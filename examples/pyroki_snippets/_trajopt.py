# 学习导读：先求带碰撞代价的起止 IK，再线性初始化轨迹，联合优化平滑、加速度、限位与扫掠碰撞约束。
# 时间步对应不同变量 ID，机器人/碰撞数组补批轴以广播到各时间点。
from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax.typing import ArrayLike


# 用起止 IK 初始化 [T,N_act] 轨迹，再优化全部时间步，保留输入 NumPy/JAX 返回约定。
def solve_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_position: ArrayLike,
    start_wxyz: ArrayLike,
    end_position: ArrayLike,
    end_wxyz: ArrayLike,
    timesteps: int,
    dt: float,
) -> ArrayLike:
    if isinstance(start_position, onp.ndarray):
        np = onp
    elif isinstance(start_position, jnp.ndarray):
        np = jnp
    else:
        raise ValueError(f"Invalid type for `ArrayLike`: {type(start_position)}")

    # 1. Solve IK for the start and end poses.
    target_link_index = robot.links.names.index(target_link_name)
    start_cfg, end_cfg = solve_iks_with_collision(
        robot=robot,
        coll=robot_coll,
        world_coll_list=world_coll,
        target_link_index=target_link_index,
        target_position_0=jnp.array(start_position),
        target_wxyz_0=jnp.array(start_wxyz),
        target_position_1=jnp.array(end_position),
        target_wxyz_1=jnp.array(end_wxyz),
    )

    # 2. Initialize the trajectory through linearly interpolating the start and end poses.
    # 插值的是关节配置，不是末端 SE(3) 轨迹；中间点可能先穿障碍，需后续优化处理。
    init_traj = jnp.linspace(start_cfg, end_cfg, timesteps)

    # 3. Optimize the trajectory.
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))

    robot = jax.tree.map(lambda x: x[None], robot)  # Add batch dimension.
    robot_coll = jax.tree.map(lambda x: x[None], robot_coll)  # Add batch dimension.

    # --- Soft costs ---
    costs: list[jaxls.Cost] = [
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.01])[None],
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([1])[None],
        ),
        pk.costs.five_point_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
            robot.joint_var_cls(jnp.arange(4, timesteps)),
            robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
            robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
            dt,
            jnp.array([0.1])[None],
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            traj_vars,
            0.02,
            5.0,
        ),
    ]

    # --- Constraints (augmented Lagrangian penalties) ---
    # Joint limits.
    costs.append(pk.costs.limit_constraint(robot, traj_vars))

    # Start / end pose constraints.
    # 名为 pose_constraint，实际约束前两步关节配置等于起始 IK 解。
    @jaxls.Cost.factory(kind="constraint_eq_zero", name="start_pose_constraint")
    def start_pose_constraint(vals: jaxls.VarValues, var) -> jax.Array:
        return (vals[var] - start_cfg).flatten()

    # 约束后两步关节配置等于终止 IK 解，从而设置末端静止边界。
    @jaxls.Cost.factory(kind="constraint_eq_zero", name="end_pose_constraint")
    def end_pose_constraint(vals: jaxls.VarValues, var) -> jax.Array:
        return (vals[var] - end_cfg).flatten()

    # 固定前两步配置，末端同样固定后两步，区别于仅固定第一/最后一个点。
    costs.append(start_pose_constraint(robot.joint_var_cls(jnp.arange(0, 2))))
    costs.append(
        end_pose_constraint(robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps)))
    )

    # Velocity limits.
    costs.append(
        pk.costs.limit_velocity_constraint(
            robot,
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            dt,
        )
    )

    # World collision avoidance using swept volumes.
    # 构造相邻配置的扫掠胶囊，输出 signed distance−0.05，供 ≥0 的世界碰撞约束使用。
    def compute_world_coll_residual(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        robot_coll: pk.collision.RobotCollision,
        world_coll_obj: pk.collision.CollGeom,
        prev_traj_vars: jaxls.Var[jax.Array],
        curr_traj_vars: jaxls.Var[jax.Array],
    ):
        coll = robot_coll.get_swept_capsules(
            robot, vals[prev_traj_vars], vals[curr_traj_vars]
        )
        dist = pk.collision.collide(
            coll.reshape((-1, 1)), world_coll_obj.reshape((1, -1))
        )  # >0 means no collision
        # 扫掠世界距离 residual≥0 对应至少 0.05 的安全距离；具体长度单位随 URDF/障碍尺度。
        return dist.flatten() - 0.05  # safety margin

    for world_coll_obj in world_coll:
        costs.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot,
                    robot_coll,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                # 这个扫掠残差为原 signed distance−margin，因此约束使用≥0，与碰撞激活残差的≤0相反。
                kind="constraint_geq_zero",
                name="world Collision (sweep)",
            )
        )

    # 4. Solve the optimization problem with augmented Lagrangian for constraints.
    # 统一交给 LeastSquaresProblem 分析并求解，数值收敛和设备执行均不由本次注释任务验证。
    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[traj_vars],
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
        )
    )
    return np.array(solution[traj_vars])


# 同时求两个变量 ID 的起止配置，以世界碰撞软项和相似度偏好选择初始化。
@jdc.jit
def solve_iks_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    target_position_0: jax.Array,
    target_wxyz_0: jax.Array,
    target_position_1: jax.Array,
    target_wxyz_1: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    joint_var_0 = robot.joint_var_cls(0)
    joint_var_1 = robot.joint_var_cls(1)
    joint_vars = robot.joint_var_cls(jnp.arange(2))
    variables = [joint_vars]

    # Soft costs: pose matching, regularization, self-collision
    # 明确列出目标/偏好/约束因子；不同示例使用的软硬条件并不相同，应逐项核对。
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var_0,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_0), target_position_0
            ),
            jnp.array(target_link_index),
            jnp.array([10.0] * 3),
            jnp.array([1.0] * 3),
        ),
        pk.costs.pose_cost(
            robot,
            joint_var_1,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_1), target_position_1
            ),
            jnp.array(target_link_index),
            jnp.array([10.0] * 3),
            jnp.array([1.0] * 3),
        ),
        pk.costs.rest_cost(
            joint_vars,
            jnp.array(joint_vars.default_factory()[None]),
            jnp.array(0.001),
        ),
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], coll),
            joint_vars,
            0.02,
            5.0,
        ),
    ]

    # Small cost to encourage the start + end configs to be close to each other.
    # 惩罚起止关节向量距离，使线性轨迹初始化尽量少跨越不同 IK 分支。
    @jaxls.Cost.factory(name="JointSimilarityCost")
    def joint_similarity_cost(vals, var_0, var_1):
        return (vals[var_0] - vals[var_1]).flatten()

    costs.append(joint_similarity_cost(joint_var_0, joint_var_1))

    # World collision as soft cost (more robust for IK initialization)
    costs.extend(
        [
            pk.costs.world_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], coll),
                joint_vars,
                jax.tree.map(lambda x: x[None], world_coll),
                0.05,
                10.0,
            )
            for world_coll in world_coll_list
        ]
    )

    # Constraint: joint limits
    costs.append(
        pk.costs.limit_constraint(
            jax.tree.map(lambda x: x[None], robot),
            joint_vars,
        ),
    )

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(verbose=False)
    )
    return sol[joint_var_0], sol[joint_var_1]
