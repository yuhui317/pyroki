# 学习导读：联合优化短关节轨迹和末端 SE(3) 轨迹，用上轮结果初始化，兼顾终点、平滑与碰撞软项。
# 当前示例采用加权软惩罚以减少外循环成本，并非所有安全条件都作为硬约束。
from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


# 在轨迹前保留一个起始配置位置，扩展上轮初值后求解，输出时移除起始行。
def solve_online_planning(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    timesteps: int,
    dt: float,
    start_cfg: onp.ndarray,
    prev_sols: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Solve online planning with collision."""

    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_indices = [robot.links.names.index(target_link_name)]

    target_poses = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    target_links = jnp.array(target_link_indices)

    # Warm start: use previous solution shifted by one step.
    # 增加一个内部起点变量；外部返回轨迹去掉第一行，仍是请求的 timesteps 长度。
    timesteps = timesteps + 1  # for start pose cost.

    sol_traj, sol_pos, sol_wxyz = _solve_online_planning_jax(
        robot,
        robot_coll,
        world_coll,
        target_poses,
        target_links,
        timesteps,
        dt,
        jnp.array(start_cfg),
        # 代码在末尾复制上一轨迹最后一行作扩展，没有在此显式删掉上一轨迹首行。
        jnp.concatenate([prev_sols, prev_sols[-1:]], axis=0),
    )
    sol_traj = sol_traj[1:]
    sol_pos = sol_pos[1:]
    sol_wxyz = sol_wxyz[1:]

    return onp.array(sol_traj), onp.array(sol_pos), onp.array(sol_wxyz)


# 联合优化各时间步关节与末端位姿，以软项连接 FK/目标/平滑/速度/碰撞。
@jdc.jit
def _solve_online_planning_jax(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_poses: jaxlie.SE3,
    target_links: jnp.ndarray,
    timesteps: jdc.Static[int],
    dt: float,
    start_cfg: jnp.ndarray,
    prev_sols: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_targets = len(target_links)

    # 对单个时间步内的多个 SE(3) 目标逐个作流形右加更新，切空间先重排为 [N_target,6]。
    def batched_rplus(
        pose: jaxlie.SE3,
        delta: jax.Array,
    ) -> jaxlie.SE3:
        return jax.vmap(jaxlie.manifold.rplus)(pose, delta.reshape(num_targets, -1))

    # Custom SE3 variable to batch across multiple joint targets.
    # This is not to be confused with SE3Vars with ids, which we use here for timesteps.
    # 一个变量值包含 N_target 个 SE(3)，而变量 ID 的批轴表示时间，两种批含义不同。
    class BatchedSE3Var(  # pylint: disable=missing-class-docstring
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity((num_targets,)),
        retract_fn=batched_rplus,
        tangent_dim=jaxlie.SE3.tangent_dim * num_targets,
    ): ...

    # --- Define Variables ---
    # 时间 ID=0..T−1 是不同关节状态；prev/next ID 切片构造相邻平滑/速度因子。
    traj_var = robot.joint_var_cls(jnp.arange(0, timesteps))
    traj_var_prev = robot.joint_var_cls(jnp.arange(0, timesteps - 1))
    traj_var_next = robot.joint_var_cls(jnp.arange(1, timesteps))
    pose_var = BatchedSE3Var(jnp.arange(0, timesteps))
    pose_var_prev = BatchedSE3Var(jnp.arange(0, timesteps - 1))
    pose_var_next = BatchedSE3Var(jnp.arange(1, timesteps))

    init_pose_vals = jaxlie.SE3(
        robot.forward_kinematics(prev_sols)[..., target_links, :]
    )

    # --- Define Costs ---
    factors: list[jaxls.Cost] = []  # Changed type hint to jaxls.Cost

    # 用 FK 将关节轨迹和自由末端位姿变量绑定，残差为两者相对 SE(3) 的 log。
    @jaxls.Cost.factory(name="SE3PoseMatchJointCost")
    def match_joint_to_pose_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
        pose_var: BatchedSE3Var,
    ):
        joint_cfg = vals[joint_var]
        target_pose = vals[pose_var]
        Ts_joint_world = robot.forward_kinematics(joint_cfg)
        residual = (
            (jaxlie.SE3(Ts_joint_world[..., target_links, :])).inverse() @ (target_pose)
        ).log()
        return residual.flatten() * 100.0

    # 正则相邻末端位姿的 SE(3) 切空间差，不按 dt 换算速度。
    @jaxls.Cost.factory(name="SE3SmoothnessCost")
    def pose_smoothness_cost(
        vals: jaxls.VarValues,
        pose_var: BatchedSE3Var,
        pose_var_prev: BatchedSE3Var,
    ):
        return (vals[pose_var].inverse() @ vals[pose_var_prev]).log().flatten() * 1.0

    # 让末端轨迹最后一步接近固定目标，平移/旋转分别使用 50/20 残差权重。
    @jaxls.Cost.factory(name="SE3PoseMatchCost")
    def pose_match_cost(
        vals: jaxls.VarValues,
        pose_var: BatchedSE3Var,
    ):
        return (
            (vals[pose_var].inverse() @ target_poses).log()
            * jnp.array([50.0] * 3 + [20.0] * 3)
        ).flatten()

    # 以 100 倍软残差让第一关节配置接近 start_cfg，不是等式约束。
    @jaxls.Cost.factory(name="MatchStartPoseCost")
    def match_start_pose_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
    ):
        return (vals[joint_var] - start_cfg).flatten() * 100.0

    # Add pose costs.
    factors.extend(
        [
            pose_match_cost(
                BatchedSE3Var(timesteps - 1),
            ),
            pose_smoothness_cost(
                pose_var_next,
                pose_var_prev,
            ),
        ]
    )

    # Need to constrain the start joint cfg.
    factors.append(match_start_pose_cost(robot.joint_var_cls(0)))

    # Add joint costs.
    # Technically we should add this as a constraint, but the augmented lagrangian formulation
    # requires us to perform multiple outer loop iterations to converge, which is slow. :(
    factors.extend(
        [
            match_joint_to_pose_cost(
                traj_var,
                pose_var,
            ),
            pk.costs.smoothness_cost(
                traj_var_prev,
                traj_var_next,
                weight=10.0,
            ),
            pk.costs.limit_velocity_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var_prev,
                traj_var_next,
                weight=1.0,
                dt=dt,
            ),
            pk.costs.limit_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=100.0,
            ),
            pk.costs.rest_cost(
                traj_var,
                jnp.array(traj_var.default_factory())[None],
                weight=0.01,
            ),
            pk.costs.manipulability_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=0.01,
                target_link_indices=target_links,
            ),
            pk.costs.self_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], robot_coll),
                traj_var,
                weight=10.0,
                margin=0.02,
            ),
        ]
    )
    factors.extend(
        [
            pk.costs.world_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], robot_coll),
                traj_var,
                jax.tree.map(lambda x: x[None], obs),
                weight=20.0,
                margin=0.1,
            )
            for obs in world_coll
        ]
    )

    # 统一交给 LeastSquaresProblem 分析并求解，数值收敛和设备执行均不由本次注释任务验证。
    solution = (
        jaxls.LeastSquaresProblem(factors, [traj_var, pose_var])
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make(
                (traj_var.with_value(prev_sols), pose_var.with_value(init_pose_vals))
            ),
            # 单次在线求解最多 20 次迭代；有限迭代软代价不能保证所有约束完全满足。
            termination=jaxls.TerminationConfig(max_iterations=20),
        )
    )
    pose_traj = solution[pose_var]
    return (
        solution[traj_var],
        pose_traj.translation(),
        pose_traj.rotation().wxyz,
    )
