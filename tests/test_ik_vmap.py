"""Tests for vmapping inverse kinematics solvers (and collision batching)."""
# 学习导读：验证 JAX vmap 对共享/批量机器人和碰撞体的适配；多数断言检查形状/拓扑兼容而非 IK 精度。
# 测试加载 robot_descriptions 的模型，本次注释任务不执行这些数值/模型加载测试。

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
from pyroki.collision import RobotCollision, Capsule
from robot_descriptions.loaders.yourdfpy import load_robot_description

# -------------------------------------------------------------------
# Solvers (JIT-compiled)
# -------------------------------------------------------------------


# 测试用单目标基本 IK，添加 rest 和关节约束，供不同 vmap 方式调用。
@jdc.jit
def _solve_ik(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    """Basic IK solver."""
    joint_var = robot.joint_var_cls(0)
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )
    # 明确列出目标/偏好/约束因子；不同示例使用的软硬条件并不相同，应逐项核对。
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot, joint_var, target_pose, target_link_index, 50.0, 10.0
        ),
        pk.costs.limit_constraint(robot, joint_var),
        pk.costs.rest_cost(joint_var, jnp.array(joint_var.default_factory()), 0.1),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(1.0),
        )
    )
    return sol[joint_var]


# 测试用带自碰撞软代价的 IK，检查 RobotCollision 是否能共享或按批映射。
@jdc.jit
def _solve_ik_coll(
    robot: pk.Robot,
    coll: RobotCollision,
    target_link_index: jax.Array,
    target_pose: jaxlie.SE3,
) -> jax.Array:
    """Collision-aware IK solver."""
    joint_var = robot.joint_var_cls(0)
    # 明确列出目标/偏好/约束因子；不同示例使用的软硬条件并不相同，应逐项核对。
    costs = [
        pk.costs.pose_cost(robot, joint_var, target_pose, target_link_index, 5.0, 1.0),
        pk.costs.rest_cost(joint_var, jnp.array(joint_var.default_factory()), 0.01),
        pk.costs.self_collision_cost(robot, coll, joint_var, 0.02, 5.0),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
        )
    )
    return sol[joint_var]


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


# 构造小型胶囊碰撞对象，批化数值叶节点后检查静态活动索引仍可使用。
def test_unit_collision_batching():
    """Unit test: RobotCollision manual batching without full URDF."""
    # 1. Create a dummy collision object (single batch)
    coll = Capsule.from_radius_height(jnp.array([0.1, 0.1]), jnp.array([1.0, 1.0]))
    robot_coll = RobotCollision(
        num_links=2,
        link_names=("l1", "l2"),
        coll=coll,
        active_idx_i=(0,),
        active_idx_j=(1,),  # Static tuples
        _geom_to_link_idx=jnp.arange(2, dtype=jnp.int32),
    )

    # 2. Manually broadcast to batch size 5 with variation
    batch_size = 5

    # 把数值叶节点堆出 5 批并添加差异，用于检查映射没有丢失批间变化。
    def broadcast_vary(leaf):
        # Broadcast (1, ...) -> (5, ...) and add variation
        return jnp.stack([leaf + i * 0.01 for i in range(batch_size)])

    batched_coll = jax.tree.map(broadcast_vary, robot_coll)

    # 3. Vmap a function over it
    # 读取静态碰撞对索引并返回两几何体半径之和，验证 PyTree/vmap 形状。
    @jax.jit
    def compute(rc: RobotCollision):
        idx_i, idx_j = jnp.array(rc.active_idx_i), jnp.array(rc.active_idx_j)
        r = rc.coll.size[..., 0]  # access radius
        return r[idx_i] + r[idx_j]

    results = jax.vmap(compute)(batched_coll)
    assert results.shape == (batch_size, 1)
    assert jnp.all(results[1:] > results[:-1])  # Verify variation was preserved


# 断言静态活动碰撞对不同的对象不能直接 tree.map 堆叠，强调批内拓扑必须一致。
def test_unit_collision_batching_topology_mismatch():
    """
    Unit test: Emphasize that RobotCollision can be batched IF AND ONLY IF
    the non-batchable elements (topological properties like link structure/indices)
    are identical across the batch.
    """
    # 1. Base collision geometry
    coll = Capsule.from_radius_height(jnp.array([0.1, 0.1]), jnp.array([1.0, 1.0]))

    # 2. Define two RobotCollision objects with DIFFERENT topologies (active_idx_j)
    # These fields are marked as jdc.Static, so they are part of the PyTree structure.
    rc1 = RobotCollision(
        num_links=2,
        link_names=("l1", "l2"),
        coll=coll,
        active_idx_i=(0,),
        active_idx_j=(1,),
        _geom_to_link_idx=jnp.arange(2, dtype=jnp.int32),
    )
    rc2 = RobotCollision(
        num_links=2,
        link_names=("l1", "l2"),
        coll=coll,
        active_idx_i=(0,),
        active_idx_j=(0,),  # <--- Difference here
        _geom_to_link_idx=jnp.arange(2, dtype=jnp.int32),
    )

    # 3. Attempting to batch them (e.g. stack) should fail because structures differ.
    try:
        jax.tree.map(lambda x, y: jnp.stack([x, y]), rc1, rc2)
    except ValueError as e:
        # Expected error: "Mismatch custom node data..."
        assert "Mismatch custom node data" in str(e)
        return

    raise AssertionError(
        "Batching incompatible RobotCollision objects should have failed!"
    )


# 共享 Robot/目标索引，把位置姿态沿第 0 轴 vmap，检查 [B,N_act] 输出。
def test_integration_ik_basic():
    """Integration: Standard usage (Shared Robot, Batched Targets)."""
    urdf = load_robot_description("panda_description")
    # URDF 加载在宿主侧完成，Robot.from_urdf 把运动学参数转成可微 JAX 数据结构。
    robot = pk.Robot.from_urdf(urdf)
    target_idx = robot.links.names.index("panda_hand")

    batch_size = 5
    target_pos = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, 3))
    target_wxyz = jnp.tile(jnp.array([0.0, 1.0, 0.0, 0.0]), (batch_size, 1))

    # Standard vmap: Robot/Index shared (None), Targets batched (0)
    solve_batch = jax.vmap(_solve_ik, in_axes=(None, None, 0, 0))
    res = solve_batch(robot, jnp.array(target_idx), target_wxyz, target_pos)
    assert res.shape == (batch_size, robot.joints.num_actuated_joints)


# 共享机器人但按批传碰撞对象和目标位姿，检查组合映射的输出形状。
def test_integration_ik_mixed_batching():
    """Integration: Mixed usage (Shared Robot, Batched Collision, Batched Targets)."""
    urdf = load_robot_description("panda_description")
    # URDF 加载在宿主侧完成，Robot.from_urdf 把运动学参数转成可微 JAX 数据结构。
    robot = pk.Robot.from_urdf(urdf)
    # Ignore adjacents to ensure collision object is non-trivial but clean
    robot_coll = RobotCollision.from_urdf(urdf, ignore_immediate_adjacents=True)
    target_idx = robot.links.names.index("panda_hand")

    batch_size = 3

    # 1. Batched Targets
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(jnp.pi, 0.0, 0.0), jnp.array([0.5, 0.0, 0.5])
    )
    # Broadcast to batch
    target_poses = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), target_pose
    )

    # 2. Batched Collision (Explicit broadcast)
    batched_coll = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), robot_coll
    )

    # 3. Vmap: Robot (None), Collision (0), Index (None), Pose (0)
    solve_batch = jax.vmap(_solve_ik_coll, in_axes=(None, 0, None, 0))
    res = solve_batch(robot, batched_coll, jnp.array(target_idx), target_poses)
    assert res.shape == (batch_size, robot.joints.num_actuated_joints)


# 机器人/碰撞/目标全部显式批化，使用默认 in_axes=0 检查形状一致性。
def test_integration_ik_fully_batched():
    """Integration: Fully batched (Robot, Collision, Targets all batched)."""
    urdf = load_robot_description("panda_description")
    # URDF 加载在宿主侧完成，Robot.from_urdf 把运动学参数转成可微 JAX 数据结构。
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf, ignore_immediate_adjacents=True)
    target_idx = robot.links.names.index("panda_hand")

    batch_size = 3

    # 1. Batched Targets
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(jnp.pi, 0.0, 0.0), jnp.array([0.5, 0.0, 0.5])
    )
    target_poses = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), target_pose
    )

    # 2. Batched Collision
    batched_coll = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), robot_coll
    )

    # 3. Batched Robot (Explicit broadcast of all array leaves)
    # This works because we converted topology indices to Static!
    batched_robot = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), robot
    )

    target_idx_batched = jnp.full((batch_size,), target_idx, dtype=jnp.int32)

    # 4. Vmap with default in_axes (all 0)
    solve_batch = jax.vmap(_solve_ik_coll)
    res = solve_batch(batched_robot, batched_coll, target_idx_batched, target_poses)
    assert res.shape == (batch_size, robot.joints.num_actuated_joints)


# 只在直接运行脚本时进入主函数/命令行流程，模块导入通常不会运行此入口。
if __name__ == "__main__":
    test_unit_collision_batching()
    test_integration_ik_basic()
    test_integration_ik_mixed_batching()
    test_integration_ik_fully_batched()
