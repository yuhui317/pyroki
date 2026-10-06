"""
Solves the basic IK problem.
"""
# 学习导读：让多个目标位姿共享同一个 JointVar(0)；批目标是多代价约束，不是多套独立关节解。

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


# 接收 [N_target,3] 位置和 [N_target,4] 姿态，名称映射后求共同关节解。
def solve_ik_with_multiple_targets(
    robot: pk.Robot,
    target_link_names: Sequence[str],
    target_wxyzs: onp.ndarray,
    target_positions: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_names: Sequence[str]. List of link names to be controlled.
        target_wxyzs: onp.ndarray. Shape: (num_targets, 4). Target orientations.
        target_positions: onp.ndarray. Shape: (num_targets, 3). Target positions.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    num_targets = len(target_link_names)
    assert target_positions.shape == (num_targets, 3)
    assert target_wxyzs.shape == (num_targets, 4)
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]

    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_wxyzs),
        jnp.array(target_positions),
        jnp.array(target_link_indices),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


# 多个目标复用变量 ID=0，批化位姿代价共同约束同一配置，再加 rest 与限位。
@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    target_joint_indices: jax.Array,
) -> jax.Array:
    JointVar = robot.joint_var_cls

    # Get the batch axes for the variable through the target pose.
    # Batch axes for the variables and cost terms (e.g., target pose) should be broadcastable!
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )
    batch_axes = target_pose.get_batch_axes()

    # 明确列出目标/偏好/约束因子；不同示例使用的软硬条件并不相同，应逐项核对。
    costs = [
        pk.costs.pose_cost_analytic_jac(
            # 给机器人数值叶节点加长度 1 批轴以对齐目标批；静态拓扑不会被数组化。
            jax.tree.map(lambda x: x[None], robot),
            # 目标批中的 ID 全为 0，所以每个目标代价引用同一关节变量，而不是各自独立解。
            JointVar(jnp.full(batch_axes, 0)),
            target_pose,
            target_joint_indices,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=JointVar.default_factory(),
            weight=1.0,
        ),
    ]
    costs.append(
        pk.costs.limit_constraint(
            robot,
            JointVar(0),
        ),
    )
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[JointVar(0)])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]
