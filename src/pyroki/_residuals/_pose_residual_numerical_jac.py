"""Pose residual with numerical (finite difference) Jacobian computation.

This module provides pose cost with numerical Jacobian computation
using finite differences. This is simpler but less accurate than the analytic
Jacobian version.

Note: This uses a custom Jacobian pattern that doesn't fit the simple
Cost.create_factory(residual_fn) pattern, so it's kept as a specialized
cost factory rather than a pure residual function.
"""
# 学习导读：用单边有限差分近似位姿残差雅可比，与解析路径共享残差/FK 缓存接口。
# 每列扰动 eps 并重新 FK，适合对照阅读；此文件不是 JAX 自动微分实现。

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

from .._robot import Robot


_PoseCostJacCache = tuple[jax.Array, jax.Array, jaxlie.SE3]


# 逐关节做单边 eps 扰动，差分 log 位姿误差并堆成加权 6×N_act 雅可比。
def _pose_cost_jac(
    vals: jaxls.VarValues,
    jac_cache: _PoseCostJacCache,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    eps: float = 1e-4,
) -> jax.Array:
    """Jacobian for pose cost with numerical computation."""
    joint_cfg = vals[joint_var]
    _, _, pose_error = jac_cache

    # 仅扰动一个关节并重新 FK，返回该列误差差分；eps 的尺度影响截断误差与数值精度。
    def finite_difference_jac(idx: jax.Array) -> jax.Array:
        joint_cfg_perturbed = joint_cfg.at[idx].add(eps)
        T_world_ee = jaxlie.SE3(
            robot.forward_kinematics(joint_cfg_perturbed)[..., target_link_index, :]
        )
        perturbed_pose_error = target_pose.inverse() @ T_world_ee
        # 差分的是 SE(3) log 残差而不是原始 7 维四元数/平移向量。
        err_diff = perturbed_pose_error.log() - pose_error.log()
        return err_diff / eps

    # 对 N_act 个扰动并行计算，再转置为 [6,N_act]；输出列顺序与直接驱动关节一致。
    jac = jax.vmap(finite_difference_jac)(jnp.arange(joint_cfg.shape[-1])).T
    assert jac.shape == (6, robot.joints.num_actuated_joints)

    return jac * jnp.array([pos_weight] * 3 + [ori_weight] * 3)[:, None]


# 生成 log(target⁻¹·actual) 残差和 FK 缓存；eps 只在自定义雅可比回调使用。
@jaxls.Cost.factory(jac_custom_with_cache_fn=_pose_cost_jac)
def pose_cost_numerical_jac(
    vals: jaxls.VarValues,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    eps: float = 1e-4,
) -> tuple[jax.Array, _PoseCostJacCache]:
    """Create a pose cost with numerical (finite difference) Jacobian.

    This is simpler but less accurate than the analytic Jacobian version.

    Args:
        vals: Variable values.
        robot: Robot model.
        joint_var: Joint configuration variable.
        target_pose: Target SE3 pose.
        target_link_index: Index of target link (int32).
        pos_weight: Weight for position error.
        ori_weight: Weight for orientation error.
        eps: Finite difference step size.

    Returns:
        Tuple of (residual, jacobian_cache).
    """
    del eps  # Unused in residual computation, only in Jacobian!
    assert target_link_index.dtype == jnp.int32
    joint_cfg = vals[joint_var]

    Ts_world_joint = robot._forward_kinematics_joints(joint_cfg)
    Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)

    T_world_ee = jaxlie.SE3(Ts_world_link[target_link_index, :])
    pose_error = target_pose.inverse() @ T_world_ee
    return (
        pose_error.log() * jnp.array([pos_weight] * 3 + [ori_weight] * 3),
        # Second argument is cache parameter, which is passed to the custom Jacobian function.
        (Ts_world_joint, Ts_world_link, pose_error),
    )
