"""Pose residual with analytic Jacobian computation.

This module provides pose cost with efficient analytic Jacobian computation,
avoiding automatic differentiation overhead.

Note: This uses a custom Jacobian pattern that doesn't fit the simple
Cost.create_factory(residual_fn) pattern, so it's kept as a specialized
cost factory rather than a pure residual function.
"""
# 学习导读：位姿残差的解析雅可比路径，利用缓存 FK、关节祖先路径和 SE(3) jlog 构造 6×N 雅可比。
# 模仿关节的导数按 multiplier 缩放并累加到被驱动关节列；joint_mask 用于屏蔽雅可比列。

from functools import partial

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

from .._robot import Robot


# 从目标关节向根遍历，标出影响目标的关节及对应直接驱动列，其余填 -1。
def _get_actuated_joints_applied_to_target(
    robot: Robot,
    target_joint_idx: jax.Array,
) -> jax.Array:
    """For each joint `i` in the robot, we return an index that is:

    1) -1 if the joint is not in the path to the target link.
    2) an actuated joint index if the joint is in the path to the target link.
        - If the joint `i` is actuated, this is just `i`.
        - If the joint `i` mimics joint `j`, this is set to `j`.

    The inputs and outputs should both be integer arrays of shape
    `(num_joints,)`.
    """

    assert target_joint_idx.shape == ()

    # 读取当前关节的直接驱动或 mimic 索引，写入路径表后继续到父关节。
    def body_fun(joint_idx, indices):
        # Find the corresponding active actuated joint.
        actuated_indices = jnp.array(robot.joints.actuated_indices, dtype=jnp.int32)
        mimic_act_indices = jnp.array(robot.joints.mimic_act_indices, dtype=jnp.int32)
        parent_indices = jnp.array(robot.joints.parent_indices, dtype=jnp.int32)

        active_act_joint = jnp.where(
            actuated_indices[joint_idx] != -1,
            # The current joint is actuated.
            actuated_indices[joint_idx],
            # The current joint is not actuated; this is -1 if not a mimic joint.
            mimic_act_indices[joint_idx],
        )

        # Find the parent of the current joint.
        parent_joint = parent_indices[joint_idx]

        # Continue traversing up the kinematic tree, using the parent joint.
        # This value may either go up or down, since there's no guarantee that
        # the kinematic tree is topologically sorted. :-)
        next_indices = indices.at[joint_idx].set(active_act_joint)
        return (parent_joint, next_indices)

    idx_applied_to_target = jnp.full(
        (robot.joints.num_joints,),
        fill_value=-1,
        dtype=jnp.int32,
    )
    idx_applied_to_target = jax.lax.while_loop(
        lambda carry: jnp.any(carry[0] >= 0),
        lambda carry: body_fun(*carry),
        (target_joint_idx, idx_applied_to_target),
    )[-1]
    return idx_applied_to_target


_PoseCostJacCache = tuple[jax.Array, jax.Array, jaxlie.SE3]


# 准备可广播的目标/变量/机器人批轴，预先求祖先关节映射，再构造带缓存的解析代价。
def pose_cost_analytic_jac(
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    joint_mask: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create a pose cost with analytic Jacobian computation.

    This is more efficient than using automatic differentiation for the Jacobian.

    Args:
        robot: Robot model.
        joint_var: Joint configuration variable.
        target_pose: Target SE3 pose.
        target_link_index: Index of target link (int32).
        pos_weight: Weight for position error.
        ori_weight: Weight for orientation error.
        joint_mask: Optional array of shape (n_actuated_joints,) with values in [0, 1].
            1.0 = joint is optimized, 0.0 = joint is locked (zero Jacobian).
            Defaults to 1.0 (all joints optimized).

    Returns:
        Cost object for pose matching.
    """
    # We only check shape lengths because there might be (1,) axes for
    # broadcasting reasons.
    assert (
        len(target_link_index.shape)
        == len(jnp.asarray(joint_var.id).shape)
        == len(robot.joints.twists.shape[:-2])
    ), "Batch axes of inputs should match"

    # Broadcast the inputs for _get_actuated_joints_applied_to_target().
    # Excluding the weights for now...
    batch_axes = jnp.broadcast_shapes(
        target_pose.get_batch_axes(),
        jnp.asarray(joint_var.id).shape,
        target_pose.get_batch_axes(),
        target_link_index.shape,
    )
    broadcast_batch_axes = partial(
        jax.tree.map,
        lambda x: jnp.broadcast_to(x, batch_axes + x.shape[len(batch_axes) :]),
    )
    get_actuated_joints = _get_actuated_joints_applied_to_target
    for _ in range(len(batch_axes)):
        get_actuated_joints = jax.vmap(get_actuated_joints)

    # Compute applied joints.
    robot = broadcast_batch_axes(robot)
    link_parent_joint_indices = jnp.array(
        robot.links.parent_joint_indices, dtype=jnp.int32
    )
    base_link_mask = link_parent_joint_indices == -1
    parent_joint_indices = jnp.where(base_link_mask, 0, link_parent_joint_indices)
    target_joint_idx = parent_joint_indices[
        tuple(jnp.arange(d) for d in parent_joint_indices.shape[:-1])
        + (target_link_index,)
    ]
    joints_applied_to_target = get_actuated_joints(
        broadcast_batch_axes(robot), broadcast_batch_axes(target_joint_idx)
    )

    return _pose_cost_analytical_jac(
        robot,
        joint_var,
        target_pose,
        target_link_index,
        pos_weight,
        ori_weight,
        joints_applied_to_target,
        joint_mask,
    )


# It's nice to pass arguments in explicitly instead of via closure in the
# `pose_cost_analytic_jac` wrapper. It helps jaxls vectorize repeated costs.
# 用缓存世界位姿计算线/角速度，转换到末端坐标并乘误差 jlog，得到加权 6×N_act 雅可比。
def _pose_cost_jac(
    vals: jaxls.VarValues,
    jac_cache: _PoseCostJacCache,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    joints_applied_to_target: jax.Array,
    joint_mask: jax.Array | float,
) -> jax.Array:
    """Jacobian for pose cost with analytic computation."""
    del vals, joint_var, target_pose  # Unused!
    Ts_world_joint, Ts_world_link, pose_error = jac_cache

    T_world_ee = jaxlie.SE3(Ts_world_link[target_link_index])
    Ts_world_joint = jaxlie.SE3(Ts_world_joint)

    R_ee_world = T_world_ee.rotation().inverse()

    # Get joint twists; these are scaled for mimic joints.
    # mimic multiplier 缩放导数；offset 已体现在缓存的 FK 位姿，不额外加在导数上。
    joint_twists = robot.joints.twists * robot.joints.mimic_multiplier[..., None]

    # Get angular velocity components (omega).
    omega_local = joint_twists[:, 3:]
    omega_wrt_world = Ts_world_joint.rotation() @ omega_local
    omega_wrt_ee = R_ee_world @ omega_wrt_world

    # Get linear velocity components (v).
    vel_local = joint_twists[:, :3]
    vel_wrt_world = Ts_world_joint.rotation() @ vel_local

    # Compute the linear velocity component (v = ω × r + v_joint).
    # 旋转关节线速度为 ω×(p_ee−p_joint)，再加移动关节自身线速度。
    vel_wrt_world = (
        jnp.cross(
            omega_wrt_world,
            T_world_ee.translation() - Ts_world_joint.translation(),
        )
        + vel_wrt_world
    )
    vel_wrt_ee = R_ee_world @ vel_wrt_world

    # Combine into spatial Jacobian.
    jac = jnp.where(
        joints_applied_to_target[:, None] != -1,
        jnp.concatenate(
            [
                vel_wrt_ee,
                omega_wrt_ee,
            ],
            axis=1,
        ),
        0.0,
    ).T
    # 几何速度 Jacobian 经误差 log 的导数转换为当前残差 Jacobian。
    jac = pose_error.jlog() @ jac

    # Jacobian of all joints => Jacobian of actuated joints.
    #
    # Because of mimic joints, the Jacobian terms from multiple joints can be
    # applied to a single actuated joint. This is summed!
    jac = (
        jnp.zeros((6, robot.joints.num_actuated_joints))
        .at[:, joints_applied_to_target]
        .add((joints_applied_to_target[None, :] != -1) * jac)
    )

    # Apply joint mask: zero out Jacobian columns for locked joints.
    # mask=0 只让本位姿因子的该列导数为零；其它因子仍可能影响对应关节。
    jac = jac * joint_mask

    # Apply weights
    # 平移三行与旋转三行分别乘残差权重，和缓存残差使用同样权重。
    weights = jnp.array([pos_weight] * 3 + [ori_weight] * 3)
    return jac * weights[:, None]


# 计算 log(target⁻¹·actual) 位姿残差，同时缓存关节 FK、连杆 FK 与误差供解析导数复用。
@jaxls.Cost.factory(jac_custom_with_cache_fn=_pose_cost_jac)
def _pose_cost_analytical_jac(
    vals: jaxls.VarValues,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    joints_applied_to_target: jax.Array,
    joint_mask: jax.Array | float,
) -> tuple[jax.Array, _PoseCostJacCache]:
    """Computes the residual for matching link poses to target poses."""
    del joints_applied_to_target, joint_mask  # Only used in Jacobian
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
