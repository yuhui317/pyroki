"""Core residual functions for pyroki costs and constraints.

These functions have the same signature as cost/constraint functions
(taking `vals: VarValues` as first arg), so they can be directly wrapped:

    limit_cost = Cost.create_factory(limit_residual)
    limit_constraint = Constraint.create_factory(limit_residual, constraint_type="leq_zero")
"""
# 学习导读：以 VarValues 读取优化变量，输出加权残差；包含位姿、限位、平滑、可操作度与碰撞。
# 权重在残差层相乘，因此最小二乘中的平方项会带权重的平方；约束符号以实际 return 表达式为准。

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Var, VarValues

from .._robot import Robot
from ..collision import CollGeom, RobotCollision, colldist_from_sdf


# --- Pose Residuals ---


# 匹配单个目标连杆：log(actual⁻¹·target) 的前三维平移、后三维旋转分别加权并展平。
def pose_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_pose: jaxlie.SE3,
    target_link_index: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    assert target_link_index.dtype == jnp.int32
    joint_cfg = vals[joint_var]
    Ts_link_world = robot.forward_kinematics(joint_cfg)
    pose_actual = jaxlie.SE3(Ts_link_world[..., target_link_index, :])
    # SE(3) log 返回 6 维切向量，前三维受旋转误差影响，不应直接等同世界 xyz 差。
    residual = (pose_actual.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


# 先以可优化的 T_world_base 乘基座相对 FK，再计算世界目标的 SE(3) 误差。
def pose_with_base_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    T_world_base_var: Var[jaxlie.SE3],
    target_pose: jaxlie.SE3,
    target_link_indices: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses relative to a mobile base."""
    assert target_link_indices.dtype == jnp.int32
    joint_cfg = vals[joint_var]
    T_world_base = vals[T_world_base_var]
    Ts_base_link = robot.forward_kinematics(joint_cfg)  # FK is T_base_link
    T_base_target_link = jaxlie.SE3(Ts_base_link[..., target_link_indices, :])
    T_world_target_link_actual = T_world_base @ T_base_target_link

    residual = (T_world_target_link_actual.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


# --- Limit Residuals ---


# 检查含 mimic/fixed 的完整配置位置范围，实际返回非负越界量；范围内为零。
def limit_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint limit violation residual.

    Returns values that are:
    - Positive when violated (joint outside limits)
    - Negative when satisfied (joint within limits)

    For inequality constraints: residual <= 0 means satisfied.
    """
    joint_cfg = vals[joint_var]
    joint_cfg_eff = robot.joints.get_full_config(joint_cfg)
    # 原 docstring 写范围内负值，但 max(0,·) 的实现实际上范围内为零、越界才为正。
    upper_violation = jnp.maximum(0.0, joint_cfg_eff - robot.joints.upper_limits_all)
    lower_violation = jnp.maximum(0.0, robot.joints.lower_limits_all - joint_cfg_eff)
    return (jnp.concatenate([upper_violation, lower_violation]) * weight).flatten()


# 用相邻直接驱动配置差除 dt 得到速度，返回超过 URDF 速度限位的非负部分。
def limit_velocity_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    prev_joint_var: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint velocity limit violation residual.

    Returns values that are:
    - Positive when violated (|velocity| > limit)
    - Zero when satisfied (|velocity| <= limit)
    """
    # dt 必须非零并与轨迹采样周期一致；旋转/移动关节对应 rad/s 或 m/s。
    joint_vel = (vals[joint_var] - vals[prev_joint_var]) / dt
    residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
    return (residual * weight).flatten()


# --- Regularization Residuals ---


# 将配置拉向给定 rest_pose，以 residual weight 逐维加权。
def rest_residual(
    vals: VarValues,
    joint_var: Var[Array],
    rest_pose: Array,
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual biasing joints towards a rest pose."""
    return ((vals[joint_var] - rest_pose) * weight).flatten()


# 同时正则关节与基座切空间位姿误差，基座参照其变量默认值。
def rest_with_base_residual(
    vals: VarValues,
    joint_var: Var[Array],
    T_world_base_var: Var[jaxlie.SE3],
    rest_pose: Array,
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual biasing joints and base towards its default pose."""
    residual_joints = vals[joint_var] - rest_pose
    residual_base = (
        vals[T_world_base_var].inverse() @ T_world_base_var.default_factory()
    ).log()
    return (jnp.concatenate([residual_joints, residual_base]) * weight).flatten()


# 惩罚相邻配置差，当前函数不除 dt，因此它是步间平滑项而非物理速度本身。
def smoothness_residual(
    vals: VarValues,
    curr_joint_var: Var[Array],
    past_joint_var: Var[Array],
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual penalizing joint configuration differences (velocity)."""
    return ((vals[curr_joint_var] - vals[past_joint_var]) * weight).flatten()


# --- Manipulability Residual ---


# 自动微分目标平移得到 3×N_act Jacobian，计算 sqrt(max(0,det(JJᵀ)))。
def _compute_manip_yoshikawa(
    cfg: Array,
    robot: Robot,
    target_link_index: jax.Array,
) -> Array:
    """Helper: Computes manipulability measure for a single link."""
    jacobian = jax.jacfwd(
        lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
    )(cfg)[target_link_index]
    # 此可操作度只用 3 维平移 Jacobian，不能据它推断完整 6D 位姿灵巧性。
    JJT = jacobian @ jacobian.T
    assert JJT.shape == (3, 3)
    return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(JJT)))


# 对指定目标连杆的平移可操作度取倒数惩罚，可沿目标索引做 vmap。
def manipulability_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_link_indices: Array,
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual penalizing low manipulability (translation)."""
    cfg = vals[joint_var]
    if target_link_indices.ndim == 0:
        vmapped_manip = _compute_manip_yoshikawa(cfg, robot, target_link_indices)
    else:
        vmapped_manip = jax.vmap(_compute_manip_yoshikawa, in_axes=(None, None, 0))(
            cfg, robot, target_link_indices
        )
    residual = 1.0 / (vmapped_manip + 1e-6)
    return (residual * weight).flatten()


# --- Collision Residuals ---


# 由当前配置计算活动自碰撞对距离，转换为安全距离内非负惩罚并展平。
def self_collision_residual(
    vals: VarValues,
    robot: Robot,
    robot_coll: RobotCollision,
    joint_var: Var[Array],
    margin: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes self-collision violation residual. Residual is >0 if collision is detected."""
    cfg = vals[joint_var]
    # signed distance 正值为分離、负值为穿透；colldist 返回≤0，所以这里额外取负得到非负残差。
    active_distances = robot_coll.compute_self_collision_distance(robot, cfg)
    return -(colldist_from_sdf(active_distances, margin) * weight).flatten()


# 计算机器人几何体与世界几何体的距离矩阵，再变成安全距离内非负惩罚。
def world_collision_residual(
    vals: VarValues,
    robot: Robot,
    robot_coll: RobotCollision,
    joint_var: Var[Array],
    world_geom: CollGeom,
    margin: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes world collision violation residual. Residual is >0 if collision is detected."""
    cfg = vals[joint_var]
    dist_matrix = robot_coll.compute_world_collision_distance(robot, cfg, world_geom)
    return -(colldist_from_sdf(dist_matrix, margin) * weight).flatten()


# --- Finite Difference Residuals ---


# 以四个邻点构造五点中心一阶差分，惩罚超出直接驱动速度限位的部分。
def five_point_velocity_residual(
    vals: VarValues,
    robot: Robot,
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes velocity limit violation using 5-point stencil.

    Returns values that are:
    - Positive when violated (|velocity| > limit)
    - Negative when satisfied (|velocity| <= limit)
    """
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]

    # 中心一阶差分系数 [-1,8,-8,1]/(12dt)，结果单位随关节为每秒。
    velocity = (-q_tp2 + 8 * q_tp1 - 8 * q_tm1 + q_tm2) / (12 * dt)
    vel_limits = robot.joints.velocity_limits
    residual = jnp.maximum(0.0, jnp.abs(velocity) - vel_limits)
    return (residual * weight).flatten()


# 用五点中心二阶差分估算加速度，返回其逐维绝对值用于软平滑。
def five_point_acceleration_residual(
    vals: VarValues,
    var_t: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint acceleration using 5-point stencil."""
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_t = vals[var_t]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]

    # 五点二阶差分除以 dt²；调用方须提供 t±1/t±2 的对齐变量 ID。
    acceleration = (-q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2) / (12 * dt**2)
    residual = jnp.abs(acceleration)
    return (residual * weight).flatten()


# 同样使用五点二阶差分，返回超过显式 acceleration_limit 的非负量。
def limit_acceleration_residual(
    vals: VarValues,
    var_t: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    acceleration_limit: Array | float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint acceleration limit violation residual using 5-point stencil.

    Returns values that are:
    - Positive when violated (|acceleration| > limit)
    - Negative when satisfied (|acceleration| <= limit)
    """
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_t = vals[var_t]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]

    # 五点二阶差分除以 dt²；调用方须提供 t±1/t±2 的对齐变量 ID。
    acceleration = (-q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2) / (12 * dt**2)
    residual = jnp.maximum(0.0, jnp.abs(acceleration) - acceleration_limit)
    return (residual * weight).flatten()


# 名称保留 five_point，实际用六个邻点的七点模板计算三阶差分 jerk。
def five_point_jerk_residual(
    vals: VarValues,
    var_t_plus_3: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    var_t_minus_3: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint jerk using 7-point stencil."""
    q_tm3 = vals[var_t_minus_3]
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]
    q_tp3 = vals[var_t_plus_3]

    # 七点三阶差分除以 dt³；函数名保留 five_point，但模板覆盖 t±1/t±2/t±3。
    jerk = (-q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3) / (
        8 * dt**3
    )
    return (jnp.abs(jerk) * weight).flatten()


# 以七点三阶差分计算 jerk，惩罚超出显式 jerk_limit 的非负部分。
def limit_jerk_residual(
    vals: VarValues,
    var_t_plus_3: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    var_t_minus_3: Var[Array],
    dt: float,
    jerk_limit: Array | float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint jerk limit violation residual using 7-point stencil.

    Returns values that are:
    - Positive when violated (|jerk| > limit)
    - Negative when satisfied (|jerk| <= limit)
    """
    q_tm3 = vals[var_t_minus_3]
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]
    q_tp3 = vals[var_t_plus_3]

    # 七点三阶差分除以 dt³；函数名保留 five_point，但模板覆盖 t±1/t±2/t±3。
    jerk = (-q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3) / (
        8 * dt**3
    )
    residual = jnp.maximum(0.0, jnp.abs(jerk) - jerk_limit)
    return (residual * weight).flatten()
