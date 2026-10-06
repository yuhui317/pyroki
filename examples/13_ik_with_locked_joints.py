"""IK with Locked Joints

Demonstrates joint locking during IK optimization by zeroing out Jacobian columns
for locked joints. Uses checkboxes to dynamically lock/unlock individual joints
without triggering JAX recompilation.
"""
# 学习导读：通过解析位姿雅可比的列掩码锁定关节，以上轮解初始化，并在输出端恢复已锁关节。
# 切换掩码值而不改变形状可复用编译图；这是示例策略，不是通用固定变量约束 API。

import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


# 用 joint_mask 屏蔽位姿雅可比列，以 prev_cfg 初始化，再按解析位姿代价/限位约束求解。
@jdc.jit
def _solve_ik(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    joint_mask: jax.Array,
    prev_cfg: jax.Array,
) -> jax.Array:
    """Solve IK with masked (locked) joints.

    Args:
        robot: Robot model.
        target_link_index: Index of the target link.
        target_wxyz: Target orientation quaternion (w, x, y, z).
        target_position: Target position (x, y, z).
        joint_mask: Array of shape (n_actuated,). 1.0 = optimize, 0.0 = lock.
        prev_cfg: Previous joint configuration (used as initial guess).

    Returns:
        Optimized joint configuration.
    """
    joint_var = robot.joint_var_cls(0)

    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )

    # 明确列出目标/偏好/约束因子；不同示例使用的软硬条件并不相同，应逐项核对。
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            target_pose,
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
            # 这里仅作用于解析位姿因子；下方仍有未加 mask 的 limit_constraint。
            joint_mask=joint_mask,
        ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]),
        )
    )
    return sol[joint_var]


# 返回 log(actual⁻¹·target) 的平移部分范数；存在旋转差时它不是直接 xyz 相减的欧氏距离。
@jdc.jit
def _compute_pose_error(
    robot: pk.Robot,
    joint_cfg: jax.Array,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    """Compute position error between actual and target pose.

    Returns:
        pos_error: Position error (Euclidean distance in meters).
    """
    Ts_world_link = robot.forward_kinematics(joint_cfg)
    actual_pose = jaxlie.SE3(Ts_world_link[target_link_index])

    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )

    pose_error = (actual_pose.inverse() @ target_pose).log()
    pos_error = jnp.linalg.norm(pose_error[:3])

    return pos_error


# 把复选框映射为固定形状关节掩码，显示解并在最后恢复锁定关节旧值。
def main():
    """Main function for IK with locked joints."""

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    # URDF 加载在宿主侧完成，Robot.from_urdf 把运动学参数转成可微 JAX 数据结构。
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    # 启动交互可视服务；scene/GUI 负责观察与目标输入，不是物理控制器。
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )

    # Create GUI folder for joint locking controls.
    reset_requested = [False]  # Use list to allow mutation in callback
    with server.gui.add_folder("Optimization"):
        # Add button to lock/unlock all.
        lock_all_btn = server.gui.add_button_group(
            label="Control (all)", options=["Lock", "Unlock", "Reset"]
        )

        # Create checkboxes for each joint (checked = unlocked/optimized).
        joint_checkboxes = []
        with server.gui.add_folder("Robot Joints"):
            for i, name in enumerate(robot.joints.actuated_names):
                # Default: all joints unlocked (checked)
                cb = server.gui.add_checkbox(f"{name}", initial_value=True)
                joint_checkboxes.append(cb)

        # 按钮回调切换全部关节勾选状态，Reset 用可变列表通知主循环重置初值。
        @lock_all_btn.on_click
        def _(_):
            for cb in joint_checkboxes:
                if lock_all_btn.value == "Lock":
                    cb.value = False
                elif lock_all_btn.value in ("Unlock", "Reset"):
                    cb.value = True

            if lock_all_btn.value == "Reset":
                reset_requested[0] = True

    # Metrics display.
    with server.gui.add_folder("Metrics"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        pos_error_handle = server.gui.add_number(
            "Position Error (m)", 0.001, disabled=True
        )

    # Get target link index.
    target_link_index = robot.links.names.index(target_link_name)

    # Store previous solution for locked joints.
    default_cfg = np.array(robot.joint_var_cls(0).default_factory())
    prev_cfg = default_cfg.copy()

    # 持续交互/播放循环，本次静态注释检查不会执行或启动它。
    while True:
        # Handle reset request.
        if reset_requested[0]:
            prev_cfg = default_cfg.copy()
            reset_requested[0] = False

        # Build joint mask from checkboxes (1.0 = optimize, 0.0 = lock).
        joint_mask = np.array([float(cb.value) for cb in joint_checkboxes])

        # Prepare arguments.
        target_link_idx_jax = jnp.array(target_link_index, dtype=jnp.int32)
        target_wxyz_jax = jnp.array(ik_target.wxyz)
        target_position_jax = jnp.array(ik_target.position)
        joint_mask_jax = jnp.array(joint_mask)
        prev_cfg_jax = jnp.array(prev_cfg)

        # Solve IK.
        start_time = time.time()
        solution = _solve_ik(
            robot=robot,
            target_link_index=target_link_idx_jax,
            target_wxyz=target_wxyz_jax,
            target_position=target_position_jax,
            # 这里仅作用于解析位姿因子；下方仍有未加 mask 的 limit_constraint。
            joint_mask=joint_mask_jax,
            prev_cfg=prev_cfg_jax,
        )

        # Update timing.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Compute accuracy.
        pos_err = _compute_pose_error(
            robot=robot,
            joint_cfg=solution,
            target_link_index=target_link_idx_jax,
            target_wxyz=target_wxyz_jax,
            target_position=target_position_jax,
        )
        pos_error_handle.value = (
            # 原 GUI 标签写 m，但此更新乘 1000；这里只记录实现，未改显示单位。
            0.99 * pos_error_handle.value + 0.01 * float(pos_err) * 1000
        )

        # For locked joints, ensure they stay at the previous value.
        # (The optimizer should produce zero update, but this is a safety net.)
        solution_np = np.array(solution)
        for i, cb in enumerate(joint_checkboxes):
            if not cb.value:  # Joint is locked
                # 输出端再次恢复已锁关节，避免其他代价导致锁定列仍发生配置改变。
                solution_np[i] = prev_cfg[i]

        # Update previous config for next iteration.
        prev_cfg = solution_np.copy()

        # Update visualizer.
        urdf_vis.update_cfg(solution_np)


# 只在直接运行脚本时进入主函数/命令行流程，模块导入通常不会运行此入口。
if __name__ == "__main__":
    main()
