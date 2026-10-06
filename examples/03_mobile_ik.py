"""Mobile IK

Same as 01_basic_ik.py, but with a mobile base!
"""
# 学习导读：Fetch 的移动基座与关节联合 IK；基座只开放 xy 平移与 z 轴旋转，以上轮结果初始化。

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks


# 使用上一轮基座/关节状态求联合 IK，并同步更新基座 frame 与关节可视模型。
def main():
    """Main function for IK with a mobile base.
    The base is fixed along the xy plane, and is biased towards being at the origin.
    """

    urdf = load_robot_description("fetch_description")
    target_link_name = "gripper_link"

    # Create robot.
    # URDF 加载在宿主侧完成，Robot.from_urdf 把运动学参数转成可微 JAX 数据结构。
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    # 启动交互可视服务；scene/GUI 负责观察与目标输入，不是物理控制器。
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0.707, 0, -0.707)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    cfg = np.array(robot.joint_var_cls(0).default_factory())

    # 持续交互/播放循环，本次静态注释检查不会执行或启动它。
    while True:
        # Solve IK.
        start_time = time.time()
        base_pos, base_wxyz, cfg = pks.solve_ik_with_base(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
            fix_base_position=(False, False, True),  # Only free along xy plane.
            fix_base_orientation=(True, True, False),  # Free along z-axis rotation.
            prev_pos=base_frame.position,
            prev_wxyz=base_frame.wxyz,
            prev_cfg=cfg,
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(cfg)
        base_frame.position = np.array(base_pos)
        base_frame.wxyz = np.array(base_wxyz)


# 只在直接运行脚本时进入主函数/命令行流程，模块导入通常不会运行此入口。
if __name__ == "__main__":
    main()
