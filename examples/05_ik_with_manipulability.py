"""IK with Manipulability

Inverse Kinematics with Manipulability using PyRoKi.
"""
# 学习导读：调节平移 Yoshikawa 可操作度代价，比较同一目标下关节解与可操作椭球变化。

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks


# 读取可操作度滑块权重，求 IK 后更新 Yoshikawa 数值和椭球显示。
def main():
    """Main function for basic IK."""

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
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    value_handle = server.gui.add_number("Yoshikawa Index", 0.001, disabled=True)
    weight_handle = server.gui.add_slider(
        "Manipulability Weight", 0.0, 10.0, 0.001, 0.0
    )
    manip_ellipse = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability",
        target_link_name=target_link_name,
    )

    # 持续交互/播放循环，本次静态注释检查不会执行或启动它。
    while True:
        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik_with_manipulability(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
            manipulability_weight=weight_handle.value,
        )

        manip_ellipse.update(solution)
        value_handle.value = manip_ellipse.manipulability

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


# 只在直接运行脚本时进入主函数/命令行流程，模块导入通常不会运行此入口。
if __name__ == "__main__":
    main()
