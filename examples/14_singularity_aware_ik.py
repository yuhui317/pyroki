"""J-PARSE Velocity IK

Singularity-aware velocity IK using J-PARSE with manipulability visualization.
"""
# 学习导读：以 J-PARSE 做单步速度 IK，显示可操作度并尝试约 50 Hz 更新；不调用 jaxls。

import time

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from pyroki_snippets._jparse import jparse_step


# 每轮做一次 J-PARSE 速度更新，显示指标并补睡眠尝试维持 0.02 秒周期。
def main():
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    # URDF 加载在宿主侧完成，Robot.from_urdf 把运动学参数转成可微 JAX 数据结构。
    robot = pk.Robot.from_urdf(urdf)
    target_link_index = robot.links.names.index(target_link_name)

    # Initial configuration (middle of joint range).
    cfg = np.array((robot.joints.lower_limits + robot.joints.upper_limits) / 2.0)

    # Set up visualizer.
    # 启动交互可视服务；scene/GUI 负责观察与目标输入，不是物理控制器。
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Target gizmo.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )

    # GUI controls.
    gamma_handle = server.gui.add_slider(
        "Gamma (singularity threshold)", 0.01, 0.5, 0.01, 0.1
    )
    nullspace_gain_handle = server.gui.add_slider("Nullspace gain", 0.0, 2.0, 0.05, 0.5)
    manipulability_handle = server.gui.add_number(
        "Manipulability", 0.001, disabled=True
    )
    show_ellipsoid_handle = server.gui.add_checkbox("Show ellipsoid", True)

    # Manipulability ellipsoid visualization.
    manip_ellipse = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability",
        target_link_name=target_link_name,
    )

    # Control loop at ~50Hz.
    # 持续交互/播放循环，本次静态注释检查不会执行或启动它。
    while True:
        start_time = time.time()

        # Take one velocity IK step.
        cfg, info = jparse_step(
            robot=robot,
            cfg=cfg,
            target_link_index=target_link_index,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
            gamma=gamma_handle.value,
            nullspace_gain=nullspace_gain_handle.value,
            dt=0.02,
        )

        # Update metrics and visualization.
        manipulability_handle.value = round(info["manipulability"], 4)
        manip_ellipse.set_visibility(show_ellipsoid_handle.value)
        manip_ellipse.update(cfg)
        urdf_vis.update_cfg(cfg)

        # Sleep to maintain ~50Hz.
        elapsed = time.time() - start_time
        time.sleep(max(0.0, 0.02 - elapsed))


# 只在直接运行脚本时进入主函数/命令行流程，模块导入通常不会运行此入口。
if __name__ == "__main__":
    main()
