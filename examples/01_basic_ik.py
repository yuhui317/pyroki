"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""
# 学习导读：首个学习入口，加载 Panda URDF→构造 Robot→拖动末端目标→调用 solve_ik→更新 Viser。
# 浏览器可视化只显示求得的关节配置，不向真实机器人发送命令。

import time

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

import pyroki_snippets as pks


# 加载 Panda 模型、创建目标控制器，在循环中调用基本 IK 并更新 URDF 可视姿态。
def main():
    """Main function for basic IK."""

    urdf = load_robot_description("panda_description")
    # --- Alternative: Load a custom URDF file ---
    # import yourdfpy
    # from pathlib import Path
    # urdf_path = Path("path/to/your/robot.urdf")
    # urdf = yourdfpy.URDF.load(urdf_path)
    # --------------------------------------------
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

    # 持续交互/播放循环，本次静态注释检查不会执行或启动它。
    while True:
        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


# 只在直接运行脚本时进入主函数/命令行流程，模块导入通常不会运行此入口。
if __name__ == "__main__":
    main()
