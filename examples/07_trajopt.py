"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""
# 学习导读：选择 Panda/UR5，优化跨越墙体的整段关节轨迹，随后用时间滑块播放；并未连接机器人控制。

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pyroki as pk
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks


# 为机器人设置起止目标和墙障碍，离线求轨迹后用滑块/自动循环播放。
def main(robot_name: Literal["ur5", "panda"] = "panda"):
    if robot_name == "ur5":
        urdf = load_robot_description("ur5_description")
        down_wxyz = np.array([0.707, 0, 0.707, 0])
        target_link_name = "ee_link"
        sphere_json_path = Path(__file__).parent / "assets" / "ur5_spheres.json"

        # For UR5 it's important to initialize the robot in a safe configuration;
        # the zero-configuration puts the robot aligned with the wall obstacle.
        default_cfg = np.zeros(6)
        default_cfg[1] = -1.308
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)

    elif robot_name == "panda":
        urdf = load_robot_description("panda_description")
        target_link_name = "panda_hand"
        down_wxyz = np.array([0, 0, 1, 0])  # for panda!
        sphere_json_path = Path(__file__).parent / "assets" / "panda_spheres.json"
        # URDF 加载在宿主侧完成，Robot.from_urdf 把运动学参数转成可微 JAX 数据结构。
        robot = pk.Robot.from_urdf(urdf)

    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)
    robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )

    # Define the trajectory problem:
    # - number of timesteps, timestep size
    timesteps, dt = 50, 0.02
    # - the start and end poses.
    start_pos, end_pos = np.array([0.5, -0.35, 0.2]), np.array([0.5, 0.35, 0.2])

    # Define the obstacles:
    # - Ground
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    # - Wall (using Box collision geometry)
    wall_height = 0.4
    wall_width = 0.1
    wall_length = 0.4
    wall_coll = pk.collision.Box.from_extent(
        extent=np.array([wall_length, wall_width, wall_height]),
        position=np.array([0.5, 0.0, wall_height / 2]),
    )
    world_coll = [ground_coll, wall_coll]

    traj = pks.solve_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_name,
        start_pos,
        down_wxyz,
        end_pos,
        down_wxyz,
        timesteps,
        dt,
    )
    traj = np.array(traj)

    # Visualize!
    # 启动交互可视服务；scene/GUI 负责观察与目标输入，不是物理控制器。
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf)
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
    server.scene.add_mesh_trimesh(
        "wall_box",
        trimesh.creation.box(
            extents=(wall_length, wall_width, wall_height),
            transform=trimesh.transformations.translation_matrix(
                np.array([0.5, 0.0, wall_height / 2])
            ),
        ),
    )
    for name, pos in zip(["start", "end"], [start_pos, end_pos]):
        server.scene.add_frame(
            f"/{name}",
            position=pos,
            wxyz=down_wxyz,
            axes_length=0.05,
            axes_radius=0.01,
        )

    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    # 持续交互/播放循环，本次静态注释检查不会执行或启动它。
    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps

        urdf_vis.update_cfg(traj[slider.value])
        time.sleep(1.0 / 10.0)


# 只在直接运行脚本时进入主函数/命令行流程，模块导入通常不会运行此入口。
if __name__ == "__main__":
    tyro.cli(main)
