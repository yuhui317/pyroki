"""IK with Mimic Joints

This is a simple test to ensure that mimic joints are handled correctly in the IK solver.

We procedurally generate a "zig-zag" chain of links with mimic joints, where:
- the first joint is driven directly,
- and the remaining joints are driven indirectly via mimic joints.
The multipliers alternate between -1 and 1, and the offsets are all 0.
"""
# 学习导读：生成带正/负 multiplier 的 mimic 链 URDF，观察直接驱动自由度如何影响间接驱动关节。

import tempfile
import time

import numpy as np
import pyroki as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf

import pyroki_snippets as pks


# 以字符串生成带 joint_0 线性 mimic 关系的连杆链，供解析与 FK/IK 示例使用。
def create_chain_xml(length: float = 0.2, num_chains: int = 5) -> str:
    # 生成单个带可视圆柱和惯性标签的 link XML；这里不会求动力学。
    def create_link(idx):
        return f"""
                <link name="link_{idx}">
                    <visual>
                    <geometry>
                        <cylinder length="{length}" radius="0.02"/>
                    </geometry>
                    </visual>
                    <inertial>
                    <mass value="0.1"/>
                    <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/>
                    </inertial>
                </link>
            """

    # 生成旋转 joint XML，首关节不写 mimic，其余引用 joint_0 并传 multiplier/offset。
    def create_joint(idx, multiplier=1.0, offset=0.0):
        mimic = f'<mimic joint="joint_0" multiplier="{multiplier}" offset="{offset}"/>'
        return f"""
            <joint name="joint_{idx}" type="revolute">
                <parent link="link_{idx}"/>
                <child link="link_{idx + 1}"/>
                <axis xyz="1 0 0"/>
                <origin xyz="0 0 {length}" rpy="0 0 0"/>
                {mimic if idx != 0 else ""}
                <limit lower="-10.0" upper="10.0" velocity="1.0"/>
            </joint>
            """

    world_joint_origin_z = length / 2.0
    xml = f"""
    <?xml version="1.0"?>
    <robot name="chain_with_mimic_joints">
    <link name="world"></link>
    <joint name="world_joint" type="revolute">
        <parent link="world"/>
        <child link="link_0"/>
        <axis xyz="1 0 0"/>
        <origin xyz="0 0 {world_joint_origin_z}" rpy="0 0 0"/>
        <limit lower="-10.0" upper="10.0" velocity="1.0"/>
    </joint>
    """
    # Create the definition + first link.
    xml += create_link(0)
    xml += create_link(1)
    xml += create_joint(0)

    # Procedurally add more links.
    assert num_chains >= 2
    for idx in range(2, num_chains):
        xml += create_link(idx)
        current_offset = 0.0
        current_multiplier = 1.0 * ((-1) ** (idx % 2))
        xml += create_joint(idx - 1, current_multiplier, current_offset)

    xml += """
    </robot>
    """
    return xml


# 把 XML 写临时 URDF 后加载，创建目标连杆选择与拖拽 IK，可观察 mimic 关节联动。
def main():
    """Main function for basic IK."""

    xml = create_chain_xml(num_chains=10, length=0.1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf") as f:
        f.write(xml)
        f.flush()
        urdf = yourdfpy.URDF.load(f.name)

    # Create robot.
    # URDF 加载在宿主侧完成，Robot.from_urdf 把运动学参数转成可微 JAX 数据结构。
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    # 启动交互可视服务；scene/GUI 负责观察与目标输入，不是物理控制器。
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    target_link_name_handle = server.gui.add_dropdown(
        "Target Link",
        robot.links.names,
        initial_value=robot.links.names[-1],
    )

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.0, 0.1, 0.1), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # 持续交互/播放循环，本次静态注释检查不会执行或启动它。
    while True:
        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name_handle.value,
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
