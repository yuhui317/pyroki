"""Hand Retargeting (Fancy)

Shadow Hand retargeting example, with costs to maintain contact with the object.
Find and unzip the shadowhand URDF at `assets/hand_retargeting/shadowhand_urdf.zip`.
"""
# 学习导读：Shadow Hand 进阶重定向，在简版上加入尺度正则和带掩码的物体接触位置残差。

import pickle
import time
from pathlib import Path
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import trimesh
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from retarget_helpers._utils import (
    MANO_TO_SHADOW_MAPPING,
    create_conn_tree,
    get_mapping_from_mano_to_shadow,
)


# 声明重定向残差的各项权重，字段用于 GUI 调节与传入 JIT 求解函数。
class RetargetingWeights(TypedDict):
    local_alignment: float
    """Local alignment weight, by matching the relative joint/keypoint positions and angles."""
    global_alignment: float
    """Global alignment weight, by matching the keypoint positions to the robot."""
    contact: float
    """Contact weight, to maintain contact between the robot and the object."""
    contact_margin: float
    """Contact margin, to stop penalizing contact when the robot is already close to the object."""
    joint_smoothness: float
    """Joint smoothness weight."""
    root_smoothness: float
    """Root translation smoothness weight."""


# 读取指定人体关键点与机器人模型，准备映射/权重，求整段轨迹并在 Viser 中逐帧播放。
def main():
    """Main function for hand retargeting."""

    asset_dir = Path(__file__).parent / "retarget_helpers" / "hand"

    robot_urdf_path = asset_dir / "shadowhand_urdf" / "shadow_hand_right.urdf"

    # 把 URDF 中资源路径相对于解压目录解析，供 yourdfpy 加载手部模型。
    def filename_handler(fname: str) -> str:
        base_path = robot_urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)

    try:
        urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Please unzip the included URDF at `retarget_helpers/hand/shadowhand_urdf.zip`."
        )

    # URDF 加载在宿主侧完成，Robot.from_urdf 把运动学参数转成可微 JAX 数据结构。
    robot = pk.Robot.from_urdf(urdf)

    # Get the mapping from MANO to Shadow Hand joints.
    shadow_link_idx, mano_joint_idx = get_mapping_from_mano_to_shadow(robot)

    # Create a mask for the MANO joints that are connected to the Shadow Hand.
    mano_mask = create_conn_tree(robot, shadow_link_idx)

    # Load source motion data.
    dexycb_motion_path = asset_dir / "dexycb_motion.pkl"
    with open(dexycb_motion_path, "rb") as f:
        dexycb_motion_data = pickle.load(f, encoding="latin1")

    # Load keypoints.
    keypoints = dexycb_motion_data["world_hand_joints"]
    assert not onp.isnan(keypoints).any()
    num_timesteps = keypoints.shape[0]
    num_mano_joints = len(MANO_TO_SHADOW_MAPPING)

    # Load mano hand contact information -- these are lists of lists,
    # len(contact_points_per_frame) = num_timesteps,
    # len(contact_points_per_frame[i]) = number of contacts in frame i,
    contact_points_per_frame = dexycb_motion_data["contact_object_points"]
    contact_indices_per_frame = dexycb_motion_data["contact_joint_indices"]

    # Now, we're going to pad this info + make a mask to indicate the padded regions.
    # We will also track the shadowhand joint indices, NOT the MANO joint indices.
    # 把每帧可变数量接触点补到固定 C_max，并以 [T,C_max] bool 掩码排除 padding。
    max_num_contacts = max(len(c) for c in contact_points_per_frame)
    padded_contact_points_per_frame = onp.zeros((num_timesteps, max_num_contacts, 3))
    padded_contact_indices_per_frame = onp.zeros(
        (num_timesteps, max_num_contacts), dtype=onp.int32
    )
    padded_contact_mask = onp.zeros((num_timesteps, max_num_contacts), dtype=onp.bool_)
    for i in range(num_timesteps):
        num_contacts = len(contact_points_per_frame[i])
        if num_contacts == 0:
            continue
        contact_shadowhand_indices = [
            robot.links.names.index(MANO_TO_SHADOW_MAPPING[j])
            for j in contact_indices_per_frame[i]
        ]
        padded_contact_points_per_frame[i, :num_contacts] = contact_points_per_frame[i]
        padded_contact_indices_per_frame[i, :num_contacts] = contact_shadowhand_indices
        padded_contact_mask[i, :num_contacts] = True

    # Load the object.
    object_mesh_vertices = dexycb_motion_data["object_mesh_vertices"]
    object_mesh_faces = dexycb_motion_data["object_mesh_faces"]
    object_pose_list = dexycb_motion_data["object_poses"]  # (N, 4, 4)
    mesh = trimesh.Trimesh(object_mesh_vertices, object_mesh_faces)

    # 启动交互可视服务；scene/GUI 负责观察与目标输入，不是物理控制器。
    server = viser.ViserServer()

    # We will transform everything by the transform below, for aesthetics.
    server.scene.add_frame(
        "/scene_offset",
        show_axes=False,
        position=(-0.15415953, -0.73598871, 0.93434792),
        wxyz=(-0.381870867, 0.92421569, 0.0, 2.0004992e-32),
    )
    base_frame = server.scene.add_frame("/scene_offset/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/scene_offset/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    object_handle = server.scene.add_mesh_trimesh("/scene_offset/object", mesh)
    server.scene.add_grid("/grid", 2.0, 2.0)

    default_weights = RetargetingWeights(
        local_alignment=10.0,
        global_alignment=1.0,
        contact=5.0,
        contact_margin=0.01,
        joint_smoothness=2.0,
        root_smoothness=2.0,
    )

    weights = pk.viewer.WeightTuner(
        server,
        default_weights,  # type: ignore
    )

    Ts_world_root, joints = None, None

    # 按钮回调重算整段根位姿和关节轨迹，更新外层状态并在计算期间禁用按钮。
    def generate_trajectory():
        nonlocal Ts_world_root, joints
        gen_button.disabled = True
        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            target_keypoints=keypoints,
            shadow_hand_link_retarget_indices=shadow_link_idx,
            mano_joint_retarget_indices=mano_joint_idx,
            mano_mask=mano_mask,
            contact_points_per_frame=jnp.array(padded_contact_points_per_frame),
            contact_indices_per_frame=jnp.array(padded_contact_indices_per_frame),
            contact_mask=jnp.array(padded_contact_mask),
            weights=weights.get_weights(),  # type: ignore
        )
        gen_button.disabled = False

    gen_button = server.gui.add_button("Retarget!")
    gen_button.on_click(lambda _: generate_trajectory())

    generate_trajectory()
    assert Ts_world_root is not None and joints is not None

    # 持续交互/播放循环，本次静态注释检查不会执行或启动它。
    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame.position = onp.array(Ts_world_root.wxyz_xyz[tstep][4:])
            urdf_vis.update_cfg(onp.array(joints[tstep]))

            server.scene.add_point_cloud(
                "/scene_offset/target_keypoints",
                onp.array(keypoints[tstep]).reshape(-1, 3),
                onp.array((0, 0, 255))[None]
                .repeat(num_mano_joints, axis=0)
                .reshape(-1, 3),
                point_size=0.005,
                point_shape="sparkle",
            )
            server.scene.add_point_cloud(
                "/scene_offset/contact_points",
                onp.array(contact_points_per_frame[tstep]).reshape(-1, 3),
                onp.array((255, 0, 0))[None]
                .repeat(len(contact_points_per_frame[tstep]), axis=0)
                .reshape(-1, 3),
                point_size=0.005,
                point_shape="circle",
            )
            object_handle.position = object_pose_list[tstep][:3, 3]
            object_handle.wxyz = R.from_matrix(object_pose_list[tstep][:3, :3]).as_quat(
                scalar_first=True
            )

        time.sleep(0.05)


# 建立每帧关节/根位姿变量和跨帧共享尺度，组合局部/全局对齐与相关平滑因子后求解。
@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    target_keypoints: jnp.ndarray,
    shadow_hand_link_retarget_indices: jnp.ndarray,
    mano_joint_retarget_indices: jnp.ndarray,
    mano_mask: jnp.ndarray,
    contact_points_per_frame: jnp.ndarray,
    contact_indices_per_frame: jnp.ndarray,
    contact_mask: jnp.ndarray,
    weights: RetargetingWeights,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the retargeting problem."""

    n_retarget = len(mano_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    # Variables.
    # 以 N_retarget×N_retarget 尺度矩阵调整机器人局部向量；重复 ID=0 意味着全序列共享一份值。
    class ManoJointsScaleVar(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((n_retarget, n_retarget))
    ): ...

    # 3 维平移偏移变量，最终左乘根位姿；简版/手部版本未在因子中使用它，不能据声明推断已优化其值。
    class OffsetVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros((3,))): ...

    # 每个时间步一个关节变量 ID，输出轨迹配置末轴为 N_act；根位姿同样按时间 ID 建立。
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    # 所有时间行的尺度变量 ID 都是 0，因此共享同一个 N_retarget×N_retarget 矩阵。
    var_smpl_joints_scale = ManoJointsScaleVar(jnp.zeros(timesteps))
    var_offset = OffsetVar(jnp.zeros(timesteps))

    # Costs and constraints.
    costs: list[jaxls.Cost] = []

    # 比较映射关键点的成对相对向量与夹角，以连接掩码排除非对应连接/对角线。
    @jaxls.Cost.factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_smpl_joints_scale: ManoJointsScaleVar,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Retargeting factor, with a focus on:
        - matching the relative joint/keypoint positions (vectors).
        - and matching the relative angles between the vectors.
        """
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link

        mano_pos = keypoints[jnp.array(mano_joint_retarget_indices)]
        robot_pos = T_world_link.translation()[
            jnp.array(shadow_hand_link_retarget_indices)
        ]

        # NxN grid of relative positions.
        delta_mano = mano_pos[:, None] - mano_pos[None, :]
        # 两两位置相减形成 [N_retarget,N_retarget,3]；对角线为零，后续显式掩掉。
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        # Vector regularization.
        # 把二维尺度矩阵扩成 [N,N,1]，广播到相对位置差 [N,N,3] 的 xyz。
        position_scale = var_values[var_smpl_joints_scale][..., None]
        residual_position_delta = (
            (delta_mano - delta_robot * position_scale)
            * (1 - jnp.eye(delta_mano.shape[0])[..., None])
            * mano_mask[..., None]
        )

        # Vector angle regularization.
        delta_mano_normalized = delta_mano / jnp.linalg.norm(
            delta_mano + 1e-6, axis=-1, keepdims=True
        )
        delta_robot_normalized = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True
        )
        # 归一化相对向量点积是夹角 cos，这里惩罚 1−cos，而不是直接 arccos 角度。
        residual_angle_delta = 1 - (delta_mano_normalized * delta_robot_normalized).sum(
            axis=-1
        )
        residual_angle_delta = (
            residual_angle_delta
            * (1 - jnp.eye(residual_angle_delta.shape[0]))
            * mano_mask
        )

        residual = (
            jnp.concatenate(
                [
                    residual_position_delta.flatten(),
                    residual_angle_delta.flatten(),
                ],
                axis=0,
            )
            * weights["local_alignment"]
        )
        return residual

    # 约束尺度接近 1、对称且非负；这些都以软残差实现，并非硬尺度边界。
    @jaxls.Cost.factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_smpl_joints_scale: ManoJointsScaleVar,
    ) -> jax.Array:
        """Regularize the scale of the retargeted joints."""
        # Close to 1.
        res_0 = (var_values[var_smpl_joints_scale] - 1.0).flatten() * 1.0
        # Symmetric.
        res_1 = (
            var_values[var_smpl_joints_scale] - var_values[var_smpl_joints_scale].T
        ).flatten() * 100.0
        # Non-negative.
        # 负尺度才激活非负性惩罚；目前是软项，有限迭代不保证严格非负。
        res_2 = jnp.clip(-var_values[var_smpl_joints_scale], min=0).flatten() * 100.0
        return jnp.concatenate([res_0, res_1, res_2])

    # 把 FK 连杆乘世界根位姿后，与人体对应关键点进行世界坐标位置软对齐。
    @jaxls.Cost.factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Soft cost to align the human keypoints to the robot, in the world frame."""
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[shadow_hand_link_retarget_indices]
        keypoint_pos = keypoints[mano_joint_retarget_indices]
        return (link_pos - keypoint_pos).flatten() * weights["global_alignment"]

    # 只惩罚相邻根位姿的平移差；这个手部实现没有直接惩罚根旋转差。
    @jaxls.Cost.factory
    def root_smoothness(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        """Smoothness cost for the robot root translation."""
        return (
            var_values[var_Ts_world_root].translation()
            - var_values[var_Ts_world_root_prev].translation()
        ).flatten() * weights["root_smoothness"]

    # 对有效接触点与对应机器人连杆位置按坐标做 margin 内零惩罚；预处理传入每帧 [C,3] 点。
    @jaxls.Cost.factory
    def contact_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        contact_points: jax.Array,  # (J, P, 3)
        contact_indices: jax.Array,  # (J,) - Actual robot joint indices.
        contact_points_mask: jax.Array,  # (J, P)
    ) -> jax.Array:
        """Cost for maintaining contact between specified robot joints and object points."""
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_T_world_root]
        T_world_link = T_world_root @ T_root_link

        contact_joint_positions_world = T_world_link.translation()[contact_indices]

        # Contact points are already in world frame (as processed in dexycb).
        # Calculate distances from each joint to its set of contact points
        # Shape contact_points: (J, P, 3), contact_joint_positions_world: (J, 3)
        # We want distance between joint J and points P for that joint.
        # residual: (J, P, 3)
        # 预处理实际单帧 contact_points 为 [C,3]、indices/mask 为 [C]；原注释中的 J/P 轴未在数据准备中体现。
        residual = contact_points - contact_joint_positions_world

        # Penalize distance beyond a margin.
        residual_penalty = jnp.maximum(
            jnp.abs(residual) - weights["contact_margin"], 0.0
        )  # (J, P, 3)

        # Apply mask.
        residual_penalty = (
            residual_penalty * contact_points_mask[..., None]
        )  # (J, P, 3)
        residual = residual_penalty.flatten() * weights["contact"]

        return residual

    # 明确列出目标/偏好/约束因子；不同示例使用的软硬条件并不相同，应逐项核对。
    costs = [
        # Costs that are relatively self-contained to the robot.
        retargeting_cost(
            var_Ts_world_root,
            var_joints,
            var_smpl_joints_scale,
            target_keypoints,
        ),
        scale_regularization(var_smpl_joints_scale),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([weights["joint_smoothness"]]),
        ),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.array([0.2]),
        ),
        # Costs that are scene-centric.
        pc_alignment_cost(
            var_Ts_world_root,
            var_joints,
            target_keypoints,
        ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        contact_cost(
            var_T_world_root=var_Ts_world_root,
            var_robot_cfg=var_joints,
            contact_points=contact_points_per_frame,
            contact_indices=contact_indices_per_frame,
            contact_points_mask=contact_mask,
        ),
    ]

    costs.append(
        pk.costs.limit_constraint(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
        ),
    )

    # 统一交给 LeastSquaresProblem 分析并求解，数值收敛和设备执行均不由本次注释任务验证。
    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[
                var_joints,
                var_Ts_world_root,
                var_smpl_joints_scale,
                var_offset,
            ],
        )
        .analyze()
        .solve()
    )
    transform = solution[var_Ts_world_root]
    # 最后把 offset 左乘到根位姿；该变量是否参与优化要检查当前版本具体因子列表。
    offset = solution[var_offset]
    transform = jaxlie.SE3.from_translation(offset) @ transform
    return transform, solution[var_joints]


# 只在直接运行脚本时进入主函数/命令行流程，模块导入通常不会运行此入口。
if __name__ == "__main__":
    main()
