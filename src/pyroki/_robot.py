# 学习导读：URDF 经解析得到关节/连杆 PyTree，再以 SE(3) 指数映射和树拓扑计算可微正运动学。
# 输入末轴是直接驱动关节，输出末轴 7 为四元数 wxyz 与平移 xyz；固定/模仿关节由 JointInfo 展开。
from __future__ import annotations

import jax
import jax_dataclasses as jdc
import jaxlie
import jaxls
import yourdfpy
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from ._robot_urdf_parser import JointInfo, LinkInfo, RobotURDFParser


# 以 PyTree 存放机器人运动学数据；joint_var_cls 和拓扑字段是静态结构，数值叶节点参与 JAX 变换。
@jdc.pytree_dataclass
class Robot:
    """A differentiable robot kinematics tree."""

    joints: JointInfo
    """Joint information for the robot."""

    links: LinkInfo
    """Link information for the robot."""

    joint_var_cls: jdc.Static[type[jaxls.Var[Array]]]
    """Variable class for the robot configuration."""

    # 解析 URDF 并创建配置变量类，默认初值为直接驱动关节限位中点；显式初值当前断言必须为一维。
    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        default_joint_cfg: Float[ArrayLike, "*batch actuated_count"] | None = None,
    ) -> Robot:
        """
        Loads a robot kinematic tree from a URDF.
        Internally tracks a topological sort of the joints.

        Args:
            urdf: The URDF to load the robot from.
            default_joint_cfg: The default joint configuration to use for optimization.
        """
        joints, links = RobotURDFParser.parse(urdf)

        # Compute default joint configuration.
        if default_joint_cfg is None:
            # 默认配置取有限位置限位中点；continuous 在解析器中若无范围会先使用 ±π。
            default_joint_cfg = (joints.lower_limits + joints.upper_limits) / 2
        else:
            default_joint_cfg = jnp.array(default_joint_cfg)
        assert default_joint_cfg.shape == (joints.num_actuated_joints,)

        # Variable class for the robot configuration.
        # 每个 Robot 实例动态创建关节变量类型，默认值捕获其关节初值；变量 ID 区分优化中的不同配置。
        class JointVar(  # pylint: disable=missing-class-docstring
            jaxls.Var[Array],
            default_factory=lambda: default_joint_cfg,
        ): ...

        robot = Robot(
            joints=joints,
            links=links,
            joint_var_cls=JointVar,
        )

        return robot

    # JIT 正运动学：输入 [*batch,N_act]，输出 [*batch,N_link,7]，按 links.names 保持连杆顺序。
    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch actuated_count"],
        unroll_fk: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch link_count 7"]:
        """Run forward kinematics on the robot's links, in the provided configuration.

        Computes the world pose of each link frame. The result is ordered
        corresponding to `self.link.names`.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch actuated_count)`.

        Returns:
            The SE(3) transforms of the links, ordered by `self.link.names`,
            in the format `(*batch, link_count, wxyz_xyz)`.
        """
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.joints.num_actuated_joints)
        return self._link_poses_from_joint_poses(
            self._forward_kinematics_joints(cfg, unroll_fk)
        )

    # 用连杆的父关节索引取位姿，根连杆使用单位变换；实际输入关节轴覆盖全部关节。
    def _link_poses_from_joint_poses(
        self, Ts_world_joint: Float[Array, "*batch actuated_count 7"]
    ) -> Float[Array, "*batch link_count 7"]:
        (*batch_axes, _, _) = Ts_world_joint.shape
        # Get the link poses.
        # self.links.parent_joint_indices is Static[tuple], cast to array for indexing
        link_parent_indices = jnp.array(
            self.links.parent_joint_indices, dtype=jnp.int32
        )

        # 根连杆无父关节，最终输出单位位姿；因此纯 Robot FK 的世界实际上锚定在 base link。
        base_link_mask = link_parent_indices == -1
        parent_joint_indices = jnp.where(base_link_mask, 0, link_parent_indices)
        identity_pose = jaxlie.SE3.identity().wxyz_xyz
        Ts_world_link = jnp.where(
            base_link_mask[..., None],
            identity_pose,
            Ts_world_joint[..., parent_joint_indices, :],
        )
        assert Ts_world_link.shape == (*batch_axes, self.links.num_links, 7)
        return Ts_world_link

    # 将直接驱动配置展开为全部关节，再按拓扑累计父子 SE(3) 变换，返回 URDF 原关节顺序。
    def _forward_kinematics_joints(
        self,
        cfg: Float[Array, "*batch actuated_count"],
        unroll_fk: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch joint_count 7"]:
        (*batch_axes, _) = cfg.shape
        assert cfg.shape == (*batch_axes, self.joints.num_actuated_joints)

        # Calculate full configuration using the dedicated method
        # mimic 位置按 multiplier·q+offset 展开，固定关节取零，q_full 末轴覆盖全部关节。
        q_full = self.joints.get_full_config(cfg)

        # Calculate delta transforms using the effective config and twists for all joints.
        # twist 末轴 6 按线分量/角分量排列；乘关节值后用 exp 得到可变关节变换。
        tangents = self.joints.twists * q_full[..., None]
        assert tangents.shape == (*batch_axes, self.joints.num_joints, 6)
        delta_Ts = jaxlie.SE3.exp(tangents)  # Shape: (*batch_axes, self.joint.count, 7)

        # Combine constant parent transform with variable joint delta transform.
        # 固定 origin 在左、关节增量在右：T_parent_child = T_origin·exp(twist·q)。
        Ts_parent_child = (
            jaxlie.SE3(self.joints.parent_transforms) @ delta_Ts
        ).wxyz_xyz
        assert Ts_parent_child.shape == (*batch_axes, self.joints.num_joints, 7)

        # Topological sort helpers
        # Static tuples must be cast to arrays for advanced indexing/argsort
        # 静态 tuple 临时转数组用于索引；先按父优先序计算，最后逆映射回原 URDF 顺序。
        topo_sort_inv = jnp.array(self.joints._topo_sort_inv, dtype=jnp.int32)
        topo_order = jnp.argsort(topo_sort_inv)

        Ts_parent_child_sorted = Ts_parent_child[..., topo_sort_inv, :]

        parent_indices = jnp.array(self.joints.parent_indices, dtype=jnp.int32)
        parent_orig_for_sorted_child = parent_indices[topo_sort_inv]
        idx_parent_joint_sorted = jnp.where(
            parent_orig_for_sorted_child == -1,
            -1,
            topo_order[parent_orig_for_sorted_child],
        )

        # Compute link transforms relative to world, indexed by sorted *joint* index.
        # fori_loop 的一步：先读取已计算父节点位姿，再乘当前父子变换，写入排序后数组。
        def compute_transform(i: int, Ts_world_link_sorted: Array) -> Array:
            parent_sorted_idx = idx_parent_joint_sorted[i]
            T_world_parent_link = jnp.where(
                parent_sorted_idx == -1,
                jaxlie.SE3.identity().wxyz_xyz,
                Ts_world_link_sorted[..., parent_sorted_idx, :],
            )
            return Ts_world_link_sorted.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent_link)
                    @ jaxlie.SE3(Ts_parent_child_sorted[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_link_init_sorted = jnp.zeros((*batch_axes, self.joints.num_joints, 7))
        # 用 JAX 控制流迭代静态关节数量，支持 batch 广播；unroll_fk 控制是否展开循环。
        Ts_world_link_sorted = jax.lax.fori_loop(
            lower=0,
            upper=self.joints.num_joints,
            body_fun=compute_transform,
            init_val=Ts_world_link_init_sorted,
            unroll=unroll_fk,
        )

        Ts_world_link_joint_indexed = Ts_world_link_sorted[..., topo_order, :]
        assert Ts_world_link_joint_indexed.shape == (
            *batch_axes,
            self.joints.num_joints,
            7,
        )  # This is the link poses indexed by parent *joint* index.

        return Ts_world_link_joint_indexed
