"""Batched URDF rendering in Viser. This requires features not yet available in
the PyPi version of Viser, we can re-enable it after the next Viser release."""
# 学习导读：批量 URDF 渲染实现被上游整段注释停用，目前模块只有说明文字，没有可调用类。
# 以下新增说明只帮助阅读停用代码，不解除注释、不启用 Viser 特性；历史依赖描述不能当作当前运行验证。

# import jax.numpy as jnp
# import jaxlie
# import numpy as onp
# import viser
# import viser.transforms as vtf
# import yourdfpy
# from jax.typing import ArrayLike
# from pyroki._robot import Robot
#
# from viser.extras import BatchedGlbHandle
#
#
# 停用类 BatchedURDF：原设计让同一 URDF 的多套配置共用批量 mesh 实例，当前不可调用。
# class BatchedURDF:
#     """
#     Helper for rendering batched URDFs in Viser.
#     Similar to `viser.extras.ViserUrdf`, but batched using `pyroki`'s batched forward kinematics.
#
#     If num_robots > 1 then the URDF meshes are rendered as batched meshes (instancing),
#     otherwise they are rendered as individual meshes.
#
#     Args:
#         target: Viser server or client handle to add URDF to.
#         urdf: URDF to render.
#         num_robots: Number of robots in the batch.
#         root_node_name: Name of the root node in the Viser scene.
#     """
#
# 停用构造器：创建 Robot、批量基座单位位姿和 mesh 句柄，要求 Viser 的实例化渲染接口。
#     def __init__(
#         self,
#         target: viser.ViserServer | viser.ClientHandle,
#         urdf: yourdfpy.URDF,
#         num_robots: int = 1,
#         root_node_name: str = "/",
#     ):
#         assert root_node_name.startswith("/")
#         robot = Robot.from_urdf(urdf)
#
#         self._urdf = urdf
#         self._robot = robot
#         self._target = target
#         self._root_node_name = root_node_name
#         self._num_robots = num_robots
#
#         # Initialize base transforms to identity.
#         self._base_transforms = jaxlie.SE3.identity(batch_axes=(num_robots,))
#         self._last_cfg = None  # Store the last configuration.
#
#         self._populate()
#
# 停用初始化：按 URDF 场景把局部 mesh 附着 link，创建单模型或批量 mesh 显示句柄。
#     def _populate(self):
#         # Initialize with the correct batch size.
#         dummy_transform = vtf.SE3.identity(batch_axes=(self._num_robots,))
#         dummy_position = dummy_transform.translation()
#         dummy_wxyz = dummy_transform.rotation().wxyz
#
#         self._meshes: dict[str, list[BatchedGlbHandle | viser.GlbHandle]] = {}
#         self._link_to_meshes: dict[str, onp.ndarray] = {}
#
#         # Check if add_batched_meshes_trimesh is available.
#         if (
#             not hasattr(self._target.scene, "add_batched_meshes_trimesh")
#             and self._num_robots > 1
#         ):
#             raise NotImplementedError(
#                 "num_robots > 1, but viser doesn't support instancing "
#                 "(add_batched_meshes_trimesh is not available)."
#             )
#
#         for mesh_name, mesh in self._urdf.scene.geometry.items():
#             link_name = self._urdf.scene.graph.transforms.parents[mesh_name]
#             if link_name not in self._meshes:
#                 self._meshes[link_name] = []
#
#             # Put mesh in the link frame.
#             T_parent_child = self._urdf.get_transform(
#                 mesh_name, self._urdf.scene.graph.transforms.parents[mesh_name]
#             )
#             mesh = mesh.copy()
#             mesh.apply_transform(T_parent_child)
#
#             if self._num_robots > 1:
#                 self._meshes[link_name].append(
#                     self._target.scene.add_batched_meshes_trimesh(  # type: ignore[attr-defined]
#                         f"{self._root_node_name}/{mesh_name}",
#                         mesh,
#                         batched_positions=dummy_position,
#                         batched_wxyzs=dummy_wxyz,
#                         lod="auto",
#                     )
#                 )
#             else:
#                 self._meshes[link_name].append(
#                     self._target.scene.add_mesh_trimesh(
#                         f"{self._root_node_name}/{mesh_name}",
#                         mesh,
#                         position=dummy_position[0],
#                         wxyz=dummy_wxyz[0],
#                     )
#                 )
#
#             self._link_to_meshes[link_name] = T_parent_child
#
# 停用清理：逐 link 删除所有 mesh 显示句柄。
#     def remove(self):
#         for meshes in self._meshes.values():
#             for mesh in meshes:
#                 mesh.remove()
#
# 停用基座更新：接受每机器人一套 wxyz_xyz 变换，并刷新已存关节配置的可视姿态。
#     def update_base_frame(self, base_transforms: ArrayLike):
#         """
#         Update the base transforms for each robot in the batch.
#
#         Args:
#             base_transforms: New base transforms. Should be a JAX-compatible array
#                              representing SE(3) transforms (e.g., a jaxlie.SE3 object)
#                              with shape (num_robots,).
#         """
#         base_transforms_jnp = jnp.array(base_transforms)
#         base_transforms_jnp = jnp.atleast_2d(base_transforms_jnp)
#         assert base_transforms_jnp.shape[0] == self._num_robots, (
#             f"Expected first dimension of base_transforms to be {self._num_robots}, got {base_transforms_jnp.shape[0]}"
#         )
#
#         self._base_transforms = jaxlie.SE3(base_transforms_jnp)
#
#         # Re-apply transforms if a configuration exists
#         if self._last_cfg is not None:
#             self._apply_transforms(self._last_cfg)
#
# 停用配置更新：把配置规范为 [N_robot,N_act] 后保存，调用 FK 刷新 mesh。
#     def update_cfg(self, cfg: ArrayLike):
#         """
#         Update the poses of the batched robots based on their configurations.
#
#         Args:
#             cfg: Batched joint configurations. Shape should be (num_robots, num_dofs), or (num_dofs,).
#         """
#         cfg_jax = jnp.array(cfg)  # in case cfg is an onp.ndarray.
#         cfg_jax = jnp.atleast_2d(cfg_jax)
#         assert cfg_jax.shape[0] == self._num_robots, (
#             f"Expected first dimension of cfg to be {self._num_robots}, got {cfg_jax.shape[0]}"
#         )
#
#         # Store the latest configuration
#         self._last_cfg = cfg_jax
#         self._apply_transforms(cfg_jax)
#
# 停用 FK 显示：选每 link 位姿、乘基座变换，再更新单个或批量 mesh 的 xyz/wxyz。
#     def _apply_transforms(self, cfg_jax: jnp.ndarray):
#         """Helper method to apply FK and base transforms to update meshes."""
#         # Ts_link_world should have shape (num_robots, num_links, ...)
#         Ts_link_world = self._robot.forward_kinematics(cfg_jax)
#
#         for link_name, meshes in self._meshes.items():
#             link_idx = self._robot.links.names.index(link_name)
#             # T_link_world has shape (num_robots, ...)
#             T_link_world = jaxlie.SE3(
#                 Ts_link_world[:, link_idx]
#             )  # Select link transforms for all robots
#
#             # Apply base transform: T_mesh_world = T_base * T_link_world
#             # Resulting shape is (num_robots, ...)
#             T_mesh_world = self._base_transforms @ T_link_world
#
#             # Extract batched positions and orientations
#             position = onp.array(T_mesh_world.translation())  # Shape (num_robots, 3)
#             wxyz = onp.array(T_mesh_world.rotation().wxyz)  # Shape (num_robots, 4)
#             for mesh in meshes:
#                 if isinstance(mesh, viser.GlbHandle):
#                     mesh.position = position[0]
#                     mesh.wxyz = wxyz[0]
#                 else:
#                     assert isinstance(mesh, BatchedGlbHandle)
#                     mesh.batched_positions = position
#                     mesh.batched_wxyzs = wxyz
