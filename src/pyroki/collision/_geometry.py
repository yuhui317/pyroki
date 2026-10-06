# 学习导读：用 PyTree 定义带 SE(3) pose 和 size 参数的半空间/球/胶囊/盒/高度图。
# 末轴存形状参数或 xyz/wxyz，前面的 batch 轴可表示样本或几何集合；to_trimesh 是宿主侧显示转换。
from __future__ import annotations

import abc
from typing import cast

import jax
import jax.numpy as jnp
import jax.scipy.ndimage
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import trimesh
from jax.typing import ArrayLike
from jaxtyping import Array, Float
from typing_extensions import Self

from ._utils import make_frame


# 所有几何的抽象基类：pose 是 SE(3)，size 末轴是类型专用形状参数。
@jdc.pytree_dataclass
class CollGeom(abc.ABC):
    """Base class for geometric objects."""

    pose: jaxlie.SE3
    """Geometry pose (position and orientation)."""

    size: Float[Array, "*batch shape_dim"]
    """Geometry shape parameters, e.g. radius, half-length."""

    # 按 pose 的批轴长度提取各数值叶节点前缀，并计算兼容广播形状。
    def get_batch_axes(self) -> tuple[int, ...]:
        """Get batch axes of the geometry."""
        batch_axes_from_pose = self.pose.get_batch_axes()
        return jnp.broadcast_shapes(
            *[
                getattr(leaf, "shape", ())[: len(batch_axes_from_pose)]
                for leaf in jax.tree.leaves(self)
            ]
        )

    # 只广播 batch 前缀，保留每个叶节点的 pose/size/height_data 特征末轴。
    def broadcast_to(self, shape: tuple[int, ...]) -> Self:
        """Broadcast geometry with given batch axes."""
        return jax.tree.map(
            lambda x: jnp.broadcast_to(
                x, shape + getattr(x, "shape", ())[len(self.pose.get_batch_axes()) :]
            ),
            self,
        )

    # 改变 batch 轴的布局，保留几何特征末轴；总数须能重排一致。
    def reshape(self, shape: tuple[int, ...]) -> Self:
        """Reshape geometry to given shape."""
        return jax.tree.map(
            lambda x: x.reshape(
                shape + getattr(x, "shape", ())[len(self.pose.get_batch_axes()) :]
            ),
            self,
        )

    # 把传入变换左乘当前 pose，广播数值数据后用 copy_and_mutate 返回新几何。
    def transform(self, transform: jaxlie.SE3) -> Self:
        """Left-multiples geometry's pose with an SE(3) transformation."""
        broadcasted_axes = jnp.broadcast_shapes(
            self.pose.get_batch_axes(), transform.get_batch_axes()
        )
        _self = self.broadcast_to(broadcasted_axes)
        with jdc.copy_and_mutate(_self) as out:
            # 左乘意味着把局部几何 pose 变到传入变换所定义的外层坐标系。
            out.pose = transform @ _self.pose
        return out

    # 从四元数 wxyz 与 xyz 构造 SE(3) 再左乘几何 pose。
    def transform_from_wxyz_position(
        self,
        wxyz: Float[ArrayLike, "*batch 4"],
        position: Float[ArrayLike, "*batch 3"],
    ) -> Self:
        """Left-multiples geometry's pose with an SE(3) transformation.

        Equivalent to `self.transform`, but doesn't require direct JAX instantiation of SE3.
        """
        position, wxyz = jnp.asarray(position), jnp.asarray(wxyz)
        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), position)
        return self.transform(pose)

    # 抽象方法：由具体类型把某个批索引转为单个显示网格。
    @abc.abstractmethod
    def _create_one_mesh(self, index: tuple[int, ...]) -> trimesh.Trimesh:
        """Helper to create a single trimesh object from batch data at a given index."""
        raise NotImplementedError

    # 宿主侧枚举全部批索引并合并网格；这不是 JIT 内的可微几何运算。
    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the (potentially batched) geometry to a single trimesh object."""
        batch_axes = self.get_batch_axes()
        if not batch_axes:
            return self._create_one_mesh(tuple())

        meshes = [
            self._create_one_mesh(idx_tuple) for idx_tuple in onp.ndindex(batch_axes)
        ]
        if not meshes:
            return trimesh.Trimesh()

        return cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))


# 以局部 z 轴为外法线定义半空间，pose.translation 是其边界平面上一点。
@jdc.pytree_dataclass
class HalfSpace(CollGeom):
    """HalfSpace geometry defined by a point and an outward normal."""

    # 取旋转矩阵第 3 列，即变换后的局部 z 单位方向。
    @property
    def normal(self) -> Float[Array, "*batch 3"]:
        """Normal vector (Z-axis of rotation matrix)."""
        # 矩阵列表示局部基轴在当前坐标系的方向，第 2 索引列为 z。
        return self.pose.rotation().as_matrix()[..., :, 2]

    # 计算法线与平面点的内积，得到平面相对原点的有向偏移。
    @property
    def offset(self) -> Float[Array, "*batch"]:
        """Offset from origin along the normal (origin = point on plane)."""
        return jnp.einsum("...i,...i->...", self.normal, self.pose.translation())

    # 广播点/法线并以 make_frame 把局部 z 轴对齐法线，构造边界坐标系。
    @staticmethod
    def from_point_and_normal(
        point: Float[ArrayLike, "*batch 3"], normal: Float[ArrayLike, "*batch 3"]
    ) -> HalfSpace:
        """Create a HalfSpace geometry from a point on the boundary and outward normal."""
        point, normal = jnp.array(point), jnp.array(normal)
        batch_axes = jnp.broadcast_shapes(point.shape[:-1], normal.shape[:-1])
        point = jnp.broadcast_to(point, batch_axes + (3,))
        normal = jnp.broadcast_to(normal, batch_axes + (3,))
        mat = make_frame(normal)
        pos = point
        pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(mat), pos
        )
        size = jnp.zeros(batch_axes + (1,), dtype=pos.dtype)
        return HalfSpace(pose=pose, size=size)

    # 用 10×10×0.01 薄盒显示无限平面的局部示意，不改变距离算法中的无限半空间。
    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        """Visualize HalfSpace as a large thin box aligned with its boundary plane."""
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        mat = onp.array(pose_i.rotation().as_matrix())
        # Visualize as a box representing the boundary plane
        plane_mesh = trimesh.creation.box(extents=[10, 10, 0.01])
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        plane_mesh.apply_transform(tf)
        return plane_mesh


# 球几何，pose 平移为球心，size[...,0] 保存半径。
@jdc.pytree_dataclass
class Sphere(CollGeom):
    """Sphere geometry."""

    # 读取 size 的半径分量，移除形状参数末轴。
    @property
    def radius(self) -> Float[Array, "*batch"]:
        """Radius of the sphere."""
        return self.size[..., 0]

    # 广播球心/半径并拼接单位 wxyz，建立批化 Sphere。
    @staticmethod
    def from_center_and_radius(
        center: Float[ArrayLike, "*batch 3"], radius: Float[ArrayLike, "*batch"]
    ) -> Sphere:
        """Create a Sphere geometry from a center point and radius."""
        center, radius = jnp.array(center), jnp.array(radius)
        batch_axes = jnp.broadcast_shapes(center.shape[:-1], radius.shape)
        center = jnp.broadcast_to(center, batch_axes + (3,))
        radius = jnp.broadcast_to(radius, batch_axes)
        pos = center
        # Create identity pose for sphere
        num_batch_elements = onp.prod(batch_axes).item() if batch_axes else 1
        quat_wxyz = jnp.stack(
            [jnp.array([1.0, 0.0, 0.0, 0.0], dtype=pos.dtype)] * num_batch_elements,
            axis=0,
        )
        quat_wxyz = quat_wxyz.reshape(batch_axes + (4,))
        wxyz_xyz = jnp.concatenate([quat_wxyz, pos], axis=-1)
        pose = jaxlie.SE3(wxyz_xyz)

        # Store radius in size[..., 0], shape_dim=1
        size = radius[..., None]
        return Sphere(pose=pose, size=size)

    # 生成低细分 icosphere 后平移到球心；球体方向不影响距离或外形。
    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        radius_val = float(self.radius[index])
        sphere_mesh = trimesh.creation.icosphere(radius=radius_val, subdivisions=1)
        # Only apply translation for sphere
        tf = onp.eye(4)
        tf[:3, 3] = pos
        sphere_mesh.apply_transform(tf)
        return sphere_mesh


# 胶囊几何：size 保存半径和中间圆柱段长度，局部 z 为轴向。
@jdc.pytree_dataclass
class Capsule(CollGeom):
    """Capsule geometry."""

    # 读取胶囊圆柱与端球共同半径。
    @property
    def radius(self) -> Float[Array, "*batch"]:
        """Radius of the capsule ends and cylinder."""
        return self.size[..., 0]

    # 读取中间圆柱段长度，不包含两端半球的额外 2r 总长。
    @property
    def height(self) -> Float[Array, "*batch"]:
        """Height of the cylindrical segment."""
        return self.size[..., 1]

    # 取 pose 旋转后的局部 z 轴，用于计算轴线端点。
    @property
    def axis(self) -> Float[Array, "*batch 3"]:
        """Axis direction (Z-axis of rotation matrix)."""
        # 矩阵列表示局部基轴在当前坐标系的方向，第 2 索引列为 z。
        return self.pose.rotation().as_matrix()[..., :, 2]

    # 广播半径、圆柱长度和可选 pose；未指定 pose 时为原点单位朝向。
    @staticmethod
    def from_radius_height(
        radius: Float[ArrayLike, "*batch"],
        height: Float[ArrayLike, "*batch"],  # Full height
        position: Float[ArrayLike, "*batch 3"] | None = None,
        wxyz: Float[ArrayLike, "*batch 4"] | None = None,
    ) -> Capsule:
        """Create Capsule geometry from radius and height."""
        if position is None:
            position = jnp.zeros((3,))
        if wxyz is None:
            wxyz = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity matrix.

        position = jnp.array(position)
        wxyz = jnp.array(wxyz)
        radius = jnp.array(radius)
        height = jnp.array(height)

        batch_axes = jnp.broadcast_shapes(
            position.shape[:-1], wxyz.shape[:-1], radius.shape, height.shape
        )
        pos = jnp.broadcast_to(position, batch_axes + (3,))
        wxyz = jnp.broadcast_to(wxyz, batch_axes + (4,))
        radius = jnp.broadcast_to(radius, batch_axes)
        height = jnp.broadcast_to(height, batch_axes)

        wxyz_xyz = jnp.concatenate([wxyz, pos], axis=-1)
        pose = jaxlie.SE3(wxyz_xyz)

        # height 在距离公式中以±height/2 定义轴线端点，外形总长还加两端半径。
        size = jnp.stack([radius, height], axis=-1)
        return Capsule(pose=pose, size=size)

    # 以 mesh 的最小包围圆柱构造胶囊近似；空 mesh 返回零尺寸几何。
    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh) -> Capsule:
        """
        Create Capsule geometry from minimum bounding cylinder of the mesh.
        """
        if mesh.is_empty:
            return Capsule(pose=jaxlie.SE3.identity(), size=jnp.zeros((2,)))
        results = trimesh.bounds.minimum_cylinder(mesh)
        radius = results["radius"]
        height = results["height"]
        tf_mat = results["transform"]
        tf = jaxlie.SE3.from_matrix(tf_mat)
        capsule = Capsule.from_radius_height(
            position=jnp.zeros((3,)),
            wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]),
            radius=radius,
            height=height,
        )
        capsule = capsule.transform(tf)
        return capsule

    # 把 icosphere 两半沿 z 分离以形成胶囊显示，再应用 pose。
    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        mat = onp.array(pose_i.rotation().as_matrix())
        radius_val = float(self.radius[index])
        height_val = abs(float(self.height[index])) / 2

        # Create sphere and stretch it to match capsule shape.
        capsule_mesh = trimesh.creation.icosphere(radius=radius_val, subdivisions=1)
        capsule_mesh.vertices = onp.where(
            capsule_mesh.vertices[:, 2][..., None] > 0,
            capsule_mesh.vertices + onp.array([0.0, 0.0, height_val]),
            capsule_mesh.vertices - onp.array([0.0, 0.0, height_val]),
        )

        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        capsule_mesh.apply_transform(tf)
        return capsule_mesh

    # 沿胶囊轴离散成 n_segments 个同半径球，新增采样轴在所有原 batch 轴之前。
    def decompose_to_spheres(self, n_segments: int) -> Sphere:
        """
        Decompose the capsule into a series of spheres along its axis.
        Args: n_segments: Number of spheres.
        Returns: Sphere object shape (n_segments, *batch, ...).
        """
        batch_axes = self.get_batch_axes()
        radii = self.radius

        # Calculate local offsets for sphere centers along z-axis.
        segment_factors = jnp.linspace(-1.0, 1.0, n_segments)
        local_offsets_vec = jnp.array([0.0, 0.0, 1.0])[None, None, :] * (
            segment_factors[:, None, None] * self.height[None, ..., None] / 2
        )

        # Create base spheres (at origin, correct radius) and transform them.
        spheres = Sphere.from_center_and_radius(
            center=jnp.zeros((n_segments,) + batch_axes + (3,)),
            radius=jnp.broadcast_to(radii, (n_segments,) + batch_axes),
        )

        # Broadcast capsule pose and apply transforms.
        capsule_pose_broadcast = jaxlie.SE3(
            jnp.broadcast_to(
                self.pose.wxyz_xyz,
                (n_segments,) + self.pose.get_batch_axes() + (7,),
            )
        )
        spheres = spheres.transform(
            capsule_pose_broadcast @ jaxlie.SE3.from_translation(local_offsets_vec)
        )
        assert spheres.get_batch_axes() == (n_segments,) + batch_axes
        return spheres

    # 用两球中心连线构造胶囊，半径取首球；仅校验批形状，没有校验两球半径相同。
    @staticmethod
    def from_sphere_pairs(sph_0: Sphere, sph_1: Sphere) -> Capsule:
        """
        Create a capsule connecting the centers of two spheres.
        Args: sph_0, sph_1: Input spheres.
        Returns: Capsule object with the same batch shape.
        """
        assert sph_0.get_batch_axes() == sph_1.get_batch_axes(), "Batch axes mismatch"

        pos0 = sph_0.pose.translation()
        pos1 = sph_1.pose.translation()
        vec = pos1 - pos0

        # Get height, safely handle zero-length case.
        x = pos1 - pos0
        # allclose 对整个输入归约为一个 bool，退化判定不是逐样本独立的。
        is_zero = jnp.allclose(x, 0.0)
        x = jnp.where(is_zero, jnp.ones_like(x), x)
        n = jnp.linalg.norm(x + 1e-6, axis=-1, keepdims=True)
        height = jax.lax.select(is_zero, jnp.zeros_like(n), n).squeeze(-1)

        transform = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.from_matrix(make_frame(vec)),
            translation=(pos0 + pos1) / 2.0,
        )

        capsule = Capsule.from_radius_height(
            position=transform.translation(),
            wxyz=transform.rotation().wxyz,
            # 连接胶囊直接使用首球半径，前后球应满足调用方期望的同半径约定。
            radius=sph_0.radius,
            height=height,
        )

        assert capsule.get_batch_axes() == sph_0.get_batch_axes()
        return capsule


# 定向盒几何，size 存局部 xyz 三个半边长。
@jdc.pytree_dataclass
class Box(CollGeom):
    """Box geometry defined by half-extents."""

    # 读取局部半边长向量 [*batch,3]。
    @property
    def half_extent(self) -> Float[Array, "*batch 3"]:
        """Half-extents (half width, half height, half depth) in local frame."""
        return self.size

    # 返回完整边长，即 size 的两倍。
    @property
    def extent(self) -> Float[Array, "*batch 3"]:
        """Full extents (width, height, depth) in local frame."""
        return self.size * 2

    # 接收完整边长与可选 pose，广播后除 2 保存半边长。
    @staticmethod
    def from_extent(
        extent: Float[ArrayLike, "*batch 3"],
        position: Float[ArrayLike, "*batch 3"] | None = None,
        wxyz: Float[ArrayLike, "*batch 4"] | None = None,
    ) -> "Box":
        """Create Box from full extents (width, height, depth)."""
        if position is None:
            position = jnp.zeros((3,))
        if wxyz is None:
            wxyz = jnp.array([1.0, 0.0, 0.0, 0.0])

        extent = jnp.array(extent)
        position = jnp.array(position)
        wxyz = jnp.array(wxyz)

        batch_axes = jnp.broadcast_shapes(
            extent.shape[:-1], position.shape[:-1], wxyz.shape[:-1]
        )
        extent = jnp.broadcast_to(extent, batch_axes + (3,))
        position = jnp.broadcast_to(position, batch_axes + (3,))
        wxyz = jnp.broadcast_to(wxyz, batch_axes + (4,))

        wxyz_xyz = jnp.concatenate([wxyz, position], axis=-1)
        pose = jaxlie.SE3(wxyz_xyz)

        # Store half-extents in size
        size = extent / 2.0
        return Box(pose=pose, size=size)

    # 列出 xyz 各轴±半边长的八种组合，返回 [*batch,8,3]。
    def get_corners_local(self) -> Float[Array, "*batch 8 3"]:
        """Get the 8 corners of the box in local frame."""
        half = self.half_extent

        # 8 corners from all combinations of +/- half extents
        signs = jnp.array(
            [
                [-1, -1, -1],
                [-1, -1, +1],
                [-1, +1, -1],
                [-1, +1, +1],
                [+1, -1, -1],
                [+1, -1, +1],
                [+1, +1, -1],
                [+1, +1, +1],
            ]
        )  # (8, 3)

        # Broadcast: half is (*batch, 3), signs is (8, 3)
        # Result: (*batch, 8, 3)
        corners = half[..., None, :] * signs
        return corners

    # 给八个局部角点应用 pose，转换到碰撞计算的坐标系。
    def get_corners_world(self) -> Float[Array, "*batch 8 3"]:
        """Get the 8 corners of the box in world frame."""
        corners_local = self.get_corners_local()  # (*batch, 8, 3)
        # Apply pose to each corner
        return self.pose.apply(corners_local)

    # 用完整边长建 trimesh 盒体，再施加 pose 用于显示。
    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        mat = onp.array(pose_i.rotation().as_matrix())
        extent_val = onp.array(self.extent[index])

        box_mesh = trimesh.creation.box(extents=extent_val)
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        box_mesh.apply_transform(tf)
        return box_mesh


# 高度图几何：height_data 末两轴 H/W，局部 x/y 网格以中心为原点，z 按高度缩放。
@jdc.pytree_dataclass
class Heightmap(CollGeom):
    """Heightmap geometry defined by a grid of height values.
    The heightmap is oriented such that its base lies on the XY plane of its local frame.
    """

    height_data: Float[Array, "*batch H W"]

    # 返回局部 x 网格间距。
    @property
    def x_scale(self) -> Float[Array, "*batch"]:
        """Grid spacing along the local X-axis."""
        return self.size[..., 0]

    # 返回局部 y 网格间距。
    @property
    def y_scale(self) -> Float[Array, "*batch"]:
        """Grid spacing along the local Y-axis."""
        return self.size[..., 1]

    # 返回原始高度数值的乘法系数。
    @property
    def height_scale(self) -> Float[Array, "*batch"]:
        """Multiplier applied to height data values."""
        return self.size[..., 2]

    # height_data 倒数第二轴是沿局部 y 的网格行数。
    @property
    def rows(self) -> int:
        """Number of rows in the height grid (along local Y)."""
        return self.height_data.shape[-2]

    # height_data 最后一轴是沿局部 x 的网格列数。
    @property
    def cols(self) -> int:
        """Number of columns in the height grid (along local X)."""
        return self.height_data.shape[-1]

    # 从高度数据/xy 间距建立单位 pose；当前 pose 无批轴，后面的断言限制批量构造。
    @staticmethod
    def from_height_data(
        height_data: Float[ArrayLike, "*batch H W"],
        x_scale: Float[ArrayLike, "*batch"],
        y_scale: Float[ArrayLike, "*batch"],
    ) -> Heightmap:
        """Create Heightmap geometry from height data and grid spacing."""
        height_data = jnp.array(height_data)
        x_scale = jnp.array(x_scale)
        y_scale = jnp.array(y_scale)
        pose = jaxlie.SE3.identity()
        size = jnp.stack([x_scale, y_scale, jnp.ones_like(x_scale)], axis=-1)
        assert height_data.shape[:-2] == size.shape[:-1]
        # from_height_data 的 pose 固定 identity() 无批轴，所以当前构造器仅支持这个断言允许的无批形式。
        assert pose.get_batch_axes() == size.shape[:-1]
        return Heightmap(pose=pose, size=size, height_data=height_data)

    # 在网格 xy 范围上向下射线采样首个命中 z，生成 2.5D 高度图；射线部分在 NumPy/trimesh 执行。
    @staticmethod
    def from_trimesh(
        mesh: trimesh.Trimesh,
        resolution: float = 0.01,
        x_bins: int | None = None,
        y_bins: int | None = None,
    ) -> Heightmap:
        """Create Heightmap geometry from a trimesh mesh using batched raycasting.
        The constructor optionally takes in `x_bins` and `y_bins` to set the grid shape,
        which is useful for JIT compilation.
        """
        if mesh.is_empty:
            return Heightmap(
                pose=jaxlie.SE3.identity(),
                size=jnp.array([resolution, resolution, 1.0]),
                height_data=jnp.zeros((0, 0)),  # Empty height data.
            )

        x_min, y_min, z_min = mesh.bounds[0]
        x_max, y_max, z_max = mesh.bounds[1]

        # Pad bounds slightly to avoid missing hits exactly at the boundary.
        z_max_padded = z_max + 0.01  # Start rays from above the highest point.
        z_min_padded = z_min - 0.01  # Use as floor if no hit.

        center = jnp.array([(x_min + x_max) / 2, (y_min + y_max) / 2, 0.0])

        # Calculate grid resolution, ensure at least 1x1 grid
        x_res = max(1, int(jnp.ceil((x_max - x_min) / resolution)))
        y_res = max(1, int(jnp.ceil((y_max - y_min) / resolution)))

        assert (x_bins is not None) == (y_bins is not None), (
            "x_bins and y_bins must both be provided or not provided"
        )
        if x_bins is not None:
            assert x_bins is not None and y_bins is not None
            x_res = x_bins
            y_res = y_bins
            resolution_x = (x_max - x_min) / x_bins
            resolution_y = (y_max - y_min) / y_bins
        else:
            resolution_x = resolution
            resolution_y = resolution

        # Create grid coordinates (pixel centers)
        x_coords = x_min + (jnp.arange(x_res) + 0.5) * resolution_x
        y_coords = y_min + (jnp.arange(y_res) + 0.5) * resolution_y

        # Create meshgrid of sampling points (shape: y_res, x_res)
        # 'xy' indexing ensures xx and yy have shapes matching (y_res, x_res)
        xx, yy = jnp.meshgrid(x_coords, y_coords, indexing="xy")

        # Prepare ray origins (casting down from above the mesh)
        # Shape: (y_res * x_res, 3)
        ray_origins = jnp.stack(
            [xx.ravel(), yy.ravel(), jnp.full(xx.shape, z_max_padded).ravel()], axis=-1
        )

        # Prepare ray directions (all pointing down)
        # Shape: (y_res * x_res, 3)
        ray_directions = jnp.tile(
            jnp.array([[0.0, 0.0, -1.0]]), (ray_origins.shape[0], 1)
        )

        # Perform batched raycasting using trimesh (requires numpy input)
        # We ask for the *first* hit since we are casting downwards.
        # trimesh 射线在宿主处理 NumPy 数组；高度图构建不是端到端可微 mesh 光线过程。
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=onp.array(ray_origins),
            ray_directions=onp.array(ray_directions),
            multiple_hits=False,  # Get only the first hit
        )

        # Initialize heightmap with a value below the mesh (z_min_padded)
        heightmap_flat = jnp.full(y_res * x_res, z_min_padded)

        # Update heights where rays hit the mesh
        # index_ray contains the indices (0 to y_res*x_res-1) of the rays that hit
        # locations[:, 2] contains the corresponding z-coordinates of the hits
        if index_ray.size > 0:
            heightmap_flat = heightmap_flat.at[index_ray].set(
                jnp.array(locations[:, 2])
            )

        # Reshape flat heightmap back to 2D grid (y_res, x_res)
        height_data = heightmap_flat.reshape((y_res, x_res))

        # Create the Heightmap object
        heightmap_obj = Heightmap.from_height_data(
            height_data,
            x_scale=jnp.array(resolution_x),
            y_scale=jnp.array(resolution_y),
        )

        # Apply transform to place the heightmap center at the mesh's xy center
        heightmap_obj = heightmap_obj.transform(jaxlie.SE3.from_translation(center))
        return heightmap_obj

    # 保留点在高度图局部 xy，替换局部 z 为双线性插值高度，再回到世界坐标。
    def project_points(
        self, points: Float[Array, "*batch 3"]
    ) -> Float[Array, "*batch 3"]:
        """Projects points onto the heightmap surface, in the world frame."""
        local_coords = self.pose.inverse().apply(points)
        height = self._interpolate_height_at_coords(points)
        intersect_points = local_coords.at[..., 2].set(height)
        return self.pose.apply(intersect_points)

    # 世界点转局部网格坐标，以 order=1 与 nearest 边界模式插值，批维展平后 vmap。
    def _interpolate_height_at_coords(
        self,
        world_coords: Float[Array, "*batch 3"],
    ) -> Float[Array, "*batch"]:
        """Interpolates heightmap surface height at given world coordinates.

        Args:
            world_coords: Coordinates in the world frame (*batch, 3).

        Returns:
            Interpolated heightmap surface height in the heightmap's local frame (*batch).
        """
        # Transform world coords to heightmap local frame.
        local_coords = self.pose.inverse().apply(world_coords)
        sx, sy = local_coords[..., 0], local_coords[..., 1]

        # Calculate continuous grid indices (r, c) from local coords (sx, sy)
        # Origin (0,0) in local frame corresponds to center of the base grid!
        # 局部原点对应网格中心，x 对列、y 对行，分别除以网格间距得到连续索引。
        c_cont = sx / self.x_scale + (self.cols - 1) / 2.0
        r_cont = sy / self.y_scale + (self.rows - 1) / 2.0

        # Interpolate height data at (r_cont, c_cont).
        batch_axes = self.get_batch_axes()
        # Ensure batch axes of coords match heightmap's batch axes.
        target_batch_shape = jnp.broadcast_shapes(batch_axes, world_coords.shape[:-1])
        coords_bc = jnp.broadcast_to(
            jnp.stack([r_cont, c_cont], axis=-1), target_batch_shape + (2,)
        )
        hm_data_bc = jnp.broadcast_to(
            self.height_data, target_batch_shape + self.height_data.shape[-2:]
        )

        if target_batch_shape:
            batch_size = onp.prod(target_batch_shape).item()
            # Reshape for vmap.
            h_data_flat = hm_data_bc.reshape((batch_size, self.rows, self.cols))
            coords_flat = coords_bc.reshape((batch_size, 2))

            # vmap over flattened batch dimension.
            vmap_interpolate = jax.vmap(
                lambda h, c: jax.scipy.ndimage.map_coordinates(
                    h, c[:, None], order=1, mode="nearest"
                ).squeeze(),
                in_axes=(0, 0),
            )
            interpolated_heights_flat = vmap_interpolate(h_data_flat, coords_flat)
            interpolated_heights = interpolated_heights_flat.reshape(target_batch_shape)
        else:
            # Non-batched case.
            interpolated_heights = jax.scipy.ndimage.map_coordinates(
                hm_data_bc,
                (coords_bc[0:1], coords_bc[1:2]),  # ([r_cont], [c_cont])
                order=1,
                # 超出高度图网格的插值采用最近边界值，不会自动生成越界碰撞报警。
                mode="nearest",
            ).squeeze()

        # Scale interpolated height
        interpolated_local_z = interpolated_heights * self.height_scale
        return interpolated_local_z

    # 按网格间距生成中心化局部顶点，返回 [*batch,H×W,3]；批量 meshgrid 分支需独立运行核对。
    def _get_vertices_local(self) -> Float[Array, "*batch H*W 3"]:
        """Computes the heightmap vertices in its local frame using JAX.

        Returns:
            Vertices array with shape (*batch, H*W, 3).
        """
        batch_axes = self.get_batch_axes()
        H, W = self.rows, self.cols

        # Create grid coordinates (centered).
        x = (jnp.arange(W) - (W - 1) / 2.0) * self.x_scale[..., None]
        y = (jnp.arange(H) - (H - 1) / 2.0) * self.y_scale[..., None]

        # Add batch dimensions for meshgrid if necessary.
        if batch_axes:
            x = jnp.broadcast_to(x, batch_axes + (W,))
            y = jnp.broadcast_to(y, batch_axes + (H,))
            # 无批 x/y 是一维网格轴；有批时 x/y 已为多维，需运行核对 meshgrid 实际支持的输入。
            xx, yy = jnp.meshgrid(x, y, indexing="xy")  # Results shape (*batch, H, W).
        else:
            # 无批 x/y 是一维网格轴；有批时 x/y 已为多维，需运行核对 meshgrid 实际支持的输入。
            xx, yy = jnp.meshgrid(x, y, indexing="xy")  # Results shape (H, W).

        # Scale height data.
        zz = self.height_data * self.height_scale[..., None, None]

        # Combine into vertices: (*batch, H, W, 3).
        vertices = jnp.stack([xx, yy, zz], axis=-1)

        # Reshape to (*batch, H*W, 3).
        vertices_flat = vertices.reshape(batch_axes + (H * W, 3))
        return vertices_flat

    # 宿主侧生成双面三角网格；行/列不足 2 时返回空网格。
    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        """Create a single trimesh object from height data at a given index.
        Also includes back-facing triangles for two-sided rendering.
        """
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        height_data_i: Float[Array, "H W"] = self.height_data[index]
        x_scale_i = float(self.x_scale[index])
        y_scale_i = float(self.y_scale[index])
        height_scale_i = float(self.height_scale[index])

        rows, cols = height_data_i.shape
        if rows < 2 or cols < 2:
            # Need at least a 2x2 grid to form a face.
            return trimesh.Trimesh()

        # Create vertex grid.
        x = onp.arange(cols) * x_scale_i
        y = onp.arange(rows) * y_scale_i
        xx, yy = onp.meshgrid(x, y)
        zz = onp.array(height_data_i) * height_scale_i

        vertices = onp.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        # Center the vertices around the origin before applying pose.
        center_offset = onp.array(
            [(cols - 1) * x_scale_i / 2.0, (rows - 1) * y_scale_i / 2.0, 0.0]
        )
        vertices -= center_offset

        # Create faces (triangles) - both front and back.
        front_faces = []
        back_faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                idx0 = r * cols + c
                idx1 = r * cols + (c + 1)
                idx2 = (r + 1) * cols + c
                idx3 = (r + 1) * cols + (c + 1)
                front_faces.append([idx0, idx1, idx2])  # Triangle 1 (front).
                front_faces.append([idx1, idx3, idx2])  # Triangle 2 (front).
                back_faces.append([idx0, idx2, idx1])  # Triangle 1 (back).
                back_faces.append([idx1, idx2, idx3])  # Triangle 2 (back).

        all_faces = front_faces + back_faces

        if not all_faces:
            return trimesh.Trimesh()

        heightmap_mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces)

        tf = onp.array(pose_i.as_matrix())
        heightmap_mesh.apply_transform(tf)

        return heightmap_mesh
