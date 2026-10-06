# 学习导读：按几何类型派发 signed distance 函数并处理批广播，另提供成对矩阵和安全距离激活。
# 原距离正值表示分离、负值表示穿透；colldist_from_sdf 激活结果非正，代价常在外部取负或平方。
from __future__ import annotations

from typing import Callable, Dict, Tuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._geometry import Box, Capsule, CollGeom, HalfSpace, Heightmap, Sphere
from ._geometry_pairs import (
    capsule_box,
    capsule_capsule,
    halfspace_box,
    halfspace_capsule,
    halfspace_sphere,
    heightmap_capsule,
    heightmap_halfspace,
    heightmap_sphere,
    sphere_box,
    sphere_capsule,
    sphere_sphere,
)

# 这里只登记明确实现的类型组合；存在 Box 类不意味着支持 Box–Box 等任意组合。
COLLISION_FUNCTIONS: Dict[
    Tuple[type[CollGeom], type[CollGeom]], Callable[..., Float[Array, "*batch"]]
] = {
    (HalfSpace, Sphere): halfspace_sphere,
    (HalfSpace, Capsule): halfspace_capsule,
    (HalfSpace, Box): halfspace_box,
    (Sphere, Sphere): sphere_sphere,
    (Sphere, Capsule): sphere_capsule,
    (Sphere, Box): sphere_box,
    (Capsule, Capsule): capsule_capsule,
    (Capsule, Box): capsule_box,
    (Heightmap, Sphere): heightmap_sphere,
    (Heightmap, Capsule): heightmap_capsule,
    (Heightmap, HalfSpace): heightmap_halfspace,
}


# 按类型对查距离函数，找不到时交换参数尝试；仍缺失则显式抛 NotImplementedError。
def _get_coll_func(
    geom1_cls: type[CollGeom], geom2_cls: type[CollGeom]
) -> Callable[[CollGeom, CollGeom], Float[Array, "*batch"]]:
    """Get appropriate collision function (distance only) for given geometry types."""
    func = COLLISION_FUNCTIONS.get((geom1_cls, geom2_cls))
    if func is not None:
        return cast(Callable[[CollGeom, CollGeom], Float[Array, "*batch"]], func)

    # 距离对称时复用交换后的已登记实现，不要求显式登记两个顺序。
    func_swapped = COLLISION_FUNCTIONS.get((geom2_cls, geom1_cls))
    if func_swapped is not None:
        return cast(
            Callable[[CollGeom, CollGeom], Float[Array, "*batch"]],
            lambda g1, g2: func_swapped(g2, g1),
        )

    raise NotImplementedError(
        f"No collision function found for {geom1_cls.__name__} and {geom2_cls.__name__}"
    )


# 广播两几何对象的 batch 轴后调用对应距离函数，返回广播后的 signed distance 张量。
def collide(geom1: CollGeom, geom2: CollGeom) -> Float[Array, "*batch"]:
    """Calculate collision distance between two geometric objects, handling broadcasting.
    Positive distance means separation, negative means penetration.
    """
    try:
        broadcast_shape = jnp.broadcast_shapes(
            geom1.get_batch_axes(), geom2.get_batch_axes()
        )
    except ValueError as e:
        raise ValueError(
            f"Cannot broadcast geometry shapes {geom1.get_batch_axes()} and {geom2.get_batch_axes()}"
        ) from e

    geom1 = geom1.broadcast_to(broadcast_shape)
    geom2 = geom2.broadcast_to(broadcast_shape)
    geom1_cls = type(geom1)
    geom2_cls = type(geom2)
    return _get_coll_func(geom1_cls, geom2_cls)(geom1, geom2)


# 把最后集合轴 N/M 扩成 N×M，利用广播计算两组几何的全对距离矩阵。
def pairwise_collide(geom1: CollGeom, geom2: CollGeom) -> Float[Array, "*batch N M"]:
    """
    Convenience wrapper around `collide` for computing pairwise distances with broadcasting.

    Args:
        geom1: First collection of geometries. Expected to have a shape like
               (*batch1, N, ...), where N is the number of geometries.
        geom2: Second collection of geometries. Expected to have a shape like
               (*batch2, M, ...), where M is the number of geometries.
               The batch dimensions (*batch1, *batch2) must be broadcast-compatible.

    Returns:
        A matrix of distances with shape (*batch_combined, N, M), where
        *batch_combined is the result of broadcasting *batch1 and *batch2.
        dist[..., i, j] is the distance between geom1[..., i, :] and geom2[..., j, :].
    """
    # Input checks.
    axes1 = geom1.get_batch_axes()
    axes2 = geom2.get_batch_axes()
    assert len(axes1) >= 1, (
        f"geom1 must have at least one batch dimension to map over, got shape {axes1}"
    )
    assert len(axes2) >= 1, (
        f"geom2 must have at least one batch dimension to map over, got shape {axes2}"
    )

    # Determine expected output shape.
    batch1_shape = axes1[:-1]
    batch2_shape = axes2[:-1]
    N = axes1[-1]
    M = axes2[-1]
    try:
        batch_combined_shape = jnp.broadcast_shapes(batch1_shape, batch2_shape)
    except ValueError as e:
        raise ValueError(
            f"Cannot broadcast non-mapped batch shapes {batch1_shape} and {batch2_shape}"
        ) from e
    # 集合轴从各自最后 batch 轴取 N/M，其余样本前缀须可广播。
    expected_output_shape = (*batch_combined_shape, N, M)
    result = collide(
        geom1.broadcast_to((*batch_combined_shape, N))
        .reshape((*batch_combined_shape, N, 1))
        .broadcast_to(expected_output_shape),
        geom2.broadcast_to((*batch_combined_shape, M))
        .reshape((*batch_combined_shape, 1, M))
        .broadcast_to(expected_output_shape),
    )
    assert result.shape == expected_output_shape, (
        f"Output shape mismatch. Expected {expected_output_shape}, got {result.shape}"
    )
    return result


# 将安全距离内的 signed distance 分段平滑到≤0 激活，超过 activation_dist 时为零。
def colldist_from_sdf(
    _dist: jax.Array,
    activation_dist: jax.Array | float,
) -> jax.Array:
    """
    Convert a signed distance field to a collision distance field,
    based on https://arxiv.org/pdf/2310.17274#page=7.39.

    This function applies a smoothing transformation, useful for converting
    raw distances into values suitable for cost functions in optimization.
    It returns values <= 0, where 0 corresponds to distances >= activation_dist,
    and increasingly negative values for deeper penetrations.

    Args:
        _dist: Signed distance field values (positive = separation, negative = penetration).
        activation_dist: The distance threshold (margin) below which the cost activates.

    Returns:
        Transformed collision distance field values (<= 0).
    """
    _dist = jnp.minimum(_dist, activation_dist)
    # 穿透区域采用线性分支，安全距离内非穿透采用二次分支，最终截到非正值。
    _dist = jnp.where(
        _dist < 0,
        _dist - 0.5 * activation_dist,
        -0.5 / (activation_dist + 1e-6) * (_dist - activation_dist) ** 2,
    )
    _dist = jnp.minimum(_dist, 0.0)
    return _dist
