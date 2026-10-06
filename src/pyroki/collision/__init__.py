"""Collision detection primitives and utilities."""
# 学习导读：公开碰撞几何、RobotCollision、距离派发及激活函数，供 costs 和示例调用。
# 几何近似与视觉网格不同，距离含义/支持类型组合应以具体实现为准。

from ._collision import colldist_from_sdf as colldist_from_sdf
from ._collision import collide as collide
from ._geometry import Box as Box
from ._geometry import Capsule as Capsule
from ._geometry import CollGeom as CollGeom
from ._geometry import HalfSpace as HalfSpace
from ._geometry import Heightmap as Heightmap
from ._geometry import Sphere as Sphere
from ._robot_collision import RobotCollision as RobotCollision
