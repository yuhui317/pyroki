# 学习导读：公开 Robot、costs、collision 与 viewer 四个入口；数值模型、优化残差和可视化分别维护。
from . import costs as costs
from . import collision as collision
from . import viewer as viewer
from ._robot import Robot as Robot

__version__ = "0.0.0"
