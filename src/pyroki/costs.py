"""Cost and constraint factories for pyroki optimization.

Costs are soft penalties (minimized via least squares).
Constraints use augmented Lagrangian method (iteratively tightened penalties).

All factories are created by wrapping residual functions with Cost.factory.
"""
# 学习导读：将纯残差包装为 jaxls.Cost 工厂；相同残差可作为软代价或增强拉格朗日约束。
# 约束 kind 区分残差≤0/其他关系，具体收敛条件由 jaxls 求解器负责。

from jaxls import Cost

from ._residuals import (
    five_point_acceleration_residual,
    five_point_jerk_residual,
    five_point_velocity_residual,
    limit_residual,
    limit_velocity_residual,
    manipulability_residual,
    pose_residual,
    pose_with_base_residual,
    rest_residual,
    rest_with_base_residual,
    self_collision_residual,
    smoothness_residual,
    world_collision_residual,
)
from ._residuals._pose_residual_analytic_jac import (
    pose_cost_analytic_jac as pose_cost_analytic_jac,
)
from ._residuals._pose_residual_numerical_jac import (
    pose_cost_numerical_jac as pose_cost_numerical_jac,
)

# Pose costs
# 纯残差默认由求解器自动求导；解析/数值自定义 Jacobian 在上面的专用工厂提供。
pose_cost = Cost.factory(pose_residual)
pose_cost_with_base = Cost.factory(pose_with_base_residual)

# Limit costs
limit_cost = Cost.factory(limit_residual)
limit_velocity_cost = Cost.factory(limit_velocity_residual)

# Regularization costs
rest_cost = Cost.factory(rest_residual)
rest_with_base_cost = Cost.factory(rest_with_base_residual)
smoothness_cost = Cost.factory(smoothness_residual)

# Manipulability cost
manipulability_cost = Cost.factory(manipulability_residual)

# Collision costs
self_collision_cost = Cost.factory(self_collision_residual)
world_collision_cost = Cost.factory(world_collision_residual)

# Finite difference costs
five_point_velocity_cost = Cost.factory(five_point_velocity_residual)
five_point_acceleration_cost = Cost.factory(five_point_acceleration_residual)
five_point_jerk_cost = Cost.factory(five_point_jerk_residual)

# Constraint factories (augmented Lagrangian penalties)
# These use kind="constraint_leq_zero" which enforces residual <= 0
# 不等式要求残差≤0；由于限位残差非负，只有无越界时的零值满足约束。
limit_constraint = Cost.factory(kind="constraint_leq_zero")(limit_residual)
limit_velocity_constraint = Cost.factory(kind="constraint_leq_zero")(
    limit_velocity_residual
)
# 世界碰撞 residual 先取负激活，因此≤0 约束与足够分離/无安全距离违例对应。
world_collision_constraint = Cost.factory(kind="constraint_leq_zero")(
    world_collision_residual
)
