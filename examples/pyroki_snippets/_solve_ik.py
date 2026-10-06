"""
Solves the basic IK problem.
"""
# 学习导读：最小 IK 封装把 NumPy 目标转 JAX，并以解析位姿雅可比和关节限位约束求解单个关节变量。
# 它是 examples/01_basic_ik.py 的数值入口；首次调用可能包含 JIT 编译开销。

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


# 校验 xyz/wxyz 形状、名称转连杆索引并调用 JIT 求解器，最后返回 NumPy 关节配置。
def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return onp.array(cfg)


# 声明 JointVar(0)，构造解析位姿代价和关节≤0 约束，经 analyze 与 dense_cholesky 求解。
@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    # 这里只有一个优化状态，ID=0；Robot 创建的变量默认工厂提供关节初值。
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]
    # 明确列出目标/偏好/约束因子；不同示例使用的软硬条件并不相同，应逐项核对。
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
    ]
    # analyze 分析变量/残差结构，solve 执行迭代；具体 LM/增强拉格朗日实现属于外部 jaxls。
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            verbose=False,
            # 选择稠密 Cholesky 线性求解；这不是独立闭式 IK 公式。
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]
