# pyroki · 完整代码树

[初步分析](analysis.md) · [源码定位](code_symbols.md) · [范围](annotation_scope.json) · [验证](annotation_validation.json)

固定基线 `388e43e1fc0d0ee382968d3dd72970fd62a0450c`，来源 `main`，学习分支 `codex/study/pyroki-code-analysis`。

86 个基线文件、0 个 gitlink；本轮注释 48 个原始文件。树不包含 `.git` 或缓存，固定基线自带的资源仍完整列出。

```text
pyroki/
├── .github/
│   └── workflows/
│       ├── docs.yml
│       ├── formatting.yaml
│       ├── pyright.yml
│       └── pytest.yml
├── .gitignore
├── LICENSE
├── README.md  [只读参考；未注释]
├── benchmark/
│   └── ik_benchmark.py  [本轮中文注释]
├── docs/
│   ├── Makefile
│   ├── README.md
│   ├── requirements.txt
│   ├── source/
│   │   ├── _static/
│   │   │   ├── basic_ik.mov
│   │   │   ├── css/
│   │   │   │   └── custom.css
│   │   │   ├── logo.svg
│   │   │   └── logo_dark.svg
│   │   ├── _templates/
│   │   │   └── sidebar/
│   │   │       └── brand.html
│   │   ├── conf.py  [本轮中文注释]
│   │   ├── examples/
│   │   │   ├── 01_basic_ik.rst
│   │   │   ├── 02_bimanual_ik.rst
│   │   │   ├── 03_mobile_ik.rst
│   │   │   ├── 04_ik_with_coll.rst
│   │   │   ├── 05_ik_with_manipulability.rst
│   │   │   ├── 06_online_planning.rst
│   │   │   ├── 07_trajopt.rst
│   │   │   ├── 08_ik_with_mimic_joints.rst
│   │   │   ├── 09_hand_retargeting.rst
│   │   │   ├── 10_humanoid_retargeting.rst
│   │   │   ├── 11_hand_retargeting_fancy.rst
│   │   │   └── 12_humanoid_retargeting_fancy.rst
│   │   ├── index.rst
│   │   └── misc/
│   │       └── writing_manual_jac.rst
│   └── update_example_docs.py  [本轮中文注释]
├── examples/
│   ├── 01_basic_ik.py  [本轮中文注释]
│   ├── 02_bimanual_ik.py  [本轮中文注释]
│   ├── 03_mobile_ik.py  [本轮中文注释]
│   ├── 04_ik_with_coll.py  [本轮中文注释]
│   ├── 05_ik_with_manipulability.py  [本轮中文注释]
│   ├── 06_online_planning.py  [本轮中文注释]
│   ├── 07_trajopt.py  [本轮中文注释]
│   ├── 08_ik_with_mimic_joints.py  [本轮中文注释]
│   ├── 09_hand_retargeting.py  [本轮中文注释]
│   ├── 10_humanoid_retargeting.py  [本轮中文注释]
│   ├── 11_hand_retargeting_fancy.py  [本轮中文注释]
│   ├── 12_humanoid_retargeting_fancy.py  [本轮中文注释]
│   ├── 13_ik_with_locked_joints.py  [本轮中文注释]
│   ├── 14_singularity_aware_ik.py  [本轮中文注释]
│   ├── assets/
│   │   ├── panda_spheres.json
│   │   └── ur5_spheres.json
│   ├── pyroki_snippets/
│   │   ├── __init__.py  [本轮中文注释]
│   │   ├── _jparse.py  [本轮中文注释]
│   │   ├── _online_planning.py  [本轮中文注释]
│   │   ├── _solve_ik.py  [本轮中文注释]
│   │   ├── _solve_ik_with_base.py  [本轮中文注释]
│   │   ├── _solve_ik_with_collision.py  [本轮中文注释]
│   │   ├── _solve_ik_with_manipulability.py  [本轮中文注释]
│   │   ├── _solve_ik_with_multiple_targets.py  [本轮中文注释]
│   │   └── _trajopt.py  [本轮中文注释]
│   └── retarget_helpers/
│       ├── _utils.py  [本轮中文注释]
│       ├── hand/
│       │   ├── dexycb_motion.pkl
│       │   └── shadowhand_urdf.zip
│       └── humanoid/
│           ├── heightmap.npy
│           ├── left_foot_contact.npy
│           ├── right_foot_contact.npy
│           └── smpl_keypoints.npy
├── pyproject.toml  [只读参考；未注释]
├── src/
│   └── pyroki/
│       ├── __init__.py  [本轮中文注释]
│       ├── _residuals/
│       │   ├── __init__.py  [本轮中文注释]
│       │   ├── _pose_residual_analytic_jac.py  [本轮中文注释]
│       │   ├── _pose_residual_numerical_jac.py  [本轮中文注释]
│       │   └── _residuals.py  [本轮中文注释]
│       ├── _robot.py  [本轮中文注释]
│       ├── _robot_urdf_parser.py  [本轮中文注释]
│       ├── collision/
│       │   ├── __init__.py  [本轮中文注释]
│       │   ├── _collision.py  [本轮中文注释]
│       │   ├── _geometry.py  [本轮中文注释]
│       │   ├── _geometry_pairs.py  [本轮中文注释]
│       │   ├── _robot_collision.py  [本轮中文注释]
│       │   └── _utils.py  [本轮中文注释]
│       ├── costs.py  [本轮中文注释]
│       ├── utils.py  [本轮中文注释]
│       └── viewer/
│           ├── __init__.py  [本轮中文注释]
│           ├── _batched_urdf.py  [本轮中文注释]
│           ├── _manipulability_ellipse.py  [本轮中文注释]
│           └── _weight_tuner.py  [本轮中文注释]
├── study/
│   ├── analysis.md  [新增学习材料]
│   ├── annotation_scope.json  [新增学习材料]
│   ├── annotation_validation.json  [新增学习材料]
│   ├── annotation_verification.json  [新增学习材料]
│   ├── code_symbols.md  [新增学习材料]
│   ├── code_tree.md  [新增学习材料]
│   ├── generate_study_index.py  [新增学习材料]
│   ├── repository.json  [新增学习材料]
│   └── verify_annotations.py  [新增学习材料]
└── tests/
    ├── test_ik_vmap.py  [本轮中文注释]
    └── viz_collision_sweep.py  [本轮中文注释]
```

## 本轮阅读入口

| 文件 | 范围 | 阅读深度 |
|---|---|---|
| [README.md](../README.md) | 只读参考，未注释 | reference: 版本/依赖/类型核对；未登记全文阅读深度 |
| [benchmark/ik_benchmark.py](../benchmark/ik_benchmark.py) | 中文注释 | full |
| [docs/source/conf.py](../docs/source/conf.py) | 中文注释 | full |
| [docs/update_example_docs.py](../docs/update_example_docs.py) | 中文注释 | full |
| [examples/01_basic_ik.py](../examples/01_basic_ik.py) | 中文注释 | full |
| [examples/02_bimanual_ik.py](../examples/02_bimanual_ik.py) | 中文注释 | full |
| [examples/03_mobile_ik.py](../examples/03_mobile_ik.py) | 中文注释 | full |
| [examples/04_ik_with_coll.py](../examples/04_ik_with_coll.py) | 中文注释 | full |
| [examples/05_ik_with_manipulability.py](../examples/05_ik_with_manipulability.py) | 中文注释 | full |
| [examples/06_online_planning.py](../examples/06_online_planning.py) | 中文注释 | full |
| [examples/07_trajopt.py](../examples/07_trajopt.py) | 中文注释 | full |
| [examples/08_ik_with_mimic_joints.py](../examples/08_ik_with_mimic_joints.py) | 中文注释 | full |
| [examples/09_hand_retargeting.py](../examples/09_hand_retargeting.py) | 中文注释 | full |
| [examples/10_humanoid_retargeting.py](../examples/10_humanoid_retargeting.py) | 中文注释 | full |
| [examples/11_hand_retargeting_fancy.py](../examples/11_hand_retargeting_fancy.py) | 中文注释 | full |
| [examples/12_humanoid_retargeting_fancy.py](../examples/12_humanoid_retargeting_fancy.py) | 中文注释 | full |
| [examples/13_ik_with_locked_joints.py](../examples/13_ik_with_locked_joints.py) | 中文注释 | full |
| [examples/14_singularity_aware_ik.py](../examples/14_singularity_aware_ik.py) | 中文注释 | full |
| [examples/pyroki_snippets/__init__.py](../examples/pyroki_snippets/__init__.py) | 中文注释 | full |
| [examples/pyroki_snippets/_jparse.py](../examples/pyroki_snippets/_jparse.py) | 中文注释 | full |
| [examples/pyroki_snippets/_online_planning.py](../examples/pyroki_snippets/_online_planning.py) | 中文注释 | full |
| [examples/pyroki_snippets/_solve_ik.py](../examples/pyroki_snippets/_solve_ik.py) | 中文注释 | full |
| [examples/pyroki_snippets/_solve_ik_with_base.py](../examples/pyroki_snippets/_solve_ik_with_base.py) | 中文注释 | full |
| [examples/pyroki_snippets/_solve_ik_with_collision.py](../examples/pyroki_snippets/_solve_ik_with_collision.py) | 中文注释 | full |
| [examples/pyroki_snippets/_solve_ik_with_manipulability.py](../examples/pyroki_snippets/_solve_ik_with_manipulability.py) | 中文注释 | full |
| [examples/pyroki_snippets/_solve_ik_with_multiple_targets.py](../examples/pyroki_snippets/_solve_ik_with_multiple_targets.py) | 中文注释 | full |
| [examples/pyroki_snippets/_trajopt.py](../examples/pyroki_snippets/_trajopt.py) | 中文注释 | full |
| [examples/retarget_helpers/_utils.py](../examples/retarget_helpers/_utils.py) | 中文注释 | full |
| [pyproject.toml](../pyproject.toml) | 只读参考，未注释 | reference: 版本/依赖/类型核对；未登记全文阅读深度 |
| [src/pyroki/__init__.py](../src/pyroki/__init__.py) | 中文注释 | full |
| [src/pyroki/_residuals/__init__.py](../src/pyroki/_residuals/__init__.py) | 中文注释 | full |
| [src/pyroki/_residuals/_pose_residual_analytic_jac.py](../src/pyroki/_residuals/_pose_residual_analytic_jac.py) | 中文注释 | full |
| [src/pyroki/_residuals/_pose_residual_numerical_jac.py](../src/pyroki/_residuals/_pose_residual_numerical_jac.py) | 中文注释 | full |
| [src/pyroki/_residuals/_residuals.py](../src/pyroki/_residuals/_residuals.py) | 中文注释 | full |
| [src/pyroki/_robot.py](../src/pyroki/_robot.py) | 中文注释 | full |
| [src/pyroki/_robot_urdf_parser.py](../src/pyroki/_robot_urdf_parser.py) | 中文注释 | full |
| [src/pyroki/collision/__init__.py](../src/pyroki/collision/__init__.py) | 中文注释 | full |
| [src/pyroki/collision/_collision.py](../src/pyroki/collision/_collision.py) | 中文注释 | full |
| [src/pyroki/collision/_geometry.py](../src/pyroki/collision/_geometry.py) | 中文注释 | full |
| [src/pyroki/collision/_geometry_pairs.py](../src/pyroki/collision/_geometry_pairs.py) | 中文注释 | full |
| [src/pyroki/collision/_robot_collision.py](../src/pyroki/collision/_robot_collision.py) | 中文注释 | full |
| [src/pyroki/collision/_utils.py](../src/pyroki/collision/_utils.py) | 中文注释 | full |
| [src/pyroki/costs.py](../src/pyroki/costs.py) | 中文注释 | full |
| [src/pyroki/utils.py](../src/pyroki/utils.py) | 中文注释 | full |
| [src/pyroki/viewer/__init__.py](../src/pyroki/viewer/__init__.py) | 中文注释 | full |
| [src/pyroki/viewer/_batched_urdf.py](../src/pyroki/viewer/_batched_urdf.py) | 中文注释 | full |
| [src/pyroki/viewer/_manipulability_ellipse.py](../src/pyroki/viewer/_manipulability_ellipse.py) | 中文注释 | full |
| [src/pyroki/viewer/_weight_tuner.py](../src/pyroki/viewer/_weight_tuner.py) | 中文注释 | full |
| [tests/test_ik_vmap.py](../tests/test_ik_vmap.py) | 中文注释 | full |
| [tests/viz_collision_sweep.py](../tests/viz_collision_sweep.py) | 中文注释 | full |

目录树和行号是定位材料，不代表所有代码已理解或运行。子模块只保留固定引用，内部文件不在本轮范围。

重新生成：`python study/generate_study_index.py`。再运行 `python study/verify_annotations.py --report study/annotation_validation.json` 审查源码保留情况；两个工具均不运行上游项目。
