# pyroki · 源码定位索引

[初步分析](analysis.md) · [代码树](code_tree.md)

行号对应当前工作区。Python AST 索引全部活动定义；只存在于注释中的停用实现不计入定义。未展开文件明确标注，生成索引不代表完成阅读。

## benchmark/ik_benchmark.py

范围：本轮中文注释；10 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `newton_raphson` | function/method | [103](../benchmark/ik_benchmark.py#L103) |
| `newton_raphson.update` | function/method | [107](../benchmark/ik_benchmark.py#L107) |
| `roberts_sequence` | function/method | [116](../benchmark/ik_benchmark.py#L116) |
| `PyrokiIkBeamHelper` | class | [128](../benchmark/ik_benchmark.py#L128) |
| `PyrokiIkBeamHelper.__init__` | function/method | [130](../benchmark/ik_benchmark.py#L130) |
| `PyrokiIkBeamHelper.solve_ik` | function/method | [157](../benchmark/ik_benchmark.py#L157) |
| `PyrokiIkBeamHelper.solve_ik.solve_one` | function/method | [165](../benchmark/ik_benchmark.py#L165) |
| `PyrokiIkBeamHelper.forward_kinematics` | function/method | [239](../benchmark/ik_benchmark.py#L239) |
| `evaluate_pyroki_ik` | function/method | [250](../benchmark/ik_benchmark.py#L250) |
| `run_full_config_collision_free_ik` | function/method | [306](../benchmark/ik_benchmark.py#L306) |

## docs/source/conf.py

范围：本轮中文注释；1 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `setup` | function/method | [247](../docs/source/conf.py#L247) |

## docs/update_example_docs.py

范围：本轮中文注释；4 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `ExampleMetadata` | class | [18](../docs/update_example_docs.py#L18) |
| `ExampleMetadata.from_path` | function/method | [27](../docs/update_example_docs.py#L27) |
| `get_example_paths` | function/method | [57](../docs/update_example_docs.py#L57) |
| `main` | function/method | [67](../docs/update_example_docs.py#L67) |

## examples/01_basic_ik.py

范围：本轮中文注释；1 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `main` | function/method | [20](../examples/01_basic_ik.py#L20) |

## examples/02_bimanual_ik.py

范围：本轮中文注释；1 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `main` | function/method | [18](../examples/02_bimanual_ik.py#L18) |

## examples/03_mobile_ik.py

范围：本轮中文注释；1 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `main` | function/method | [18](../examples/03_mobile_ik.py#L18) |

## examples/04_ik_with_coll.py

范围：本轮中文注释；3 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `main` | function/method | [24](../examples/04_ik_with_coll.py#L24) |
| `_get_link_frame_paths` | function/method | [129](../examples/04_ik_with_coll.py#L129) |
| `_create_collision_handles` | function/method | [155](../examples/04_ik_with_coll.py#L155) |

## examples/05_ik_with_manipulability.py

范围：本轮中文注释；1 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `main` | function/method | [18](../examples/05_ik_with_manipulability.py#L18) |

## examples/06_online_planning.py

范围：本轮中文注释；1 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `main` | function/method | [20](../examples/06_online_planning.py#L20) |

## examples/07_trajopt.py

范围：本轮中文注释；1 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `main` | function/method | [26](../examples/07_trajopt.py#L26) |

## examples/08_ik_with_mimic_joints.py

范围：本轮中文注释；4 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `create_chain_xml` | function/method | [25](../examples/08_ik_with_mimic_joints.py#L25) |
| `create_chain_xml.create_link` | function/method | [27](../examples/08_ik_with_mimic_joints.py#L27) |
| `create_chain_xml.create_joint` | function/method | [43](../examples/08_ik_with_mimic_joints.py#L43) |
| `main` | function/method | [89](../examples/08_ik_with_mimic_joints.py#L89) |

## examples/09_hand_retargeting.py

范围：本轮中文注释；10 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `RetargetingWeights` | class | [35](../examples/09_hand_retargeting.py#L35) |
| `main` | function/method | [47](../examples/09_hand_retargeting.py#L47) |
| `main.filename_handler` | function/method | [55](../examples/09_hand_retargeting.py#L55) |
| `main.generate_trajectory` | function/method | [157](../examples/09_hand_retargeting.py#L157) |
| `solve_retargeting` | function/method | [215](../examples/09_hand_retargeting.py#L215) |
| `solve_retargeting.ManoJointsScaleVar` | class | [230](../examples/09_hand_retargeting.py#L230) |
| `solve_retargeting.OffsetVar` | class | [235](../examples/09_hand_retargeting.py#L235) |
| `solve_retargeting.retargeting_cost` | function/method | [249](../examples/09_hand_retargeting.py#L249) |
| `solve_retargeting.pc_alignment_cost` | function/method | [315](../examples/09_hand_retargeting.py#L315) |
| `solve_retargeting.root_smoothness` | function/method | [332](../examples/09_hand_retargeting.py#L332) |

## examples/10_humanoid_retargeting.py

范围：本轮中文注释；8 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `RetargetingWeights` | class | [31](../examples/10_humanoid_retargeting.py#L31) |
| `main` | function/method | [39](../examples/10_humanoid_retargeting.py#L39) |
| `main.generate_trajectory` | function/method | [101](../examples/10_humanoid_retargeting.py#L101) |
| `solve_retargeting` | function/method | [141](../examples/10_humanoid_retargeting.py#L141) |
| `solve_retargeting.SmplJointsScaleVarG1` | class | [165](../examples/10_humanoid_retargeting.py#L165) |
| `solve_retargeting.OffsetVar` | class | [170](../examples/10_humanoid_retargeting.py#L170) |
| `solve_retargeting.retargeting_cost` | function/method | [184](../examples/10_humanoid_retargeting.py#L184) |
| `solve_retargeting.pc_alignment_cost` | function/method | [244](../examples/10_humanoid_retargeting.py#L244) |

## examples/11_hand_retargeting_fancy.py

范围：本轮中文注释；12 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `RetargetingWeights` | class | [34](../examples/11_hand_retargeting_fancy.py#L34) |
| `main` | function/method | [50](../examples/11_hand_retargeting_fancy.py#L50) |
| `main.filename_handler` | function/method | [58](../examples/11_hand_retargeting_fancy.py#L58) |
| `main.generate_trajectory` | function/method | [156](../examples/11_hand_retargeting_fancy.py#L156) |
| `solve_retargeting` | function/method | [216](../examples/11_hand_retargeting_fancy.py#L216) |
| `solve_retargeting.ManoJointsScaleVar` | class | [234](../examples/11_hand_retargeting_fancy.py#L234) |
| `solve_retargeting.OffsetVar` | class | [239](../examples/11_hand_retargeting_fancy.py#L239) |
| `solve_retargeting.retargeting_cost` | function/method | [253](../examples/11_hand_retargeting_fancy.py#L253) |
| `solve_retargeting.scale_regularization` | function/method | [319](../examples/11_hand_retargeting_fancy.py#L319) |
| `solve_retargeting.pc_alignment_cost` | function/method | [337](../examples/11_hand_retargeting_fancy.py#L337) |
| `solve_retargeting.root_smoothness` | function/method | [354](../examples/11_hand_retargeting_fancy.py#L354) |
| `solve_retargeting.contact_cost` | function/method | [367](../examples/11_hand_retargeting_fancy.py#L367) |

## examples/12_humanoid_retargeting_fancy.py

范围：本轮中文注释；13 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `RetargetingWeights` | class | [32](../examples/12_humanoid_retargeting_fancy.py#L32) |
| `main` | function/method | [48](../examples/12_humanoid_retargeting_fancy.py#L48) |
| `main.generate_trajectory` | function/method | [115](../examples/12_humanoid_retargeting_fancy.py#L115) |
| `solve_retargeting` | function/method | [161](../examples/12_humanoid_retargeting_fancy.py#L161) |
| `solve_retargeting.SmplJointsScaleVarG1` | class | [194](../examples/12_humanoid_retargeting_fancy.py#L194) |
| `solve_retargeting.OffsetVar` | class | [199](../examples/12_humanoid_retargeting_fancy.py#L199) |
| `solve_retargeting.retargeting_cost` | function/method | [213](../examples/12_humanoid_retargeting_fancy.py#L213) |
| `solve_retargeting.scale_regularization` | function/method | [273](../examples/12_humanoid_retargeting_fancy.py#L273) |
| `solve_retargeting.pc_alignment_cost` | function/method | [291](../examples/12_humanoid_retargeting_fancy.py#L291) |
| `solve_retargeting.floor_contact_cost` | function/method | [308](../examples/12_humanoid_retargeting_fancy.py#L308) |
| `solve_retargeting.root_smoothness` | function/method | [374](../examples/12_humanoid_retargeting_fancy.py#L374) |
| `solve_retargeting.skating_cost` | function/method | [386](../examples/12_humanoid_retargeting_fancy.py#L386) |
| `solve_retargeting.world_collision_cost` | function/method | [425](../examples/12_humanoid_retargeting_fancy.py#L425) |

## examples/13_ik_with_locked_joints.py

范围：本轮中文注释；4 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `_solve_ik` | function/method | [26](../examples/13_ik_with_locked_joints.py#L26) |
| `_compute_pose_error` | function/method | [86](../examples/13_ik_with_locked_joints.py#L86) |
| `main` | function/method | [112](../examples/13_ik_with_locked_joints.py#L112) |
| `main._` | function/method | [151](../examples/13_ik_with_locked_joints.py#L151) |

## examples/14_singularity_aware_ik.py

范围：本轮中文注释；1 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `main` | function/method | [19](../examples/14_singularity_aware_ik.py#L19) |

## examples/pyroki_snippets/__init__.py

范围：本轮中文注释；0 个定义。

[模块入口](../examples/pyroki_snippets/__init__.py)：没有活动类/函数定义。

## examples/pyroki_snippets/_jparse.py

范围：本轮中文注释；8 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `compute_jacobian` | function/method | [26](../examples/pyroki_snippets/_jparse.py#L26) |
| `compute_jacobian.get_pose_components` | function/method | [60](../examples/pyroki_snippets/_jparse.py#L60) |
| `jparse_pseudoinverse` | function/method | [72](../examples/pyroki_snippets/_jparse.py#L72) |
| `pinv` | function/method | [183](../examples/pyroki_snippets/_jparse.py#L183) |
| `damped_least_squares` | function/method | [189](../examples/pyroki_snippets/_jparse.py#L189) |
| `manipulability_measure` | function/method | [208](../examples/pyroki_snippets/_jparse.py#L208) |
| `inverse_condition_number` | function/method | [215](../examples/pyroki_snippets/_jparse.py#L215) |
| `jparse_step` | function/method | [223](../examples/pyroki_snippets/_jparse.py#L223) |

## examples/pyroki_snippets/_online_planning.py

范围：本轮中文注释；8 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `solve_online_planning` | function/method | [15](../examples/pyroki_snippets/_online_planning.py#L15) |
| `_solve_online_planning_jax` | function/method | [62](../examples/pyroki_snippets/_online_planning.py#L62) |
| `_solve_online_planning_jax.batched_rplus` | function/method | [76](../examples/pyroki_snippets/_online_planning.py#L76) |
| `_solve_online_planning_jax.BatchedSE3Var` | class | [85](../examples/pyroki_snippets/_online_planning.py#L85) |
| `_solve_online_planning_jax.match_joint_to_pose_cost` | function/method | [110](../examples/pyroki_snippets/_online_planning.py#L110) |
| `_solve_online_planning_jax.pose_smoothness_cost` | function/method | [125](../examples/pyroki_snippets/_online_planning.py#L125) |
| `_solve_online_planning_jax.pose_match_cost` | function/method | [134](../examples/pyroki_snippets/_online_planning.py#L134) |
| `_solve_online_planning_jax.match_start_pose_cost` | function/method | [145](../examples/pyroki_snippets/_online_planning.py#L145) |

## examples/pyroki_snippets/_solve_ik.py

范围：本轮中文注释；2 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `solve_ik` | function/method | [17](../examples/pyroki_snippets/_solve_ik.py#L17) |
| `_solve_ik_jax` | function/method | [49](../examples/pyroki_snippets/_solve_ik.py#L49) |

## examples/pyroki_snippets/_solve_ik_with_base.py

范围：本轮中文注释；4 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `solve_ik_with_base` | function/method | [13](../examples/pyroki_snippets/_solve_ik_with_base.py#L13) |
| `_solve_ik_jax` | function/method | [70](../examples/pyroki_snippets/_solve_ik_with_base.py#L70) |
| `_solve_ik_jax.retract_fn` | function/method | [82](../examples/pyroki_snippets/_solve_ik_with_base.py#L82) |
| `_solve_ik_jax.ConstrainedSE3Var` | class | [89](../examples/pyroki_snippets/_solve_ik_with_base.py#L89) |

## examples/pyroki_snippets/_solve_ik_with_collision.py

范围：本轮中文注释；2 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `solve_ik_with_collision` | function/method | [18](../examples/pyroki_snippets/_solve_ik_with_collision.py#L18) |
| `_solve_ik_with_collision_jax` | function/method | [58](../examples/pyroki_snippets/_solve_ik_with_collision.py#L58) |

## examples/pyroki_snippets/_solve_ik_with_manipulability.py

范围：本轮中文注释；2 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `solve_ik_with_manipulability` | function/method | [18](../examples/pyroki_snippets/_solve_ik_with_manipulability.py#L18) |
| `_solve_ik_jax` | function/method | [59](../examples/pyroki_snippets/_solve_ik_with_manipulability.py#L59) |

## examples/pyroki_snippets/_solve_ik_with_multiple_targets.py

范围：本轮中文注释；2 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `solve_ik_with_multiple_targets` | function/method | [18](../examples/pyroki_snippets/_solve_ik_with_multiple_targets.py#L18) |
| `_solve_ik_jax` | function/method | [54](../examples/pyroki_snippets/_solve_ik_with_multiple_targets.py#L54) |

## examples/pyroki_snippets/_trajopt.py

范围：本轮中文注释；6 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `solve_trajopt` | function/method | [16](../examples/pyroki_snippets/_trajopt.py#L16) |
| `solve_trajopt.start_pose_constraint` | function/method | [95](../examples/pyroki_snippets/_trajopt.py#L95) |
| `solve_trajopt.end_pose_constraint` | function/method | [100](../examples/pyroki_snippets/_trajopt.py#L100) |
| `solve_trajopt.compute_world_coll_residual` | function/method | [121](../examples/pyroki_snippets/_trajopt.py#L121) |
| `solve_iks_with_collision` | function/method | [172](../examples/pyroki_snippets/_trajopt.py#L172) |
| `solve_iks_with_collision.joint_similarity_cost` | function/method | [228](../examples/pyroki_snippets/_trajopt.py#L228) |

## examples/retarget_helpers/_utils.py

范围：本轮中文注释；4 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `create_conn_tree` | function/method | [8](../examples/retarget_helpers/_utils.py#L8) |
| `create_conn_tree.is_direct_chain_connection` | function/method | [18](../examples/retarget_helpers/_utils.py#L18) |
| `get_humanoid_retarget_indices` | function/method | [152](../examples/retarget_helpers/_utils.py#L152) |
| `get_mapping_from_mano_to_shadow` | function/method | [211](../examples/retarget_helpers/_utils.py#L211) |

## src/pyroki/__init__.py

范围：本轮中文注释；0 个定义。

[模块入口](../src/pyroki/__init__.py)：没有活动类/函数定义。

## src/pyroki/_residuals/__init__.py

范围：本轮中文注释；0 个定义。

[模块入口](../src/pyroki/_residuals/__init__.py)：没有活动类/函数定义。

## src/pyroki/_residuals/_pose_residual_analytic_jac.py

范围：本轮中文注释；5 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `_get_actuated_joints_applied_to_target` | function/method | [24](../src/pyroki/_residuals/_pose_residual_analytic_jac.py#L24) |
| `_get_actuated_joints_applied_to_target.body_fun` | function/method | [42](../src/pyroki/_residuals/_pose_residual_analytic_jac.py#L42) |
| `pose_cost_analytic_jac` | function/method | [82](../src/pyroki/_residuals/_pose_residual_analytic_jac.py#L82) |
| `_pose_cost_jac` | function/method | [163](../src/pyroki/_residuals/_pose_residual_analytic_jac.py#L163) |
| `_pose_cost_analytical_jac` | function/method | [245](../src/pyroki/_residuals/_pose_residual_analytic_jac.py#L245) |

## src/pyroki/_residuals/_pose_residual_numerical_jac.py

范围：本轮中文注释；3 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `_pose_cost_jac` | function/method | [26](../src/pyroki/_residuals/_pose_residual_numerical_jac.py#L26) |
| `_pose_cost_jac.finite_difference_jac` | function/method | [42](../src/pyroki/_residuals/_pose_residual_numerical_jac.py#L42) |
| `pose_cost_numerical_jac` | function/method | [61](../src/pyroki/_residuals/_pose_residual_numerical_jac.py#L61) |

## src/pyroki/_residuals/_residuals.py

范围：本轮中文注释；16 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `pose_residual` | function/method | [26](../src/pyroki/_residuals/_residuals.py#L26) |
| `pose_with_base_residual` | function/method | [48](../src/pyroki/_residuals/_residuals.py#L48) |
| `limit_residual` | function/method | [76](../src/pyroki/_residuals/_residuals.py#L76) |
| `limit_velocity_residual` | function/method | [99](../src/pyroki/_residuals/_residuals.py#L99) |
| `rest_residual` | function/method | [123](../src/pyroki/_residuals/_residuals.py#L123) |
| `rest_with_base_residual` | function/method | [134](../src/pyroki/_residuals/_residuals.py#L134) |
| `smoothness_residual` | function/method | [150](../src/pyroki/_residuals/_residuals.py#L150) |
| `_compute_manip_yoshikawa` | function/method | [164](../src/pyroki/_residuals/_residuals.py#L164) |
| `manipulability_residual` | function/method | [180](../src/pyroki/_residuals/_residuals.py#L180) |
| `self_collision_residual` | function/method | [203](../src/pyroki/_residuals/_residuals.py#L203) |
| `world_collision_residual` | function/method | [219](../src/pyroki/_residuals/_residuals.py#L219) |
| `five_point_velocity_residual` | function/method | [238](../src/pyroki/_residuals/_residuals.py#L238) |
| `five_point_acceleration_residual` | function/method | [267](../src/pyroki/_residuals/_residuals.py#L267) |
| `limit_acceleration_residual` | function/method | [291](../src/pyroki/_residuals/_residuals.py#L291) |
| `five_point_jerk_residual` | function/method | [321](../src/pyroki/_residuals/_residuals.py#L321) |
| `limit_jerk_residual` | function/method | [348](../src/pyroki/_residuals/_residuals.py#L348) |

## src/pyroki/_robot.py

范围：本轮中文注释；7 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `Robot` | class | [20](../src/pyroki/_robot.py#L20) |
| `Robot.from_urdf` | function/method | [34](../src/pyroki/_robot.py#L34) |
| `Robot.from_urdf.JointVar` | class | [58](../src/pyroki/_robot.py#L58) |
| `Robot.forward_kinematics` | function/method | [73](../src/pyroki/_robot.py#L73) |
| `Robot._link_poses_from_joint_poses` | function/method | [97](../src/pyroki/_robot.py#L97) |
| `Robot._forward_kinematics_joints` | function/method | [120](../src/pyroki/_robot.py#L120) |
| `Robot._forward_kinematics_joints.compute_transform` | function/method | [163](../src/pyroki/_robot.py#L163) |

## src/pyroki/_robot_urdf_parser.py

范围：本轮中文注释；12 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `JointInfo` | class | [19](../src/pyroki/_robot_urdf_parser.py#L19) |
| `JointInfo._map_to_full_joint_space` | function/method | [65](../src/pyroki/_robot_urdf_parser.py#L65) |
| `JointInfo.get_full_config` | function/method | [113](../src/pyroki/_robot_urdf_parser.py#L113) |
| `LinkInfo` | class | [133](../src/pyroki/_robot_urdf_parser.py#L133) |
| `RobotURDFParser` | class | [142](../src/pyroki/_robot_urdf_parser.py#L142) |
| `RobotURDFParser._topologically_sort_joints` | function/method | [147](../src/pyroki/_robot_urdf_parser.py#L147) |
| `RobotURDFParser.parse` | function/method | [218](../src/pyroki/_robot_urdf_parser.py#L218) |
| `RobotURDFParser._get_act_joint_idx_and_mimic` | function/method | [354](../src/pyroki/_robot_urdf_parser.py#L354) |
| `RobotURDFParser._get_joint_twist` | function/method | [388](../src/pyroki/_robot_urdf_parser.py#L388) |
| `RobotURDFParser._get_T_parent_joint` | function/method | [410](../src/pyroki/_robot_urdf_parser.py#L410) |
| `RobotURDFParser._get_joint_limits` | function/method | [440](../src/pyroki/_robot_urdf_parser.py#L440) |
| `RobotURDFParser._get_joint_limit_vel` | function/method | [472](../src/pyroki/_robot_urdf_parser.py#L472) |

## src/pyroki/collision/__init__.py

范围：本轮中文注释；0 个定义。

[模块入口](../src/pyroki/collision/__init__.py)：没有活动类/函数定义。

## src/pyroki/collision/_collision.py

范围：本轮中文注释；4 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `_get_coll_func` | function/method | [45](../src/pyroki/collision/_collision.py#L45) |
| `collide` | function/method | [67](../src/pyroki/collision/_collision.py#L67) |
| `pairwise_collide` | function/method | [88](../src/pyroki/collision/_collision.py#L88) |
| `colldist_from_sdf` | function/method | [142](../src/pyroki/collision/_collision.py#L142) |

## src/pyroki/collision/_geometry.py

范围：本轮中文注释；45 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `CollGeom` | class | [24](../src/pyroki/collision/_geometry.py#L24) |
| `CollGeom.get_batch_axes` | function/method | [34](../src/pyroki/collision/_geometry.py#L34) |
| `CollGeom.broadcast_to` | function/method | [45](../src/pyroki/collision/_geometry.py#L45) |
| `CollGeom.reshape` | function/method | [55](../src/pyroki/collision/_geometry.py#L55) |
| `CollGeom.transform` | function/method | [65](../src/pyroki/collision/_geometry.py#L65) |
| `CollGeom.transform_from_wxyz_position` | function/method | [77](../src/pyroki/collision/_geometry.py#L77) |
| `CollGeom._create_one_mesh` | function/method | [92](../src/pyroki/collision/_geometry.py#L92) |
| `CollGeom.to_trimesh` | function/method | [97](../src/pyroki/collision/_geometry.py#L97) |
| `HalfSpace` | class | [114](../src/pyroki/collision/_geometry.py#L114) |
| `HalfSpace.normal` | function/method | [119](../src/pyroki/collision/_geometry.py#L119) |
| `HalfSpace.offset` | function/method | [126](../src/pyroki/collision/_geometry.py#L126) |
| `HalfSpace.from_point_and_normal` | function/method | [132](../src/pyroki/collision/_geometry.py#L132) |
| `HalfSpace._create_one_mesh` | function/method | [149](../src/pyroki/collision/_geometry.py#L149) |
| `Sphere` | class | [165](../src/pyroki/collision/_geometry.py#L165) |
| `Sphere.radius` | function/method | [170](../src/pyroki/collision/_geometry.py#L170) |
| `Sphere.from_center_and_radius` | function/method | [176](../src/pyroki/collision/_geometry.py#L176) |
| `Sphere._create_one_mesh` | function/method | [200](../src/pyroki/collision/_geometry.py#L200) |
| `Capsule` | class | [214](../src/pyroki/collision/_geometry.py#L214) |
| `Capsule.radius` | function/method | [219](../src/pyroki/collision/_geometry.py#L219) |
| `Capsule.height` | function/method | [225](../src/pyroki/collision/_geometry.py#L225) |
| `Capsule.axis` | function/method | [231](../src/pyroki/collision/_geometry.py#L231) |
| `Capsule.from_radius_height` | function/method | [238](../src/pyroki/collision/_geometry.py#L238) |
| `Capsule.from_trimesh` | function/method | [272](../src/pyroki/collision/_geometry.py#L272) |
| `Capsule._create_one_mesh` | function/method | [293](../src/pyroki/collision/_geometry.py#L293) |
| `Capsule.decompose_to_spheres` | function/method | [315](../src/pyroki/collision/_geometry.py#L315) |
| `Capsule.from_sphere_pairs` | function/method | [351](../src/pyroki/collision/_geometry.py#L351) |
| `Box` | class | [390](../src/pyroki/collision/_geometry.py#L390) |
| `Box.half_extent` | function/method | [395](../src/pyroki/collision/_geometry.py#L395) |
| `Box.extent` | function/method | [401](../src/pyroki/collision/_geometry.py#L401) |
| `Box.from_extent` | function/method | [407](../src/pyroki/collision/_geometry.py#L407) |
| `Box.get_corners_local` | function/method | [437](../src/pyroki/collision/_geometry.py#L437) |
| `Box.get_corners_world` | function/method | [461](../src/pyroki/collision/_geometry.py#L461) |
| `Box._create_one_mesh` | function/method | [468](../src/pyroki/collision/_geometry.py#L468) |
| `Heightmap` | class | [484](../src/pyroki/collision/_geometry.py#L484) |
| `Heightmap.x_scale` | function/method | [493](../src/pyroki/collision/_geometry.py#L493) |
| `Heightmap.y_scale` | function/method | [499](../src/pyroki/collision/_geometry.py#L499) |
| `Heightmap.height_scale` | function/method | [505](../src/pyroki/collision/_geometry.py#L505) |
| `Heightmap.rows` | function/method | [511](../src/pyroki/collision/_geometry.py#L511) |
| `Heightmap.cols` | function/method | [517](../src/pyroki/collision/_geometry.py#L517) |
| `Heightmap.from_height_data` | function/method | [523](../src/pyroki/collision/_geometry.py#L523) |
| `Heightmap.from_trimesh` | function/method | [541](../src/pyroki/collision/_geometry.py#L541) |
| `Heightmap.project_points` | function/method | [639](../src/pyroki/collision/_geometry.py#L639) |
| `Heightmap._interpolate_height_at_coords` | function/method | [649](../src/pyroki/collision/_geometry.py#L649) |
| `Heightmap._get_vertices_local` | function/method | [712](../src/pyroki/collision/_geometry.py#L712) |
| `Heightmap._create_one_mesh` | function/method | [746](../src/pyroki/collision/_geometry.py#L746) |

## src/pyroki/collision/_geometry_pairs.py

范围：本轮中文注释；15 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `_halfspace_sphere_dist` | function/method | [16](../src/pyroki/collision/_geometry_pairs.py#L16) |
| `halfspace_sphere` | function/method | [31](../src/pyroki/collision/_geometry_pairs.py#L31) |
| `halfspace_capsule` | function/method | [43](../src/pyroki/collision/_geometry_pairs.py#L43) |
| `_sphere_sphere_dist` | function/method | [65](../src/pyroki/collision/_geometry_pairs.py#L65) |
| `sphere_sphere` | function/method | [78](../src/pyroki/collision/_geometry_pairs.py#L78) |
| `sphere_capsule` | function/method | [90](../src/pyroki/collision/_geometry_pairs.py#L90) |
| `capsule_capsule` | function/method | [104](../src/pyroki/collision/_geometry_pairs.py#L104) |
| `heightmap_sphere` | function/method | [131](../src/pyroki/collision/_geometry_pairs.py#L131) |
| `heightmap_capsule` | function/method | [154](../src/pyroki/collision/_geometry_pairs.py#L154) |
| `heightmap_halfspace` | function/method | [195](../src/pyroki/collision/_geometry_pairs.py#L195) |
| `halfspace_box` | function/method | [239](../src/pyroki/collision/_geometry_pairs.py#L239) |
| `_sphere_box_dist` | function/method | [271](../src/pyroki/collision/_geometry_pairs.py#L271) |
| `sphere_box` | function/method | [294](../src/pyroki/collision/_geometry_pairs.py#L294) |
| `_point_to_box_sdf` | function/method | [304](../src/pyroki/collision/_geometry_pairs.py#L304) |
| `capsule_box` | function/method | [327](../src/pyroki/collision/_geometry_pairs.py#L327) |

## src/pyroki/collision/_robot_collision.py

范围：本轮中文注释；10 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `RobotCollision` | class | [27](../src/pyroki/collision/_robot_collision.py#L27) |
| `RobotCollision.from_urdf` | function/method | [47](../src/pyroki/collision/_robot_collision.py#L47) |
| `RobotCollision.from_sphere_decomposition` | function/method | [116](../src/pyroki/collision/_robot_collision.py#L116) |
| `RobotCollision._compute_collision_pair_indices` | function/method | [210](../src/pyroki/collision/_robot_collision.py#L210) |
| `RobotCollision._get_trimesh_collision_geometries` | function/method | [281](../src/pyroki/collision/_robot_collision.py#L281) |
| `RobotCollision.at_config` | function/method | [364](../src/pyroki/collision/_robot_collision.py#L364) |
| `RobotCollision.get_link_collision_meshes` | function/method | [397](../src/pyroki/collision/_robot_collision.py#L397) |
| `RobotCollision.get_swept_capsules` | function/method | [428](../src/pyroki/collision/_robot_collision.py#L428) |
| `RobotCollision.compute_self_collision_distance` | function/method | [479](../src/pyroki/collision/_robot_collision.py#L479) |
| `RobotCollision.compute_world_collision_distance` | function/method | [516](../src/pyroki/collision/_robot_collision.py#L516) |

## src/pyroki/collision/_utils.py

范围：本轮中文注释；5 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `make_frame` | function/method | [13](../src/pyroki/collision/_utils.py#L13) |
| `normalize` | function/method | [36](../src/pyroki/collision/_utils.py#L36) |
| `normalize_with_norm` | function/method | [45](../src/pyroki/collision/_utils.py#L45) |
| `closest_segment_point` | function/method | [58](../src/pyroki/collision/_utils.py#L58) |
| `closest_segment_to_segment_points` | function/method | [73](../src/pyroki/collision/_utils.py#L73) |

## src/pyroki/costs.py

范围：本轮中文注释；0 个定义。

[模块入口](../src/pyroki/costs.py)：没有活动类/函数定义。

## src/pyroki/utils.py

范围：本轮中文注释；3 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `stopwatch` | function/method | [14](../src/pyroki/utils.py#L14) |
| `_log` | function/method | [25](../src/pyroki/utils.py#L25) |
| `jax_log` | function/method | [30](../src/pyroki/utils.py#L30) |

## src/pyroki/viewer/__init__.py

范围：本轮中文注释；0 个定义。

[模块入口](../src/pyroki/viewer/__init__.py)：没有活动类/函数定义。

## src/pyroki/viewer/_batched_urdf.py

范围：本轮中文注释；0 个定义。

[模块入口](../src/pyroki/viewer/_batched_urdf.py)：没有活动类/函数定义。

## src/pyroki/viewer/_manipulability_ellipse.py

范围：本轮中文注释；7 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `ManipulabilityEllipse` | class | [18](../src/pyroki/viewer/_manipulability_ellipse.py#L18) |
| `ManipulabilityEllipse.__init__` | function/method | [22](../src/pyroki/viewer/_manipulability_ellipse.py#L22) |
| `ManipulabilityEllipse._create_mesh_handle` | function/method | [68](../src/pyroki/viewer/_manipulability_ellipse.py#L68) |
| `ManipulabilityEllipse.set_target_link` | function/method | [90](../src/pyroki/viewer/_manipulability_ellipse.py#L90) |
| `ManipulabilityEllipse.update` | function/method | [112](../src/pyroki/viewer/_manipulability_ellipse.py#L112) |
| `ManipulabilityEllipse.set_visibility` | function/method | [187](../src/pyroki/viewer/_manipulability_ellipse.py#L187) |
| `ManipulabilityEllipse.remove` | function/method | [204](../src/pyroki/viewer/_manipulability_ellipse.py#L204) |

## src/pyroki/viewer/_weight_tuner.py

范围：本轮中文注释；4 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `WeightTuner` | class | [14](../src/pyroki/viewer/_weight_tuner.py#L14) |
| `WeightTuner.__init__` | function/method | [32](../src/pyroki/viewer/_weight_tuner.py#L32) |
| `WeightTuner.get_weights` | function/method | [100](../src/pyroki/viewer/_weight_tuner.py#L100) |
| `WeightTuner.reset_weights` | function/method | [116](../src/pyroki/viewer/_weight_tuner.py#L116) |

## tests/test_ik_vmap.py

范围：本轮中文注释；9 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `_solve_ik` | function/method | [21](../tests/test_ik_vmap.py#L21) |
| `_solve_ik_coll` | function/method | [54](../tests/test_ik_vmap.py#L54) |
| `test_unit_collision_batching` | function/method | [86](../tests/test_ik_vmap.py#L86) |
| `test_unit_collision_batching.broadcast_vary` | function/method | [103](../tests/test_ik_vmap.py#L103) |
| `test_unit_collision_batching.compute` | function/method | [112](../tests/test_ik_vmap.py#L112) |
| `test_unit_collision_batching_topology_mismatch` | function/method | [123](../tests/test_ik_vmap.py#L123) |
| `test_integration_ik_basic` | function/method | [165](../tests/test_ik_vmap.py#L165) |
| `test_integration_ik_mixed_batching` | function/method | [183](../tests/test_ik_vmap.py#L183) |
| `test_integration_ik_fully_batched` | function/method | [215](../tests/test_ik_vmap.py#L215) |

## tests/viz_collision_sweep.py

范围：本轮中文注释；3 个定义。

| 符号 | 类型 | 当前行号 |
|---|---|---|
| `_solve_ik_jax` | function/method | [39](../tests/viz_collision_sweep.py#L39) |
| `solve_ik` | function/method | [75](../tests/viz_collision_sweep.py#L75) |
| `main` | function/method | [99](../tests/viz_collision_sweep.py#L99) |

## C/C++ 注释落点

只列本轮中文注释对应的下一条源码行；不是全仓 C++ 符号解析，不覆盖子模块。

| 文件 / 行号 | 中文说明 |
|---|---|
| — | 本轮未注释 C/C++ 文件 |

原始 Python：48 文件、262 活动定义；C/C++：0 注释落点。

新增学习工具不算上游符号，其完整中文说明在工具源码中。重新生成使用 [generate_study_index.py](generate_study_index.py)；静态验证使用 [verify_annotations.py](verify_annotations.py)。
