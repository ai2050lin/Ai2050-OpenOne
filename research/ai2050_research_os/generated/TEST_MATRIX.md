# 测试矩阵

> 自动生成于账本基线 `2026-08-11`。请修改 `registry/`，不要手工修改本文件。

| 测试 | 状态 | 成本 | 目标 | 前置 | 候选 | 拼图 |
|---|---|---|---|---|---|---|
| `TB00` 构念分型与接口解耦 | 预注册中 | low | 分离内容正确、格式正确、协议服从、解码成功和自然生成。 | — | H04, H05, H08 | P00, P02, P07 |
| `TB01` 测量与干预相机校准 | 计划中 | low | 在已知真值中验证可识别性、拒答、规范变化和干预特异性。 | TB00 | H03, H04, H05, H06, H07 | P01, P03, P06 |
| `TB02` 行为正交资格 | 计划中 | medium | 在全新世界中按原子构念授权内容、角色、操作、格式和生成对象。 | TB00 | H04, H05, H08 | P02, P07, P09 |
| `TB03` 全域低成本扫描 | 计划中 | medium | 在统一材料上构建层、位置、组件、生成步和缓存的全局观察轮廓。 | TB02 | H02, H06, H07 | P04 |
| `TB04` 未来响应张量试采集 | 计划中 | high | 在少量冻结事件和合法干预上建立可重算的类型化未来响应张量。 | TB01, TB02, TB03 | H03, H04, H05, H06, H07 | P01, P03, P06, P07 |
| `TB05` 结构发现与候选机制竞赛 | 计划中 | medium | 用同一响应张量比较固定方向、旋转子空间、联盟、因子化、路由和动力状态。 | TB04 | H00, H01, H02, H03, H04, H05, H06, H07, H08 | P04, P05, P06, P07, P09 |
| `TB06` 主动决定性干预 | 计划中 | high | 在预算内选择候选预测分歧最大、覆盖最高且控制合法的干预。 | TB05 | H00, H02, H03, H04, H05, H06, H07, H08 | P05, P06, P07 |
| `TB07` 一次性封存确认 | 计划中 | high | 在新世界、新组合、新干预和留一模型上批量裁决候选强版本。 | TB06 | H00, H01, H02, H03, H04, H05, H06, H07, H08 | P05, P06, P07 |
| `TB08` 同一对象因果闭环 | 计划中 | very_high | 让封存幸存机制完成充分、必要、救援、中介和最小联盟链。 | TB07 | H03, H04, H05, H06, H07, H08 | P05, P06, P07, P09 |
| `TB09` 自回归生成闭环 | 计划中 | very_high | 验证每个真实词元反馈、缓存更新、错误累积和停止由候选机制解释。 | TB08 | H05, H06, H07 | P08 |
| `TB10` 操作组合 | 计划中 | high | 从单操作结构预测未见组合、顺序和作用域。 | TB08 | H08 | P09 |
| `TB11` 跨协议与跨模型功能同构 | 计划中 | very_high | 寻找保持未来响应和操作交换结构的功能映射，而非物理坐标对齐。 | TB08 | H00, H02, H04, H05, H06, H07, H08 | P10, P12 |
| `TB12` 训练形成 | 计划中 | very_high | 分别预测是否形成、何时形成、形成哪种机制，并以训练干预验证。 | TB08 | H00, H03, H06, H07 | P11 |

## TB00：构念分型与接口解耦

- 目标：分离内容正确、格式正确、协议服从、解码成功和自然生成。
- 轴：content, format, prompt, decoder, evaluator
- 分区：calibration, discovery, confirmation
- 控制：correct_content_wrong_format, wrong_content_correct_format, synonym, multiple_reference, negation, quotation, self_correction
- 指标：content_accuracy, format_accuracy, semantic_accuracy, sequence_logprob, first_token_rank, eos_accuracy
- 通过：每个构念在其专属必需分区独立通过；不得使用异质合取总门。
- 失败：构念不能被四格控制或独立评价器分离时，登记 construct_failure。
- 产物：typed_construct_contract, typed_behavior_vector, authorization_matrix

## TB01：测量与干预相机校准

- 目标：在已知真值中验证可识别性、拒答、规范变化和干预特异性。
- 轴：target_type, observation_contract, intervention_operator, gauge, numeric_layout
- 分区：calibration, blind_confirmation
- 控制：identifiability_twins, matched_null, same_answer_carrier, wrong_donor, random_same_norm, off_manifold_sentinel
- 指标：target_recovery, abstention_accuracy, null_effect, carrier_effect, unseen_intervention_error, gauge_equivariance
- 通过：目标恢复、合法拒答、匹配控制、未见干预预测和数值布局声明分别通过。
- 失败：相机失败则禁止进入自然模型；观察不足则登记 identifiability_failure。
- 产物：instrument_registry, intervention_semantics_contract, known_truth_audit

## TB02：行为正交资格

- 目标：在全新世界中按原子构念授权内容、角色、操作、格式和生成对象。
- 轴：world, content, role, operation, expression, model
- 分区：discovery, confirmation, unseen_composition
- 控制：surface_only, semantic_neighbor, same_answer_wrong_binding, prior_cancelled_candidates, free_generation
- 指标：cluster_accuracy, worst_partition_accuracy, content_format_confusion, calibration, full_sequence_probability
- 通过：按原子作用域逐项授权；所有声明必需分区分别达到冻结门。
- 失败：只关闭失败原子和接口，不由家族总门吞并已通过对象。
- 产物：behavior_qualification_matrix, eligible_object_registry

## TB03：全域低成本扫描

- 目标：在统一材料上构建层、位置、组件、生成步和缓存的全局观察轮廓。
- 轴：layer, token_role, component, generation_step, protocol, model
- 分区：discovery_only
- 控制：label_permutation, matched_surface, random_world, fixed_color_scale, cluster_stability
- 指标：candidate_margin, prefix_margin, sequence_logprob, attention_delta, mlp_delta, change_point_score, cache_summary
- 通过：全部预注册事件统一采集，缺失率和数值审计通过；本阶段不产生机制确认。
- 失败：采集或观察构念不稳定则回到 TB00/TB01，不通过换热点修补。
- 产物：observational_tensor, trajectory_atlas, change_point_candidates

## TB04：未来响应张量试采集

- 目标：在少量冻结事件和合法干预上建立可重算的类型化未来响应张量。
- 轴：event, intervention, readout, protocol, generation_step, model
- 分区：calibration, discovery, heldout_intervention
- 控制：matched_null, wrong_donor, same_answer_wrong_binding, random_same_norm, zero, recompute, cache_mode
- 指标：response_reproducibility, control_specificity, direction_accuracy, trajectory_error, state_validity
- 通过：响应可独立复算，matched 控制近零，未知干预留出规则可执行。
- 失败：控制或状态合法性失败时停止扩大张量，登记 instrument_failure 或 specificity_failure。
- 产物：response_tensor_pilot, response_data_dictionary, intervention_audit

## TB05：结构发现与候选机制竞赛

- 目标：用同一响应张量比较固定方向、旋转子空间、联盟、因子化、路由和动力状态。
- 轴：model_family, structure_complexity, world_holdout, intervention_holdout, rollout_horizon
- 分区：discovery, selection
- 控制：design_only, lookup_table, unrestricted_oracle, label_permutation, complexity_matched_random
- 指标：normalized_predictive_gain, mdl, world_holdout_loss, intervention_holdout_loss, multi_step_loss, calibration
- 通过：每个候选输出冻结预测、复杂度、死亡条件和最大分歧实验；不在本阶段确认。
- 失败：不能超过设计基线或无法产生区别性预测的候选降级或拒答。
- 产物：candidate_models, hypothesis_scoreboard, disagreement_matrix

## TB06：主动决定性干预

- 目标：在预算内选择候选预测分歧最大、覆盖最高且控制合法的干预。
- 轴：hypothesis_pair, intervention, event, protocol_pair, world, cost
- 分区：selection_only
- 控制：pair_coverage, cost_match, no_confirmation_access, negative_control_quota
- 指标：expected_disagreement, pair_coverage, information_gain, cost, control_coverage
- 通过：达到冻结的假说对覆盖和控制配额，且确认集保持密封。
- 失败：候选不可分则登记 observational_equivalence，不任意增加实验。
- 产物：decisive_intervention_set, pair_coverage_report, sealed_predictions

## TB07：一次性封存确认

- 目标：在新世界、新组合、新干预和留一模型上批量裁决候选强版本。
- 轴：new_world, new_paraphrase, new_composition, new_intervention, heldout_model
- 分区：confirmation_only
- 控制：frozen_code, frozen_thresholds, independent_audit, worst_partition_gate
- 指标：normalized_predictive_gain, direction_ci_lower, cross_protocol_retention, cross_model_retention, mdl_ratio
- 通过：所有声明必需分区分别通过；最多一至两个候选进入闭环。
- 失败：独特预测失败即在声明适用域限定否决，不换层、模板或子集重开。
- 产物：sealed_results, batch_verdicts, independent_audit

## TB08：同一对象因果闭环

- 目标：让封存幸存机制完成充分、必要、救援、中介和最小联盟链。
- 轴：sufficiency, necessity, rescue, mediation, minimality, context
- 分区：closure_discovery, closure_confirmation
- 控制：wrong_donor, same_answer_wrong_binding, random_support, delete_one, joint_delete, recompute
- 指标：behavior_transfer, necessity_drop, rescue_gain, wrong_rescue_advantage, mediation_fraction, coalition_stability
- 通过：同一对象的正确链通过，错误救援和匹配控制失败，最小联盟跨世界稳定。
- 失败：明确登记最高闭合层级，不用局部充分性替代完整功能状态。
- 产物：causal_closure_report, minimal_coalition, mediation_audit

## TB09：自回归生成闭环

- 目标：验证每个真实词元反馈、缓存更新、错误累积和停止由候选机制解释。
- 轴：generation_step, prefix, sampling, kv_cache, eos, protocol
- 分区：closure_discovery, closure_confirmation
- 控制：teacher_forcing, free_rollout, cache_recompute, wrong_prefix, sampling_seed, eos_control
- 指标：prefix_margin, sequence_probability, content_accuracy, format_accuracy, cache_consistency, eos_accuracy, rollout_divergence
- 通过：完整轨迹和缓存闭合；首词元、候选评分或外部 trie 不得替代自然 rollout。
- 失败：登记 generation_failure，并保持 L4/L5 局部结论不被吞并。
- 产物：generation_trajectory_atlas, cache_closure_report, eos_audit

## TB10：操作组合

- 目标：从单操作结构预测未见组合、顺序和作用域。
- 轴：operation_a, operation_b, order, scope, protocol
- 分区：composition_discovery, unseen_composition_confirmation
- 控制：independent_concat, lookup_table, length_match, commutative_control, invalid_composition
- 指标：composition_error, order_prediction, scope_accuracy, abstention_accuracy, description_length
- 通过：未见组合显著优于独立拼接和查表复杂度基线，并正确拒答无定义组合。
- 失败：关闭操作代数强版本，保留已确认原子局部机制。
- 产物：operation_algebra_table, composition_verdict

## TB11：跨协议与跨模型功能同构

- 目标：寻找保持未来响应和操作交换结构的功能映射，而非物理坐标对齐。
- 轴：source_model, target_model, protocol, event, operation
- 分区：model_pair_discovery, heldout_model_confirmation
- 控制：layer_number_match, cca_baseline, random_event_match, behavior_only_match, reverse_map
- 指标：response_preservation, commuting_error, cross_protocol_retention, heldout_model_loss, map_complexity
- 通过：功能映射在留一模型中保持响应与操作结构，并超过坐标和行为表面基线。
- 失败：降级为模型特定或局部模体，不用层号近似冒充同构。
- 产物：functional_maps, cross_model_verdict, commuting_diagram_audit

## TB12：训练形成

- 目标：分别预测是否形成、何时形成、形成哪种机制，并以训练干预验证。
- 轴：task, vocabulary, architecture, seed, checkpoint, training_intervention
- 分区：formation_discovery, formation_confirmation, heldout_factor
- 控制：design_only_baseline, training_scalar_baseline, seed_permutation, right_censoring, future_blindness
- 指标：formation_auc, onset_survival_loss, mechanism_class_accuracy, incremental_gain, intervention_shift
- 通过：三种目标分账通过强设计基线，并由形成前干预产生预注册方向变化。
- 失败：分类、起点和实现类型分别裁决；右删失不伪造成起点数值。
- 产物：formation_predictor, survival_model, training_causal_audit
