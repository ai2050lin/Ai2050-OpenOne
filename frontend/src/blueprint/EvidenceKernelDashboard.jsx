import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Database,
  ExternalLink,
  FlaskConical,
  Network,
  Target,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

const KERNEL_BASE = '/vis_data/research_kernel';
const ATLAS_MANIFEST = '/vis_data/pattern_family_neuron_atlas/v1/manifest.json';

const MODULES = [
  { id: 'atlas', title: '当前图谱状态', icon: Network, color: '#f59e0b' },
  { id: 'progress', title: '证据覆盖进度', icon: Target, color: '#22d3ee' },
  { id: 'claims', title: '机制主张', icon: CheckCircle2, color: '#34d399' },
  { id: 'runs', title: '可追溯运行', icon: FlaskConical, color: '#60a5fa' },
  { id: 'gaps', title: '开放缺口', icon: AlertTriangle, color: '#fb923c' },
];

const DIMENSION_LABELS = {
  model_coverage: '三模型覆盖',
  color_case_coverage: '颜色案例覆盖',
  real_trace_coverage: '真实 Trace 覆盖',
  real_unit_address_coverage: '真实单元地址覆盖',
  single_unit_causal_coverage: '单神经元因果覆盖',
  heldout_prediction_coverage: '留出预测覆盖',
  clean_closure_coverage: '干净闭合覆盖',
};

const EVIDENCE_STAGES = [
  { label: '自然观测', level: 'L1-L2' },
  { label: '稳定路径', level: 'L3' },
  { label: '组件归因', level: 'L4' },
  { label: '因果必要/充分', level: 'L5-L6' },
  { label: '生成与闭合', level: 'L7-L8' },
];

const formatNumber = (value) => new Intl.NumberFormat('zh-CN').format(Number(value || 0));

async function fetchJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.json();
}

async function fetchJsonl(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return (await response.text())
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function SourceLink({ path, label = '查看来源' }) {
  return (
    <a
      href={path}
      target="_blank"
      rel="noreferrer"
      style={{ color: '#7dd3fc', fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 4, overflowWrap: 'anywhere' }}
    >
      <ExternalLink size={12} />
      {label}
    </a>
  );
}

function ProgressBar({ ratio }) {
  const value = Math.max(0, Math.min(1, Number(ratio || 0)));
  return (
    <div style={{ height: 5, marginTop: 5, background: 'rgba(148,163,184,0.13)', borderRadius: 2, overflow: 'hidden' }}>
      <div style={{ width: `${value * 100}%`, height: '100%', background: value === 1 ? '#34d399' : '#22d3ee', borderRadius: 2 }} />
    </div>
  );
}

function AtlasDetail({ atlas }) {
  const metrics = atlas?.metrics || {};
  const phase393 = atlas?.phase393_audit || {};
  const phase396 = atlas?.phase396_audit || {};
  const phase397 = atlas?.phase397_audit || {};
  const phase398 = atlas?.phase398_audit || {};
  const phase399 = atlas?.phase399_audit || {};
  const phase400 = atlas?.phase400_audit || {};
  const phase401 = atlas?.phase401_audit || {};
  const phase402 = atlas?.phase402_audit || {};
  const phase405 = atlas?.phase405_audit || {};
  const phase406 = atlas?.phase406_audit || {};
  const phase407 = atlas?.phase407_audit || {};
  const phase408 = atlas?.phase408_audit || {};
  const phase409 = atlas?.phase409_audit || {};
  const phase410 = atlas?.phase410_audit || {};
  const phase411 = atlas?.phase411_audit || {};
  const phase412 = atlas?.phase412_audit || {};
  const phase413 = atlas?.phase413_audit || {};
  const phase414 = atlas?.phase414_audit || {};
  const rows = [
    ['模式族物理映射', `${metrics.mapped_family_count || 0}/${metrics.family_count || 0}`],
    ['模型 × 模式族分区', formatNumber(metrics.model_family_partition_count)],
    ['注册机制', formatNumber(metrics.registered_mechanism_count)],
    ['Prompt-model 案例', formatNumber(metrics.prompt_model_case_count)],
    ['全层组件事件', formatNumber(metrics.component_event_count)],
    ['路径签名', formatNumber(metrics.path_signature_count)],
    ['跨模型集合读出候选', formatNumber(metrics.phase330_cross_model_set_readout_specific_mechanism_count)],
    ['Phase 334 局部传播通过', formatNumber(metrics.phase334_local_propagation_pass_count)],
    ['Phase 334 自然必要性候选', formatNumber(metrics.phase334_natural_necessity_candidate_count)],
    ['Phase 391 局部父节点布局', formatNumber(metrics.phase391_physical_local_parent_layout_count)],
    ['Phase 393 属性内容切换', `${metrics.phase393_attribute_answer_switch_count || 0}/${metrics.phase393_attribute_direction_count || 0}`],
    ['Phase 393 深度特异模型', `${metrics.phase393_depth_specificity_pass_model_count || 0}/3`],
    ['Phase 396 字段上下文切换', `${metrics.phase396_same_literal_answer_switch_count || 0}/${metrics.phase396_physical_direction_count || 0}`],
    ['Phase 396 同位置内容切换', `${metrics.phase396_same_position_content_switch_count || 0}/${metrics.phase396_physical_direction_count || 0}`],
    ['Phase 396 字段物理复现模型', `${metrics.phase396_field_context_transport_pass_model_count || 0}/3`],
    ['Phase 395 跨任务共享状态', `${metrics.phase395_crosssurface_shared_state_count || 0}/1`],
    ['Phase 397 多任务完整合格组', `${metrics.phase397_behavior_qualified_group_count || 0}/144`],
    ['Phase 397 行为合格任务面', `${metrics.phase397_eligible_surface_count || 0}/${metrics.phase397_registered_surface_count || 0}`],
    ['Phase 397 三分割关系签名', `${(metrics.phase397_discovery_observational_pass_cell_count || 0) + (metrics.phase397_calibration_observational_pass_cell_count || 0) + (metrics.phase397_physical_observational_pass_cell_count || 0)}/27`],
    ['Phase 397 因果关系载体', `${metrics.phase397_causal_relation_pass_cell_count || 0}/9`],
    ['Phase 397 关系答案切换', `${metrics.phase397_relation_answer_switch_count || 0}/${metrics.phase397_causal_direction_count || 0}`],
    ['Phase 398 联合析因完整组', `${metrics.phase398_qualified_group_count || 0}/72`],
    ['Phase 398 ROQ 物理复现', `${metrics.phase398_ROQ_physical_pass_cell_count || 0}/9`],
    ['Phase 398 单位置因果单元', `${metrics.phase398_causal_pass_cell_count || 0}/9`],
    ['Phase 398 同顺序答案切换', `${metrics.phase398_same_order_answer_switch_count || 0}/${metrics.phase398_causal_direction_count || 0}`],
    ['Phase 399 多位置动态完整组', `${metrics.phase399_qualified_parallel_group_count || 0}/112`],
    ['Phase 399 行为合格任务面', `${metrics.phase399_eligible_surface_count || 0}/${metrics.phase399_registered_surface_count || 0}`],
    ['Phase 399 物理必需事件单元', `${metrics.phase399_required_event_physical_cell_count || 0}/9`],
    ['Phase 399 物理有序链单元', `${metrics.phase399_ordered_chain_physical_cell_count || 0}/9`],
    ['Phase 399 跨模型动态链', `${metrics.phase399_crossmodel_chain_surface_count || 0}/3`],
    ['Phase 399 联合因果干预', formatNumber(metrics.phase399_joint_causal_intervention_count)],
    ['Phase 400 发现集部分序图', `${metrics.phase400_discovery_partial_order_graph_cell_count || 0}/6`],
    ['Phase 400 跨模型发现候选', `${metrics.phase400_discovery_crossmodel_isomorphism_surface_count || 0}/2`],
    ['Phase 400 预测门合格单元', `${metrics.phase400_discovery_prediction_pass_cell_count || 0}/6`],
    ['Phase 400 校准守恒单元', `${metrics.phase400_calibration_quality_group_model_cell_count || 0}/${metrics.phase400_calibration_group_model_cell_count || 0}`],
    ['Phase 400 已使用物理留出', `${metrics.phase400_physical_holdout_case_count || 0}/384`],
    ['Phase 400 新增神经元节点', formatNumber(metrics.phase400_new_neuron_node_count)],
    ['Phase 401 语义正确案例', `${metrics.phase401_behavior_semantic_correct_case_count || 0}/${metrics.phase401_behavior_candidate_case_count || 0}`],
    ['Phase 401 批形状敏感案例', `${metrics.phase401_batch_sensitive_pilot_case_count || 0}/${metrics.phase401_batch_pilot_case_count || 0}`],
    ['Phase 401 同形状账本', `${metrics.phase401_instrument_quality_pass_case_count || 0}/${metrics.phase401_instrument_case_count || 0}`],
    ['Phase 401 严格局部边层', `${metrics.phase401_strict_local_edge_passing_layer_count || 0}/${metrics.phase401_model_surface_layer_count || 0}`],
    ['Phase 401 特异直接物理边', `${metrics.phase401_direct_local_physical_model_surface_count || 0}/${metrics.phase401_model_surface_count || 0}`],
    ['Phase 401 跨模型局部边', `${metrics.phase401_crossmodel_local_edge_surface_count || 0}/2`],
    ['Phase 401 已使用物理留出', `${metrics.phase401_physical_holdout_case_count || 0}/384`],
    ['Phase 401 新增神经元节点', formatNumber(metrics.phase401_new_neuron_node_count)],
    ['Phase 402 语义正确案例', `${metrics.phase402_behavior_semantic_correct_case_count || 0}/${metrics.phase402_behavior_candidate_case_count || 0}`],
    ['Phase 402 行为合格任务面', `${metrics.phase402_eligible_surface_count || 0}/${metrics.phase402_registered_surface_count || 0}`],
    ['Phase 402 多父分区账本', `${metrics.phase402_instrument_pass_row_count || 0}/${metrics.phase402_instrument_row_count || 0}`],
    ['Phase 402 严格局部联合单元', `${metrics.phase402_strict_local_joint_cell_count || 0}/${metrics.phase402_joint_group_layer_subset_count || 0}`],
    ['Phase 402 模型级多父候选', `${metrics.phase402_model_level_joint_parent_candidate_count || 0}/12`],
    ['Phase 402 跨模型多父任务面', `${metrics.phase402_crossmodel_joint_parent_surface_count || 0}/2`],
    ['Phase 402 已使用物理留出', `${metrics.phase402_physical_holdout_case_count || 0}/288`],
    ['Phase 402 新增神经元节点', formatNumber(metrics.phase402_new_neuron_node_count)],
    ['Phase 403 状态操作正确案例', `${metrics.phase403_predictive_state_correct_count || 0}/${metrics.phase403_predictive_state_case_count || 0}`],
    ['Phase 403 跨模型状态族', `${metrics.phase403_crossmodel_state_family_count || 0}/3`],
    ['Phase 404 有限候选正确案例', `${metrics.phase404_finite_candidate_correct_count || 0}/${metrics.phase404_direct_state_case_count || 0}`],
    ['Phase 404 全词表目标首词', `${metrics.phase404_global_top_target_count || 0}/${metrics.phase404_direct_state_case_count || 0}`],
    ['Phase 405 自然未来目标首词', `${metrics.phase405_natural_top_target_count || 0}/${metrics.phase405_natural_future_case_count || 0}`],
    ['Phase 405 严格模型族单元', `${metrics.phase405_model_family_natural_group_pass_count || 0}/9`],
    ['Phase 405 跨模型状态族', `${metrics.phase405_crossmodel_state_family_count || 0}/3`],
    ['Phase 405 新增神经元节点', formatNumber(metrics.phase405_new_neuron_node_count)],
    ['Phase 406 正式发现案例', formatNumber(metrics.phase406_formal_discovery_case_count)],
    ['Phase 406 H12 语义正确', `${metrics.phase406_H12_sequence_semantic_correct_count || 0}/${metrics.phase406_formal_discovery_case_count || 0}`],
    ['Phase 406 正式组通过', `${metrics.phase406_formal_group_pass_count || 0}/72`],
    ['Phase 406 跨模型状态族', `${metrics.phase406_crossmodel_state_family_count || 0}/3`],
    ['Phase 406 新增神经元节点', formatNumber(metrics.phase406_new_neuron_node_count)],
    ['Phase 407 正式发现案例', formatNumber(metrics.phase407_formal_discovery_case_count)],
    ['Phase 407 语义正确', `${metrics.phase407_semantic_correct_count || 0}/${metrics.phase407_formal_discovery_case_count || 0}`],
    ['Phase 407 完整响应', `${metrics.phase407_complete_response_count || 0}/${metrics.phase407_formal_discovery_case_count || 0}`],
    ['Phase 407 四门完整组', `${metrics.phase407_fully_semantic_gated_group_count || 0}/${metrics.phase407_formal_group_count || 0}`],
    ['Phase 407 非有限路径', formatNumber(metrics.phase407_nonfinite_generation_path_count)],
    ['Phase 407 跨模型状态族', `${metrics.phase407_crossmodel_state_family_count || 0}/3`],
    ['Phase 407 新增神经元节点', formatNumber(metrics.phase407_new_neuron_node_count)],
    ['Phase 408 正式发现案例', formatNumber(metrics.phase408_formal_discovery_case_count)],
    ['Phase 408 注册响应', `${metrics.phase408_registered_response_observed_count || 0}/${metrics.phase408_formal_discovery_case_count || 0}`],
    ['Phase 408 允许响应', `${metrics.phase408_allowed_response_observed_count || 0}/${metrics.phase408_formal_discovery_case_count || 0}`],
    ['Phase 408 条件可分组', `${metrics.phase408_condition_separation_pass_group_count || 0}/108`],
    ['Phase 408 表面词汇稳定组', `${metrics.phase408_surface_lexical_stability_pass_group_count || 0}/108`],
    ['Phase 408 功能分区组', `${metrics.phase408_functional_group_pass_count || 0}/108`],
    ['Phase 408 发现跨模型族', `${metrics.phase408_discovery_crossmodel_partition_family_count || 0}/3`],
    ['Phase 408 校准跨模型族', `${metrics.phase408_calibration_crossmodel_partition_family_count || 0}/3`],
    ['Phase 408 行为留出跨模型族', `${metrics.phase408_behavioral_crossmodel_partition_family_count || 0}/3`],
    ['Phase 408 新增神经元节点', formatNumber(metrics.phase408_new_neuron_node_count)],
    ['Phase 409 协议注册案例', formatNumber(metrics.phase409_registered_abstract_case_count)],
    ['Phase 409 未来提示哈希', formatNumber(metrics.phase409_future_model_prompt_hash_count)],
    ['Phase 409 双规则一致', `${metrics.phase409_dual_rule_agreement_count || 0}/${metrics.phase409_rule_engine_scenario_count || 0}`],
    ['Phase 409 外部规则复核', `${metrics.phase409_external_rule_review_count || 0}/1`],
    ['Phase 409 采集器等价', `${metrics.phase409_collector_equivalence_count || 0}/1`],
    ['Phase 409 模型案例', formatNumber(metrics.phase409_model_case_count)],
    ['Phase 409 新增神经元节点', formatNumber(metrics.phase409_new_neuron_node_count)],
    ['Phase 410 正交状态轴', `${metrics.phase410_orthogonal_axis_count || 0}/6`],
    ['Phase 410 h3 顺序审计', `${(metrics.phase410_h3_order_variant_count || 0) - (metrics.phase410_h3_order_symmetry_failure_count || 0)}/${metrics.phase410_h3_order_variant_count || 0}`],
    ['Phase 410 语法有限全集', `${(metrics.phase410_grammar_finite_case_count || 0) - (metrics.phase410_grammar_failure_count || 0)}/${metrics.phase410_grammar_finite_case_count || 0}`],
    ['Phase 410 外部审阅者', `${metrics.phase410_completed_external_reviewer_count || 0}/${metrics.phase410_required_external_reviewer_count || 0}`],
    ['Phase 410 密封采集器等价', `${metrics.phase410_sealed_model_collector_case_count || 0}/165`],
    ['Phase 410 模型案例', formatNumber(metrics.phase410_model_case_count)],
    ['Phase 410 新增神经元节点', formatNumber(metrics.phase410_new_neuron_node_count)],
    ['Phase 411 有限语义合同', `${(metrics.phase411_finite_semantic_case_count || 0) - (metrics.phase411_finite_semantic_failure_count || 0)}/${metrics.phase411_finite_semantic_case_count || 0}`],
    ['Phase 411 有限语义新增解析', formatNumber(metrics.phase411_semantic_only_resolved_case_count)],
    ['Phase 411 注册状态操作', `${metrics.phase411_registered_operation_count || 0}/46`],
    ['Phase 411 操作组合闭包', `${(metrics.phase411_operation_composition_case_count || 0) - (metrics.phase411_operation_composition_failure_count || 0)}/${metrics.phase411_operation_composition_case_count || 0}`],
    ['Phase 411 历史规则协变', `${(metrics.phase411_history_covariance_case_count || 0) - (metrics.phase411_history_covariance_failure_count || 0)}/${metrics.phase411_history_covariance_case_count || 0}`],
    ['Phase 411 粗分区不稳定操作单元', formatNumber(metrics.phase411_coarse_unstable_operation_cell_count)],
    ['Phase 411 外部审阅者', `${metrics.phase411_completed_external_reviewer_count || 0}/${metrics.phase411_required_external_reviewer_count || 0}`],
    ['Phase 411 双人接受条目', `${metrics.phase411_review_accepted_item_count || 0}/65`],
    ['Phase 411 密封采集器等价', `${metrics.phase411_sealed_model_collector_case_count || 0}/165`],
    ['Phase 411 模型案例', formatNumber(metrics.phase411_model_case_count)],
    ['Phase 411 新增神经元节点', formatNumber(metrics.phase411_new_neuron_node_count)],
    ['Phase 412 类型化观察者协变', `${(metrics.phase412_observer_operation_cell_count || 0) - (metrics.phase412_typed_observer_unstable_cell_count || 0)}/${metrics.phase412_observer_operation_cell_count || 0}`],
    ['Phase 412 固定角色失败已解释', `${metrics.phase412_role_transport_explained_cell_count || 0}/${metrics.phase412_fixed_observer_unstable_cell_count || 0}`],
    ['Phase 412 观察者作用组合', `${(metrics.phase412_observer_action_composition_case_count || 0) - (metrics.phase412_observer_action_composition_failure_count || 0)}/${metrics.phase412_observer_action_composition_case_count || 0}`],
    ['Phase 412 有限分区穷举', `${metrics.phase412_finite_partition_count || 0}/${metrics.phase412_finite_partition_count || 0}`],
    ['Phase 412 全局非平凡商', `${metrics.phase412_global_qualifying_nontrivial_partition_count || 0}/${metrics.phase412_nontrivial_partition_count || 0}`],
    ['Phase 412 外部角色索引分区束', formatNumber(metrics.phase412_role_indexed_partition_bundle_count)],
    ['Phase 412 已注册不可逆操作', `${metrics.phase412_registered_irreversible_operation_count || 0}/7`],
    ['Phase 412 已注册跨族桥', `${metrics.phase412_registered_cross_family_bridge_count || 0}/4`],
    ['Phase 412 外部审阅者', `${metrics.phase412_completed_external_reviewer_count || 0}/${metrics.phase412_required_external_reviewer_count || 0}`],
    ['Phase 412 密封采集器等价', `${metrics.phase412_sealed_model_collector_case_count || 0}/165`],
    ['Phase 412 模型案例', formatNumber(metrics.phase412_model_case_count)],
    ['Phase 412 新增神经元节点', formatNumber(metrics.phase412_new_neuron_node_count)],
    ['Phase 413 材料主张审计', `${metrics.phase413_source_claim_count || 0}/${metrics.phase413_source_claim_count || 0}`],
    ['Phase 413 终端相同有限轨迹', `${metrics.phase413_same_terminal_path_count || 0}/${metrics.phase413_synthetic_path_count || 0}`],
    ['Phase 413 中间不同轨迹对', `${metrics.phase413_internal_distinct_path_pair_count || 0}/${metrics.phase413_synthetic_path_pair_count || 0}`],
    ['Phase 413 一步相同但未来不同', `${metrics.phase413_future_different_pair_count || 0}/${metrics.phase413_future_state_pair_count || 0}`],
    ['Phase 413 通道置换原生输出不变', `${metrics.phase413_native_output_invariant_channel_case_count || 0}/${metrics.phase413_channel_permutation_case_count || 0}`],
    ['Phase 413 固定通道读数反例', `${metrics.phase413_fixed_coordinate_probe_failure_count || 0}/${metrics.phase413_channel_permutation_case_count || 0}`],
    ['Phase 413 候选面板合同', `${metrics.phase413_candidate_panel_case_count || 0}/${metrics.phase413_candidate_panel_case_count || 0}`],
    ['Phase 413 合格层内局部概率读出', `${metrics.phase413_qualified_direct_layer_local_readout_count || 0}/${metrics.phase413_direct_layer_local_readout_count || 0}`],
    ['Phase 413 外部审阅者', `${metrics.phase413_completed_external_reviewer_count || 0}/${metrics.phase413_required_external_reviewer_count || 0}`],
    ['Phase 413 密封采集器等价', `${metrics.phase413_sealed_model_collector_case_count || 0}/165`],
    ['Phase 413 模型案例', formatNumber(metrics.phase413_model_case_count)],
    ['Phase 413 新增神经元节点', formatNumber(metrics.phase413_new_neuron_node_count)],
    ['Phase 414 混合证据目录分类', `${metrics.phase414_catalog_item_count || 0}/${metrics.phase414_catalog_item_count || 0}`],
    ['Phase 414 目录严格机制闭合', `${metrics.phase414_catalog_mechanism_closed_count || 0}/${metrics.phase414_catalog_item_count || 0}`],
    ['Phase 414 完整状态续跑恒等', `${metrics.phase414_natural_replay_exact_count || 0}/${metrics.phase414_natural_replay_cell_count || 0}`],
    ['Phase 414 层间终端核变化', `${metrics.phase414_layerwise_terminal_kernel_variation_count || 0}/${metrics.phase414_natural_replay_case_count || 0}`],
    ['Phase 414 不完整状态反例', `${metrics.phase414_incomplete_state_counterexample_count || 0}/${metrics.phase414_natural_replay_cell_count || 0}`],
    ['Phase 414 观察者索引轨迹', `${metrics.phase414_varying_observer_trajectory_count || 0}/${metrics.phase414_observer_trajectory_count || 0}`],
    ['Phase 414 可变长度语义事件', `${metrics.phase414_candidate_event_count || 0}/${metrics.phase414_candidate_event_count || 0}`],
    ['Phase 414 跨 tokenizer 语义对齐', `${metrics.phase414_cross_tokenizer_semantic_alignment_count || 0}/${metrics.phase414_cross_tokenizer_semantic_event_count || 0}`],
    ['Phase 414 合格观察者', `${metrics.phase414_qualified_observer_count || 0}/${metrics.phase414_observer_method_count || 0}`],
    ['Phase 414 外部审阅者', `${metrics.phase414_completed_external_reviewer_count || 0}/${metrics.phase414_required_external_reviewer_count || 0}`],
    ['Phase 414 密封采集器等价', `${metrics.phase414_sealed_model_collector_case_count || 0}/165`],
    ['Phase 414 模型案例', formatNumber(metrics.phase414_model_case_count)],
    ['Phase 414 新增神经元节点', formatNumber(metrics.phase414_new_neuron_node_count)],
    ['跨模型行为必要性', formatNumber(metrics.phase330_cross_model_behavior_necessity_mechanism_count)],
    ['单神经元因果', formatNumber(metrics.single_unit_causal_count)],
    ['完整自然链', formatNumber(metrics.full_natural_chain_pass_count)],
  ];

  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.7 }}>
        {phase414.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。Phase414 将完整自然状态续跑校准为仪器恒等门：${metrics.phase414_natural_replay_exact_count || 0}/${metrics.phase414_natural_replay_cell_count || 0} 个有限续跑单元复现同一终端核，层间终端核变化为 ${metrics.phase414_layerwise_terminal_kernel_variation_count || 0}/${metrics.phase414_natural_replay_case_count || 0}；删去角色、历史或缓存后出现 ${metrics.phase414_incomplete_state_counterexample_count || 0}/${metrics.phase414_natural_replay_cell_count || 0} 个反例。观察者索引轨迹变化 ${metrics.phase414_varying_observer_trajectory_count || 0}/${metrics.phase414_observer_trajectory_count || 0} 只是合成可读性审计，合格观察者仍为 ${metrics.phase414_qualified_observer_count || 0}/${metrics.phase414_observer_method_count || 0}。96 项目录是混合证据账本，不是完成百分比分母；模型、物理、因果和神经元证据均未增加。`
          : phase413.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。Phase413 明确区分终端原生预测核和中间诊断读出：${metrics.phase413_same_terminal_path_count || 0}/${metrics.phase413_synthetic_path_count || 0} 条有限轨迹共享同一终端分布，但 ${metrics.phase413_internal_distinct_path_pair_count || 0}/${metrics.phase413_synthetic_path_pair_count || 0} 条轨迹对拥有不同中间过程；${metrics.phase413_future_different_pair_count || 0}/${metrics.phase413_future_state_pair_count || 0} 个状态对在一步分布相同而未来不同。通道置换保持原生输出 ${metrics.phase413_native_output_invariant_channel_case_count || 0}/${metrics.phase413_channel_permutation_case_count || 0}，同时给出固定通道读数反例 ${metrics.phase413_fixed_coordinate_probe_failure_count || 0}/${metrics.phase413_channel_permutation_case_count || 0}。合格层内局部概率读出仍为 ${metrics.phase413_qualified_direct_layer_local_readout_count || 0}/${metrics.phase413_direct_layer_local_readout_count || 0}，模型、物理和神经元证据均未增加。`
          : phase412.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。Phase412 复现了 ${metrics.phase412_fixed_observer_unstable_cell_count || 0} 个固定查询角色失败，并确认其中 ${metrics.phase412_role_transport_explained_cell_count || 0} 个都来自实体重命名时未同步运输查询角色。状态、查询角色和响应类联合变换后，类型化协变通过 ${(metrics.phase412_observer_operation_cell_count || 0) - (metrics.phase412_typed_observer_unstable_cell_count || 0)}/${metrics.phase412_observer_operation_cell_count || 0}；${metrics.phase412_finite_partition_count || 0} 个有限分区已穷举，全局合格非平凡商为 ${metrics.phase412_global_qualifying_nontrivial_partition_count || 0}/${metrics.phase412_nontrivial_partition_count || 0}。知识族的 ${metrics.phase412_role_indexed_partition_bundle_count || 0} 个外部角色索引分区束不是模型状态证据。外部审阅者和密封采集器仍分别为 ${metrics.phase412_completed_external_reviewer_count || 0}/${metrics.phase412_required_external_reviewer_count || 0}、${metrics.phase412_sealed_model_collector_case_count || 0}/165，所以不显示新的模型行为、物理路径或神经元。`
          : phase411.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。Phase411 在 ${formatNumber(metrics.phase411_finite_semantic_case_count)} 个冻结模板案例上完成严格精确通道与有限注册语义通道审计，并验证 ${metrics.phase411_registered_operation_count || 0} 个外部状态操作、${metrics.phase411_operation_composition_case_count || 0} 个组合和 ${metrics.phase411_history_covariance_case_count || 0} 个历史协变案例。固定查询角色下有 ${metrics.phase411_coarse_unstable_operation_cell_count || 0} 个粗分区操作单元不保持；该结果仍需区分状态变换和查询角色运输。外部审阅者仍为 ${metrics.phase411_completed_external_reviewer_count || 0}/${metrics.phase411_required_external_reviewer_count || 0}，密封采集器为 ${metrics.phase411_sealed_model_collector_case_count || 0}/165，所以不显示新的模型行为、物理路径或神经元。`
          : phase410.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。Phase410 已把动态响应拆成 ${metrics.phase410_orthogonal_axis_count || 0} 个正交轴，并完成 ${(metrics.phase410_h3_order_variant_count || 0) - (metrics.phase410_h3_order_symmetry_failure_count || 0)}/${metrics.phase410_h3_order_variant_count || 0} 个 h3 顺序变体和 ${(metrics.phase410_grammar_finite_case_count || 0) - (metrics.phase410_grammar_failure_count || 0)}/${metrics.phase410_grammar_finite_case_count || 0} 个语法有限全集案例的机器审计。外部审阅者为 ${metrics.phase410_completed_external_reviewer_count || 0}/${metrics.phase410_required_external_reviewer_count || 0}，密封真实模型采集器等价为 ${metrics.phase410_sealed_model_collector_case_count || 0}/165，因此不显示新的行为、物理路径或神经元。`
          : phase409.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。Phase409 已冻结 ${formatNumber(metrics.phase409_registered_abstract_case_count)} 个动态响应与历史条件抽象案例、${formatNumber(metrics.phase409_future_model_prompt_hash_count)} 个未来三模型提示哈希；双机器规则引擎一致 ${metrics.phase409_dual_rule_agreement_count || 0}/${metrics.phase409_rule_engine_scenario_count || 0}。外部规则复核和增量采集器逐词元等价仍为 0/1，模型案例为 0，因此这里只显示协议节点，不显示新的物理路径或神经元。`
          : phase408.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。Phase408 在 ${formatNumber(metrics.phase408_formal_discovery_case_count)} 个正式案例中，把状态可分性、标签对齐、数值有效性、生成事件和右删失分别记录；${metrics.phase408_functional_group_pass_count || 0}/108 个组形成冻结功能分区，发现、校准、行为留出的跨模型候选分别为 ${metrics.phase408_discovery_crossmodel_partition_family_count || 0}/3、${metrics.phase408_calibration_crossmodel_partition_family_count || 0}/3、${metrics.phase408_behavioral_crossmodel_partition_family_count || 0}/3。历史仍固定为空，因此不授权物理路径或神经元节点。`
          : phase407.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。Phase407 在 5,760 个正式发现案例中分开记录语义完成、句界和模型停止：3,935 例达到注册语义，3,933 例形成完整响应；只有 10/108 个组同时通过表面、接口、历史和序列门，九个模型×语言族单元均未达门，跨模型状态族为 0/3。129 条 GLM4 非有限路径单列为运行时警告，所有物理和神经元门保持关闭。`
          : phase406.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。条件化 H12 短序列在 5,760 个正式发现案例中恢复 3,512 例语义答案，但 72 个正式组仅通过 5 个，且未达到任何模型族门；跨模型条件状态族为 0/3。宽松词汇上界与 H48 诊断均未改变停止决定，所有下游物理和神经元门保持关闭。`
          : phase405.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。Phase403-405 依次分离了更新执行、有限候选读出和自然未来首词；最终九个模型×语言族单元的严格自然组通过数均为零，跨模型预测状态族为 0/3。校准、行为留出和物理留出均未使用，3D 不新增层、头、通道或神经元节点。`
          : phase402.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。四类互斥注意力 K/V 父分区账本通过 3,328/3,328；13,728 个联合组层子集中只有 8 个早层局部单元通过全部门，模型级和跨模型候选均为零。校准与物理留出未使用，3D 不新增神经元、头或通道节点。`
          : phase401.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。同执行形状组件账本通过 96/96；独立批敏感性审计有 7/192 个案例发生字段变化。真实关系替换虽然产生局部响应，但严格与敏感性审计均为 0/208 个合格层、0/6 个特异直接物理边。校准和物理留出均未使用，3D 不新增神经元、头或通道节点。`
          : phase400.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。区间部分序发现图通过 5/6 个模型任务单元，拥有关系出现 1/2 个发现集跨模型候选；但答案预测门为 0/6，校准守恒合同仅通过 23/24 个组模型单元。物理留出保持 0/384，3D 不新增神经元、头或通道节点。`
          : phase399.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。来源到查询、查询整合和终端三类聚合事件在 9/9 个模型任务单元及三份独立数据中复现；但冻结峰值顺序只在 DS7B 角色填槽中三次通过，跨模型任务为 0/3。3D 蓝色事件是模型特异聚合观测，不是注意力头、神经元或因果绑定路径。`
          : phase398.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。查询末端的 ROQ 联合轨迹在三任务、三模型和三份独立数据中均为 9/9，并跨词汇方向复现；但单查询位置搬运通过 0/9 个因果单元，只产生 10/432 次答案切换。3D 粉色节点是顺序条件化聚合轨迹，不是神经元、可移植状态或完整绑定算法。`
          : phase397.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。同位置值状态中的关系签名在三任务、三模型和三份独立数据上复现，但隔离搬运通过 0/9 个因果单元、产生 0/144 次答案切换。3D 蓝色节点是聚合观测签名，不是注意力头、MLP 神经元或可移植绑定规则。`
          : phase396.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。字段抽取中的同字面值上下文状态已在三模型独立物理组复现，但跨任务共享门失败；3D 新增的是聚合词元状态锚点，不是注意力头或 MLP 神经元。抽象绑定算法、自然必要性、完整语言路径与单神经元因果仍为零。`
          : phase393.status
          ? `当前图谱已推进到 Phase ${atlas?.phase || '-'}。局部父节点布局已跨模型物理复现，受控属性内容搬运也已通过三模型独立留出；但错误深度同样有效，所以不能把它标成自然深度路径。完整语言路径与神经元因果仍为零。`
          : `当前图谱已经完成九个模式族和三个模型的统一物理覆盖，并继续推进到 Phase ${atlas?.phase || '-'} 的自然必要性审计。严格边界仍然是：局部传播已经出现，但跨模型行为必要性、单神经元因果和完整自然闭合均未通过。`}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 8 }}>
        {rows.map(([label, value]) => (
          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, padding: '8px 10px', borderBottom: '1px solid rgba(148,163,184,0.12)' }}>
            <span style={{ color: '#94a3b8', fontSize: 11 }}>{label}</span>
            <strong style={{ color: value === '0' ? '#fda4af' : '#e2e8f0', fontSize: 11, fontFamily: 'monospace' }}>{value}</strong>
          </div>
        ))}
      </div>
      <div style={{ color: '#fbbf24', fontSize: 11, lineHeight: 1.65 }}>
        {atlas?.evidence_boundary?.statement || atlas?.evidence_boundary || ''}
      </div>
      <SourceLink path={ATLAS_MANIFEST} label="最新物理图谱 manifest" />
    </div>
  );
}

function ProgressDetail({ progress }) {
  const dimensions = Object.entries(progress?.dimensions || {});
  return (
    <div style={{ display: 'grid', gap: 10 }}>
      {dimensions.map(([id, row]) => {
        const ratio = Math.max(0, Math.min(1, Number(row.ratio || 0)));
        return (
          <div key={id}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: 12 }}>
              <span style={{ color: '#dbeafe' }}>{DIMENSION_LABELS[id] || id}</span>
              <span style={{ color: '#94a3b8' }}>{row.valid}/{row.required} · {(ratio * 100).toFixed(1)}%</span>
            </div>
            <ProgressBar ratio={ratio} />
          </div>
        );
      })}
      <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.6 }}>{progress?.boundary}</div>
      <SourceLink path={`${KERNEL_BASE}/progress.json`} label="progress.json" />
    </div>
  );
}

function ClaimsDetail({ claims }) {
  return (
    <div style={{ display: 'grid', gap: 10 }}>
      {claims.map((claim) => (
        <div key={claim.claim_id} style={{ paddingBottom: 10, borderBottom: '1px solid rgba(148,163,184,0.12)' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7, alignItems: 'center' }}>
            <strong style={{ color: '#e2e8f0', fontSize: 12 }}>{claim.claim_id}</strong>
            <span style={{ color: '#34d399', fontSize: 10 }}>{claim.evidence_level}</span>
            <span style={{ color: '#fbbf24', fontSize: 10 }}>{claim.status}</span>
          </div>
          <div style={{ color: '#cbd5e1', fontSize: 11, lineHeight: 1.6, marginTop: 5 }}>{claim.claim_text}</div>
          <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 5 }}>范围：{JSON.stringify(claim.scope)}</div>
          {claim.negative_evidence?.map((item) => (
            <div key={item} style={{ color: '#fda4af', fontSize: 10, marginTop: 4 }}>限制：{item}</div>
          ))}
          <div style={{ color: '#7dd3fc', fontSize: 10, marginTop: 5 }}>下一测试：{claim.next_test}</div>
        </div>
      ))}
      <SourceLink path={`${KERNEL_BASE}/claims.jsonl`} label="claims.jsonl" />
    </div>
  );
}

function RunsDetail({ runs }) {
  return (
    <div style={{ display: 'grid', gap: 9 }}>
      {runs.map((run) => (
        <div key={run.run_id} style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, paddingBottom: 9, borderBottom: '1px solid rgba(148,163,184,0.12)' }}>
          <div style={{ minWidth: 0 }}>
            <div style={{ color: '#dbeafe', fontSize: 12, overflowWrap: 'anywhere' }}>{run.run_id}</div>
            <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 3 }}>
              {run.model} · Phase {run.phase} · {run.evidence_level} · case {run.case_count} · unit {run.unit_count} · event {run.trace_event_count}
            </div>
          </div>
          <SourceLink path={`${KERNEL_BASE}/${run.manifest_path}`} label="manifest" />
        </div>
      ))}
      <SourceLink path={`${KERNEL_BASE}/manifest.json`} label="总 manifest" />
    </div>
  );
}

function GapsDetail({ gaps }) {
  return (
    <div style={{ display: 'grid', gap: 9 }}>
      {gaps.map((gap) => (
        <div key={gap.gap_id} style={{ paddingBottom: 9, borderBottom: '1px solid rgba(148,163,184,0.12)' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
            <strong style={{ color: '#fef3c7', fontSize: 12 }}>{gap.title}</strong>
            <span style={{ color: gap.status === 'open' ? '#f59e0b' : '#34d399', fontSize: 10 }}>{gap.priority} · {gap.status}</span>
          </div>
          <div style={{ color: '#7dd3fc', fontSize: 10, marginTop: 5 }}>下一测试：{gap.next_test}</div>
          {gap.filled_by?.length > 0 && <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 4 }}>来源：{gap.filled_by.join(', ')}</div>}
        </div>
      ))}
      <SourceLink path={`${KERNEL_BASE}/gaps.jsonl`} label="gaps.jsonl" />
    </div>
  );
}

export function EvidenceKernelDashboard() {
  const [manifest, setManifest] = useState(null);
  const [progress, setProgress] = useState(null);
  const [atlas, setAtlas] = useState(null);
  const [claims, setClaims] = useState([]);
  const [gaps, setGaps] = useState([]);
  const [expanded, setExpanded] = useState(false);
  const [active, setActive] = useState('atlas');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    Promise.allSettled([
      fetchJson(`${KERNEL_BASE}/manifest.json`),
      fetchJson(`${KERNEL_BASE}/progress.json`),
      fetchJsonl(`${KERNEL_BASE}/claims.jsonl`),
      fetchJsonl(`${KERNEL_BASE}/gaps.jsonl`),
      fetchJson(ATLAS_MANIFEST),
    ]).then((results) => {
      if (!mounted) return;
      const [manifestResult, progressResult, claimsResult, gapsResult, atlasResult] = results;
      if (manifestResult.status === 'fulfilled') setManifest(manifestResult.value);
      if (progressResult.status === 'fulfilled') setProgress(progressResult.value);
      if (claimsResult.status === 'fulfilled') setClaims(claimsResult.value);
      if (gapsResult.status === 'fulfilled') setGaps(gapsResult.value);
      if (atlasResult.status === 'fulfilled') setAtlas(atlasResult.value);
      const failures = results.filter((result) => result.status === 'rejected');
      setError(failures.length ? `${failures.length} 个证据数据源读取失败，以下状态为部分结果。` : '');
      setLoading(false);
    });
    return () => { mounted = false; };
  }, []);

  const metrics = atlas?.metrics || {};
  const state = (() => {
    if (loading) return { label: '正在读取证据', detail: '正在同步研究内核和最新物理图谱。', color: '#60a5fa' };
    if (!atlas && !manifest) return { label: '证据源不可用', detail: '无法读取统一证据内核，请检查发布数据。', color: '#fb7185' };
    if (metrics.full_natural_chain_pass_count > 0 && metrics.single_unit_causal_count > 0) {
      return { label: '存在严格闭合机制', detail: '至少一条路径同时通过单元因果和完整自然链。', color: '#34d399' };
    }
    if ((metrics.phase414_catalog_item_count || 0) > 0) {
      return {
        label: '完整状态续跑已校准为恒等门，观察者仍未合格',
        detail: `Phase414 的完整自然状态续跑恒等为 ${metrics.phase414_natural_replay_exact_count || 0}/${metrics.phase414_natural_replay_cell_count || 0}，层间终端核变化 ${metrics.phase414_layerwise_terminal_kernel_variation_count || 0}/${metrics.phase414_natural_replay_case_count || 0}；这不是候选逐层收缩曲线。不完整状态反例为 ${metrics.phase414_incomplete_state_counterexample_count || 0}/${metrics.phase414_natural_replay_cell_count || 0}，合格观察者为 ${metrics.phase414_qualified_observer_count || 0}/${metrics.phase414_observer_method_count || 0}。外部审阅与真实采集器门仍关闭。`,
        color: '#f59e0b',
      };
    }
    if ((metrics.phase413_source_claim_count || 0) > 0) {
      return {
        label: '终端预测核边界已冻结，中间局部概率读出仍为空',
        detail: `Phase413 构造 ${metrics.phase413_same_terminal_path_count || 0}/${metrics.phase413_synthetic_path_count || 0} 条终端相同轨迹和 ${metrics.phase413_internal_distinct_path_pair_count || 0}/${metrics.phase413_synthetic_path_pair_count || 0} 条中间不同轨迹对；通道置换原生输出不变 ${metrics.phase413_native_output_invariant_channel_case_count || 0}/${metrics.phase413_channel_permutation_case_count || 0}，固定通道读数反例 ${metrics.phase413_fixed_coordinate_probe_failure_count || 0}/${metrics.phase413_channel_permutation_case_count || 0}。合格层内局部概率读出为 ${metrics.phase413_qualified_direct_layer_local_readout_count || 0}/${metrics.phase413_direct_layer_local_readout_count || 0}，外部审阅与真实采集器门仍关闭。`,
        color: '#f59e0b',
      };
    }
    if ((metrics.phase412_finite_partition_count || 0) > 0) {
      return {
        label: '类型化协变通过，全局非平凡状态商仍为空',
        detail: `Phase412 将 Phase411 的 ${metrics.phase412_fixed_observer_unstable_cell_count || 0} 个固定角色失败全部校准为查询角色运输问题；类型化协变为 ${(metrics.phase412_observer_operation_cell_count || 0) - (metrics.phase412_typed_observer_unstable_cell_count || 0)}/${metrics.phase412_observer_operation_cell_count || 0}。有限分区穷举 ${metrics.phase412_finite_partition_count || 0}/${metrics.phase412_finite_partition_count || 0}，全局合格非平凡商 ${metrics.phase412_global_qualifying_nontrivial_partition_count || 0}/${metrics.phase412_nontrivial_partition_count || 0}。外部审阅和真实采集器门仍关闭。`,
        color: '#f59e0b',
      };
    }
    if ((metrics.phase411_finite_semantic_case_count || 0) > 0) {
      return {
        label: '有限协议闭包通过，模型语义与物理执行仍关闭',
        detail: `Phase411 的有限语义合同通过 ${(metrics.phase411_finite_semantic_case_count || 0) - (metrics.phase411_finite_semantic_failure_count || 0)}/${metrics.phase411_finite_semantic_case_count || 0}，操作组合通过 ${(metrics.phase411_operation_composition_case_count || 0) - (metrics.phase411_operation_composition_failure_count || 0)}/${metrics.phase411_operation_composition_case_count || 0}。固定查询角色下有 ${metrics.phase411_coarse_unstable_operation_cell_count || 0} 个粗分区不稳定单元；该结果不能在未审计查询角色运输时直接升级为功能状态否定。外部审阅者 ${metrics.phase411_completed_external_reviewer_count || 0}/${metrics.phase411_required_external_reviewer_count || 0}、密封采集器 ${metrics.phase411_sealed_model_collector_case_count || 0}/165，模型和物理案例均为零。`,
        color: '#f59e0b',
      };
    }
    if ((metrics.phase410_orthogonal_axis_count || 0) > 0) {
      return {
        label: '正交动态预检通过，科学执行门仍关闭',
        detail: `Phase410 的六轴状态合同、h3 顺序镜像和语法有限全集机器审计已通过；外部审阅者为 ${metrics.phase410_completed_external_reviewer_count || 0}/${metrics.phase410_required_external_reviewer_count || 0}，密封模型采集器等价为 ${metrics.phase410_sealed_model_collector_case_count || 0}/165。模型、物理、因果和神经元案例均为零。`,
        color: '#f59e0b',
      };
    }
    if ((metrics.phase409_registered_abstract_case_count || 0) > 0) {
      return {
        label: '动态历史协议已冻结，模型执行尚未授权',
        detail: `Phase409 注册 ${metrics.phase409_registered_abstract_case_count || 0} 个抽象案例和 ${metrics.phase409_future_model_prompt_hash_count || 0} 个未来提示哈希；机器双规则一致 ${metrics.phase409_dual_rule_agreement_count || 0}/${metrics.phase409_rule_engine_scenario_count || 0}，但外部规则复核和采集器逐词元等价均未完成。模型、物理、因果和神经元案例均为零。`,
        color: '#f59e0b',
      };
    }
    if ((metrics.phase408_formal_discovery_case_count || 0) > 0) {
      return {
        label: '排他响应分区已审计，内部机制尚未授权',
        detail: `Phase408 完成 ${metrics.phase408_formal_discovery_case_count || 0} 个发现案例；功能分区组为 ${metrics.phase408_functional_group_pass_count || 0}/108，行为留出跨模型族为 ${metrics.phase408_behavioral_crossmodel_partition_family_count || 0}/3。响应分区仍是观测对象，历史、物理、因果和神经元门未越级开放。`,
        color: '#f59e0b',
      };
    }
    if ((metrics.phase407_formal_discovery_case_count || 0) > 0) {
      return { label: '事件账本完整，跨模型状态未成立', detail: 'Phase407 的注册语义为 3,935/5,760，四门完整组为 10/108，但九个模型族单元均未达门、跨模型为 0/3；校准、物理、因果和神经元门保持关闭。', color: '#f59e0b' };
    }
    if ((metrics.phase406_formal_discovery_case_count || 0) > 0) {
      return { label: '短序列恢复存在，跨模型状态未成立', detail: 'Phase406 的 H12 序列语义正确为 3,512/5,760，但正式组仅 5/72、跨模型为 0/3；校准、物理、因果和神经元门保持关闭。', color: '#f59e0b' };
    }
    if ((metrics.phase405_natural_future_case_count || 0) > 0) {
      return { label: '响应丰富但预测状态尚未成立', detail: 'Phase403-405 已完成三种功能状态协议；九个严格模型族单元均未通过，跨模型状态、物理路径和神经元门保持关闭。', color: '#f59e0b' };
    }
    if ((metrics.phase402_behavior_candidate_case_count || 0) > 0) {
      return { label: '多父局部迹象未形成模型级机制', detail: '分区账本为 3,328/3,328；严格局部联合单元为 8/13,728，但模型级与跨模型候选均为零，所有下游门保持关闭。', color: '#f59e0b' };
    }
    if ((metrics.phase401_behavior_candidate_case_count || 0) > 0) {
      return { label: '局部响应未形成特异功能边', detail: '执行与语义合同已修复，账本为 96/96；局部边为 0/208 层，校准、物理留出、因果与神经元门均关闭。', color: '#f59e0b' };
    }
    if ((metrics.phase400_behavior_candidate_case_count || 0) > 0) {
      return { label: '部分序候选未通过验证门', detail: '发现集出现聚合区间图，但预测为 0/6，校准合同为 23/24；物理留出、因果干预与神经元扫描均未开放。', color: '#f59e0b' };
    }
    if ((metrics.phase393_attribute_transport_pass_model_count || 0) === 3) {
      return { label: '属性内容可搬运，深度路径未定位', detail: '三模型独立留出确认受控属性内容搬运；错误深度同样有效，尚无完整路径或神经元级闭合。', color: '#f59e0b' };
    }
    if (metrics.phase334_local_propagation_pass_count > 0) {
      return { label: '局部传播已出现，机制未闭合', detail: '图谱覆盖完整，已有局部传播证据；自然必要性、单神经元因果和完整生成链仍未通过。', color: '#f59e0b' };
    }
    return { label: '物理图谱已建立，等待因果升级', detail: '已有观测和组件候选，但尚未形成严格因果闭合。', color: '#22d3ee' };
  })();

  const summaries = useMemo(() => ({
    atlas: `${metrics.mapped_family_count || 0}/${metrics.family_count || 0} 模式族 · Phase ${atlas?.phase || '-'}`,
    progress: `${Object.keys(progress?.dimensions || {}).length} 个独立分母`,
    claims: `${claims.length} 条主张`,
    runs: `${manifest?.runs?.filter((run) => run.status === 'complete').length || 0}/${manifest?.runs?.length || 0} 完成`,
    gaps: `${gaps.filter((gap) => gap.status === 'open').length} 个开放缺口`,
  }), [atlas?.phase, claims.length, gaps, manifest?.runs, metrics.family_count, metrics.mapped_family_count, progress?.dimensions]);

  const summaryMetrics = [
    ['模式族映射', `${metrics.mapped_family_count || 0}/${metrics.family_count || 0}`],
    ['注册机制', formatNumber(metrics.registered_mechanism_count)],
    ['物理组件事件', formatNumber(metrics.component_event_count)],
    ['Phase401 同形状账本', `${metrics.phase401_instrument_quality_pass_case_count || 0}/${metrics.phase401_instrument_case_count || 0}`],
    ['Phase401 局部边层', `${metrics.phase401_strict_local_edge_passing_layer_count || 0}/${metrics.phase401_model_surface_layer_count || 0}`],
    ['Phase402 多父分区账本', `${metrics.phase402_instrument_pass_row_count || 0}/${metrics.phase402_instrument_row_count || 0}`],
    ['Phase402 严格局部联合单元', `${metrics.phase402_strict_local_joint_cell_count || 0}/${metrics.phase402_joint_group_layer_subset_count || 0}`],
    ['Phase404 候选/自然首词', `${metrics.phase404_finite_candidate_correct_count || 0}/${metrics.phase404_global_top_target_count || 0}`],
    ['Phase405 自然目标首词', `${metrics.phase405_natural_top_target_count || 0}/${metrics.phase405_natural_future_case_count || 0}`],
    ['Phase405 跨模型状态族', `${metrics.phase405_crossmodel_state_family_count || 0}/3`],
    ['Phase406 H12 语义正确', `${metrics.phase406_H12_sequence_semantic_correct_count || 0}/${metrics.phase406_formal_discovery_case_count || 0}`],
    ['Phase406 正式组通过', `${metrics.phase406_formal_group_pass_count || 0}/72`],
    ['Phase406 跨模型状态族', `${metrics.phase406_crossmodel_state_family_count || 0}/3`],
    ['Phase407 注册语义正确', `${metrics.phase407_semantic_correct_count || 0}/${metrics.phase407_formal_discovery_case_count || 0}`],
    ['Phase407 四门完整组', `${metrics.phase407_fully_semantic_gated_group_count || 0}/${metrics.phase407_formal_group_count || 0}`],
    ['Phase407 跨模型状态族', `${metrics.phase407_crossmodel_state_family_count || 0}/3`],
    ['Phase408 注册响应', `${metrics.phase408_registered_response_observed_count || 0}/${metrics.phase408_formal_discovery_case_count || 0}`],
    ['Phase408 功能分区组', `${metrics.phase408_functional_group_pass_count || 0}/108`],
    ['Phase408 发现跨模型族', `${metrics.phase408_discovery_crossmodel_partition_family_count || 0}/3`],
    ['Phase408 行为留出跨模型族', `${metrics.phase408_behavioral_crossmodel_partition_family_count || 0}/3`],
    ['Phase409 协议注册案例', formatNumber(metrics.phase409_registered_abstract_case_count)],
    ['Phase409 双规则一致', `${metrics.phase409_dual_rule_agreement_count || 0}/${metrics.phase409_rule_engine_scenario_count || 0}`],
    ['Phase409 外部规则复核', `${metrics.phase409_external_rule_review_count || 0}/1`],
    ['Phase409 模型案例', formatNumber(metrics.phase409_model_case_count)],
    ['Phase410 h3 顺序审计', `${(metrics.phase410_h3_order_variant_count || 0) - (metrics.phase410_h3_order_symmetry_failure_count || 0)}/${metrics.phase410_h3_order_variant_count || 0}`],
    ['Phase410 语法有限全集', `${(metrics.phase410_grammar_finite_case_count || 0) - (metrics.phase410_grammar_failure_count || 0)}/${metrics.phase410_grammar_finite_case_count || 0}`],
    ['Phase410 外部审阅者', `${metrics.phase410_completed_external_reviewer_count || 0}/${metrics.phase410_required_external_reviewer_count || 0}`],
    ['Phase410 密封采集器等价', `${metrics.phase410_sealed_model_collector_case_count || 0}/165`],
    ['Phase410 模型案例', formatNumber(metrics.phase410_model_case_count)],
    ['Phase411 有限语义合同', `${(metrics.phase411_finite_semantic_case_count || 0) - (metrics.phase411_finite_semantic_failure_count || 0)}/${metrics.phase411_finite_semantic_case_count || 0}`],
    ['Phase411 注册状态操作', `${metrics.phase411_registered_operation_count || 0}/46`],
    ['Phase411 操作组合闭包', `${(metrics.phase411_operation_composition_case_count || 0) - (metrics.phase411_operation_composition_failure_count || 0)}/${metrics.phase411_operation_composition_case_count || 0}`],
    ['Phase411 历史规则协变', `${(metrics.phase411_history_covariance_case_count || 0) - (metrics.phase411_history_covariance_failure_count || 0)}/${metrics.phase411_history_covariance_case_count || 0}`],
    ['Phase411 粗分区不稳定单元', formatNumber(metrics.phase411_coarse_unstable_operation_cell_count)],
    ['Phase411 外部审阅者', `${metrics.phase411_completed_external_reviewer_count || 0}/${metrics.phase411_required_external_reviewer_count || 0}`],
    ['Phase411 密封采集器等价', `${metrics.phase411_sealed_model_collector_case_count || 0}/165`],
    ['Phase411 模型案例', formatNumber(metrics.phase411_model_case_count)],
    ['Phase412 类型化观察者协变', `${(metrics.phase412_observer_operation_cell_count || 0) - (metrics.phase412_typed_observer_unstable_cell_count || 0)}/${metrics.phase412_observer_operation_cell_count || 0}`],
    ['Phase412 固定角色失败已解释', `${metrics.phase412_role_transport_explained_cell_count || 0}/${metrics.phase412_fixed_observer_unstable_cell_count || 0}`],
    ['Phase412 有限分区穷举', `${metrics.phase412_finite_partition_count || 0}/${metrics.phase412_finite_partition_count || 0}`],
    ['Phase412 全局非平凡商', `${metrics.phase412_global_qualifying_nontrivial_partition_count || 0}/${metrics.phase412_nontrivial_partition_count || 0}`],
    ['Phase412 外部角色索引分区束', formatNumber(metrics.phase412_role_indexed_partition_bundle_count)],
    ['Phase412 不可逆操作', `${metrics.phase412_registered_irreversible_operation_count || 0}/7`],
    ['Phase412 跨族桥', `${metrics.phase412_registered_cross_family_bridge_count || 0}/4`],
    ['Phase412 外部审阅者', `${metrics.phase412_completed_external_reviewer_count || 0}/${metrics.phase412_required_external_reviewer_count || 0}`],
    ['Phase412 密封采集器等价', `${metrics.phase412_sealed_model_collector_case_count || 0}/165`],
    ['Phase412 模型案例', formatNumber(metrics.phase412_model_case_count)],
    ['Phase413 终端相同有限轨迹', `${metrics.phase413_same_terminal_path_count || 0}/${metrics.phase413_synthetic_path_count || 0}`],
    ['Phase413 中间不同轨迹对', `${metrics.phase413_internal_distinct_path_pair_count || 0}/${metrics.phase413_synthetic_path_pair_count || 0}`],
    ['Phase413 一步相同但未来不同', `${metrics.phase413_future_different_pair_count || 0}/${metrics.phase413_future_state_pair_count || 0}`],
    ['Phase413 固定通道读数反例', `${metrics.phase413_fixed_coordinate_probe_failure_count || 0}/${metrics.phase413_channel_permutation_case_count || 0}`],
    ['Phase413 合格层内局部概率读出', `${metrics.phase413_qualified_direct_layer_local_readout_count || 0}/${metrics.phase413_direct_layer_local_readout_count || 0}`],
    ['Phase413 外部审阅者', `${metrics.phase413_completed_external_reviewer_count || 0}/${metrics.phase413_required_external_reviewer_count || 0}`],
    ['Phase413 密封采集器等价', `${metrics.phase413_sealed_model_collector_case_count || 0}/165`],
    ['Phase413 模型案例', formatNumber(metrics.phase413_model_case_count)],
    ['Phase414 混合证据目录', `${metrics.phase414_catalog_item_count || 0}/${metrics.phase414_catalog_item_count || 0}`],
    ['Phase414 目录严格机制闭合', `${metrics.phase414_catalog_mechanism_closed_count || 0}/${metrics.phase414_catalog_item_count || 0}`],
    ['Phase414 完整状态续跑恒等', `${metrics.phase414_natural_replay_exact_count || 0}/${metrics.phase414_natural_replay_cell_count || 0}`],
    ['Phase414 层间终端核变化', `${metrics.phase414_layerwise_terminal_kernel_variation_count || 0}/${metrics.phase414_natural_replay_case_count || 0}`],
    ['Phase414 不完整状态反例', `${metrics.phase414_incomplete_state_counterexample_count || 0}/${metrics.phase414_natural_replay_cell_count || 0}`],
    ['Phase414 观察者索引轨迹', `${metrics.phase414_varying_observer_trajectory_count || 0}/${metrics.phase414_observer_trajectory_count || 0}`],
    ['Phase414 跨 tokenizer 语义事件', `${metrics.phase414_cross_tokenizer_semantic_alignment_count || 0}/${metrics.phase414_cross_tokenizer_semantic_event_count || 0}`],
    ['Phase414 合格观察者', `${metrics.phase414_qualified_observer_count || 0}/${metrics.phase414_observer_method_count || 0}`],
    ['Phase414 外部审阅者', `${metrics.phase414_completed_external_reviewer_count || 0}/${metrics.phase414_required_external_reviewer_count || 0}`],
    ['Phase414 密封采集器等价', `${metrics.phase414_sealed_model_collector_case_count || 0}/165`],
    ['Phase414 模型案例', formatNumber(metrics.phase414_model_case_count)],
    ['单神经元因果', formatNumber(metrics.single_unit_causal_count)],
    ['完整自然链', formatNumber(metrics.full_natural_chain_pass_count)],
  ];

  return (
    <section style={{ display: 'grid', gap: 10 }}>
      <button
        type="button"
        aria-expanded={expanded}
        aria-controls="evidence-kernel-detail"
        onClick={() => setExpanded((value) => !value)}
        style={{
          width: '100%',
          padding: 18,
          color: '#e2e8f0',
          textAlign: 'left',
          fontFamily: 'inherit',
          cursor: 'pointer',
          borderRadius: 8,
          border: `1px solid ${expanded ? state.color : 'rgba(148,163,184,0.16)'}`,
          borderLeft: `3px solid ${state.color}`,
          background: expanded ? 'rgba(15,23,42,0.56)' : 'rgba(15,23,42,0.24)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
          <div style={{ display: 'flex', gap: 10, minWidth: 0 }}>
            <Database size={20} color="#22d3ee" style={{ flex: '0 0 auto' }} />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                <h4 style={{ margin: 0, fontSize: 16, color: '#f8fafc' }}>统一证据内核</h4>
                <span style={{ color: '#94a3b8', fontSize: 10 }}>Phase {atlas?.phase || manifest?.phase || '-'}</span>
                <span style={{ color: state.color, fontSize: 10, fontWeight: 800 }}>{state.label}</span>
              </div>
              <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.6, marginTop: 4 }}>{state.detail}</div>
            </div>
          </div>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: '#7dd3fc', fontSize: 11, whiteSpace: 'nowrap' }}>
            {expanded ? '收起详细证据' : '查看详细证据'}
            {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8, marginTop: 14 }}>
          {summaryMetrics.map(([label, value]) => (
            <div key={label} style={{ padding: '8px 10px', borderTop: '1px solid rgba(148,163,184,0.12)' }}>
              <div style={{ color: '#94a3b8', fontSize: 10 }}>{label}</div>
              <div style={{ color: value === '0' ? '#fda4af' : '#e2e8f0', fontSize: 15, fontWeight: 800, fontFamily: 'monospace', marginTop: 3 }}>{value}</div>
            </div>
          ))}
        </div>
      </button>

      {expanded && (
        <div id="evidence-kernel-detail" style={{ display: 'grid', gap: 12, padding: '2px 0 4px' }}>
          {error && <div style={{ color: '#fda4af', fontSize: 11 }}>{error}</div>}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(100px, 1fr))', gap: 6, overflowX: 'auto', paddingBottom: 2 }}>
            {EVIDENCE_STAGES.map((stage, index) => {
              const reached = index <= 2;
              return (
                <div key={stage.label} style={{ minWidth: 100, padding: '8px 9px', borderTop: `2px solid ${reached ? '#22d3ee' : 'rgba(148,163,184,0.18)'}`, background: reached ? 'rgba(8,47,73,0.22)' : 'rgba(15,23,42,0.2)' }}>
                  <div style={{ color: reached ? '#bae6fd' : '#64748b', fontSize: 10, fontWeight: 700 }}>{stage.label}</div>
                  <div style={{ color: '#64748b', fontSize: 9, marginTop: 2 }}>{stage.level}</div>
                </div>
              );
            })}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 8 }}>
            {MODULES.map((module) => {
              const Icon = module.icon;
              const selected = active === module.id;
              return (
                <button
                  key={module.id}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => setActive(module.id)}
                  style={{
                    padding: 11,
                    minHeight: 68,
                    textAlign: 'left',
                    color: '#e2e8f0',
                    border: `1px solid ${selected ? module.color : 'rgba(148,163,184,0.14)'}`,
                    borderRadius: 6,
                    background: selected ? 'rgba(8,47,73,0.3)' : 'rgba(15,23,42,0.28)',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                  }}
                >
                  <Icon size={16} color={module.color} />
                  <div style={{ fontSize: 11, fontWeight: 700, marginTop: 7 }}>{module.title}</div>
                  <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 3 }}>{summaries[module.id]}</div>
                </button>
              );
            })}
          </div>

          <div style={{ borderTop: `2px solid ${MODULES.find((module) => module.id === active)?.color || '#22d3ee'}`, background: 'rgba(2,6,23,0.42)', padding: 14 }}>
            {active === 'atlas' && <AtlasDetail atlas={atlas} />}
            {active === 'progress' && <ProgressDetail progress={progress} />}
            {active === 'claims' && <ClaimsDetail claims={claims} />}
            {active === 'runs' && <RunsDetail runs={manifest?.runs || []} />}
            {active === 'gaps' && <GapsDetail gaps={gaps} />}
          </div>
        </div>
      )}
    </section>
  );
}
