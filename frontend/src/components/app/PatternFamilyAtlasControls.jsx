import { Eye, EyeOff, Network, SlidersHorizontal } from 'lucide-react';
import './PatternFamilyAtlasControls.css';

const FOCUS_OPTIONS = [
  { value: 'key', label: '关键候选' },
  { value: 'natural', label: '自然交叉' },
  { value: 'registered', label: '注册集合' },
  { value: 'cross_model', label: '跨模型' },
  { value: 'refined', label: '扩展审计' },
  { value: 'interface_path', label: '接口脉络' },
  { value: 'dynamic_path', label: '动态时序' },
  { value: 'natural_necessity', label: '自然必要性' },
  { value: 'competition', label: '竞争路径' },
];

export function PatternFamilyAtlasControls({
  enabled,
  onEnabledChange,
  atlas,
  familyId,
  onFamilyChange,
  modelKey,
  onModelChange,
  evidenceFocus,
  onEvidenceFocusChange,
  maxUnits,
  onMaxUnitsChange,
  variant = 'floating',
  showModel = true,
  showDetails = true,
}) {
  if (!enabled) {
    return (
      <button
        type="button"
        className={`pattern-atlas-restore pattern-atlas-restore--${variant}`}
        onClick={() => onEnabledChange(true)}
        title="打开模式族物理叠层"
        aria-label="打开模式族物理叠层"
      >
        <Network size={18} />
      </button>
    );
  }

  const metrics = atlas.partition?.metrics;
  const latestMetrics = atlas.manifest?.metrics || {};
  const hasPhase393 = Number(latestMetrics.phase393_attribute_direction_count || 0) > 0;
  const mappingStatus = atlas.mapped ? '物理候选已叠加' : '真实单元尚未映射';
  const mappingTone = atlas.mapped ? 'mapped' : 'unmapped';

  return (
    <section className={`pattern-atlas-controls pattern-atlas-controls--${variant}`} aria-label="模式族物理叠层控制">
      <header className="pattern-atlas-controls__header">
        <div className="pattern-atlas-controls__title">
          <Network size={16} />
          <span>语言模式族物理叠层</span>
          <span className={`pattern-atlas-status pattern-atlas-status--${mappingTone}`}>{mappingStatus}</span>
        </div>
        <button
          type="button"
          className="pattern-atlas-icon-button"
          onClick={() => onEnabledChange(false)}
          title="隐藏模式族叠层"
          aria-label="隐藏模式族叠层"
        >
          <EyeOff size={16} />
        </button>
      </header>

      <div className="pattern-atlas-controls__body">
        <label className="pattern-atlas-field">
          <span>模式族</span>
          <select value={familyId} onChange={(event) => onFamilyChange(event.target.value)}>
            {atlas.families.map((family) => {
              const mapped = (family.physical_mapping?.models || []).length > 0;
              return (
                <option key={family.family_id} value={family.family_id}>
                  {family.family_name}{mapped ? '' : ' · 未映射'}
                </option>
              );
            })}
          </select>
        </label>

        {showModel && (
          <label className="pattern-atlas-field pattern-atlas-field--model">
            <span>模型</span>
            <select value={modelKey} onChange={(event) => onModelChange(event.target.value)} aria-label="模型">
              <option value="qwen3-4b">Qwen3</option>
              <option value="glm4-9b">GLM4</option>
              <option value="ds7b">DS7B</option>
            </select>
          </label>
        )}

        <label className="pattern-atlas-field pattern-atlas-field--evidence">
          <span>证据范围</span>
          <select
            value={evidenceFocus}
            onChange={(event) => onEvidenceFocusChange(event.target.value)}
            disabled={!atlas.mapped}
            aria-label="证据范围"
          >
            {FOCUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>

        <label className="pattern-atlas-range">
          <SlidersHorizontal size={14} />
          <span>关键候选 {maxUnits}</span>
          <input
            type="range"
            min="12"
            max="96"
            step="12"
            value={maxUnits}
            onChange={(event) => onMaxUnitsChange(Number(event.target.value))}
            disabled={!atlas.mapped}
          />
        </label>

        <button
          type="button"
          className="pattern-atlas-icon-button pattern-atlas-icon-button--visible"
          onClick={() => onEnabledChange(true)}
          title="当前显示模式族脉络"
          aria-label="当前显示模式族脉络"
        >
          <Eye size={16} />
        </button>
      </div>

      {hasPhase393 && (
        <div className="pattern-atlas-controls__latest" aria-label="Phase393 最新研究边界">
          <strong>P393</strong>
          <span>属性搬运 {latestMetrics.phase393_attribute_answer_switch_count || 0}/{latestMetrics.phase393_attribute_direction_count || 0}</span>
          <span>错误深度 {latestMetrics.phase393_wrong_depth_attribute_switch_count || 0}/{latestMetrics.phase393_attribute_direction_count || 0}</span>
          <span>深度特异 {latestMetrics.phase393_depth_specificity_pass_model_count || 0}/3</span>
          <span>完整路径 0</span>
          <span>神经元路径 0</span>
        </div>
      )}

      {showDetails && <footer className="pattern-atlas-controls__footer">
        {atlas.loading ? (
          <span>正在读取证据分区</span>
        ) : atlas.error ? (
          <span className="is-error">{atlas.error}</span>
        ) : atlas.mapped ? (
          <>
            <span>{atlas.model}</span>
            <span>{metrics?.candidate_layer_count || 0} 个关键层</span>
            <span>{metrics?.unique_unit_count || 0} 个唯一候选</span>
            <span>{metrics?.component_set_member_count || 0} 个集合成员</span>
            <span>{metrics?.natural_overlap_count || 0} 个自然交叉</span>
            <span>{metrics?.group_supported_candidate_count || 0} 个组级支持候选</span>
            <span>{metrics?.expanded_confirmed_candidate_count || 0} 个扩大确认候选</span>
            <span>{metrics?.natural_retrieval_path_count || 0} 条自然检索路径</span>
            <span>{metrics?.full_natural_chain_pass_count || 0} 条严格自然闭合</span>
            <span>{metrics?.upstream_residual_mediation_edge_count || 0} 条上游中介候选</span>
            <span>{metrics?.full_vocabulary_mediation_path_count || 0} 条全词表路径</span>
            <span>{metrics?.phase330_mechanism_count || 0} 个九族机制</span>
            <span>{metrics?.phase330_component_member_count || 0} 个新组件成员</span>
            <span>{metrics?.phase330_registered_causal_case_count || 0} 个注册留出案例</span>
            <span>{metrics?.phase330_local_set_readout_specific_mechanism_count || 0} 个本模型集合读出支持</span>
            <span>{metrics?.phase330_cross_model_set_readout_specific_mechanism_count || 0} 个跨模型读出支持</span>
            <span>{metrics?.phase331_refined_mechanism_count || 0} 个扩展审计机制</span>
            <span>{metrics?.phase331_full_gate_pass_count || 0} 个完整门槛通过</span>
            <span>{metrics?.phase331_behavior_mechanism_closed_count || 0} 个行为机制闭合</span>
            <span>{metrics?.phase332_interface_path_member_count || 0} 个接口路径成员</span>
            <span>{metrics?.phase332_stable_shared_member_count || 0} 个稳定共享成员</span>
            <span>{metrics?.phase332_specific_interface_branch_member_count || 0} 个接口分支成员</span>
            <span>{metrics?.phase332_full_gate_pass_count || 0} 个接口路径完整门</span>
            <span>{metrics?.phase333_dynamic_event_count || 0} 个动态事件锚点</span>
            <span>{metrics?.phase333_stable_sequence_count || 0} 个稳定时序接口</span>
            <span>{metrics?.phase333_specific_block_cell_count || 0} 个特异状态块单元</span>
            <span>{metrics?.phase333_compensation_candidate_count || 0} 条补偿候选</span>
            <span>{metrics?.phase333_full_gate_pass_count || 0} 个动态路径完整门</span>
            <span>{metrics?.phase334_candidate_node_count || 0} 个自然必要性候选</span>
            <span>{metrics?.phase334_baseline_eligible_cell_count || 0} 个基线合格单元</span>
            <span>{metrics?.phase334_natural_necessity_candidate_count || 0} 个局部必要性单元</span>
            <span>{metrics?.phase334_propagation_candidate_count || 0} 条下游传播候选</span>
            <span>{metrics?.phase334_local_gate_pass_count || 0} 个局部完整门</span>
            <span>{metrics?.phase334_cross_model_gate_count || 0} 个跨模型必要性门</span>
            <span>{latestMetrics.phase391_physical_local_parent_layout_count || 0} 个物理局部父节点布局</span>
            <span>{latestMetrics.phase393_attribute_answer_switch_count || 0}/{latestMetrics.phase393_attribute_direction_count || 0} 属性内容切换</span>
            <span>{latestMetrics.phase393_wrong_depth_attribute_switch_count || 0}/{latestMetrics.phase393_attribute_direction_count || 0} 错误深度切换</span>
            <span>{latestMetrics.phase393_attribute_transport_pass_model_count || 0}/3 属性搬运模型</span>
            <span>{latestMetrics.phase393_depth_specificity_pass_model_count || 0}/3 深度特异模型</span>
            <span>{Math.round((metrics?.phase330_heldout_peak_10pct_rate || 0) * 100)}% 留出峰层命中</span>
            <span>{metrics?.tokenwise_beats_pooled_count || 0} 条逐词元正向胜出</span>
            <span>{metrics?.blocker_decline_pass_count || 0} 条阻挡者下降</span>
            <span>{metrics?.carrier_member_mediation_pass_count || 0} 条成员中介</span>
            <span>{metrics?.top1_unlock_pass_count || 0} 条首选解锁</span>
            <span>{metrics?.causal_path_edge_count || 0} 条因果路径边</span>
            <span className="pattern-atlas-boundary">单神经元门 {metrics?.single_unit_intervention_gate_open_count || 0} · 完整语言路径 0 · 深度特异路径 {latestMetrics.phase393_depth_specificity_pass_model_count || 0}/3</span>
          </>
        ) : (
          <span>{atlas.family?.family_name || '当前模式族'} · 等待真实物理映射</span>
        )}
      </footer>}
    </section>
  );
}
