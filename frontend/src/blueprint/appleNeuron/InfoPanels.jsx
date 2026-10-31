/**
 * AppleNeuron3D 信息面板组件
 * 从 AppleNeuron3DTab.jsx 拆分而来
 */

import { Html } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef, useState } from 'react';

import {
  LAYER_COUNT, DFF, IMPORTED_QUERY_NODE_MAX,
  ROLE_COLORS, DIMENSION_LABELS,
  FRUIT_COLORS,
  HARD_PROBLEM_EXPERIMENT_LABELS,
  THEORY_OBJECT_RESEARCH_MAP,
  DEFAULT_LANGUAGE_FOCUS,
} from './constants';

import {
  toSafeNumber, formatPreviewValue,
  shouldShowResearchAssetInTopRight,
  buildAutoDisplayProfile,
  nodeDisplayGroup,
} from './utils';

const smallActionButtonStyle = {
  borderRadius: 8,
  border: '1px solid rgba(122, 162, 255, 0.5)',
  background: 'rgba(28, 53, 102, 0.75)',
  color: '#dbe9ff',
  fontSize: 12,
  padding: '7px 10px',
  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
};

const panelCardStyle = {
  borderRadius: 14,
  padding: 14,
  border: '1px solid rgba(118, 170, 255, 0.25)',
  background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
};

const inputStyle = {
  width: '100%',
  borderRadius: 8,
  border: '1px solid rgba(122, 162, 255, 0.3)',
  background: 'rgba(7, 12, 25, 0.8)',
  color: '#dbe9ff',
  padding: '8px 10px',
  fontSize: 12,
};

const textAreaStyle = {
  width: '100%',
  borderRadius: 8,
  border: '1px solid rgba(122, 162, 255, 0.3)',
  background: 'rgba(7, 12, 25, 0.8)',
  color: '#dbe9ff',
  padding: '8px 10px',
  fontSize: 12,
  resize: 'vertical',
};

const fixedFileControlWidth = 240;


const inferScanOptionConcept = (fileMeta) => {
  const lower = String(fileMeta?.name || fileMeta?.path || '').toLowerCase();
  if (lower.includes('unified_math_structure_decode')) return '统一解码';
  if (lower.includes('agi_research_stage_bundle_manifest')) return '阶段实验总清单';
  if (lower.includes('agi_four_tasks_suite_manifest')) return '四任务总清单';
  if (lower.includes('multidim_encoding_probe')) return '三维编码探针';
  if (lower.includes('multidim_causal_ablation')) return '三维因果消融';
  if (lower.includes('multidim_multiseed_stability')) return '三维多 Seed 稳定性';
  if (lower.includes('minimal_causal_circuit_search')) return '最小因果子回路';
  if (lower.includes('variable_binding_hard_verification')) return '变量绑定验证';
  if (lower.includes('unified_coordinate_system_test')) return '统一坐标测试';
  if (lower.includes('concept_family_parallel_scale')) return '概念族并行尺度';
  if (lower.includes('dynamic_binding_stress_test')) return '动态绑定压力测试';
  if (lower.includes('long_horizon_causal_trace_test')) return '长程因果链路';
  if (lower.includes('local_credit_assignment_proxy_test')) return '局部信用代理测试';
  if (lower.includes('triplet_targeted_causal_scan')) return '三元组定向因果';
  if (lower.includes('triplet_targeted_multiseed_stability')) return '三元组多 Seed 稳定性';
  if (lower.includes('mass_noun') || lower.includes('noun_scan') || lower.includes('encoding_scan')) return '名词编码扫描';
  return String(fileMeta?.name || '研究资产');
};

function AppleSwitchMechanismInsightsPanel({ workspace, compact = false }) {
  const appleSwitchMechanismData = workspace?.appleSwitchMechanismData || null;
  const selected = workspace?.selected || null;
  const setSelected = workspace?.setSelected || null;
  const nodes = workspace?.nodes || [];
  const [filterModel, setFilterModel] = useState('all');
  const [filterKind, setFilterKind] = useState('all');
  const [filterDirection, setFilterDirection] = useState('all');
  const [filterCircuit, setFilterCircuit] = useState('all');
  const [filterKeyword, setFilterKeyword] = useState('');

  if (!isAppleSwitchMechanismPayload(appleSwitchMechanismData)) {
    return null;
  }

  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;
  const modelEntries = Object.entries(appleSwitchMechanismData.models || {});
  const filteredModelEntries = modelEntries.filter(([modelKey]) => filterModel === 'all' || modelKey === filterModel);
  const normalizedKeyword = String(filterKeyword || '').trim().toLowerCase();
  const activeModelKey = selected?.detailType === 'apple_switch_unit'
    ? selected.modelKey
    : (filterModel !== 'all' ? filterModel : (appleSwitchMechanismData.models?.deepseek7b ? 'deepseek7b' : modelEntries[0]?.[0]));
  const activeModel = appleSwitchMechanismData.models?.[activeModelKey] || filteredModelEntries[0]?.[1] || modelEntries[0]?.[1] || null;
  const selectedUnit = selected?.detailType === 'apple_switch_unit' ? selected.appleSwitchUnit : null;
  const nodeByUnitId = Object.fromEntries(
    (Array.isArray(nodes) ? nodes : [])
      .filter((node) => node?.detailType === 'apple_switch_unit')
      .map((node) => [node.unitId, node])
  );

  const filterUnits = (units = []) => units.filter((unit) => {
    if (filterKind !== 'all' && unit?.kind !== filterKind) {
      return false;
    }
    if (filterDirection !== 'all') {
      const lateMean = Number(unit?.signed_effect?.late_mean_signed_contrast_switch_coupling || 0);
      const resolvedDirection = lateMean > 0 ? 'reverse' : 'forward';
      if (resolvedDirection !== filterDirection) {
        return false;
      }
    }
    if (filterCircuit === 'in' && !unit?.is_final_circuit_member) {
      return false;
    }
    if (filterCircuit === 'out' && unit?.is_final_circuit_member) {
      return false;
    }
    if (normalizedKeyword) {
      const haystack = [
        unit?.unit_id,
        unit?.role,
        getAppleSwitchUnitRoleLabel(unit?.role),
        unit?.kind === 'mlp_neuron' ? 'mlp 神经元' : '注意力头',
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      if (!haystack.includes(normalizedKeyword)) {
        return false;
      }
    }
    return true;
  });

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>苹果切换机制总览</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7, marginBottom: 8 }}>
          <div>{`统一资产: 已导入`}</div>
          <div>{`峰值层匹配率: ${(toSafeNumber(appleSwitchMechanismData?.aggregate_stability?.peak_layer_match_rate, 0) * 100).toFixed(1)}%`}</div>
          <div>{`当前聚焦模型: ${activeModel?.model_name || activeModelKey || '-'}`}</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: compact ? 'repeat(2, minmax(0, 1fr))' : 'repeat(5, minmax(0, 1fr))', gap: 8, marginBottom: 10 }}>
          <select value={filterModel} onChange={(e) => setFilterModel(e.target.value)} style={inputStyle}>
            <option value="all">全部模型</option>
            {modelEntries.map(([modelKey, modelPayload]) => (
              <option key={`apple-switch-filter-model-${modelKey}`} value={modelKey}>{modelPayload?.model_name || modelKey}</option>
            ))}
          </select>
          <select value={filterKind} onChange={(e) => setFilterKind(e.target.value)} style={inputStyle}>
            <option value="all">全部类型</option>
            <option value="attention_head">注意力头</option>
            <option value="mlp_neuron">MLP 神经元</option>
          </select>
          <select value={filterDirection} onChange={(e) => setFilterDirection(e.target.value)} style={inputStyle}>
            <option value="all">全部方向</option>
            <option value="forward">正向支撑</option>
            <option value="reverse">反向校正</option>
          </select>
          <select value={filterCircuit} onChange={(e) => setFilterCircuit(e.target.value)} style={inputStyle}>
            <option value="all">全部回路成员</option>
            <option value="in">仅最小回路内</option>
            <option value="out">仅最小回路外</option>
          </select>
          <input
            value={filterKeyword}
            onChange={(e) => setFilterKeyword(e.target.value)}
            placeholder="关键词: H:2:26 / 锚点 / 头"
            style={inputStyle}
          />
        </div>
        <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.7, marginBottom: 8 }}>
          {`筛选方式: 模型 + 类型 + 方向 + 最小回路成员 + 关键词，可同时生效。`}
        </div>
        <div style={{ display: 'grid', gap: 8 }}>
          {filteredModelEntries.map(([modelKey, modelPayload]) => {
            const visibleUnits = filterUnits(modelPayload?.core_units || []);
            return (
            <div
              key={`apple-switch-model-${modelKey}`}
              style={{
                borderRadius: 10,
                border: `1px solid ${APPLE_SWITCH_MODEL_COLORS[modelKey] || '#6ea8ff'}55`,
                padding: '8px 10px',
                background: 'rgba(255,255,255,0.03)',
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 700, color: APPLE_SWITCH_MODEL_COLORS[modelKey] || '#dbeafe' }}>
                {modelPayload?.model_name || modelKey}
              </div>
              <div style={{ marginTop: 4, fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
                <div>{`敏感层: L${toSafeNumber(modelPayload?.best_sensitive_layer?.layer_index, 0)}`}</div>
                <div>{`共享底座最强层: L${toSafeNumber(modelPayload?.best_shared_layer?.layer_index, 0)}`}</div>
                <div>{`核心单元: ${Array.isArray(modelPayload?.core_units) ? modelPayload.core_units.length : 0}`}</div>
                <div>{`筛选后单元: ${visibleUnits.length}`}</div>
                <div>{`最小回路规模: ${Array.isArray(modelPayload?.effective_circuit?.final_subset) ? modelPayload.effective_circuit.final_subset.length : 0}`}</div>
              </div>
              <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
                {visibleUnits.slice(0, compact ? 4 : 8).map((unit) => {
                  const node = nodeByUnitId[unit.unit_id];
                  const clickable = typeof setSelected === 'function' && node;
                  return (
                    <button
                      key={`apple-switch-unit-${modelKey}-${unit.unit_id}`}
                      type="button"
                      onClick={() => clickable && setSelected(node)}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        borderRadius: 8,
                        border: '1px solid rgba(255,255,255,0.1)',
                        padding: '6px 8px',
                        background: node?.id === selected?.id ? 'rgba(56,189,248,0.16)' : 'rgba(255,255,255,0.02)',
                        color: '#dbeafe',
                        cursor: clickable ? 'pointer' : 'default',
                      }}
                    >
                      <div style={{ fontSize: 11, fontWeight: 700 }}>{unit.unit_id}</div>
                      <div style={{ marginTop: 2, fontSize: 10, color: '#9bb3de' }}>
                        {`${getAppleSwitchUnitRoleLabel(unit.role)} | 有效 ${toSafeNumber(unit?.scores?.effective_score, 0).toFixed(3)} | ${unit?.signed_effect?.direction_label || '-'}`}
                      </div>
                      <div style={{ marginTop: 2, fontSize: 10, color: '#7ea2c9' }}>
                        {`${unit.kind === 'mlp_neuron' ? 'MLP 神经元' : '注意力头'} | ${unit.is_final_circuit_member ? '最小回路内' : '最小回路外'}`}
                      </div>
                    </button>
                  );
                })}
                {visibleUnits.length === 0 ? (
                  <div style={{ fontSize: 11, color: '#8ea5c5', lineHeight: 1.6 }}>
                    当前筛选条件下没有命中单元，可以放宽一个条件再看。
                  </div>
                ) : null}
              </div>
            </div>
          );
          })}
        </div>
      </div>

      {activeModel ? (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>
            {selectedUnit ? `${selected.unitId} 逐层过程` : `${activeModel.model_name || activeModelKey} 层过程时间线`}
          </div>
          <div style={{ display: 'grid', gap: 6, maxHeight: compact ? 220 : 320, overflowY: 'auto' }}>
            {selectedUnit ? (
              (selectedUnit?.process_timeline || []).map((row) => {
                const signedValue = toSafeNumber(row?.signed_contrast_switch_coupling, 0);
                const relativeDrop = toSafeNumber(row?.relative_separation_drop, 0);
                const signedWidth = Math.min(100, Math.abs(signedValue) * 650);
                const relativeWidth = Math.min(100, Math.abs(relativeDrop) * 1800);
                return (
                  <div key={`apple-switch-process-${selected.unitId}-${row.layer_index}`} style={{ display: 'grid', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 11, color: '#dbeafe' }}>
                      <span>{`L${row.layer_index}`}</span>
                      <span>{row.direction_label}</span>
                    </div>
                    <div style={{ height: 5, borderRadius: 999, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                      <div style={{ width: `${signedWidth}%`, height: '100%', background: signedValue <= 0 ? '#38bdf8' : '#fb7185' }} />
                    </div>
                    <div style={{ height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                      <div style={{ width: `${relativeWidth}%`, height: '100%', background: '#f59e0b' }} />
                    </div>
                    <div style={{ fontSize: 10, color: '#8ea5c5' }}>
                      {`signed=${signedValue.toFixed(4)} | relative_drop=${relativeDrop.toFixed(4)} | pc1=${toSafeNumber(row?.pc1_explained_variance_ratio, 0).toFixed(3)}`}
                    </div>
                  </div>
                );
              })
            ) : (
              (activeModel?.layer_summary || []).map((row) => {
                const sharedWidth = Math.min(100, toSafeNumber(row?.shared_active_neuron_count, 0) * 4);
                const splitWidth = Math.min(100, Math.abs(toSafeNumber(row?.excess_switch_drop, 0)) * 1800);
                return (
                  <div key={`apple-switch-layer-summary-${activeModelKey}-${row.layer_index}`} style={{ display: 'grid', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 11, color: '#dbeafe' }}>
                      <span>{`L${row.layer_index}`}</span>
                      <span>{row.process_label}</span>
                    </div>
                    <div style={{ height: 5, borderRadius: 999, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                      <div style={{ width: `${sharedWidth}%`, height: '100%', background: '#34d399' }} />
                    </div>
                    <div style={{ height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                      <div style={{ width: `${splitWidth}%`, height: '100%', background: '#f59e0b' }} />
                    </div>
                    <div style={{ fontSize: 10, color: '#8ea5c5' }}>
                      {`共享=${toSafeNumber(row?.shared_active_neuron_count, 0)} | 分裂强度=${toSafeNumber(row?.excess_switch_drop, 0).toFixed(4)} | Jaccard=${toSafeNumber(row?.active_jaccard, 0).toFixed(3)}`}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

const buildScanContentLabel = (data, fileMeta) => {
  const nounRecords = Array.isArray(data?.noun_records) ? data.noun_records : [];
  if (nounRecords.length > 0) {
    const pairs = nounRecords
      .slice(0, 2)
      .map((row) => {
        const noun = String(row?.noun || '').trim();
        const category = String(row?.category || '未分类').trim() || '未分类';
        if (!noun) {
          return null;
        }
        return `${noun}-${category}`;
      })
      .filter(Boolean);
    if (pairs.length > 0) {
      return pairs.join(' / ');
    }
  }

  if (data?.experiment_id && HARD_PROBLEM_EXPERIMENT_LABELS[data.experiment_id]) {
    return HARD_PROBLEM_EXPERIMENT_LABELS[data.experiment_id];
  }
  if (data?.suite_id === 'agi_four_tasks_suite_v1') {
    return '四任务验证';
  }
  if (data?.bundle_id === 'agi_research_stage_bundle_v1') {
    return '阶段实验总清单';
  }
  if (isAppleSwitchMechanismPayload(data)) {
    return '苹果切换机制';
  }
  if (isUnifiedDecodePayload(data)) {
    return '风格-逻辑-语法';
  }
  if (data?.dimensions?.style && data?.dimensions?.logic && data?.dimensions?.syntax) {
    return '风格-逻辑-语法';
  }
  return inferScanOptionConcept(fileMeta);
};

const formatScanOptionLabel = (fileMeta, contentLabel = '') => {
  const conceptLabel = contentLabel || inferScanOptionConcept(fileMeta);
  const mtime = String(fileMeta?.mtime_iso || '').slice(0, 19).replace('T', ' ');
  return mtime ? `${conceptLabel} | ${mtime}` : conceptLabel;
};



export function AppleNeuronEncodingInfoPanels({ workspace, compact = false }) {
  const nodes = workspace?.nodes || [];
  const summary = workspace?.summary || {};
  const metrics = workspace?.modeMetrics || [];
  const multidimProbe = workspace?.multidimProbeData || null;
  const multidimCausal = workspace?.multidimCausalData || null;
  const hardProblemResults = workspace?.hardProblemResults || {};
  const unifiedDecodeResult = workspace?.unifiedDecodeResult || null;
  const bundleManifest = workspace?.bundleManifest || null;
  const fourTasksManifest = workspace?.fourTasksManifest || null;
  const activeDim = workspace?.multidimActiveDimension || 'style';

  const layerRows = useMemo(() => {
    const map = new Map();
    nodes.forEach((node) => {
      const key = Number.isFinite(node.layer) ? node.layer : 0;
      const row = map.get(key) || { layer: key, count: 0, strength: 0 };
      row.count += 1;
      row.strength += Number(node.strength || 0);
      map.set(key, row);
    });
    return Array.from(map.values())
      .sort((a, b) => a.layer - b.layer)
      .map((row) => ({
        ...row,
        strength: row.count ? row.strength / row.count : 0,
      }));
  }, [nodes]);

  const maxCount = useMemo(() => Math.max(1, ...layerRows.map((row) => row.count)), [layerRows]);
  const hardProblemRows = useMemo(() => Object.entries(hardProblemResults), [hardProblemResults]);
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <AppleSwitchMechanismInsightsPanel workspace={workspace} compact={compact} />

      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>层级编码签名</div>
        <div style={{ display: 'grid', gap: 6, maxHeight: compact ? 130 : 170, overflowY: 'auto' }}>
          {layerRows.map((row) => (
            <div key={`layer-sign-${row.layer}`} style={{ display: 'grid', gridTemplateColumns: '44px 1fr', gap: 8, alignItems: 'center' }}>
              <div style={{ fontSize: 11, color: '#9bb3de' }}>{`L${row.layer}`}</div>
              <div style={{ display: 'grid', gap: 3 }}>
                <div style={{ height: 5, background: 'rgba(255,255,255,0.08)', borderRadius: 8, overflow: 'hidden' }}>
                  <div style={{ width: `${(row.count / maxCount) * 100}%`, height: '100%', background: '#67dfff' }} />
                </div>
                <div style={{ height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 8, overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(100, row.strength * 1000000)}%`, height: '100%', background: '#f59e0b' }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>数据机制指标</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>{`核心神经元: ${(summary.micro || 0) + (summary.macro || 0) + (summary.route || 0)}`}</div>
          <div>{`当前词元: ${summary.currentToken || '-'} (${((summary.currentTokenProb || 0) * 100).toFixed(1)}%)`}</div>
          <div>{`显示策略: ${summary.displayStrategy === 'auto' ? '自动聚焦' : summary.displayStrategy === 'all' ? '全部显示' : '手动筛选'}`}</div>
          <div>{`可见概念集: ${summary.visibleQuerySets || 0} / 隐藏概念集: ${summary.hiddenQuerySets || 0}`}</div>
        </div>
        {metrics.length > 0 ? (
          <div style={{ marginTop: 8, display: 'grid', gap: 4 }}>
            {metrics.map((metric, idx) => (
              <div key={`m-${metric.label}-${idx}`} style={{ fontSize: 11, color: '#9bb3de' }}>{`${metric.label}: ${metric.value}`}</div>
            ))}
          </div>
        ) : null}
        {multidimProbe ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`多维探针: 已导入`}</div>
            <div>{`当前维度: ${DIMENSION_LABELS[activeDim] || activeDim}`}</div>
            {Number.isFinite(multidimCausal?.diagonal_advantage?.[activeDim]) ? (
              <div>{`对角优势: ${toSafeNumber(multidimCausal?.diagonal_advantage?.[activeDim], 0).toFixed(4)}`}</div>
            ) : (
              <div style={{ color: '#7f95bb' }}>未导入三维因果消融结果</div>
            )}
          </div>
        ) : null}
        {hardProblemRows.length > 0 ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`硬伤实验导入: ${hardProblemRows.length}`}</div>
            {hardProblemRows.map(([expId, payload]) => {
              const title = HARD_PROBLEM_EXPERIMENT_LABELS[expId] || payload?.title || expId;
              const mm = payload?.metrics || {};
              if (expId === 'hard_problem_dynamic_binding_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 稳定性=${toSafeNumber(mm.binding_stability_index, 0).toFixed(3)} | 交换错率=${toSafeNumber(mm.role_swap_error_rate, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'hard_problem_long_horizon_trace_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 长程衰减=${toSafeNumber(mm.long_horizon_decay, 0).toFixed(3)} | 传输稳定=${toSafeNumber(mm.layer_transport_stability_mean, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'hard_problem_local_credit_assignment_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 局部充分=${toSafeNumber(mm.local_sufficiency_mean, 0).toFixed(3)} | 局部选择=${toSafeNumber(mm.local_selectivity_mean, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'triplet_targeted_causal_scan_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 三联分离=${toSafeNumber(mm.triplet_separability_index, 0).toFixed(3)} | 轴特异=${toSafeNumber(mm.axis_specificity_index, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'triplet_targeted_multiseed_stability_v1') {
                const seqMargin = toSafeNumber(mm?.global_mean_causal_margin_seq_logprob?.mean, 0);
                const posRatio = toSafeNumber(mm?.global_positive_causal_margin_ratio?.mean, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: seq边际均值=${seqMargin.toFixed(4)} | 正边际比例=${(posRatio * 100).toFixed(1)}%`}
                  </div>
                );
              }
              if (expId === 'hard_problem_variable_binding_verification_v1') {
                const meanDelta = toSafeNumber(mm?.mean_delta, 0);
                const improvedDims = toSafeNumber(mm?.improved_dimension_count, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 平均提升=${meanDelta.toFixed(4)} | 提升维度=${improvedDims}`}
                  </div>
                );
              }
              if (expId === 'minimal_causal_circuit_search_v1') {
                const drop = toSafeNumber(mm?.global?.intervention_drop_mean, 0);
                const repr = toSafeNumber(mm?.global?.reproducibility_jaccard_mean, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 干预下降=${drop.toFixed(4)} | 复现Jaccard=${repr.toFixed(4)}`}
                  </div>
                );
              }
              if (expId === 'unified_coordinate_system_test_v1') {
                const us = toSafeNumber(mm?.unified_coordinate_score, 0);
                const orth = toSafeNumber(mm?.probe_orthogonality?.orthogonality_index, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 统一分数=${us.toFixed(4)} | 正交性=${orth.toFixed(4)}`}
                  </div>
                );
              }
              if (expId === 'concept_family_parallel_scale_v1') {
                const appleShared = toSafeNumber(mm?.apple_chain_summary?.shared_base_ratio_vs_micro_union?.mean, 0);
                const catShared = toSafeNumber(mm?.cat_chain_summary?.shared_base_ratio_vs_micro_union?.mean, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 苹果共享=${appleShared.toFixed(4)} | 猫共享=${catShared.toFixed(4)}`}
                  </div>
                );
              }
              return <div key={`hp-${expId}`}>{`${title}: 已导入`}</div>;
            })}
          </div>
        ) : null}
        {unifiedDecodeResult ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`统一解码: 已导入`}</div>
            <div>{`假设通过率: ${(toSafeNumber(unifiedDecodeResult?.hypothesis_test?.pass_ratio, 0) * 100).toFixed(1)}%`}</div>
            <div>{`探针文件数: ${toSafeNumber(unifiedDecodeResult?.axis_stability?.n_probe_files, 0)}`}</div>
          </div>
        ) : null}
        {bundleManifest ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`批量清单: 已导入`}</div>
            <div>{`seed=${toSafeNumber(bundleManifest?.config?.seed, 0)} | 统一解码=${bundleManifest?.config?.run_unified_decoder ? '开启' : '关闭'}`}</div>
          </div>
        ) : null}
        {fourTasksManifest ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`四任务清单: 已导入`}</div>
            <div>{`all_success=${fourTasksManifest?.all_success ? 'true' : 'false'} | 任务数=${Object.keys(fourTasksManifest?.return_codes || {}).length}`}</div>
          </div>
        ) : null}
      </div>

      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>硬伤实验与统一编码说明</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7, display: 'grid', gap: 6, maxHeight: compact ? 220 : 300, overflowY: 'auto' }}>
          <div style={{ color: '#cfe2ff', fontWeight: 700 }}>硬伤实验节点（红/蓝/橙/紫）</div>
          <div>1. 来源：导入 `agi_research_result.v1` 的实验指标。</div>
          <div>2. 映射：每个关键指标生成一个节点，颜色代表实验类型。</div>
          <div>3. 位置：层号与神经元号由指标哈希映射到主 3D 空间（用于对比，不代表真实单一神经元定位）。</div>
          <div>4. 大小与亮度：由指标强度决定；误差类指标（error/collision/decay）按“越小越好”反向映射。</div>
          <div style={{ color: '#cfe2ff', fontWeight: 700, marginTop: 4 }}>统一编码节点（绿色）</div>
          <div>1. 来源：`unified_math_structure_decode.json` 的融合结果。</div>
          <div>2. 映射：按 style / logic / syntax 三个维度生成节点簇。</div>
          <div>3. 层位：优先使用 dominant layer pattern，把节点放到对应层；无模式时使用回退层。</div>
          <div>4. 强度：综合 `profile_cosine_mean` 与 `diagonal_advantage`，用于显示“轴稳定 + 因果可分离”程度。</div>
          <div style={{ color: '#cfe2ff', fontWeight: 700, marginTop: 4 }}>读图顺序（建议）</div>
          <div>1. 先看图例颜色区分实验类型。</div>
          <div>2. 再看模型说明中的指标数值（均值/通过率）。</div>
          <div>3. 最后点选节点查看 `metric/value/source`，判断该可视化是“证据节点”还是“结构节点”。</div>
        </div>
      </div>
    </div>
  );
}

export function AppleNeuronResearchAssetInfoPanel({ workspace, compact = false }) {
  const selectedScanPath = workspace?.selectedScanPath || '';
  const scanPreviewData = workspace?.scanPreviewData || null;
  const scanPreviewLoading = workspace?.scanPreviewLoading || false;
  const scanPreviewError = workspace?.scanPreviewError || '';
  const languageFocus = workspace?.languageFocus || DEFAULT_LANGUAGE_FOCUS;
  const scanPreview = useMemo(
    () => buildArtifactPreview(scanPreviewData, selectedScanPath),
    [scanPreviewData, selectedScanPath]
  );
  const scanPreviewTheory = useMemo(
    () => THEORY_OBJECT_RESEARCH_MAP[scanPreview?.theoryObject] || THEORY_OBJECT_RESEARCH_MAP.family_patch,
    [scanPreview?.theoryObject]
  );
  const showInTopRight = shouldShowResearchAssetInTopRight(scanPreview, selectedScanPath);
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  if (!showInTopRight) {
    return null;
  }

  return (
    <div style={cardStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline', marginBottom: 8 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff' }}>{`${scanPreview.typeLabel} | ${scanPreview.title}`}</div>
        <div style={{ fontSize: 10, color: '#7ea2c9' }}>{scanPreviewLoading ? '预览加载中...' : '预览就绪'}</div>
      </div>
      <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>{scanPreview.subtitle}</div>
      {scanPreviewError ? (
        <div style={{ marginTop: 8, fontSize: 11, color: '#ff9fb0' }}>{scanPreviewError}</div>
      ) : null}

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>理论映射</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>{scanPreviewTheory.summary}</div>
          <div>{`3D 关注点：${scanPreviewTheory.sceneHint}`}</div>
          <div>{`当前研究层：${languageFocus?.researchLayer || 'static_encoding'}`}</div>
        </div>
        <div style={{ display: 'grid', gap: 4 }}>
          {scanPreviewTheory.metrics.map((item) => (
            <div key={`artifact-data-${item.label}`} style={{ display: 'grid', gridTemplateColumns: '110px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
              <span>{item.label}</span>
              <span style={{ color: '#dbe9ff', fontWeight: 700 }}>{item.value}</span>
            </div>
          ))}
        </div>
        {scanPreview.analysisLines.map((line) => (
          <div key={`topright-line-${line}`} style={{ fontSize: 11, color: '#8fd4ff', lineHeight: 1.6 }}>
            {`• ${line}`}
          </div>
        ))}
      </div>
      {scanPreview.metricRows.length > 0 ? (
        <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 4 }}>
          <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>关键指标</div>
          {scanPreview.metricRows.map((item) => (
            <div key={`preview-metric-${item.label}`} style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
              <span>{item.label}</span>
              <span style={{ color: '#dbe9ff', fontWeight: 700 }}>{item.value}</span>
            </div>
          ))}
        </div>
      ) : null}

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>原始数据</div>
        <pre
          style={{
            margin: 0,
            maxHeight: compact ? 220 : 320,
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontSize: 11,
            color: '#cfe2ff',
            background: 'rgba(7, 12, 25, 0.82)',
            border: '1px solid rgba(122, 162, 255, 0.22)',
            borderRadius: 10,
            padding: 10,
          }}
        >
          {scanPreview?.rawJson || '暂无原始数据'}
        </pre>
      </div>
    </div>
  );
}

/* WaveRing 已移除 */

function PulseColumn({
  position = [0, 0, 0],
  color = '#ffffff',
  height = 1,
  radius = 0.06,
  speed = 1,
  phase = 0,
  opacity = 0.72,
}) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    const t = state.clock.elapsedTime * speed + phase;
    const sy = 0.84 + (Math.sin(t) + 1) * 0.24;
    ref.current.scale.set(1, sy, 1);
  });
  return (
    <mesh ref={ref} position={position}>
      <cylinderGeometry args={[radius, radius, height, 10]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.92} transparent opacity={opacity} />
    </mesh>
  );
}

function AppleNeuronAnimationOverlay({
  animationMode = 'none',
  nodes = [],
  selected = null,
  prediction = null,
  scanMechanismData = null,
}) {
  const coreNodes = useMemo(() => (Array.isArray(nodes) ? nodes.filter((node) => node?.role !== 'background') : []), [nodes]);
  const familyView = useMemo(
    () => buildFamilyPatchViewModel(coreNodes, selected, scanMechanismData),
    [coreNodes, scanMechanismData, selected]
  );

  const familyCenter = familyView.familyCenter;
  const conceptCenter = familyView.conceptCenter;
  const siblingCenter = familyView.siblingCenter;
  const routeNodes = coreNodes.filter((node) => node.role === 'route');
  const routeCenter = averagePosition(routeNodes, shiftPosition(conceptCenter, 0.6, 0.1, 1.4));
  const protocolCenter = shiftPosition(routeCenter, 2.8, 0.6, 1.2);
  const readoutPort = shiftPosition(protocolCenter, 3.2, 0.8, 0);
  const stagePath = [
    shiftPosition(familyCenter, -1.4, -0.8, -2.2),
    shiftPosition(conceptCenter, -0.4, 0.2, -0.6),
    shiftPosition(routeCenter, 0.4, -0.2, 0.9),
    protocolCenter,
  ];
  const successorPath = [
    shiftPosition(conceptCenter, -1.0, -0.5, -0.4),
    conceptCenter,
    shiftPosition(conceptCenter, 1.1, 0.55, 0.45),
    shiftPosition(conceptCenter, 2.2, 0.95, 1.1),
  ];
  const counterfactualPathA = [
    familyCenter,
    conceptCenter,
    shiftPosition(conceptCenter, 1.2, 0.7, 0.8),
    shiftPosition(conceptCenter, 2.2, 1.2, 1.2),
  ];
  const counterfactualPathB = [
    familyCenter,
    conceptCenter,
    shiftPosition(conceptCenter, 1.05, -0.75, -0.6),
    shiftPosition(conceptCenter, 2.0, -1.25, -1.1),
  ];
  const layerRelayNodes = coreNodes
    .filter((node) => node.role === 'query')
    .slice()
    .sort((a, b) => a.layer - b.layer)
    .filter((node, idx, arr) => idx === 0 || node.layer !== arr[idx - 1].layer)
    .slice(0, 5);
  const minimalWitness = familyView.selectedConceptMinimal?.subset_flat_indices
    ? familyView.instanceWitness.slice(0, Math.min(5, familyView.selectedConceptMinimal.subset_flat_indices.length))
    : familyView.instanceWitness.slice(0, 4);
  const siblingLabel = familyView.uniqueSiblingConcepts[0] || 'sibling';
  const animationLabel = APPLE_ANIMATION_OPTIONS.find((opt) => opt.id === animationMode)?.label || '动画';

  if (animationMode === 'none' || !selected) {
    return null;
  }

  return (
    <group>
      <Text position={shiftPosition(conceptCenter, 0, 1.45, -0.1)} color="#f8fafc" fontSize={0.17} anchorX="center" anchorY="middle">
        {animationLabel}
      </Text>
      {animationMode === 'family_patch_formation' && (
        <>
          {familyView.prototypeWitness.slice(0, 6).map((node, idx) => (
            <TheoryRunner
              key={`anim-family-form-${node.id}`}
              path={[node.position, blendPosition(node.position, familyCenter, 0.55), familyCenter]}
              color={idx < 2 ? '#ffffff' : '#7dd3fc'}
              size={0.065}
              speed={0.24 + idx * 0.02}
              phase={idx * 0.18}
            />
          ))}
        </>
      )}
      {animationMode === 'instance_offset' && (
        <>
          <Line points={[familyCenter, conceptCenter]} color="#f8b4ff" transparent opacity={0.9} lineWidth={2.4} />
          <TheoryRunner path={[familyCenter, conceptCenter]} color="#fff7ff" size={0.09} speed={0.44} phase={0.14} />
          <Text position={shiftPosition(conceptCenter, 0, 0.72, 0)} color="#f8b4ff" fontSize={0.14} anchorX="center" anchorY="middle">
            {'Δ concept'}
          </Text>
        </>
      )}
      {animationMode === 'attribute_fiber' && (
        <>
          <Line points={[shiftPosition(conceptCenter, -1.4, -0.7, 0), shiftPosition(conceptCenter, 1.4, 0.7, 0)]} color="#34d399" transparent opacity={0.88} lineWidth={2} />
          <Line points={[shiftPosition(conceptCenter, -1.4, 0.7, 0), shiftPosition(conceptCenter, 1.4, -0.7, 0)]} color="#60a5fa" transparent opacity={0.82} lineWidth={2} />
          <Line points={[shiftPosition(conceptCenter, 0, -1.0, -0.45), shiftPosition(conceptCenter, 0, 1.0, 0.45)]} color="#f59e0b" transparent opacity={0.8} lineWidth={2} />
          <TheoryRunner path={[shiftPosition(conceptCenter, -1.4, -0.7, 0), conceptCenter, shiftPosition(conceptCenter, 1.4, 0.7, 0)]} color="#34d399" size={0.07} speed={0.38} phase={0.08} />
          <TheoryRunner path={[shiftPosition(conceptCenter, -1.4, 0.7, 0), conceptCenter, shiftPosition(conceptCenter, 1.4, -0.7, 0)]} color="#60a5fa" size={0.07} speed={0.35} phase={0.32} />
          <TheoryBeacon position={conceptCenter} color="#f8b4ff" size={0.1} pulse={0.12} speed={1.5} phase={0.22} />
        </>
      )}
      {animationMode === 'successor_transport' && (
        <>
          <Line points={successorPath} color="#f59e0b" transparent opacity={0.92} lineWidth={2.2} />
          <TheoryRunner path={successorPath} color="#fff7d6" size={0.08} speed={0.46} phase={0.08} />
          <TheoryRunner path={successorPath} color="#f59e0b" size={0.07} speed={0.28} phase={0.42} />
        </>
      )}
      {animationMode === 'protocol_bridge' && (
        <>
          <Line points={[conceptCenter, routeCenter, protocolCenter, readoutPort]} color="#fde68a" transparent opacity={0.88} lineWidth={2.1} />
          <TheoryRunner path={[conceptCenter, routeCenter, protocolCenter, readoutPort]} color="#ffffff" size={0.08} speed={0.42} phase={0.16} />
        </>
      )}
      {animationMode === 'cross_layer_relay' && (
        <>
          {layerRelayNodes.map((node, idx) => (
            idx < layerRelayNodes.length - 1 ? (
              <Line key={`relay-line-${node.id}`} points={[node.position, layerRelayNodes[idx + 1].position]} color="#38bdf8" transparent opacity={0.42} lineWidth={1.5} />
            ) : null
          ))}
          {layerRelayNodes.length > 1 ? (
            <TheoryRunner path={layerRelayNodes.map((node) => node.position)} color="#dff6ff" size={0.07} speed={0.34} phase={0.12} />
          ) : null}
        </>
      )}
      {animationMode === 'ablation_shockwave' && (
        <>
          {familyView.instanceWitness.slice(0, 3).map((node) => (
            <Line key={`ablate-${node.id}`} points={[conceptCenter, node.position]} color="#fb7185" transparent opacity={0.6} lineWidth={1.6} />
          ))}
        </>
      )}
      {animationMode === 'counterfactual_split' && (
        <>
          <Line points={counterfactualPathA} color="#7dd3fc" transparent opacity={0.82} lineWidth={2} />
          <Line points={counterfactualPathB} color="#fb7185" transparent opacity={0.82} lineWidth={2} />
          <TheoryRunner path={counterfactualPathA} color="#dff6ff" size={0.07} speed={0.34} phase={0.08} />
          <TheoryRunner path={counterfactualPathB} color="#ffd5df" size={0.07} speed={0.34} phase={0.38} />
          <Text position={shiftPosition(counterfactualPathA[counterfactualPathA.length - 1], 0, 0.45, 0)} color="#dff6ff" fontSize={0.12}>{'actual'}</Text>
          <Text position={shiftPosition(counterfactualPathB[counterfactualPathB.length - 1], 0, -0.45, 0)} color="#ffd5df" fontSize={0.12}>{siblingLabel}</Text>
        </>
      )}
      {animationMode === 'minimal_circuit_peeloff' && (
        <>
          {minimalWitness.map((node, idx) => (
            <Line key={`minimal-${node.id}`} points={[conceptCenter, node.position]} color={idx < 2 ? '#ffffff' : '#f97316'} transparent opacity={0.72} lineWidth={idx < 2 ? 2.0 : 1.3} />
          ))}
        </>
      )}
      {animationMode === 'margin_breathing' && (
        <>
          <Line points={[familyCenter, siblingCenter]} color="#a7f3d0" transparent opacity={0.36} lineWidth={1.4} />
        </>
      )}
      {animationMode === 'offset_sparsity' && (
        <>
          {familyView.instanceWitness.slice(0, 6).map((node, idx) => (
            <PulseColumn
              key={`offset-sparse-${node.id}`}
              position={shiftPosition(conceptCenter, -0.7 + idx * 0.28, -1.0, 0)}
              color={idx < 3 ? '#f8b4ff' : '#c084fc'}
              height={0.42 + idx * 0.18}
              radius={0.045}
              speed={0.9 + idx * 0.12}
              phase={idx * 0.2}
              opacity={0.68}
            />
          ))}
        </>
      )}
      {animationMode === 'prototype_instance_tug' && (
        <>
          <Line points={[familyCenter, conceptCenter]} color="#7dd3fc" transparent opacity={0.72} lineWidth={2} />
          <Line points={[siblingCenter, conceptCenter]} color="#f8b4ff" transparent opacity={0.72} lineWidth={2} />
          <TheoryRunner path={[familyCenter, conceptCenter]} color="#dff6ff" size={0.07} speed={0.3} phase={0.12} />
          <TheoryRunner path={[siblingCenter, conceptCenter]} color="#f8b4ff" size={0.07} speed={0.3} phase={0.46} />
        </>
      )}
      {animationMode === 'stage_transition' && (
        <>
          <Line points={stagePath} color="#dff6ff" transparent opacity={0.38} lineWidth={1.6} />
          <TheoryRunner path={stagePath} color="#ffffff" size={0.07} speed={0.28} phase={0.18} />
        </>
      )}
    </group>
  );
}

function AppleNeuronFamilyPatchInspector({ workspace, compact = false }) {
  const nodes = workspace?.nodes || [];
  const selected = workspace?.selected || null;
  const currentTheoryObject = workspace?.currentTheoryObject || null;
  const scanMechanismData = workspace?.scanMechanismData || null;
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;
  const familyPatchView = useMemo(
    () => buildFamilyPatchViewModel(nodes, selected, scanMechanismData),
    [nodes, scanMechanismData, selected]
  );

  if (!selected || !['family_patch', 'concept_section'].includes(currentTheoryObject?.id || '')) {
    return null;
  }

  const minimal = familyPatchView.selectedConceptMinimal;
  const counterfactualList = familyPatchView.selectedConceptCounterfactuals || [];
  const firstCounterfactual = counterfactualList[0] || null;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>family patch 分解视图</div>
      <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7, display: 'grid', gap: 4 }}>
        <div>{`概念: ${selected?.concept || selected?.label || '-'}`}</div>
        <div>{`类别: ${selected?.category || '未分类'}`}</div>
        <div>{`family 节点数: ${familyPatchView.familyNodes.length}`}</div>
        <div>{`实例节点数: ${familyPatchView.conceptNodes.length}`}</div>
        <div>{`同族兄弟概念: ${familyPatchView.uniqueSiblingConcepts.length}`}</div>
        <div>{`offset 几何长度: ${familyPatchView.offsetNorm.toFixed(3)}`}</div>
      </div>

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>数学解释</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>{`prototype: B_${selected?.category || 'family'}`}</div>
          <div>{`instance: Δ_${selected?.concept || selected?.label || 'c'}`}</div>
          <div>{'state ≈ family prototype + instance offset + attribute/context corrections'}</div>
        </div>
        <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6 }}>
          {`蓝色 ring 表示家族原型核，粉色 ring 表示当前概念的实例偏移核，浅绿色 ring 表示同族兄弟概念的相对中心。`}
        </div>
      </div>

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>神经元见证</div>
        <div style={{ display: 'grid', gap: 4 }}>
          {familyPatchView.prototypeWitness.length > 0 ? (
            familyPatchView.prototypeWitness.slice(0, 4).map((node) => (
              <div key={`family-proto-row-${node.id}`} style={{ display: 'grid', gridTemplateColumns: '88px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
                <span>prototype</span>
                <span style={{ color: '#dbe9ff', fontWeight: 700 }}>{`${node.label} | L${node.layer} N${node.neuron}`}</span>
              </div>
            ))
          ) : (
            <div style={{ fontSize: 11, color: '#7ea2c9' }}>当前没有可用的 family witness 节点。</div>
          )}
          {familyPatchView.instanceWitness.length > 0 ? (
            familyPatchView.instanceWitness.slice(0, 4).map((node) => (
              <div key={`family-inst-row-${node.id}`} style={{ display: 'grid', gridTemplateColumns: '88px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
                <span>instance</span>
                <span style={{ color: '#f5d0fe', fontWeight: 700 }}>{`${node.label} | L${node.layer} N${node.neuron}`}</span>
              </div>
            ))
          ) : null}
        </div>
      </div>

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>因果证据</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>{`最小回路: ${minimal ? `subset=${toSafeNumber(minimal?.subset_size, 0)} | recovery=${toSafeNumber(minimal?.recovery_ratio, 0).toFixed(3)}` : '未导入'}`}</div>
          <div>{`反事实对: ${counterfactualList.length}`}</div>
          {firstCounterfactual ? (
            <div>{`首个反事实: ${firstCounterfactual?.noun || '-'} -> ${firstCounterfactual?.counterfactual_noun || '-'} | margin=${toSafeNumber(firstCounterfactual?.specificity_margin_seq_logprob, 0).toFixed(6)}`}</div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function AppleNeuronSelectedLegendPanels({ workspace, compact = false }) {
  const selected = workspace?.selected || null;
  const summary = workspace?.summary || {};
  const displayStrategy = workspace?.displayStrategy || 'auto';
  const setDisplayStrategy = workspace?.setDisplayStrategy || (() => {});
  const manualDisplayGroups = workspace?.manualDisplayGroups || {};
  const setManualDisplayGroups = workspace?.setManualDisplayGroups || (() => {});
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <AppleNeuronFamilyPatchInspector workspace={workspace} compact={compact} />

      <div style={{ ...cardStyle, minHeight: 160 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>选中神经元</div>
        {selected ? (
          <div style={{ fontSize: 12, color: '#9eb4dd', display: 'grid', gap: 6 }}>
            <div style={{ color: '#e5eeff', fontWeight: 700 }}>{selected.label}</div>
            <div>{`角色: ${selected.role}`}</div>
            {'fruit' in selected ? <div>{`水果: ${selected.fruit}`}</div> : null}
            {'concept' in selected ? <div>{`概念: ${selected.concept}`}</div> : null}
            {'category' in selected ? <div>{`类别: ${selected.category}`}</div> : null}
            <div>{`层 / 神经元: L${selected.layer} / N${selected.neuron}`}</div>
            <div>{`强度: ${selected.strength.toExponential(3)}`}</div>
            <div>{`${selected.metric}: ${selected.value.toExponential(3)}`}</div>
            <div style={{ color: '#6f84ad' }}>{`来源: ${selected.source}`}</div>
          </div>
        ) : (
          <div style={{ fontSize: 12, color: '#7d93bd' }}>请在 3D 场景中点击高亮神经元。</div>
        )}
      </div>

      <div style={{ ...cardStyle, fontSize: 12, color: '#9eb4dd', lineHeight: 1.7 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>显示与降噪策略</div>
        <div style={{ display: 'grid', gridTemplateColumns: compact ? '1fr' : 'repeat(3, 1fr)', gap: 6 }}>
          {[
            { id: 'auto', label: '自动聚焦', desc: '随分析类型切换重点' },
            { id: 'all', label: '全部显示', desc: '不过滤任何节点' },
            { id: 'manual', label: '手动筛选', desc: '按类别开关显示' },
          ].map((opt) => (
            <button
              key={`legend-display-${opt.id}`}
              type="button"
              onClick={() => setDisplayStrategy(opt.id)}
              title={opt.desc}
              style={{
                borderRadius: 8,
                border: `1px solid ${displayStrategy === opt.id ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.35)'}`,
                background: displayStrategy === opt.id ? 'rgba(24, 101, 134, 0.38)' : 'rgba(7, 12, 25, 0.82)',
                color: '#dbe9ff',
                fontSize: 11,
                padding: '7px 8px',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
        {displayStrategy === 'manual' ? (
          <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
            {[
              { id: 'core', label: '核心/基础节点' },
              { id: 'query', label: '输入概念节点' },
              { id: 'multidim', label: '多维编码节点' },
              { id: 'hard', label: '硬伤实验节点' },
              { id: 'unified', label: '统一解码节点' },
              { id: 'background', label: '背景网络节点' },
            ].map((item) => (
              <label key={`legend-manual-group-${item.id}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: '#9eb4dd' }}>
                <input
                  type="checkbox"
                  checked={manualDisplayGroups[item.id] !== false}
                  onChange={(e) => setManualDisplayGroups((prev) => ({ ...prev, [item.id]: e.target.checked }))}
                />
                <span>{item.label}</span>
              </label>
            ))}
          </div>
        ) : null}
        <div style={{ marginTop: 8, fontSize: 11, color: '#7ea2c9' }}>
          {displayStrategy === 'auto'
            ? '自动模式：因果类分析突出硬伤实验，结构类分析突出统一编码与多维节点。'
            : displayStrategy === 'all'
              ? '全部模式：所有节点同等显示，不做降噪。'
              : '手动模式：按勾选结果控制各类节点显示。'}
        </div>
      </div>

      <div style={{ ...cardStyle, fontSize: 12, color: '#9eb4dd', lineHeight: 1.7 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>图例</div>
        <div><span style={{ color: ROLE_COLORS.micro }}>●</span> 微观神经元</div>
        <div><span style={{ color: ROLE_COLORS.macro }}>●</span> 中观神经元</div>
        <div><span style={{ color: ROLE_COLORS.route }}>●</span> 共享路径神经元</div>
        <div><span style={{ color: ROLE_COLORS.fruitGeneral }}>●</span> 类别通用神经元</div>
        <div><span style={{ color: ROLE_COLORS.style }}>●</span> 风格维度神经元</div>
        <div><span style={{ color: ROLE_COLORS.logic }}>●</span> 逻辑维度神经元</div>
        <div><span style={{ color: ROLE_COLORS.syntax }}>●</span> 句法维度神经元</div>
        <div><span style={{ color: ROLE_COLORS.hardBinding }}>●</span> 硬伤实验-动态绑定</div>
        <div><span style={{ color: ROLE_COLORS.hardLong }}>●</span> 硬伤实验-长程链路</div>
        <div><span style={{ color: ROLE_COLORS.hardLocal }}>●</span> 硬伤实验-局部信用</div>
        <div><span style={{ color: ROLE_COLORS.hardTriplet }}>●</span> 硬伤实验-三元组定向因果</div>
        <div><span style={{ color: ROLE_COLORS.unifiedDecode }}>●</span> 统一解码节点</div>
        <div><span style={{ color: '#84f1ff' }}>●</span> 输入概念神经元</div>
        <div><span style={{ color: ROLE_COLORS.background }}>●</span> 背景网络采样</div>
        <div style={{ color: '#6f84ad', marginTop: 8 }}>
          {`核心集合: ${(summary.micro || 0) + (summary.macro || 0) + (summary.route || 0)} | 多维集合: ${summary.multidimNodes || 0} | 硬伤: ${summary.hardProblemNodes || 0} | 统一解码: ${summary.unifiedDecodeNodes || 0}`}
        </div>
      </div>
    </div>
  );
}

export function AppleNeuronCategoryComparePanel({ workspace, compact = false }) {
  const summary = workspace?.summary || {};
  const querySets = workspace?.querySets || [];
  const categoryStats = summary?.categoryStats || {};
  const categoryRows = Object.entries(categoryStats)
    .map(([name, stat]) => ({ name, ...stat }))
    .sort((a, b) => b.neurons - a.neurons);
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>类别神经元比较</div>
      <div style={{ fontSize: 12, color: '#92a6cc', lineHeight: 1.6 }}>
        {`总神经元 ${summary.total || 0} | 查询神经元 ${summary.query || 0} | 类别数 ${categoryRows.length}`}
      </div>
      <div style={{ fontSize: 11, color: '#7ea2c9', marginTop: 4 }}>
        {`当前词元: ${summary.currentToken || '-'} (${((summary.currentTokenProb || 0) * 100).toFixed(1)}%) | 多维节点: ${summary.multidimNodes || 0}`}
      </div>
      <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
        {categoryRows.length === 0 ? (
          <div style={{ fontSize: 11, color: '#6f84ad' }}>暂无类别数据，请在左侧输入概念和类别后生成。</div>
        ) : (
          categoryRows.map((row) => (
            <div key={`cat-${row.name}`} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#9eb4dd' }}>
              <span>{row.name}</span>
              <span>{`${row.concepts} 概念 / ${row.neurons} 神经元`}</span>
            </div>
          ))
        )}
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: '#6f84ad' }}>{`已生成概念集: ${querySets.length}`}</div>
    </div>
  );
}

export function AppleNeuronCompareFilterPanel({ workspace, compact = false }) {
  const querySets = workspace?.querySets || [];
  const queryVisibility = workspace?.queryVisibility || {};
  const setQuerySetVisible = workspace?.setQuerySetVisible;
  const setAllQuerySetVisible = workspace?.setAllQuerySetVisible;
  const summary = workspace?.summary || {};
  const visibleCount = querySets.filter((set) => queryVisibility[set.id] !== false).length;
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>Compare Filter</div>
      <div style={{ fontSize: 12, color: '#92a6cc', lineHeight: 1.6 }}>
        {`按输入名称筛选：已显示 ${visibleCount}/${querySets.length}`}
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        <button type="button" onClick={() => setAllQuerySetVisible?.(true)} style={smallActionButtonStyle}>全选</button>
        <button type="button" onClick={() => setAllQuerySetVisible?.(false)} style={smallActionButtonStyle}>全不选</button>
      </div>
      <div style={{ marginTop: 10, display: 'grid', gap: 8, maxHeight: compact ? 180 : 220, overflowY: 'auto' }}>
        {querySets.length === 0 ? (
          <div style={{ fontSize: 11, color: '#6f84ad' }}>暂无输入名称。请先在左侧生成概念神经元。</div>
        ) : (
          querySets.map((set) => (
            <label key={`qf-${set.id}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd' }}>
              <input
                type="checkbox"
                checked={queryVisibility[set.id] !== false}
                onChange={(e) => setQuerySetVisible?.(set.id, e.target.checked)}
              />
              <span style={{ color: set.color }}>●</span>
              <span>{`${set.name} [${set.category}] (${set.nodes.length})`}</span>
            </label>
          ))
        )}
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: '#6f84ad' }}>
        {`当前词元: ${summary.currentToken || '-'} (${((summary.currentTokenProb || 0) * 100).toFixed(1)}%)`}
      </div>
    </div>
  );
}

export function AppleNeuronGeneratedConceptSetsPanel({ workspace, compact = false }) {
  const querySets = workspace?.querySets || [];
  const queryVisibility = workspace?.queryVisibility || {};
  const setQuerySetVisible = workspace?.setQuerySetVisible;
  const setAllQuerySetVisible = workspace?.setAllQuerySetVisible;
  const removeQuerySet = workspace?.removeQuerySet;
  const visibleCount = querySets.filter((set) => queryVisibility[set.id] !== false).length;
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>已生成概念集</div>
      <div style={{ fontSize: 11, color: '#92a6cc', lineHeight: 1.6 }}>
        {`当前共 ${querySets.length} 组概念，其中显示 ${visibleCount} 组。这里统一处理显隐和清理。`}
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        <button type="button" onClick={() => setAllQuerySetVisible?.(true)} style={smallActionButtonStyle}>全显示</button>
        <button type="button" onClick={() => setAllQuerySetVisible?.(false)} style={smallActionButtonStyle}>全隐藏</button>
      </div>
      <div style={{ marginTop: 10, display: 'grid', gap: 8, maxHeight: compact ? 220 : 280, overflowY: 'auto' }}>
        {querySets.length === 0 ? (
          <div style={{ fontSize: 11, color: '#6f84ad' }}>暂未生成概念集，请先在左侧输入名词和类别。</div>
        ) : (
          querySets.map((set) => (
            <div key={`generated-set-${set.id}`} style={{ display: 'grid', gridTemplateColumns: '20px 1fr auto', gap: 8, alignItems: 'center', fontSize: 12, color: '#9eb4dd' }}>
              <input
                type="checkbox"
                checked={queryVisibility[set.id] !== false}
                onChange={(e) => setQuerySetVisible?.(set.id, e.target.checked)}
              />
              <span style={{ overflowWrap: 'anywhere' }}>
                <span style={{ color: set.color }}>●</span>
                {` ${set.name} [${set.category}] (${set.nodes.length})`}
              </span>
              <button
                type="button"
                onClick={() => removeQuerySet?.(set.id)}
                style={{ ...smallActionButtonStyle, padding: '2px 8px', fontSize: 11 }}
              >
                删除
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export function AppleNeuronMultidimSettingsPanel({ workspace, compact = false }) {
  const multidimProbeData = workspace?.multidimProbeData || null;
  const multidimCausalData = workspace?.multidimCausalData || null;
  const multidimTopN = workspace?.multidimTopN ?? 96;
  const setMultidimTopN = workspace?.setMultidimTopN;
  const multidimVisible = workspace?.multidimVisible || { style: true, logic: true, syntax: true };
  const setMultidimVisible = workspace?.setMultidimVisible;
  const multidimActiveDimension = workspace?.multidimActiveDimension || 'style';
  const setMultidimActiveDimension = workspace?.setMultidimActiveDimension;
  const multidimLayerProfile = workspace?.multidimLayerProfile || [];
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>三维编码设置</div>
      <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 10 }}>
        管理 `style / logic / syntax` 三维探针的可见性、TopN 和当前显示维度。
      </div>
      <div style={{ display: 'grid', gap: 8 }}>
        <div style={{ fontSize: 11, color: '#7ea2c9' }}>
          {multidimProbeData
            ? `已导入探针，当前显示维度: ${DIMENSION_LABELS[multidimActiveDimension] || multidimActiveDimension}，层谱点数: ${multidimLayerProfile?.length || 0}`
            : '未导入三维探针 JSON（multidim_encoding_probe.json）'}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
          <div style={{ fontSize: 12, color: '#9eb4dd' }}>TopN</div>
          <input
            type="number"
            min={16}
            max={256}
            value={multidimTopN}
            onChange={(e) => setMultidimTopN?.(Number(e.target.value))}
            style={inputStyle}
          />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
          {['style', 'logic', 'syntax'].map((dim) => (
            <button
              key={`multidim-panel-${dim}`}
              type="button"
              onClick={() => setMultidimActiveDimension?.(dim)}
              style={{
                borderRadius: 8,
                border: `1px solid ${multidimActiveDimension === dim ? ROLE_COLORS[dim] : 'rgba(122,162,255,0.35)'}`,
                background: multidimActiveDimension === dim ? 'rgba(42,71,132,0.82)' : 'rgba(7, 12, 25, 0.82)',
                color: '#dbe9ff',
                fontSize: 11,
                padding: '6px 8px',
                cursor: 'pointer',
              }}
            >
              {DIMENSION_LABELS[dim]}
            </button>
          ))}
        </div>
        <div style={{ display: 'grid', gap: 6 }}>
          {['style', 'logic', 'syntax'].map((dim) => (
            <label key={`multidim-vis-${dim}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd' }}>
              <input
                type="checkbox"
                checked={multidimVisible[dim] !== false}
                onChange={(e) => setMultidimVisible?.((prev) => ({ ...prev, [dim]: e.target.checked }))}
              />
              <span style={{ color: ROLE_COLORS[dim] }}>●</span>
              <span>{`${DIMENSION_LABELS[dim]}神经元`}</span>
            </label>
          ))}
        </div>
        <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6 }}>
          {multidimCausalData
            ? `对角优势 style=${toSafeNumber(multidimCausalData?.diagonal_advantage?.style, 0).toFixed(4)} / logic=${toSafeNumber(multidimCausalData?.diagonal_advantage?.logic, 0).toFixed(4)} / syntax=${toSafeNumber(multidimCausalData?.diagonal_advantage?.syntax, 0).toFixed(4)}`
            : '未导入三维因果消融 JSON（multidim_causal_ablation.json）'}
        </div>
      </div>
    </div>
  );
}

