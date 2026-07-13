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
    ['跨模型行为必要性', formatNumber(metrics.phase330_cross_model_behavior_necessity_mechanism_count)],
    ['单神经元因果', formatNumber(metrics.single_unit_causal_count)],
    ['完整自然链', formatNumber(metrics.full_natural_chain_pass_count)],
  ];

  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.7 }}>
        {phase402.status
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
