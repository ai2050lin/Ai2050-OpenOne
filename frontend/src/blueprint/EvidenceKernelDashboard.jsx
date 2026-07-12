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
    ['跨模型行为必要性', formatNumber(metrics.phase330_cross_model_behavior_necessity_mechanism_count)],
    ['单神经元因果', formatNumber(metrics.single_unit_causal_count)],
    ['完整自然链', formatNumber(metrics.full_natural_chain_pass_count)],
  ];

  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.7 }}>
        当前图谱已经完成九个模式族和三个模型的统一物理覆盖，并继续推进到 Phase {atlas?.phase || '-'} 的自然必要性审计。严格边界仍然是：局部传播已经出现，但跨模型行为必要性、单神经元因果和完整自然闭合均未通过。
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 8 }}>
        {rows.map(([label, value]) => (
          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, padding: '8px 10px', borderBottom: '1px solid rgba(148,163,184,0.12)' }}>
            <span style={{ color: '#94a3b8', fontSize: 11 }}>{label}</span>
            <strong style={{ color: value === '0' ? '#fda4af' : '#e2e8f0', fontSize: 11, fontFamily: 'monospace' }}>{value}</strong>
          </div>
        ))}
      </div>
      <div style={{ color: '#fbbf24', fontSize: 11, lineHeight: 1.65 }}>{atlas?.evidence_boundary}</div>
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
    ['局部传播通过', formatNumber(metrics.phase334_local_propagation_pass_count)],
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
