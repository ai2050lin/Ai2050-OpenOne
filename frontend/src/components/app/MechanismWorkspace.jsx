import {
  Activity,
  ArrowUpRight,
  BarChart3,
  ChevronDown,
  ChevronUp,
  Eye,
  FileSearch,
  GitCompareArrows,
  Presentation,
  ScanSearch,
  Wrench,
} from 'lucide-react';
import { useMemo, useState } from 'react';

import { useResearchSnapshot } from '../../researchKernel/useResearchSnapshot';

import './MechanismWorkspace.css';

const MECHANISM_WORKSPACE_MODES = [
  { id: 'observe', label: '机制观察', icon: Eye, detail: '逐层观察状态、组件和候选竞争' },
  { id: 'compare', label: '证据比较', icon: GitCompareArrows, detail: '比较原始、反事实和干预结果' },
  { id: 'present', label: '成果展示', icon: Presentation, detail: '只显示达到声明证据等级的结果' },
];

const DOCK_TABS = [
  { id: 'candidates', label: '候选竞争', icon: BarChart3 },
  { id: 'compare', label: '反事实', icon: GitCompareArrows },
  { id: 'interventions', label: '干预', icon: Wrench },
  { id: 'provenance', label: '证据来源', icon: FileSearch },
];

function toNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function displayText(value, fallback = '-') {
  if (value == null || value === '') return fallback;
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (Array.isArray(value)) {
    const text = value
      .map((item) => displayText(item?.text ?? item?.label ?? item, ''))
      .filter(Boolean)
      .join(' ');
    return text || fallback;
  }
  return displayText(value.text ?? value.label ?? value.name ?? value.id, fallback);
}

function normalizeScores(candidateField) {
  const scores = Array.isArray(candidateField?.scores)
    ? candidateField.scores
    : Array.isArray(candidateField?.candidates)
      ? candidateField.candidates
      : [];
  return scores
    .map((item, index) => ({
      label: displayText(item?.label ?? item?.token ?? item?.name, `候选 ${index + 1}`),
      score: toNumber(item?.score ?? item?.logit ?? item?.margin ?? item, 0),
      probability: toNumber(item?.probability ?? item?.prob, null),
      role: item?.role || null,
    }))
    .sort((left, right) => right.score - left.score);
}

function candidateSnapshots(mechanismCase, trace, forwardData) {
  const registered = asArray(mechanismCase?.candidate_margins)
    .filter((snapshot) => snapshot && typeof snapshot === 'object');
  if (registered.length) {
    return registered.map((snapshot, index) => ({
      layer: toNumber(snapshot.layer, index),
      eventType: snapshot.event_type || 'registered_candidate_margin',
      field: {
        ...snapshot,
        scores: snapshot.scores || snapshot.candidates,
      },
      source: 'mechanism_case',
    }));
  }

  const traceSnapshots = asArray(trace?.events)
    .filter((event) => event?.candidate_field)
    .map((event) => ({
      layer: toNumber(event.layer, -1),
      eventType: event.event_type || event.component || 'trace_event',
      field: event.candidate_field,
      source: 'real_component_trace',
    }));
  if (traceSnapshots.length) return traceSnapshots;

  return asArray(forwardData?.layers)
    .filter((layer) => layer?.candidate_field)
    .map((layer) => ({
      layer: toNumber(layer.layer, -1),
      eventType: 'forward_layer',
      field: layer.candidate_field,
      source: 'forward_data',
    }));
}

function nearestSnapshot(snapshots, currentLayer) {
  if (!snapshots.length) return null;
  if (currentLayer == null) return snapshots[snapshots.length - 1];
  return snapshots.reduce((best, snapshot) => {
    const distance = Math.abs(snapshot.layer - currentLayer);
    return !best || distance < best.distance ? { snapshot, distance } : best;
  }, null)?.snapshot || null;
}

function displayMargin(field, scores) {
  const explicit = toNumber(field?.margin, null);
  if (explicit != null) return explicit;
  if (scores.length < 2) return null;
  return scores[0].score - scores[1].score;
}

export function MechanismModeSwitch({ mode, onModeChange }) {
  return (
    <section className="mechanism-mode-switch" aria-label="主空间研究模式">
      <div className="mechanism-mode-switch__label">
        <ScanSearch size={13} />主空间模式
      </div>
      <div className="mechanism-mode-switch__options">
        {MECHANISM_WORKSPACE_MODES.map((option) => {
          const Icon = option.icon;
          return (
            <button
              key={option.id}
              type="button"
              className={mode === option.id ? 'is-active' : ''}
              onClick={() => onModeChange?.(option.id)}
              title={option.detail}
            >
              <Icon size={13} />
              <span>{option.label}</span>
            </button>
          );
        })}
      </div>
      <p>{MECHANISM_WORKSPACE_MODES.find((option) => option.id === mode)?.detail}</p>
    </section>
  );
}

function CandidateView({ snapshot }) {
  if (!snapshot) {
    return <div className="mechanism-dock__empty">当前结果没有候选分数。运行真实Trace或加载包含candidate_margins的MechanismCase。</div>;
  }

  const scores = normalizeScores(snapshot.field);
  const maxMagnitude = Math.max(1, ...scores.map((item) => Math.abs(item.score)));
  const margin = displayMargin(snapshot.field, scores);
  const target = displayText(snapshot.field?.target_label ?? snapshot.field?.target, '');
  const competitor = displayText(snapshot.field?.competitor_label ?? snapshot.field?.competitor, '');

  return (
    <div className="mechanism-candidates">
      <div className="mechanism-candidates__summary">
        <span>L{snapshot.layer} · {snapshot.eventType}</span>
        <strong className={margin != null && margin >= 0 ? 'is-positive' : 'is-negative'}>
          Margin {margin == null ? '-' : margin.toFixed(3)}
        </strong>
        <span>{target ? `目标 ${target}` : '目标未声明'}{competitor ? ` / 对手 ${competitor}` : ''}</span>
      </div>
      <div className="mechanism-candidates__list">
        {scores.slice(0, 7).map((candidate, index) => (
          <div className="mechanism-candidate" key={`${candidate.label}:${index}`}>
            <span>{index + 1}</span>
            <strong>{candidate.label}</strong>
            <i><b style={{ width: `${Math.max(3, Math.abs(candidate.score) / maxMagnitude * 100)}%` }} /></i>
            <output>{candidate.score.toFixed(3)}</output>
            <small>{candidate.probability == null ? '' : `${(candidate.probability * 100).toFixed(1)}%`}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function CompareView({ mechanismCase }) {
  const counterfactual = mechanismCase?.counterfactual;
  const controls = Array.isArray(mechanismCase?.negative_controls) ? mechanismCase.negative_controls : [];
  if (!counterfactual && !controls.length) {
    return <div className="mechanism-dock__empty">尚未加载反事实或matched-null负控。当前页面只展示自然运行，不能据此判断条件效应。</div>;
  }
  return (
    <div className="mechanism-compare">
      <section>
        <span>原始样本</span>
        <strong>{displayText(mechanismCase?.sample?.prompt ?? mechanismCase?.sample?.text, '已登记样本')}</strong>
      </section>
      <ArrowUpRight size={18} />
      <section>
        <span>反事实</span>
        <strong>{displayText(counterfactual?.prompt ?? counterfactual?.text ?? counterfactual?.label, '已登记反事实')}</strong>
      </section>
      <section className="mechanism-compare__controls">
        <span>匹配负控</span>
        <strong>{controls.length} 组</strong>
      </section>
    </div>
  );
}

function InterventionView({ mechanismCase }) {
  const interventions = asArray(mechanismCase?.interventions)
    .filter((item) => item && typeof item === 'object');
  if (!interventions.length) {
    return <div className="mechanism-dock__empty">当前数据没有Ablation、Patch或Restore记录。观测轨迹不会被显示为因果路径。</div>;
  }
  return (
    <div className="mechanism-interventions">
      {interventions.slice(0, 8).map((item, index) => (
        <div key={item.id || index}>
          <span>{displayText(item?.type ?? item?.operation, 'intervention')}</span>
          <strong>{displayText(item?.component ?? item?.target, `记录 ${index + 1}`)}</strong>
          <output>{toNumber(item?.margin_delta ?? item?.effect, 0).toFixed(3)}</output>
          <small>{displayText(item?.status ?? item?.evidence_status, '未标记')}</small>
        </div>
      ))}
    </div>
  );
}

function ProvenanceView({ mechanismCase, activeFileMeta, trace }) {
  const { snapshot, error } = useResearchSnapshot();
  const current = snapshot?.current;
  const rows = [
    ['Case', mechanismCase?.case_id || trace?.run_id || activeFileMeta?.id || '-'],
    ['模型', mechanismCase?.model || trace?.model || activeFileMeta?.model || '-'],
    ['结果类型', mechanismCase?.result_type || activeFileMeta?.result_type || activeFileMeta?.type || '内部状态观察'],
    ['证据等级', mechanismCase?.evidence_level || '自然观测 / 未声明'],
    ['状态', mechanismCase?.status || 'exploratory'],
    ['协议冻结', mechanismCase?.protocol_frozen === true ? '是' : '未声明'],
    ['数据来源', activeFileMeta?.label || activeFileMeta?.filename || trace?.run_id || '内置真实Trace'],
  ];
  return (
    <dl className="mechanism-provenance">
      {rows.map(([label, value]) => (
        <div key={label}><dt>{label}</dt><dd>{String(value)}</dd></div>
      ))}
      <div className="mechanism-provenance__boundary">
        <dt>证据边界</dt>
        <dd>{mechanismCase?.evidence_boundary || current?.bottleneck || error || 'Canonical Snapshot 读取中'}</dd>
      </div>
    </dl>
  );
}

export function MechanismWorkspaceDock({
  mode,
  currentLayer,
  mechanismCase,
  trace,
  forwardData,
  activeFileMeta,
  hidden = false,
}) {
  const [expanded, setExpanded] = useState(false);
  const [tab, setTab] = useState('candidates');
  const snapshots = useMemo(
    () => candidateSnapshots(mechanismCase, trace, forwardData),
    [forwardData, mechanismCase, trace]
  );
  const snapshot = useMemo(
    () => nearestSnapshot(snapshots, currentLayer),
    [currentLayer, snapshots]
  );
  const scores = normalizeScores(snapshot?.field);
  const margin = displayMargin(snapshot?.field, scores);

  if (hidden) return null;

  return (
    <aside className={`mechanism-dock ${expanded ? 'is-expanded' : ''}`} aria-label="机制观察详情">
      <button type="button" className="mechanism-dock__summary" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}>
        <Activity size={14} />
        <strong>{MECHANISM_WORKSPACE_MODES.find((item) => item.id === mode)?.label || '机制观察'}</strong>
        <span>{currentLayer == null ? 'Layer未播放' : `当前 L${currentLayer}`}</span>
        <span>{scores[0] ? `领先 ${scores[0].label}` : '候选数据未加载'}</span>
        <output className={margin != null && margin >= 0 ? 'is-positive' : 'is-negative'}>
          {margin == null ? 'Margin -' : `Margin ${margin.toFixed(3)}`}
        </output>
        {expanded ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
      </button>

      {expanded && (
        <div className="mechanism-dock__body">
          <nav aria-label="机制详情栏目">
            {DOCK_TABS.map((item) => {
              const Icon = item.icon;
              return (
                <button key={item.id} type="button" className={tab === item.id ? 'is-active' : ''} onClick={() => setTab(item.id)}>
                  <Icon size={13} />{item.label}
                </button>
              );
            })}
          </nav>
          <div className="mechanism-dock__content">
            {tab === 'candidates' && <CandidateView snapshot={snapshot} />}
            {tab === 'compare' && <CompareView mechanismCase={mechanismCase} />}
            {tab === 'interventions' && <InterventionView mechanismCase={mechanismCase} />}
            {tab === 'provenance' && <ProvenanceView mechanismCase={mechanismCase} activeFileMeta={activeFileMeta} trace={trace} />}
          </div>
        </div>
      )}
    </aside>
  );
}
