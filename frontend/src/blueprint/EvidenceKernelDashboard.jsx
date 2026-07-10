import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Database,
  ExternalLink,
  FlaskConical,
  Target,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

const KERNEL_BASE = '/vis_data/research_kernel';

const MODULES = [
  { id: 'progress', title: '总体证据进度', icon: Target, color: '#22d3ee' },
  { id: 'claims', title: '机制主张', icon: CheckCircle2, color: '#34d399' },
  { id: 'runs', title: '实验运行', icon: FlaskConical, color: '#60a5fa' },
  { id: 'gaps', title: '开放缺口', icon: AlertTriangle, color: '#f59e0b' },
];

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

function ProgressDetail({ progress }) {
  const dimensions = Object.entries(progress?.dimensions || {});
  return (
    <div style={{ display: 'grid', gap: 10 }}>
      {dimensions.map(([id, row]) => {
        const ratio = Math.max(0, Math.min(1, Number(row.ratio || 0)));
        return (
          <div key={id}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: 12 }}>
              <span style={{ color: '#dbeafe' }}>{id}</span>
              <span style={{ color: '#94a3b8' }}>{row.valid}/{row.required} · {(ratio * 100).toFixed(1)}%</span>
            </div>
            <div style={{ height: 5, marginTop: 5, background: 'rgba(148,163,184,0.13)', borderRadius: 2 }}>
              <div style={{ width: `${ratio * 100}%`, height: '100%', background: ratio === 1 ? '#34d399' : '#22d3ee', borderRadius: 2 }} />
            </div>
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
        <div key={run.run_id} style={{ display: 'grid', gridTemplateColumns: 'minmax(180px, 1fr) auto', gap: 12, paddingBottom: 9, borderBottom: '1px solid rgba(148,163,184,0.12)' }}>
          <div>
            <div style={{ color: '#dbeafe', fontSize: 12 }}>{run.run_id}</div>
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
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
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
  const [claims, setClaims] = useState([]);
  const [gaps, setGaps] = useState([]);
  const [active, setActive] = useState('progress');
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    Promise.all([
      fetchJson(`${KERNEL_BASE}/manifest.json`),
      fetchJsonl(`${KERNEL_BASE}/claims.jsonl`),
      fetchJsonl(`${KERNEL_BASE}/gaps.jsonl`),
    ]).then(([nextManifest, nextClaims, nextGaps]) => {
      if (!mounted) return;
      setManifest(nextManifest);
      setClaims(nextClaims);
      setGaps(nextGaps);
      setError('');
    }).catch((reason) => {
      if (mounted) setError(reason?.message || 'Evidence Kernel 加载失败');
    });
    return () => { mounted = false; };
  }, []);

  const summaries = useMemo(() => ({
    progress: `${Object.keys(manifest?.progress?.dimensions || {}).length} 个独立分母`,
    claims: `${claims.length} 条主张`,
    runs: `${manifest?.runs?.length || 0} 个可追溯运行`,
    gaps: `${gaps.filter((gap) => gap.status === 'open').length} 个开放缺口`,
  }), [claims.length, gaps, manifest]);

  return (
    <section style={{ marginBottom: 28 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 12 }}>
        <Database size={20} color="#22d3ee" />
        <div>
          <h2 style={{ margin: 0, fontSize: 18, color: '#e0f2fe' }}>Evidence Kernel / 统一证据内核</h2>
          <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 3 }}>所有数字均由有效记录数与明确目标数计算；单提示成功不计为机制闭合。</div>
        </div>
      </div>
      {error && <div style={{ color: '#fb7185', fontSize: 12, marginBottom: 10 }}>{error}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8 }}>
        {MODULES.map((module) => {
          const Icon = module.icon;
          const selected = active === module.id;
          return (
            <button
              key={module.id}
              type="button"
              onClick={() => setActive(selected ? '' : module.id)}
              style={{
                padding: 12,
                minHeight: 76,
                textAlign: 'left',
                color: '#e2e8f0',
                border: `1px solid ${selected ? module.color : 'rgba(148,163,184,0.16)'}`,
                background: selected ? 'rgba(8,47,73,0.35)' : 'rgba(15,23,42,0.58)',
                borderRadius: 6,
                cursor: 'pointer',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Icon size={17} color={module.color} />
                {selected ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
              </div>
              <div style={{ fontSize: 12, fontWeight: 700, marginTop: 9 }}>{module.title}</div>
              <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 3 }}>{summaries[module.id]}</div>
            </button>
          );
        })}
      </div>
      {active && (
        <div style={{ marginTop: 8, border: '1px solid rgba(34,211,238,0.18)', background: 'rgba(2,6,23,0.56)', borderRadius: 6, padding: 14 }}>
          {active === 'progress' && <ProgressDetail progress={manifest?.progress} />}
          {active === 'claims' && <ClaimsDetail claims={claims} />}
          {active === 'runs' && <RunsDetail runs={manifest?.runs || []} />}
          {active === 'gaps' && <GapsDetail gaps={gaps} />}
        </div>
      )}
    </section>
  );
}
