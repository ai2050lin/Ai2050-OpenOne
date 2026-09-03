import { AlertTriangle, Boxes, BrainCircuit, MonitorCog, ShieldCheck } from 'lucide-react';

import { useResearchSnapshot } from '../researchKernel/useResearchSnapshot';

const MODULES = [
  { id: 'registry', title: 'Registry', status: 'ready', summary: 'Core metadata and experiment registry are loaded.' },
  { id: 'evidence', title: 'Evidence OS', status: 'ready', summary: 'Evidence index and run artifacts are available.' },
  { id: 'readout', title: 'Readout Traces', status: 'partial', summary: 'Layer and output traces are present but not fully cleaned.' },
  { id: 'wp01', title: 'WP01 Control', status: 'blocked', summary: 'Next route wait condition not yet satisfied.' },
  { id: 'causal', title: 'Causal Extractor', status: 'blocked', summary: 'Causal components depend on deeper alignment evidence.' },
  { id: 'client', title: '3D/Visual Client', status: 'candidate', summary: 'Heatmap + trace path route is the recommended path.' },
  { id: 'cross', title: 'Cross Model', status: 'partial', summary: 'Qwen3 / GLM4 / DS7B comparison in progress.' },
];

const TASKS = [
  { id: 'preflight', priority: 'P0', title: 'Run preflight checks', detail: 'run_ready + tokenizer sanity' },
  { id: 'freeze', priority: 'P0', title: 'Freeze experiment environment', detail: 'build stable manifest and lock run context' },
  { id: 'adjudication', priority: 'P1', title: 'Evidence adjudication', detail: 'move only when run trace is reproducible' },
  { id: 'closure', priority: 'P2', title: 'Claim closure', detail: 'close one falsifiable claim per cycle' },
];

const STATUS = {
  ready: { label: 'ready', color: '#34d399' },
  partial: { label: 'partial', color: '#f59e0b' },
  candidate: { label: 'candidate', color: '#60a5fa' },
  blocked: { label: 'blocked', color: '#fb7185' },
};

function statusTone(status) {
  return STATUS[status] || STATUS.partial;
}

function statusIcon(status) {
  if (status === 'ready') return ShieldCheck;
  if (status === 'partial') return AlertTriangle;
  if (status === 'candidate') return MonitorCog;
  return BrainCircuit;
}

function statusMark(status) {
  const tone = statusTone(status);
  const Icon = statusIcon(status);
  return { tone, Icon };
}

function IconStatus({ status }) {
  const { tone, Icon } = statusMark(status);
  return <Icon size={15} color={tone.color} />;
}

export const SystemStatusTab = () => {
  const { snapshot, error } = useResearchSnapshot();
  const current = snapshot?.current;
  const counts = snapshot?.counts || {};

  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <section style={{ display: 'grid', gap: 8, padding: '12px 14px', border: '1px solid rgba(148,163,184,0.14)', background: 'rgba(15,23,42,0.2)' }}>
        <div style={{ color: '#a7f3d0', fontSize: 10, fontWeight: 900, letterSpacing: 1.3 }}>RESEARCH SYSTEM STATUS</div>
        <div style={{ color: '#f8fafc', fontSize: 22 }}>
          {current ? `${current.campaign_id || 'Campaign'} ${current.campaign_name || ''}` : 'No active campaign'}
        </div>
        <div style={{ color: '#94a3b8', fontSize: 12 }}>{current?.bottleneck || error || 'Loading snapshot...'}</div>
      </section>

      <section style={{ display: 'grid', gap: 8 }}>
        <div style={{ color: '#f8fafc', fontSize: 16, fontWeight: 800 }}>System Modules</div>
        <div style={{ color: '#94a3b8', fontSize: 12 }}>Status is shown as health, not as step control.</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px,1fr))', gap: 8 }}>
          {MODULES.map((mod) => {
            const tone = statusTone(mod.status);
            return (
              <div key={mod.id} style={{ padding: 10, borderTop: `2px solid ${tone.color}`, background: 'rgba(15,23,42,0.28)', borderRadius: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: tone.color }}>
                  <IconStatus status={mod.status} />
                  <strong>{mod.title}</strong>
                </div>
                <div style={{ color: tone.color, fontSize: 10, marginTop: 6 }}>{tone.label}</div>
                <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.6, marginTop: 6 }}>{mod.summary}</div>
              </div>
            );
          })}
        </div>
      </section>

      <section style={{ display: 'grid', gap: 8 }}>
        <div style={{ color: '#f8fafc', fontSize: 16, fontWeight: 800 }}>Current Data</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(110px,1fr))', gap: 8 }}>
          {[
            ['Hypotheses', counts.hypotheses || 0],
            ['Puzzles', counts.puzzles || 0],
            ['Evidence', counts.evidence || 0],
            ['Constructs', counts.constructs || 0],
            ['Runs', counts.runs || 0],
          ].map(([label, value]) => (
            <div key={label} style={{ padding: '10px', background: 'rgba(15,23,42,0.25)', borderRadius: 6 }}>
              <div style={{ color: '#94a3b8', fontSize: 10 }}>{label}</div>
              <div style={{ color: '#f8fafc', fontSize: 18, fontWeight: 800, fontFamily: 'monospace' }}>{value}</div>
            </div>
          ))}
        </div>
      </section>

      <section style={{ display: 'grid', gap: 8 }}>
        <div style={{ color: '#f8fafc', fontSize: 16, fontWeight: 800 }}>Next Actions</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(190px,1fr))', gap: 8 }}>
          {TASKS.map((task) => (
            <div key={task.id} style={{ padding: 10, background: 'rgba(15,23,42,0.28)', borderRadius: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6 }}>
                <div style={{ color: '#f8fafc', fontWeight: 800 }}>{task.title}</div>
                <span style={{ color: '#fbbf24', fontSize: 10 }}>{task.priority}</span>
              </div>
              <div style={{ marginTop: 6, color: '#94a3b8', fontSize: 11 }}>{task.detail}</div>
            </div>
          ))}
        </div>
      </section>

      <section style={{ display: 'grid', gap: 6 }}>
        <div style={{ color: '#7dd3fc', fontSize: 11, display: 'flex', gap: 10, alignItems: 'center' }}>
          <Boxes size={12} />
          <span>Use current evidence/manifest links as read-only verification trail</span>
        </div>
        <div style={{ color: '#64748b', fontSize: 10 }}>For deeper status details, open the audit report artifact.</div>
      </section>
    </div>
  );
};
