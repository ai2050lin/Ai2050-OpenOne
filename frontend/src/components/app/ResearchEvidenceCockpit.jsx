import {
  AlertTriangle,
  ArrowRight,
  Check,
  Circle,
  FlaskConical,
  X,
} from 'lucide-react';

import { useResearchSnapshot } from '../../researchKernel/useResearchSnapshot';

import './ResearchEvidenceCockpit.css';

const STATUS_ICONS = {
  passed: Check,
  blocked: AlertTriangle,
  pending: Circle,
};

export function ResearchEvidenceCockpit({ onSelectGate }) {
  const { snapshot, error } = useResearchSnapshot();
  const current = snapshot?.current;
  const gates = (snapshot?.summaries?.evidence?.latest || []).slice().reverse().map((item) => ({
    ...item,
    status: item.polarity === 'positive' ? 'passed' : item.polarity === 'negative' ? 'blocked' : 'pending',
    shortLabel: item.id,
    value: item.grade,
    label: item.title,
  }));
  const stage = snapshot?.framework?.current_stage_id || 'R';
  const frameworkLabel = snapshot?.framework?.title || '研发框架';

  return (
    <section className="research-evidence-cockpit" aria-label="当前研究证据面板">
      <header className="research-evidence-cockpit__header">
        <div>
          <span>当前框架：{frameworkLabel}</span>
          <strong>{stage} · {current?.campaign_name || '未定义 Campaign'}</strong>
        </div>
        <span className="research-evidence-cockpit__state">
          {current ? `${current.campaign_id} · ${current.campaign_status}` : '读取中'}
        </span>
      </header>

      <p>{current?.bottleneck || error || '正在读取 Canonical Snapshot...'}</p>

      <div className="research-evidence-cockpit__gates">
        {gates.map((gate) => {
          const Icon = STATUS_ICONS[gate.status] || Circle;
          return (
            <button
              key={gate.id}
              type="button"
              className={`research-evidence-cockpit__gate is-${gate.status}`}
              onClick={() => onSelectGate?.(gate.id)}
              title={gate.label}
            >
              <Icon size={12} />
              <span>{gate.shortLabel}</span>
              <b>{gate.value}</b>
            </button>
          );
        })}
      </div>

      <button type="button" className="research-evidence-cockpit__next" onClick={() => onSelectGate?.('causal')}>
        <FlaskConical size={13} />
        <span>当前未通过项：选择下一条验证任务</span>
        <ArrowRight size={13} />
      </button>
    </section>
  );
}

export function ResearchEvidenceDrawer({ gateId, onClose }) {
  const { snapshot } = useResearchSnapshot();
  const item = (snapshot?.summaries?.evidence?.latest || []).find((candidate) => candidate.id === gateId);
  const gate = item
    ? {
      ...item,
      status: item.polarity === 'positive' ? 'passed' : item.polarity === 'negative' ? 'blocked' : 'pending',
      value: item.grade,
      label: item.title,
      summary: `基于 ${item.polarity} 结果`,
      detail: '请结合实验 run、artifact、审计记录与契约字段判断可否进入下一步。',
      level: `Closure L${item.closure_level}`,
    }
    : null;
  if (!gate) return null;

  return (
    <aside className={`research-evidence-drawer is-${gate.status}`} aria-live="polite">
      <div className="research-evidence-drawer__marker" />
      <div className="research-evidence-drawer__content">
        <div className="research-evidence-drawer__eyebrow">{gate.level} · {gate.value}</div>
        <h2>{gate.label}</h2>
        <p>{gate.summary}</p>
        <div>{gate.detail}</div>
      </div>
      <button type="button" className="research-evidence-drawer__close" onClick={onClose} title="关闭详情" aria-label="关闭详情">
        <X size={17} />
      </button>
    </aside>
  );
}
