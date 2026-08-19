import {
  AlertTriangle,
  ArrowRight,
  Check,
  Circle,
  FlaskConical,
  X,
} from 'lucide-react';

import {
  CURRENT_RESEARCH_STATE,
  RESEARCH_EVIDENCE_GATES,
  getResearchEvidenceGate,
} from '../../researchKernel/currentResearchState';

import './ResearchEvidenceCockpit.css';

const STATUS_ICONS = {
  passed: Check,
  blocked: AlertTriangle,
  pending: Circle,
};

export function ResearchEvidenceCockpit({ onSelectGate }) {
  return (
    <section className="research-evidence-cockpit" aria-label="当前研究证据状态">
      <header className="research-evidence-cockpit__header">
        <div>
          <span>PHASE {CURRENT_RESEARCH_STATE.phase}</span>
          <strong>{CURRENT_RESEARCH_STATE.title}</strong>
        </div>
        <span className="research-evidence-cockpit__state">{CURRENT_RESEARCH_STATE.statusLabel}</span>
      </header>

      <p>{CURRENT_RESEARCH_STATE.summary}</p>

      <div className="research-evidence-cockpit__gates">
        {RESEARCH_EVIDENCE_GATES.map((gate) => {
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
        <span>当前停止点：统一充分状态未通过</span>
        <ArrowRight size={13} />
      </button>
    </section>
  );
}

export function ResearchEvidenceDrawer({ gateId, onClose }) {
  const gate = getResearchEvidenceGate(gateId);
  if (!gate) return null;

  return (
    <aside className={`research-evidence-drawer is-${gate.status}`} aria-live="polite">
      <div className="research-evidence-drawer__marker" />
      <div className="research-evidence-drawer__content">
        <div className="research-evidence-drawer__eyebrow">{gate.phase} · {gate.value}</div>
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
