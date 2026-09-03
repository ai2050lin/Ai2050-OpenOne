import { AlertTriangle, BrainCircuit, CheckCircle2, Circle, Compass, Target } from 'lucide-react';

import { useResearchSnapshot } from '../researchKernel/useResearchSnapshot';

import './ProjectRoadmapTab.css';

const WORKING_HYPOTHESES = [
  ['Core path', 'representation + policy transfer remains the controlling uncertainty'],
  ['Validation path', 'traceability and contract gates must be aligned before claim upgrade'],
  ['Task path', 'token-level effects should show robustness trend'],
  ['Control path', 'patch/replay/restore must be isolated in test route'],
  ['Closure path', 'no new route is added only to make UI look complete'],
];

const CRACKING_TARGETS = [
  {
    index: '01',
    title: 'Reduce embedding drift',
    body: 'Collect larger samples on the same trace family and recheck nearest-neighbor continuity.',
    question: 'Does embedding trend stay stable across three campaigns?',
  },
  {
    index: '02',
    title: 'Close hidden-state interpretation gap',
    body: 'Use run + audit pair to check whether layer-level signatures survive perturbations.',
    question: 'Can hidden-state trend still explain behavior changes?',
  },
  {
    index: '03',
    title: 'Upgrade one falsifiable claim',
    body: 'Pick one candidate claim and run 2x more runs only when gate quality is above minimum.',
    question: 'Can this claim be rejected or upgraded cleanly?',
  },
];

function EvidenceIcon({ status }) {
  if (status === 'passed') return <CheckCircle2 size={13} />;
  if (status === 'blocked') return <AlertTriangle size={13} />;
  return <Circle size={12} />;
}

export const ProjectRoadmapTab = () => {
  const { snapshot, loading, error } = useResearchSnapshot();
  const current = snapshot?.current;
  const evidenceCards = (snapshot?.summaries?.evidence?.latest || []).slice(-6).reverse().map((item) => ({
    id: item.id,
    status: item.polarity === 'positive' ? 'passed' : item.polarity === 'negative' ? 'blocked' : 'pending',
    shortLabel: item.id,
    value: item.grade,
    framework: `Closure L${item.closure_level}`,
  }));

  return (
    <div className="project-outline">
      <header className="project-outline__header">
        <div>
          <span>RESEARCH OUTLINE</span>
          <h1>Theory-to-System Route</h1>
          <p>{loading ? 'Loading...' : error || 'Framework-first path, evidence-first traceability.'}</p>
        </div>
        <div className="project-outline__phase">
          <span>Current Campaign</span>
          <strong>{current?.campaign_id || 'unbound'}</strong>
          <small>{current ? current.campaign_status : 'no status'}</small>
        </div>
      </header>

      <section className="project-outline__section">
        <div className="project-outline__section-heading">
          <Compass size={18} />
          <div>
            <h2>Framework Chain</h2>
            <p>Route shown as current working chain, not phase list.</p>
          </div>
        </div>
        <div className="project-outline__chain">
          {WORKING_HYPOTHESES.map(([title, body], index) => (
            <article key={title}>
              <div><BrainCircuit size={15} /><span>{String(index + 1).padStart(2, '0')}</span></div>
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="project-outline__section">
        <div className="project-outline__section-heading">
          <Target size={18} />
          <div>
            <h2>Current Evidence Stream</h2>
            <p>Recent evidence cards keep traceability; not used as global tabs.</p>
          </div>
        </div>
        <div className="project-outline__gates">
          {evidenceCards.length ? evidenceCards.map((gate) => (
            <article key={gate.id} className={`is-${gate.status}`}>
              <div><EvidenceIcon status={gate.status} /><span>{gate.shortLabel}</span></div>
              <strong>{gate.value}</strong>
              <small>{gate.framework}</small>
            </article>
          )) : <p>No evidence cards in current snapshot.</p>}
        </div>
      </section>

      <section className="project-outline__section">
        <div className="project-outline__section-heading">
          <BrainCircuit size={18} />
          <div><h2>Cracking Targets</h2><p>Prioritized to reduce noise and close quickly.</p></div>
        </div>
        <div className="project-outline__targets">
          {CRACKING_TARGETS.map((target) => (
            <article key={target.index}>
              <span>{target.index}</span>
              <h3>{target.title}</h3>
              <p>{target.body}</p>
              <strong>{target.question}</strong>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
};
