import { AlertTriangle, CheckCircle2, Crosshair, FlaskConical, Gauge, Route } from 'lucide-react';

import { useResearchSnapshot } from '../researchKernel/useResearchSnapshot';

import './ResearchProgressTab.css';

const TRACKS = [
  {
    id: 'discovery',
    framework: 'R0 Discovery',
    objective: 'define token-to-representation boundary.',
    status: 'implemented in current traces',
  },
  {
    id: 'representation',
    framework: 'R1 Representation',
    objective: 'stabilize latent symbol mapping across runs.',
    status: 'partially stable; needs tighter closure',
  },
  {
    id: 'causality',
    framework: 'R2 Causality',
    objective: 'test whether transitions are causal, not just correlated.',
    status: 'running targeted ablation on one route',
  },
];

const METRICS = [
  ['Evidence accepted', 'up', 'latest positive evidence increased'],
  ['Main bottleneck', 'open', 'run-to-run transfer of representation is not closed yet'],
  ['Trace continuity', 'medium', 'artifact lineage is intact for selected runs'],
];

export const ResearchProgressTab = () => {
  const { snapshot, loading, error } = useResearchSnapshot();
  const current = snapshot?.current;
  const framework = snapshot?.framework?.title || 'Current Research Framework';
  const bottleneck = current?.bottleneck || 'No current bottleneck in snapshot';
  const nextDecision = current?.next_decision || 'Collect more high-quality traces and lock a falsifiable target';

  return (
    <div className="research-progress">
      <header className="research-progress__header">
        <div>
          <span>FRAMEWORK-FIRST VIEW</span>
          <h1>Research Framework Dashboard</h1>
          <p>{loading ? 'Loading...' : error || bottleneck}</p>
        </div>
        <div className="research-progress__verdict">
          <AlertTriangle size={17} />
          <span>Current campaign</span>
          <strong>{current ? `${current.campaign_id || 'C000'} ${current.campaign_status || ''}`.trim() : 'unlinked'}</strong>
        </div>
      </header>

      <section className="research-progress__section">
        <div className="research-progress__heading">
          <Route size={19} />
          <div>
            <h2>{framework}</h2>
            <p>{bottleneck}</p>
          </div>
        </div>
        <div className="research-progress__stages">
          {TRACKS.map((track) => (
            <article className="is-active" key={track.id}>
              <span>{track.framework}</span>
              <strong>{track.objective}</strong>
              <small>{track.status}</small>
            </article>
          ))}
        </div>
      </section>

      <section className="research-progress__evidence">
        <div>
          <div className="research-progress__heading">
            <CheckCircle2 size={19} />
            <div>
              <h2>Evidence Trail</h2>
              <p>Used for traceability, not primary navigation</p>
            </div>
          </div>
          <ul>
            {(snapshot?.summaries?.evidence?.latest || []).slice(0, 6).map((item) => (
              <li key={item.id}>
                {item.grade} / {item.polarity || 'unknown'} / {item.title}
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="research-progress__section">
        <div className="research-progress__heading">
          <Gauge size={19} />
          <div>
            <h2>Operational Signals</h2>
            <p>Only key deltas to trigger the next step</p>
          </div>
        </div>
        <div className="research-progress__formulas">
          {METRICS.map(([label, value, note]) => (
            <article key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
              <p>{note}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="research-progress__next">
        <Crosshair size={23} />
        <div>
          <span>NEXT VERIFIABLE TARGET</span>
          <h2>{nextDecision}</h2>
        </div>
        <FlaskConical size={20} />
      </section>
    </div>
  );
};
