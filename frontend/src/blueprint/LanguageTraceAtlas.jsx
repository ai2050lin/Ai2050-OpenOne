import {
  Activity,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  CircleDot,
  GitBranch,
  Map as MapIcon,
  Network,
  Route,
} from 'lucide-react';
import { useMemo, useState } from 'react';

import { LanguageAnalysisSectionHeader } from './LanguageAnalysisSectionHeader';
import {
  LANGUAGE_TRACES,
  TRACE_DIMENSIONS,
  TRACE_EVIDENCE_STAGES,
  tr,
} from './languageAnalysisData';

function EvidenceStrip({ evidence, lang }) {
  return (
    <div className="trace-evidence-strip" aria-label={lang === 'en' ? 'Evidence levels' : '证据等级'}>
      {TRACE_EVIDENCE_STAGES.map((stage) => {
        const reached = evidence.includes(stage.id);
        return (
          <div
            className={`trace-evidence-step${reached ? ' is-reached' : ''}`}
            key={stage.id}
            title={tr(stage.label, lang)}
          >
            <span aria-hidden="true" />
            <small>{tr(stage.label, lang)}</small>
          </div>
        );
      })}
    </div>
  );
}

function TraceCard({ trace, active, lang, onToggle }) {
  return (
    <button
      type="button"
      className={`language-trace-card${active ? ' is-active' : ''}`}
      style={{ '--trace-accent': trace.accent }}
      aria-expanded={active}
      onClick={onToggle}
    >
      <div className="language-trace-card-head">
        <div>
          <span className="language-trace-phase">{trace.phase}</span>
          <h4>{tr(trace.title, lang)}</h4>
        </div>
        {active ? <ChevronUp size={17} aria-hidden="true" /> : <ChevronDown size={17} aria-hidden="true" />}
      </div>
      <p>{tr(trace.summary, lang)}</p>
      <div className="language-trace-card-meta">
        <span>{tr(trace.status, lang)}</span>
        <span>{trace.models.join(' · ')}</span>
      </div>
      <EvidenceStrip evidence={trace.evidence} lang={lang} />
    </button>
  );
}

function TraceDetail({ trace, lang, onClose }) {
  if (!trace) return null;

  return (
    <div className="language-trace-detail" style={{ '--trace-accent': trace.accent }}>
      <div className="language-trace-detail-head">
        <div>
          <div className="language-detail-kicker">{trace.phase} · {tr(trace.status, lang)}</div>
          <h3>{tr(trace.title, lang)}</h3>
        </div>
        <button type="button" className="language-detail-close" onClick={onClose} title={lang === 'en' ? 'Collapse detail' : '收起详情'}>
          <ChevronUp size={17} aria-hidden="true" />
        </button>
      </div>

      <div className="language-mechanism-band">
        <Route size={19} aria-hidden="true" />
        <div>
          <h4>{lang === 'en' ? 'Current Mechanism Description' : '当前机制说明'}</h4>
          <p>{tr(trace.mechanism, lang)}</p>
        </div>
      </div>

      <div className="language-trace-trajectory">
        {trace.trajectory.map((step) => (
          <div className="language-trajectory-step" key={`${trace.id}-${step.stage}`}>
            <span>{step.stage}</span>
            <h5>{tr(step.title, lang)}</h5>
            <p>{tr(step.detail, lang)}</p>
          </div>
        ))}
      </div>

      <div className="language-trace-detail-grid">
        <div className="language-detail-block">
          <div className="language-detail-block-title">
            <Network size={16} aria-hidden="true" />
            {lang === 'en' ? 'Neural Network Features' : '深度神经网络内的特征'}
          </div>
          <dl className="language-network-features">
            {trace.networkFeatures.map((feature) => (
              <div key={tr(feature.label, 'zh')}>
                <dt>{tr(feature.label, lang)}</dt>
                <dd>{tr(feature.value, lang)}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="language-detail-block">
          <div className="language-detail-block-title">
            <CircleDot size={16} aria-hidden="true" />
            {lang === 'en' ? 'Established Observations' : '已经建立的观察'}
          </div>
          <ul className="language-detail-list">
            {trace.findings.map((finding) => <li key={tr(finding, 'zh')}>{tr(finding, lang)}</li>)}
          </ul>
        </div>

        <div className="language-detail-block language-detail-block--warning">
          <div className="language-detail-block-title">
            <AlertTriangle size={16} aria-hidden="true" />
            {lang === 'en' ? 'Boundaries and Confounds' : '边界与混杂'}
          </div>
          <ul className="language-detail-list">
            {trace.limits.map((limit) => <li key={tr(limit, 'zh')}>{tr(limit, lang)}</li>)}
          </ul>
        </div>
      </div>

      <div className="language-detail-source">
        {lang === 'en' ? 'Evidence source' : '证据来源'}: {trace.source}
      </div>
    </div>
  );
}

export function LanguageTraceAtlas({ lang }) {
  const [activeDimension, setActiveDimension] = useState('operation');
  const [activeTraceId, setActiveTraceId] = useState('');

  const groups = useMemo(() => {
    const grouped = new Map();
    LANGUAGE_TRACES.forEach((trace) => {
      const group = tr(trace.dimensions[activeDimension], lang);
      if (!grouped.has(group)) grouped.set(group, []);
      grouped.get(group).push(trace);
    });
    return [...grouped.entries()];
  }, [activeDimension, lang]);

  const activeTrace = LANGUAGE_TRACES.find((trace) => trace.id === activeTraceId);
  const repeatedCount = LANGUAGE_TRACES.filter((trace) => trace.evidence.includes('repeated')).length;
  const causalCount = LANGUAGE_TRACES.filter((trace) => trace.evidence.includes('causal')).length;
  const closedCount = LANGUAGE_TRACES.filter((trace) => trace.evidence.includes('closed')).length;

  return (
    <section className="language-analysis-section" id="trace-atlas">
      <LanguageAnalysisSectionHeader
        index="02"
        icon={MapIcon}
        accent="#34d399"
        lang={lang}
        title={{ zh: '图谱脉络临摹', en: 'Trace Atlas Cartography' }}
        subtitle={{
          zh: '持续记录语言变化在网络中的出现、扩散、汇聚和输出。局部闭合失败只降低证据等级，不阻断图谱增长。',
          en: 'Continuously map how language changes emerge, spread, converge, and reach output. Local closure failure lowers evidence level without stopping atlas growth.',
        }}
      />

      <div className="language-atlas-summary">
        <div><span>{LANGUAGE_TRACES.length}</span><small>{lang === 'en' ? 'mapped traces' : '候选脉络'}</small></div>
        <div><span>{repeatedCount}</span><small>{lang === 'en' ? 'repeated' : '重复出现'}</small></div>
        <div><span>{causalCount}</span><small>{lang === 'en' ? 'local causal support' : '局部因果支持'}</small></div>
        <div><span>{closedCount}</span><small>{lang === 'en' ? 'fully closed' : '完整闭合'}</small></div>
        <div className="language-atlas-strategy">
          <GitBranch size={18} aria-hidden="true" />
          <span>{lang === 'en' ? 'Cartography first · closure deferred' : '临摹优先 · 闭合后置'}</span>
        </div>
      </div>

      <div className="language-dimension-control" role="tablist" aria-label={lang === 'en' ? 'Atlas dimension' : '图谱观察维度'}>
        {TRACE_DIMENSIONS.map((dimension) => (
          <button
            type="button"
            role="tab"
            aria-selected={activeDimension === dimension.id}
            className={activeDimension === dimension.id ? 'is-active' : ''}
            key={dimension.id}
            onClick={() => setActiveDimension(dimension.id)}
          >
            {tr(dimension.label, lang)}
          </button>
        ))}
      </div>

      <div className="language-trace-groups">
        {groups.map(([group, traces]) => (
          <div className="language-trace-group" key={group}>
            <div className="language-trace-group-label">
              <Activity size={14} aria-hidden="true" />
              <span>{group}</span>
              <small>{traces.length}</small>
            </div>
            <div className="language-trace-grid">
              {traces.map((trace) => (
                <TraceCard
                  key={trace.id}
                  trace={trace}
                  lang={lang}
                  active={activeTraceId === trace.id}
                  onToggle={() => setActiveTraceId(activeTraceId === trace.id ? '' : trace.id)}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <TraceDetail trace={activeTrace} lang={lang} onClose={() => setActiveTraceId('')} />
    </section>
  );
}
