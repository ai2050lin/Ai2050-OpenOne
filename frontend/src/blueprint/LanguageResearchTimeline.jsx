import { ChevronDown, ChevronUp, Clock3, History } from 'lucide-react';
import { useState } from 'react';

import { LanguageAnalysisSectionHeader } from './LanguageAnalysisSectionHeader';
import { CURRENT_RESEARCH_MILESTONES, tr } from './languageAnalysisData';

function MilestoneDetail({ milestone, lang }) {
  const fields = [
    { label: { zh: '核心问题', en: 'Core question' }, value: milestone.objective },
    { label: { zh: '阶段成果', en: 'Result' }, value: milestone.result },
    { label: { zh: '主要教训', en: 'Lesson' }, value: milestone.lesson },
    { label: { zh: '路线影响', en: 'Impact' }, value: milestone.impact },
  ];

  return (
    <div className="language-milestone-detail" style={{ '--milestone-tone': milestone.tone }}>
      {fields.map((field) => (
        <div key={tr(field.label, 'zh')}>
          <span>{tr(field.label, lang)}</span>
          <p>{tr(field.value, lang)}</p>
        </div>
      ))}
    </div>
  );
}

export function LanguageResearchTimeline({ lang }) {
  const [activeMilestoneId, setActiveMilestoneId] = useState('research-os-wp00');

  return (
    <section className="language-analysis-section" id="research-history">
      <LanguageAnalysisSectionHeader
        index="03"
        icon={History}
        accent="#f59e0b"
        lang={lang}
        title={{ zh: '研究记录', en: 'Research Record' }}
        subtitle={{
          zh: '只保留改变研究对象、证据标准或整体路线的大节点，局部实验继续留在原始 Phase 记录中。',
          en: 'Only milestones that changed the research object, evidence standard, or overall route are shown here.',
        }}
      />

      <div className="language-research-timeline">
        {CURRENT_RESEARCH_MILESTONES.map((milestone, index) => {
          const active = activeMilestoneId === milestone.id;
          return (
            <article className={`language-milestone${active ? ' is-active' : ''}`} key={milestone.id} style={{ '--milestone-tone': milestone.tone }}>
              <div className="language-milestone-marker" aria-hidden="true">
                <span>{String(index + 1).padStart(2, '0')}</span>
              </div>
              <div className="language-milestone-content">
                <button
                  type="button"
                  className="language-milestone-button"
                  aria-expanded={active}
                  onClick={() => setActiveMilestoneId(active ? '' : milestone.id)}
                >
                  <div>
                    <div className="language-milestone-phase"><Clock3 size={13} aria-hidden="true" />{milestone.phase}</div>
                    <h3>{tr(milestone.title, lang)}</h3>
                    <p>{tr(milestone.summary, lang)}</p>
                  </div>
                  {active ? <ChevronUp size={18} aria-hidden="true" /> : <ChevronDown size={18} aria-hidden="true" />}
                </button>
                {active && <MilestoneDetail milestone={milestone} lang={lang} />}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
