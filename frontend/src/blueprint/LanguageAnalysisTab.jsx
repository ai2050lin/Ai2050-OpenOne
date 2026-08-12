import { BookOpen, Map, Microscope, Network } from 'lucide-react';

import './LanguageAnalysisTab.css';
import { LanguageFeatureAnalysis } from './LanguageFeatureAnalysis';
import { LanguageResearchTimeline } from './LanguageResearchTimeline';
import { LanguageTraceAtlas } from './LanguageTraceAtlas';

const PAGE_SECTIONS = [
  { id: 'language-features', index: '01', zh: '语言特性分析', en: 'Language Features', icon: Microscope },
  { id: 'trace-atlas', index: '02', zh: '图谱脉络临摹', en: 'Trace Cartography', icon: Map },
  { id: 'research-history', index: '03', zh: '研究记录', en: 'Research Record', icon: Network },
];

export const LanguageAnalysisTab = ({ lang = 'zh' }) => {
  const isEnglish = lang === 'en';

  return (
    <div className="language-analysis-page">
      <header className="language-analysis-page-header">
        <div className="language-analysis-page-title">
          <BookOpen size={27} color="#22d3ee" aria-hidden="true" />
          <div>
            <span>{isEnglish ? 'LANGUAGE MECHANISM RESEARCH' : '语言机制研究'}</span>
            <h1>{isEnglish ? 'Language Analysis' : '语言分析'}</h1>
            <p>
              {isEnglish
                ? 'Study language patterns from relational encoding and repeated traces to sequence-aligned causal state transitions.'
                : '从模式、相对编码和重复物理脉络出发，推进到与真实序列决策路径对齐的因果状态转移。'}
            </p>
          </div>
        </div>

        <nav className="language-analysis-section-index" aria-label={isEnglish ? 'Page sections' : '页面章节'}>
          {PAGE_SECTIONS.map((section) => {
            const Icon = section.icon;
            return (
              <a href={`#${section.id}`} key={section.id}>
                <span>{section.index}</span>
                <Icon size={15} aria-hidden="true" />
                <strong>{isEnglish ? section.en : section.zh}</strong>
              </a>
            );
          })}
        </nav>
      </header>

      <LanguageFeatureAnalysis lang={lang} />
      <LanguageTraceAtlas lang={lang} />
      <LanguageResearchTimeline lang={lang} />
    </div>
  );
};
