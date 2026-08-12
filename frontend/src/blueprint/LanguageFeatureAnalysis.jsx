import { Brain, Database, Layers, Sparkles, Target } from 'lucide-react';

import { LanguageAnalysisSectionHeader } from './LanguageAnalysisSectionHeader';
import { LANGUAGE_FRAMEWORK_COLUMNS, tr } from './languageAnalysisData';

const FRAMEWORK_ICONS = {
  database: Database,
  target: Target,
  layers: Layers,
};

function FrameworkColumn({ column, lang }) {
  const Icon = FRAMEWORK_ICONS[column.icon] || Brain;

  return (
    <article className="language-framework-column" style={{ '--framework-accent': column.accent }}>
      <div className="language-framework-title">
        <Icon size={18} aria-hidden="true" />
        <h3>{tr(column.title, lang)}</h3>
      </div>
      <div className="language-framework-items">
        {column.items.map((item) => (
          <div className="language-framework-item" key={tr(item.title, 'zh')}>
            <h4>{tr(item.title, lang)}</h4>
            <p>{tr(item.body, lang)}</p>
          </div>
        ))}
      </div>
    </article>
  );
}

export function LanguageFeatureAnalysis({ lang }) {
  const isEnglish = lang === 'en';

  return (
    <section className="language-analysis-section" id="language-features">
      <LanguageAnalysisSectionHeader
        index="01"
        icon={Brain}
        accent="#22d3ee"
        lang={lang}
        title={{ zh: '语言特性分析', en: 'Language Feature Analysis' }}
        subtitle={{
          zh: '把知识、推理、语法和生成控制统一为模式，研究其相对编码、复用差分和多尺度条件计算。',
          en: 'Treat knowledge, reasoning, syntax, and generation control as patterns, then study relational encoding, reuse-difference, and multiscale conditional computation.',
        }}
      />

      <div className="language-framework-grid">
        {LANGUAGE_FRAMEWORK_COLUMNS.map((column) => (
          <FrameworkColumn key={column.id} column={column} lang={lang} />
        ))}
      </div>

      <div className="language-core-hypothesis">
        <Sparkles size={18} color="#fbbf24" aria-hidden="true" />
        <div>
          <h3>{isEnglish ? 'Core Objective and Working Hypothesis' : '核心目标与工作假说'}</h3>
          <p>
            {isEnglish
              ? 'Language may arise from relational pattern networks, structural reuse, minimal differences, and conditioned state transitions. Phase 1140 supports process over static coordinates and sequence-aligned intervention, but no cross-model invariant has been established.'
              : '语言能力可能由模式网络中的相对关系、结构复用、最小差分和条件化状态转移共同实现。当前 Phase 1140 的证据支持“过程优先于静态坐标”和“干预必须对齐序列决策路径”，但尚未找到跨模型稳定的最小计算单元或编码不变量。'}
          </p>
        </div>
      </div>
    </section>
  );
}
