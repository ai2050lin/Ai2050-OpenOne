import { tr } from './languageAnalysisData';
import { createElement } from 'react';

export function LanguageAnalysisSectionHeader({
  index,
  title,
  subtitle,
  icon,
  accent,
  lang,
}) {
  return (
    <header className="language-section-heading">
      <div className="language-section-index" style={{ color: accent }}>
        {index}
      </div>
      <div className="language-section-heading-copy">
        <div className="language-section-title-row">
          {createElement(icon, { size: 20, color: accent, 'aria-hidden': true })}
          <h2>{tr(title, lang)}</h2>
        </div>
        <p>{tr(subtitle, lang)}</p>
      </div>
    </header>
  );
}
