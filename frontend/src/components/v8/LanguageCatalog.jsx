import { Fragment, useState } from 'react';
import catalog from '../../../../shared/dist_protocol/lang_catalog.json';

const domainCount = domain => domain.groups.reduce((total, group) => total + group.items.length, 0);
const domainsById = new Map(catalog.domains.map(domain => [domain.id, domain]));
const catalogItems = catalog.domains.flatMap(domain => domain.groups.flatMap(group => group.items));
export const languageCatalogCount = catalogItems.length;
export const countMeasuredLanguageItems = measurements => catalogItems.filter(item => measurements.has(item.measurement)).length;

function CatalogItems({ domain, onOpenMeasurement }) {
  return <div className="v8-catalog-groups" aria-label={`${domain.name}条目`}>
    {domain.groups.map(group => <section key={group.name}>
      <h4>{group.name}</h4>
      {group.items.map(item => <details className="v8-catalog-item" key={item.id}>
        <summary>{item.name}</summary>
        <p className="v8-note">{item.example}</p>
        {item.measurement && <button type="button" className="v6-linkbtn" onClick={() => onOpenMeasurement(item.measurement)}>
          查看对应实验条目
        </button>}
      </details>)}
    </section>)}
  </div>;
}

export default function LanguageCatalog({ onOpenMeasurement }) {
  const [selection, setSelection] = useState(null);

  function toggleSection(section) {
    setSelection(current => current?.section === section.id ? null : { section: section.id, domain: section.domains[0] });
  }

  return <section className="v8-route" aria-label="语言目录">
    <div className="v8-direction-cards v8-overview-cards">
      {catalog.sections.map(section => {
        const open = selection?.section === section.id;
        const domains = section.domains.map(id => domainsById.get(id));
        return <Fragment key={section.id}>
          <button type="button" className="v8-card v8-direction v8-overview-card v8-catalog-card"
            aria-label={section.name} aria-describedby={`catalog-${section.id}-summary`}
            aria-expanded={open} aria-controls={open ? 'language-catalog-items' : undefined}
            onClick={() => toggleSection(section)}>
            <strong>{section.name}</strong>
            <span className="v8-note" id={`catalog-${section.id}-summary`}>{section.summary}</span>
            <span className="v8-overview-action" aria-hidden="true">{open ? '收起 ↑' : '查看条目 →'}</span>
          </button>
          {open && <section id="language-catalog-items" className="v8-card v8-overview-detail" aria-label={`${section.name}条目`}>
            {domains.length > 1 && <div className="v8-catalog-switch" role="group" aria-label="选择条目类别">
              {domains.map(domain => <button type="button" className="fw-tbtn" key={domain.id}
                aria-pressed={selection.domain === domain.id} onClick={() => setSelection({ section: section.id, domain: domain.id })}>
                {domain.name}
              </button>)}
            </div>}
            <CatalogItems key={selection.domain} domain={domainsById.get(selection.domain)} onOpenMeasurement={onOpenMeasurement} />
          </section>}
        </Fragment>;
      })}
    </div>
  </section>;
}
