import { useId, useState } from 'react';
import './research_panels.css';

// One selected workspace at a time; retain visited forms while switching tabs.
export default function ResearchPanels({ items, label, initial = '', collapsible = false, value, onChange }) {
  const entries = items.filter(Boolean);
  const [local, setLocal] = useState(initial);
  const selected = value ?? local;
  const [visited, setVisited] = useState(() => new Set(initial ? [initial] : []));
  const uid = useId();
  const choose = id => {
    const next = collapsible && selected === id ? '' : id;
    setLocal(next); onChange?.(next);
    setVisited(previous => new Set([...previous, id]));
  };
  return <section className="rnd-panels" aria-label={label}>
    <div className="rnd-tabs" role="tablist" aria-label={label}>
      {entries.map(item => <button key={item.id} type="button" role="tab" id={`${uid}-${item.id}-tab`}
        aria-controls={`${uid}-${item.id}-panel`} aria-selected={selected === item.id} onClick={() => choose(item.id)}>{item.label}</button>)}
    </div>
    {entries.map(item => <div key={item.id} role="tabpanel" id={`${uid}-${item.id}-panel`}
      aria-labelledby={`${uid}-${item.id}-tab`} hidden={selected !== item.id} className="rnd-panel">
      {(visited.has(item.id) || selected === item.id) && item.render()}
    </div>)}
  </section>;
}
