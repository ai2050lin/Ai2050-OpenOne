import { useState } from 'react';

export default function ResearchDisclosure({ title, onFirstOpen, children, initiallyOpen=false }) {
  const [visited, setVisited] = useState(initiallyOpen);
  return <details className="v8-card v8-overview-fold" open={initiallyOpen||undefined} onToggle={event => {
    if (visited || !event.currentTarget.open) return;
    setVisited(true);
    onFirstOpen?.();
  }}>
    <summary>{title}</summary>
    {visited && <div className="v8-research-fold-content">{children}</div>}
  </details>;
}
