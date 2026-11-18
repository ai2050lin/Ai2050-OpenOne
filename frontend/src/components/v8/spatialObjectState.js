import { useMemo, useSyncExternalStore } from 'react';

const KEYS = { layer: 'sblock', module: 'smodule', index: 'sindex', coordinate: 'scoord', mode: 'smode' };
const EMPTY = { layer: null, module: '', index: null, coordinate: null, mode: '' };
export function readSpatialObject(search) {
  const q = new URLSearchParams(search), n = key => /^\d+$/.test(q.get(key) || '') ? Number(q.get(key)) : null;
  if (q.has('sblock')) return { layer: n('sblock'), module: q.get('smodule') || '', index: n('sindex'), coordinate: n('scoord'), mode: q.get('smode') || '' };
  if (q.get('pkind') === 'embedding' || q.get('ttab') === 'embedding') return { ...EMPTY, layer: 0, module: 'embedding', index: n('prow'), coordinate: n('pcol') };
  if (['pblock', 'pkind', 'phead', 'punit'].some(k => q.has(k))) {
    const mlp = q.get('pkind') === 'mlp' || (!q.has('pkind') && q.has('punit'));
    return { ...EMPTY, layer: (n('pblock') ?? 0) + 1, module: mlp ? 'mlp' : 'attn', index: n(mlp ? 'punit' : 'phead') ?? 0 };
  }
  if (q.get('ptab') === 'neurons' && n('block') != null) return { ...EMPTY, layer: n('block') + 1, module: 'mlp', index: n('unit') };
  if (q.has('tblock') || q.has('ttab')) {
    const head = q.get('ttab') === 'attention' || (!q.has('tunit') && q.has('thead'));
    return { ...EMPTY, layer: (n('tblock') ?? 0) + 1, module: head ? 'attn' : 'mlp', index: n(head ? 'thead' : 'tunit') ?? 0 };
  }
  return { ...EMPTY, layer: n('rlayer') ?? 24 };
}
export function writeSpatialObject(selection) {
  const url = new URL(window.location.href);
  for (const [field, key] of Object.entries(KEYS)) {
    const value = selection[field];
    if (value == null || value === '') url.searchParams.delete(key); else url.searchParams.set(key, String(value));
  }
  for (const key of ['pblock', 'pkind', 'phead', 'punit', 'tblock', 'thead', 'tunit', 'ttab', 'rlayer']) url.searchParams.delete(key);
  if (url.href !== window.location.href) { window.history.replaceState(window.history.state, '', url); window.dispatchEvent(new Event('spatial-object')); }
}
const subscribe = callback => {
  for (const event of ['spatial-object', 'popstate', 'research-context']) window.addEventListener(event, callback);
  return () => { for (const event of ['spatial-object', 'popstate', 'research-context']) window.removeEventListener(event, callback); };
};
export function useSpatialObject() {
  const search = useSyncExternalStore(subscribe, () => window.location.search, () => '');
  return useMemo(() => readSpatialObject(search), [search]);
}
