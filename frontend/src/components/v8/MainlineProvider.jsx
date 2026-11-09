import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE, getJson } from '../app/rdc_fusion/api.js';
import { bundledState, MainlineContext, readSnapshot, useMainline } from './data/mainlineState.js';

export function MainlineProvider({ children }) {
  const [current, setCurrent] = useState({ state: bundledState, revision: null, source: 'bundled', error: '' });
  const accepted = useRef(null), busy = useRef(false);
  const refresh = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    const key = `ai2050.mainline:${API_BASE || window.location.origin}`;
    try {
      if (!accepted.current) {
        try {
          const cached = JSON.parse(localStorage.getItem(key));
          if (cached) {
            accepted.current = await readSnapshot(cached);
            setCurrent({ ...accepted.current, source: 'cached', error: '' });
          }
        } catch { /* A corrupt or unavailable cache is never a server record. */ }
      }
      const envelope = await getJson('/api/research/mainline', { timeoutMs: 12000, networkOnly: true });
      const next = await readSnapshot(envelope, accepted.current);
      accepted.current = next;
      try { localStorage.setItem(key, JSON.stringify(envelope)); } catch { /* Private mode / quota. */ }
      setCurrent(old => old.revision === next.revision && old.source === 'server' && !old.error ? old : { ...next, source: 'server', error: '' });
    } catch (reason) {
      setCurrent(old => ({ ...old, source: accepted.current ? 'cached' : 'bundled', error: reason.message }));
    } finally { busy.current = false; }
  }, []);
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 30000);
    window.addEventListener('focus', refresh);
    window.addEventListener('ai2050:research-refresh', refresh);
    return () => { clearInterval(timer); window.removeEventListener('focus', refresh); window.removeEventListener('ai2050:research-refresh', refresh); };
  }, [refresh]);
  return <MainlineContext.Provider value={{ ...current.state.view, ...current, refresh }}>{children}</MainlineContext.Provider>;
}

export function MainlineStatus() {
  const { state, source, error, revision, refresh } = useMainline();
  const label = source === 'server' ? '中心研究记录' : source === 'cached' ? '已校验缓存 · 中心暂不可达' : '客户端随包记录 · 尚未接通中心记录';
  return <div className="v8-note mainline-status" aria-label="研究记录同步" title={error || state.scientific_note}>
    <span>{label} · v{state.sequence} · {state.active_branch.id}：{state.active_branch.summary}</span>
    <button type="button" className="v6-linkbtn" onClick={refresh}>同步研究记录</button>
    <details><summary>版本与来源</summary><p>{state.updated_at} · {revision || '客户端随包版本'}<br />{state.scientific_note}{error && <><br />{error}</>}</p></details>
  </div>;
}
