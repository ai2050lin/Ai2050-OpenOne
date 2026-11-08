/* 本机后端发现与实时运行（Phase 335 起用于控制台，Phase 346 起工作台共用）。
   没有本机后端时诚实显示不可用，不做模拟。实时观察是未登记观察，不产生研究结论。 */
import { useEffect, useState } from 'react';

import { API_BASE } from '../app/rdc_fusion/api.js';
import { request } from './MechanismTraceExplorer.jsx';
import { rewriteCurve } from './data/assocFlow.js';

const CANDIDATES = () => [...new Set([API_BASE, 'http://127.0.0.1:5001', 'http://127.0.0.1:5002'])];

/* 找一个能响应 probe 路径的后端：同源中心或本机节点。probe 默认是实时运行的状态接口 */
export function useLocalBase(probe = '/api/live/status') {
  const [state, setState] = useState({ base: null, status: null, checked: false });
  useEffect(() => {
    let dead = false;
    (async () => {
      for (const b of CANDIDATES()) {
        try { const s = await request(b, probe); if (!dead) setState({ base: b, status: s, checked: true }); return; } catch { /* 试下一个 */ }
      }
      if (!dead) setState({ base: null, status: null, checked: true });
    })();
    return () => { dead = true; };
  }, [probe]);
  return state;
}
export const useLiveBase = () => useLocalBase('/api/live/status');

export async function runOnce(base, text) {
  const d = await request(base, '/api/live/run', { method: 'POST', body: JSON.stringify({ text, max_tokens: 1024 }) });
  const pos = d.readout_positions.at(-1);
  const hidden = [];
  for (let l = 0; l <= d.n_layers; l++) {
    const f = await request(base, `/api/traces/${d.trace_id}/field?block=${l}&position=${pos}&boundary=${l ? 'hidden_state' : 'embedding'}`);
    hidden.push(f.values);
  }
  return { id: d.trace_id, tokens: d.tokens, top: d.output_top || [], curve: rewriteCurve(hidden), elapsed: d.elapsed_ms };
}
