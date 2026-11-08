/* 本机后端发现与实时运行（Phase 335 起用于控制台，Phase 346 起工作台共用）。
   没有本机后端时诚实显示不可用，不做模拟。实时观察是未登记观察，不产生研究结论。 */
import { useEffect, useState } from 'react';

import { API_BASE } from '../app/rdc_fusion/api.js';
import { request } from './MechanismTraceExplorer.jsx';
import { rewriteCurve } from './data/assocFlow.js';

const CANDIDATES = () => [...new Set([API_BASE, 'http://127.0.0.1:5001', 'http://127.0.0.1:5002'])];

/* 找一个能响应 probe 路径的后端：同源中心或本机节点。accept 判断响应是不是要找的服务 */
export function useLocalBase(probe = '/api/live/status', accept = () => true) {
  const [state, setState] = useState({ base: null, status: null, checked: false });
  useEffect(() => {
    let dead = false;
    (async () => {
      for (const b of CANDIDATES()) {
        try { const s = await request(b, probe); if (accept(s)) { if (!dead) setState({ base: b, status: s, checked: true }); return; } } catch { /* 试下一个 */ }
      }
      if (!dead) setState({ base: null, status: null, checked: true });
    })();
    return () => { dead = true; };
  }, [probe]);   // eslint-disable-line react-hooks/exhaustive-deps
  return state;
}
/* 实时前向服务：只认 live_status.v1（中心自己的 /api/live/status 是节点连接状态，不能运行模型） */
export const useLiveBase = () => useLocalBase('/api/live/status', s => s?.schema === 'live_status.v1');

/* 本机模型的加载状态：idle / loading / ready / error / unavailable；加载中每 1.5 秒刷新 */
export function useLiveModel(base) {
  const [st, setSt] = useState({ status: null, busy: false, msg: '' });
  const refresh = async () => { if (!base) return null; try { const s = await request(base, '/api/live/status'); setSt(x => ({ ...x, status: s })); return s; } catch (e) { setSt(x => ({ ...x, msg: e.message })); return null; } };
  useEffect(() => { refresh(); }, [base]);   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (st.status?.status !== 'loading') return undefined;
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, [st.status?.status]);   // eslint-disable-line react-hooks/exhaustive-deps
  const act = async kind => {
    setSt(x => ({ ...x, busy: true, msg: '' }));
    try { const r = await request(base, `/api/live/${kind}`, { method: 'POST' }); setSt(x => ({ ...x, msg: r.message || '' })); }
    catch (e) { setSt(x => ({ ...x, msg: e.message })); }
    await refresh(); setSt(x => ({ ...x, busy: false }));
  };
  return { ...st, load: () => act('load'), unload: () => act('unload'), refresh };
}

export async function runOnce(base, text) {
  const d = await request(base, '/api/live/run', { method: 'POST', body: JSON.stringify({ text, max_tokens: 1024 }) });
  const pos = d.readout_positions.at(-1);
  const hidden = [];
  for (let l = 0; l <= d.n_layers; l++) {
    const f = await request(base, `/api/traces/${d.trace_id}/field?block=${l}&position=${pos}&boundary=${l ? 'hidden_state' : 'embedding'}`);
    hidden.push(f.values);
  }
  return { id: d.trace_id, tokens: d.tokens, readout: d.readout_positions, top: d.output_top || [], curve: rewriteCurve(hidden), elapsed: d.elapsed_ms };
}

/* 实时运行 bundle（/api/traces/{id}/bundle，float32 小端）→ 与冻结样本相同接口的取值器 */
export function f32(o) {     // float32 小端（实时轨迹 bundle）
  const bin = atob(o.b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return { a: new Float32Array(u.buffer), shape: o.shape };
}
export function liveSampler(b, res) {
  if (!b || !res) return null;
  const hs = f32(b.hs), at = f32(b.attn), mlp = b.mlp ? f32(b.mlp) : null, ao = b.attn_out ? f32(b.attn_out) : null, mo = b.mlp_out ? f32(b.mlp_out) : null;
  const [NL, DM] = hs.shape, [NB, NH, T] = at.shape, NF = mlp ? mlp.shape[1] : 9728;
  const row = (m, i, w) => m.a.subarray(i * w, (i + 1) * w);
  return {
    P: 1, NL, DM, NB, NF, NH, T, live: true, hasMlp: Boolean(mlp), hasSplit: Boolean(ao && mo),
    d: { tokpos: Array.from({ length: T }, (_, i) => i), lens: null, model_snapshot: b.model_snapshot, ntok: T, readout: b.readout_positions, tokens: res.tokens },
    hs: (_p, l) => row(hs, l, DM),
    mlp: blk => (mlp ? row(mlp, blk, NF) : null),
    att: (blk, h) => at.a.subarray((blk * NH + h) * T, (blk * NH + h + 1) * T),
    ao: blk => (ao ? row(ao, blk, DM) : null), mo: blk => (mo ? row(mo, blk, DM) : null),
  };
}
