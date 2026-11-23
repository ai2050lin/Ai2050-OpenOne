/* 本机后端发现与实时运行（Phase 335 起用于控制台，Phase 346 起工作台共用）。
   没有本机后端时诚实显示不可用，不做模拟。实时观察是未登记观察，不产生研究结论。 */
import { useEffect, useState } from 'react';

import { API_BASE, LOCAL_ANALYSIS_BASES } from '../../config.js';
import { request } from './MechanismTraceExplorer.jsx';
import { rewriteCurve } from './data/assocFlow.js';

const CANDIDATES = () => [...new Set([API_BASE, ...LOCAL_ANALYSIS_BASES])];
const loopback = new Set(['localhost','127.0.0.1','[::1]']);
function localBases() {
  const origin=new URL(API_BASE||window.location.href,window.location.href);
  const configured=loopback.has(origin.hostname)&&['http:','https:'].includes(origin.protocol)
    ? [API_BASE||origin.origin] : [];
  return [...new Set([...configured,...LOCAL_ANALYSIS_BASES])];
}

const acceptAll = () => true;
const isLiveService = status => status?.schema === 'live_status.v1';

/* 找一个能响应 probe 路径的后端：同源中心或本机节点。accept 判断响应是不是要找的服务 */
export function useLocalBase(probe = '/api/live/status', accept = acceptAll, localOnly = false, retry = 0) {
  const [state, setState] = useState({});
  useEffect(() => {
    if(!probe)return;
    let dead = false;
    (async () => {
      let fallback=null;
      for (const base of (localOnly?localBases():CANDIDATES())) {
        if(dead)return;
        try {
          const status=await request(base,probe,{signal:AbortSignal.timeout(3000)});
          if(!accept(status))continue;
          const result={probe,retry,base,status,checked:true};
          if(localOnly&&status.available===false){fallback??=result;continue;}
          if(!dead)setState(result);
          return;
        } catch { /* 试下一个 */ }
      }
      if (!dead) setState(fallback||{probe,retry,base:null,status:null,checked:true});
    })();
    return () => { dead = true; };
  }, [probe,accept,localOnly,retry]);
  return state.probe===probe&&state.retry===retry?state:{base:null,status:null,checked:false};
}
/* 实时输入只交给本机采集服务。 */
export function useLiveBase() {
  const [attempt,setAttempt]=useState(0);
  const service=useLocalBase('/api/live/status',isLiveService,true,attempt);
  return {...service,reconnect:()=>setAttempt(previous=>previous+1)};
}

/* 本机模型的加载状态：idle / loading / ready / error / unavailable；加载中每 1.5 秒刷新 */
export function useLiveModel(base) {
  const [st, setSt] = useState({ status: null, busy: false, msg: '' });
  const refresh = async () => { if (base==null) return null; try { const s = await request(base, '/api/live/status', {signal:AbortSignal.timeout(5000)}); setSt(x => ({ ...x, base, status: s })); return s; } catch (e) { setSt(x => ({ ...x, base, msg: e.message })); return null; } };
  useEffect(() => { refresh(); }, [base]);   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (st.status?.status !== 'loading') return undefined;
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, [st.status?.status]);   // eslint-disable-line react-hooks/exhaustive-deps
  const act = async (kind, snapshot) => {
    setSt(x => ({ ...x, busy: true, msg: '' }));
    try { const r = await request(base, `/api/live/${kind}`, { method: 'POST',
      ...(snapshot?{body:JSON.stringify({expected_snapshot:snapshot})}:{}),signal:AbortSignal.timeout(30000) }); setSt(x => ({ ...x, msg: r.message || '' })); }
    catch (e) { setSt(x => ({ ...x, msg: e.message })); }
    await refresh(); setSt(x => ({ ...x, busy: false }));
  };
  return { ...st, status:st.base===base?st.status:null, load: snapshot => act('load',snapshot), unload: () => act('unload'), refresh };
}

export async function runOnce(base, text, expectedSnapshot, promptFormat='chat') {
  const body = { text, max_tokens: 1024, expected_snapshot:expectedSnapshot, ...(promptFormat==='rendered'?{prompt_format:promptFormat}:{}) };
  const d = await request(base, '/api/live/run', { method: 'POST', body: JSON.stringify(body) });
  if (expectedSnapshot && d.model_snapshot !== expectedSnapshot) {
    throw Error('采集轨迹与当前模型快照不一致。请选择与本机权重匹配的模型档案。');
  }
  if (promptFormat==='rendered' && (d.prompt!==text || d.provenance?.prompt_format!=='rendered')) {
    throw Error('本机服务未保留归档原始输入。请更新本机实时采集服务后重试。');
  }
  const pos = d.readout_positions.at(-1);
  if (pos !== d.tokens.length - 1 || pos < 0) throw Error('轨迹缺少最终输入位置的读出。');
  const bundle = await request(base, `/api/traces/${d.trace_id}/bundle?position=${pos}`);
  if (bundle.trace_id !== d.trace_id || bundle.model_snapshot !== d.model_snapshot || bundle.position !== pos) {
    throw Error('读出的轨迹、模型或位置与本次采集不一致。');
  }
  const { a, shape: [layers, width] } = f32(bundle.hs);
  if (layers !== d.n_layers + 1 || width !== d.d_model || a.length !== layers * width || !a.every(Number.isFinite)) {
    throw Error('最终位置的逐层 HiddenState 缺失或损坏。');
  }
  const hidden = Array.from({ length: layers }, (_, l) => a.subarray(l * width, (l + 1) * width));
  return { id: d.trace_id, base, snapshot: d.model_snapshot, bundle, hidden, tokens: d.tokens, readout: d.readout_positions, top: d.output_top || [], curve: rewriteCurve(hidden), elapsed: d.elapsed_ms };
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
