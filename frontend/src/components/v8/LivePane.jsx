/* 研究工作台 · 空间 · 实时分析：输入一段文本，在本机中心上跑一次真实前向传播，
   逐层步进观察每个注意力头与 MLP 单元的响应和参数（「下一步」从嵌入层走到输出头）。
   数据来自 /api/live/*（live_trace.v1 / live_block.v1），模型权重从本机加载（AI2050_LIVE_MODEL_DIR，
   与自然语料普查同款 Qwen3-4B：36 块 × 32 头 × d_ff 9728）。
   本机没有权重、或处于只读网页模式时显示诚实空态，不做任何模拟。
   神经元语义与普查打通：块的 MLP 前列单元会标出普查「已解释 / 模板选择性」标记（/api/neurons/layer）。 */
import { useEffect, useMemo, useRef, useState } from 'react';

import { API_BASE } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { GRADE_COLOR, seq } from './v8Util.js';

async function api(base, path, opts) {
  const res = await fetch(`${base}${path}`, {
    cache: 'no-store',
    ...(opts?.body ? { headers: { 'Content-Type': 'application/json' } } : {}),
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`);
  return data;
}

/* 前向服务候选：当前中心 → 本机 5001（完整中心）→ 本机 5002（live-only 轻量服务）。
   「实时分析」不该受「客户端连的是哪个中心」约束——任何本机有前向能力的端口都自动接上。 */
const LIVE_BASES = ['http://127.0.0.1:5001', 'http://127.0.0.1:5002'];

/* 带符号热图颜色：v ∈ [-1, 1]，正 = 红、负 = 蓝（与平台「涨红跌绿」无关，这里是参数方向） */
function heat(v) {
  const a = Math.max(0, Math.min(1, Math.abs(v)));
  return v >= 0 ? `rgba(214,64,69,${0.06 + 0.84 * a})` : `rgba(42,120,214,${0.06 + 0.84 * a})`;
}

function Bars({ values, highlight = -1, format }) {
  const max = Math.max(...values, 1e-9);
  return (
    <div className="lv-bars">
      {values.map((v, i) => (
        <div key={i} className={'lv-bar' + (i === highlight ? ' on' : '')} title={`#${i} = ${v}`}>
          <i style={{ height: `${Math.max(2, (v / max) * 100)}%` }} />
          {format && <span>{format(i)}</span>}
        </div>
      ))}
    </div>
  );
}

function Heatmap({ rows, label }) {
  const flat = rows.flat();
  const max = Math.max(...flat.map(Math.abs), 1e-9);
  return (
    <div className="lv-heat" title={label}>
      <b>{label}</b>
      <div className="lv-heat-g" style={{ gridTemplateColumns: `repeat(${rows[0]?.length || 1}, 1fr)` }}>
        {rows.flat().map((v, i) => <i key={i} style={{ background: heat(v / max) }} />)}
      </div>
    </div>
  );
}

const stepLabel = (step, L) => (step === 0 ? '嵌入层 hs[0]' : step <= L ? `块 ${step - 1}（输出 hs[${step}]）` : '输出头');

export default function LivePane({ census, trace, setTrace, step, setStep }) {
  const [text, setText] = useState('The capital of France is');
  const [maxTokens, setMaxTokens] = useState(24);
  const [status, setStatus] = useState(null);
  const [liveBase, setLiveBase] = useState(null);  // 实际提供前向服务的基址（可能不是当前中心）
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState('');
  const [head, setHead] = useState(null);          // 选中的注意力头（null = 只看总览）
  const [tokenIdx, setTokenIdx] = useState(null);  // 选中的 token 位置（null = 最后一个）
  const [blockDoc, setBlockDoc] = useState(null);  // 当前块参数缩略图
  const [layerCensus, setLayerCensus] = useState(null);  // 当前块普查数据（已解释 / 选择性）
  const [playing, setPlaying] = useState(false);   // 自动逐层播放：3D 层塔随之逐层点亮
  const playTimer = useRef(null);

  const L = trace?.n_layers || 0;
  const T = trace?.n_tokens || 0;
  const tokIdx = tokenIdx != null && tokenIdx < T ? tokenIdx : Math.max(0, T - 1);

  /* 状态探测：按候选依次找「有实时前向能力」的服务（ready/loading 即用）；全失败时分类报错并每 5 秒重试。
     加载中每 2 秒复查；手动「重试」按钮触发 probeTick。
     不按运行形态设门槛：当前中心、本机 5001/5002 谁有权重谁上——连线上中心的客户端也能纯本地跑前向。 */
  const [probeTick, setProbeTick] = useState(0);
  useEffect(() => {
    let dead = false;
    let timer = null;
    const tick = async () => {
      const bases = [...new Set([API_BASE, ...LIVE_BASES])];
      let firstDoc = null;                       // 第一个有响应但不 ready 的（用于分类失败原因）
      for (const base of bases) {
        let d;
        try { d = await api(base, '/api/live/status'); } catch { continue; }
        if (d.status === 'ready' || d.status === 'loading') {
          if (dead) return;
          setLiveBase(base);
          setStatus(d);
          if (d.status === 'loading') timer = setTimeout(tick, 2000);
          return;
        }
        if (!firstDoc) firstDoc = { base, doc: d };
      }
      if (dead) return;
      setLiveBase(null);
      if (firstDoc && typeof firstDoc.doc.available === 'boolean') {
        setStatus(firstDoc.doc);                 // 明确不可用：中心有前向服务但权重缺失等
      } else if (firstDoc) {
        setStatus({ available: false, status: 'incompatible',
                    reason: `${firstDoc.base} 返回了旧版只读状态接口（无实时前向能力），本机 5001/5002 也没有可用前向服务` });
      } else {
        setStatus({ available: false, status: 'unreachable',
                    reason: `当前中心与本机 5001/5002 都连不上`, tried: bases.join(' → '), retrying: true });
        timer = setTimeout(tick, 5000);
      }
    };
    void tick();
    return () => { dead = true; if (timer) clearTimeout(timer); };
  }, [probeTick]);

  /* 自动播放：每 0.9 秒推进一层，到输出头停；手动步进不打断 */
  useEffect(() => {
    if (!playing || !trace) return undefined;
    if (step >= trace.n_layers + 1) { setPlaying(false); return undefined; }
    playTimer.current = setTimeout(() => setStep(s => Math.min(s + 1, trace.n_layers + 1)), 900);
    return () => clearTimeout(playTimer.current);
  }, [playing, step, trace, setStep]);

  /* 当前块：参数缩略图 + 普查数据（已解释等级、选择性前列），切层即取 */
  const b = step >= 1 && step <= L ? step - 1 : null;
  useEffect(() => {
    setBlockDoc(null);
    setLayerCensus(null);
    if (b == null || !trace) return undefined;
    let dead = false;
    const fwd = liveBase || API_BASE;
    api(fwd, `/api/live/block?block=${b}`).then(d => { if (!dead) setBlockDoc(d); }).catch(() => {});
    if (census) {
      api(API_BASE, `/api/neurons/layer?model_key=${encodeURIComponent(census)}&block=${b}`)
        .then(d => { if (!dead && !d.pending) setLayerCensus(d); }).catch(() => {});
    }
    return () => { dead = true; };
  }, [b, trace, census, liveBase]);

  const run = async () => {
    setRunning(true);
    setRunError('');
    setHead(null);
    try {
      const d = await api(liveBase || API_BASE, '/api/live/run',
                          { method: 'POST', body: JSON.stringify({ text, max_tokens: maxTokens }) });
      setTrace(d);
      setTokenIdx(null);
      setStep(0);                                   // 从嵌入层开始步进
    } catch (reason) {
      setRunError(String(reason.message || reason));
    } finally {
      setRunning(false);
    }
  };

  /* 块内归一化用的最大值 */
  const headMax = useMemo(() => (b != null && trace
    ? Math.max(...trace.layers[b].heads.out_norm.flat(), 1e-9) : 1), [b, trace]);
  const explained = layerCensus?.explained || {};
  const selUnits = useMemo(() => {
    const s = new Set();
    for (const run_ of layerCensus?.selectivity || []) {
      for (const units of Object.values(run_.top || {})) for (const [u] of units) s.add(u);
    }
    return s;
  }, [layerCensus]);

  const st = status?.status;
  if (st === 'unreachable' || st === 'unavailable' || st === 'error' || st === 'incompatible') {
    return (
      <Empty>
        {st === 'unreachable' && (<>实时分析不可用：当前中心（{API_BASE || '同源 ' + window.location.origin}）与本机 5001/5002 都连不上。
          实时分析要跑一次真实前向——需要本机模型权重（AI2050_LIVE_MODEL_DIR，缺省 D:/develop/model/Qwen/Qwen3-4B）；
          桌面客户端启动时会自动拉起本机轻量前向服务（5002）。
          {status.retrying ? '每 5 秒自动重试中…' : ''}</>)}
        {st === 'incompatible' && `实时分析不可用：${status.reason}。`}
        {(st === 'unavailable' || st === 'error')
          && `实时分析不可用：${status.reason || status.error || '前向服务没有响应'}。需要本机模型权重（AI2050_LIVE_MODEL_DIR，缺省 D:/develop/model/Qwen/Qwen3-4B）。`}
        <div style={{ marginTop: 8 }}>
          <button type="button" className="lv-run" onClick={() => setProbeTick(t => t + 1)}>重试</button>
        </div>
      </Empty>
    );
  }

  const layer = b != null && trace ? trace.layers[b] : null;
  const isFinal = trace && step === L + 1;

  return (
    <div className="lv-root">
      <div className="lv-hd">
        <p className="v8-nprog-say">
          输入一段文本，看模型逐块怎么算：每个注意力头写了什么、MLP 哪些单元被点亮、参数长什么样。
          {st === 'ready' && <>当前 <b>{status.model_name}</b>（{status.device === 'cuda' ? 'GPU' : 'CPU'}）就绪，「▶ 播放」自动逐层推进。</>}
          {st === 'loading' && <>{status.progress || '模型加载中…'}</>}
        </p>
      </div>

      <div className="lv-input">
        <textarea value={text} rows={2} onChange={e => setText(e.target.value)}
                  placeholder="输入要分析的文本…" aria-label="分析文本" />
        <div className="lv-input-tools">
          <label>最多 token
            <select value={maxTokens} onChange={e => setMaxTokens(Number(e.target.value))} aria-label="最大 token 数">
              {[8, 16, 24, 32].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <button type="button" className="lv-run" disabled={running || st !== 'ready' || !text.trim()} onClick={run}>
            {running ? '前向计算中…' : '运行分析'}
          </button>
          {runError && <span className="lv-err" role="alert">{runError}</span>}
        </div>
        {trace && (
          <div className="lv-tokens">
            {trace.tokens.map((t, i) => (
              <button key={i} type="button" className={'lv-tok' + (i === tokIdx ? ' on' : '')}
                      title={`#${i} · 嵌入范数 ${trace.layers[0].resid_in[i]}`}
                      onClick={() => setTokenIdx(i)}>{t}</button>
            ))}
            {trace.truncated && <span className="lv-note">（已截断到前 {maxTokens} 个 token）</span>}
          </div>
        )}
      </div>

      {!trace && st === 'ready' && <Empty>输入文本并点击「运行分析」，从这里开始逐层观察模型怎么算。</Empty>}
      {st !== 'ready' && !trace && <Empty>模型加载中，请稍候…</Empty>}

      {trace && (
        <>
          <div className="lv-steps">
            <button type="button" className="lv-play" disabled={!trace} onClick={() => {
              if (playing) setPlaying(false);
              else { if (step >= L + 1) setStep(0); setPlaying(true); }     // 在终点按播放：从头再走一遍
            }} title="自动逐层播放：左侧 3D 层塔会逐层点亮（每层 0.9 秒）">{playing ? '⏸ 暂停' : '▶ 播放'}</button>
            <button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>← 上一步</button>
            <input type="range" min={0} max={L + 1} value={step} aria-label="层步进"
                   onChange={e => setStep(Number(e.target.value))} />
            <button type="button" className="primary" disabled={step === L + 1} onClick={() => setStep(step + 1)}>下一步 →</button>
            <span className="lv-pos">{stepLabel(step, L)}{step <= L && step > 0 ? ` · 第 ${step} / ${L + 1} 步` : step === 0 ? ' · 起点' : ' · 终点'}</span>
          </div>

          {step === 0 && (
            <div className="lv-card">
              <h4>嵌入层 hs[0]：文本变成向量</h4>
              <p>每个 token 被查表变成 {trace.d_model} 维向量。条形 = 各 token 嵌入向量的范数（长度）；点击上方 token 查看它的旅程。下一步进入块 0。</p>
              <Bars values={trace.layers[0].resid_in} highlight={tokIdx} />
            </div>
          )}

          {layer && (
            <div className="lv-card">
              <h4>块 {b}：32 个注意力头</h4>
              <p>每格 = 一个头。颜色 = 该头对这段输入的写出能量（输出向量范数，块内归一化）；点击选中，下方展开它的注意力分布与参数。</p>
              <div className="lv-heads">
                {layer.heads.out_norm.map((row, h) => {
                  const m = row.reduce((a, x) => a + x, 0) / (T || 1);
                  return (
                    <button key={h} type="button" className={'lv-head' + (head === h ? ' on' : '')}
                            style={{ background: seq(m / headMax) }} title={`头 ${h} · 平均写出范数 ${m.toFixed(2)}`}
                            onClick={() => setHead(head === h ? null : h)}>{h}</button>
                  );
                })}
              </div>
              {head != null && (
                <div className="lv-head-detail">
                  <div className="lv-head-col">
                    <b>头 {head} 对最后 token「{trace.tokens[tokIdx]}」的注意力</b>
                    <Bars values={layer.heads.attn_last[head]} highlight={tokIdx} format={i => trace.tokens[i]} />
                    <ul className="lv-pairs">
                      {layer.heads.top_pairs[head].map(([i, j, w], k) => (
                        <li key={k}>{trace.tokens[i] || '(起点)'} → {trace.tokens[j]}：<b>{w}</b></li>
                      ))}
                    </ul>
                  </div>
                  <div className="lv-head-col">
                    {blockDoc ? (
                      <>
                        <b>头 {head} 的参数缩略图（均值池化 {blockDoc.heads[head].q.length}×{blockDoc.heads[head].q[0].length}）</b>
                        <div className="lv-heat-row">
                          <Heatmap rows={blockDoc.heads[head].q} label="W_q（查询）" />
                          <Heatmap rows={blockDoc.heads[head].o} label="W_o（写出）" />
                        </div>
                        <p className="lv-note">红 = 正、蓝 = 负，深浅 = 幅度。这是 128×{trace.d_model} 实矩阵的缩略鸟瞰；KV 头只有 {trace.n_kv_heads} 个（GQA），多个查询头共享。</p>
                      </>
                    ) : <p className="lv-note">读取参数缩略图…</p>}
                  </div>
                </div>
              )}
            </div>
          )}

          {layer && (
            <div className="lv-card">
              <h4>块 {b}：MLP 神经元（d_ff = {trace.d_ff}）</h4>
              <p>对 token「{trace.tokens[tokIdx]}」激活最强的 {layer.mlp.top[tokIdx]?.length || 0} 个中间单元。标出普查结果：<i style={{ color: GRADE_COLOR['文本可复现'] }}>■ 已解释</i> · <i style={{ color: GRADE_COLOR['只在语料中可识别'] }}>■ 语料可识别</i> · <b>S</b> = 模板选择性前列。</p>
              <div className="lv-units">
                {(layer.mlp.top[tokIdx] || []).map(([u, v]) => {
                  const g = explained[u];
                  return (
                    <div key={u} className="lv-unit" title={`单元 ${u} · 激活 ${v}${g ? ` · 普查：${g}` : ''}`}>
                      <span className="lv-unit-id">#{u}</span>
                      <span className="lv-unit-bar"><i style={{ width: `${Math.min(100, Math.abs(v) / Math.max(...(layer.mlp.top[tokIdx] || [[0, 1]]).map(x => Math.abs(x[1]))) * 100)}%`,
                        background: v >= 0 ? seq(0.75) : 'rgba(42,120,214,0.85)' }} /></span>
                      <span className="lv-unit-val">{v}</span>
                      <span className="lv-unit-marks">
                        {g && <i className="lv-mark" style={{ background: GRADE_COLOR[g] || '#94a3b8' }} title={g} />}
                        {selUnits.has(u) && <b className="lv-mark-s" title="模板选择性前列">S</b>}
                      </span>
                    </div>
                  );
                })}
              </div>
              {blockDoc && (
                <div className="lv-heat-row">
                  <Heatmap rows={blockDoc.mlp.gate} label="W_gate" />
                  <Heatmap rows={blockDoc.mlp.up} label="W_up" />
                  <Heatmap rows={blockDoc.mlp.down} label="W_down" />
                </div>
              )}
              <p className="lv-note">残差流入范数（该块接收到的向量长度，逐 token）：</p>
              <Bars values={layer.resid_in} highlight={tokIdx} />
            </div>
          )}

          {isFinal && (
            <div className="lv-card">
              <h4>输出头：预测下一个词</h4>
              <p>最终隐状态经输出投影得到全词表打分。下方是最后一个 token 位置的前 8 名（softmax 概率）。</p>
              <div className="lv-logits">
                {trace.output.last_top.map(([t, p], i) => (
                  <div key={i} className="lv-logit">
                    <span className="lv-unit-id">{t || '␣'}</span>
                    <span className="lv-unit-bar"><i style={{ width: `${p * 100}%`, background: seq(0.75) }} /></span>
                    <span className="lv-unit-val">{(p * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <p className="lv-note">每个位置模型的接续（top-1）：{trace.output.per_pos_top1.map((t, i) => `${trace.tokens[i] || '␣'}→${t}`).join(' · ')}</p>
              <p className="lv-note">旅程结束——36 个块里，注意力头负责「从别的 token 搬信息」，MLP 负责「加工与记忆」。在「神经元」tab 里可以深入普查这些单元的长期统计。</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
