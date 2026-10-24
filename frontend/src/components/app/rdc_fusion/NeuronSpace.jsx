/* 封存激活 · 神经元 3D（neuron3d.v1，design/atlas_neuron3d_scheme_v1.md）
   数据 = GET /api/ai-rnd/atlas/neuron3d/{model}（grid/白名单/npz 清单，派生器产物）
   单位 = 残差流坐标 (layer,dim)；**不是 MLP 神经元**（残差基由模型自选，单义性另论，见因子构成条）
        + GET /api/ai-rnd/atlas/neuron/{model}/{layer}/slice（单层激活切片，median over seed×row）
        + GET /api/ai-rnd/atlas/neuron/{model}/{layer}/{dim}（单神经元：统计 + 正/负极 + registry 回链）
   场景：Y=层楼层（当前层实心网格 + 相邻层幽灵楼板），X/Z=原生坐标索引网格；
        颜色 = 激活值发散色带（负=琥珀 / 0=浅灰 / 正=emerald，p10–p90 归一）。
   纪律（scheme §0/§7）：前端零标签生成——未入册点只显示数字；feature 标记 = 特征级 registry 回链，
   不声称占据神经元坐标；位置恒注记「坐标索引，非模型真实几何」。 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE } from '../../../config.js';

const MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b'];

function colorFor(v, p10, p90) {
  /* 发散色带（浅色主题）：0 点锚定浅灰，正向 emerald 加深，负向琥珀加深 */
  const posMax = Math.max(1e-6, p90);
  const negMax = Math.max(1e-6, -p10);
  const mix = (a, b, t) => a.map((c, i) => Math.round(c + (b[i] - c) * t));
  const GRAY = [226, 232, 240], GREEN = [4, 120, 87], AMBER = [180, 83, 9];
  let rgb;
  if (v >= 0) rgb = mix(GRAY, GREEN, Math.min(1, Math.sqrt(v / posMax)));
  else rgb = mix(GRAY, AMBER, Math.min(1, Math.sqrt(-v / negMax)));
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

/* 因子构成条：全交叉方差分解（模板 / 类别 / 实体 / 交互），份额之和为 1。
   4 类因子固定配色；数值全部来自服务端分解，前端只做展示，不做任何再解释。 */
const FACTOR_ROWS = [
  ['t', '句式(模板)', '#0369a1'],
  ['c', '类别(内容)', '#1e5c22'],
  ['e', '实体(身份)', '#8a4b16'],
  ['x', '交互/组合', '#94a3b8'],
];
function FactorBar({ title, mean, frac, nDims, compact }) {
  if (!mean) return null;
  const v = { t: mean.t || 0, c: mean.c || 0, e: mean.e || 0,
    x: (mean.te || 0) + (mean.tc || 0) + (mean.ec || 0) + (mean.tec || 0) };
  const total = v.t + v.c + v.e + v.x || 1;
  return (
    <div style={{ marginBottom: compact ? 8 : 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <b style={{ fontSize: 11.5 }}>{title}</b>
        {title ? <span style={{ fontSize: 10, color: '#94a3b8' }}>{nDims} dims</span> : null}
      </div>
      <div style={{ display: 'flex', height: 7, borderRadius: 4, overflow: 'hidden', margin: '4px 0 3px', background: '#eef1f5' }}>
        {FACTOR_ROWS.map(([k, lb, col]) => (
          <span key={k} style={{ width: `${(100 * v[k]) / total}%`, background: col }}
            title={`${lb} ${(100 * v[k]).toFixed(1)}%`} />
        ))}
      </div>
      <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.5 }}>
        {FACTOR_ROWS.map(([k, lb, col]) => (
          <span key={k} style={{ marginRight: 7 }}>
            <i style={{ display: 'inline-block', width: 6, height: 6, borderRadius: 2, background: col, marginRight: 3 }} />
            {lb} {(100 * v[k]).toFixed(0)}%
          </span>
        ))}
      </div>
      {frac !== undefined && (
        <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2, lineHeight: 1.45 }}>
          类别份额 &gt; 0.5 的坐标占比 {(100 * frac).toFixed(2)}%（内容单义坐标的直接计数）
        </div>
      )}
    </div>
  );
}

export default function NeuronSpace() {
  const [model, setModel] = useState('qwen3-4b');
  const [doc, setDoc] = useState(null);
  const [err, setErr] = useState('');
  const [layer, setLayer] = useState(12);
  const [slice, setSlice] = useState(null);      // {values,p10,p90,n_dims}
  const [sliceState, setSliceState] = useState('idle'); // idle|loading|ready|fail
  const [sel, setSel] = useState(null);          // {layer,dim}
  const [detail, setDetail] = useState(null);
  const [detailErr, setDetailErr] = useState('');
  const [jump, setJump] = useState('');
  const [hover, setHover] = useState(null);
  const [tickV, setTickV] = useState(0);          // 相机变化 → 重绘
  const bump = () => setTickV(t => t + 1);
  const cvRef = useRef(null);
  const cam = useRef({ yaw: 0.55, pitch: 0.42, zoom: 1, drag: null, moved: false, panX: 0, panY: 0 });
  const projected = useRef([]);

  /* neuron3d 派生档（grid/白名单/npz 清单） */
  useEffect(() => {
    let dead = false; setDoc(null); setErr(''); setSel(null); setDetail(null); setSlice(null);
    fetch(`${API_BASE}/api/ai-rnd/atlas/neuron3d/${model}`).then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(d => { if (!dead) setDoc(d); }).catch(e => { if (!dead) setErr(String(e.message || e)); });
    return () => { dead = true; };
  }, [model]);

  const grid = doc?.grid;                         // {n_layers,d_model,n_rows,n_templates}
  const npz = doc?.npz;
  const nLayers = grid?.n_layers || 0;
  const nDims = grid?.d_model || 0;
  const nTpl = grid?.n_templates || grid?.n_seeds || 0;   // 旧派生档字段名为 n_seeds（实为模板轴）
  useEffect(() => { if (nLayers && layer >= nLayers) setLayer(Math.floor(nLayers / 2)); }, [nLayers]);

  /* 单层切片 */
  useEffect(() => {
    if (!nLayers) return;
    let dead = false; setSliceState('loading');
    fetch(`${API_BASE}/api/ai-rnd/atlas/neuron/${model}/${layer}/slice`).then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(d => { if (!dead) { setSlice(d); setSliceState('ready'); } })
      .catch(() => { if (!dead) setSliceState('fail'); });
    return () => { dead = true; };
  }, [model, layer, nLayers]);

  /* 选中 → 单神经元详情 */
  useEffect(() => {
    if (!sel) { setDetail(null); setDetailErr(''); return; }
    let dead = false; setDetailErr('');
    fetch(`${API_BASE}/api/ai-rnd/atlas/neuron/${model}/${sel.layer}/${sel.dim}`).then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(d => { if (!dead) setDetail(d); })
      .catch(e => { if (!dead) setDetailErr(String(e.message || e)); });
    return () => { dead = true; };
  }, [sel, model]);

  /* 读出层标记（该模型）与特征标记 */
  const readoutLayer = useMemo(() => {
    const p = (doc?.points || []).find(p => p.kind === 'readout_layer' && p.model === model);
    return p ? p.layer : null;
  }, [doc, model]);
  const features = useMemo(() => (doc?.points || []).filter(p => p.kind === 'feature'), [doc]);

  /* ── 渲染：canvas 伪 3D ── */
  useEffect(() => {
    const cv = cvRef.current;
    if (!cv || !nDims || sliceState !== 'ready') return;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { yaw, pitch, zoom, panX, panY } = cam.current;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const scale = Math.min(W, H) * 0.42 * zoom;
    const proj = (x, y, z) => {
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const f = 6 / (6 + z2);
      return [W / 2 + (x1 * scale + panX) * f, H / 2 - (y1 * scale + panY) * f, z2];
    };
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, W, H);

    /* 楼板（当前层 ±1/±2 幽灵层） */
    const S = 1.0, floorH = 0.34;
    const corner = (x, y, z) => proj(x * S - S / 2, y * floorH * 4, z * S * 0.72 - S * 0.36);
    for (const off of [-2, -1, 1, 2]) {
      const ly = layer + off;
      if (ly < 0 || ly >= nLayers) continue;
      const c = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([a, b]) => corner(a, off, b));
      ctx.strokeStyle = off === 0 ? 'rgba(15,23,42,0.5)' : 'rgba(148,163,184,0.28)';
      ctx.lineWidth = 1;
      ctx.beginPath(); c.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.stroke();
      const lab = proj(-S / 2 - 0.06, off * floorH * 4, -S * 0.36);
      ctx.fillStyle = 'rgba(100,116,139,0.8)'; ctx.font = '10px ui-monospace,Consolas,monospace';
      ctx.fillText('L' + ly, lab[0] - 24, lab[1] + 3);
    }
    /* 读出层标记：该层楼板描金 */
    if (readoutLayer != null && Math.abs(readoutLayer - layer) <= 2 && readoutLayer !== layer) {
      const off = readoutLayer - layer;
      const c = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([a, b]) => corner(a, off, b));
      ctx.strokeStyle = '#b45309'; ctx.lineWidth = 2; ctx.setLineDash([6, 3]);
      ctx.beginPath(); c.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.stroke();
      ctx.setLineDash([]);
    }

    /* 当前层神经元网格：64 列 × ceil(n/64) 行 */
    const cols = Math.ceil(Math.sqrt(nDims * 1.9));
    const ptList = [];
    const vals = slice.values, p10arr = slice.p10, p90arr = slice.p90;
    /* 全层统一色标（分位聚合，保证层内可比） */
    const sP10 = [...p10arr].sort((a, b) => a - b)[Math.floor(p10arr.length * 0.02)];
    const sP90 = [...p90arr].sort((a, b) => a - b)[Math.floor(p90arr.length * 0.98)];
    for (let d = 0; d < nDims; d++) {
      const gx = d % cols, gz = Math.floor(d / cols);
      const x = (gx / (cols - 1) - 0.5) * S;
      const z = (gz / (Math.ceil(nDims / cols) - 1) - 0.5) * S * 0.72;
      const q = proj(x, 0, z);
      ptList.push({ dim: d, sx: q[0], sy: q[1], depth: q[2] });
    }
    ptList.sort((a, b) => b.depth - a.depth);
    for (const p of ptList) {
      const v = vals[p.dim];
      const near = sel && sel.dim === p.dim;
      ctx.fillStyle = colorFor(v, sP10, sP90);
      ctx.globalAlpha = near ? 1 : 0.92;
      ctx.beginPath(); ctx.arc(p.sx, p.sy, near ? 4.4 : 2.1, 0, Math.PI * 2); ctx.fill();
      if (near) {
        ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(p.sx, p.sy, 7.4, 0, Math.PI * 2); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    /* 层号标签 */
    ctx.fillStyle = '#0f172a'; ctx.font = '600 12px ui-monospace,Consolas,monospace';
    ctx.fillText(`L${layer} · ${nDims} dims · median over ${nTpl}×${grid.n_rows}`, 12, 20);
    projected.current = ptList;
  }, [doc, model, layer, slice, sliceState, nDims, nLayers, sel, readoutLayer, tickV]);

  if (err) return <div className="v6-empty" style={{ padding: 24 }}>封存激活 3D 不可用：{err}（服务端派生档缺失时运行 python server/derive_neuron3d.py）</div>;
  if (!doc) return <div className="v6-empty" style={{ padding: 24 }}>读取 neuron3d 派生档…</div>;

  const nearest = e => {
    const box = cvRef.current.getBoundingClientRect();
    const mx = e.clientX - box.left, my = e.clientY - box.top;
    let best = null, bd = 100;
    for (const it of projected.current) {
      const d = (it.sx - mx) ** 2 + (it.sy - my) ** 2;
      if (d < bd) { bd = d; best = it; }
    }
    return { best, mx, my };
  };
  const onDown = e => { cam.current.drag = { x: e.clientX, y: e.clientY, yaw: cam.current.yaw, pitch: cam.current.pitch, panX: cam.current.panX, panY: cam.current.panY, btn: e.button }; cam.current.moved = false; };
  const onMove = e => {
    const d = cam.current.drag;
    if (d) {
      if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 3) cam.current.moved = true;
      if (d.btn === 2) { cam.current.panX = d.panX + (e.clientX - d.x); cam.current.panY = d.panY + (e.clientY - d.y); }
      else {
        cam.current.yaw = d.yaw + (e.clientX - d.x) * 0.008;
        cam.current.pitch = Math.max(-1.3, Math.min(1.3, d.pitch + (e.clientY - d.y) * 0.008));
      }
      bump();
      return;
    }
    if (sliceState === 'ready') {
      const { best, mx, my } = nearest(e);
      setHover(best ? { x: mx, y: my, dim: best.dim } : null);
    }
  };
  const onUp = e => {
    const d = cam.current.drag; cam.current.drag = null;
    if (d && !cam.current.moved && sliceState === 'ready') {
      const { best } = nearest(e);
      setSel(best ? { layer, dim: best.dim } : null);
    }
  };
  const doJump = () => {
    const m = String(jump || '').trim().match(/^L(\d+)[\s.,:D]*D?(\d+)$/i);
    if (!m) return;
    const L = +m[1], D = +m[2];
    if (L >= 0 && L < nLayers && D >= 0 && D < nDims) { setLayer(L); setSel({ layer: L, dim: D }); }
  };

  const st = detail?.stats;
  return (
    <div style={{ display: 'flex', gap: 10, height: '100%', minHeight: 480 }}>
      <div style={{ flex: 1, position: 'relative', border: '1px solid #e2e8f0', borderRadius: 10, overflow: 'hidden', background: '#f8fafc' }}>
        <div style={{ position: 'absolute', top: 8, left: 10, right: 10, display: 'flex', gap: 8, alignItems: 'center', zIndex: 2 }}>
          <select value={model} onChange={e => setModel(e.target.value)} style={{ fontSize: 12 }}>
            {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
          <input value={jump} onChange={e => setJump(e.target.value)} onKeyDown={e => e.key === 'Enter' && doJump()}
            placeholder="跳转 L12.D758" style={{ fontSize: 12, width: 120 }} />
          <button type="button" className="fw-tbtn" onClick={doJump}>跳转</button>
          <input type="range" min={0} max={nLayers - 1} value={layer} onChange={e => setLayer(+e.target.value)} style={{ flex: 1 }} aria-label="层滑块" />
          <b className="fw-mono" style={{ fontSize: 12 }}>L{layer}</b>
        </div>
        <canvas ref={cvRef}
          onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp}
          onMouseLeave={() => { cam.current.drag = null; setHover(null); }}
          onWheel={e => { e.preventDefault(); cam.current.zoom = Math.max(0.35, Math.min(6, cam.current.zoom * (e.deltaY > 0 ? 0.9 : 1.1))); bump(); }}
          onContextMenu={e => e.preventDefault()}
          style={{ width: '100%', height: '100%', display: 'block', cursor: hover ? 'pointer' : 'grab' }}
          role="img" aria-label={`封存激活 3D：${model} L${layer}，${nDims} 个残差坐标`} />
        {hover && slice && (
          <div className="v6-tip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
            {`D${hover.dim} · median ${slice.values[hover.dim]}`}{` · p90 ${slice.p90[hover.dim]}`}
          </div>
        )}
        {sliceState === 'loading' && <div className="v6-empty" style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(248,250,252,0.75)' }}>加载封存激活切片…（首次加载 npz 约 1–3 秒）</div>}
        {sliceState === 'fail' && <div className="v6-empty" style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>切片加载失败（服务未启动或 npz 缺失）</div>}
        <div style={{ position: 'absolute', bottom: 8, left: 12, right: 12, fontSize: 10.5, color: '#64748b', display: 'flex', justifyContent: 'space-between', gap: 8 }}>
          <span>位置 = 原生坐标索引（不重排），非模型真实几何 · 数据 = phase3169 封存 collect npz{npz?.file_sha256_8 ? ` · sha8 ${npz.file_sha256_8}` : ''}</span>
          <span>左键旋转 · 右键平移 · 滚轮缩放 · 点击坐标看证据</span>
        </div>
      </div>

      {/* ── 详情卡：概览 / 极点 / 电路 / 跨模型 ── */}
      <aside style={{ width: 340, flexShrink: 0, border: '1px solid #e2e8f0', borderRadius: 10, padding: '10px 12px', overflowY: 'auto', background: '#fff' }}>
        {slice?.factor_mean && <FactorBar title={`本层因子构成 · L${layer}`} mean={slice.factor_mean.mean}
          frac={slice.factor_mean.frac_cs_gt_0_5} nDims={slice.factor_mean.n_dims} compact />}
        {!sel && <div className="v6-empty">点击场景中的残差坐标查看证据；或在上方输入 L层数.D维度 跳转。<br /><br />本视图画的是<b>残差流坐标 (layer, dim)</b>，不是 MLP 神经元——两者的单义性不同，见右侧因子构成。纪律注记：前端零标签生成，未入册坐标只显示数字证据。</div>}
        {sel && !detail && !detailErr && <div className="v6-empty">读取 D{sel.dim} 证据…</div>}
        {sel && detailErr && <div className="v6-empty">读取失败：{detailErr}</div>}
        {sel && detail && (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <b className="fw-mono">{model} · L{detail.layer}.D{detail.dim}</b>
              <button type="button" className="fw-tbtn ghost" onClick={() => setSel(null)} aria-label="关闭">×</button>
            </div>
            <div style={{ fontSize: 11, color: '#64748b', margin: '2px 0 8px' }}>{detail.coords_meaning}</div>
            <table className="v6-mini-table">
              <tbody>
                <tr><th>min</th><td className="num">{st.min}</td><th>p10</th><td className="num">{st.p10}</td></tr>
                <tr><th>median</th><td className="num">{st.median}</td><th>mean</th><td className="num">{st.mean}</td></tr>
                <tr><th>p90</th><td className="num">{st.p90}</td><th>max</th><td className="num">{st.max}</td></tr>
                <tr><th>样本池</th><td className="num" colSpan={3}>{st.n_templates || st.n_seeds} 模板 × {st.n_rows} 行（phase3169 封存）</td></tr>
              </tbody>
            </table>
            {detail.factor && (
              <>
                <h5 className="v6-h5" style={{ marginTop: 10 }}>因子构成（这一维在做什么）</h5>
                {detail.factor.degenerate
                  ? <div style={{ fontSize: 11, color: '#64748b' }}>{detail.factor.reading}</div>
                  : <FactorBar title="" nDims={1} mean={detail.factor.shares} />}
                {!detail.factor.degenerate && (
                  <div style={{ fontSize: 10, color: '#94a3b8', lineHeight: 1.45 }}>{detail.factor.reading}</div>
                )}
              </>
            )}
            <h5 className="v6-h5" style={{ marginTop: 10 }}>正极 / 负极 · 真实片段</h5>
            {[['正极', detail.poles.positive, '#047857'], ['负极', detail.poles.negative, '#b45309']].map(([name, arr, col]) => (
              <div key={name} style={{ marginBottom: 6 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: col, marginBottom: 3 }}>{name}</div>
                {arr.map((p, pi) => (
                  <div key={name + '-' + pi + '-' + p.row + '-' + p.value}
                    style={{ fontSize: 10.5, marginBottom: 3, paddingLeft: 7, borderLeft: `2px solid ${col}33` }}>
                    <div style={{ lineHeight: 1.4 }}>{p.text || (p.text === '' ? '（空）' : `r${p.row}·t${p.template ?? p.seed}`)}</div>
                    <div className="fw-mono" style={{ fontSize: 9.5, color: '#94a3b8' }}>
                      {p.value}{p.template !== undefined ? ` · 模板 t${p.template}` : ''}{p.diagonal === false ? ' · 类别≠实体真类' : ''}
                    </div>
                  </div>
                ))}
              </div>
            ))}
            <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 3, lineHeight: 1.45 }}>
              {detail.poles.method}。文本来自封存 execution.json 的模板与实体表（零标注）；「类别≠实体真类」标记的是矛盾格，极值常集中在这里。
            </div>
            <h5 className="v6-h5" style={{ marginTop: 10 }}>registry 回链</h5>
            {detail.registered.readout_layer?.length > 0
              ? <div style={{ fontSize: 11.5 }}>本层为 <b>读出层标记</b>（{detail.registered.readout_layer[0].note}）</div>
              : <div style={{ fontSize: 11.5, color: '#64748b' }}>{detail.registered.note}</div>}
            {detail.registered_features_index.length > 0 && (
              <details style={{ marginTop: 6 }}>
                <summary style={{ fontSize: 11, cursor: 'pointer' }}>在册特征索引（{detail.registered_features_index.length}）——特征级标记，无原生坐标</summary>
                {detail.registered_features_index.map(f => (
                  <div key={f.id} style={{ fontSize: 10.5 }}><b className="fw-mono">{f.id}</b> · {f.family} · {f.e_level}</div>
                ))}
              </details>
            )}
            <h5 className="v6-h5" style={{ marginTop: 10 }}>电路 / 跨模型</h5>
            <div style={{ fontSize: 11, color: '#64748b' }}>挂账（scheme §6 步 D/E）：上下游归因边需离线烘焙（down_proj@gate_proj 有效传递），E3 干预结果与三模型并排在研究线产物入册后解锁——当前不伪造数据。</div>
          </>
        )}
      </aside>
    </div>
  );
}
