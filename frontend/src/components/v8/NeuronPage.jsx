/* 主线 ②：神经元。左：3D 层塔（嵌入层在下、输出头在上，中间是全部 Transformer 块），点击选中一层；
   右：该层内部结构（按 config.json）→ MLP 中间单元阵列（9,728 个，按真实索引排列，可显式排序）→ 单元卡。
   层塔与阵列都只画真实数据：没有逐头数据的注意力只画结构；层板与网格的位置只是排列，不代表模型几何。 */
import { useEffect, useMemo, useRef, useState } from 'react';

import NeuronUnitCard from '../app/rdc_fusion/NeuronUnitCard.jsx';
import { useApi } from '../app/rdc_fusion/api.js';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { READOUT_LABEL, seq, shortModel, unitColors } from './v8Util.js';

const DEG = Math.PI / 180;
const OVERLAYS = [['sel', '选择性'], ['expl', '已解释'], ['t1', 'T1 检测器数'], ['dens', '强激活率中位数'], ['none', '无']];
const COLOR_MODES = [['type', '分型'], ['dens', '强激活率'], ['regime', '主导符号区'], ['expl', '解释等级'], ['sel', '选择性']];

function inside(pt, pts) {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

function towerValues(tower, ov, selRun, ro) {
  const n = tower.n_layers;
  if (ov === 'sel') {
    const s = tower.selectivity.find(x => x.file === selRun) || tower.selectivity[0];
    return s ? Array.from({ length: n }, (_, b) => (s.by_readout?.[ro] || s.by_readout?.[Object.keys(s.by_readout)[0]] || [])[b] || 0) : null;
  }
  if (ov === 'expl') return tower.blocks.map(b => Object.values(b.explained || {}).reduce((s, v) => s + v, 0));
  if (ov === 't1') return tower.blocks.map(b => (b.types ? b.types[0] : 0));
  if (ov === 'dens') return tower.blocks.map(b => b.dens_median || 0);
  return null;
}

function Tower({ tower, block, onPick, values }) {
  const cv = useRef(null), hits = useRef([]), drag = useRef(null);
  const [view, setView] = useState({ yaw: -30 * DEG, pitch: 16 * DEG, zoom: 1 });
  const [hover, setHover] = useState(null);
  const [w, setW] = useState(340);
  const H = 600;
  useEffect(() => {
    const el = cv.current?.parentElement;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = H * dpr;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);
    const n = tower.n_layers + 2, span = 4.2, s = Math.min(w * 0.3, H * 0.2) * view.zoom;
    const yOf = i => -span / 2 + (span * i) / (n - 1);
    const P = (x, y, z) => {
      const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      return [w / 2 + x1 * s, H / 2 - (y * cp - z1 * sp) * s];
    };
    const max = values ? Math.max(1e-9, ...values) : 1;
    const out = [];
    for (let i = 0; i < n; i += 1) {
      const y = yOf(i), b = i - 1, isB = i > 0 && i < n - 1, D = 0.6;
      const pts = [P(-1, y, -D), P(1, y, -D), P(1, y, D), P(-1, y, D)];
      ctx.beginPath(); pts.forEach(([a, c2], k) => (k ? ctx.lineTo(a, c2) : ctx.moveTo(a, c2))); ctx.closePath();
      const v = isB && values ? values[b] : null, sel = isB && b === block;
      ctx.fillStyle = !isB ? 'rgba(148,163,184,0.28)' : v == null ? 'rgba(241,245,249,0.9)' : seq(0.06 + (0.94 * v) / max);
      ctx.globalAlpha = 0.93; ctx.fill(); ctx.globalAlpha = 1;
      ctx.lineWidth = sel ? 2.4 : 0.8; ctx.strokeStyle = sel ? '#e34948' : '#94a3b8'; ctx.stroke();
      const label = !isB ? (i === 0 ? '嵌入层' : '输出头（unembedding）') : `块 ${b}`;
      out.push({ b: isB ? b : null, pts, v, label });
      if (!isB || b % 5 === 0 || sel) {
        const lc = pts.slice().sort((a2, c2) => a2[0] - c2[0])[0];
        ctx.font = `${sel ? 600 : 400} 10.5px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = sel ? '#e34948' : '#475569'; ctx.textAlign = 'right';
        ctx.fillText(!isB ? (i === 0 ? '嵌入' : '输出头') : `块 ${b}`, lc[0] - 6, lc[1] + 3);
      }
    }
    hits.current = out.reverse();
  }, [tower, block, values, view, w]);
  const locate = e => {
    const r = cv.current.getBoundingClientRect();
    const pt = [e.clientX - r.left, e.clientY - r.top];
    return { pt, hit: hits.current.find(h => inside(pt, h.pts)) };
  };
  useEffect(() => {
    const c = cv.current;
    const wheel = e => { e.preventDefault(); setView(v => ({ ...v, zoom: Math.max(0.5, Math.min(2.5, v.zoom * (e.deltaY < 0 ? 1.1 : 0.91))) })); };
    c?.addEventListener('wheel', wheel, { passive: false });
    return () => c?.removeEventListener('wheel', wheel);
  }, []);
  return (
    <div className="v8-tower-canvas">
      <canvas ref={cv} style={{ width: w, height: H }} role="img" aria-label={`${tower.n_layers} 个块的层塔；数值见悬停与右侧`}
        onMouseDown={e => { drag.current = { x: e.clientX, y: e.clientY, view, moved: false }; }}
        onMouseMove={e => {
          if (drag.current) {
            const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
            if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
            setView({ ...drag.current.view, yaw: drag.current.view.yaw + dx * 0.008, pitch: Math.max(-1.2, Math.min(1.2, drag.current.view.pitch + dy * 0.008)) });
            return;
          }
          const { pt, hit } = locate(e);
          setHover(hit ? { ...hit, x: pt[0], y: pt[1] } : null);
        }}
        onMouseUp={e => { const d = drag.current; drag.current = null; if (d && !d.moved) { const { hit } = locate(e); if (hit?.b != null) onPick(hit.b); } }}
        onMouseLeave={() => { drag.current = null; setHover(null); }} />
      {hover && <div className="v8-tip" style={{ left: Math.min(hover.x + 12, w - 170), top: hover.y + 10 }}>{hover.label}{hover.v != null && <small>叠加量 {Number(hover.v).toLocaleString()}</small>}</div>}
      <button type="button" className="fw-tbtn ghost v8-reset" onClick={() => setView({ yaw: -30 * DEG, pitch: 16 * DEG, zoom: 1 })}>重置视角</button>
    </div>
  );
}

function Blk({ id, name, desc, tag, cls, mod, set }) {
  return (
    <button type="button" className={`v8-blk ${cls} ${mod === id ? 'on' : ''}`} onClick={() => set({ mod: id })}>
      <b>{name}</b><span>{desc}</span><em>{tag}</em>
    </button>
  );
}

function BlockStructure({ block, cfg, skeleton, mod, set }) {
  const h = cfg?.hidden_size ?? skeleton?.d_model, ff = cfg?.intermediate_size ?? skeleton?.d_ff;
  const heads = cfg?.num_attention_heads, kv = cfg?.num_key_value_heads, hd = cfg?.head_dim;
  return (
    <section className="v8-card">
      <h3>块 {block} 的内部结构 <small>输入 hs[{block}] → 输出 hs[{block + 1}]；维度来自 {cfg ? 'config.json' : '结果摘要（未找到 config.json，部分字段留空）'}</small></h3>
      <div className="v8-block">
        <div className="v8-blk res"><b>输入残差</b><span>{h ?? '—'} 维（hs[{block}]）</span></div><div className="v8-arrow">↓</div>
        <Blk id="norm" name="RMSNorm" desc="逐位置归一化 × γ" tag="无单独分析" cls="plan" mod={mod} set={set} /><div className="v8-arrow">↓</div>
        <Blk id="attn" name="自注意力" desc={heads ? `${heads} 个查询头 / ${kv} 组 KV（${kv && kv < heads ? 'GQA' : 'MHA'}）· 每头 ${hd} 维 · RoPE θ=${cfg?.rope_theta?.toLocaleString?.() ?? '—'}` : '头数未记录'} tag="规划中：逐头分析" cls="plan" mod={mod} set={set} />
        <div className="v8-arrow">↓ ＋残差</div>
        <Blk id="norm" name="RMSNorm" desc="逐位置归一化 × γ" tag="无单独分析" cls="plan" mod={mod} set={set} /><div className="v8-arrow">↓</div>
        <Blk id="mlp" name="MLP" desc={`gate / up ${h ?? '—'}→${ff ?? '—'} · SwiGLU A = silu(G)⊙V · down ${ff ?? '—'}→${h ?? '—'}`} tag={`${ff?.toLocaleString?.() ?? '—'} 个中间单元 · 已普查`} cls="has" mod={mod} set={set} />
        <div className="v8-arrow">↓ ＋残差</div>
        <div className="v8-blk res"><b>输出残差</b><span>{h ?? '—'} 维（hs[{block + 1}]）</span></div>
      </div>
    </section>
  );
}

function UnitGrid({ layer, mode, selRun, ro, sorted, unit, onPick }) {
  const cv = useRef(null);
  const [tip, setTip] = useState(null);
  const n = layer.n_units, cols = 128, cell = 5, rows = Math.ceil(n / cols);
  const [colors, score, legend] = useMemo(() => unitColors(layer, mode, selRun, ro), [layer, mode, selRun, ro]);
  const order = useMemo(() => {
    const idx = Array.from({ length: n }, (_, i) => i);
    return sorted ? idx.sort((a, b) => score[b] - score[a] || a - b) : idx;
  }, [n, score, sorted]);
  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, c.width, c.height);
    order.forEach((u, pos) => { ctx.fillStyle = colors[u]; ctx.fillRect((pos % cols) * cell, Math.floor(pos / cols) * cell, cell - 1, cell - 1); });
    if (unit != null) {
      const pos = order.indexOf(unit);
      if (pos >= 0) { ctx.strokeStyle = '#e34948'; ctx.lineWidth = 2; ctx.strokeRect((pos % cols) * cell - 2, Math.floor(pos / cols) * cell - 2, cell + 3, cell + 3); }
    }
  }, [order, colors, unit, cols]);
  const at = e => {
    const r = cv.current.getBoundingClientRect();
    const pos = Math.floor((e.clientY - r.top) / cell) * cols + Math.floor((e.clientX - r.left) / cell);
    return { pos, u: pos >= 0 && pos < n ? order[pos] : null, x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const col = layer.columns;
  return (
    <div>
      <div className="v8-gridwrap">
        <canvas ref={cv} width={cols * cell} height={rows * cell} style={{ width: cols * cell, height: rows * cell }}
          aria-label={`块 ${layer.block} 的 ${n} 个 MLP 中间单元${sorted ? '（已按着色量排序）' : '（按真实索引）'}`}
          onMouseMove={e => { const h = at(e); setTip(h.u != null ? h : null); }} onMouseLeave={() => setTip(null)}
          onClick={e => { const h = at(e); if (h.u != null) onPick(h.u); }} />
        {tip && (
          <div className="v8-tip" style={{ left: Math.min(tip.x + 12, cols * cell - 220), top: tip.y + 12 }}>
            <b>B{layer.block}·U{tip.u}</b>{sorted && <small>排序位置 {tip.pos}</small>}
            <small>{layer.type_names[col.type[tip.u]]} · 强激活率 {col.dens[tip.u]} · 主导 {layer.regime_names[col.regime[tip.u]]}</small>
            <small>读出首位 {JSON.stringify(col.ro[tip.u])}{col.hit[tip.u] ? ` · 命中 ${JSON.stringify(col.hit[tip.u])}（集中度 ${col.pconc[tip.u]}）` : ''}</small>
            {layer.explained?.[tip.u] && <small>解释：{layer.explained[tip.u]}</small>}
          </div>
        )}
      </div>
      <div className="v8-lg">{legend.map(l => <span key={l.label}><i style={{ background: l.color }} />{l.label}</span>)}</div>
    </div>
  );
}

function LanguageLinks({ layer, unit, set }) {
  const hits = (layer.selectivity || []).flatMap(s => Object.entries(s.top || {}).flatMap(([r, tops]) =>
    tops.filter(t => t[0] === unit).map(t => ({ s, r, d: t[1], pos: t[2] }))));
  return (
    <section className="v8-card v8-langlinks">
      <h4>对应的语言特征</h4>
      {hits.length ? hits.map(({ s, r, d, pos }) => (
        <div key={s.file + r} className="v8-link-row">
          {s.item ? <button type="button" className="v6-linkbtn" onClick={() => set({ page: 'lang', item: s.item, piece: '' })}>{s.item} →</button> : <span>{s.tm_id}</span>}
          <small>{s.tm_id} v{s.version} · {s.factor} · {READOUT_LABEL[r] || r}读出 · 留出 d = {d} · 偏向 {pos}（该层选择性前列）</small>
        </div>
      )) : <p className="v8-note">这个单元不在任何模板选择性运行的前列名单里。完整的模板选择性记录见下方单元卡「模板选择性」。</p>}
    </section>
  );
}

export default function NeuronPage({ model, skeleton, skeletons, st, set }) {
  const census = skeleton?.census_run ? skeleton.census_key : null;
  const tw = useApi(census ? `/api/neurons/tower?model_key=${encodeURIComponent(census)}` : null);
  const tower = tw.data && !tw.data.pending ? tw.data : null;
  const block = st.block != null && tower && st.block < tower.n_layers ? st.block : 18;
  const ly = useApi(tower ? `/api/neurons/layer?model_key=${encodeURIComponent(census)}&block=${block}&run=${encodeURIComponent(tower.run)}` : null);
  const layer = ly.data && !ly.data.pending ? ly.data : null;
  const values = useMemo(() => (tower ? towerValues(tower, st.ov, st.sel, st.ro) : null), [tower, st.ov, st.sel, st.ro]);
  if (!census) {
    const withCensus = Object.values(skeletons || {}).filter(s => s.census_run);
    return (
      <Empty action={withCensus.map(s => <button key={s.model} type="button" className="fw-tbtn" onClick={() => set({ model: s.model })}>切换到 {shortModel(s.model)}</button>)}>
        {shortModel(model)} 还没有自然语料普查，神经元主线不可用（普查是逐单元统计的来源）。
      </Empty>
    );
  }
  if (tw.error) return <Empty>层塔数据未接入：{tw.error}。中心 1.0.8 及以前没有逐层神经元摘要（/api/neurons/tower、/api/neurons/layer），中心升级后在左下角「更新数据」即可使用。</Empty>;
  if (!tower) return <Empty>{tw.data?.note || '读取层塔…'}</Empty>;
  const selRuns = tower.selectivity || [];
  const selRun = selRuns.find(s => s.file === st.sel) || selRuns[0];
  const ros = Object.keys(selRun?.by_readout || {});
  const ro = ros.includes(st.ro) ? st.ro : ros[0];
  const cfg = skeleton?.config;
  return (
    <div className="v8-neuron">
      <aside className="v8-tower">
        <div className="v8-tower-hd">
          <b>全部层 · {shortModel(model)}</b>
          <span className="v8-seg">{OVERLAYS.map(([k, t]) => <button key={k} type="button" className={st.ov === k ? 'on' : ''} onClick={() => set({ ov: k })}>{t}</button>)}</span>
        </div>
        {st.ov === 'sel' && selRun && (
          <div className="v8-tools">
            <select value={selRun.file} onChange={e => set({ sel: e.target.value })} aria-label="选择性运行">
              {selRuns.map(s => <option key={s.file} value={s.file}>{s.item || s.tm_id} · {s.factor}（{s.hi} 对 {s.lo}）</option>)}
            </select>
            <span className="v8-seg">{ros.map(x => <button key={x} type="button" className={x === ro ? 'on' : ''} onClick={() => set({ ro: x })}>{READOUT_LABEL[x] || x}</button>)}</span>
          </div>
        )}
        <Tower tower={tower} block={block} onPick={b => set({ block: b, unit: null })} values={values} />
        <p className="v8-note">每块板 = 一个 Transformer 块；颜色 = 所选叠加量（浅 → 深，按各块最大值归一化）。拖动旋转、滚轮缩放、点击选中一层。层板只是排列，不代表模型几何。
          {tower.summaries_ready < tower.n_layers && ` 逐层摘要已生成 ${tower.summaries_ready}/${tower.n_layers} 块。`}</p>
      </aside>
      <main className="v8-layerp">
        <BlockStructure block={block} cfg={cfg} skeleton={skeleton} mod={st.mod} set={set} />
        {st.mod === 'attn' && <Empty>自注意力目前没有逐头分析数据（研究计划：注意力头是否作为第二类节点）。接入数据前只画结构，不画示意数值。</Empty>}
        {st.mod === 'norm' && <Empty>归一化层只有 γ 参数，没有单元级分析。点 MLP 查看神经元阵列。</Empty>}
        {st.mod === 'mlp' && (
          <section className="v8-card">
            <h3>MLP 中间单元阵列 · 块 {block} <small>{tower.d_ff.toLocaleString()} 个单元，{sorted(st) ? '已按当前着色量排序（不再是真实索引顺序）' : '按真实索引排列：左上角为 0 号，每行 128 个'}</small></h3>
            <div className="v8-tools">
              <span className="v8-seg">{COLOR_MODES.map(([k, t]) => <button key={k} type="button" className={st.color === k ? 'on' : ''} onClick={() => set({ color: k })}>{t}</button>)}</span>
              <label className="v8-toggle"><input type="checkbox" checked={sorted(st)} onChange={e => set({ sort: e.target.checked ? '1' : '' })} />按着色量排序</label>
            </div>
            {ly.error && <Empty>该层摘要不可用：{ly.error}<ExtraPackPrompt pack={ly.missingPack} /></Empty>}
            {!ly.error && !layer && <Empty>{ly.data?.note || '读取该层…'}</Empty>}
            {layer && <UnitGrid layer={layer} mode={st.color} selRun={selRun?.file} ro={ro} sorted={sorted(st)} unit={st.unit} onPick={u => set({ unit: u })} />}
            {layer && <p className="v8-note">分型规则：{layer.rules?.source}。分型是描述性分类，不是证据等级；选择性只标出各运行的前列单元。</p>}
          </section>
        )}
        {st.mod === 'mlp' && st.unit != null && layer && (
          <div className="v8-unit">
            <LanguageLinks layer={layer} unit={st.unit} set={set} />
            <NeuronUnitCard modelKey={census} block={block} unit={st.unit} run={tower.run} />
          </div>
        )}
        {st.mod === 'mlp' && st.unit == null && layer && <Empty>在阵列里点一个神经元，查看它的单元卡：激活分布、正负极上下文、直接读出、对应的语言特征与检验记录。</Empty>}
      </main>
    </div>
  );
}

function sorted(st) {
  return st.sort === '1';
}
