/* 神经元面板（研究工作台 · 空间）：选中块的内部结构（按 config.json）→ MLP 中间单元阵列（9,728 个，按真实索引排列，可显式排序）→ 单元卡。
   层塔在空间的左侧常驻（Tower.jsx）；这里只画选中的块。注意力在有逐头数据前只画结构；网格位置只是排列，不代表模型几何。 */
import { useEffect, useMemo, useRef, useState } from 'react';

import NeuronUnitCard from '../app/rdc_fusion/NeuronUnitCard.jsx';
import { neuronUnitContext, useAiContext } from '../app/rdc_fusion/aiContext.js';
import { useApi } from '../app/rdc_fusion/api.js';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { READOUT_LABEL, unitColors } from './v8Util.js';

const COLOR_MODES = [['type', '分型'], ['dens', '强激活率'], ['regime', '主导符号区'], ['expl', '解释等级'], ['sel', '选择性']];

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

export default function NeuronPane({ census, tower, skeleton, block, st, set }) {
  const ly = useApi(`/api/neurons/layer?model_key=${encodeURIComponent(census)}&block=${block}&run=${encodeURIComponent(tower.run)}`);
  const layer = ly.data && !ly.data.pending ? ly.data : null;
  const selRuns = tower.selectivity || [];
  const selRun = selRuns.find(s => s.file === st.sel) || selRuns[0];
  const ros = Object.keys(selRun?.by_readout || {});
  const ro = ros.includes(st.ro) ? st.ro : ros[0];
  const cfg = skeleton?.config;
  const unit = st.mod === 'mlp' ? st.unit : null;
  useAiContext(useMemo(() => neuronUnitContext(layer, unit, {                        // 「问 AI」侧栏附带选中单元的数据
    census, block, links: (layer?.selectivity || []).flatMap(s => Object.entries(s.top || {}).flatMap(([r, tops]) =>
      tops.filter(t => t[0] === unit).map(t => ({ item: s.item, tm_id: s.tm_id, version: s.version, factor: s.factor, readout: r, heldout_d: t[1], direction: t[2] })))),
  }), [layer, unit, census, block]));
  return (
    <div className="v8-layerp">
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
    </div>
  );
}

function sorted(st) {
  return st.sort === '1';
}
