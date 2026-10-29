/* 神经元面板（研究工作台 · 空间）· 三层漏斗版：
   ① 本块速览（数据派生的摘要句 + 分型占比 + 关键数字）→ ② 精选单元条（解释等级 / 选择性 / 强激活，
      点击直达单元卡）→ ③ 全量阵列（原 canvas，功能不变）→ ④ 单元卡。
   层塔在空间左侧常驻（Tower.jsx）。注意力/归一化没有单元级分析：结构图里直接注明，不再做成点进去空白的死胡同。 */
import { useEffect, useMemo, useRef, useState } from 'react';

import NeuronUnitCard from '../app/rdc_fusion/NeuronUnitCard.jsx';
import { neuronUnitContext, useAiContext } from '../app/rdc_fusion/aiContext.js';
import { useApi } from '../app/rdc_fusion/api.js';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { GRADE_COLOR, READOUT_LABEL, TYPE_COLOR, unitColors } from './v8Util.js';

const COLOR_MODES = [['type', '分型'], ['dens', '强激活率'], ['regime', '主导符号区'], ['expl', '解释等级'], ['sel', '选择性']];

/* ── ① 本块速览 ─────────────────────────────────────────── */
function BlockDigest({ layer }) {
  const n = layer.n_units;
  const col = layer.columns;
  const hist = layer.type_names.map((_, k) => ({ k, name: layer.type_names[k], n: 0 }));
  for (const t of col.type) hist[t].n += 1;
  const dominant = hist.reduce((a, b) => (b.n > a.n ? b : a));
  const ex = layer.explained || {};
  const grades = { 文本可复现: 0, 只在语料中可识别: 0, 未解释: 0 };
  for (const v of Object.values(ex)) if (v in grades) grades[v] += 1;
  const sel = layer.selectivity || [];
  const densSorted = [...col.dens].sort((a, b) => a - b);
  const densMedian = densSorted.length ? densSorted[Math.floor(densSorted.length / 2)] : 0;
  const pct = k => `${((k / n) * 100).toFixed(1)}%`;
  return (
    <section className="v8-card v8-digest">
      <h3>块 {layer.block} 普查发现了什么 <small>数字全部来自本层语料普查，句式是固定模板、不含推测</small></h3>
      <p className="v8-digest-say">
        这一块有 <b>{n.toLocaleString()}</b> 个 MLP 中间单元。
        {Object.values(grades).some(v => v > 0)
          ? <>其中 <b>{Object.entries(grades).filter(([, v]) => v > 0).map(([g, v]) => `${g} ${v} 个`).join('、')}</b>（做过解释检验的共 {Object.values(grades).reduce((a, b) => a + b, 0)} 个）；</>
          : '尚无单元做过解释检验；'}
        {sel.length
          ? <>模板选择性方面，<b>{sel.length} 个运行</b>（如 {sel.map(s => s.item || s.tm_id).slice(0, 2).join('、')}）在本层有前列单元；</>
          : '没有模板选择性运行覆盖本层；'}
        最常见的类型是 <b style={{ color: TYPE_COLOR[dominant.k] }}>{dominant.name}</b>（{pct(dominant.n)}）。
      </p>
      <div className="v8-digest-bar" title={hist.map(h => `${h.name} ${h.n}（${pct(h.n)}）`).join(' · ')}>
        {hist.filter(h => h.n > 0).map(h => (
          <i key={h.k} style={{ background: TYPE_COLOR[h.k], width: pct(h.n) }} title={`${h.name}：${h.n} 个（${pct(h.n)}）`} />
        ))}
      </div>
      <div className="v8-digest-stats">
        <span><b style={{ color: GRADE_COLOR['文本可复现'] }}>{grades['文本可复现']}</b>文本可复现</span>
        <span><b style={{ color: GRADE_COLOR['只在语料中可识别'] }}>{grades['只在语料中可识别']}</b>语料可识别</span>
        <span><b>{sel.reduce((a, s) => a + Object.values(s.n_selected || {}).reduce((x, y) => x + y, 0), 0).toLocaleString()}</b>选择性前列单元（按运行计）</span>
        <span><b>{densMedian.toFixed(4)}</b>强激活率中位数</span>
      </div>
      <div className="v8-lg">{hist.map(h => <span key={h.k}><i style={{ background: TYPE_COLOR[h.k] }} />{h.name}</span>)}</div>
    </section>
  );
}

/* ── ② 精选单元条 ───────────────────────────────────────── */
function PickedUnits({ layer, selRun, unit, onPick }) {
  const col = layer.columns;
  const picks = (() => {
    const seen = new Set(); const out = [];
    const push = (u, why) => {
      if (u == null || !Number.isInteger(u) || u < 0 || u >= layer.n_units || seen.has(u) || out.length >= 8) return;
      seen.add(u); out.push({ u, why });
    };
    const gradeRank = { 文本可复现: 0, 只在语料中可识别: 1, 未解释: 2 };
    Object.entries(layer.explained || {})
      .sort((a, b) => gradeRank[a[1]] - gradeRank[b[1]] || a[0] - b[0])
      .forEach(([u, g]) => push(+u, g));                                     // ① 解释等级最高的
    for (const [r, tops] of Object.entries(selRun?.top || {})) {             // ② 选择性 |d| 前列（两方向都要）
      const label = READOUT_LABEL[r] || r;
      [...tops].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 3)
        .forEach(([u, d, pos]) => push(u, `${label}读出偏向 ${pos}（d=${d}）`));
    }
    [...col.dens.keys()].sort((a, b) => col.dens[b] - col.dens[a])           // ③ 强激活率补足
      .forEach(u => push(u, `强激活率 ${(col.dens[u] * 100).toFixed(1)}%`));
    return out;
  })();
  return (
    <section className="v8-card">
      <h3>从这 {picks.length} 个开始看 <small>已解释的在前，其次选择性最强，再按强激活率补足——点一个直达它的单元卡</small></h3>
      <div className="v8-picked">
        {picks.map(({ u, why }) => (
          <button key={u} type="button" className={'v8-pick' + (unit === u ? ' on' : '')} onClick={() => onPick(u)}>
            <b><i style={{ background: TYPE_COLOR[col.type[u]] }} />U{u}</b>
            <span>{why}</span>
            <em>{layer.type_names[col.type[u]]}</em>
          </button>
        ))}
      </div>
    </section>
  );
}

/* ── 结构图：MLP 为主角；注意力 / 归一化注明「无单元级分析」，不再做成死胡同按钮 ── */
function BlockStructure({ block, cfg, skeleton }) {
  const h = cfg?.hidden_size ?? skeleton?.d_model, ff = cfg?.intermediate_size ?? skeleton?.d_ff;
  const heads = cfg?.num_attention_heads, kv = cfg?.num_key_value_heads, hd = cfg?.head_dim;
  return (
    <section className="v8-card">
      <h3>块 {block} 的内部结构 <small>输入 hs[{block}] → 输出 hs[{block + 1}]；维度来自 {cfg ? 'config.json' : '结果摘要（未找到 config.json，部分字段留空）'}</small></h3>
      <div className="v8-block">
        <div className="v8-blk res"><b>输入残差</b><span>{h ?? '—'} 维（hs[{block}]）</span></div><div className="v8-arrow">↓</div>
        <div className="v8-blk plan"><b>RMSNorm</b><span>逐位置归一化 × γ</span><em>只有 γ 参数，无单元级分析</em></div><div className="v8-arrow">↓</div>
        <div className="v8-blk plan"><b>自注意力</b><span>{heads ? `${heads} 个查询头 / ${kv} 组 KV（${kv && kv < heads ? 'GQA' : 'MHA'}）· 每头 ${hd} 维` : '头数未记录'}</span><em>逐头分析在研究计划中（想看逐头实时计算，切到「实时分析」输入文本）</em></div><div className="v8-arrow">↓ ＋残差</div>
        <div className="v8-blk plan"><b>RMSNorm</b><span>逐位置归一化 × γ</span><em>同上</em></div><div className="v8-arrow">↓</div>
        <div className="v8-blk has on"><b>MLP ↓ 本页主体</b><span>{`gate / up ${h ?? '—'}→${ff ?? '—'} · SwiGLU A = silu(G)⊙V · down ${ff ?? '—'}→${h ?? '—'}`}</span><em>{ff?.toLocaleString?.() ?? '—'} 个中间单元 · 已普查</em></div>
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
      <BlockStructure block={block} cfg={cfg} skeleton={skeleton} />
      {st.mod === 'attn' && <Empty>自注意力目前没有逐头分析数据（研究计划：注意力头是否作为第二类节点）。逐头实时计算可切到「实时分析」输入文本观察。</Empty>}
      {st.mod === 'norm' && <Empty>归一化层只有 γ 参数，没有单元级分析。单元分析看 MLP（默认视图）。</Empty>}
      {st.mod === 'mlp' && (ly.error ? <Empty>该层摘要不可用：{ly.error}<ExtraPackPrompt pack={ly.missingPack} /></Empty>
        : !layer ? <Empty>{ly.data?.note || '读取该层…'}</Empty>
          : <>
              <BlockDigest layer={layer} />
              <PickedUnits layer={layer} selRun={selRun} unit={unit} onPick={u => set({ unit: u })} />
              {unit != null && (
                <div className="v8-unit">
                  <LanguageLinks layer={layer} unit={unit} set={set} />
                  <NeuronUnitCard modelKey={census} block={block} unit={unit} run={tower.run} />
                </div>
              )}
              <section className="v8-card">
                <h3>全部单元阵列 · 块 {block} <small>{tower.d_ff.toLocaleString()} 个单元，{sorted(st) ? '已按当前着色量排序（不再是真实索引顺序）' : '按真实索引排列：左上角为 0 号，每行 128 个；悬停看单个单元，点击选中'}</small></h3>
                <div className="v8-tools">
                  <span className="v8-seg">{COLOR_MODES.map(([k, t]) => <button key={k} type="button" title={COLOR_MODE_HINT[k] || ''} className={st.color === k ? 'on' : ''} onClick={() => set({ color: k })}>{t}</button>)}</span>
                  <label className="v8-toggle"><input type="checkbox" checked={sorted(st)} onChange={e => set({ sort: e.target.checked ? '1' : '' })} />按着色量排序</label>
                </div>
                <UnitGrid layer={layer} mode={st.color} selRun={selRun?.file} ro={ro} sorted={sorted(st)} unit={unit} onPick={u => set({ unit: u })} />
                <p className="v8-note">分型规则：{layer.rules?.source}。分型是描述性分类，不是证据等级；选择性只标出各运行的前列单元。</p>
              </section>
            </>)}
    </div>
  );
}

const COLOR_MODE_HINT = {
  type: '按普查分型着色：这个单元像什么（单 token 检测器 / 上下文检测器 / 稀疏强激活…）',
  dens: '颜色深 = 这个单元被强激活的频率高（对数色阶）',
  regime: '按激活主导符号区着色：正正 / 负负 / 正负 / 负正',
  expl: '只给做过解释检验的单元着色：文本可复现 > 语料可识别 > 未解释',
  sel: '只给当前模板选择性运行的前列单元着色：偏向哪一侧',
};

function sorted(st) {
  return st.sort === '1';
}
