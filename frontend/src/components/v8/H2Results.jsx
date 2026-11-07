/* 关联机制 · 主线 H2 的真实结果（MA-H2，Phase 324）。数据随构建读入 research/mainline/H2/report.json。
   只是 discovery 上的候选观察：逐层读出是近似，注意力去向不等于计算使用，确认要到 H4 的路径干预。 */
import { useState } from 'react';

import report from '../../../../research/mainline/H2/report.json';
import './client_guide.css';

const DOC = '#2a78d6', REAL = '#eb6834';
const POS_LABEL = { ans: '答案位置', qname: '问题中的目标名', scope: '范围词', tname: '记录中的目标名', tcol: '记录中的颜色', lastcol: '最后一个颜色' };
const FACTOR_LABEL = { scope: '范围', draft: '草稿', form: '记录形式', pos: '目标位置', k: '记录数' };

export function LineChart({ series, label, yLabel, band, height = 210 }) {
  const [hover, setHover] = useState(null);
  const W = 640, H = height, x0 = 44, y0 = 12, w = W - x0 - 64, h = H - y0 - 30;
  const n = series[0].values.length;
  const all = series.flatMap(s => s.values);
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all), pad = (hi - lo) * 0.06 || 1;
  const Y = v => y0 + h * (1 - (v - lo + pad) / (hi - lo + 2 * pad));
  const X = i => x0 + (w * i) / (n - 1);
  const ticks = [lo, 0, hi].filter((v, i, a) => a.indexOf(v) === i);
  /* 末端直接标注：按纵向位置排序后逐个下推，相邻标签至少相隔 13 像素，整体再居中回原位附近 */
  const raw = series.map(s => Y(s.values[n - 1]));
  const order = raw.map((y, i) => [y, i]).sort((a, b) => a[0] - b[0]);
  const placed = [];
  order.forEach(([y], j) => { placed[j] = j ? Math.max(y, placed[j - 1] + 13) : y; });
  const shift = (order.reduce((s, [y]) => s + y, 0) - placed.reduce((s, y) => s + y, 0)) / Math.max(1, placed.length);
  const ends = [];
  order.forEach(([, i], j) => { ends[i] = Math.min(y0 + h, Math.max(y0 + 6, placed[j] + shift)); });
  return <figure className="h2-fig">
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
      {band && <rect x={X(band[0])} y={y0} width={X(band[1]) - X(band[0])} height={h} fill="#eef3fa"/>}
      {ticks.map(v => <g key={v}><line x1={x0} x2={x0 + w} y1={Y(v)} y2={Y(v)} stroke={v === 0 ? '#b9c4d2' : '#eceff4'}/>
        <text x={x0 - 6} y={Y(v) + 3} fontSize="10" fill="#566373" textAnchor="end">{v.toFixed(1)}</text></g>)}
      {[0, 6, 12, 18, 24, 30, n - 1].map(i => <text key={i} x={X(i)} y={H - 10} fontSize="10" fill="#566373" textAnchor="middle">{i}</text>)}
      <text x={x0 + w} y={H - 10} fontSize="10" fill="#566373" textAnchor="end" dx="40">层</text>
      {series.map((s, k) => <g key={s.name}>
        <polyline points={s.values.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')} fill="none" stroke={s.color} strokeWidth="2"/>
        <text x={X(n - 1) + 6} y={ends[k] + 4} fontSize="11" fill="#16202c">{s.name}</text></g>)}
      {hover != null && <line x1={X(hover)} x2={X(hover)} y1={y0} y2={y0 + h} stroke="#16202c" strokeDasharray="3 3"/>}
      {Array.from({ length: n }, (_, i) => <rect key={i} x={X(i) - w / (2 * (n - 1))} y={y0} width={w / (n - 1)} height={h} fill="transparent"
        onMouseEnter={() => setHover(i)}><title>{`第 ${i} 层：${series.map(s => `${s.name} ${s.values[i].toFixed(3)}`).join('，')}`}</title></rect>)}
    </svg>
    <figcaption className="v8-note">{yLabel}{hover != null ? ` · 第 ${hover} 层：${series.map(s => `${s.name} ${s.values[hover].toFixed(3)}`).join('，')}` : ''}</figcaption>
  </figure>;
}

export default function H2Results() {
  const lens = report.lens;
  const att = report.attention;
  const factors = Object.keys(FACTOR_LABEL);
  const positions = Object.keys(POS_LABEL);
  const cellText = (f, p) => {
    const e = report.effects[f][p];
    if (e.auc[0] != null && e.auc[0] >= 0.9) return '词身份混淆';
    return e.first_readable == null ? '—' : `第 ${e.first_readable} 层`;
  };
  return <section className="v8-card cg-section" aria-label="主线 H2 结果">
    <h3>一份资料、一个问题：颜色在哪里分出来 <span className="mp-tag ok">真实数据 · Phase 324</span></h3>
    <p className="v8-note">主线切片 A：资料里给一个物品配一种与常识相反的颜色，再问“按这份资料”或“在现实世界里”是什么颜色。Qwen3-4B bf16，discovery {report.n_analyzed} 条（已排除预注册的 2 格）。
      这里是候选观察，确认要到 H4 的干预。</p>
    <div className="lg3-legend" aria-hidden="true">
      <span><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke={DOC} strokeWidth="2"/></svg>按资料问</span>
      <span><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke={REAL} strokeWidth="2"/></svg>按现实问</span>
      <span><i style={{ display: 'inline-block', width: 14, height: 10, background: '#eef3fa', border: '1px solid #dfe5ec' }}/>分化区</span>
    </div>
    <div className="cg-grid2">
      <div>
        <h4>逐层读出：资料色比常识色高多少</h4>
        <LineChart label="逐层读出的资料色减常识色" yLabel="纵轴：资料色 K 与常识色 P 的首 token logit 差（逐层读出，近似）。"
          series={[{ name: '按资料问', color: DOC, values: lens.mean_K_minus_P.doc }, { name: '按现实问', color: REAL, values: lens.mean_K_minus_P.real }]} band={[24, 33]}/>
        <p className="v8-note">第 24 层前两条线都在 0 附近，看不出偏向；第 24–33 层（浅蓝区）同时分开。最终答对率：按资料 {(lens.final_correct_share.doc * 100).toFixed(1)}%，按现实 {(lens.final_correct_share.real * 100).toFixed(1)}%。</p>
      </div>
      <div>
        <h4>注意力：答案位置有多少看向记录里的颜色</h4>
        <LineChart label="答案位置对目标颜色的注意力" yLabel="纵轴：答案位置对目标记录颜色 token 的注意力，32 个头平均。"
          series={[{ name: '按资料问', color: DOC, values: att.tcol.layer_mean.doc }, { name: '按现实问', color: REAL, values: att.tcol.layer_mean.real }]} band={[21, 29]}/>
        <p className="v8-note">读颜色的注意力集中在第 21–29 层；按现实问时明显更少。</p>
      </div>
    </div>
    <h4>最常读记录颜色的注意力头</h4>
    <div className="v8-weight-scroll"><table className="v6-table"><thead><tr><th>头</th><th>按资料问</th><th>按现实问</th></tr></thead>
      <tbody>{att.top_heads_to_tcol.slice(0, 8).map(r => <tr key={`${r.layer}-${r.head}`}><td>L{r.layer}H{r.head}</td><td>{r.doc.toFixed(3)}</td><td>{r.real.toFixed(3)}</td></tr>)}</tbody></table></div>
    <p className="v8-note">这些头按现实问时几乎不读资料色，形态更像“读取时就按范围门控”。注意力权重高不等于模型用了它，H4 会对这些头做干预。</p>
    <details className="mp-fold"><summary>各因素从第几层开始读得出</summary><div className="mp-fold-body">
      <div className="v8-weight-scroll"><table className="v6-table"><thead><tr><th>因素</th>{positions.map(p => <th key={p}>{POS_LABEL[p]}</th>)}</tr></thead>
        <tbody>{factors.map(f => <tr key={f}><td>{FACTOR_LABEL[f]}</td>{positions.map(p => <td key={p}>{cellText(f, p)}</td>)}</tr>)}</tbody></table></div>
      <p className="v8-note">判据：实体留出的读出 AUC ≥ 0.75，且比第 0 层高 0.10。“—”为未达判据。范围和草稿在记录位置读不出，因为它们出现在记录之后，这是预期的负对照。
        “词身份混淆”：两种范围的答案前缀末 token 不同（“是”与“通常是”），第 0 层就能分开，不能算上下文计算。</p>
    </div></details>
    <details className="mp-fold"><summary>已知问题</summary><div className="mp-fold-body">
      <p className="v8-note">预注册的 R1 / R2 判据取第 8–32 层“任一层资料色领先”，这些层接近噪声，判为“{lens.verdict}”（{(lens.real_K_lead_share * 100).toFixed(1)}%）但没有信息量，已标为不可用。答案前缀的末 token 将在 H3 前统一。</p>
    </div></details>
  </section>;
}
