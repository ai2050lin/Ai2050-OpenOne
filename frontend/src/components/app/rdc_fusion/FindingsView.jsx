/* 规律分析（analysis.v1）：规律陈述 + 层轨迹（留出探针准确率、校正迹 η²）+ 句式不变性矩阵
   + 置换零分布 + 判据。所有数字来自 /api/analysis/{tm}；判据固定写在服务端代码里并随结果返回。 */
import { useState } from 'react';
import { fmtNum } from './api.js';
import { Empty, StatusPill, divColor, inkOn } from './ui.jsx';
import { CATEGORICAL } from './Space3D.jsx';

const KIND = { supported: ['pass', '支持'], negative: ['fail', '未支持'], warning: ['warn', '注意'], info: ['na', '描述'] };

/* 折线图：x = 层（有序），每个序列一条 2px 线 + 8px 标记，末端直接标注；悬停显示数值 */
function LineChart({ layers, series, yMax = 1, yLabel, refLines = [], height = 200 }) {
  const [hover, setHover] = useState(null);
  const W = 460, H = height, L = 44, R = 104, T = 24, B = 28;
  const x = i => L + (layers.length === 1 ? (W - L - R) / 2 : (i / (layers.length - 1)) * (W - L - R));
  const y = v => T + (1 - Math.max(0, Math.min(yMax, v)) / yMax) * (H - T - B);
  const ticks = [0, yMax / 2, yMax];
  // 末端直接标注：按 y 排序后至少相隔 12px，避免数值相同时标签重叠
  const ends = series.map((s, si) => {
    let li = -1;
    s.values.forEach((v, i) => { if (v != null) li = i; });
    return li < 0 ? null : { si, y: y(s.values[li]), x: x(li) };
  }).filter(Boolean).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 12) ends[i].y = ends[i - 1].y + 12;
  const labelY = Object.fromEntries(ends.map(e => [e.si, e.y]));
  const refs = refLines.map(r => ({ ...r, ly: y(r.value) })).sort((a, b) => a.ly - b.ly);
  for (let i = 1; i < refs.length; i++) if (refs[i].ly - refs[i - 1].ly < 12) refs[i].ly = refs[i - 1].ly + 12;
  return (
    <div className="v6-tip-host">
      <svg viewBox={`0 0 ${W} ${H}`} className="v6-line" role="img" aria-label={yLabel}>
        {ticks.map(t => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="#e2e8f0" />
            <text x={L - 6} y={y(t) + 3} textAnchor="end" className="tick">{t.toFixed(1)}</text>
          </g>
        ))}
        {layers.map((l, i) => <text key={l} x={x(i)} y={H - 8} textAnchor="middle" className="tick">{l}</text>)}
        <text x={L - 30} y={12} className="tick">{yLabel}</text>
        {refs.map(r => (
          <g key={r.label}>
            <line x1={L} x2={W - R} y1={y(r.value)} y2={y(r.value)} stroke="#94a3b8" strokeDasharray="4 3" />
            <text x={W - R + 4} y={r.ly + 3} className="tick">{r.label}</text>
          </g>
        ))}
        {series.map((s, si) => {
          const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v), v, layers[i]])).filter(Boolean);
          if (!pts.length) return null;
          const last = pts[pts.length - 1];
          return (
            <g key={s.name}>
              <polyline points={pts.map(p => `${p[0]},${p[1]}`).join(' ')} fill="none" stroke={s.color} strokeWidth="2" />
              {pts.map((p, i) => (
                <g key={i} onMouseEnter={() => setHover({ x: p[0], y: p[1], text: `${s.name} · ${p[3]}：${fmtNum(p[2])}` })} onMouseLeave={() => setHover(null)}>
                  <circle cx={p[0]} cy={p[1]} r="10" fill="transparent" />
                  <circle cx={p[0]} cy={p[1]} r="4" fill={s.color} stroke="#fff" strokeWidth="2" />
                </g>
              ))}
              <text x={last[0] + 8} y={(labelY[si] ?? last[1]) + 4} className="lbl">{s.name}</text>
            </g>
          );
        })}
      </svg>
      {hover && <div className="v6-tip" style={{ left: `${(hover.x / W) * 100}%`, top: hover.y + 8 }}>{hover.text}</div>}
    </div>
  );
}

function NullHist({ hist, observed, range, label }) {
  if (!hist) return null;
  const max = Math.max(...hist, 1);
  const W = 300, H = 90, L = 6, R = 6, T = 8, B = 18;
  const bw = (W - L - R) / hist.length;
  const xo = observed == null ? null : L + ((observed - range[0]) / (range[1] - range[0])) * (W - L - R);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="v6-hist" role="img" aria-label={label}>
      {hist.map((c, i) => <rect key={i} x={L + i * bw + 1} width={Math.max(bw - 2, 1)} y={T + (1 - c / max) * (H - T - B)} height={(c / max) * (H - T - B)} rx="1.5" fill="#9ec5f4" />)}
      {xo != null && <line x1={xo} x2={xo} y1={T - 4} y2={H - B} stroke="#0f172a" strokeWidth="2" />}
      <text x={L} y={H - 4} className="tick">{range[0]}</text>
      <text x={W - R} y={H - 4} textAnchor="end" className="tick">{range[1]}</text>
      {xo != null && <text x={Math.min(xo + 4, W - 60)} y={T + 6} className="lbl">观测 {fmtNum(observed)}</text>}
    </svg>
  );
}

export default function FindingsView({ analysis }) {
  const layers = analysis.layers.filter(l => !l.insufficient);
  if (!layers.length) return <Empty>discovery 样本不足，无法分析。</Empty>;
  const tags = layers.map(l => l.tag);
  const factors = analysis.factors;
  const accSeries = factors.map((f, i) => ({ name: f, color: CATEGORICAL[i], values: layers.map(l => l.factors[f]?.splits?.lockbox?.probe_acc ?? null) }));
  const frameSeries = factors.map((f, i) => ({ name: f + '·句式', color: CATEGORICAL[i], values: layers.map(l => l.factors[f]?.splits?.lockbox_frame?.probe_acc ?? null) })).filter(s => s.values.some(v => v != null));
  const etaSeries = [...factors.map((f, i) => ({ name: f, color: CATEGORICAL[i], values: layers.map(l => l.trace_eta2?.[f] ?? null) })),
    { name: '词身份', color: CATEGORICAL[factors.length], values: layers.map(l => l.trace_eta2?.filler ?? null) }];
  const main = factors[0];
  const best = layers.reduce((a, l) => ((l.factors[main]?.splits?.lockbox?.probe_acc ?? -1) > (a.factors[main]?.splits?.lockbox?.probe_acc ?? -1) ? l : a), layers[0]);
  const perm = best.factors[main]?.perm;
  const inv = (best.invariance || [])[0];
  const etaMax = Math.max(0.2, ...etaSeries.flatMap(s => s.values.filter(v => v != null)));
  return (
    <div className="v6-findings">
      <section className="v6-find-list">
        {analysis.findings.map(f => (
          <article key={f.id} className={'v6-find ' + f.kind}>
            <header><StatusPill status={KIND[f.kind]?.[0] || 'na'} label={KIND[f.kind]?.[1]} /><b>{f.title}</b>{f.level && <em>{f.level}</em>}</header>
            <p>{f.text}</p>
            {f.scope && <small>适用范围：{f.scope}</small>}
          </article>
        ))}
      </section>
      <section className="v6-find-charts">
        <div>
          <h5 className="v6-h5">未见词表（lockbox）探针准确率 · 按层</h5>
          <LineChart layers={tags} series={accSeries} yLabel="准确率"
            refLines={factors.map(f => ({ label: `${f} 机会`, value: best.factors[f]?.chance ?? 0 }))} />
        </div>
        {frameSeries.length > 0 && (
          <div>
            <h5 className="v6-h5">未见句式（lockbox_frame）探针准确率</h5>
            <LineChart layers={tags} series={frameSeries} yLabel="准确率" refLines={[{ label: `${main} 机会`, value: best.factors[main]?.chance ?? 0 }]} />
          </div>
        )}
        <div>
          <h5 className="v6-h5">校正迹 η²（discovery）· 按层</h5>
          <LineChart layers={tags} series={etaSeries} yMax={Math.min(1, Math.ceil(etaMax * 10) / 10)} yLabel="ε²" />
        </div>
        {perm && (
          <div>
            <h5 className="v6-h5">置换零分布（{best.tag} 层 · {main} · {perm.split} · {perm.n_perm} 次）</h5>
            <div className="v6-hists">
              <div><small>探针准确率 p={fmtNum(perm.acc_p, 4)}</small><NullHist hist={perm.acc_null_hist} observed={best.factors[main].splits[perm.split]?.probe_acc} range={[0, 1]} label="探针准确率零分布" /></div>
              <div><small>方向复现 cos p={fmtNum(perm.rep_p, 4)}</small><NullHist hist={perm.rep_null_hist} observed={best.factors[main].splits[perm.split]?.rep_cos_white} range={[-1, 1]} label="方向复现零分布" /></div>
            </div>
            <p className="v6-note">打乱单位 = {best.factors[main].unit}（按{best.factors[main].unit === 'filler' ? '填充词' : best.factors[main].unit === 'frame' ? '句式' : '样本'}整体打乱标签，保留词内/句内结构）。</p>
          </div>
        )}
        {inv && (
          <div>
            <h5 className="v6-h5">{main} 方向的句式不变性（{best.tag} 层，白化）</h5>
            <div className="v6-matrix" style={{ gridTemplateColumns: `minmax(70px,max-content) repeat(${inv.levels.length}, 40px)` }}>
              {inv.matrix.flatMap((row, i) => [
                <em key={`l${i}`} className="v6-matrix-label">{inv.levels[i]}</em>,
                ...row.map((v, j) => <span key={`${i}-${j}`} style={{ width: 40, height: 28, background: divColor(v ?? 0), color: inkOn(v ?? 0, true) }} title={`${inv.levels[i]} × ${inv.levels[j]}：${fmtNum(v)}`}>{v == null ? '' : v.toFixed(2)}</span>),
              ])}
            </div>
            <p className="v6-note">每格 = 在该两个句式内分别计算的 {main} 水平方向的平均 cos；平均 {fmtNum(inv.mean_offdiag)}（白化）/ {fmtNum(inv.mean_offdiag_raw)}（原始）。</p>
          </div>
        )}
        <div>
          <h5 className="v6-h5">判据（固定于代码，随结果返回）</h5>
          <table className="v6-mini-table">
            <tbody>{Object.entries(analysis.criteria).map(([k, v]) => <tr key={k}><th scope="row" className="mono">{k}</th><td className="num">{String(v)}</td></tr>)}</tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
