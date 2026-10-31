/* 机制报告的二维图：层 × 头热力图、逐层赛跑曲线、逐层成分条、判据板。
   色彩：正负量用蓝 ↔ 红、灰色中点（ui.jsx divColor），每张图按自身最大绝对值对称归一并在图例写明；
   类别（线索）用固定顺序的类别色，同时给出文字图例。 */
import { useState } from 'react';

import { divColor } from '../rdc_fusion/ui.jsx';
import { CATEGORICAL_COLORS } from './mechColors.js';
import { checkList, fmt, maxAbs, voidList } from './mechUtil.js';

/* 层 × 头热力图：行 = 层（自上而下 0 → L−1），列 = 头。highlight：需要描边的 'L.H' 集合 */
export function HeadHeatmap({ matrix, title, legend, highlight, onPick, picked }) {
  const [hover, setHover] = useState(null);
  if (!Array.isArray(matrix) || !matrix.length) return null;
  const m = maxAbs(matrix) || 1;
  const nh = Math.max(...matrix.map(r => r?.length || 0));
  const hv = hover && matrix[hover[0]]?.[hover[1]];
  return (
    <figure className="wb9-heat">
      <figcaption><b>{title}</b><small>{legend}；色标 ±{fmt(m, 3)}（本图最大绝对值）</small></figcaption>
      <div className="wb9-heat-grid" style={{ gridTemplateColumns: `28px repeat(${nh}, minmax(0, 1fr))` }}
           onMouseLeave={() => setHover(null)} role="img" aria-label={`${title}：${matrix.length} 层 × ${nh} 头`}>
        {matrix.map((row, l) => [
          <span key={`l${l}`} className="wb9-heat-row">{l % 5 === 0 ? l : ''}</span>,
          ...Array.from({ length: nh }, (_, h) => {
            const v = row?.[h];
            const name = `${l}.${h}`;
            const cls = (highlight?.has(name) ? ' hl' : '') + (picked === name ? ' on' : '');
            return (
              <i key={name} className={'wb9-heat-cell' + cls} style={{ background: Number.isFinite(v) ? divColor(v / m) : '#e2e8f0' }}
                 onMouseEnter={() => setHover([l, h])} onClick={onPick ? () => onPick(name) : undefined} />
            );
          }),
        ])}
      </div>
      <div className="wb9-heat-foot">
        <span className="wb9-scale"><i style={{ background: divColor(-1) }} />−{fmt(m, 3)}<i style={{ background: divColor(0) }} />0<i style={{ background: divColor(1) }} />+{fmt(m, 3)}</span>
        <span>{hover ? `头 ${hover[0]}.${hover[1]}：${fmt(hv, 4)}` : '悬停查看数值；行 = 层，列 = 头'}</span>
      </div>
    </figure>
  );
}

/* 逐层赛跑曲线：每条线 = 一种线索在全部过门句子上的平均（上下文答案 − 记忆答案）logit lens 读数；
   横轴 hs 序号（0 = 嵌入层），纵轴 > 0 表示上下文答案领先。 */
export function RaceChart({ series, height = 260 }) {
  const lines = (series || []).filter(s => Array.isArray(s.values) && s.values.length);
  if (!lines.length) return null;
  const W = 640, H = height, L = 44, R = 12, T = 12, B = 30;
  const n = Math.max(...lines.map(s => s.values.length));
  const all = lines.flatMap(s => s.values).filter(Number.isFinite);
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const x = i => L + ((W - L - R) * i) / Math.max(1, n - 1);
  const y = v => T + ((H - T - B) * (hi - v)) / ((hi - lo) || 1);
  const ticks = [lo, 0, hi].filter((v, i, a) => a.indexOf(v) === i);
  return (
    <figure className="wb9-race">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="逐层赛跑曲线">
        {ticks.map(v => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke={v === 0 ? '#64748b' : '#e2e8f0'} strokeDasharray={v === 0 ? '' : '3 3'} />
            <text x={L - 6} y={y(v) + 4} textAnchor="end" fontSize="11" fill="#64748b">{fmt(v, 1)}</text>
          </g>
        ))}
        {Array.from({ length: n }, (_, i) => i).filter(i => i % 6 === 0 || i === n - 1).map(i => (
          <text key={i} x={x(i)} y={H - 10} textAnchor="middle" fontSize="11" fill="#64748b">{i === 0 ? '嵌入' : i}</text>
        ))}
        {lines.map((s, k) => (
          <polyline key={s.id} fill="none" stroke={CATEGORICAL_COLORS[k % CATEGORICAL_COLORS.length]} strokeWidth={s.bold ? 2.6 : 1.6}
                    strokeDasharray={s.dashed ? '5 3' : ''}
                    points={s.values.map((v, i) => (Number.isFinite(v) ? `${x(i)},${y(v)}` : '')).filter(Boolean).join(' ')} />
        ))}
        <text x={W - R} y={T + 10} textAnchor="end" fontSize="11" fill="#475569">↑ 上下文答案领先</text>
        <text x={W - R} y={H - B - 4} textAnchor="end" fontSize="11" fill="#475569">↓ 记忆答案领先</text>
      </svg>
      <figcaption className="wb9-legend">
        {lines.map((s, k) => (
          <span key={s.id}><i style={{ background: CATEGORICAL_COLORS[k % CATEGORICAL_COLORS.length] }} />{s.label}</span>
        ))}
      </figcaption>
    </figure>
  );
}

/* 逐层成分条：两组正负量（注意力合计、MLP 合计）按层并排；共享一个对称色标 */
export function LayerBars({ rows, height = 64 }) {
  const valid = (rows || []).filter(r => Array.isArray(r.values));
  if (!valid.length) return null;
  const n = Math.max(...valid.map(r => r.values.length));
  const m = Math.max(1e-9, ...valid.flatMap(r => r.values.map(v => Math.abs(v || 0))));
  return (
    <div className="wb9-lbars">
      {valid.map(r => (
        <div key={r.id} className="wb9-lbar">
          <span>{r.label}</span>
          <div style={{ height }}>
            {Array.from({ length: n }, (_, i) => {
              const v = r.values[i] || 0;
              const h = (Math.abs(v) / m) * 50;
              return (
                <span key={i} title={`层 ${i}：${fmt(v, 3)}`}>
                  <i style={{ height: `${h}%`, [v >= 0 ? 'bottom' : 'top']: '50%', background: divColor(v >= 0 ? 0.85 : -0.85) }} />
                </span>
              );
            })}
          </div>
        </div>
      ))}
      <small>每根柱 = 一层，向上（蓝）推高上下文答案、向下（红）推高记忆答案；两行共用色标 ±{fmt(m, 3)}。</small>
    </div>
  );
}

/* 判据板：未通过与通过两栏同样大小、同样醒目；失效的判据（不计通过 / 未通过）另列在下面 */
export function CheckBoard({ checks, voids, empty = '报告里没有判据' }) {
  const list = checkList(checks);
  const vl = voidList(voids);
  if (!list.length && !vl.length) return <p className="v6-note">{empty}</p>;
  const bad = list.filter(c => !c.ok);
  const good = list.filter(c => c.ok);
  return (
    <div className="wb9-checks">
      <div className="bad">
        <h5>✗ 未通过 <b>{bad.length}</b></h5>
        {bad.length ? bad.map(c => <p key={c.key}>{c.label}<code>{c.key}</code></p>) : <p className="muted">无</p>}
      </div>
      <div className="good">
        <h5>✓ 通过 <b>{good.length}</b></h5>
        {good.length ? good.map(c => <p key={c.key}>{c.label}<code>{c.key}</code></p>) : <p className="muted">无</p>}
      </div>
      {vl.length > 0 && (
        <div className="void">
          <h5>⊘ 判据失效 <b>{vl.length}</b><small>不计通过、不计未通过</small></h5>
          {vl.map(c => <p key={c.key}>{c.label}<code>{c.key} · {c.why}</code></p>)}
        </div>
      )}
    </div>
  );
}
