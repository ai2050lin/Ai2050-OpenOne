/* v8 语言项页的三张图（SVG，无外部依赖）：逐层曲线、峰值层点云、逐块选择性柱状图。
   悬停每个标记都有原始数值；颜色按固定顺序分配，图例带文字。 */
import { useState } from 'react';

import { CATEGORICAL, READOUT_LABEL, seq } from './v8Util.js';

const BASELINE_TEXT = { identical: '各样本 token 相同', not_separable: '不可分', separable: '已可分（token 身份）', missing: '无记录' };

export function LayerCurve({ readouts, nLayers }) {
  const ros = Object.keys(readouts || {});
  const [hover, setHover] = useState(null);
  if (!ros.length) return null;
  const W = 680, H = 240, L = 40, R = 14, T = 16, B = 30;
  const xmax = nLayers || Math.max(...ros.flatMap(r => readouts[r].curve.map(c => c.layer)));
  const ymax = Math.max(1e-6, ...ros.flatMap(r => readouts[r].curve.map(c => c.sep || 0))) * 1.1;
  const x = l => L + ((W - L - R) * l) / xmax, y = v => T + (H - T - B) * (1 - v / ymax);
  const step = Math.max(1, Math.ceil(ymax / 4));
  return (
    <div className="v8-chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="逐层留出投影间隔">
        {Array.from({ length: Math.floor(ymax / step) + 1 }, (_, i) => i * step).map(v => (
          <g key={v}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="#eef2f7" />
            <text x={L - 6} y={y(v) + 3} fontSize="10" fill="#94a3b8" textAnchor="end">{v}</text></g>
        ))}
        {Array.from({ length: Math.floor(xmax / 4) + 1 }, (_, i) => i * 4).map(l => (
          <text key={l} x={x(l)} y={H - 12} fontSize="10" fill="#94a3b8" textAnchor="middle">{l === 0 ? 'hs[0]' : `L${l}`}</text>
        ))}
        <rect x={x(0) - 6} y={T} width="12" height={H - T - B} fill="#f1f5f9" />
        <text x={x(0)} y={T - 4} fontSize="9.5" fill="#64748b" textAnchor="middle">嵌入层</text>
        {ros.map((r, k) => {
          const c = readouts[r].curve, col = CATEGORICAL[k % CATEGORICAL.length], pk = readouts[r].peak;
          return (
            <g key={r}>
              <polyline fill="none" stroke={col} strokeWidth="2" points={c.map(p => `${x(p.layer)},${y(p.sep || 0)}`).join(' ')} />
              {c.map(p => (
                <circle key={p.layer} cx={x(p.layer)} cy={y(p.sep || 0)} r="3.6" fill={p.readable ? col : '#fff'} stroke={col} strokeWidth="1.5"
                        onMouseEnter={() => setHover({ r, ...p, x: x(p.layer), y: y(p.sep || 0) })} onMouseLeave={() => setHover(null)} />
              ))}
              {pk && <text x={x(pk.layer)} y={y(pk.sep) - 9} fontSize="10.5" fill={col} textAnchor="middle" fontWeight="600">峰 L{pk.layer}</text>}
            </g>
          );
        })}
        {hover && (
          <g pointerEvents="none">
            <rect x={Math.min(hover.x + 8, W - 190)} y={Math.max(hover.y - 46, 2)} width="182" height="40" rx="6" fill="#0f172a" />
            <text x={Math.min(hover.x + 16, W - 182)} y={Math.max(hover.y - 30, 18)} fontSize="11" fill="#f8fafc">
              {READOUT_LABEL[hover.r] || hover.r} · {hover.layer === 0 ? 'hs[0]' : `L${hover.layer}`} · 间隔 {(hover.sep || 0).toFixed(2)}
            </text>
            <text x={Math.min(hover.x + 16, W - 182)} y={Math.max(hover.y - 15, 33)} fontSize="11" fill="#cbd5e1">
              {hover.readable ? '● 未见词表上可读出' : '○ 未达判据'}{hover.acc != null ? ` · 准确率 ${hover.acc}` : ''}
            </text>
          </g>
        )}
      </svg>
      <div className="v8-lg">
        {ros.map((r, k) => (
          <span key={r}><i style={{ background: CATEGORICAL[k % CATEGORICAL.length] }} />{READOUT_LABEL[r] || r}读出位置 · 嵌入层基线：{BASELINE_TEXT[readouts[r].baseline?.state] || '—'}</span>
        ))}
        <span>● 实心 = 未见词表上可读出 · ○ 空心 = 未达判据</span>
      </div>
    </div>
  );
}

export function ScatterPlot({ scatter }) {
  const [hover, setHover] = useState(null);
  if (!scatter?.pts?.length) return null;
  const W = 380, H = 280, P = 18, pts = scatter.pts;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const X = v => P + ((W - 2 * P) * (v - x0)) / (x1 - x0 || 1), Y = v => H - P - ((H - 2 * P) * (v - y0)) / (y1 - y0 || 1);
  const levels = scatter.levels?.length ? scatter.levels : [...new Set(pts.map(p => p[2]))];
  const col = lv => CATEGORICAL[Math.max(0, levels.indexOf(lv)) % CATEGORICAL.length];
  return (
    <div className="v8-chart v8-scatter">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${scatter.tag} 点云`}>
        <rect x="0" y="0" width={W} height={H} rx="8" fill="#fbfdff" />
        {pts.map((p, i) => {
          const held = p[3] !== 'discovery';
          return (
            <circle key={i} cx={X(p[0])} cy={Y(p[1])} r={held ? 3.8 : 2.8} fill={held ? '#fff' : col(p[2])} stroke={held ? col(p[2]) : '#fff'}
                    strokeWidth={held ? 1.6 : 0.6} onMouseEnter={() => setHover({ p, x: X(p[0]), y: Y(p[1]) })} onMouseLeave={() => setHover(null)} />
          );
        })}
        <text x={W / 2} y={H - 4} fontSize="10" fill="#94a3b8" textAnchor="middle">LDA 轴 1</text>
      </svg>
      {hover && (
        <div className="v8-tip" style={{ left: `${(100 * hover.x) / W}%`, top: `${(100 * hover.y) / H}%` }}>
          {hover.p[4]}<small>{hover.p[2]} · {hover.p[3]}</small>
        </div>
      )}
      <div className="v8-lg">
        {levels.map(lv => <span key={lv}><i style={{ background: col(lv) }} />{lv}</span>)}
        <span>实心 = 发现集 · 空心环 = 留出（confirm / 未见词表 / 未见句式）{scatter.sampled ? ` · 按序抽样 ${pts.length}/${scatter.n_total}` : ''}</span>
      </div>
    </div>
  );
}

export function SelBars({ counts, block, onPick }) {
  if (!counts?.length) return null;
  const W = 680, H = 150, L = 36, B = 22, T = 8, max = Math.max(1, ...counts), bw = (W - L - 6) / counts.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="逐块选择性单元数">
      {[0, max / 2, max].map(v => {
        const yy = T + (H - T - B) * (1 - v / max);
        return <g key={v}><line x1={L} x2={W} y1={yy} y2={yy} stroke="#eef2f7" /><text x={L - 5} y={yy + 3} fontSize="10" fill="#94a3b8" textAnchor="end">{Math.round(v)}</text></g>;
      })}
      {counts.map((n, b) => {
        const h = ((H - T - B) * n) / max;
        return (
          <g key={b} style={{ cursor: 'pointer' }} onClick={() => onPick(b)}>
            <rect x={L + b * bw} y={T} width={bw} height={H - T - B} fill="transparent" />
            <rect x={L + b * bw + 1} y={H - B - h} width={Math.max(1, bw - 2)} height={h} rx="2" fill={b === block ? '#e34948' : seq(0.25 + (0.7 * n) / max)}>
              <title>块 {b}：{n} 个选择性单元</title>
            </rect>
            {b % 5 === 0 && <text x={L + b * bw + bw / 2} y={H - 7} fontSize="10" fill="#94a3b8" textAnchor="middle">{b}</text>}
          </g>
        );
      })}
    </svg>
  );
}
