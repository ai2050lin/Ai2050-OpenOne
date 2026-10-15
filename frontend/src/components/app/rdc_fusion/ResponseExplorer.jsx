/* 空间 · 实验响应（v6）：所有视图都绑定到真实结果；没有选中数据时只显示空态。
   v3 模板 → GET /api/agg/{tm}/v2（跨节点精确合并的充分统计量）：
     η² 剖面（层 × 因子 × 维度分箱，可取逐维完整值）/ 条件结构（格均值 cos + MDS）/
     留出复现（discovery 方向在 confirm / lockbox / lockbox_frame 上的 cos）/ 跨模型 / 行为门。
   legacy 模板 → GET /api/results 摘要 + /api/results/{sha} 的 cos 矩阵（单节点，不可精确合并）。 */
import { useMemo, useState } from 'react';
import { getJson, useApi, fmtNum } from './api.js';
import { Card, Empty, SourceTag, StatusPill, Tabs, divColor, inkOn, seqColor } from './ui.jsx';
import Space3D from './Space3D.jsx';
import FindingsView from './FindingsView.jsx';
import { packFile, packUrl } from './catalogUtil.js';

const CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const SPLIT_LABEL = { discovery: 'discovery 发现', confirm: 'confirm 留出', lockbox: 'lockbox 未见词表', lockbox_frame: 'lockbox_frame 未见句式' };
const VIEWS = [
  { id: 'space3d', label: '3D 响应空间', sub: '每点 = 一条样本' },
  { id: 'findings', label: '规律分析', sub: '留出 · 置换 · 不变性' },
  { id: 'eta', label: 'η² 剖面', sub: '层 × 因子 × 维度' },
  { id: 'structure', label: '条件结构', sub: '格均值 cos · MDS' },
  { id: 'replication', label: '留出复现', sub: '预测 vs 实测' },
  { id: 'models', label: '跨模型', sub: '功能结构对照' },
  { id: 'behavior', label: '行为门', sub: '逐格准确率' },
];

function useTip() {
  const [tip, setTip] = useState(null);
  const bind = text => ({
    onMouseMove: event => {
      const host = event.currentTarget.closest('.v6-tip-host');
      if (!host) return;
      const box = host.getBoundingClientRect();
      setTip({ x: event.clientX - box.left + 12, y: event.clientY - box.top + 12, text });
    },
    onMouseLeave: () => setTip(null),
  });
  const node = tip ? <div className="v6-tip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div> : null;
  return [bind, node];
}

/* ---------------- η² 剖面 ---------------- */
function EtaProfile({ tm, bucket, agg, layerTag, split }) {
  const [bind, tipNode] = useTip();
  const [full, setFull] = useState({ key: '', values: null, error: '' });
  const layer = agg.layers.find(row => row.tag === layerTag) || agg.layers[agg.layers.length - 1];
  const entry = layer?.by_split?.[split];
  if (!entry) return <Empty>该层在 {SPLIT_LABEL[split] || split} 划分没有数据。</Empty>;
  if (entry.insufficient) return <Empty>样本不足（N={entry.N}，条件格 {entry.C}），无法分解。</Empty>;
  const factors = Object.entries(entry.factors);
  const vmax = Math.max(1e-9, ...factors.flatMap(([, f]) => f.bins.max));
  const loadFull = async factor => {
    const key = `${layer.tag}|${factor}`;
    setFull({ key, values: null, error: '' });
    try {
      const data = await getJson(`/api/agg/${encodeURIComponent(tm)}/v2?bucket=${encodeURIComponent(bucket)}&layer=${encodeURIComponent(layer.tag)}&factor=${encodeURIComponent(factor)}`);
      const row = data.buckets?.[0]?.agg?.layers?.find(l => l.tag === layer.tag)?.full_eta2;
      setFull({ key, values: row?.values || null, error: row ? '' : '服务未返回逐维值' });
    } catch (reason) { setFull({ key, values: null, error: String(reason.message || reason) }); }
  };
  const downloadCsv = (factor, values) => {
    const csv = 'dim,eta2\n' + values.map((v, i) => `${i},${v}`).join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const a = document.createElement('a');
    a.href = url; a.download = `${tm}_${layer.tag}_${factor}_eta2.csv`; a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="v6-tip-host">
      <div className="v6-eta-meta">
        <span>N = <b>{entry.N}</b></span><span>条件格 <b>{entry.C}</b></span>
        <span>格间 η² 均值 <b>{fmtNum(entry.cells)}</b></span>
        {entry.interaction_residual !== undefined && <span>交互残余 <b>{fmtNum(entry.interaction_residual)}</b></span>}
        {entry.within !== undefined && <span>格内 <b>{fmtNum(entry.within)}</b></span>}
      </div>
      <table className="v6-eta-table">
        <thead><tr><th>因子</th><th>水平数</th><th>η² 均值</th><th>零假设基线</th><th>超出基线</th><th>p99</th><th className="wide">维度分箱（每格 = 该段维度 η² 最大值；色深 = 大）</th><th /></tr></thead>
        <tbody>
          {factors.map(([name, f]) => (
            <tr key={name}>
              <th scope="row">{name}</th>
              <td>{f.k}</td>
              <td className="num">{fmtNum(f.mean)}</td>
              <td className="num muted">{fmtNum(f.null)}</td>
              <td className="num"><b>{fmtNum(f.excess)}</b></td>
              <td className="num">{fmtNum(f.p99)}</td>
              <td className="wide">
                <div className="v6-strip" role="img" aria-label={`${name} η² 维度分箱`}>
                  {f.bins.max.map((value, i) => (
                    <span key={i} style={{ background: seqColor(value / vmax) }}
                      {...bind(`${name} · 维度 ${f.bins.edges[i]}–${f.bins.edges[i + 1] - 1}\n最大 η² ${fmtNum(value)} · 平均 ${fmtNum(f.bins.mean[i])}`)} />
                  ))}
                </div>
                <div className="v6-top">最高维：{f.top.slice(0, 6).map(([d, v]) => <code key={d}>{d}:{fmtNum(v, 3)}</code>)}</div>
                {full.key === `${layer.tag}|${name}` && full.values && (
                  <div className="v6-full">
                    <div className="v6-strip fine">
                      {full.values.map((value, i) => <span key={i} style={{ background: seqColor(value / vmax) }} {...bind(`维度 ${i} · η² ${fmtNum(value)}`)} />)}
                    </div>
                    <button type="button" className="fw-tbtn" onClick={() => downloadCsv(name, full.values)}>下载 CSV（{full.values.length} 维）</button>
                  </div>
                )}
                {full.key === `${layer.tag}|${name}` && full.error && <div className="fw-ai-err">{full.error}</div>}
              </td>
              <td><button type="button" className="fw-tbtn" onClick={() => loadFull(name)}>逐维完整值</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="v6-legend-seq"><span>0</span><i style={{ background: `linear-gradient(90deg, ${seqColor(0)}, ${seqColor(0.5)}, ${seqColor(1)})` }} /><span>{fmtNum(vmax)}</span><em>同一色标覆盖本层全部因子；分箱为显示聚合，原始逐维值可下载。</em></div>
      <h5 className="v6-h5">各层摘要（η² 均值，{SPLIT_LABEL[split] || split}）</h5>
      <table className="v6-mini-table">
        <thead><tr><th>层</th>{factors.map(([name]) => <th key={name}>{name}</th>)}</tr></thead>
        <tbody>
          {agg.layers.map(row => (
            <tr key={row.tag} className={row.tag === layer.tag ? 'on' : ''}>
              <th scope="row">{row.tag}</th>
              {factors.map(([name]) => {
                const v = row.by_split?.[split]?.factors?.[name]?.mean;
                return <td key={name} className="num">{fmtNum(v)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {tipNode}
    </div>
  );
}

/* ---------------- 条件结构：cos 矩阵 + 经典 MDS ---------------- */
function classicalMds(cos) {
  const n = cos.length;
  if (n < 3) return null;
  const d2 = cos.map(row => row.map(c => Math.max(0, 2 - 2 * c)));
  const rowMean = d2.map(r => r.reduce((a, b) => a + b, 0) / n);
  const all = rowMean.reduce((a, b) => a + b, 0) / n;
  const B = d2.map((r, i) => r.map((v, j) => -0.5 * (v - rowMean[i] - rowMean[j] + all)));
  const power = (M, deflate) => {
    let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1) + 1.1);
    let lambda = 0;
    for (let it = 0; it < 200; it++) {
      let w = M.map(r => r.reduce((s, x, j) => s + x * v[j], 0));
      if (deflate) { const dot = w.reduce((s, x, j) => s + x * deflate.v[j], 0); w = w.map((x, j) => x - dot * deflate.v[j]); }
      const norm = Math.sqrt(w.reduce((s, x) => s + x * x, 0)) || 1;
      lambda = norm; v = w.map(x => x / norm);
    }
    return { v, lambda };
  };
  const e1 = power(B);
  const B2 = B.map((r, i) => r.map((x, j) => x - e1.lambda * e1.v[i] * e1.v[j]));
  const e2 = power(B2, e1);
  return e1.v.map((x, i) => [x * Math.sqrt(e1.lambda), e2.v[i] * Math.sqrt(Math.max(e2.lambda, 0))]);
}

function ConditionStructure({ layer, colorFactor }) {
  const [bind, tipNode] = useTip();
  const cc = layer?.cell_cos;
  const coords = useMemo(() => (cc ? classicalMds(cc.matrix) : null), [cc]);
  if (!cc) return <Empty>该层没有 discovery 格均值（样本不足）。</Empty>;
  const keys = cc.keys;
  const parse = key => Object.fromEntries(key.split('|').map(part => part.split('=')));
  const short = key => key.replace(/\w+=/g, '').replace(/\|/g, '·');
  const dense = keys.length > 10;
  const levels = [...new Set(keys.map(k => parse(k)[colorFactor]))].sort();
  const colorOf = key => {
    const index = levels.indexOf(parse(key)[colorFactor]);
    return index >= 0 && index < CATEGORICAL.length ? CATEGORICAL[index] : '#94a3b8';
  };
  const cell = Math.max(14, Math.min(34, Math.floor(560 / Math.max(keys.length, 1))));
  let pts = null;
  if (coords) {
    const xs = coords.map(p => p[0]); const ys = coords.map(p => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    pts = coords.map(([x, y]) => [24 + (x - x0) / ((x1 - x0) || 1) * 250, 20 + (y - y0) / ((y1 - y0) || 1) * 240]);
  }
  return (
    <div className="v6-tip-host v6-structure">
      <div>
        <h5 className="v6-h5">格均值方向 cos（discovery，已减去总均值）</h5>
        <div className="v6-matrix" style={{ gridTemplateColumns: `minmax(90px,max-content) repeat(${keys.length}, ${cell}px)` }} role="img" aria-label="条件格 cos 矩阵">
          {cc.matrix.flatMap((row, i) => [
            <em key={`l-${i}`} className="v6-matrix-label" title={keys[i]}>{short(keys[i])}</em>,
            ...row.map((v, j) => (
              <span key={`${i}-${j}`} style={{ width: cell, height: cell, background: divColor(v), color: inkOn(v, true) }}
                {...bind(`${keys[i]}\n× ${keys[j]}\ncos = ${fmtNum(v)}`)}>{cell >= 28 ? v.toFixed(1) : ''}</span>
            )),
          ])}
        </div>
        <div className="v6-legend-seq"><span>−1</span><i style={{ background: `linear-gradient(90deg, ${divColor(-1)}, ${divColor(0)}, ${divColor(1)})` }} /><span>+1</span><em>行列顺序 = 条件键字典序（固定，不按相似度重排）</em></div>
      </div>
      <div>
        <h5 className="v6-h5">经典 MDS（距离 = 2 − 2cos；二维投影，仅示意相对位置）</h5>
        {dense && pts && <p className="v6-note">条件格较多（{keys.length}），标签隐藏；悬停点查看条件。</p>}
        {pts ? (
          <svg viewBox="0 0 360 290" className="v6-mds" role="img" aria-label="条件格 MDS 投影">
            {pts.map(([x, y], i) => (
              <g key={keys[i]} {...bind(keys[i])}>
                <circle cx={x} cy={y} r="11" fill="transparent" />
                <circle cx={x} cy={y} r="5" fill={colorOf(keys[i])} stroke="#fff" strokeWidth="2" />
                {!dense && <text x={x + 8} y={y + 3}>{short(keys[i])}</text>}
              </g>
            ))}
          </svg>
        ) : <Empty>条件格少于 3 个，不做投影。</Empty>}
        <div className="v6-legend">
          <span>着色因子：{colorFactor}</span>
          {levels.map((level, index) => <span key={level}><i style={{ background: index < CATEGORICAL.length ? CATEGORICAL[index] : '#94a3b8' }} />{level}</span>)}
        </div>
      </div>
      {tipNode}
    </div>
  );
}

/* ---------------- 留出复现 ---------------- */
function Replication({ agg, prereg }) {
  const [bind, tipNode] = useTip();
  const threshold = prereg?.min_lockbox_cos;
  const splits = ['confirm', 'lockbox', 'lockbox_frame'].filter(sp => agg.layers.some(l => l.replication?.[sp]));
  if (!splits.length) return <Empty>还没有留出划分的数据（需要 confirm / lockbox 分片结果）。</Empty>;
  const factors = agg.factors;
  return (
    <div className="v6-tip-host">
      <p className="v6-note">分数 = discovery 的因子水平方向（水平均值 − 总均值）与留出划分同一水平方向的平均 cos。留出划分的填充词从未参与发现；阈值 {threshold ?? '未预注册'} 冻结于模板 prereg。</p>
      {splits.map(sp => (
        <div key={sp} className="v6-rep-block">
          <h5 className="v6-h5">{SPLIT_LABEL[sp]}</h5>
          <table className="v6-rep-table">
            <thead><tr><th>因子 \ 层</th>{agg.layers.map(l => <th key={l.tag}>{l.tag}</th>)}<th>RDM 一致性</th></tr></thead>
            <tbody>
              {factors.map(f => (
                <tr key={f}>
                  <th scope="row">{f}</th>
                  {agg.layers.map(l => {
                    const rec = l.replication?.[sp]?.[f];
                    const v = rec?.mean_cos;
                    const pass = threshold != null && v != null && sp !== 'confirm' ? v >= threshold : null;
                    return (
                      <td key={l.tag} {...bind(rec ? Object.entries(rec.levels).map(([lvl, c]) => `${lvl}: ${fmtNum(c)}`).join('\n') : '无数据')}>
                        <div className="v6-bar"><i style={{ width: `${Math.max(0, v ?? 0) * 100}%` }} />{threshold != null && <em style={{ left: `${threshold * 100}%` }} />}</div>
                        <span className="num">{fmtNum(v)}</span>{pass !== null && <StatusPill status={pass ? 'pass' : 'fail'} label={pass ? '达阈' : '未达'} />}
                      </td>
                    );
                  })}
                  <td className="num">{f === factors[0] ? agg.layers.map(l => `${l.tag} ${fmtNum(l.rdm_agreement?.[sp]?.pearson)}`).join(' · ') : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      {tipNode}
    </div>
  );
}

/* ---------------- 跨模型 ---------------- */
function upperPearson(a, b) {
  const xs = [], ys = [];
  for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) { xs.push(a[i][j]); ys.push(b[i][j]); }
  if (xs.length < 3) return null;
  const m = v => v.reduce((s, x) => s + x, 0) / v.length;
  const mx = m(xs), my = m(ys);
  let sxy = 0, sxx = 0, syy = 0;
  xs.forEach((x, k) => { sxy += (x - mx) * (ys[k] - my); sxx += (x - mx) ** 2; syy += (ys[k] - my) ** 2; });
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : null;
}
function CrossModel({ buckets, factors }) {
  const ready = buckets.filter(b => b.agg);
  if (ready.length < 2) return <Empty>跨模型对照需要 ≥2 个模型桶（不同 model_id / model_rev / 精度）的合并结果；当前 {ready.length} 个。</Empty>;
  const tags = ready[0].agg.layers.map(l => l.tag);
  const rdm = (a, b, tag) => {
    const la = a.agg.layers.find(l => l.tag === tag)?.cell_cos; const lb = b.agg.layers.find(l => l.tag === tag)?.cell_cos;
    if (!la || !lb) return null;
    const shared = la.keys.filter(k => lb.keys.includes(k));
    const pick = (cc) => shared.map(k1 => shared.map(k2 => cc.matrix[cc.keys.indexOf(k1)][cc.keys.indexOf(k2)]));
    return upperPearson(pick(la), pick(lb));
  };
  return (
    <div>
      <p className="v6-note">不同模型的坐标不能一一对应：这里只比较功能结构——因子 η²（discovery）、未见词表复现分数，以及共享条件格上的 cos 结构相关（RDM）。</p>
      <table className="v6-rep-table">
        <thead><tr><th>指标</th><th>层</th>{ready.map(b => <th key={b.bucket} className="mono">{b.bucket}</th>)}</tr></thead>
        <tbody>
          {factors.flatMap(f => tags.map(tag => (
            <tr key={`${f}-${tag}`}>
              <th scope="row">η²({f})</th><td>{tag}</td>
              {ready.map(b => <td key={b.bucket} className="num">{fmtNum(b.agg.layers.find(l => l.tag === tag)?.by_split?.discovery?.factors?.[f]?.mean)}</td>)}
            </tr>
          )))}
          {factors.map(f => (
            <tr key={`rep-${f}`}>
              <th scope="row">lockbox cos({f})</th><td>最佳层</td>
              {ready.map(b => {
                const vals = b.agg.layers.map(l => l.replication?.lockbox?.[f]?.mean_cos).filter(v => v != null);
                return <td key={b.bucket} className="num">{vals.length ? fmtNum(Math.max(...vals)) : '—'}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <h5 className="v6-h5">模型间 RDM 相关（共享条件格，discovery）</h5>
      <table className="v6-mini-table">
        <thead><tr><th>模型对</th>{tags.map(t => <th key={t}>{t}</th>)}</tr></thead>
        <tbody>
          {ready.flatMap((a, i) => ready.slice(i + 1).map(b => (
            <tr key={a.bucket + b.bucket}><th scope="row" className="mono">{a.bucket} ~ {b.bucket}</th>{tags.map(t => <td key={t} className="num">{fmtNum(rdm(a, b, t))}</td>)}</tr>
          )))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------- 行为门 ---------------- */
function BehaviorTable({ bucket }) {
  const cells = Object.entries(bucket.behavior || {});
  if (!cells.length) return <Empty>该桶没有 probe 行为结果。</Empty>;
  return (
    <div>
      <p className="v6-note">阈值 {Math.round((bucket.behavior_threshold ?? 0.8) * 100)}%（模板 prereg.min_behavior_acc）。{bucket.gate_applied ? `合并时已排除 ${bucket.gate_excluded_cells.length} 个未达标格。` : '当前未应用行为门（显示全部格）。'}</p>
      <table className="v6-mini-table">
        <thead><tr><th>条件格</th><th>probe 数</th><th>top-1 准确率</th><th>行为门</th></tr></thead>
        <tbody>
          {cells.map(([cell, rec]) => (
            <tr key={cell}><th scope="row" className="mono">{cell}</th><td className="num">{rec.n}</td><td className="num">{rec.acc == null ? '—' : `${(rec.acc * 100).toFixed(0)}%`}</td><td><StatusPill status={rec.pass ? 'pass' : 'fail'} /></td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------- legacy 模板 ---------------- */
function LegacyResults({ tm }) {
  const results = useApi(`/api/results?tm_id=${encodeURIComponent(tm)}&kind=real&limit=50`);
  const [sel, setSel] = useState('');
  const detail = useApi(sel ? `/api/results/${sel}` : null);
  const [bind, tipNode] = useTip();
  if (!results.data) return <Empty>{results.error ? `结果库未接入（${results.error}）` : '读取结果…'}</Empty>;
  const rows = results.data.results;
  if (!rows.length) return <Empty>该 legacy 模板还没有 real 结果。</Empty>;
  const cos = detail.data?.cos;
  return (
    <div className="v6-tip-host">
      <p className="v6-note">legacy 模板只有单节点摘要（means.npz），不能跨节点精确合并、没有留出划分与金丝雀；用于回看历史结果。</p>
      <table className="v6-mini-table">
        <thead><tr><th>结果</th><th>模型</th><th>分析</th><th>摘要</th><th /></tr></thead>
        <tbody>
          {rows.map(r => {
            const d = r.summary_digest || {};
            const brief = d.eta2_by_factor ? Object.entries(d.eta2_by_factor).map(([k, v]) => `${k} ${fmtNum(v)}`).join(' · ')
              : d.topk_jaccard ? `head Jaccard ${Object.values(d.topk_jaccard).map(v => fmtNum(v, 2)).join(' / ')}`
                : d.eta2_by_cond ? Object.entries(d.eta2_by_cond).map(([k, v]) => `${k} ${fmtNum(v)}`).join(' · ') : (d.status || '—');
            return (
              <tr key={r.sha} className={sel === r.sha ? 'on' : ''}>
                <th scope="row" className="mono">{r.sha.slice(0, 10)}…</th><td>{r.model_id || '—'}</td><td>{d.analysis || '—'}</td><td className="dim2">{brief}</td>
                <td><button type="button" className="fw-tbtn" onClick={() => setSel(r.sha)}>cos 矩阵</button></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {sel && !cos && <Empty>{detail.loading ? '读取结果文件…' : '该结果没有可提取的 cos 矩阵。'}</Empty>}
      {cos && (
        <div className="v6-matrix" style={{ gridTemplateColumns: `repeat(${cos.keys.length}, 30px)` }}>
          {cos.matrix.flatMap((row, i) => row.map((v, j) => (
            <span key={`${i}-${j}`} style={{ width: 30, height: 30, background: divColor(v), color: inkOn(v, true) }} {...bind(`${cos.keys[i]} × ${cos.keys[j]}\ncos = ${fmtNum(v)}`)}>{v.toFixed(1)}</span>
          )))}
        </div>
      )}
      {tipNode}
    </div>
  );
}

export default function ResponseExplorer({ initialTm, initialBucket, initialLayer, onContext }) {
  const templates = useApi('/api/templates', { interval: 60000 });
  const list = (templates.data?.templates || []).slice().sort((a, b) =>
    (a.protocol === 'v3' ? 0 : 1) - (b.protocol === 'v3' ? 0 : 1) || a.tm_id.localeCompare(b.tm_id));
  const [picked, setPicked] = useState(initialTm || '');
  const tm = picked || list.find(t => t.protocol === 'v3')?.tm_id || list[0]?.tm_id || '';
  const tpl = list.find(t => t.tm_id === tm);
  const isV3 = tpl?.protocol === 'v3';
  const [gate, setGate] = useState(true);
  const agg = useApi(tm && isV3 ? `/api/agg/${encodeURIComponent(tm)}/v2?gate=${gate}` : null);
  const [bucketPick, setBucketPick] = useState(initialBucket || '');
  const buckets = agg.data?.buckets || [];
  const bucket = buckets.find(b => b.bucket === bucketPick) || buckets.find(b => b.agg && b.complete) || buckets.find(b => b.agg) || buckets[0];
  const [view, setView] = useState('space3d');
  const [method, setMethod] = useState('');
  const [colorBy, setColorBy] = useState('');
  const [hiddenSplits, setHiddenSplits] = useState(() => new Set());
  const [layerPick, setLayerPick] = useState(initialLayer || '');
  const [split, setSplit] = useState('discovery');
  const [colorFactor, setColorFactor] = useState('');
  const layers = bucket?.agg?.layers || [];
  const layerTag = layers.find(l => l.tag === layerPick)?.tag || layers[layers.length - 1]?.tag;
  const layer = layers.find(l => l.tag === layerTag);
  const factors = agg.data?.factors || tpl?.factors || [];
  const wantAnalysis = isV3 && bucket?.agg && (view === 'space3d' || view === 'findings');
  const analysis = useApi(wantAnalysis ? `/api/analysis/${encodeURIComponent(tm)}?bucket=${encodeURIComponent(bucket.bucket)}&gate=${gate}` : null);
  const an = analysis.data?.available ? analysis.data : null;
  const methodV = method || (factors[0] ? `lda:${factors[0]}` : 'pca');
  const colorByV = colorBy || (factors[0] ? `factor:${factors[0]}` : 'split');
  const visibleSplits = useMemo(() => new Set(['discovery', 'confirm', 'lockbox', 'lockbox_frame'].filter(sp => !hiddenSplits.has(sp))), [hiddenSplits]);
  const synthetic = Boolean(bucket?.synthetic || String(bucket?.bucket || '').startsWith('合成·'));

  const choose = value => { setPicked(value); setBucketPick(''); setLayerPick(''); onContext?.({ tm: value }); };

  if (!templates.data) {
    return <Empty>{templates.error ? `中心服务未接入（${templates.error}）。实验响应只显示真实运行产物，不使用演示数据。` : '读取模板…'}</Empty>;
  }
  if (!list.length) return <Empty>中心服务还没有任何模板。</Empty>;
  return (
    <div className="v6-response">
      <div className="v6-toolbar">
        <label>模板
          <select value={tm} onChange={event => choose(event.target.value)}>
            {list.map(t => <option key={t.tm_id} value={t.tm_id}>{t.tm_id} · {t.name}{t.protocol === 'v3' ? '' : '（legacy）'}</option>)}
          </select>
        </label>
        {isV3 && (
          <>
            <label>模型桶
              <select value={bucket?.bucket || ''} onChange={event => setBucketPick(event.target.value)} disabled={!buckets.length}>
                {!buckets.length && <option value="">无合并结果</option>}
                {buckets.map(b => <option key={b.bucket} value={b.bucket}>{b.bucket}{b.complete ? '' : `（分片 ${b.shards_present.length}/${b.shards_expected}）`}</option>)}
              </select>
            </label>
            <label>层
              <select value={layerTag || ''} onChange={event => setLayerPick(event.target.value)} disabled={!layers.length || (view === 'space3d' && methodV === 'stack')}>
                {layers.map(l => <option key={l.tag} value={l.tag}>{l.tag}</option>)}
              </select>
            </label>
            {view === 'space3d' && (
              <>
                <label>坐标
                  <select value={methodV} onChange={event => setMethod(event.target.value)}>
                    {factors.slice(0, 2).map(f => <option key={f} value={`lda:${f}`}>LDA · {f}</option>)}
                    <option value="pca">PCA（原始）</option>
                    <option value="stack">层堆叠 · {factors[0]} 轨迹</option>
                  </select>
                </label>
                <label>着色
                  <select value={colorByV} onChange={event => setColorBy(event.target.value)}>
                    {factors.map(f => <option key={f} value={`factor:${f}`}>{f}</option>)}
                    <option value="split">划分</option>
                    <option value="correct">模型作答对错（行为门）</option>
                  </select>
                </label>
              </>
            )}
            {view === 'eta' && (
              <label>划分
                <select value={split} onChange={event => setSplit(event.target.value)}>
                  {(bucket?.agg?.splits || ['discovery']).map(sp => <option key={sp} value={sp}>{SPLIT_LABEL[sp] || sp}</option>)}
                </select>
              </label>
            )}
            {view === 'structure' && (
              <label>着色
                <select value={colorFactor || factors[0] || ''} onChange={event => setColorFactor(event.target.value)}>
                  {factors.map(f => <option key={f} value={f}>{f}</option>)}
                </select>
              </label>
            )}
            <label className="fw-chk"><input type="checkbox" checked={gate} onChange={event => setGate(event.target.checked)} /> 应用行为门</label>
          </>
        )}
        {isV3 && bucket?.agg && (
          <a className="fw-tbtn" href={packUrl(tm, bucket.bucket)} download={packFile(tm, bucket.bucket)} title="研究包 respack.v1：语料、合并统计、规律分析（含 3D 坐标）、充分统计量、逐条向量；可在「打开本地包」离线查看">下载研究包 ↓</a>
        )}
        <span className="v6-toolbar-src"><SourceTag state={isV3 ? (agg.data ? 'live' : agg.loading ? 'loading' : 'none') : 'live'} label={isV3 ? 'AGG-v2' : '结果库'} /></span>
      </div>

      {tpl && (
        <div className="v6-ctx">
          <b>{tpl.name}</b>
          <span>{tpl.dim}</span>
          <span className="mono">design {String(tpl.design_sha || '').slice(0, 12)}</span>
          {isV3 && tpl.by_split && <span>{Object.entries(tpl.by_split).map(([k, v]) => `${k} ${v}`).join(' · ')}</span>}
          {bucket && <span>分片 {bucket.shards_present.length}/{bucket.shards_expected}{bucket.complete ? '' : '（未完整：统计只覆盖已回传分片）'}</span>}
          {bucket?.excluded?.length > 0 && <span className="warn">排除 {bucket.excluded.length} 份：{[...new Set(bucket.excluded.map(e => e.reason))].join('、')}</span>}
        </div>
      )}

      {synthetic && (
        <div className="v6-synthetic">
          <b>合成数据 · 方法与流程验证</b>
          <span>该模型桶来自沙盒中心的 runner 合成模式（按已知规律生成），只用于检验分析方法与客户端流程，不代表任何真实模型，不能写入研究记录。</span>
        </div>
      )}

      {!isV3 && tm && <Card title="legacy 结果"><LegacyResults tm={tm} /></Card>}
      {isV3 && (
        <>
          <Tabs tabs={VIEWS} value={view} onChange={setView} label="实验响应视图" />
          {!agg.data && <Empty>{agg.error ? `合并统计读取失败（${agg.error}）` : '合并统计计算中…'}</Empty>}
          {agg.data && !bucket && <Empty>该模板还没有 real 结果。去「研发 · 分发」领取分片任务后，这里显示跨节点合并统计。</Empty>}
          {bucket && !bucket.agg && <Empty>该桶的结果全部被排除（{bucket.excluded.map(e => e.reason).join('、') || '无可合并统计'}）。</Empty>}
          {bucket?.agg && (
            <Card>
              {(view === 'space3d' || view === 'findings') && !an && (
                <Empty>{analysis.loading ? '规律分析计算中（置换检验约需数秒）…' : (analysis.data?.reason || analysis.error || '规律分析不可用')}</Empty>
              )}
              {view === 'space3d' && an && (
                <>
                  <div className="v6-split-toggles">
                    {['discovery', 'confirm', 'lockbox', 'lockbox_frame'].map(sp => (
                      <label key={sp} className="fw-chk"><input type="checkbox" checked={!hiddenSplits.has(sp)}
                        onChange={() => setHiddenSplits(prev => { const n = new Set(prev); if (n.has(sp)) n.delete(sp); else n.add(sp); return n; })} /> {SPLIT_LABEL[sp]}</label>
                    ))}
                    <span className="v6-note">{an.n_items} 条样本 · 向量空间 {an.space}{an.n_excluded ? ` · 行为门排除 ${an.n_excluded} 条` : ''} · 拖动旋转、滚轮缩放、悬停看样本</span>
                  </div>
                  <Space3D analysis={an} layerTag={layerTag} method={methodV} colorBy={colorByV} visibleSplits={visibleSplits} />
                  <p className="v6-note">
                    {methodV === 'stack' ? an.stack?.note
                      : methodV === 'pca' ? '坐标 = discovery 中心化后的前三个主成分（原始空间）。'
                        : `坐标 = 白化空间中「${methodV.slice(4)}」各水平质心的判别方向（只用 discovery 拟合），所有划分投影到同一组轴；轴的长度与方向只表示相对结构，不是模型的原生坐标。`}
                  </p>
                </>
              )}
              {view === 'findings' && an && <FindingsView analysis={an} />}
              {view === 'eta' && <EtaProfile tm={tm} bucket={bucket.bucket} agg={bucket.agg} layerTag={layerTag} split={split} />}
              {view === 'structure' && <ConditionStructure layer={layer} colorFactor={colorFactor || factors[0]} />}
              {view === 'replication' && <Replication agg={bucket.agg} prereg={agg.data.prereg} />}
              {view === 'models' && <CrossModel buckets={buckets} factors={factors} />}
              {view === 'behavior' && <BehaviorTable bucket={bucket} />}
            </Card>
          )}
          {agg.data && <p className="v6-note">{agg.data.policy}</p>}
        </>
      )}
    </div>
  );
}
