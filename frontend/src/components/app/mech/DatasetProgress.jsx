/* 数据集进度：每个数据集在两条研究线上的阶段状态（神经元逆向 · Goodfire 式；注意力语言图谱）、逐层可视化与可贡献的计算。
   数据：/api/progress/datasets（dataset_progress.v1，只读本机数据包）。状态只描述产物是否存在、判据有何结论，不是证据等级；
   未支持的检验与通过的同样显示。层条的位置只是层序，不代表模型几何。 */
import { useEffect, useRef, useState } from 'react';

import { DESKTOP, useApi } from '../rdc_fusion/api.js';
import { Empty } from '../rdc_fusion/ui.jsx';
import { STATUS_META, STATUS_ORDER, TRACKS, contributeLine, groupDatasets, layerBars, reached, stageCounts, statusMeta } from './progressUtil.js';

function Cell({ cell, stage, onPick, active }) {
  const m = statusMeta(cell?.status);
  return (
    <td className="dp-cell">
      <button type="button" className={active ? 'on' : ''} style={{ background: m.bg, color: m.ink }} onClick={onPick}
              title={`${stage.id} ${stage.label}：${m.label}${cell?.value ? ` · ${cell.value}` : ''}`}
              aria-label={`${stage.id} ${stage.label} ${m.label}`}>{m.sym}</button>
    </td>
  );
}

function StageBar({ datasets, track, stage }) {
  const c = stageCounts(datasets, track, stage.id);
  const total = c.applicable || 1;
  return (
    <div className="dp-stagebar" title={STATUS_ORDER.filter(s => s !== 'na' && c[s]).map(s => `${statusMeta(s).label} ${c[s]}`).join('，')}>
      <span className="dp-stagebar-l">{stage.id} {stage.label}</span>
      <span className="dp-stagebar-b">
        {STATUS_ORDER.filter(s => s !== 'na' && c[s]).map(s => (
          <i key={s} style={{ width: `${(c[s] / total) * 100}%`, background: statusMeta(s).bg === 'transparent' ? '#fff' : statusMeta(s).bg }} />
        ))}
      </span>
      <span className="dp-stagebar-n">{c.done + c.partial} / {c.applicable}</span>
    </div>
  );
}

function LayerStrip({ layer }) {
  if (!layer?.values?.length) return <p className="dp-none">这条线还没有逐层数据。</p>;
  const { bars, max } = layerBars(layer.values, layer.kind);
  const ok = layer.values_ok ? layerBars(layer.values_ok, 'sequential').bars : null;
  const marks = new Set(layer.marks || []);
  const n = bars.length, W = 100 / n, H = 64, mid = layer.kind === 'diverging' ? H / 2 : H;
  return (
    <figure className="dp-strip">
      <figcaption>{layer.label}</figcaption>
      <svg viewBox={`0 0 100 ${H + 12}`} preserveAspectRatio="none" role="img" aria-label={layer.label}>
        {layer.kind === 'diverging' && <line x1="0" x2="100" y1={mid} y2={mid} stroke="#94a3b8" strokeWidth="0.3" />}
        {bars.map((b, i) => {
          if (b === null) return <rect key={i} x={i * W + W * 0.15} width={W * 0.7} y={H - 1} height="1" fill="#e2e8f0" />;
          const h = Math.max(Math.abs(b) * (layer.kind === 'diverging' ? H / 2 : H), 0.6);
          const y = layer.kind === 'diverging' ? (b >= 0 ? mid - h : mid) : H - h;
          const fill = layer.kind === 'diverging' ? (b >= 0 ? '#2a78d6' : '#d6453d') : layer.kind === 'progress' ? '#cde2fb' : '#3987e5';
          return (
            <g key={i}>
              <rect x={i * W + W * 0.15} width={W * 0.7} y={y} height={h} fill={fill}>
                <title>{`层 ${i}：${layer.values[i]}${ok ? `（已解释 ${layer.values_ok[i]}）` : ''}`}</title>
              </rect>
              {ok && ok[i] !== null && ok[i] > 0 && (
                <rect x={i * W + W * 0.15} width={W * 0.7} y={H - (layer.values_ok[i] / max) * H} height={(layer.values_ok[i] / max) * H} fill="#184f95" />
              )}
              {marks.has(i) && <rect x={i * W + W * 0.15} width={W * 0.7} y={H + 3} height="3" fill="#0f172a" />}
            </g>
          );
        })}
      </svg>
      <div className="dp-axis"><span>层 0</span><span>层 {n - 1}</span></div>
      {layer.note && <small>{layer.note}{marks.size ? '；底部黑点 = 达到可读出判据的层' : ''}。条的位置只是层序，不代表模型几何。</small>}
    </figure>
  );
}

function TrackDetail({ ds, track, stages, onLink, focus }) {
  const t = TRACKS.find(x => x.id === track);
  return (
    <div className={`dp-track ${focus ? 'focus' : ''}`}>
      <h4>{t.label}<small>{t.sub}</small></h4>
      <ol className="dp-stages">
        {stages.map(s => {
          const c = ds[track]?.[s.id] || {};
          const m = statusMeta(c.status);
          return (
            <li key={s.id} className={`st-${c.status || 'unavailable'}`}>
              <span className="dp-badge" style={{ background: m.bg, color: m.ink }}>{m.sym} {m.label}</span>
              <div>
                <b>{s.id} {s.label}</b>{c.value && <span className="dp-val">{c.value}</span>}
                {c.detail && <small>{c.detail}</small>}
                {c.sources?.length > 0 && <small className="mono">来源：{c.sources.join('、')}</small>}
              </div>
              {c.link && <button type="button" className="fw-tbtn ghost" onClick={() => onLink(c.link)}>查看 →</button>}
            </li>
          );
        })}
      </ol>
      <LayerStrip layer={ds.layers?.[track]} />
    </div>
  );
}

function Contribute({ ds, onGo }) {
  const c = contributeLine(ds.contribute);
  const [copied, setCopied] = useState(false);
  const cmd = ds.contribute?.claim_command;
  return (
    <div className={`dp-contrib ${c.open ? 'open' : ''}`}>
      <div>
        <b>{c.open ? '● 有可领取的计算' : '○ 当前没有待领取的分片'}</b>
        <span>{c.text}</span>
        {ds.contribute?.note && <small>{ds.contribute.note}</small>}
      </div>
      <div className="dp-contrib-act">
        {cmd && (
          <button type="button" className="fw-tbtn ghost" onClick={() => {
            navigator.clipboard?.writeText(cmd).then(() => { setCopied(true); window.setTimeout(() => setCopied(false), 1500); }).catch(() => {});
          }}>{copied ? '已复制' : '复制领题命令'}</button>
        )}
        <button type="button" className="fw-tbtn" onClick={() => onGo('contribute')}>
          {DESKTOP ? '贡献算力 →' : '下载客户端贡献算力 →'}
        </button>
      </div>
    </div>
  );
}

export default function DatasetProgress({ onGo, onItem }) {
  const api = useApi('/api/progress/datasets');
  const [pick, setPick] = useState(null);         // {id, track?}
  const detailRef = useRef(null);
  useEffect(() => {                                 // 详情在矩阵下方：选中后滚到详情
    if (pick && detailRef.current) detailRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [pick]);
  if (api.loading) return <Empty>读取…</Empty>;
  if (api.error) return <Empty>未接入：数据集进度不在本机数据包里（{api.error}）。更新数据包后显示。</Empty>;
  const doc = api.data;
  if (!doc?.datasets?.length) return <Empty>未接入：还没有任何数据集的进度。</Empty>;
  const stages = doc.stages || {};
  const groups = groupDatasets(doc.datasets);
  const sel = doc.datasets.find(d => d.id === pick?.id);
  const link = l => {
    if (l.item && onItem) onItem(l.item);
    else onGo(l.lens, l.sub);
  };
  const toggle = (id, track) => setPick(p => (p?.id === id && (!track || p.track === track) ? null : { id, track }));
  const n = doc.datasets.length;
  const reachedAll = track => doc.datasets.filter(d => (stages[track] || []).some(s => ['done', 'partial', 'negative'].includes(d[track]?.[s.id]?.status))).length;
  return (
    <>
      <div className="dp-summary">
        {TRACKS.map(t => (
          <div key={t.id} className="dp-sum">
            <h4>{t.label}<small>{reachedAll(t.id)} / {n} 个数据集已开始</small></h4>
            {(stages[t.id] || []).map(s => <StageBar key={s.id} datasets={doc.datasets} track={t.id} stage={s} />)}
          </div>
        ))}
      </div>
      <div className="dp-legend">
        {STATUS_ORDER.map(s => <span key={s}><i style={{ background: STATUS_META[s].bg, color: STATUS_META[s].ink, border: s === 'na' ? '1px dashed #cbd5e1' : 0 }}>{STATUS_META[s].sym}</i>{STATUS_META[s].label}</span>)}
        <span className="dp-legend-note">每条横条 = 各数据集在该阶段的状态分布（不适用不计入）；数字 = 完成或部分 / 适用的数据集数。不是转化率。</span>
      </div>
      <div className="dp-matrix-wrap">
        <table className="dp-matrix">
          <thead>
            <tr>
              <th rowSpan={2}>数据集</th>
              {TRACKS.map(t => <th key={t.id} colSpan={(stages[t.id] || []).length} className="dp-trackhd">{t.label}</th>)}
              <th rowSpan={2}>贡献算力</th>
            </tr>
            <tr>{TRACKS.flatMap(t => (stages[t.id] || []).map(s => <th key={s.id} className="dp-stagehd" title={s.desc}>{s.id}<span>{s.label}</span></th>))}</tr>
          </thead>
          {groups.map(g => (
            <tbody key={g.kind}>
              <tr className="dp-group"><td colSpan={2 + TRACKS.reduce((a, t) => a + (stages[t.id] || []).length, 0)}>{g.label}<small>{g.rows.length}</small></td></tr>
              {g.rows.map(d => {
                const c = contributeLine(d.contribute);
                return (
                  <tr key={d.id} className={pick?.id === d.id ? 'sel' : ''}>
                    <th scope="row">
                      <button type="button" className="dp-name" onClick={() => toggle(d.id)}>
                        {d.name}
                        <small>{(d.models || []).join(' · ') || d.desc || ''}</small>
                        <span className="dp-reach">
                          {TRACKS.map(t => <i key={t.id} title={`${t.label}：走到第 ${reached(d, t.id, stages[t.id])} 步`}
                                              style={{ width: `${(reached(d, t.id, stages[t.id]) / Math.max((stages[t.id] || []).length, 1)) * 100}%` }}
                                              className={t.id} />)}
                        </span>
                      </button>
                    </th>
                    {TRACKS.flatMap(t => (stages[t.id] || []).map(s => (
                      <Cell key={`${t.id}${s.id}`} cell={d[t.id]?.[s.id]} stage={s} active={pick?.id === d.id && pick?.track === t.id}
                            onPick={() => toggle(d.id, t.id)} />
                    )))}
                    <td className="dp-ctd">
                      <button type="button" className={`dp-cbtn ${c.open ? 'open' : ''}`} onClick={() => toggle(d.id)} title={c.text}>
                        {c.open ? '● 可领取' : '○ 无'}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          ))}
        </table>
      </div>
      {sel ? (
        <div className="dp-detail" ref={detailRef} aria-label={`${sel.name} 的进度`}>
          <header>
            <h3>{sel.name}</h3>
            <small>{sel.desc || ''}{sel.models?.length ? ` · 模型：${sel.models.join('、')}` : ''}</small>
            <button type="button" className="fw-tbtn ghost" onClick={() => setPick(null)}>收起</button>
          </header>
          <div className="dp-tracks">
            {TRACKS.map(t => (
              <TrackDetail key={t.id} ds={sel} track={t.id} stages={stages[t.id] || []} onLink={link} focus={pick.track === t.id} />
            ))}
          </div>
          <Contribute ds={sel} onGo={onGo} />
        </div>
      ) : <p className="v6-note">点数据集名称或任一格，查看两条线的逐阶段说明、逐层图与可贡献的计算。</p>}
      {doc.note && <p className="v6-note">{doc.note}</p>}
    </>
  );
}
