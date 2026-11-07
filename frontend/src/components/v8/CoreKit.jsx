/* 页面公用部件（Phase 333 / 334，AGENTS.md 第 10 节“设计风格”）。
   Phase 334 按线上版（server.ai2050.com）的结构调整：页面标题 + 卡片式核心模块 + 左对齐的折叠行。 */
import { useState } from 'react';

import './core_kit.css';

/* 环形进度：value 为 0–1；null 表示没有分母，只画灰环 */
export function Ring({ value, label, size = 120, color = '#183f77', ariaLabel }) {
  const r = size / 2 - 9, c = 2 * Math.PI * r;
  const v = value == null || !Number.isFinite(value) ? 0 : Math.max(0, Math.min(1, value));
  return <svg className="ck-ring" width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={ariaLabel || label}>
    <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#eef1f5" strokeWidth="10" />
    {v > 0 && <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
      strokeDasharray={`${Math.max(c * v, 1).toFixed(1)} ${c.toFixed(1)}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} />}
    <text x="50%" y="52%" textAnchor="middle" dominantBaseline="middle" fontSize={size / 5.4} fontWeight="700" fill="#172033">{label}</text>
  </svg>;
}

/* 细进度条：value 为 0–1；null 时不画 */
export function Bar({ value, color = '#183f77', label }) {
  const v = value == null || !Number.isFinite(value) ? null : Math.max(0, Math.min(1, value));
  return <span className="ck-bar" role={v == null ? undefined : 'progressbar'} aria-label={label} aria-valuenow={v == null ? undefined : Math.round(v * 100)}>
    {v != null && <i style={{ width: `${v > 0 ? Math.max(v * 100, 1) : 0}%`, background: color }} />}</span>;
}

/* 条形关卡：n / d */
export function GateBar({ name, n, d, color = '#183f77', onClick }) {
  const w = d > 0 ? (n / d) * 100 : 0;
  const body = <><span>{name}</span><i><b style={{ width: `${w}%`, background: color }} /></i><span className="v8-note">{n} / {d}</span></>;
  return onClick ? <button type="button" className="ck-gate" onClick={onClick}>{body}</button> : <div className="ck-gate">{body}</div>;
}

/* 页面标题：与线上版一致的 h2 + 一句说明；右侧可放范围或操作 */
export function PageHead({ title, sub, note, children, id }) {
  return <header className="v8-overview-heading ck-head"><div>
    <h2 id={id}>{title}</h2>{sub && <p>{sub}</p>}{note && <p className="v8-note">{note}</p>}</div>{children}</header>;
}

/* 核心模块的外框：卡片，左上角一个小标题 */
export function CoreCard({ question, children, hint, label }) {
  return <section className="v8-card ck-core" aria-label={label || (typeof question === 'string' ? question : undefined)}>
    {question && <h3 className="ck-q">{question}</h3>}
    {children}
    {hint && <p className="ck-hint">{hint}</p>}
  </section>;
}

function FoldRow({ item, open }) {
  const [shown, setShown] = useState(open);
  return <details className={`v8-card v8-overview-fold ck-fold${item.pro ? ' wb-pro' : ''}`} open={shown} onToggle={event => setShown(event.currentTarget.open)}>
    <summary><span>{item.label}</span>{item.meta != null && <small className="v8-note">{item.meta}</small>}</summary>
    {shown && <div className="ck-fold-body">{item.render()}</div>}
  </details>;
}

/* 细节：线上版的左对齐折叠行，默认收起，点开才渲染。items = [{ id, label, meta, render, pro }] */
export function PillTools({ items, label = '更多', initial = '' }) {
  const shown = items.filter(Boolean);
  if (!shown.length) return null;
  return <div className="ck-folds" role="group" aria-label={label}>
    {shown.map(item => <FoldRow key={item.id} item={item} open={item.id === initial} />)}
  </div>;
}
export const FoldList = PillTools;
