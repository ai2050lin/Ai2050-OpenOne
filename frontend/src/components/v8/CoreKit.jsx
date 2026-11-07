/* 精简版页面的公用部件（Phase 333，AGENTS.md 第 10 节“设计风格”）：
   每页首屏只放一个可视化的核心模块；其余功能收进 PillTools，默认收起，点一个只展开一项，再点同一个收起。 */
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

/* 条形关卡：n / d */
export function GateBar({ name, n, d, color = '#183f77', onClick }) {
  const w = d > 0 ? (n / d) * 100 : 0;
  const body = <><span>{name}</span><i><b style={{ width: `${w}%`, background: color }} /></i><span className="v8-note">{n} / {d}</span></>;
  return onClick ? <button type="button" className="ck-gate" onClick={onClick}>{body}</button> : <div className="ck-gate">{body}</div>;
}

/* 核心模块的外框 */
export function CoreCard({ question, children, hint, label }) {
  return <section className="ck-core" aria-label={label || question}>
    {question && <p className="ck-q">{question}</p>}
    {children}
    {hint && <p className="ck-hint">{hint}</p>}
  </section>;
}

/* 收起的操作台：items = [{ id, label, render, pro }]；pro 只在专业模式显示（沿用 wb-pro） */
export function PillTools({ items, label = '更多', initial = '' }) {
  const [open, setOpen] = useState(initial);
  const shown = items.filter(Boolean);
  if (!shown.length) return null;
  const cur = shown.find(item => item.id === open);
  return <>
    <nav className="ck-pills" aria-label={label}>{shown.map(item =>
      <button type="button" key={item.id} className={item.pro ? 'wb-pro' : undefined} aria-pressed={open === item.id} aria-expanded={open === item.id}
        onClick={() => setOpen(o => (o === item.id ? '' : item.id))}>{item.label}</button>)}</nav>
    {cur && <section className="v8-card ck-panel" aria-label={cur.label}>{cur.render()}</section>}
  </>;
}
