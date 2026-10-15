/* v6 共享小组件：数据源标记、状态胶囊、空态、证据门链、色阶。
   色阶来自 dataviz 参考调色板：顺序量 = 单一蓝色由浅到深；正负量 = 蓝 ↔ 红、灰色中点。
   状态总是「符号 + 文字」，不只靠颜色。 */

import { getCachedUser, getSession } from './api.js';

export const SEQ_STOPS = ['#f3f7fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
const DIV_NEG = [227, 73, 72];   // #e34948
const DIV_MID = [240, 239, 236]; // #f0efec
const DIV_POS = [42, 120, 214];  // #2a78d6

function hexRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function mix(a, b, t) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}
/* v ∈ [0,1] → 顺序蓝色阶 */
export function seqColor(v) {
  const t = Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0)) * (SEQ_STOPS.length - 1);
  const i = Math.min(SEQ_STOPS.length - 2, Math.floor(t));
  const c = mix(hexRgb(SEQ_STOPS[i]), hexRgb(SEQ_STOPS[i + 1]), t - i);
  return `rgb(${c.join(',')})`;
}
/* v ∈ [-1,1] → 红-灰-蓝 */
export function divColor(v) {
  const t = Math.max(-1, Math.min(1, Number.isFinite(v) ? v : 0));
  const c = t >= 0 ? mix(DIV_MID, DIV_POS, t) : mix(DIV_MID, DIV_NEG, -t);
  return `rgb(${c.join(',')})`;
}
/* 色块上文字用深/浅墨色 */
export function inkOn(v, diverging = false) {
  const a = diverging ? Math.abs(v) : v;
  return a > 0.55 ? '#ffffff' : '#1e293b';
}

export function SourceTag({ state, label }) {
  const text = state === 'live' ? 'LIVE' : state === 'loading' ? '读取中' : '未接入';
  return <span className={'v6-src ' + (state || 'none')} title={label ? `${label} 数据源` : undefined}>{label ? `${label} · ` : ''}{text}</span>;
}

const STATUS = {
  pass: ['✓', '通过', 'good'], done: ['✓', '完成', 'good'], ok: ['✓', '一致', 'good'], reference: ['◎', '参考', 'info'],
  partial: ['◐', '部分', 'warn'], warn: ['!', '注意', 'warn'], pending: ['…', '待数据', 'muted'], empty: ['○', '空格', 'muted'],
  fail: ['✗', '未通过', 'bad'], drift: ['✗', '漂移', 'bad'], not_run: ['–', '未运行', 'muted'], na: ['–', '不适用', 'muted'],
  passed: ['✓', '审计通过', 'good'], rejected: ['✗', '已驳回', 'bad'], needs_rerun: ['↻', '需重跑', 'warn'],
  missing: ['?', '缺失', 'warn'], incomplete: ['?', '不完整', 'warn'], unreferenced: ['–', '未登记', 'muted'],
};
export function StatusPill({ status, label, title }) {
  const [icon, text, tone] = STATUS[status] || ['·', status || '—', 'muted'];
  return <span className={'v6-pill ' + tone} title={title}><i aria-hidden="true">{icon}</i>{label || text}</span>;
}

export function Empty({ children, action }) {
  return <div className="v6-empty"><span>{children}</span>{action}</div>;
}

export function Card({ title, extra, children, className = '', sub }) {
  return (
    <section className={'v6-card ' + className}>
      {(title || extra) && (
        <header className="v6-card-hd">
          <div><h4>{title}</h4>{sub && <small>{sub}</small>}</div>
          {extra && <div className="v6-card-extra">{extra}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

/* 证据门链：10 道门，自动门由分布式协议判定，干预/生成门显示「未运行」 */
export function GateChain({ gates, compact = false }) {
  if (!gates?.length) return null;
  return (
    <ol className={'v6-gates' + (compact ? ' compact' : '')} aria-label="证据门链">
      {gates.map((gate, index) => (
        <li key={gate.id} className={'v6-gate ' + (gate.status || 'pending')} title={`${gate.desc || ''}${gate.note ? '\n当前：' + gate.note : ''}`}>
          <span className="n">{index + 1}</span>
          <b>{gate.label}</b>
          {!compact && <StatusPill status={gate.status} />}
          {!compact && gate.note && <small>{gate.note}</small>}
        </li>
      ))}
    </ol>
  );
}

export function Tabs({ tabs, value, onChange, label }) {
  return (
    <div className="v6-tabs" role="tablist" aria-label={label}>
      {tabs.map(tab => (
        <button key={tab.id} type="button" role="tab" aria-selected={value === tab.id}
          className={value === tab.id ? 'on' : ''} onClick={() => onChange(tab.id)}>
          {tab.label}{tab.badge ? <span className="v6-badge">{tab.badge}</span> : null}
          {tab.sub && <small>{tab.sub}</small>}
        </button>
      ))}
    </div>
  );
}

export function AdminTokenField({ token, onChange }) {
  const user = getCachedUser();
  if (token && user?.role === 'admin' && token === getSession()) {
    return (
      <span className="v6-token via-account" title="已登录管理员账号：写操作使用账号会话授权，不需要再填管理令牌">
        <span>管理权限</span><em>管理员账号 {user.display_name || user.username}</em>
      </span>
    );
  }
  return (
    <label className="v6-token" title="写操作（冻结模板、审计裁决）需要中心节点管理令牌（或用管理员账号登录）；只保存在本标签页会话中">
      <span>管理令牌</span>
      <input type="password" autoComplete="off" value={token} placeholder="X-Admin-Token"
        onChange={event => onChange(event.target.value.trim())} />
      {token ? <em>已填写</em> : <em className="muted">未填写</em>}
    </label>
  );
}
