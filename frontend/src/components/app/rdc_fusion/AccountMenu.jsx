/* 顶栏账号菜单：当前账号、修改密码、我的节点、退出登录；管理员另有「账号管理」（网站 ?page=accounts，AccountAdmin.jsx）与「数据管理」（?page=data，DataAdmin.jsx）。 */
import { useEffect, useRef, useState } from 'react';
import { authJson } from './api.js';
import { useAuth } from './authContext.js';
import { StatusPill } from './ui.jsx';
import { siteHref } from '../../v8/v8Util.js';
import './accounts.css';

function PasswordForm({ onDone }) {
  const [f, setF] = useState({ old: '', next: '', confirm: '' });
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);
  const submit = async e => {
    e.preventDefault();
    if (f.next !== f.confirm) { setMsg({ ok: false, text: '两次输入的新密码不一致' }); return; }
    setBusy(true);
    try {
      const r = await authJson('/api/auth/password', { method: 'POST', body: { old_password: f.old, new_password: f.next } });
      setMsg({ ok: true, text: r.note });
      setF({ old: '', next: '', confirm: '' });
      window.setTimeout(onDone, 1500);
    } catch (reason) { setMsg({ ok: false, text: reason.message }); }
    setBusy(false);
  };
  return (
    <form className="v6-acct-form" onSubmit={submit}>
      <input type="password" placeholder="原密码" autoComplete="current-password" required value={f.old} onChange={e => setF(v => ({ ...v, old: e.target.value }))} />
      <input type="password" placeholder="新密码（≥ 8 位，含字母和数字）" autoComplete="new-password" required value={f.next} onChange={e => setF(v => ({ ...v, next: e.target.value }))} />
      <input type="password" placeholder="确认新密码" autoComplete="new-password" required value={f.confirm} onChange={e => setF(v => ({ ...v, confirm: e.target.value }))} />
      <button type="submit" className="fw-tbtn primary" disabled={busy}>{busy ? '保存中…' : '保存新密码'}</button>
      {msg && <div className={msg.ok ? 'v6-ok' : 'fw-ai-err'}>{msg.text}</div>}
    </form>
  );
}

function MyNodes() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState('');
  useEffect(() => {
    let dead = false;
    authJson('/api/auth/my-nodes').then(d => { if (!dead) setData(d); }).catch(e => { if (!dead) setErr(e.message); });
    return () => { dead = true; };
  }, []);
  if (err) return <div className="fw-ai-err">{err}</div>;
  if (!data) return <small className="muted">读取…</small>;
  if (!data.nodes.length) return <small className="muted">还没有绑定节点：在 总览 / 题库「参与测试」生成绑定码后注册节点。</small>;
  return (
    <ul className="v6-acct-nodes">
      {data.nodes.map(n => (
        <li key={n.node_id}><StatusPill status={n.online ? 'ok' : 'na'} label={n.online ? '在线' : '离线'} />
          <span><b>{n.name}</b><small>{n.node_id}{n.gpu ? ` · ${n.gpu}` : ''}</small></span><span className="muted">{n.results} 份</span></li>
      ))}
    </ul>
  );
}

/* 账号菜单：只读数据包模式下同样可用（登录、注销与账号操作都走 /api/auth/*，只在点击时连接服务器）。 */
export default function AccountMenu() {
  return <OnlineAccountMenu />;
}

function OnlineAccountMenu() {
  const { user, offline, guest, logout, enterAuth } = useAuth();
  const [open, setOpen] = useState(false);
  const [panel, setPanel] = useState('');
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const onDoc = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = e => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey); };
  }, [open]);
  if (!user) return null;
  const name = user.display_name || user.username;
  const roleLabel = guest ? '访客' : (user.role === 'admin' ? '管理员' : '成员');
  const roleStatus = user.role === 'admin' ? 'pass' : 'na';
  return (
    <div className="v6-acct" ref={ref}>
      <button type="button" className="v6-acct-btn" aria-haspopup="menu" aria-expanded={open} onClick={() => { setOpen(v => !v); setPanel(''); }}
        title={`${name}（${roleLabel}）${offline ? ' · 中心服务暂不可达' : ''}`}>
        <span className="v6-acct-avatar" aria-hidden="true">{name.slice(0, 1).toUpperCase()}</span>
        <span className="v6-acct-name">{name}</span>
        {offline && <span className="v6-acct-off" aria-label="离线">!</span>}
      </button>
      {open && (
        <div className="v6-acct-menu" role="menu">
          <div className="v6-acct-head">
            <b>{name}</b><small className="mono">{user.username}</small>
            <StatusPill status={roleStatus} label={roleLabel} />
            {offline && <small className="warn">中心服务暂不可达：显示的是本机保存的登录信息</small>}
          </div>
          {guest ? (
            <>
              <button type="button" role="menuitem" className="v6-acct-item" onClick={() => { setOpen(false); enterAuth(); }}>登录 / 注册</button>
              <small className="v6-acct-panel muted">访客模式：可直接浏览全部内容；登录后可领题测试、绑定节点、署名贡献。</small>
            </>
          ) : (
            <>
              <button type="button" role="menuitem" className="v6-acct-item" onClick={() => setPanel(panel === 'nodes' ? '' : 'nodes')}>我的节点</button>
              {panel === 'nodes' && <div className="v6-acct-panel"><MyNodes /></div>}
              <button type="button" role="menuitem" className="v6-acct-item" onClick={() => setPanel(panel === 'pw' ? '' : 'pw')}>修改密码</button>
              {panel === 'pw' && <div className="v6-acct-panel"><PasswordForm onDone={() => setPanel('')} /></div>}
              {user.role === 'admin' && <a role="menuitem" className="v6-acct-item" href={siteHref('accounts')} onClick={() => setOpen(false)}>账号管理</a>}
              {user.role === 'admin' && <a role="menuitem" className="v6-acct-item" href={siteHref('data')} onClick={() => setOpen(false)}>数据管理</a>}
              <button type="button" role="menuitem" className="v6-acct-item danger" onClick={() => { setOpen(false); logout(); }}>退出登录</button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
