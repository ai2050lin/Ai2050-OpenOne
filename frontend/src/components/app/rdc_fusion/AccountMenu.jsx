/* 顶栏账号菜单：当前账号、修改密码、我的节点、退出登录；管理员另有「账号管理」（角色、停用、创建账号）。 */
import { useEffect, useRef, useState } from 'react';
import { authJson, fmtAgoTs } from './api.js';
import { useAuth } from './authContext.js';
import { Empty, StatusPill } from './ui.jsx';

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

function UserAdmin({ me, onClose }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState('');
  const [nonce, setNonce] = useState(0);
  const [nu, setNu] = useState({ username: '', password: '', display_name: '', role: 'member' });
  const [msg, setMsg] = useState(null);
  useEffect(() => {
    let dead = false;
    authJson('/api/auth/users').then(d => { if (!dead) setData(d); }).catch(e => { if (!dead) setErr(e.message); });
    return () => { dead = true; };
  }, [nonce]);
  const update = async (u, body) => {
    setMsg(null);
    try { await authJson(`/api/auth/users/${u.user_id}`, { method: 'POST', body }); setNonce(n => n + 1); }
    catch (e) { setMsg({ ok: false, text: e.message }); }
  };
  const create = async e => {
    e.preventDefault();
    setMsg(null);
    try {
      const r = await authJson('/api/auth/users', { method: 'POST', body: nu });
      setMsg({ ok: true, text: `已创建账号 ${r.user.username}` });
      setNu({ username: '', password: '', display_name: '', role: 'member' });
      setNonce(n => n + 1);
    } catch (e2) { setMsg({ ok: false, text: e2.message }); }
  };
  return (
    <div className="v6-modal-back" role="presentation" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="v6-modal" role="dialog" aria-modal="true" aria-label="账号管理">
        <header><b>账号管理</b>{data && <small>自助注册：{data.registration === 'open' ? '开放' : '关闭（AI2050_REGISTRATION=closed）'}</small>}
          <button type="button" className="fw-tbtn ghost" onClick={onClose} aria-label="关闭">×</button></header>
        {err && <div className="fw-ai-err">{err}</div>}
        {!data && !err && <Empty>读取…</Empty>}
        {data && (
          <table className="v6-mini-table v6-acct-users">
            <thead><tr><th>账号</th><th>角色</th><th>节点</th><th>最近登录</th><th>状态</th><th /></tr></thead>
            <tbody>
              {data.users.map(u => (
                <tr key={u.user_id}>
                  <th scope="row"><b>{u.display_name}</b><small className="mono"> {u.username}</small>{u.user_id === me.user_id && <em className="v6-tag">我</em>}</th>
                  <td>
                    <select value={u.role} onChange={e => update(u, { role: e.target.value })} aria-label={`${u.username} 的角色`}>
                      <option value="member">成员</option><option value="admin">管理员</option>
                    </select>
                  </td>
                  <td className="num">{u.nodes}</td>
                  <td>{u.last_login ? fmtAgoTs(u.last_login) : '—'}</td>
                  <td><StatusPill status={u.disabled ? 'fail' : 'ok'} label={u.disabled ? '已停用' : '正常'} /></td>
                  <td>{u.user_id !== me.user_id && (
                    <button type="button" className="fw-tbtn ghost" onClick={() => update(u, { disabled: !u.disabled })}>{u.disabled ? '启用' : '停用'}</button>
                  )}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <form className="v6-acct-create" onSubmit={create}>
          <b>创建账号</b>
          <input placeholder="用户名" required value={nu.username} onChange={e => setNu(v => ({ ...v, username: e.target.value }))} />
          <input placeholder="显示名称（可选）" value={nu.display_name} onChange={e => setNu(v => ({ ...v, display_name: e.target.value }))} />
          <input type="password" placeholder="初始密码" autoComplete="new-password" required value={nu.password} onChange={e => setNu(v => ({ ...v, password: e.target.value }))} />
          <select value={nu.role} onChange={e => setNu(v => ({ ...v, role: e.target.value }))} aria-label="新账号角色"><option value="member">成员</option><option value="admin">管理员</option></select>
          <button type="submit" className="fw-tbtn primary">创建</button>
        </form>
        {msg && <div className={msg.ok ? 'v6-ok' : 'fw-ai-err'}>{msg.text}</div>}
        <p className="v6-note">管理员可以冻结模板、审计结果、运行自动测试与 AI 解读（账号会话即管理授权）；成员可以浏览、下载研究包、绑定自己的节点参与测试。停用账号会让其所有登录立即失效；系统至少保留一个可用的管理员。</p>
      </div>
    </div>
  );
}

export default function AccountMenu() {
  const { user, offline, guest, logout, enterAuth } = useAuth();
  const [open, setOpen] = useState(false);
  const [panel, setPanel] = useState('');
  const [admin, setAdmin] = useState(false);
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
              {user.role === 'admin' && <button type="button" role="menuitem" className="v6-acct-item" onClick={() => { setAdmin(true); setOpen(false); }}>账号管理</button>}
              <button type="button" role="menuitem" className="v6-acct-item danger" onClick={() => { setOpen(false); logout(); }}>退出登录</button>
            </>
          )}
        </div>
      )}
      {admin && <UserAdmin me={user} onClose={() => setAdmin(false)} />}
    </div>
  );
}
