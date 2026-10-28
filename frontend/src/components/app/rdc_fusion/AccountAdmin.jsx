/* 账号管理页（网站 ?page=accounts，管理员）：搜索与筛选、注册策略、创建账号；逐个账号改角色、资料、重置密码、强制下线、停用 / 启用、删除；
   自己的账号可修改密码（要原密码，其他设备上的登录随之失效）。
   全部操作走 /api/auth/*（只在点击时连接中心），网页与桌面客户端相同；账号只用于署名与授权，不影响结果的合并口径与证据等级。 */
import { useEffect, useState } from 'react';
import { authJson, fmtAgoTs } from './api.js';
import { useAuth } from './authContext.js';
import { ROLE_LABEL, countBy, filterUsers, userState } from './accountsUtil.js';
import { Empty, StatusPill } from './ui.jsx';
import './accounts.css';

const EMPTY_NEW = { username: '', password: '', display_name: '', email: '', role: 'member' };

function RowPanel({ u, mode, onDone, onCancel }) {
  const [f, setF] = useState({ display_name: u.display_name || '', email: u.email || '', old: '', password: '', confirm: '', del: '' });
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const run = async (path, body) => {
    setBusy(true); setErr('');
    try { const r = await authJson(path, { method: 'POST', body }); onDone(r); }
    catch (e) { setErr(e.message); setBusy(false); }
  };
  const submit = e => {
    e.preventDefault();
    if (mode === 'edit') run(`/api/auth/users/${u.user_id}`, { display_name: f.display_name, email: f.email });
    else if (mode === 'password' || mode === 'own-password') {
      if (f.password !== f.confirm) { setErr('两次输入的新密码不一致'); return; }
      if (mode === 'own-password') run('/api/auth/password', { old_password: f.old, new_password: f.password });
      else run(`/api/auth/users/${u.user_id}/password`, { new_password: f.password });
    } else if (mode === 'delete') run(`/api/auth/users/${u.user_id}/delete`, { confirm: f.del });
  };
  const field = (k, props) => <input {...props} value={f[k]} onChange={e => setF(v => ({ ...v, [k]: e.target.value }))} />;
  return (
    <form className={'acm-panel' + (mode === 'delete' ? ' danger' : '')} onSubmit={submit}>
      {mode === 'edit' && (<>
        <label>显示名称{field('display_name', { maxLength: 60 })}</label>
        <label>邮箱{field('email', { type: 'email', maxLength: 120, placeholder: '可留空' })}</label>
      </>)}
      {mode === 'own-password' && <label>原密码{field('old', { type: 'password', autoComplete: 'current-password', required: true })}</label>}
      {(mode === 'password' || mode === 'own-password') && (<>
        <label>新密码{field('password', { type: 'password', autoComplete: 'new-password', required: true, minLength: 8 })}</label>
        <label>确认新密码{field('confirm', { type: 'password', autoComplete: 'new-password', required: true, minLength: 8 })}</label>
        <small>{mode === 'own-password' ? '修改后，你在其他设备和浏览器上的登录都会失效；当前这个页面保持登录。'
          : '重置后该账号的全部登录立即失效，登录锁定一并解除；请通过其他渠道把新密码告诉对方。'}</small>
      </>)}
      {mode === 'delete' && (<>
        <small>永久删除 <b>{u.username}</b>：账号、登录与绑定码都会删除，不能恢复；名下 {u.nodes} 个节点改为无主（已上传的结果保留，去掉署名）。只想暂时禁止登录请用「停用」。</small>
        <label>输入用户名确认{field('del', { required: true, autoComplete: 'off', placeholder: u.username })}</label>
      </>)}
      <div className="acm-panel-act">
        <button type="submit" className={'fw-tbtn ' + (mode === 'delete' ? 'acm-danger' : 'primary')}
                disabled={busy || (mode === 'delete' && f.del.trim().toLowerCase() !== u.username.toLowerCase())}>
          {busy ? '处理中…' : { edit: '保存', password: '重置密码', 'own-password': '修改密码', delete: '永久删除' }[mode]}
        </button>
        <button type="button" className="fw-tbtn" onClick={onCancel}>取消</button>
        {err && <span className="acm-err" role="alert">{err}</span>}
      </div>
    </form>
  );
}

function CreateForm({ onDone }) {
  const [nu, setNu] = useState(EMPTY_NEW);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async e => {
    e.preventDefault();
    setBusy(true); setErr('');
    try {
      const r = await authJson('/api/auth/users', { method: 'POST', body: nu });
      setNu(EMPTY_NEW);
      onDone(`已创建账号 ${r.user.username}（${ROLE_LABEL[r.user.role]}）`);
    } catch (e2) { setErr(e2.message); }
    setBusy(false);
  };
  const field = (k, props) => <input {...props} value={nu[k]} onChange={e => setNu(v => ({ ...v, [k]: e.target.value }))} />;
  return (
    <form className="acm-create" onSubmit={submit}>
      <label>用户名{field('username', { required: true, autoComplete: 'off', placeholder: '3–32 位字母、数字、_ . -' })}</label>
      <label>显示名称{field('display_name', { placeholder: '可选' })}</label>
      <label>邮箱{field('email', { type: 'email', placeholder: '可选' })}</label>
      <label>初始密码{field('password', { type: 'password', required: true, minLength: 8, autoComplete: 'new-password' })}</label>
      <label>角色
        <select value={nu.role} onChange={e => setNu(v => ({ ...v, role: e.target.value }))}>
          <option value="member">成员</option><option value="admin">管理员</option>
        </select>
      </label>
      <button type="submit" className="fw-tbtn primary" disabled={busy}>{busy ? '创建中…' : '创建账号'}</button>
      {err && <span className="acm-err" role="alert">{err}</span>}
    </form>
  );
}

export default function AccountAdmin() {
  const { user, guest, enterAuth } = useAuth();
  const isAdmin = !!user && !guest && user.role === 'admin';
  const [data, setData] = useState(null);
  const [err, setErr] = useState('');
  const [nonce, setNonce] = useState(0);
  const [filter, setFilter] = useState({ q: '', role: '', state: '' });
  const [open, setOpen] = useState({ id: '', mode: '' });
  const [msg, setMsg] = useState(null);
  const [showCreate, setShowCreate] = useState(false);

  useEffect(() => {
    if (!isAdmin) return undefined;
    let dead = false;
    authJson('/api/auth/users').then(d => { if (!dead) { setData(d); setErr(''); } }).catch(e => { if (!dead) setErr(e.message); });
    return () => { dead = true; };
  }, [nonce, isAdmin]);

  if (!isAdmin) {
    return (<Empty action={(!user || guest) && <button type="button" className="fw-tbtn primary" onClick={enterAuth}>登录</button>}>
      {user && !guest ? '账号管理只对管理员开放；当前账号是成员。' : '账号管理需要管理员账号登录。'}
    </Empty>);
  }

  const reload = text => { setOpen({ id: '', mode: '' }); if (text) setMsg({ ok: true, text }); setNonce(n => n + 1); };
  const act = async (path, body, text) => {
    setMsg(null);
    try { const r = await authJson(path, { method: 'POST', body }); reload(typeof text === 'function' ? text(r) : text); }
    catch (e) { setMsg({ ok: false, text: e.message }); }
  };
  const users = data?.users || [];
  const shown = filterUsers(users, filter);
  const n = countBy(users);
  const toggle = (u, mode) => { setMsg(null); setOpen(o => (o.id === u.user_id && o.mode === mode ? { id: '', mode: '' } : { id: u.user_id, mode })); };

  return (
    <div className="acm">
      <header className="acm-head">
        <div>
          <h1>账号管理</h1>
          <p>管理本中心的全部账号。管理员可以冻结模板、审计结果和管理账号；成员可以浏览、绑定自己的节点并署名贡献。系统至少保留一个可用的管理员。</p>
        </div>
        <button type="button" className="fw-tbtn" onClick={() => reload()}>刷新</button>
      </header>

      {err && <div className="acm-err block" role="alert">{err}</div>}
      {!data && !err && <Empty>读取账号…</Empty>}

      {data && (<>
        <section className="acm-bar" aria-label="概况与注册策略">
          <dl className="acm-counts">
            <div><dt>账号</dt><dd>{n.total}</dd></div>
            <div><dt>管理员</dt><dd>{n.admin}</dd></div>
            <div><dt>已停用</dt><dd>{n.disabled}</dd></div>
            <div><dt>锁定中</dt><dd>{n.locked}</dd></div>
          </dl>
          <div className="acm-reg">
            <span>自助注册</span>
            <div className="acm-seg" role="group" aria-label="自助注册">
              {[['open', '开放'], ['closed', '关闭']].map(([k, label]) => (
                <button key={k} type="button" aria-pressed={data.registration === k} className={data.registration === k ? 'on' : ''}
                        onClick={() => data.registration !== k && act('/api/auth/registration', { policy: k }, `自助注册已${label}`)}>{label}</button>
              ))}
            </div>
            <small>{data.registration_source === 'db' ? '在本页设置，优先于服务器环境变量' : '来自服务器环境变量 AI2050_REGISTRATION（缺省开放）'}；关闭后只能由管理员创建账号。</small>
          </div>
        </section>

        <section className="acm-tools" aria-label="筛选">
          <input type="search" placeholder="搜索用户名、显示名称、邮箱" value={filter.q} aria-label="搜索账号"
                 onChange={e => setFilter(f => ({ ...f, q: e.target.value }))} />
          <select value={filter.role} aria-label="按角色筛选" onChange={e => setFilter(f => ({ ...f, role: e.target.value }))}>
            <option value="">全部角色</option><option value="admin">管理员</option><option value="member">成员</option>
          </select>
          <select value={filter.state} aria-label="按状态筛选" onChange={e => setFilter(f => ({ ...f, state: e.target.value }))}>
            <option value="">全部状态</option><option value="active">正常</option><option value="locked">锁定中</option><option value="disabled">已停用</option>
          </select>
          <span className="acm-shown">显示 {shown.length} / {users.length}</span>
          <button type="button" className={'fw-tbtn' + (showCreate ? ' on' : ' primary')} onClick={() => setShowCreate(v => !v)}>{showCreate ? '收起' : '＋ 创建账号'}</button>
        </section>
        {showCreate && <CreateForm onDone={text => { setShowCreate(false); reload(text); }} />}
        {msg && <div className={msg.ok ? 'acm-ok' : 'acm-err block'} role={msg.ok ? 'status' : 'alert'}>{msg.text}</div>}

        <div className="acm-table-wrap">
          <table className="acm-table">
            <thead><tr><th>账号</th><th>角色</th><th>状态</th><th className="num">节点</th><th className="num">登录</th><th>最近登录</th><th>注册于</th><th>操作</th></tr></thead>
            <tbody>
              {shown.map(u => {
                const me = u.user_id === user.user_id;
                const st = userState(u);
                const isOpen = open.id === u.user_id;
                return [
                  <tr key={u.user_id} className={isOpen ? 'open' : ''}>
                    <th scope="row">
                      <b>{u.display_name}</b>{me && <em className="acm-me">我</em>}
                      <small className="mono">{u.username}{u.email ? ` · ${u.email}` : ''}</small>
                    </th>
                    <td>
                      <select value={u.role} aria-label={`${u.username} 的角色`}
                              onChange={e => act(`/api/auth/users/${u.user_id}`, { role: e.target.value }, `${u.username} 已设为${ROLE_LABEL[e.target.value]}`)}>
                        <option value="member">成员</option><option value="admin">管理员</option>
                      </select>
                    </td>
                    <td><StatusPill status={st.status} label={st.label} title={st.key === 'locked' ? '连续输错密码，5 分钟后自动解除；重置密码也会解除' : undefined} /></td>
                    <td className="num">{u.nodes}</td>
                    <td className="num" title="未过期的登录会话数">{u.sessions}</td>
                    <td>{u.last_login ? fmtAgoTs(u.last_login) : '—'}</td>
                    <td>{u.created_at ? new Date(u.created_at * 1000).toLocaleDateString() : '—'}</td>
                    <td className="acm-act">
                      <button type="button" className={'fw-tbtn' + (isOpen && open.mode === 'edit' ? ' on' : '')} onClick={() => toggle(u, 'edit')}>资料</button>
                      {me && <button type="button" className={'fw-tbtn' + (isOpen && open.mode === 'own-password' ? ' on' : '')} onClick={() => toggle(u, 'own-password')}>修改密码</button>}
                      {!me && (<>
                        <button type="button" className={'fw-tbtn' + (isOpen && open.mode === 'password' ? ' on' : '')} onClick={() => toggle(u, 'password')}>重置密码</button>
                        <button type="button" className="fw-tbtn" disabled={!u.sessions} title={u.sessions ? '注销该账号的全部登录' : '没有登录中的会话'}
                                onClick={() => act(`/api/auth/users/${u.user_id}/logout`, undefined, r => `${u.username} 已强制下线（${r.revoked} 个登录）`)}>强制下线</button>
                        <button type="button" className="fw-tbtn"
                                onClick={() => act(`/api/auth/users/${u.user_id}`, { disabled: !u.disabled }, `${u.username} 已${u.disabled ? '启用' : '停用'}`)}>{u.disabled ? '启用' : '停用'}</button>
                        <button type="button" className={'fw-tbtn acm-danger-ghost' + (isOpen && open.mode === 'delete' ? ' on' : '')} onClick={() => toggle(u, 'delete')}>删除</button>
                      </>)}
                    </td>
                  </tr>,
                  isOpen && (
                    <tr key={u.user_id + ':panel'} className="acm-panel-row">
                      <td colSpan={8}>
                        <RowPanel u={u} mode={open.mode} onCancel={() => setOpen({ id: '', mode: '' })}
                                  onDone={r => reload(open.mode === 'delete' ? `已删除 ${r.deleted}${r.nodes_unowned ? `，${r.nodes_unowned} 个节点改为无主` : ''}`
                                    : open.mode === 'password' ? `${u.username} 的密码已重置，其登录已全部失效`
                                    : open.mode === 'own-password' ? '你的密码已修改；其他设备上的登录已失效' : `${u.username} 的资料已保存`)} />
                      </td>
                    </tr>
                  ),
                ];
              })}
            </tbody>
          </table>
          {!shown.length && <Empty>没有符合条件的账号</Empty>}
        </div>
        <p className="acm-note">自己的账号可在本页「修改密码」，退出登录在右上角菜单。忘记管理员密码时，在中心服务器上运行 <code>deploy/reset_password.py</code>。</p>
      </>)}
    </div>
  );
}
