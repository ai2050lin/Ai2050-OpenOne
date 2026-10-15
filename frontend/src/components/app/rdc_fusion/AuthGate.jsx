/* 账号入口（accounts.v1）：客户端第一次打开显示注册界面；登录后令牌保存在本机，再次打开直接进入主界面。
   启动时用 GET /api/auth/status 校验已保存的会话：有效 → 主界面；失效或过期 → 登录界面；
   中心服务暂时不可达但本机有登录记录 → 先进入主界面（各视图显示「未接入」），不把用户挡在门外。 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { API_BASE, authJson, clearSession, getCachedUser, getLastUsername, getSession, setSession } from './api.js';
import { AuthContext } from './authContext.js';

function Field({ label, hint, ...props }) {
  return (
    <label className="v6-auth-field">
      <span>{label}{hint && <small>{hint}</small>}</span>
      <input {...props} />
    </label>
  );
}

function AuthScreen({ info, offlineError, onRetry, onDone }) {
  const canRegister = !info || info.can_register;
  const firstUser = info && !info.has_users;
  const [mode, setMode] = useState(() => (firstUser || (!getLastUsername() && canRegister) ? 'register' : 'login'));
  const [f, setF] = useState({ username: getLastUsername(), password: '', confirm: '', display_name: '', email: '' });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const set = k => e => setF(v => ({ ...v, [k]: e.target.value }));
  const effectiveMode = canRegister ? mode : 'login';

  const submit = async e => {
    e.preventDefault();
    setErr('');
    if (effectiveMode === 'register' && f.password !== f.confirm) { setErr('两次输入的密码不一致'); return; }
    setBusy(true);
    try {
      const body = effectiveMode === 'register'
        ? { username: f.username.trim(), password: f.password, display_name: f.display_name.trim(), email: f.email.trim(), client: 'web' }
        : { username: f.username.trim(), password: f.password, client: 'web' };
      const res = await authJson(`/api/auth/${effectiveMode}`, { method: 'POST', body });
      setSession(res.token, res.user);
      onDone(res.user);
    } catch (reason) {
      setErr(reason.message);
    }
    setBusy(false);
  };

  return (
    <div className="v6-auth">
      <div className="v6-auth-card" role="main">
        <header className="v6-auth-hd">
          <span className="v6-auth-logo" aria-hidden="true">A2</span>
          <div>
            <h1>Ai2050 Atlas</h1>
            <p>分布式 LLM 语言编码研究平台：领题测试、3D 观察、语言谱图挖掘</p>
          </div>
        </header>
        {offlineError && (
          <div className="v6-auth-offline" role="alert">
            <b>连不上中心服务</b>
            <span>{API_BASE} · {offlineError}。确认后端（server.server，默认 :5001）已启动后重试。</span>
            <button type="button" className="fw-tbtn" onClick={onRetry}>重试连接</button>
          </div>
        )}
        {canRegister && (
          <div className="v6-auth-tabs" role="tablist" aria-label="注册或登录">
            <button type="button" role="tab" aria-selected={effectiveMode === 'register'} className={effectiveMode === 'register' ? 'on' : ''} onClick={() => { setMode('register'); setErr(''); }}>注册</button>
            <button type="button" role="tab" aria-selected={effectiveMode === 'login'} className={effectiveMode === 'login' ? 'on' : ''} onClick={() => { setMode('login'); setErr(''); }} disabled={firstUser}>登录</button>
          </div>
        )}
        {firstUser && <p className="v6-auth-note">这是该中心的第一个账号，将自动成为<b>管理员</b>：可以冻结模板、审计结果、运行自动测试、管理其他账号。</p>}
        {info && !canRegister && <p className="v6-auth-note">本中心关闭了自助注册：请用管理员为你创建的账号登录。</p>}
        <form onSubmit={submit} className="v6-auth-form">
          <Field label="用户名" hint={effectiveMode === 'register' ? '3–32 位：字母、数字、_ . -' : ''} autoComplete="username" required
            value={f.username} onChange={set('username')} autoFocus={effectiveMode === 'login' ? undefined : true} />
          {effectiveMode === 'register' && (
            <>
              <Field label="显示名称" hint="可选，用于贡献者署名" value={f.display_name} onChange={set('display_name')} autoComplete="nickname" />
              <Field label="邮箱" hint="可选" type="email" value={f.email} onChange={set('email')} autoComplete="email" />
            </>
          )}
          <Field label="密码" hint={effectiveMode === 'register' ? '至少 8 位，同时包含字母和数字' : ''} type="password" required
            autoComplete={effectiveMode === 'register' ? 'new-password' : 'current-password'} value={f.password} onChange={set('password')}
            autoFocus={effectiveMode === 'login' && Boolean(f.username) ? true : undefined} />
          {effectiveMode === 'register' && (
            <Field label="确认密码" type="password" required autoComplete="new-password" value={f.confirm} onChange={set('confirm')} />
          )}
          {err && <div className="fw-ai-err" role="alert">{err}</div>}
          <button type="submit" className="fw-tbtn primary v6-auth-submit" disabled={busy || Boolean(offlineError)}>
            {busy ? '请稍候…' : effectiveMode === 'register' ? (firstUser ? '创建管理员账号并进入' : '注册并进入') : '登录'}
          </button>
        </form>
        <p className="v6-auth-foot">登录状态保存在本机浏览器，下次打开直接进入；在右上角账号菜单中退出登录。中心：<span className="mono">{API_BASE}</span></p>
      </div>
    </div>
  );
}

export default function AuthGate({ children }) {
  const [state, setState] = useState(() => ({ phase: 'checking', user: getCachedUser(), info: null, offline: '' }));
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let dead = false;
    authJson('/api/auth/status').then(info => {
      if (dead) return;
      if (info.user) {
        setSession(getSession(), info.user);                 // 刷新本机缓存的账号信息（角色可能已变）
        setState({ phase: 'in', user: info.user, info, offline: '' });
      } else {
        clearSession();
        setState({ phase: 'auth', user: null, info, offline: '' });
      }
    }).catch(reason => {
      if (dead) return;
      const cached = getCachedUser();
      if (getSession() && cached) setState({ phase: 'in', user: cached, info: null, offline: reason.message });
      else setState({ phase: 'auth', user: null, info: null, offline: reason.message });
    });
    return () => { dead = true; };
  }, [nonce]);

  const logout = useCallback(async () => {
    try { await authJson('/api/auth/logout', { method: 'POST' }); } catch { /* 离线时只清本机 */ }
    clearSession();
    setState({ phase: 'checking', user: null, info: null, offline: '' });
    setNonce(n => n + 1);
  }, []);
  const refresh = useCallback(() => setNonce(n => n + 1), []);
  const ctx = useMemo(() => ({ user: state.user, offline: Boolean(state.offline), logout, refresh }), [state.user, state.offline, logout, refresh]);

  if (state.phase === 'checking') {
    return <div className="v6-auth v6-auth-checking" aria-busy="true"><span className="v6-auth-logo" aria-hidden="true">A2</span><span>正在连接中心服务…</span></div>;
  }
  if (state.phase === 'auth') {
    return (
      <AuthScreen info={state.info} offlineError={state.offline} onRetry={refresh}
        onDone={user => setState({ phase: 'in', user, info: null, offline: '' })} />
    );
  }
  return <AuthContext.Provider value={ctx}>{children}</AuthContext.Provider>;
}
