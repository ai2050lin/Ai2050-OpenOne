/* 客户端 API 公共层（v6）：统一 API_BASE、轮询读取、写操作与管理令牌。
   纪律：读取失败只返回 error，组件显示「未接入」空态，不回退演示数据。 */
import { useCallback, useEffect, useState } from 'react';

import { API_BASE } from '../../../config.js';

export { API_BASE };

export async function getJson(path, { signal } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { signal, cache: 'no-store' });
  } catch (reason) {
    if (reason?.name === 'AbortError') throw reason;
    throw new Error('中心服务暂时无法连接，请稍后重试');
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(detailText(body.detail) || `HTTP ${response.status}`);
  return body;
}

export async function postJson(path, payload, { adminToken, method = 'POST' } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (adminToken) headers['X-Admin-Token'] = adminToken;
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { method, headers, body: JSON.stringify(payload ?? {}) });
  } catch {
    throw new Error('中心服务暂时无法连接，请稍后重试');
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(detailText(body.detail) || `HTTP ${response.status}`);
    error.status = response.status;
    error.detail = body.detail;
    throw error;
  }
  return body;
}

export function detailText(detail) {
  if (!detail) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail?.errors)) return detail.errors.join('；');
  try { return JSON.stringify(detail); } catch { return String(detail); }
}

/* 读取 + 可选轮询。path 为 null 时不请求。返回 {data, error, loading, reload}。
   setState 只在异步回调中调用（不在 effect 体内同步调用）；切换 path 时旧数据不串用。 */
export function useApi(path, { interval = 0, pollWhilePending = false } = {}) {
  const [state, setState] = useState({ key: null, data: null, error: '' });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (!path) return undefined;
    const controller = new AbortController();
    let retry = null;
    const run = () => getJson(path, { signal: controller.signal })
      .then(data => {
        setState({ key: path, data, error: '' });
        // 服务端后台计算中（data.pending）：15 秒后自动复查，直到出结果
        if (pollWhilePending && data?.pending) retry = window.setTimeout(run, 15000);
      })
      .catch(reason => {
        if (reason?.name === 'AbortError') return;
        setState(previous => ({ key: path, data: previous.key === path ? previous.data : null, error: String(reason.message || reason) }));
      });
    run();
    const timer = interval > 0 ? window.setInterval(run, interval) : null;
    return () => { controller.abort(); if (timer) window.clearInterval(timer); if (retry) window.clearTimeout(retry); };
  }, [path, interval, nonce, pollWhilePending]);
  const reload = useCallback(() => setNonce(value => value + 1), []);
  const current = state.key === path;
  return {
    data: current ? state.data : null,
    error: current ? state.error : '',
    loading: Boolean(path) && !current,
    reload,
  };
}

/* 中心后台快照（/api/platform/snapshot/<name>）：合并全部模板、谱图等重计算端点由中心定时算好，公开页面只读快照。
   返回与 useApi 相同的形状；快照尚未算完时 loading 为 true。 */
export function useSnapshot(name, { interval = 0 } = {}) {
  const r = useApi(name ? `/api/platform/snapshot/${name}` : null, { interval: interval || 0 });
  const pending = Boolean(r.data?.pending);
  return { data: pending ? null : (r.data?.data ?? null), error: r.error || r.data?.error || '', loading: r.loading || pending,
           computedAt: r.data?.computed_at, reload: r.reload };
}

/* 账号会话（accounts.v1）：登录后令牌存在 localStorage，再次打开客户端直接进入主界面；退出登录即清除。
   令牌只随 /api/auth/* 请求以 Authorization 头发送（普通读取不带，避免每个 GET 都触发预检）；
   管理员账号的会话同时可作为写操作的管理令牌。 */
const SESSION_KEY = 'ai2050-session';
const USER_KEY = 'ai2050-user';
const LAST_USER_KEY = 'ai2050-last-username';

function store(op, key, value) {
  try {
    if (op === 'get') return window.localStorage.getItem(key);
    if (op === 'set') window.localStorage.setItem(key, value);
    if (op === 'del') window.localStorage.removeItem(key);
  } catch { /* 隐私模式或存储被禁用：会话只在本次打开内有效 */ }
  return null;
}
export function getSession() { return store('get', SESSION_KEY) || ''; }
export function getCachedUser() {
  try { return JSON.parse(store('get', USER_KEY) || 'null'); } catch { return null; }
}
export function getLastUsername() { return store('get', LAST_USER_KEY) || ''; }
export function setSession(token, user) {
  if (token) store('set', SESSION_KEY, token);
  if (user) {
    store('set', USER_KEY, JSON.stringify(user));
    store('set', LAST_USER_KEY, user.username);
  }
}
export function clearSession() { store('del', SESSION_KEY); store('del', USER_KEY); }

/* /api/auth/* 调用：自动带会话令牌 */
export async function authJson(path, { method = 'GET', body } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  const token = getSession();
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { method, headers, cache: 'no-store', body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    const error = new Error('中心服务暂时无法连接，请稍后重试');
    error.offline = true;
    throw error;
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(detailText(data.detail) || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

/* 管理令牌：手填的令牌只存在本标签页 sessionStorage（不写 cookie / localStorage，不随 URL 分享）；
   没有手填且当前登录的是管理员账号时，用账号会话代替（服务器接受管理员会话作为管理令牌）。 */
const TOKEN_KEY = 'ai2050-admin-token';
export function useAdminToken() {
  const [token, setTokenState] = useState(() => {
    try { return window.sessionStorage.getItem(TOKEN_KEY) || ''; } catch { return ''; }
  });
  const setToken = useCallback(value => {
    setTokenState(value);
    try {
      if (value) window.sessionStorage.setItem(TOKEN_KEY, value);
      else window.sessionStorage.removeItem(TOKEN_KEY);
    } catch { /* 隐私模式 */ }
  }, []);
  const user = getCachedUser();
  const effective = token || (user?.role === 'admin' ? getSession() : '');
  return [effective, setToken];
}

export function fmtNum(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (typeof value !== 'number') return String(value);
  if (Math.abs(value) >= 1000) return value.toLocaleString();
  return value.toFixed(digits);
}

export function fmtAgoTs(ts) {
  if (!ts) return '—';
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return '刚刚';
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
  return `${Math.floor(diff / 86400)} 天前`;
}
