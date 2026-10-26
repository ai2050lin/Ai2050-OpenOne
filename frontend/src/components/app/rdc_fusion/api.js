/* 客户端 API 公共层（v6）：统一 API_BASE、读取（本地数据包优先，不轮询）、写操作与管理令牌。
   离线客户端（config.js 的 OFFLINE_CLIENT）只在线读取站点配置，其余读取一律来自本机数据包与附加包，写操作不可用。
   纪律：读取失败只返回 error，组件显示「未接入」空态，不回退演示数据。 */
import { useCallback, useEffect, useState, useSyncExternalStore } from 'react';

import { getExtraEntry, getLocal, getOnDemand, isOnDemand, packHint, packInfo, putOnDemand, storeTick, subscribe } from './dataStore.js';

import { API_BASE, OFFLINE_CLIENT, WEB_URL } from '../../../config.js';

export { API_BASE, OFFLINE_CLIENT, WEB_URL };

/* 离线客户端唯一在线读取的接口。 */
const ONLINE_PATHS = new Set(['/api/platform/config']);

/* 离线客户端：从本机数据包取一条；没有时抛出带 pack（应下载的附加包 id）的错误。 */
async function offlineJson(path) {
  const local = getLocal(path);
  if (local !== undefined) return local;
  const extra = await getExtraEntry(path);
  if (extra !== undefined) return extra;
  const pack = packHint(path);
  const info = packInfo();
  const error = new Error(!info.version ? '离线客户端：还没有下载数据（页面左下角「下载数据」）'
    : pack && !info.extras[pack] ? '离线客户端：这项数据在附加包里，下载后显示'
      : '离线客户端：已下载的数据包里没有这项数据（可在网页版查看）');
  error.pack = pack && !info.extras[pack] ? pack : null;
  error.offline = true;
  throw error;
}

let configOnce = null;             // 离线客户端：站点配置每次打开页面只读一次，各组件共用

export async function getJson(path, { signal } = {}) {
  if (OFFLINE_CLIENT && !ONLINE_PATHS.has(path.split('?')[0])) return offlineJson(path);
  if (OFFLINE_CLIENT) {
    if (!configOnce) configOnce = fetchJson(path).catch(reason => { configOnce = null; throw reason; });
    return configOnce;
  }
  return fetchJson(path, { signal });
}

async function fetchJson(path, { signal } = {}) {
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
  if (OFFLINE_CLIENT) throw new Error(`离线客户端不能写入；请在网页版操作：${WEB_URL}`);
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

/* 读取：优先本地数据（手动下载的数据包，见 dataStore.js），命中就不访问服务器；
   没命中时在线读取一次。不做定时轮询，需要最新数据时调用 reload()（管理类页面的「刷新」按钮）：
   reload 之后改为在线读取，不再用本地数据包里的旧值。online：管理类页面始终在线读取（冻结、派发后要马上看到变化）。
   实验响应的合并统计与规律分析体积大、不进数据包：第一次在线读取后存进本地，之后直接读本地。
   pollWhilePending：服务端返回 pending（后台计算中）时 15 秒后自动复查，直到出结果。
   离线客户端：只读本机数据包与附加包（getJson 内处理），error 时 missingPack 给出应下载的附加包 id；下载后自动重读。
   返回 {data, error, loading, reload, local, missingPack}；setState 只在异步回调中调用；切换 path 时旧数据不串用。 */
export function useApi(path, { pollWhilePending = false, online = false } = {}) {
  useSyncExternalStore(subscribe, storeTick);              // 数据包载入 / 更新后重新渲染
  const ready = packInfo().ready;
  const local = path ? getLocal(path) : undefined;
  const [state, setState] = useState({ key: null, data: null, error: '', pack: null });
  const [nonce, setNonce] = useState(0);
  const useLocal = (!online || OFFLINE_CLIENT) && (nonce === 0 || OFFLINE_CLIENT) && local !== undefined;
  const needFetch = Boolean(path) && ((online && !OFFLINE_CLIENT) || ready) && !useLocal;
  const extrasKey = OFFLINE_CLIENT && path && !ONLINE_PATHS.has(path.split('?')[0])
    ? JSON.stringify(packInfo().extras) : '';                // 附加包下载后重读（站点配置不受影响）
  useEffect(() => {
    if (!needFetch) return undefined;
    const controller = new AbortController();
    let retry = null;
    let dead = false;
    const run = () => getJson(path, { signal: controller.signal })
      .then(data => {
        if (dead) return;
        setState({ key: path, data, error: '', pack: null });
        if (!OFFLINE_CLIENT && isOnDemand(path)) putOnDemand(path, data);
        // 服务端后台计算中（data.pending）：15 秒后自动复查，直到出结果
        if (pollWhilePending && data?.pending) retry = window.setTimeout(run, 15000);
      })
      .catch(reason => {
        if (reason?.name === 'AbortError' || dead) return;
        setState(previous => ({ key: path, data: previous.key === path ? previous.data : null, error: String(reason.message || reason),
                                pack: reason?.pack || null }));
      });
    if (!OFFLINE_CLIENT && isOnDemand(path) && nonce === 0) {
      getOnDemand(path).then(cached => {
        if (dead) return;
        if (cached !== undefined) setState({ key: path, data: cached, error: '', pack: null });
        else run();
      });
    } else {
      run();
    }
    return () => { dead = true; controller.abort(); if (retry) window.clearTimeout(retry); };
  }, [path, nonce, pollWhilePending, needFetch, extrasKey]);
  const reload = useCallback(() => setNonce(value => value + 1), []);
  if (useLocal) return { data: local, error: '', loading: false, reload, local: true, missingPack: null };
  const current = state.key === path;
  return {
    data: current ? state.data : null,
    error: current ? state.error : '',
    loading: Boolean(path) && !current,
    reload,
    local: false,
    missingPack: current ? state.pack : null,
  };
}

/* 中心后台快照（/api/platform/snapshot/<name>）：合并全部模板、谱图等重计算端点由中心定时算好，公开页面只读快照。
   返回与 useApi 相同的形状；快照尚未算完时 loading 为 true。 */
export function useSnapshot(name) {
  const r = useApi(name ? `/api/platform/snapshot/${name}` : null);
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
  if (OFFLINE_CLIENT) {
    const error = new Error('离线客户端不连接账号服务');
    error.offline = true;
    throw error;
  }
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
