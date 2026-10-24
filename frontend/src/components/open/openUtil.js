/* 开放平台外壳（OpenPlatform）的共享函数（非组件）。数据全部来自 /api/platform/*、/api/census/*、/api/auth/*。 */
import { API_BASE, detailText, getSession } from '../app/rdc_fusion/api.js';

export const TABS = [
  { id: 'home', label: '首页', desc: '研究进展' },
  { id: 'learn', label: '学习', desc: '方法与概念' },
  { id: 'atlas', label: '图谱', desc: '语言谱图 · 神经元卡' },
  { id: 'join', label: '参与', desc: '用你的 GPU 跑一个分片' },
];

export function readTab() {
  try {
    const t = new URLSearchParams(window.location.search).get('tab');
    return TABS.some(x => x.id === t) ? t : 'home';
  } catch {
    return 'home';
  }
}

export function writeTab(tab) {
  try {
    const url = new URL(window.location.href);
    url.searchParams.set('tab', tab);
    window.history.replaceState(null, '', url);
  } catch { /* 地址栏不可写时只保留页面状态 */ }
}

export function fmtInt(v) {
  return v == null ? '—' : Number(v).toLocaleString();
}

export function fmtTokens(v) {
  if (v == null) return '—';
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
  return String(v);
}

export function fmtAgo(ts) {
  if (!ts) return '—';
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return '刚刚';
  if (s < 3600) return `${Math.round(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.round(s / 3600)} 小时前`;
  return `${Math.round(s / 86400)} 天前`;
}

/* 普查作业分片状态 → 显示（顺序：完成在前） */
export const SHARD_STATES = [
  { id: 'done', label: '完成', color: '#256abf' },
  { id: 'replica_pending', label: '等复算', color: '#9ec5f4' },
  { id: 'running', label: '运行中', color: '#e5b29b' },
  { id: 'disputed', label: '复算不一致', color: '#c2410c' },
  { id: 'queued', label: '待领取', color: '#e2e8f0' },
];

/* 下载参与者一键包：登录时走 /api/auth/kit（会话令牌只在请求头里，内嵌一次性绑定码），否则匿名包。 */
export async function downloadKit(loggedIn) {
  const headers = {};
  const token = getSession();
  const path = loggedIn && token ? '/api/auth/kit' : '/api/platform/kit';
  if (loggedIn && token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(`${API_BASE}${path}`, { headers, cache: 'no-store' });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(detailText(body.detail) || `HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'ai2050-node.zip';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
