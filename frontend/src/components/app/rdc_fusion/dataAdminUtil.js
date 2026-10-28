/* 数据管理页（DataAdmin.jsx）的标签与格式（纯函数，组件文件只导出组件）。 */

export const SOURCE_LABEL = { result: '模式族结果', replica: '重复上传', census: '普查分片' };

/* 上传状态 → StatusPill 的状态键与文字；符号与文字同时给出 */
const STATE = {
  passed: ['passed', '审计通过'], rejected: ['rejected', '已驳回'], needs_rerun: ['needs_rerun', '需重跑'],
  unaudited: ['pending', '待审计'], accepted: ['pass', '校验通过'], disputed: ['warn', '复算不一致'],
  byte_identical_replicate: ['ok', '逐字节一致'], identical_bytes_different_model: ['warn', '字节相同·模型不同'], replica: ['ok', '副本'],
};
export function uploadState(status) {
  const [key, label] = STATE[status] || ['na', status || '—'];
  return { status: key, label };
}

export function fmtBytes(n) {
  if (n == null) return '—';
  if (n >= 2 ** 40) return `${(n / 2 ** 40).toFixed(2)} TiB`;
  if (n >= 2 ** 30) return `${(n / 2 ** 30).toFixed(2)} GiB`;
  if (n >= 2 ** 20) return `${(n / 2 ** 20).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${n} B`;
}

export function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  const p = v => String(v).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

export function ownerLabel(item) {
  if (!item.owner) return '无主节点';
  return item.owner.display_name && item.owner.display_name !== item.owner.username
    ? `${item.owner.display_name}（${item.owner.username}）` : item.owner.username;
}

/* 分组计数 → [[键, 数量], …]，数量倒序 */
export function entries(obj) {
  return Object.entries(obj || {}).sort((a, b) => b[1] - a[1]);
}

/* 上传列表查询串：只写非空条件 */
export function uploadsQuery(f, offset, limit) {
  const q = new URLSearchParams();
  for (const k of ['source', 'user', 'kind', 'status', 'q']) if (f[k] && f[k] !== 'all') q.set(k, f[k]);
  q.set('limit', String(limit));
  if (offset) q.set('offset', String(offset));
  return q.toString();
}
