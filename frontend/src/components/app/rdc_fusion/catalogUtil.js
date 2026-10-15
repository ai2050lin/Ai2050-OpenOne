/* 题库共享工具（非组件模块，供 CatalogPanel / HomeOverview / AutoTestPanel 使用）。 */
import { API_BASE } from './api.js';

export const STATUS_META = {
  done: ['pass', '已完成'], running: ['partial', '测试中'], open: ['pending', '待测试'],
  pilot: ['warn', '试跑中'], draft: ['empty', '未冻结'],
};
export const STATUS_ORDER = ['done', 'running', 'open', 'pilot', 'draft'];

/* 服务器给出的领题命令里中心地址可能是占位符：换成客户端当前连接的中心 */
export function claimCommand(item) {
  const cmd = item.claim_command || '';
  return cmd.includes('<center>') ? cmd.replace(/--server \S+/, `--server ${API_BASE}`) : cmd;
}

/* 研究包下载地址（模型桶含 | 与中文，需要编码） */
export function packUrl(tmId, bucket) {
  return `${API_BASE}/api/export/${encodeURIComponent(tmId)}?bucket=${encodeURIComponent(bucket || '')}`;
}

/* 下载文件名（与服务器命名一致；只保留 ASCII 安全字符） */
export function safeName(text) {
  return String(text || '').replace('合成·', 'SYN_').replace(/[^A-Za-z0-9._-]+/g, '_').replace(/_+/g, '_').replace(/^_+|_+$/g, '') || 'pack';
}
export function packFile(tmId, bucket) {
  return `respack_${safeName(tmId)}__${safeName(bucket)}.zip`;
}
