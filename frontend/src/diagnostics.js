/* 全局错误落盘（client_architecture_review P0-2）：渲染进程的未捕获异常与 Promise 拒绝
   写入 localStorage 环形缓冲，供「连接与运行形态」页查看与导出诊断包。
   设计约束：绝不抛出、绝不阻塞渲染——诊断本身失败时静默放弃。 */

const KEY = 'ai2050-errors';
const MAX = 50;

function load() {
  try { return JSON.parse(window.localStorage.getItem(KEY) || '[]'); } catch { return []; }
}

function save(entries) {
  try { window.localStorage.setItem(KEY, JSON.stringify(entries.slice(-MAX))); } catch { /* 存储不可用则放弃 */ }
}

/* 单条错误入缓冲；返回是否写入。 */
export function recordError(kind, message, detail) {
  try {
    const entry = {
      ts: new Date().toISOString(),
      kind,                                    // 'error' | 'unhandledrejection' | 'manual'
      message: String(message || '').slice(0, 400),
      detail: String(detail || '').slice(0, 1200),
      page: typeof window !== 'undefined' ? window.location.href.slice(0, 200) : '',
    };
    const entries = load();
    entries.push(entry);
    save(entries);
    try { window.dispatchEvent(new CustomEvent('ai2050-errors-updated')); } catch { /* 无监听者 */ }
    return true;
  } catch { return false; }
}

export function getErrors() { try { return load(); } catch { return []; } }
export function clearErrors() { try { window.localStorage.removeItem(KEY); try { window.dispatchEvent(new CustomEvent('ai2050-errors-updated')); } catch { /* 同上 */ } } catch { /* 同上 */ } }
export function errorCount() { return getErrors().length; }

/* 诊断包：环境摘要 + 最近错误，文本形式便于粘贴到 issue。 */
export function exportDiagnostics() {
  const info = {
    exported_at: new Date().toISOString(),
    href: typeof window !== 'undefined' ? window.location.href : '',
    user_agent: typeof navigator !== 'undefined' ? navigator.userAgent : '',
  };
  const lines = [
    'AI2050 客户端诊断包',
    JSON.stringify(info, null, 2),
    `最近错误 ${getErrors().length} 条：`,
    ...getErrors().map(e => `[${e.ts}] ${e.kind}: ${e.message}${e.detail ? `\n    ${e.detail.split('\n').join('\n    ')}` : ''}`),
  ];
  return lines.join('\n');
}

/* 在入口挂一次：捕获未被 ErrorBoundary 接住的异常（事件回调、异步、资源加载）。 */
export function installGlobalErrorCapture() {
  if (typeof window === 'undefined' || window.__ai2050ErrorCapture) return;
  window.__ai2050ErrorCapture = true;
  window.addEventListener('error', event => {
    recordError('error', event.message, event.filename ? `${event.filename}:${event.lineno}:${event.colno}` : (event.error?.stack || ''));
  });
  window.addEventListener('unhandledrejection', event => {
    const r = event.reason;
    if (r && (r.name === 'AbortError' || r.name === 'TimeoutError')) return;   // 请求取消/超时不是故障
    recordError('unhandledrejection', r?.message || String(r), r?.stack || '');
  });
}
