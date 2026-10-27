/* 运行形态与 API 基址统一出口（v1 · 2026-10-14）
   ────────────────────────────────────────────────────────────────
   纪律：本文件是 API 基址的**唯一真源**。业务代码一律从这里 import，
   不得再出现字面量 'http://localhost:5001' 或裸 127.0.0.1。

   三种运行形态（优先级从高到低）：
     ① Electron 安装包：preload 在页面脚本之前把 { apiBase, mode } 经
        contextBridge.exposeInMainWorld 复制进主世界的 window.__AI2050__
        （服务器地址由用户首启填写，持久化到 userData/config.json；
          经 webPreferences.additionalArguments 同步传入 preload，时序早于本模块求值）
        —— 注意：contextIsolation 下 preload 直接赋值 window.__AI2050__ 只落在隔离世界，
        主世界读不到，必须走 contextBridge（见 electron/preload.js 注释）。
     ② nginx 同源托管（CentOS 部署）：基址为空字符串 '',
        fetch('/api/...') 走相对路径，无 CORS、无硬编码域名
     ③ vite dev：读 .env.development 的 VITE_API_BASE（默认本机 5001）

   取值一律去掉末尾斜杠，避免 `${base}/x` 产生双斜杠。 */

function readRuntime(key) {
  if (typeof window === 'undefined') return '';
  const bag = window.__AI2050__;
  return bag && typeof bag[key] === 'string' ? bag[key].trim() : '';
}

function stripTrailingSlash(value) {
  return String(value || '').replace(/\/+$/, '');
}

/** 运行形态标记（供界面显示口径，不参与数据真伪判定）。 */
function detectMode(base) {
  const runtime = readRuntime('mode');
  if (runtime) return runtime;
  if (isDesktop()) return 'electron';
  if (base === '') return 'same-origin';
  return base.includes('localhost') || base.includes('127.0.0.1') ? 'local' : 'remote';
}

/** 是否 Electron 安装包形态（主世界标记，由 preload 经 contextBridge 注入）。 */
function isDesktop() {
  return typeof window !== 'undefined' && Boolean(window.ai2050Desktop && window.ai2050Desktop.isDesktop);
}

/**
 * 中心服务（:5001）基址。
 * - Electron：preload 注入的服务器地址（必填，不可退到 localhost）
 * - nginx 同源托管：空串 → fetch('/api/...') 相对路径，无 CORS、无硬编码域名
 * - vite dev：.env.development 的 VITE_API_BASE
 *
 * 纪律：Electron 形态下注入缺失（preload 未跑/被绕过）时**不退到 localhost**——
 * 那会让客户端静默请求用户本机的 5001 端口，装在别人机器上必然连不通。
 * 此时返回空串（同源语义），请求打不上会明确失败并由界面显示「未接入」空态。
 */
/* 生产构建（vite build）缺省同源：中心服务（deploy/center_app.py 或 nginx）同时托管客户端，
   fetch('/api/...') 走相对路径；需要指向别的中心时在构建前设置 VITE_API_BASE。开发模式缺省本机 5001。 */
const ENV_API_BASE = import.meta.env.VITE_API_BASE;
const API_BASE_RAW = isDesktop()
  ? stripTrailingSlash(readRuntime('apiBase'))
  : stripTrailingSlash(
      readRuntime('apiBase') || ENV_API_BASE || (import.meta.env.PROD ? '' : 'http://localhost:5001')
    );

export const API_BASE = stripTrailingSlash(API_BASE_RAW);

/** 分析服务；缺省与中心服务同源（CentOS 部署只跑一个进程）。 */
export const ANALYSIS_API_BASE = stripTrailingSlash(
  readRuntime('analysisApiBase') || import.meta.env.VITE_ANALYSIS_API_BASE || API_BASE
);

/** 训练服务（:8000，独立进程）。 */
export const TRAINING_API_BASE = stripTrailingSlash(
  readRuntime('trainingApiBase') || import.meta.env.VITE_TRAINING_API_BASE || 'http://localhost:8000'
);

/* RDC 分片研究服务（:5002 update / :5003 query / :5004 construction）。
   独立开发进程，CentOS 部署默认不运行 → 空串走同源，nginx 未反代时按 404 诚实失败，
   界面显示「未接入」空态，不伪造数据。 */
export const RDC_UPDATE_API = stripTrailingSlash(
  readRuntime('rdcUpdateApi') || import.meta.env.VITE_RDC_UPDATE_API || ''
);
export const RDC_QUERY_API = stripTrailingSlash(
  readRuntime('rdcQueryApi') || import.meta.env.VITE_RDC_QUERY_API || ''
);
export const RDC_CONSTRUCTION_API = stripTrailingSlash(
  readRuntime('rdcConstructionApi') || import.meta.env.VITE_RDC_CONSTRUCTION_API || ''
);

/** 前端自身地址（用于拼下载链接）。同源托管时用 location.origin。 */
export const FRONTEND_BASE = stripTrailingSlash(
  readRuntime('frontendBase') ||
  import.meta.env.VITE_FRONTEND_BASE ||
  (typeof window !== 'undefined' && window.location ? window.location.origin : '')
);

/** 运行形态标记，供界面显示「本机 / 服务器」等口径（不参与数据判定）。 */
export const RUNTIME_MODE = detectMode(API_BASE);

/** 只读数据包模式（OFFLINE_CLIENT）：使用中不实时访问服务器。只在这些操作时连接服务器——登录 / 注销 / 账号（/api/auth/*）、
 *  读取站点配置、下载主数据包与附加包、下载参与者一键包；其余读取一律来自本机数据包。
 *  对所有形态生效（网站与研究工作台，网页版、远端 vite dev、Electron 安装包），只有连接本机后端的开发保持在线。
 *  VITE_CLIENT_MODE=online|offline 可显式指定。 */
const CLIENT_MODE = import.meta.env.VITE_CLIENT_MODE;
export const ADMIN_PAGE = typeof window !== 'undefined' && Boolean(window.location)
  && /^\/(workbench|rdc-fusion)(\/|$)/.test(window.location.pathname);
export const OFFLINE_CLIENT = CLIENT_MODE === 'offline' || (CLIENT_MODE !== 'online' && RUNTIME_MODE !== 'local');
/** 连接的是另一台主机上的中心（远端客户端 / 安装包），而不是同源托管的网页：账号菜单里给出「网页版」链接 */
export const REMOTE_CENTER = RUNTIME_MODE === 'remote' || RUNTIME_MODE === 'electron';
/** 管理工位的实时连接：只在中心同源托管的研究工作台里，研发工位（题库之外的模板、冻结、分发、审计、普查）
 *  显式标记 online 的读取与写操作才连接服务器，且只在打开该工位、点击操作时发生。 */
export const ADMIN_LIVE = !OFFLINE_CLIENT || (ADMIN_PAGE && !REMOTE_CENTER && CLIENT_MODE !== 'offline');

/** 网页版地址（离线客户端里「去网页版操作」的链接）。 */
export const WEB_URL = API_BASE || (typeof window !== 'undefined' && window.location ? window.location.origin : '');

/** 把相对路径拼成完整 URL（下载链接用；同源托管下返回站点绝对地址）。 */
export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

export default {
  API_BASE,
  ANALYSIS_API_BASE,
  TRAINING_API_BASE,
  RDC_UPDATE_API,
  RDC_QUERY_API,
  RDC_CONSTRUCTION_API,
  FRONTEND_BASE,
  RUNTIME_MODE,
  apiUrl,
};
