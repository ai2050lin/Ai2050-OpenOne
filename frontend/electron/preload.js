/**
 * Electron preload —— 在任何页面脚本执行前注入服务器配置。
 *
 * 时序关键：config.js 的模块顶层在 import 时立即求值 API_BASE，
 * 配置必须**早于**页面脚本。主进程通过 webPreferences.additionalArguments
 * 同步传入（process.argv），preload 在此直接挂到 window，
 * 因此 config.js 第一次读 window.__AI2050__ 就能拿到值。
 *
 * ⚠ contextIsolation 语义（实测确认，勿改回直接赋值）：
 *   contextIsolation: true 时，preload 与页面主世界是**两个不同的 JS 全局**。
 *   preload 里 `window.__AI2050__ = cfg` 只写在隔离世界，主世界读不到；
 *   而 config.js 拿不到 apiBase 会退到 `ai2050Desktop.isDesktop → ''` 分支，
 *   于是 fetch('/api/...') 打到 file:// 协议上——全部接口静默失败、无任何报错。
 *   唯一正确通道是 contextBridge.exposeInMainWorld（它把值**复制**进主世界）。
 */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

/* 形如 --ai2050-server=https://server.ai2050.com */
const argv = process.argv || [];
let server = '';
let mode = 'electron';
for (let i = 0; i < argv.length; i += 1) {
  const a = String(argv[i]);
  if (a.startsWith('--ai2050-server=')) server = a.slice('--ai2050-server='.length);
  if (a === '--ai2050-server' && argv[i + 1]) server = String(argv[i + 1]);
  if (a.startsWith('--ai2050-mode=')) mode = a.slice('--ai2050-mode='.length);
}

const config = { apiBase: server || '', mode };

/* 主世界唯一入口：config.js 读 window.__AI2050__。
   contextBridge 会把普通对象按值复制进主世界（只读快照）。 */
contextBridge.exposeInMainWorld('__AI2050__', config);

/* 只读能力，不泄漏 Node */
contextBridge.exposeInMainWorld('ai2050Desktop', {
  isDesktop: true,
  platform: process.platform,
  mode,
  server: config.apiBase,
  getServer: () => ipcRenderer.invoke('ai2050:get-server'),
  /* 本机节点服务（贡献算力）：主进程只转发白名单内的接口，本机令牌不进页面 */
  node: {
    info: () => ipcRenderer.invoke('ai2050:node-info'),
    request: (method, path, body) => ipcRenderer.invoke('ai2050:node-request', { method, path, body }),
    restart: () => ipcRenderer.invoke('ai2050:node-restart'),
    pickPython: () => ipcRenderer.invoke('ai2050:node-pick-python'),
    pickModelsDir: () => ipcRenderer.invoke('ai2050:node-pick-models-dir'),
    onInfo: cb => {
      const h = (_e, info) => cb(info);
      ipcRenderer.on('ai2050:node-info', h);
      return () => ipcRenderer.removeListener('ai2050:node-info', h);
    },
  },
  /* AI 助手：配置与 Key 在主进程，页面只能看到「已设置 · 尾号」；回答经 ai2050:ai-event 流式推回 */
  ai: (() => {
    const call = async (ch, arg) => {
      const r = await ipcRenderer.invoke(ch, arg);
      if (!r || !r.ok) throw new Error((r && r.error) || 'AI 助手不可用');
      return r.data;
    };
    return {
      config: () => call('ai2050:ai-config'),
      save: profile => call('ai2050:ai-save', profile),
      remove: id => call('ai2050:ai-remove', { id }),
      settings: s => call('ai2050:ai-settings', s),
      test: id => call('ai2050:ai-test', { id }),
      chat: req => call('ai2050:ai-chat', req),
      cancel: id => call('ai2050:ai-cancel', { id }),
      openLogs: () => call('ai2050:ai-open-logs'),
      onEvent: cb => {
        const h = (_e, ev) => cb(ev);
        ipcRenderer.on('ai2050:ai-event', h);
        return () => ipcRenderer.removeListener('ai2050:ai-event', h);
      },
    };
  })(),
  /* 计算组件安装（主进程执行；页面只发起、取消与查看进度） */
  runtime: {
    info: () => ipcRenderer.invoke('ai2050:runtime-info'),
    install: mirror => ipcRenderer.invoke('ai2050:runtime-install', { mirror }),
    cancel: () => ipcRenderer.invoke('ai2050:runtime-cancel'),
    pickDir: () => ipcRenderer.invoke('ai2050:runtime-pick-dir'),
    onInfo: cb => {
      const h = (_e, info) => cb(info);
      ipcRenderer.on('ai2050:runtime-info', h);
      return () => ipcRenderer.removeListener('ai2050:runtime-info', h);
    },
  },
});

/* 隔离世界侧自留一份（仅供调试/后续 preload 内部使用；主世界不依赖它）。 */
try {
  Object.defineProperty(window, '__AI2050_ISOLATED__', {
    value: config,
    writable: false,
    configurable: false,
    enumerable: true,
  });
} catch {
  window.__AI2050_ISOLATED__ = config;
}