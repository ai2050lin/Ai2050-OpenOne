/**
 * Electron preload —— 在任何页面脚本执行前注入服务器配置。
 *
 * 时序关键：config.js 的模块顶层在 import 时立即求值 API_BASE，
 * 配置必须**早于**页面脚本。主进程通过 webPreferences.additionalArguments
 * 同步传入（process.argv），preload 在此直接挂到 window，
 * 因此 config.js 第一次读 window.__AI2050__ 就能拿到值。
 *
 * 说明：contextIsolation 开启时 preload 依然能写自己的隔离世界 window，
 * 渲染进程主世界读到的即是同一份（隔离世界共享同一 DOM window 的自定义属性投放，
 * 故这里直接赋值 + contextBridge 双保险）。
 */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

/* 形如 --ai2050-server=https://platform.ai2050.com */
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

/* 写入页面全局：config.js 读 window.__AI2050__ */
try {
  Object.defineProperty(window, '__AI2050__', {
    value: config,
    writable: false,
    configurable: false,
    enumerable: true,
  });
} catch {
  window.__AI2050__ = config;
}

/* 只读能力，不泄漏 Node */
contextBridge.exposeInMainWorld('ai2050Desktop', {
  isDesktop: true,
  platform: process.platform,
  mode,
  server: config.apiBase,
  getServer: () => ipcRenderer.invoke('ai2050:get-server'),
});
