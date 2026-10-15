/**
 * 服务器地址输入框的 preload：只做一件事——把用户输入回传主进程。
 * 刻意不暴露任何其他能力。
 */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('ai2050Prompt', {
  submit: value => ipcRenderer.send('ai2050:server-submit', value),
});
