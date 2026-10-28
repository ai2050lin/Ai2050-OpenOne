/**
 * AI2050 客户端 Electron 主进程
 * ────────────────────────────────────────────────────────────────
 * 职责：
 *   1) 读取用户配置的服务器地址（默认 https://server.ai2050.com）
 *   2) 在页面加载前把地址注入 window.__AI2050__（前端 src/config.js 的唯一读取入口）
 *   3) 首启（无配置）询问服务器地址，持久化到 userData/config.json
 *
 * 纪律：服务器地址来自用户显式配置；连不上时界面显示「未接入」空态，不伪造数据。
 */
'use strict';

const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const http = require('http');
const https = require('https');
const { NodeSupervisor } = require('./nodeSupervisor');
const { RuntimeInstaller, readRuntime } = require('./runtimeInstaller');

const DEFAULT_SERVER = 'https://server.ai2050.com';
const CONFIG_FILE = () => path.join(app.getPath('userData'), 'config.json');

/* ── 配置读写 ─────────────────────────────────────────────── */
function readConfig() {
  /* 沙盒联调：AI2050_CLIENT_SERVER 直接指定中心，不读写用户配置里的服务器 */
  if (process.env.AI2050_CLIENT_SERVER) {
    let saved = {};
    try { saved = JSON.parse(fs.readFileSync(CONFIG_FILE(), 'utf8')); } catch { /* 无配置 */ }
    return { ...saved, server: normalizeServer(process.env.AI2050_CLIENT_SERVER), verified: true };
  }
  try {
    const cfg = JSON.parse(fs.readFileSync(CONFIG_FILE(), 'utf8'));
    if (typeof cfg.server === 'string' && cfg.server.trim()) return cfg;
  } catch {
    /* 首次运行 / 文件损坏 → 走询问流程 */
  }
  return null;
}

function writeConfig(cfg) {
  if (process.env.AI2050_CLIENT_SERVER) {
    let saved = {};
    try { saved = JSON.parse(fs.readFileSync(CONFIG_FILE(), 'utf8')); } catch { /* 无配置 */ }
    cfg = { ...cfg, server: saved.server, verified: saved.verified };     // 环境变量指定的中心不写进用户配置
    if (!cfg.server) delete cfg.server;
  }
  fs.mkdirSync(path.dirname(CONFIG_FILE()), { recursive: true });
  fs.writeFileSync(CONFIG_FILE(), JSON.stringify(cfg, null, 2), 'utf8');
}

function normalizeServer(input) {
  let s = String(input || '').trim();
  if (!s) return null;
  if (!/^https?:\/\//i.test(s)) s = 'http://' + s;
  return s.replace(/\/+$/, '');
}

/** 探测可达性：GET {server}/api/health（开放平台中心）或旧版 /health，8s 超时。返回 {ok, status} */
function probe(server, timeoutMs = 8000) {
  return new Promise(resolve => {
    let settled = false;
    const done = r => {
      if (settled) return;
      settled = true;
      resolve(r);
    };
    try {
      const mod = server.startsWith('https://') ? https : http;
      const req = mod.get(server + '/api/health', { timeout: timeoutMs }, res => {
        res.resume();
        if (res.statusCode === 200) { done({ ok: true, status: 200 }); return; }
        const legacy = mod.get(server + '/health', { timeout: timeoutMs }, r2 => {
          r2.resume();
          done({ ok: r2.statusCode === 200, status: r2.statusCode });
        });
        legacy.on('timeout', () => { legacy.destroy(); done({ ok: false, status: 0 }); });
        legacy.on('error', () => done({ ok: false, status: res.statusCode }));
      });
      req.on('timeout', () => { req.destroy(); done({ ok: false, status: 0 }); });
      req.on('error', () => done({ ok: false, status: 0 }));
    } catch {
      done({ ok: false, status: 0 });
    }
  });
}

/* ── 自定义地址输入（模态小窗，独立 preload 走 IPC）────────── */
let promptWindow = null;

function customServerDialog(parent, current) {
  return new Promise(resolve => {
    if (promptWindow) { promptWindow.focus(); return; }

    const w = new BrowserWindow({
      parent: parent || undefined,
      modal: !!parent,
      width: 540,
      height: 300,
      resizable: false,
      minimizable: false,
      maximizable: false,
      title: '自定义服务器地址',
      autoHideMenuBar: true,
      webPreferences: {
        preload: path.join(__dirname, 'serverPrompt.js'),
        contextIsolation: true,
        nodeIntegration: false,
      },
    });
    promptWindow = w;

    let answered = false;
    const finish = value => {
      if (answered) return;
      answered = true;
      ipcMain.removeListener('ai2050:server-submit', onSubmit);
      if (!w.isDestroyed()) w.destroy();
      promptWindow = null;
      resolve(value);
    };
    const onSubmit = (_e, value) => finish(value === null ? null : normalizeServer(value));
    ipcMain.on('ai2050:server-submit', onSubmit);

    w.on('closed', () => finish(null));
    w.loadFile(path.join(__dirname, 'serverPrompt.html'));
    w.webContents.once('did-finish-load', () => {
      w.webContents.executeJavaScript(
        'window.__setServer(' + JSON.stringify(current || DEFAULT_SERVER) + ')'
      ).catch(() => {});
    });
  });
}

/* ── 首次运行：选择服务器 ─────────────────────────────────── */
async function ensureServer(parent) {
  const existing = readConfig();
  if (existing) return existing.server;

  const choice = await dialog.showMessageBox(parent || undefined, {
    type: 'question',
    title: 'AI2050 客户端 — 连接研究平台',
    message: '选择要连接的研究平台服务器',
    detail:
      '默认服务器：' + DEFAULT_SERVER + '\n\n' +
      '• 使用默认服务器 —— 连接到官方平台\n' +
      '• 自定义地址 —— 填写自建部署的域名或 IP\n\n' +
      '地址保存在本机，可随时通过「文件 → 切换服务器」修改。',
    buttons: ['使用默认服务器', '自定义地址…', '退出'],
    defaultId: 0,
    cancelId: 2,
    noLink: true,
  }).then(r => r.response);

  if (choice === 2) return null;
  const picked = choice === 0 ? DEFAULT_SERVER : await customServerDialog(parent, '');
  const server = normalizeServer(picked);
  if (!server) return null;

  const p = await probe(server);
  writeConfig({ server, verified: p.ok, verifiedAt: new Date().toISOString() });

  if (!p.ok) {
    await dialog.showMessageBox(parent || undefined, {
      type: 'warning',
      title: 'AI2050 客户端',
      message: '暂时连不上该服务器',
      detail:
        '已保存地址：' + server + '\n\n' +
        '客户端仍会启动，但图谱 / 结果库等面板会显示「未接入」空态（不会伪造数据）。\n' +
        '请检查网络或地址是否正确，稍后可通过「文件 → 切换服务器」修改。',
      buttons: ['继续'],
    });
  }
  return server;
}

/* ── 主窗口 ───────────────────────────────────────────────── */
let mainWindow = null;

function createWindow(server) {
  const base = server || '';
  const win = new BrowserWindow({
    width: 1600,
    height: 980,
    minWidth: 1280,
    minHeight: 800,
    show: false,
    backgroundColor: '#f7f8fa',
    title: 'AI2050 语言机制研究平台',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      /* 服务器地址经 additionalArguments 同步传给 preload：
         preload 在页面脚本之前执行，config.js 首读即拿到（executeJavaScript 太晚）。 */
      additionalArguments: [
        '--ai2050-server=' + base,
        '--ai2050-mode=electron',
      ],
    },
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });

  // 打包后 dist 在 app.asar 内；开发时指向 frontend/dist
  win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  win.once('ready-to-show', () => win.show());
  win.on('closed', () => { if (mainWindow === win) mainWindow = null; });

  mainWindow = win;
  return win;
}

/* ── 菜单 ─────────────────────────────────────────────────── */
function buildMenu() {
  const template = [
    {
      label: '文件',
      submenu: [
        {
          label: '切换服务器…',
          click: async () => {
            const cur = (readConfig() || {}).server || '';
            const custom = await customServerDialog(mainWindow, cur);
            if (!custom) return;
            const p = await probe(custom);
            writeConfig({ server: custom, verified: p.ok, verifiedAt: new Date().toISOString() });
            await dialog.showMessageBox(mainWindow, {
              type: p.ok ? 'info' : 'warning',
              message: p.ok ? '服务器已切换' : '地址已保存，但暂时连不上',
              detail: (p.ok ? custom + '\n\n正在重新加载界面…'
                          : custom + '\n\n界面会显示「未接入」空态，请检查地址或网络。'),
            });
            nodeSup.stop().then(() => nodeSup.start(custom));
            if (mainWindow) mainWindow.reload();
          },
        },
        { type: 'separator' },
        { label: '重新加载界面', accelerator: 'CmdOrCtrl+R', click: () => mainWindow && mainWindow.reload() },
        { type: 'separator' },
        { role: 'quit', label: '退出' },
      ],
    },
    {
      label: '视图',
      submenu: [
        { role: 'resetZoom', label: '实际大小' },
        { role: 'zoomIn', label: '放大' },
        { role: 'zoomOut', label: '缩小' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: '全屏' },
        { role: 'toggleDevTools', label: '开发者工具' },
      ],
    },
    {
      label: '帮助',
      submenu: [
        {
          label: '在浏览器中打开服务器',
          click: () => shell.openExternal((readConfig() || {}).server || DEFAULT_SERVER),
        },
        { type: 'separator' },
        {
          label: '关于 AI2050',
          click: () => {
            const cfg = readConfig() || {};
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              message: 'AI2050 语言机制研究平台',
              detail:
                '客户端版本：' + app.getVersion() + '\n' +
                '服务器：' + (cfg.server || '未配置') + '\n' +
                '连通性：' + (cfg.verified ? '已验证' : '未验证或失败') + '\n\n' +
                '全部实验数据来自服务器端，客户端不内置数据。\n' +
                '连接不可用时面板显示「未接入」空态，不伪造结果。',
            });
          },
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

/* ── 生命周期 ─────────────────────────────────────────────── */
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  });

  app.whenReady().then(async () => {
    const server = await ensureServer(null);
    if (!server) { app.quit(); return; }
    buildMenu();
    createWindow(server);
    nodeSup.start(server);
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createWindow((readConfig() || {}).server || DEFAULT_SERVER);
      }
    });
  });

  app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });

  /* 退出前关掉自己启动的本机节点服务（运行中的任务被结束，中心在租约到期后重派） */
  let nodeStopped = false;
  app.on('before-quit', e => {
    if (nodeStopped) return;
    e.preventDefault();
    nodeStopped = true;
    nodeSup.stop().finally(() => app.quit());
  });
}

ipcMain.handle('ai2050:get-server', () => (readConfig() || {}).server || DEFAULT_SERVER);

/* ── 本机节点服务（贡献算力）：界面只经这几个 IPC 调用，令牌不离开主进程 ── */
const runtimeDir = () => (readConfig() || {}).runtimeDir || path.join(app.getPath('userData'), 'runtime');
const lockFile = [path.join(process.resourcesPath || '', 'node', 'runtime', 'requirements-win-cu130.lock'),
  path.join(__dirname, '..', '..', 'deploy', 'runtime', 'requirements-win-cu130.lock')].find(p => fs.existsSync(p));
const nodeSup = new NodeSupervisor({
  app, readConfig, writeConfig,
  runtimePython: () => (readRuntime(runtimeDir()) || {}).python || '',
  onChange: info => { if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('ai2050:node-info', info); },
});
ipcMain.handle('ai2050:node-info', () => nodeSup.info());
ipcMain.handle('ai2050:node-request', (_e, req) => nodeSup.request(req && req.method, req && req.path, req && req.body));
ipcMain.handle('ai2050:node-restart', () => nodeSup.restart());
ipcMain.handle('ai2050:node-pick-python', async () => {
  const r = await dialog.showOpenDialog(mainWindow, {
    title: '选择运行任务的 Python（需要 torch 与 transformers）',
    properties: ['openFile'],
    filters: process.platform === 'win32' ? [{ name: 'python.exe', extensions: ['exe'] }] : [],
  });
  if (r.canceled || !r.filePaths[0]) return nodeSup.info();
  await nodeSup.setPython(r.filePaths[0]);
  return nodeSup.start();
});

/* ── 计算组件（Python + torch + transformers）：主进程安装，不需要系统里已有 Python ── */
const runtime = new RuntimeInstaller({
  dir: runtimeDir, lockFile,
  onChange: info => { if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('ai2050:runtime-info', info); },
});
ipcMain.handle('ai2050:runtime-info', () => runtime.info());
ipcMain.handle('ai2050:runtime-install', async (_e, req) => {
  const info = await runtime.install({ mirror: req && req.mirror });
  // 装好后：用户没有手动指定 Python 时，节点服务改用新环境
  if (info.state === 'done' && !(readConfig() || {}).nodePython && nodeSup.python !== info.installed.python) await nodeSup.restart();
  return runtime.info();
});
ipcMain.handle('ai2050:runtime-cancel', () => runtime.cancel());
ipcMain.handle('ai2050:runtime-pick-dir', async () => {
  if (runtime.state === 'running') return runtime.info();
  const r = await dialog.showOpenDialog(mainWindow, {
    title: '选择计算组件的安装目录（约 6 GB，建议放在空间大的盘）', properties: ['openDirectory', 'createDirectory'],
  });
  if (!r.canceled && r.filePaths[0]) {
    const dir = path.join(r.filePaths[0], 'ai2050-runtime');
    writeConfig({ ...(readConfig() || {}), runtimeDir: dir });
  }
  return runtime.info();
});
ipcMain.handle('ai2050:node-pick-models-dir', async () => {
  const r = await dialog.showOpenDialog(mainWindow, { title: '选择模型下载目录（每个模型 1–30 GB）', properties: ['openDirectory', 'createDirectory'] });
  return r.canceled || !r.filePaths[0] ? '' : r.filePaths[0];
});
