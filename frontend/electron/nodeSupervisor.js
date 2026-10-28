/**
 * 本机节点服务的监护（贡献算力客户端 P2，设计见 design/contributor_client_v1.md）
 * ────────────────────────────────────────────────────────────────
 * 主进程启动 `python node_agent.py serve --port 0`，读 <home>/service.json 得到端口与本机令牌，
 * 界面只经 IPC 调用白名单内的接口：令牌留在主进程，页面拿不到，也不能直接访问 127.0.0.1。
 * 已有一个正在运行的服务（如命令行启动的）时直接接管连接，不重复启动，退出时也不关它。
 * 服务意外退出时按 2 s 起倍增（上限 60 s）重启；10 分钟内连续 5 次失败后停下，界面显示原因与输出尾部。
 * --synthetic 只能由环境变量 AI2050_NODE_SYNTHETIC=1 打开（沙盒中心联调），界面不能打开。
 */
'use strict';

const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');

/* 界面可调用的接口（/shutdown 只由主进程在退出时调用） */
const ROUTES = new Set([
  'GET /status', 'GET /device', 'GET /config', 'GET /models', 'GET /logs', 'GET /history',
  'PUT /config', 'POST /models/check', 'POST /models/download', 'POST /models/cancel', 'POST /bind', 'POST /start', 'POST /pause', 'POST /resume', 'POST /stop',
]);
const READY_TIMEOUT = 30000;
const MAX_FAILS = 5;
const FAIL_WINDOW = 10 * 60 * 1000;

function nodeHome() {
  if (process.env.AI2050_NODE_HOME) return process.env.AI2050_NODE_HOME;
  const base = process.env.AI2050_HOME || path.join(os.homedir(), '.ai2050');
  return path.join(base, 'node_service');           // 与 node_service.default_home() 相同：命令行与客户端共用凭据
}

function exists(p) {
  try { return Boolean(p) && fs.statSync(p).isFile(); } catch { return false; }
}

/* Windows 商店的 python.exe 只是安装提示，不能用 */
function pathPythons() {
  if (process.platform !== 'win32') return ['python3', 'python'];
  try {
    return execFileSync('where.exe', ['python'], { encoding: 'utf8', windowsHide: true, timeout: 5000 })
      .split(/\r?\n/).map(s => s.trim()).filter(s => s && !/WindowsApps/i.test(s));
  } catch {
    return [];
  }
}

function httpJson(port, token, method, route, body, timeoutMs = 15000) {
  return new Promise(resolve => {
    const data = body === undefined ? null : Buffer.from(JSON.stringify(body));
    const req = http.request({
      host: '127.0.0.1', port, path: route, method, timeout: timeoutMs,
      headers: { 'X-Local-Token': token, ...(data ? { 'Content-Type': 'application/json', 'Content-Length': data.length } : {}) },
    }, res => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        let parsed = null;
        try { parsed = JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { /* 非 JSON */ }
        resolve({ status: res.statusCode, data: parsed });
      });
    });
    req.on('timeout', () => { req.destroy(); resolve({ status: 0, data: { detail: '本机节点服务没有响应' } }); });
    req.on('error', e => resolve({ status: 0, data: { detail: `连不上本机节点服务：${e.code || e.message}` } }));
    if (data) req.write(data);
    req.end();
  });
}

class NodeSupervisor {
  constructor({ app, readConfig, writeConfig, onChange, runtimePython }) {
    this.app = app;
    this.readConfig = readConfig;
    this.writeConfig = writeConfig;
    this.onChange = onChange || (() => {});
    this.runtimePython = runtimePython || (() => '');
    this.home = nodeHome();
    this.synthetic = process.env.AI2050_NODE_SYNTHETIC === '1';
    this.server = '';
    this.state = 'idle';           // idle · starting · running · attached · stopped · no-python · no-script · failed
    this.error = '';
    this.python = '';
    this.script = '';
    this.port = 0;
    this.token = '';
    this.child = null;
    this.tail = [];
    this.fails = [];
    this.timer = null;
    this.quitting = false;
  }

  info() {
    return {
      state: this.state, error: this.error, python: this.python, script: this.script, home: this.home,
      port: this.port, owned: Boolean(this.child), server: this.server, synthetic: this.synthetic,
      tail: this.tail.slice(-40), platform: process.platform,
    };
  }

  _set(state, error = '') {
    this.state = state;
    this.error = error;
    this.onChange(this.info());
  }

  _push(text) {
    for (const line of String(text).split(/\r?\n/)) {
      if (line.trim()) this.tail.push(line.slice(0, 500));
    }
    if (this.tail.length > 200) this.tail.splice(0, this.tail.length - 200);
  }

  resolveScript() {
    const packed = path.join(process.resourcesPath || '', 'node', 'node_agent.py');
    const dev = path.join(__dirname, '..', '..', 'deploy', 'node_agent.py');
    return [this.app.isPackaged ? packed : dev, packed, dev].find(exists) || '';
  }

  /* 优先级：用户在界面里选的 → 环境变量 → 客户端安装的计算组件（runtimeInstaller）→ 开发仓库的 .venv → PATH */
  resolvePython() {
    const cfg = this.readConfig() || {};
    const cands = [
      cfg.nodePython,
      process.env.AI2050_PYTHON,
      this.runtimePython(),
      this.app.isPackaged ? '' : path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe'),
      this.app.isPackaged ? '' : path.join(__dirname, '..', '..', '.venv', 'bin', 'python3'),
    ];
    const found = cands.find(exists);
    if (found) return found;
    return pathPythons().find(p => !path.isAbsolute(p) || exists(p)) || '';
  }

  readInfo() {
    try { return JSON.parse(fs.readFileSync(path.join(this.home, 'service.json'), 'utf8')); } catch { return null; }
  }

  async start(server) {
    if (server !== undefined) this.server = server;
    if (this.child || this.state === 'starting') return this.info();
    clearTimeout(this.timer);
    this.quitting = false;

    // 已有服务在运行（命令行启动、或上次客户端没有正常退出）：直接接管连接
    const prev = this.readInfo();
    if (prev && prev.port && prev.token) {
      const r = await httpJson(prev.port, prev.token, 'GET', '/status', undefined, 3000);
      if (r.status === 200) {
        Object.assign(this, { port: prev.port, token: prev.token, python: this.python || '', script: '' });
        this._set('attached');
        return this.info();
      }
    }

    this.script = this.resolveScript();
    if (!this.script) { this._set('no-script', '找不到 node_agent.py（安装包不完整）'); return this.info(); }
    this.python = this.resolvePython();
    if (!this.python) { this._set('no-python', '没有找到可用的 Python：请在下面选择 python.exe'); return this.info(); }

    this._set('starting');
    const args = ['-u', this.script, 'serve', '--home', this.home, '--port', '0'];
    if (this.server) args.push('--server-override', this.server);
    if (this.synthetic) args.push('--synthetic');
    const t0 = Date.now() / 1000;
    let child;
    try {
      child = spawn(this.python, args, {
        cwd: path.dirname(this.script), windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
      });
    } catch (e) {
      this._fail(`无法启动 Python：${e.message}`);
      return this.info();
    }
    this.child = child;
    child.stdout.on('data', d => this._push(d));
    child.stderr.on('data', d => this._push(d));
    child.on('error', e => this._push(`启动失败：${e.message}`));
    child.on('exit', code => this._exited(child, code));

    // venv 的 python.exe 在 Windows 上会再启动一个解释器，pid 不一定相同：按启动时间认 service.json
    const deadline = Date.now() + READY_TIMEOUT;
    while (Date.now() < deadline && this.child === child) {
      const info = this.readInfo();
      if (info && info.port && info.token && info.started >= t0 - 1) {
        const r = await httpJson(info.port, info.token, 'GET', '/status', undefined, 3000);
        if (r.status === 200) {
          Object.assign(this, { port: info.port, token: info.token, pid: info.pid });
          this.fails = [];
          this._set('running');
          return this.info();
        }
      }
      await new Promise(r => setTimeout(r, 300));
    }
    if (this.child === child) {
      this._kill(child);
      this._fail('本机节点服务 30 秒内没有就绪');
    }
    return this.info();
  }

  _exited(child, code) {
    if (this.child !== child) return;
    this.child = null;
    this.port = 0;
    this.token = '';
    if (this.quitting) { this._set('stopped'); return; }
    this._push(`本机节点服务退出（代码 ${code}）`);
    this._fail(`本机节点服务意外退出（代码 ${code}）`);
  }

  _fail(msg) {
    const now = Date.now();
    this.fails = this.fails.filter(t => now - t < FAIL_WINDOW).concat(now);
    if (this.fails.length >= MAX_FAILS) {
      this._set('failed', `${msg}；10 分钟内连续失败 ${this.fails.length} 次，已停止自动重启`);
      return;
    }
    const delay = Math.min(60000, 2000 * 2 ** (this.fails.length - 1));
    this._set('failed', `${msg}；${Math.round(delay / 1000)} 秒后重启`);
    clearTimeout(this.timer);
    this.timer = setTimeout(() => { if (!this.quitting && !this.child) this.start(); }, delay);
  }

  _kill(child) {
    if (!child || child.exitCode !== null) return;
    if (process.platform === 'win32') {
      for (const pid of [this.pid, child.pid].filter(Boolean)) {
        try { execFileSync('taskkill.exe', ['/pid', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' }); } catch { /* 已退出 */ }
      }
    } else {
      child.kill('SIGTERM');
    }
  }

  async request(method, route, body) {
    const m = String(method || 'GET').toUpperCase();
    const p = String(route || '');
    if (!ROUTES.has(`${m} ${p.split('?')[0]}`)) return { status: 403, data: { detail: '不允许的本机接口' } };
    if (!this.port || !this.token) return { status: 0, data: { detail: '本机节点服务没有运行' } };
    const r = await httpJson(this.port, this.token, m, p, body);
    if (r.status === 0 && this.state === 'attached') this._set('failed', '接管的本机节点服务已不可达');
    return r;
  }

  async restart() {
    await this.stop();
    this.fails = [];
    return this.start();
  }

  setPython(file) {
    const cfg = this.readConfig() || {};
    this.writeConfig({ ...cfg, nodePython: file });
    return this.restart();
  }

  /* 只关自己启动的服务；接管的服务（命令行启动）保持运行 */
  async stop() {
    this.quitting = true;
    clearTimeout(this.timer);
    const child = this.child;
    if (!child) {
      this.port = 0;
      this.token = '';
      if (this.state !== 'idle') this._set('stopped');
      return this.info();
    }
    await httpJson(this.port, this.token, 'POST', '/shutdown', undefined, 3000);
    const gone = await new Promise(resolve => {
      if (child.exitCode !== null) { resolve(true); return; }
      const t = setTimeout(() => resolve(false), 5000);
      child.once('exit', () => { clearTimeout(t); resolve(true); });
    });
    if (!gone) this._kill(child);
    this.child = null;
    this.port = 0;
    this.token = '';
    this._set('stopped');
    return this.info();
  }
}

module.exports = { NodeSupervisor, ROUTES, nodeHome };
