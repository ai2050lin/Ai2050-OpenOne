#!/usr/bin/env node
/* AI2050 前端常驻服务守护器（dev_server.mjs · v1 · 2026-10-19）
   ═══════════════════════════════════════════════════════════════════
   要解决的真实故障（反复出现「VSCode 启动客户端，访问不了」）：

     ① vite 是"一次性前台进程"，没有任何监护。它一死，5173 就空着，
        界面上表现为浏览器 ERR_CONNECTION_REFUSED，而后端 python 还活着
        —— 于是出现「5001 在、5173 没了」这种半死状态。
     ② 进程死了不留痕迹。终端一关、窗口一 reload，报错就没了，
        下次只能靠重开碰运气，故障永远定位不到。
     ③ 半死端口（在监听、HTTP 不应答）被旧脚本判成"已被占用"直接抛错退出。
     ④ 浏览器在 vite 真正 bind 之前就被打开 → 必然 ERR_CONNECTION_REFUSED。

   本脚本：把 vite 交给一个守护进程托管
     · 幂等：健康就复用，不重复拉起
     · 回收：端口被本项目僵死 vite 占着 → 清掉重建；是别人的进程 → 如实报错不擅自杀
     · 就绪门：等到 HTTP 200 且页面含 id="root" 才打印地址（浏览器不会再撞空端口）
     · 自愈：vite 异常退出按 600ms→1.2s→2.4s→4.8s→8s 退避重启，1 分钟内超过 5 次才放弃
     · 可观测：全部输出带时间戳落到 .run/devserver.log
     · 脱管：检测到 VSCode 终端时，守护进程 detach 出去，窗口 reload / 终端被关都不影响服务

   用法：
     node scripts/dev_server.mjs                 默认：确保可用（VSCode 下自动脱管）
     node scripts/dev_server.mjs --foreground    前台守护（Ctrl+C 停止，日志照旧落盘）
     node scripts/dev_server.mjs --detach        强制脱管
     node scripts/dev_server.mjs --status        查状态
     node scripts/dev_server.mjs --stop          停止并等端口真正释放
     node scripts/dev_server.mjs --port 5174     换端口（仅显式指定时）
     node scripts/dev_server.mjs --mode preview  托管 vite preview

   登录态文件：.run/devserver.json   日志：.run/devserver.log（.gitignore 已覆盖 *.log）
   纪律：不伪造成功。端口被无关进程占用时给出明确原因与备选端口，绝不静默换端口
        （静默换端口会让书签/快捷键指向一个没人监听的地址，是更坏的体验）。
*/
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import os from 'node:os';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.dirname(HERE);
const FRONTEND = path.join(REPO, 'frontend');
const RUN_DIR = path.join(REPO, '.run');
const STATE_FILE = path.join(RUN_DIR, 'devserver.json');
const LOG_FILE = path.join(RUN_DIR, 'devserver.log');
const VITE_BIN = path.join(FRONTEND, 'node_modules', 'vite', 'bin', 'vite.js');

const argv = process.argv.slice(2);
const has = (n) => argv.includes('--' + n);
const opt = (n, d) => {
  const i = argv.indexOf('--' + n);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : d;
};

const PORT = Number(opt('port', process.env.AI2050_PORT || 5173));
const MODE = opt('mode', 'preview') === 'preview' && has('mode') ? 'preview' : 'dev';
const MAX_RESTART = Number(opt('max-restart', 5));
const RESTART_WINDOW_MS = 60_000;
const READY_TIMEOUT_MS = Number(opt('ready-timeout', 120_000));

/* 在 VSCode 集成终端里默认脱管：VSCode reload 窗口 / 回收终端 / 调试会话重连失败
   都可能把前台子进程带走，而服务不该随编辑器消失。
   显式 --foreground / --detach 优先。 */
const IN_VSCODE = /^vscode$/i.test(process.env.TERM_PROGRAM || '') ||
                  Boolean(process.env.VSCODE_PID) || Boolean(process.env.VSCODE_IPC_HOOK_CLI);
const WANT_DETACH = has('detach') ? true : has('foreground') ? false : IN_VSCODE;

const T0 = Date.now();
const stamp = () => {
  const d = new Date();
  const p = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
         `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
};
const say = (msg) => console.log(`[AI2050] ${msg}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function logLine(msg) {
  try {
    fs.mkdirSync(RUN_DIR, { recursive: true });
    fs.appendFileSync(LOG_FILE, `[${stamp()}] ${msg}\n`, 'utf8');
  } catch { /* 日志不可写不应拖垮主流程 */ }
}
function logTail(lines = 20) {
  try {
    return fs.readFileSync(LOG_FILE, 'utf8').split(/\r?\n/).filter(Boolean).slice(-lines).join('\n');
  } catch {
    return '(无日志)';
  }
}

/* ── 状态文件 ─────────────────────────────────────────────── */
function readState() {
  try { return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8')); } catch { return null; }
}
function writeState(patch) {
  const next = { ...(readState() || {}), ...patch, updatedAt: stamp() };
  try {
    fs.mkdirSync(RUN_DIR, { recursive: true });
    fs.writeFileSync(STATE_FILE, JSON.stringify(next, null, 2), 'utf8');
  } catch { /* ignore */ }
  return next;
}
function clearState() {
  try { fs.rmSync(STATE_FILE, { force: true }); } catch { /* ignore */ }
}
function alive(pid) {
  if (!pid || Number.isNaN(Number(pid))) return false;
  try { process.kill(Number(pid), 0); return true; } catch (e) { return e.code === 'EPERM'; }
}

/* ── 健康探针：HTTP 200 且确实是本应用的页面 ──────────────── */
function probe(port = PORT, timeout = 1500) {
  return new Promise((resolve) => {
    const req = http.get({ host: '127.0.0.1', port, path: '/', timeout }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (c) => { if (body.length < 8192) body += c; });
      res.on('end', () => {
        const ours = res.statusCode === 200 && body.includes('id="root"');
        resolve({ ok: ours, status: res.statusCode });
      });
    });
    req.on('timeout', () => { req.destroy(); resolve({ ok: false, status: 0, reason: 'timeout' }); });
    req.on('error', (e) => resolve({ ok: false, status: 0, reason: e.code || e.message }));
  });
}

/* ── 端口占用者 ───────────────────────────────────────────── */
function listenerPid(port = PORT) {
  try {
    if (process.platform === 'win32') {
      const r = spawnSync('netstat', ['-ano', '-p', 'tcp'], { encoding: 'utf8', windowsHide: true });
      const m = new RegExp(`^\\s*TCP\\s+\\S*:${port}\\s+\\S+\\s+LISTENING\\s+(\\d+)`, 'mi').exec(r.stdout || '');
      return m ? Number(m[1]) : null;
    }
    const r = spawnSync('lsof', ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN', '-t'], { encoding: 'utf8' });
    const pid = Number((r.stdout || '').trim().split(/\s+/)[0]);
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch { return null; }
}

function procInfo(pid) {
  try {
    if (process.platform === 'win32') {
      const t = spawnSync('tasklist', ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true });
      const name = ((t.stdout || '').split(/\r?\n/)[0] || '').split('","')[0].replace(/^"|"$/g, '').trim() || 'unknown';
      const c = spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command',
        `(Get-CimInstance Win32_Process -Filter "ProcessId=${pid}").CommandLine`], { encoding: 'utf8', windowsHide: true });
      return { name, cmd: (c.stdout || '').trim() };
    }
    const r = spawnSync('ps', ['-p', String(pid), '-o', 'comm=,args='], { encoding: 'utf8' });
    const line = (r.stdout || '').trim();
    return { name: line.split(/\s+/)[0] || 'unknown', cmd: line };
  } catch { return { name: 'unknown', cmd: '' }; }
}

function killTree(pid) {
  if (!pid) return;
  try {
    if (process.platform === 'win32') {
      spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
    } else {
      try { process.kill(-Number(pid), 'SIGTERM'); } catch { process.kill(Number(pid), 'SIGTERM'); }
    }
  } catch { /* ignore */ }
}

async function waitPortFree(port = PORT, timeout = 12_000) {
  const until = Date.now() + timeout;
  while (Date.now() < until) {
    if (listenerPid(port) === null) return true;
    await sleep(300);
  }
  return listenerPid(port) === null;
}

function looksLikeOurVite(info) {
  const cmd = String(info.cmd || '').replace(/\//g, '\\').toLowerCase();
  const root = FRONTEND.replace(/\//g, '\\').toLowerCase();
  return cmd.includes(root) && cmd.includes('vite');
}

/* 保持 vite 原始行格式：VSCode launch.json 的 serverReadyAction 靠这条正则开浏览器 */
function printUrl() {
  console.log(`  Local:   http://localhost:${PORT}/`);
}

/* ══════════════════════════════════════════════════════════════
   守护主体：拉起 vite、报告就绪、异常退出重启
   ══════════════════════════════════════════════════════════════ */
async function supervise() {
  if (!fs.existsSync(VITE_BIN)) {
    logLine('FATAL 找不到 vite：' + VITE_BIN);
    say('找不到 vite，请先在 frontend 目录安装依赖（npm ci）');
    process.exit(2);
  }
  fs.mkdirSync(RUN_DIR, { recursive: true });
  const fd = fs.openSync(LOG_FILE, 'a');
  let child = null;
  let stopping = false;
  let announced = false;
  const restarts = [];

  const shutdown = (sig) => {
    if (stopping) return;
    stopping = true;
    logLine(`守护器收到 ${sig}，停止 vite${child ? ` (pid ${child.pid})` : ''}`);
    if (child && child.pid) killTree(child.pid);
    clearState();
    setTimeout(() => process.exit(0), 300);
  };
  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK']) {
    try { process.on(sig, () => shutdown(sig)); } catch { /* 平台不支持则忽略 */ }
  }

  writeState({ supervisorPid: process.pid, port: PORT, mode: MODE, startedAt: stamp(), detached: WANT_DETACH });
  logLine(`守护器启动 pid=${process.pid} node=${process.version} port=${PORT} mode=${MODE} vscode=${IN_VSCODE} detach=${WANT_DETACH}`);

  /* 就绪播报：只在真正 HTTP 200 时打印一次（重启后再报） */
  const announceLoop = async () => {
    while (!stopping) {
      if (await probe(PORT, 900)) {
        if (!announced) {
          announced = true;
          say(`就绪（${((Date.now() - T0) / 1000).toFixed(1)}s）`);
          printUrl();
          logLine('就绪：HTTP 200');
        }
      } else {
        announced = false;
      }
      await sleep(1000);
    }
  };
  announceLoop();

  for (;;) {
    const args = [VITE_BIN];
    if (MODE === 'preview') args.push('preview');
    args.push('--port', String(PORT), '--strictPort');

    fs.writeSync(fd, `\n[${stamp()}] ── 启动 vite (${args.slice(1).join(' ')}) ──\n`);
    child = spawn(process.execPath, args, {
      cwd: FRONTEND,
      env: { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' },
      stdio: ['ignore', fd, fd],
      windowsHide: true,
    });
    writeState({ vitePid: child.pid });

    const code = await new Promise((resolve) => {
      child.on('exit', (c, sig) => resolve({ c, sig }));
      child.on('error', (e) => {
        fs.writeSync(fd, `[${stamp()}] spawn 失败：${e.message}\n`);
        resolve({ c: -1, sig: null });
      });
    });

    if (stopping) break;
    fs.writeSync(fd, `[${stamp()}] vite 退出 code=${code.c} signal=${code.sig}\n`);
    writeState({ vitePid: null, lastExit: code.c, lastExitAt: stamp() });
    announced = false;

    const now = Date.now();
    restarts.push(now);
    while (restarts.length && now - restarts[0] > RESTART_WINDOW_MS) restarts.shift();

    if (code.c === 0) {
      logLine('vite 正常退出（code 0），守护器一并结束');
      clearState();
      process.exit(0);
    }
    if (restarts.length > MAX_RESTART) {
      logLine(`FATAL ${RESTART_WINDOW_MS / 1000}s 内重启超过 ${MAX_RESTART} 次，放弃`);
      say(`vite 反复退出，已放弃。日志：${LOG_FILE}`);
      clearState();
      process.exit(1);
    }
    const backoff = Math.min(8000, 600 * 2 ** (restarts.length - 1));
    logLine(`${backoff}ms 后重启（窗口内第 ${restarts.length} 次）`);
    await sleep(backoff);
  }
  clearState();
  process.exit(0);
}

async function waitReady(supPid, timeout = READY_TIMEOUT_MS) {
  const until = Date.now() + timeout;
  while (Date.now() < until) {
    if (await probe(PORT, 900)) return true;
    if (supPid && !alive(supPid)) return false;
    await sleep(350);
  }
  return false;
}

/* ══════════════════════════════════════════════════════════════
   ensure：幂等确保可用
   ══════════════════════════════════════════════════════════════ */
async function ensure() {
  const first = await probe();
  if (first.ok) {
    const st = readState() || {};
    say(`已在运行（${PORT}），直接复用${st.startedAt ? `，启动于 ${st.startedAt}` : ''}`);
    logLine('ensure: 复用已健康实例');
    printUrl();
    return 0;
  }

  /* 已有守护进程在跑但尚未就绪 → 等它，不要重复拉起 */
  const st = readState();
  if (st && alive(st.supervisorPid) && Number(st.port) === PORT) {
    say(`守护进程已在启动中（PID ${st.supervisorPid}），等待就绪…`);
    if (await waitReady(st.supervisorPid)) { printUrl(); return 0; }
    say('等待超时，日志尾部：');
    console.error(logTail(15));
    return 1;
  }

  /* 端口被占：区分「我们自己的僵死 vite」与「别人的进程」 */
  const occupant = listenerPid(PORT);
  if (occupant) {
    const info = procInfo(occupant);
    if (looksLikeOurVite(info)) {
      say(`端口 ${PORT} 被本项目僵死的 vite 占着（PID ${occupant}），回收后重建`);
      logLine(`ensure: 回收僵死 vite pid=${occupant}`);
      killTree(occupant);
      if (!(await waitPortFree(PORT))) {
        say(`僵死进程未能释放端口，请手动结束 PID ${occupant}`);
        return 1;
      }
    } else {
      say(`端口 ${PORT} 已被其它进程占用：${info.name}（PID ${occupant}）`);
      if (info.cmd) say(`  命令行：${info.cmd.slice(0, 160)}`);
      say('它不属于本项目，我不会擅自结束它。请关闭它，或显式换端口：');
      say(`  node scripts/dev_server.mjs --port 5174`);
      logLine(`ensure: 端口被外部进程占用 pid=${occupant} name=${info.name}`);
      return 1;
    }
  }

  /* 内存水位（本机提交内存长期 55–60/65 GB，vite 分配失败是首要嫌疑之一） */
  const freeGB = os.freemem() / 1024 ** 3;
  if (freeGB < 1.2) say(`可用物理内存仅 ${freeGB.toFixed(1)}GB，vite 可能因分配失败退出（会记入日志）`);

  if (!WANT_DETACH) return supervise();

  /* 脱管：守护进程独立于本终端，VSCode reload/关终端都不影响 */
  const sup = spawn(process.execPath, [fileURLToPath(import.meta.url), '--supervise',
    '--port', String(PORT), '--mode', MODE, '--detach'], {
    cwd: REPO, detached: true, stdio: 'ignore', windowsHide: true,
    env: { ...process.env, AI2050_DEVSERVER: '1' },
  });
  sup.unref();
  say(`守护进程已脱管（PID ${sup.pid}），等待就绪…`);

  if (await waitReady(sup.pid)) {
    /* 就绪后再确认一次：脱管进程若被外部策略回收，这里要如实报失败，
       而不是打印一个已经没人监听的地址。 */
    await sleep(1200);
    if (!(await probe(PORT, 1200)) || !alive(sup.pid)) {
      say('刚就绪又掉线了：守护进程已被终止（可能是环境限制了后台进程）');
      say(`请改用前台守护：node scripts/dev_server.mjs --foreground`);
      console.error(logTail(12));
      return 1;
    }
    printUrl();
    return 0;
  }
  say('启动失败，日志尾部：');
  console.error(logTail(25));
  return 1;
}

/* ══════════════════════════════════════════════════════════════ */
async function status() {
  const r = await probe();
  const st = readState() || {};
  const pid = listenerPid(PORT);
  say(`端口 ${PORT}：${r.ok ? 'HTTP 200 · 正常' : '不可用'}`);
  say(`守护进程：${alive(st.supervisorPid) ? `PID ${st.supervisorPid}（存活${st.detached ? '，已脱管' : ''}）` : '未运行'}`);
  say(`vite 进程：${st.vitePid ? `PID ${st.vitePid}` : '无'}`);
  if (pid && pid !== st.vitePid) say(`端口占用者：${procInfo(pid).name}（PID ${pid}）`);
  if (st.startedAt) say(`启动于 ${st.startedAt}，最近心跳 ${st.updatedAt || '-'}`);
  say(`日志：${LOG_FILE}`);
  if (r.ok) printUrl();
  return r.ok ? 0 : 1;
}

async function stop() {
  const st = readState() || {};
  const targets = [...new Set([st.supervisorPid, st.vitePid, listenerPid(PORT)].filter(Boolean))];
  if (!targets.length) { say('没有记录到运行中的服务'); clearState(); return 0; }
  for (const p of targets) {
    if (alive(p)) { say(`结束 PID ${p} 及其子进程`); killTree(p); }
  }
  clearState();
  if (!(await waitPortFree(PORT))) {
    say(`端口 ${PORT} 仍被占用（PID ${listenerPid(PORT)}），请手动处理`);
    return 1;
  }
  say(`已停止，端口 ${PORT} 已释放`);
  return 0;
}

/* ── 入口 ─────────────────────────────────────────────────── */
const main = async () => {
  if (has('supervise')) return supervise();
  if (has('stop')) return stop();
  if (has('status')) return status();
  return ensure();
};

main()
  .then((code) => process.exit(code || 0))
  .catch((e) => { say('内部错误：' + (e && e.stack ? e.stack : e)); process.exit(1); });
