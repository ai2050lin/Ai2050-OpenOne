/**
 * 计算组件安装（贡献算力客户端 P3，设计见 design/contributor_client_v1.md §5.3）
 * ────────────────────────────────────────────────────────────────
 * 在 <运行时目录>（缺省 userData/runtime，可换到空间大的盘）里装一套与开发机验证过的组合相同的 Python 环境：
 *   1. 下载 uv 的 PyPI wheel（固定版本，按 sha256 核对；PyPI 或清华镜像，内容相同）并用 Windows 自带 tar 解出 uv.exe
 *   2. uv python install 固定版本的 CPython；uv venv
 *   3. uv pip install --require-hashes -r requirements-win-cu130.lock（每个包都按锁文件里的 sha256 核对；
 *      torch 来自 download.pytorch.org 的 cu130 索引，其余来自 PyPI 或镜像）
 *   4. 在新环境里检测 torch / CUDA / transformers，写 runtime.json
 * 只用 Node 内置模块；不需要系统里已有 Python。缓存（UV_CACHE_DIR）也放在运行时目录里，不占系统盘。
 */
'use strict';

const { spawn, execFileSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const https = require('https');
const http = require('http');
const path = require('path');

const RUNTIME_ID = 'win-cu130-py3.14.6-1';            // 锁文件或版本变化时升级，旧环境提示重装
const PYTHON_VERSION = '3.14.6';
const UV = {
  version: '0.11.28',
  file: 'uv-0.11.28-py3-none-win_amd64.whl',
  sha256: 'f4fcf2c8d9f1444b900e6b8dbbb828825fb76eca01acd18aeaa5c90240408cda',
  size: 27603677,
  path: '/packages/40/bc/d67b18cddd54c503c7bad2b189a47fd7a1d07ea10b9212624f892b985498/uv-0.11.28-py3-none-win_amd64.whl',
};
const MIRRORS = {
  pypi: { label: 'PyPI（官方）', files: 'https://files.pythonhosted.org', index: 'https://pypi.org/simple' },
  tuna: { label: '清华镜像', files: 'https://pypi.tuna.tsinghua.edu.cn', index: 'https://pypi.tuna.tsinghua.edu.cn/simple' },
};
const TORCH_INDEX = 'https://download.pytorch.org/whl/cu130';
const MIN_DRIVER = 580;                               // CUDA 13.0 需要的 NVIDIA 驱动主版本
const NEED_GIB = 8;                                   // 环境 + 缓存约 5–6 GiB，留余量
const PROBE = 'import json\nd={}\ntry:\n import torch\n d["torch"]=torch.__version__\n d["cuda"]=torch.version.cuda\n'
  + ' d["cuda_available"]=bool(torch.cuda.is_available())\nexcept Exception as e:\n d["torch_error"]=str(e)[:200]\n'
  + 'try:\n import transformers\n d["transformers"]=transformers.__version__\nexcept Exception as e:\n'
  + ' d["transformers_error"]=str(e)[:200]\nprint(json.dumps(d))\n';

function venvPython(dir) {
  return path.join(dir, 'venv', 'Scripts', 'python.exe');
}

function readRuntime(dir) {
  try {
    const r = JSON.parse(fs.readFileSync(path.join(dir, 'runtime.json'), 'utf8'));
    return fs.existsSync(r.python) ? r : null;
  } catch {
    return null;
  }
}

function freeGiB(dir) {
  try {
    let p = dir;
    while (!fs.existsSync(p) && path.dirname(p) !== p) p = path.dirname(p);
    const s = fs.statfsSync(p);
    return Math.round((s.bavail * s.bsize) / 2 ** 30 * 10) / 10;
  } catch {
    return null;
  }
}

function driverMajor() {
  const exe = fs.existsSync('C:\\Windows\\System32\\nvidia-smi.exe') ? 'C:\\Windows\\System32\\nvidia-smi.exe' : 'nvidia-smi';
  try {
    const out = execFileSync(exe, ['--query-gpu=driver_version', '--format=csv,noheader'], { encoding: 'utf8', windowsHide: true, timeout: 15000 });
    const v = parseInt(String(out).trim().split(/\r?\n/)[0], 10);
    return Number.isFinite(v) ? v : null;
  } catch {
    return null;
  }
}

/* 流式下载并计算 sha256；不一致则删除。跟随最多 5 次重定向。 */
function download(url, dest, { sha256, size, onBytes, signal }, hops = 0) {
  return new Promise((resolve, reject) => {
    const mod = url.startsWith('https://') ? https : http;
    const req = mod.get(url, { timeout: 60000 }, res => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location && hops < 5) {
        res.resume();
        resolve(download(new URL(res.headers.location, url).toString(), dest, { sha256, size, onBytes, signal }, hops + 1));
        return;
      }
      if (res.statusCode !== 200) { res.resume(); reject(new Error(`下载失败：HTTP ${res.statusCode}（${url}）`)); return; }
      const tmp = dest + '.part';
      const out = fs.createWriteStream(tmp);
      const h = crypto.createHash('sha256');
      let n = 0;
      res.on('data', c => { h.update(c); n += c.length; if (onBytes) onBytes(n); });
      res.pipe(out);
      out.on('finish', () => {
        const got = h.digest('hex');
        if ((size && n !== size) || (sha256 && got !== sha256)) {
          fs.rmSync(tmp, { force: true });
          reject(new Error(`${path.basename(dest)} 的大小或 sha256 与固定值不一致，已删除（下载被篡改或不完整）`));
          return;
        }
        fs.renameSync(tmp, dest);
        resolve(got);
      });
      out.on('error', reject);
    });
    req.on('timeout', () => req.destroy(new Error('下载超时')));
    req.on('error', reject);
    if (signal) signal.addEventListener('abort', () => req.destroy(new Error('已取消')), { once: true });
  });
}

class RuntimeInstaller {
  /**
   * opts.dir() → 运行时目录；opts.lockFile → 锁文件路径；opts.onChange(info)；
   * 测试可注入 opts.run(cmd, args, env) → Promise<{code, out}>、opts.download、opts.driver()、opts.free()、opts.filesBase、opts.extract。
   */
  constructor(opts) {
    this.opts = opts;
    this.state = 'idle';            // idle · running · done · failed · cancelled
    this.step = '';
    this.error = '';
    this.log = [];
    this.bytes = 0;
    this.abort = null;
    this.child = null;
  }

  info() {
    const dir = this.opts.dir();
    const installed = readRuntime(dir);
    return {
      dir, installed, current: Boolean(installed && installed.id === RUNTIME_ID), runtimeId: RUNTIME_ID,
      state: this.state, step: this.step, error: this.error, log: this.log.slice(-60), bytes: this.bytes, uvSize: UV.size,
      free_gib: (this.opts.free || freeGiB)(dir), need_gib: NEED_GIB, driver: (this.opts.driver || driverMajor)(),
      min_driver: MIN_DRIVER, mirrors: Object.fromEntries(Object.entries(MIRRORS).map(([k, v]) => [k, v.label])),
    };
  }

  _emit() { if (this.opts.onChange) this.opts.onChange(this.info()); }

  _push(text) {
    for (const line of String(text).split(/\r?\n/)) if (line.trim()) this.log.push(line.slice(0, 400));
    if (this.log.length > 400) this.log.splice(0, this.log.length - 400);
  }

  _run(cmd, args, env) {
    if (this.opts.run) return this.opts.run(cmd, args, env);
    return new Promise((resolve, reject) => {
      const child = spawn(cmd, args, { env: { ...process.env, ...env }, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
      this.child = child;
      let out = '';
      const on = d => { out += d; this._push(d); this._emit(); };
      child.stdout.on('data', on);
      child.stderr.on('data', on);
      child.on('error', reject);
      child.on('exit', code => { this.child = null; resolve({ code, out }); });
    });
  }

  async _cmd(label, cmd, args, env) {
    this.step = label;
    this._push(`> ${label}`);
    this._emit();
    const r = await this._run(cmd, args, env);
    if (this.state === 'cancelled') throw new Error('已取消');
    if (r.code !== 0) throw new Error(`${label}失败（退出码 ${r.code}）`);
    return r.out;
  }

  async install({ mirror = 'pypi' } = {}) {
    if (this.state === 'running') return this.info();
    const m = MIRRORS[mirror] || MIRRORS.pypi;
    const dir = this.opts.dir();
    Object.assign(this, { state: 'running', error: '', log: [], bytes: 0, step: '检查' });
    this.abort = new AbortController();
    this._emit();
    try {
      if (process.platform !== 'win32' && !this.opts.run) throw new Error('当前只支持 Windows + NVIDIA');
      const drv = (this.opts.driver || driverMajor)();
      if (drv == null) throw new Error('没有检测到 NVIDIA 显卡驱动（nvidia-smi）');
      if (drv < MIN_DRIVER) throw new Error(`NVIDIA 驱动版本 ${drv} 太旧：CUDA 13.0 需要 ${MIN_DRIVER} 以上，请先更新驱动`);
      const free = (this.opts.free || freeGiB)(dir);
      if (free != null && free < NEED_GIB) throw new Error(`${dir} 所在磁盘剩余 ${free} GiB，至少需要 ${NEED_GIB} GiB：请换一个目录`);
      const lock = this.opts.lockFile;
      if (!lock || !fs.existsSync(lock)) throw new Error('安装包缺少锁文件 requirements-win-cu130.lock');

      fs.mkdirSync(path.join(dir, 'downloads'), { recursive: true });
      const env = { UV_CACHE_DIR: path.join(dir, 'cache'), UV_PYTHON_INSTALL_DIR: path.join(dir, 'python'), UV_NO_CONFIG: '1' };
      const uvExe = path.join(dir, 'bin', 'uv.exe');
      if (!fs.existsSync(uvExe)) {
        this.step = `下载 uv ${UV.version}`;
        this._push(`> ${this.step}（${m.label}，${(UV.size / 2 ** 20).toFixed(1)} MB，按 sha256 核对）`);
        this._emit();
        const whl = path.join(dir, 'downloads', UV.file);
        await (this.opts.download || download)((this.opts.filesBase || m.files) + UV.path, whl, {
          sha256: UV.sha256, size: UV.size, signal: this.abort.signal, onBytes: n => { this.bytes = n; },
        });
        const ex = path.join(dir, 'downloads', 'uv-extract');
        fs.rmSync(ex, { recursive: true, force: true });
        fs.mkdirSync(ex, { recursive: true });
        if (this.opts.extract) await this.opts.extract(whl, ex);
        else execFileSync('C:\\Windows\\System32\\tar.exe', ['-xf', whl, '-C', ex], { windowsHide: true });
        const inner = path.join(ex, `uv-${UV.version}.data`, 'scripts', 'uv.exe');
        if (!fs.existsSync(inner)) throw new Error('uv wheel 里没有 uv.exe');
        fs.mkdirSync(path.dirname(uvExe), { recursive: true });
        fs.copyFileSync(inner, uvExe);
        fs.rmSync(ex, { recursive: true, force: true });
      }
      await this._cmd(`安装 Python ${PYTHON_VERSION}`, uvExe, ['python', 'install', PYTHON_VERSION], env);
      const py = venvPython(dir);
      await this._cmd('创建虚拟环境', uvExe, ['venv', path.join(dir, 'venv'), '--python', PYTHON_VERSION,
        '--python-preference', 'only-managed', '--allow-existing'], env);
      await this._cmd('安装 torch / transformers（约 3 GB，按锁文件 sha256 核对）', uvExe, ['pip', 'install', '--python', py,
        '--require-hashes', '--no-deps', '-r', lock, '--index-url', m.index, '--extra-index-url', TORCH_INDEX,
        '--index-strategy', 'unsafe-best-match'], env);
      const out = await this._cmd('检测 torch / CUDA', py, ['-c', PROBE], {});
      const probe = JSON.parse(String(out).trim().split(/\r?\n/).pop());
      if (!probe.cuda_available || !probe.transformers) {
        throw new Error(`环境已装好但检测不通过：${probe.torch_error || probe.transformers_error || 'torch 检测不到 CUDA'}`);
      }
      const rec = { id: RUNTIME_ID, python: py, python_version: PYTHON_VERSION, uv: UV.version, mirror, torch: probe.torch,
        cuda: probe.cuda, transformers: probe.transformers, lock_sha256: crypto.createHash('sha256').update(fs.readFileSync(lock)).digest('hex'),
        installed_at: new Date().toISOString() };
      fs.writeFileSync(path.join(dir, 'runtime.json'), JSON.stringify(rec, null, 1));
      this._push(`✓ 计算组件可用：torch ${probe.torch} · CUDA ${probe.cuda} · transformers ${probe.transformers}`);
      this.state = 'done';
      this.step = '';
    } catch (e) {
      if (this.state !== 'cancelled') { this.state = 'failed'; this.error = e.message; }
      this._push(this.state === 'cancelled' ? '已取消（已下载的包留在缓存里，重装时复用）' : `✗ ${e.message}`);
    } finally {
      this.abort = null;
      this._emit();
    }
    return this.info();
  }

  cancel() {
    if (this.state !== 'running') return this.info();
    this.state = 'cancelled';
    if (this.abort) this.abort.abort();
    if (this.child) {
      try { execFileSync('taskkill.exe', ['/pid', String(this.child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' }); } catch { /* 已退出 */ }
    }
    this._emit();
    return this.info();
  }
}

module.exports = { RuntimeInstaller, readRuntime, venvPython, download, RUNTIME_ID, UV, MIRRORS, PYTHON_VERSION, TORCH_INDEX, MIN_DRIVER };
