// 计算组件安装器的测试（node --test electron/）：下载核对、驱动与磁盘门、命令顺序与参数、检测失败、取消。
'use strict';
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const { test } = require('node:test');

const R = require('./runtimeInstaller');

function tmpdir() { return fs.mkdtempSync(path.join(os.tmpdir(), 'ai2050-rt-')); }

function serve(routes) {
  return new Promise(resolve => {
    const srv = http.createServer((req, res) => {
      const r = routes[req.url];
      if (!r) { res.statusCode = 404; res.end(); return; }
      if (r.redirect) { res.statusCode = 302; res.setHeader('Location', r.redirect); res.end(); return; }
      res.end(r.body);
    }).listen(0, '127.0.0.1', () => resolve({ srv, base: `http://127.0.0.1:${srv.address().port}` }));
  });
}

test('download verifies sha256 and size, follows redirects', async () => {
  const body = Buffer.from('uv wheel bytes '.repeat(1000));
  const sha = crypto.createHash('sha256').update(body).digest('hex');
  const { srv, base } = await serve({ '/a': { redirect: '/b' }, '/b': { body } });
  const dir = tmpdir();
  try {
    assert.equal(await R.download(`${base}/a`, path.join(dir, 'ok.whl'), { sha256: sha, size: body.length }), sha);
    assert.deepEqual(fs.readFileSync(path.join(dir, 'ok.whl')), body);
    await assert.rejects(R.download(`${base}/b`, path.join(dir, 'bad.whl'), { sha256: '0'.repeat(64) }), /sha256/);
    assert.equal(fs.existsSync(path.join(dir, 'bad.whl')), false);
    assert.equal(fs.existsSync(path.join(dir, 'bad.whl.part')), false);
    await assert.rejects(R.download(`${base}/b`, path.join(dir, 'short.whl'), { size: body.length + 1 }), /大小/);
  } finally {
    srv.close();
  }
});

function fakeInstaller(dir, { driver = 610, free = 100, probe = { torch: '2.13.0+cu130', cuda: '13.0', cuda_available: true, transformers: '5.14.1' }, hang = false } = {}) {
  const lock = path.join(dir, 'req.lock');
  fs.writeFileSync(lock, 'torch==2.13.0+cu130 --hash=sha256:00\n');
  const calls = [];
  const inst = new R.RuntimeInstaller({
    dir: () => path.join(dir, 'rt'), lockFile: lock, driver: () => driver, free: () => free,
    download: async (url, dest) => { calls.push(['download', url]); fs.writeFileSync(dest, 'whl'); },
    extract: async (whl, ex) => {
      const p = path.join(ex, `uv-${R.UV.version}.data`, 'scripts');
      fs.mkdirSync(p, { recursive: true });
      fs.writeFileSync(path.join(p, 'uv.exe'), 'exe');
    },
    run: async (cmd, args, env) => {
      calls.push([path.basename(cmd), ...args, env]);
      if (hang) await new Promise(r => setTimeout(r, 200));
      if (args[0] === 'venv') { fs.mkdirSync(path.dirname(R.venvPython(path.join(dir, 'rt'))), { recursive: true }); fs.writeFileSync(R.venvPython(path.join(dir, 'rt')), ''); }
      if (args[0] === '-c') return { code: 0, out: 'noise\n' + JSON.stringify(probe) };
      return { code: 0, out: '' };
    },
  });
  return { inst, calls, lock };
}

test('install runs the pinned steps in order with hash-locked pip', async () => {
  const dir = tmpdir();
  const { inst, calls, lock } = fakeInstaller(dir);
  const info = await inst.install({ mirror: 'tuna' });
  assert.equal(info.state, 'done', info.error);
  assert.equal(calls[0][0], 'download');
  assert.ok(calls[0][1].startsWith('https://pypi.tuna.tsinghua.edu.cn/packages/') && calls[0][1].endsWith(R.UV.file));
  const [py, venv, pip, probe] = calls.slice(1);
  assert.deepEqual(py.slice(0, 4), ['uv.exe', 'python', 'install', R.PYTHON_VERSION]);
  assert.equal(venv[1], 'venv');
  assert.ok(venv.includes('only-managed'));
  for (const a of ['--require-hashes', '--no-deps', lock, R.TORCH_INDEX, 'https://pypi.tuna.tsinghua.edu.cn/simple']) assert.ok(pip.includes(a), a);
  assert.equal(pip.at(-1).UV_CACHE_DIR, path.join(dir, 'rt', 'cache'));   // 缓存不落系统盘
  assert.equal(probe[0], 'python.exe');
  assert.ok(info.installed && info.current);
  assert.equal(info.installed.torch, '2.13.0+cu130');
  assert.equal(info.installed.lock_sha256, crypto.createHash('sha256').update(fs.readFileSync(lock)).digest('hex'));
  const again = fakeInstaller(dir);                                      // uv 已在：不再下载
  await again.inst.install();
  assert.notEqual(again.calls[0][0], 'download');
});

test('old driver and small disk stop before downloading', async () => {
  const a = fakeInstaller(tmpdir(), { driver: 575 });
  const ia = await a.inst.install();
  assert.equal(ia.state, 'failed');
  assert.match(ia.error, /驱动/);
  assert.equal(a.calls.length, 0);
  const b = fakeInstaller(tmpdir(), { driver: null });
  assert.match((await b.inst.install()).error, /NVIDIA/);
  const c = fakeInstaller(tmpdir(), { free: 3 });
  assert.match((await c.inst.install()).error, /磁盘/);
  assert.equal(c.calls.length, 0);
});

test('probe without CUDA is a failure and writes no runtime.json', async () => {
  const dir = tmpdir();
  const { inst } = fakeInstaller(dir, { probe: { torch: '2.13.0', cuda_available: false, transformers: '5.14.1' } });
  const info = await inst.install();
  assert.equal(info.state, 'failed');
  assert.match(info.error, /CUDA/);
  assert.equal(info.installed, null);
  assert.equal(fs.existsSync(path.join(dir, 'rt', 'runtime.json')), false);
});

test('cancel stops the install', async () => {
  const { inst } = fakeInstaller(tmpdir(), { hang: true });
  const p = inst.install();
  await new Promise(r => setTimeout(r, 50));
  inst.cancel();
  const info = await p;
  assert.equal(info.state, 'cancelled');
  assert.equal(info.installed, null);
});
