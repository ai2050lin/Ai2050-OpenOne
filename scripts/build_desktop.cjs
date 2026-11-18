'use strict';

const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { checkWrite, checkedDir, storageDir, StorageBlocked } = require('../frontend/electron/storagePolicy');

const frontend = path.resolve(__dirname, '../frontend');
const GiB = 2 ** 30;

try {
  const [mode, ...args] = process.argv.slice(2);
  if (!['pack', 'dist', 'dir'].includes(mode)) throw new Error('未知的客户端打包方式');
  const out = path.resolve(frontend, process.env.AI2050_DESKTOP_OUT_DIR || 'dist-installer');
  // The default staging and final outputs share a volume. Budget both before starting.
  checkWrite(out, 4 * GiB, '客户端安装包及暂存');
  const cache = checkedDir(storageDir('reusable', 'electron-builder'), GiB, '打包工具缓存');
  const temp = checkedDir(storageDir('staging', 'desktop'), GiB, '客户端打包暂存');
  if (mode !== 'pack') checkWrite(path.join(frontend, 'dist'), GiB / 2, '前端构建');
  const env = { ...process.env, TEMP: temp, TMP: temp, TMPDIR: temp, ELECTRON_BUILDER_CACHE: cache,
    ELECTRON_CACHE: checkedDir(storageDir('reusable', 'electron'), GiB, 'Electron 下载缓存'),
    npm_config_cache: checkedDir(storageDir('reusable', 'npm'), 0, '前端依赖缓存') };
  const run = (entry, argv) => {
    const script = path.join(frontend, 'node_modules', entry);
    const result = spawnSync(process.execPath, [script, ...argv], { cwd: frontend, env, stdio: 'inherit' });
    if (result.error) throw result.error;
    if (result.status !== 0) process.exit(result.status ?? 1);
  };
  if (mode !== 'pack') run('vite/bin/vite.js', ['build']);
  checkWrite(out, 2 * GiB, '客户端安装包');
  run('electron-builder/out/cli/cli.js', [...(mode === 'dist' ? ['--win', '--x64'] : ['--dir']), ...args,
    `--config.directories.output=${out}`]);
} catch (error) {
  console.error(error.message);
  if (error instanceof StorageBlocked && process.platform === 'win32' && process.stdin.isTTY && !process.env.CI) {
    const cmd = 'Add-Type -AssemblyName PresentationFramework; ' +
      '[void][System.Windows.MessageBox]::Show($env:AI2050_DISK_NOTICE, "AI2050 磁盘检查")';
    spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', cmd], {
      env: { ...process.env, AI2050_DISK_NOTICE: error.message }, windowsHide: true,
    });
  }
  process.exitCode = 1;
}
