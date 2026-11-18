'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const file = process.resourcesPath && path.join(process.resourcesPath, 'node', 'storage_policy.json');
const policy = JSON.parse(fs.readFileSync(file && fs.existsSync(file) ? file :
  path.join(__dirname, '..', '..', 'deploy', 'storage_policy.json'), 'utf8'));
const GiB = 2 ** 30;

function storageDir(kind, ...parts) {
  const root = process.env.AI2050_STORAGE_ROOT ||
    (process.platform === 'win32' ? policy.windows_root : path.join(os.homedir(), '.ai2050', 'storage'));
  return path.join(root, policy.directories[kind], ...parts);
}

function actualDestination(target) {
  let current = path.resolve(target);
  const missing = [];
  while (true) {
    try {
      fs.lstatSync(current);
      const existing = fs.realpathSync.native(current);
      return { existing, actual: path.join(existing, ...missing.reverse()) };
    } catch (error) {
      if (error.code !== 'ENOENT' || current === path.dirname(current)) throw error;
      // A dangling junction also reports ENOENT from realpath. Do not guess its volume.
      if (fs.lstatSync(current, { throwIfNoEntry: false })) throw error;
      missing.push(path.basename(current));
      current = path.dirname(current);
    }
  }
}

class StorageBlocked extends Error {
  constructor(reason, target, bytes, message) {
    super(`${message}（实际路径 ${target}，预计新增 ${(bytes / GiB).toFixed(2)} GiB）。请选择空间充足的项目盘。`);
    this.reason = reason;
  }
}

function checkWrite(target, bytes = 0, operation = '项目写入') {
  if (!Number.isSafeInteger(bytes) || bytes < 0) throw new TypeError('无效的写入预算');
  let resolved, free;
  try {
    resolved = actualDestination(target);
    free = fs.statfsSync(resolved.existing, { bigint: true });
  } catch {
    throw new StorageBlocked('unavailable', target, bytes, `${operation}的目标卷不可用`);
  }
  const { actual } = resolved;
  const windows = process.platform === 'win32';
  const drive = actual.replace(/^\\\\\?\\/, '').slice(0, 2).toUpperCase();
  if (windows && drive === policy.blocked_windows_drive) {
    throw new StorageBlocked('system_drive', actual, bytes, `${operation}将写入 C 盘，已暂停`);
  }
  const reserve = windows ? policy.windows_reserve_gib : policy.portable_reserve_gib;
  const available = free.bavail * free.bsize;
  if (available < BigInt(reserve * GiB) + BigInt(bytes)) {
    throw new StorageBlocked('space', actual, bytes,
      `${operation}空间不足，当前 ${(Number(available) / GiB).toFixed(2)} GiB，保留线 ${reserve} GiB`);
  }
  return actual;
}

function checkedDir(target, bytes, operation) {
  const actual = checkWrite(target, bytes, operation);
  fs.mkdirSync(actual, { recursive: true });
  return actual;
}

module.exports = { actualDestination, checkWrite, checkedDir, storageDir, StorageBlocked };
