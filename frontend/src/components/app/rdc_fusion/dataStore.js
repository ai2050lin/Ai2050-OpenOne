/* 客户端本地数据（datapack.v1）：手动下载的数据包存在浏览器 IndexedDB，页面优先读本地，命中就不访问服务器。
   - 主数据包：GET /api/platform/datapack（gzip JSON），条目以规范化请求路径为键（与 deploy/datapack.py 的 canonical 相同）。
   - 附加包：GET /api/platform/datapack/extra/{id}，体积大的部分按需手动下载：tm:<模式族>（实验响应 3D 页）、
     cards:<模型>（单元卡）。条目逐条存进 IndexedDB（x|<路径>），读取时按需取出，不常驻内存。
   - 在线网页版的按需缓存：实验响应的合并统计与规律分析第一次在线读取后存进本地（od|<主包版本>|<路径>）。
   浏览器禁用 IndexedDB 时退化为只在内存里保存（关闭页面即失效）。 */

const DB_NAME = 'ai2050-data';
const STORE = 'kv';
const ON_DEMAND = [/^\/api\/agg\/[^/]+\/v2$/, /^\/api\/analysis\/[^/]+$/];

const state = { ready: false, version: null, createdAt: null, bytes: 0, entries: new Map(), extras: {}, tick: 0 };
const listeners = new Set();
const memory = new Map();          // IndexedDB 不可用时的退路
let dbPromise = null;
let initPromise = null;

function emit() {
  state.tick += 1;
  listeners.forEach(fn => fn());
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function storeTick() {
  return state.tick;
}

export function packInfo() {
  return { ready: state.ready, version: state.version, createdAt: state.createdAt, bytes: state.bytes, n: state.entries.size,
           extras: state.extras, tick: state.tick };
}

/* 规范化请求路径：路径 + 按键排序、解码后的查询参数（与服务端相同）。 */
export function canonical(path) {
  const i = path.indexOf('?');
  if (i < 0) return path;
  const params = [...new URLSearchParams(path.slice(i + 1)).entries()];
  params.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0)));
  return path.slice(0, i) + (params.length ? '?' + params.map(([k, v]) => `${k}=${v}`).join('&') : '');
}

/* 一条请求应在哪个附加包里（与 deploy/datapack.py 的 pack_hint 相同）。 */
export function packHint(path) {
  const [p, q = ''] = path.split('?');
  const m = /^\/api\/(?:agg\/([^/]+)\/v2|analysis\/([^/]+))$/.exec(p);
  if (m) return `tm:${decodeURIComponent(m[1] || m[2])}`;
  if (p === '/api/neurons/unit' || p === '/api/neurons/layer') {
    const mk = new URLSearchParams(q).get('model_key');
    return mk ? `cards:${mk}` : null;
  }
  if (p === '/api/neurons/circuit') {
    const mk = new URLSearchParams(q).get('model_key');
    return mk ? `circuit:${mk}` : null;
  }
  return null;
}

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise(resolve => {
    try {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
  return dbPromise;
}

async function idb(op, key, value) {
  const db = await openDb();
  if (!db) {
    if (op === 'get') return memory.get(key);
    if (op === 'put') memory.set(key, value);
    else if (op === 'clear') memory.clear();
    else memory.delete(key);
    return true;
  }
  return new Promise(resolve => {
    try {
      const tx = db.transaction(STORE, op === 'get' ? 'readonly' : 'readwrite');
      const st = tx.objectStore(STORE);
      const req = op === 'get' ? st.get(key) : op === 'put' ? st.put(value, key) : op === 'clear' ? st.clear() : st.delete(key);
      req.onsuccess = () => resolve(op === 'get' ? req.result : true);
      req.onerror = () => resolve(undefined);
    } catch {
      resolve(undefined);
    }
  });
}

/* 一个事务里写入多条（附加包可能有上千条）。 */
async function idbPutMany(pairs, deleteKeys = []) {
  const db = await openDb();
  if (!db) {
    deleteKeys.forEach(k => memory.delete(k));
    pairs.forEach(([k, v]) => memory.set(k, v));
    return;
  }
  await new Promise(resolve => {
    try {
      const tx = db.transaction(STORE, 'readwrite');
      const st = tx.objectStore(STORE);
      deleteKeys.forEach(k => st.delete(k));
      pairs.forEach(([k, v]) => st.put(v, k));
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
      tx.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
}

/* 启动时从本地载入主数据包与附加包清单（只做一次）。 */
export function initDataStore() {
  if (!initPromise) {
    initPromise = (async () => {
      const pack = await idb('get', 'pack');
      if (pack?.version && pack.entries) {
        state.version = pack.version;
        state.createdAt = pack.created_at;
        state.bytes = pack.bytes || 0;
        state.entries = new Map(Object.entries(pack.entries));
      }
      state.extras = (await idb('get', 'xmeta')) || {};
      state.ready = true;
      emit();
    })();
  }
  return initPromise;
}

export function getLocal(path) {
  return state.entries.get(canonical(path));
}

/* 附加包里的条目（异步，从 IndexedDB 取）；没有时返回 undefined。 */
export async function getExtraEntry(path) {
  const id = packHint(path);
  if (!id || !state.extras[id]) return undefined;
  return idb('get', `x|${canonical(path)}`);
}

export function isOnDemand(path) {
  const p = path.split('?')[0];
  return ON_DEMAND.some(re => re.test(p));
}

export async function getOnDemand(path) {
  if (!state.version) return undefined;
  return idb('get', `od|${state.version}|${canonical(path)}`);
}

export async function putOnDemand(path, data) {
  if (!state.version || data?.pending) return;
  await idb('put', `od|${state.version}|${canonical(path)}`, data);
}

/* 下载一个 gzip JSON 数据包并解析；onProgress(已下载字节, 总字节)。 */
async function fetchPack(url, onProgress) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got, total);
  }
  const blob = new Blob(chunks);
  const text = await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).text();
  return { doc: JSON.parse(text), bytes: got };
}

/* 下载并保存主数据包。 */
export async function downloadDataPack(apiBase, onProgress) {
  const { doc, bytes } = await fetchPack(`${apiBase}/api/platform/datapack`, onProgress);
  if (doc.schema !== 'datapack.v1') throw new Error('不是 datapack.v1 数据包');
  const old = state.version;
  await idb('put', 'pack', { version: doc.version, created_at: doc.created_at, bytes, entries: doc.entries });
  state.version = doc.version;
  state.createdAt = doc.created_at;
  state.bytes = bytes;
  state.entries = new Map(Object.entries(doc.entries));
  if (old && old !== doc.version) clearOldOnDemand(doc.version);
  emit();
  return packInfo();
}

/* 下载并保存一个附加包（替换同一个包的旧条目）。 */
export async function downloadExtra(apiBase, id, onProgress) {
  const { doc, bytes } = await fetchPack(`${apiBase}/api/platform/datapack/extra/${encodeURIComponent(id)}`, onProgress);
  if (doc.schema !== 'datapack_extra.v1' || doc.id !== id) throw new Error('不是对应的附加包');
  const oldKeys = (await idb('get', `xkeys|${id}`)) || [];
  const keys = Object.keys(doc.entries).map(k => `x|${k}`);
  await idbPutMany([...Object.entries(doc.entries).map(([k, v]) => [`x|${k}`, v]), [`xkeys|${id}`, keys]],
    oldKeys.filter(k => !keys.includes(k)));
  state.extras = { ...state.extras, [id]: { version: doc.version, created_at: doc.created_at, bytes, n: keys.length } };
  await idb('put', 'xmeta', state.extras);
  emit();
  return state.extras[id];
}

async function clearOldOnDemand(keepVersion) {
  const db = await openDb();
  if (!db) return;
  try {
    const tx = db.transaction(STORE, 'readwrite');
    const req = tx.objectStore(STORE).openCursor();
    req.onsuccess = () => {
      const cur = req.result;
      if (!cur) return;
      if (String(cur.key).startsWith('od|') && !String(cur.key).startsWith(`od|${keepVersion}|`)) cur.delete();
      cur.continue();
    };
  } catch { /* 清理失败不影响使用 */ }
}

export async function clearDataStore() {
  await idb('clear');
  state.version = null;
  state.createdAt = null;
  state.bytes = 0;
  state.entries = new Map();
  state.extras = {};
  emit();
}
