/* 客户端本地数据（datapack.v1）：手动下载的数据包存在浏览器 IndexedDB，页面优先读本地，命中就不访问服务器。
   - 数据包：GET /api/platform/datapack（gzip JSON），条目以规范化请求路径为键（与 deploy/datapack.py 的 canonical 相同）。
   - 按需缓存：不进数据包的大数据（实验响应的合并统计与规律分析）第一次打开时在线读取，存进本地并标记数据包版本；
     数据包更新后旧缓存不再使用。
   - 单元卡与管理类接口不缓存，打开时在线读取一次。
   浏览器禁用 IndexedDB 时退化为只在内存里保存（关闭页面即失效）。 */

const DB_NAME = 'ai2050-data';
const STORE = 'kv';
const ON_DEMAND = [/^\/api\/agg\/[^/]+\/v2$/, /^\/api\/analysis\/[^/]+$/];

const state = { ready: false, version: null, createdAt: null, bytes: 0, entries: new Map(), tick: 0 };
const listeners = new Set();
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
  return { ready: state.ready, version: state.version, createdAt: state.createdAt, bytes: state.bytes, n: state.entries.size };
}

/* 规范化请求路径：路径 + 按键排序、解码后的查询参数（与服务端相同）。 */
export function canonical(path) {
  const i = path.indexOf('?');
  if (i < 0) return path;
  const params = [...new URLSearchParams(path.slice(i + 1)).entries()];
  params.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0)));
  return path.slice(0, i) + (params.length ? '?' + params.map(([k, v]) => `${k}=${v}`).join('&') : '');
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
  if (!db) return undefined;
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

/* 启动时从本地载入数据包（只做一次）。 */
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
      state.ready = true;
      emit();
    })();
  }
  return initPromise;
}

export function getLocal(path) {
  return state.entries.get(canonical(path));
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

/* 下载并保存数据包；onProgress(已下载字节, 总字节)。 */
export async function downloadDataPack(apiBase, onProgress) {
  const response = await fetch(`${apiBase}/api/platform/datapack`, { cache: 'no-store' });
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
  const doc = JSON.parse(text);
  if (doc.schema !== 'datapack.v1') throw new Error('不是 datapack.v1 数据包');
  const old = state.version;
  await idb('put', 'pack', { version: doc.version, created_at: doc.created_at, bytes: got, entries: doc.entries });
  state.version = doc.version;
  state.createdAt = doc.created_at;
  state.bytes = got;
  state.entries = new Map(Object.entries(doc.entries));
  if (old && old !== doc.version) clearOldOnDemand(doc.version);
  emit();
  return packInfo();
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
  emit();
}
