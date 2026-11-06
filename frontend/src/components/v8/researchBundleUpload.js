const MiB = 1024 * 1024;
const BLOCK = 512;
const SHA = /^[0-9a-f]{64}$/;
const INDEX = '/api/traces/research-space';
const ARTIFACT = '/api/research-library/artifacts/';
const decoder = new TextDecoder('utf-8', { fatal: true });
const memberName = /^(?:MANIFEST\.json|objects\/[0-9a-f]{64}\.json)$/;

function field(header, start, end) {
  return decoder.decode(header.subarray(start, end)).split('\0', 1)[0].trim();
}

function octal(header, start, end) {
  const value = field(header, start, end);
  if (!/^[0-7]+$/.test(value)) throw new Error('研究包的 tar 长度或校验字段无效。');
  return Number.parseInt(value, 8);
}

// The release builder writes a POSIX tar, with optional PAX mtime records.
// Keep file contents in memory. No archive path is written to disk.
function readTar(bytes) {
  const entries = new Map();
  let offset = 0;
  let hasPax = false;
  while (offset + BLOCK <= bytes.length) {
    const header = bytes.subarray(offset, offset + BLOCK);
    if (header.every(byte => byte === 0)) {
      if (hasPax || bytes.length - offset < 2 * BLOCK || !bytes.subarray(offset).every(byte => byte === 0))
        throw new Error('研究包的 tar 结束记录无效。');
      return entries;
    }
    const checksum = header.reduce((sum, byte, index) => sum + (index >= 148 && index < 156 ? 32 : byte), 0);
    if (field(header, 257, 263) !== 'ustar' || checksum !== octal(header, 148, 156))
      throw new Error('研究包的 tar 文件头校验失败。');
    const size = octal(header, 124, 136);
    const start = offset + BLOCK;
    const next = start + Math.ceil(size / BLOCK) * BLOCK;
    if (size > 16 * MiB || next > bytes.length) throw new Error('研究包成员超出 16 MiB 或数据不完整。');
    const raw = bytes.subarray(start, start + size);
    const type = header[156];
    offset = next;
    if (type === 120) {
      const text = decoder.decode(raw);
      if (hasPax || !/^\d+ mtime=-?\d+(?:\.\d+)?\n$/.test(text) || Number(text.split(' ', 1)[0]) !== size)
        throw new Error('研究包含有不支持的 tar 扩展。');
      hasPax = true;
      continue;
    }
    const name = field(header, 0, 100);
    if ((type !== 48 && type !== 0) || field(header, 345, 500) || !memberName.test(name) || entries.has(name))
      throw new Error('研究包含有未知、重复或非普通文件成员。');
    hasPax = false;
    entries.set(name, raw);
  }
  throw new Error('研究包的 tar 数据不完整。');
}

async function unpack(file) {
  if (!file.size || file.size > 64 * MiB) throw new Error('请选择不超过 64 MiB 的研究包。');
  if (typeof DecompressionStream === 'undefined' || !globalThis.crypto?.subtle)
    throw new Error('请使用支持安全解压与 SHA256 的新版客户端。');
  let total = 0;
  const bounded = new TransformStream({
    transform(chunk, controller) {
      total += chunk.byteLength;
      if (total > 128 * MiB) throw new Error('研究包解压后超过 128 MiB。');
      controller.enqueue(chunk);
    },
  });
  const stream = file.stream().pipeThrough(new DecompressionStream('gzip')).pipeThrough(bounded);
  return readTar(new Uint8Array(await new Response(stream).arrayBuffer()));
}

export async function prepareResearchBundle(file) {
  const entries = await unpack(file);
  const rawManifest = entries.get('MANIFEST.json');
  if (!rawManifest || rawManifest.length > 4 * MiB) throw new Error('研究包缺少有效的 MANIFEST.json。');
  const manifest = JSON.parse(decoder.decode(rawManifest));
  if (manifest?.schema !== 'ai2050_research_upload.v1' || !Array.isArray(manifest.objects) ||
      !manifest.objects.length || manifest.objects.length !== entries.size - 1 ||
      !Array.isArray(manifest.studies) || !manifest.studies.length)
    throw new Error('研究包清单无效或与文件数量不符。');
  const objects = new Map();
  for (const { sha256: sha } of manifest.objects) {
    const raw = entries.get('objects/' + sha + '.json');
    if (!SHA.test(sha) || objects.has(sha) || !raw) throw new Error('研究包清单存在无效或重复的对象。');
    const actual = new Uint8Array(await crypto.subtle.digest('SHA-256', raw));
    const hex = Array.from(actual, byte => byte.toString(16).padStart(2, '0')).join('');
    if (hex !== sha) throw new Error('研究对象 SHA256 不匹配：' + sha.slice(0, 12));
    if (JSON.parse(decoder.decode(raw))?.kind !== 'real') throw new Error('研究包只能发布真实产物。');
    objects.set(sha, raw);
  }
  for (const study of manifest.studies) {
    if (typeof study?.run_id !== 'string' || !study.run_id.trim() ||
        !SHA.test(study.model_snapshot) || !objects.has(study.report_sha256))
      throw new Error('研究包缺少有效的工作包身份或报告引用。');
  }
  return { objects, studies: manifest.studies };
}

export async function uploadResearchBundle(bundle, request, progress) {
  const initial = await request(INDEX);
  if (initial?.schema !== 'research_space_index.v1' || !Array.isArray(initial.artifacts))
    throw new Error('当前中心没有可用的研究归档索引。');
  const existing = new Set(initial.artifacts.map(row => row.sha));
  let done = 0;
  let uploaded = 0;
  for (const [sha, raw] of bundle.objects) {
    if (!existing.has(sha)) {
      // Send exact JSON bytes so the server preserves the archived number representations.
      const response = await request('/api/research-library/ingest', { method: 'POST', rawBody: raw });
      if (response?.sha256 !== sha) throw new Error('中心返回的对象哈希与研究包不符：' + sha.slice(0, 12));
      uploaded++;
    }
    progress(++done, bundle.objects.size);
  }
  const published = await request(INDEX);
  for (const study of bundle.studies) {
    const entry = published?.studies?.find(item => item.run_id === study.run_id && item.model_snapshot === study.model_snapshot);
    if (!entry || ['report_sha256', 'binding_sha256', 'workpackage_sha256'].some(key => study[key] && entry[key] !== study[key]))
      throw new Error('中心尚未发布工作包：' + study.run_id);
    const report = await request(ARTIFACT + study.report_sha256);
    if (report?.run_id !== study.run_id || report?.model_snapshot !== study.model_snapshot)
      throw new Error('公开报告的研究或模型快照不一致：' + study.run_id);
  }
  return { total: done, uploaded, studies: bundle.studies };
}
