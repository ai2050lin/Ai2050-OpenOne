/* 本地研究包 / 谱图包读取（respack.v1 / atlaspack.v1）：在浏览器内解 zip，不上传、不依赖服务器。
   只支持 zip 的「存储」与「deflate」两种方式（中心生成的包就是这两种）；解压用浏览器自带的 DecompressionStream。 */

function findEocd(dv, len) {
  for (let i = len - 22; i >= Math.max(0, len - 65557); i -= 1) {
    if (dv.getUint32(i, true) === 0x06054b50) return i;
  }
  return -1;
}

export function readZipIndex(buffer) {
  const u8 = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  const eocd = findEocd(dv, u8.length);
  if (eocd < 0) throw new Error('不是 zip 文件');
  const count = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const dec = new TextDecoder();
  const files = {};
  for (let k = 0; k < count; k += 1) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('zip 目录损坏');
    const method = dv.getUint16(p + 10, true);
    const csize = dv.getUint32(p + 20, true);
    const nlen = dv.getUint16(p + 28, true);
    const elen = dv.getUint16(p + 30, true);
    const clen = dv.getUint16(p + 32, true);
    const off = dv.getUint32(p + 42, true);
    const name = dec.decode(u8.subarray(p + 46, p + 46 + nlen));
    const start = off + 30 + dv.getUint16(off + 26, true) + dv.getUint16(off + 28, true);
    files[name] = { method, data: u8.subarray(start, start + csize) };
    p += 46 + nlen + elen + clen;
  }
  return files;
}

export async function inflateEntry(entry) {
  if (entry.method === 0) return entry.data;
  if (entry.method !== 8) throw new Error(`不支持的压缩方式 ${entry.method}`);
  if (typeof DecompressionStream === 'undefined') throw new Error('浏览器不支持 DecompressionStream，请使用新版 Chrome / Edge / Firefox');
  const stream = new Blob([entry.data]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function readJson(files, name) {
  if (!files[name]) return null;
  return JSON.parse(new TextDecoder().decode(await inflateEntry(files[name])));
}

/* 读取一个本地包：返回 { kind: 'respack' | 'atlaspack', name, manifest, analysis?, agg?, atlas? } */
export async function loadLocalPack(file) {
  const files = readZipIndex(await file.arrayBuffer());
  const manifest = await readJson(files, 'manifest.json');
  if (!manifest) throw new Error('包里没有 manifest.json');
  if (manifest.schema === 'respack.v1') {
    const analysis = await readJson(files, 'analysis.json');
    if (!analysis) throw new Error('研究包里没有 analysis.json');
    return { kind: 'respack', name: file.name, manifest, analysis, agg: await readJson(files, 'agg.json') };
  }
  if (manifest.schema === 'atlaspack.v1') {
    const atlas = await readJson(files, 'atlas.json');
    if (!atlas) throw new Error('谱图包里没有 atlas.json');
    return { kind: 'atlaspack', name: file.name, manifest, atlas };
  }
  throw new Error(`不认识的包格式：${manifest.schema || '未知'}`);
}
