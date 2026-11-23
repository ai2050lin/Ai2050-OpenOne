import packed from './b23-corpus-data.json';

function unpack(base64, Type) {
  const bytes = Uint8Array.from(atob(base64), ch => ch.charCodeAt(0));
  return new Type(bytes.buffer);
}

const native = new DataView(new ArrayBuffer(4));
function bf16(word) {
  native.setUint32(0, word * 65536, true);
  return native.getFloat32(0, true);
}

let decoded;
function arrays() {
  decoded ??= {
    means: unpack(packed.stats.mean, Float64Array),
    deviations: unpack(packed.stats.std, Float64Array),
    repeats: unpack(packed.stats.repeat, Uint8Array),
    keywords: unpack(packed.stats.terms, Uint32Array),
    poles: packed.stats.poles.map(([ids, values]) => [
      unpack(ids, Uint16Array), unpack(values, Uint16Array),
    ]),
  };
  return decoded;
}

function lookup(index) {
  if (!Number.isInteger(index) || index < 0 || index >= packed.width) return null;
  const {means, deviations, repeats, keywords, poles} = arrays();
  const samples = poles.map(([ids, values]) => {
    const [di, position, tokenId, text] = packed.examples[ids[index]];
    const source = packed.docs[di];
    return {di, document: source.doc_id, source, position, tokenId, text, activation: bf16(values[index])};
  });
  return {
    state: 'atlas', run: packed.run, hashes: packed.stats.hashes,
    mean: means[index], std: deviations[index],
    words: Array.from(keywords.subarray(3 * index, 3 * index + 3), id => packed.words[id]).filter(Boolean),
    confirm: repeats[index] >> 4, holdout: repeats[index] & 15, samples,
  };
}

const {model, snapshot, block, width, run, counts, rows, source, hashes, positions} = packed;
export default {model, snapshot, block, width, run, counts, rows, source, hashes, positions, lookup};
