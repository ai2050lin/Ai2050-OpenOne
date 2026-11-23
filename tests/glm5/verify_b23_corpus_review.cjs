// Compare the exact bundled lookup and full-field bytes to the native archive.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const esbuild = require('../../frontend/node_modules/esbuild');

const root = path.resolve(__dirname, '../..');
const preview = path.join(root, 'research/glm5/docs/previews/client-review');
const dir = path.join(root, 'tests/glm5/result/goodfire_middle_20261113_v2/qwen14');
const read = name => JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8'));

async function verify() {
  const {outputFiles} = esbuild.buildSync({
    entryPoints: [path.join(preview, 'b23-corpus.js')], write: false, bundle: true,
    platform: 'browser', format: 'esm',
  });
  const {default: corpus} = await import('data:text/javascript;base64,' + Buffer.from(outputFiles[0].contents).toString('base64'));
  const manifest = read('manifest.json');
  const docs = read('documents.json'), stats = read('B23-profile.json').fields.A, cues = read('B23-cues.json').units;
  assert.equal(corpus.snapshot, manifest.model_snapshot);
  assert.equal(corpus.run, manifest.run_id);
  assert.equal(corpus.block, 23);
  assert.deepEqual(corpus.positions, docs.positions);
  assert.deepEqual(corpus.counts, {documents: 1024, readouts: 8, tokens: 8192});
  for (const invalid of [-1, 17408, null, '7708', 0.5]) assert.equal(corpus.lookup(invalid), null);
  const fd = fs.openSync(path.join(dir, 'B23-A.npy'), 'r');
  const cache = new Map(), scalar = Buffer.alloc(4);
  try {
    const header = Buffer.alloc(128);
    assert.equal(fs.readSync(fd, header, 0, 128, 0), 128);
    assert.equal(header[6], 1);
    const offset = 10 + header.readUInt16LE(8);
    assert.equal(offset, 128);
    assert.match(header.toString('latin1', 10), /'descr': '<u2'.*'fortran_order': False.*'shape': \(1024, 8, 17408\)/);
    const width = 17408;
    assert.equal(corpus.width, width);
    function rowAt(flatIndex) {
      if (!cache.has(flatIndex)) {
        const bytes = Buffer.alloc(2 * width);
        assert.equal(fs.readSync(fd, bytes, 0, bytes.length, offset + bytes.length * flatIndex), bytes.length);
        cache.set(flatIndex, bytes);
      }
      return cache.get(flatIndex);
    }
    for (let unit = 0; unit < width; unit++) {
      const card = corpus.lookup(unit);
      assert.equal(card.run, manifest.run_id);
      assert.equal(card.mean, stats.mean[unit]);
      assert.equal(card.std, stats.std[unit]);
      assert.deepEqual(card.words, cues[unit].positive_keywords.slice(0, 3));
      assert.equal(card.confirm, cues[unit].repeat.confirm.hits);
      assert.equal(card.holdout, cues[unit].repeat.holdout.hits);
      for (const [pole, sample] of card.samples.entries()) {
        const side = pole === 0 ? 'positive_indices' : 'negative_indices';
        const flat = stats.holdout[side][unit][0];
        const doc = docs.docs[Math.floor(flat / 8)], position = docs.positions[flat % 8];
        const raw = rowAt(flat);
        scalar.writeUInt16LE(raw.readUInt16LE(2 * unit), 2);
        assert.equal(sample.activation, scalar.readFloatLE(0));
        assert.equal(sample.di, doc.doc_index);
        assert.equal(sample.document, doc.doc_id);
        assert.equal(sample.position, position);
        assert.equal(sample.tokenId, doc.tokens[position]);
        assert.equal(doc.split, 'holdout');
        assert.equal(sample.source.revision, doc.revision);
        assert.equal(sample.source.license, doc.license);
      }
    }
    assert.deepEqual(corpus.rows.map(row => row.key), ['positive', 'negative']);
    const focus = corpus.lookup(7708);
    const checkedRows = corpus.rows.map((row, i) => {
      const {bits, ...identity} = row;
      const raw = rowAt(row.readoutIndex), sample = focus.samples[i];
      assert.equal(row.readoutIndex, row.documentIndex * 8 + row.slot);
      assert.deepEqual(Buffer.from(bits, 'base64'), raw);
      assert.equal(row.doc, sample.document);
      assert.equal(row.position, sample.position);
      assert.equal(row.text, sample.text);
      assert.equal(row.activation, sample.activation);
      return {...identity, row_sha256: crypto.createHash('sha256').update(raw).digest('hex')};
    });
    return {
      verified_at: new Date().toISOString(), snapshot: corpus.snapshot, run: corpus.run, manifest: corpus.source,
      shape: [1024, 8, width], native_dtype: 'little-endian BF16 bits (uint16)',
      profiles_checked: width, pole_activations_checked: 2 * width, current_unit: 7708,
      checked_rows: checkedRows, candidate_terms: focus.words, mean: focus.mean, std: focus.std,
      confirm_keyword_hits: focus.confirm, holdout_keyword_hits: focus.holdout, each_denominator: 4,
      archive_sha256: corpus.hashes,
      scope: 'Published holdout record review. Two complete fixed-readout rows from the natural corpus, selected for U7708. Model inference was not run.',
    };
  } finally {
    fs.closeSync(fd);
  }
}

verify().then(report => console.log(JSON.stringify(report))).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
