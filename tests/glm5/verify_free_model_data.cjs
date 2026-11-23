// Engineering contract tests for free-model-data.js. Run --receipt to retain the verification record.
'use strict';
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const repo = path.resolve(__dirname, '../..');
const preview = path.join(repo, 'research/glm5/docs/previews');
const resultDir = path.join(__dirname, 'result');
const weightRoot = process.env.AI2050_PREVIEW_QWEN_ROOT || 'D:/develop/model/Qwen';
const glmRoot = process.env.AI2050_PREVIEW_GLM_ROOT || 'D:/develop/model/ZhipuAI/glm-4-9b-chat-hf';
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
const json = name => JSON.parse(fs.readFileSync(path.join(resultDir, name), 'utf8'));
global.window = global;
for (const name of ['model-records-data', 'model-explorer-data', 'explorer-system-field'])
  require(path.join(preview, name + '.js'));
require(path.join(repo, 'frontend/src/components/v8/data/nativeHeadChannels.js'));
require(path.join(preview, 'encoding-workbench-data.js'));
const adapterPath = path.join(preview, 'free-model-data.js');
require(adapterPath);
const api = FreeModelData;

// File API adapter retaining real random access and logging every requested interval.
function localFile(id, name) {
  const fd = fs.openSync(path.join(id === 'GLM-4-9B' ? glmRoot : path.join(weightRoot, id), name), 'r');
  const size = fs.fstatSync(fd).size;
  const lengths = [];
  return {
    name, size, lengths, close() { fs.closeSync(fd); },
    slice(a, b) {
      assert.ok(a >= 0 && a <= b && b <= size);
      lengths.push(b - a);
      return {async arrayBuffer() {
        const raw = Buffer.alloc(b - a);
        assert.equal(fs.readSync(fd, raw, 0, raw.length, a), raw.length);
        return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
      }};
    }
  };
}

function sharedGroupAndTrace() {
  const reference = json('workspace_full_native/source-check.json');
  const m = api.model('Qwen3-4B');
  const hashes = {};
  for (const field of [...api.fields(m.id, 'head', 26, 14), ...api.fields(m.id, 'ffn', 26, 9727)]) {
    const actual = sha(Buffer.from(field.values.buffer, field.values.byteOffset, field.values.byteLength));
    assert.equal(field.position, reference.absolute_position);
    assert.equal(field.tokenId, reference.token_id);
    assert.equal(field.snapshot, reference.model_snapshot);
    assert.equal(field.traceId, reference.trace_id);
    assert.equal(actual, reference.vectors[field.measure].returned_f32_le_sha256);
    hashes[field.measure] = actual;
  }
  for (let b = 0; b < m.layers; b++) {
    const groupValues = new Map();
    for (let h = 0; h < m.heads; h++) {
      const fields = api.fields(m.id, 'head', b, h);
      assert.deepEqual(fields.map(f => f.measure), ['head_z','head_q','head_k','head_v']);
      const a = fields[0].attention;
      assert.equal(a.weights.length, reference.absolute_position + 1);
      assert.equal(a.tokens.length, a.weights.length);
      assert.equal(a.tokenIds[a.queryPosition], reference.token_id);
      assert.ok(a.weights.every(x => x >= 0 && x <= 1));
      assert.ok(Math.abs(a.weights.reduce((sum, v) => sum + v, 0) - 1) < 0.0078125);
      for (const f of fields) {
        assert.equal(f.kvGroup, Math.floor(h / (m.heads / m.kvHeads)));
        assert.equal(f.values.length, m.headDim);
        assert.ok(f.values.every(Number.isFinite));
        if (!['head_k','head_v'].includes(f.measure)) continue;
        const key = f.kvGroup + f.measure;
        const prev = groupValues.get(key);
        if (prev) assert.deepEqual(prev, f.values);
        else groupValues.set(key, f.values);
      }
    }
    assert.equal(api.fields(m.id, 'ffn', b, m.ffn - 1)[0].values.length, m.ffn);
    assert.equal(api.fields(m.id, 'hidden', b, m.hidden - 1)[0].values.length, m.hidden);
  }
  assert.equal(api.fields(m.id, 'hidden', 0)[0].boundary, '词嵌入输出');
  assert.equal(api.fields(m.id, 'hidden', m.layers)[0].boundary, '最终归一化后');
  return {snapshot: m.snapshot, positions: [reference.absolute_position], blocks: m.layers, headsPerBlock: m.heads,
    ffnWidth: m.ffn, hiddenReadouts: m.layers + 1, attentionRows: m.layers * m.heads,
    sourceTokensPerAttentionRow: reference.absolute_position + 1, kvGroupSize: m.heads / m.kvHeads,
    existingReference: 'workspace_full_native/source-check.json', vectorSha256: hashes};
}

function archive14() {
  const m = api.model('Qwen3-14B');
  const heads = api.fields(m.id, 'head', 26, 36);
  const ffn = api.fields(m.id, 'ffn', 36, 2429);
  const hidden = api.fields(m.id, 'hidden', 40, m.hidden - 1);
  assert.deepEqual([heads.length, ffn.length, hidden.length], [3, 4, 4]);
  for (const field of [...heads, ...ffn, ...hidden]) {
    assert.ok(field.caseId && field.run && field.source && field.sha256);
    assert.ok(field.position >= 0 && field.tokenId >= 0);
    assert.equal(field.snapshot, m.snapshot);
  }
  assert.ok(heads.every(x => x.values.length === m.headDim && x.rows.length === 5));
  assert.ok(ffn.every(x => x.values.length === m.ffn));
  assert.ok(hidden.every(x => x.values.length === m.hidden && x.boundary.includes('归一化前')));
  assert.deepEqual(ffn.map(x => x.position), hidden.map(x => x.position));
  assert.equal(api.fields(m.id, 'head', 26, 35).length, 0);
  assert.equal(api.fields(m.id, 'ffn', 36, m.ffn).length, 0);
  return {snapshot: m.snapshot, headCases: heads.map(x => x.caseId), ffnPositions: ffn.map(x => x.position),
    hiddenFinalBoundary: hidden[0].boundary, missingHeadProducesEmpty: true};
}

async function weightChecks(m) {
  assert.equal(m.weightBytes, m.total * 2);
  assert.equal(m.physicalTensors, m.tied ? 398 : 443);
  assert.equal(api.tensor(m.id, 'lm_head.weight').alias, m.tied ? 'model.embed_tokens.weight' : null);
  for (const t of m.tensors) assert.ok(Number.isFinite(api.tile(m.id, t.name, 0, 0, 1, 1).values[0][0]));
  const earlier = json('phase2764_real_parameter_reads.json').models.find(x =>
    x.model === (m.id === 'GLM-4-9B' ? 'glm-4-9b-chat-hf' : m.id));
  for (const t of earlier.last_layer_parameter_checks)
    assert.equal(api.tile(m.id, t.name, ...t.start, 1, 1).values[0][0], t.value);
  if (m.id === 'GLM-4-9B') {
    assert.equal(m.architecture, 'GlmForCausalLM');
    assert.deepEqual([m.layers, m.heads, m.kvHeads, m.hidden, m.ffn], [40, 32, 2, 4096, 13696]);
    assert.equal(m.fusedGateUp, true);
    assert.equal(api.tensor(m.id, 'model.layers.0.mlp.gate_proj.weight'), null);
    assert.deepEqual(api.tensor(m.id, 'model.layers.0.mlp.gate_up_proj.weight').shape, [27392, 4096]);
    assert.deepEqual(api.tensor(m.id, 'model.layers.0.self_attn.k_proj.bias').shape, [256]);
    assert.ok(Number.isFinite(api.tile(m.id, 'model.layers.26.mlp.gate_up_proj.weight', m.ffn+2429, 0, 1, 1).values[0][0]));
    assert.equal(api.fields(m.id, 'head', 26, 14).length, 0);
    assert.equal(api.fields(m.id, 'ffn', 36, 2429).length, 0);
    assert.equal(api.fields(m.id, 'hidden', 40, 0).length, 0);
    assert.equal(m.lensTokens.find(x => x.label === '红色').id, 101520);
  }

  const q = api.tensor(m.id, 'model.layers.26.self_attn.q_proj.weight');
  const tOut = api.tensor(m.id, 'lm_head.weight');
  const tNorm = api.tensor(m.id, 'model.norm.weight');
  const tokenRows = await Promise.all(m.lensTokenIds.map(token => api.readWeightVector(m.id, tOut.name, token)));
  const norm = await api.readWeightVector(m.id, tNorm.name);
  assert.equal(norm.values.length, m.hidden);
  assert.ok(tokenRows.every(x => x.values.length === m.hidden && x.kind === 'sample'));
  assert.equal(await api.readWeightVector(m.id, tOut.name, 42), null);
  const files = Array.from(new Set([q.file, tOut.file, tNorm.file]), n => localFile(m.id, n));
  try {
    await api.bindShards(m.id, files);
    assert.deepEqual(api.boundShards(m.id), files.map(f => f.name));
    const r = 77, c = m.hidden - 19;
    const imported = await api.readTile(m.id, q.name, r, c, 3, 4);
    assert.ok(imported.values.flat().every(Number.isFinite));
    assert.equal(imported.kind, 'local'); assert.equal(imported.verifiedContent, false);
    assert.ok(api.tile(m.id, q.name, r, c, 3, 4).values.flat().every(v => v === null));
    const lead = (m.id === 'Qwen3-14B' ? 36 : 14) * m.headDim;
    assert.deepEqual((await api.readTile(m.id, q.name, lead, 0, 8, 8)).values, api.tile(m.id, q.name, lead, 0, 8, 8).values);
    const last = await api.readTile(m.id, q.name, q.shape[0]-1, q.shape[1]-1, 8, 8);
    assert.deepEqual(last.actualShape, [1, 1]);
    assert.deepEqual(last.values, api.tile(m.id, q.name, q.shape[0]-1, q.shape[1]-1, 1, 1).values);
    assert.throws(() => api.tile(m.id, q.name, -1, 0, 1, 1));
    assert.throws(() => api.tile(m.id, q.name, q.shape[0], 0, 1, 1));
    assert.deepEqual((await api.readWeightVector(m.id, tNorm.name)).values, norm.values);
    for (const [i, token] of m.lensTokenIds.entries()) {
      const actual = await api.readWeightVector(m.id, tOut.name, token);
      assert.ok(actual.kind === 'local');
      assert.deepEqual(actual.values, tokenRows[i].values);
    }
    assert.equal((await api.readWeightVector(m.id, tOut.name, 42)).values.length, m.hidden);
    await assert.rejects(api.bindShards(m.id, [{name: q.file, size: 19}]));
    const bad = files[0];
    const corrupted = {name: bad.name, size: bad.size, slice(a, b) {
      if (a !== 8) return bad.slice(a, b);
      return {async arrayBuffer() {
        const raw = await bad.slice(a, b).arrayBuffer();
        new Uint8Array(raw)[0] ^= 1;
        return raw;
      }};
    }};
    await assert.rejects(api.bindShards(m.id, [corrupted]), /指纹/);
    assert.deepEqual(api.boundShards(m.id), files.map(f => f.name));
    const largestReadBytes = Math.max(...files.flatMap(f => f.lengths));
    assert.ok(largestReadBytes < 1024 * 1024);
    api.clearShards(m.id);
    assert.equal(api.boundShards(m.id).length, 0);
    assert.equal((await api.readTile(m.id, q.name, r, c, 1, 1)).values[0][0], null);
    return {model: m.id, snapshot: m.snapshot, indexedTensorCount: m.physicalTensors, lookupNames: m.tensors.length,
      uniqueParameterCount: m.total, tiedEmbeddingAndReadout: m.tied, nativeDtype: 'BF16',
      lensTokens: m.lensTokenIds, lensRowsVerifiedAgainstNative: tokenRows.length,
      excerptSha256: m.excerptSha256, nativeFinalLayerReferencesChecked: earlier.last_layer_parameter_checks.length,
      matchedFinalCell: last.values[0][0], rejectedBadShapeAndHeader: true, maxSingleFileReadBytes: largestReadBytes};
  } finally {for (const f of files) f.close();}
}

(async () => {
  const report = {schema: 'free_model_data_engineering_check.v1', timestamp: new Date().toISOString(),
    adapterSha256: sha(fs.readFileSync(adapterPath)), inference: false, fullModelRehashed: false,
    archive4B: sharedGroupAndTrace(), archive14B: archive14(), weights: []};
  for (const model of api.models) report.weights.push(await weightChecks(model));
  if (process.argv.includes('--receipt')) {
    const target = path.join(resultDir, 'free_model_data_qa.json');
    fs.writeFileSync(target + '.new', JSON.stringify(report, null, 2) + '\n');
    fs.renameSync(target + '.new', target);
  }
  console.log('PASS · three model catalogues, indexed native vector reads, archival boundaries, all traced 4B heads and five independent vector hashes');
})().catch(error => {console.error(error); process.exitCode = 1;});
