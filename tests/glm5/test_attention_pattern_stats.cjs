const assert = require('node:assert/strict');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

async function main() {
  const file = path.resolve(__dirname,'../../frontend/src/components/v8/attentionPatternStats.js');
  await import(pathToFileURL(file).href);
  const {summarize} = globalThis.AI2050AttentionStats;
  const uniform = summarize(Float32Array.of(.25,.25,.25,.25),{topK:2});
  assert.equal(uniform.entropyBits,2);
  assert.equal(uniform.effectiveSources,4);
  assert.equal(uniform.topMass,.5);
  assert.deepEqual(uniform.ranked.map(r=>r.index),[0,1,2,3]);
  const one = summarize([0,1,0]);
  assert.equal(one.entropyBits,0);
  assert.equal(one.effectiveSources,1);
  const rounded = summarize([.5,.25,.249],{topK:1});
  assert.ok(rounded.ok);
  assert.ok(Math.abs(rounded.topMass - .5/.999)<1e-12);
  const tokens = summarize([.1,.1,.2,.6]);
  const groups = summarize([.2,.2,.6]);
  assert.ok(tokens.entropyBits > groups.entropyBits);
  assert.equal(groups.ranked[0].index,2);
  assert.equal(groups.topK,3);
  for (const missing of [[],null,[.4,.3],[.1,-.1,1],[.5,.499,NaN],[1.001],[.4,.7],[0,0],[undefined,1],new DataView(new ArrayBuffer(4))]) {
    assert.equal(summarize(missing).ok,false);
  }
  assert.equal(summarize([1],{topK:0}).ok,false);
  console.log('Attention-pattern formulas, tie ranks, complete rows, grouping and quantization checks passed.');
}

main().catch(e=>{console.error(e);process.exitCode=1;});
