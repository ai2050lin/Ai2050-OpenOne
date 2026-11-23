/* Load only archived, opened results for the file preview. Each result keeps its own unit of measurement. */
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {pathToFileURL} = require('node:url');

module.exports = function databaseData(root, evidence) {
  const file = rel => path.join(root,rel);
  const read = p => JSON.parse(fs.readFileSync(p,'utf8'));
  const results = file('tests/glm5/result');
  const promptByPhase = {};
  for (const [i,run] of evidence.cues.entries()) {
    const buffer = fs.readFileSync(path.join(results,run.id,'confirm-materials.json'));
    if (createHash('sha256').update(buffer).digest('hex')!==run.provenance['confirm-materials.json']) throw Error('Material snapshot mismatch '+run.id);
    promptByPhase[i===0?2983:2990]=Object.fromEntries(JSON.parse(buffer).map(r=>[r.id,r.prompt]));
  }
  const dir=path.join(results,'routine/run-t7dfn8_7');
  const causalReceipt=read(path.join(dir,'server-receipt.json'));
  const causalFile=process.env.AI2050_REVIEW_CAUSAL_CASES || 'D:/ai2050_native/scratch/routine/run-t7dfn8_7/case_results.jsonl';
  function verified(p,name){
    const bytes=fs.readFileSync(p);
    if (createHash('sha256').update(bytes).digest('hex')!==causalReceipt.artifacts.find(a=>a.name===name)?.sha256)
      throw Error('Causal archive mismatch: '+name);
    return bytes.toString('utf8');
  }
  const causalResults = new Map(verified(causalFile,'case_results.jsonl').trim().split(/\r?\n/).map(s=>{const r=JSON.parse(s);return [r.case_id,r];}));
  const causal=JSON.parse(verified(path.join(dir,'questions.json'),'questions.json')).map(q=>{
    const r=causalResults.get(q.id);
    if(!r || r.arm!==q.arm || r.run_id!==causalReceipt.run_id)throw Error('Mismatched causal observation '+q.id);
    return [q.source_id,q.arm,q.arm==='base'?q.prompt:null,q.codes[q.target_record],
      q.codes[q.choice_ids.indexOf(r.output_id)] ?? ('token '+r.output_id),r.correct,r.margin,r.output_id];
  });
  const colorQuestions=read(path.join(results,'color_reuse_v1/greedy/study-package/questions.json'));
  const colors=Object.fromEntries(colorQuestions.rows.filter(r=>r.arm==='base').map(r=>[r.id,r.prompt]));
  const h0=file('research/mainline/H0');
  const raw=read(path.join(h0,'run_v5.json'));
  const prereg=read(path.join(h0,'MA-H0-v5.prereg.json'));
  const words=Object.fromEntries(fs.readFileSync(file('shared/dist_protocol/lexicon/ENT-XF_nouns.v2.tsv'),'utf8')
    .split(/\r?\n/).filter(s=>s.trim()&&!s.startsWith('#')).map(s=>{const v=s.split('\t');return [v[0],[v[4],v[5]]];}));
  const bilingual=Object.entries(raw.langs).flatMap(([lang,block])=>block.records.map((r,i)=>{
    if(!words[r.entity])throw Error('Missing term '+r.entity);
    return [lang,i+1,r.entity,r.cell,r.correct,r.choice,r.scores];
  }));
  return {promptByPhase, colors, causal, causalFile:pathToFileURL(causalFile).href, bilingual, words,
    h0Questions:Object.fromEntries(Object.entries(prereg.text).map(([lang,t])=>[lang,t.q]))};
};
