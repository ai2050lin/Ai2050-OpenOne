const {test}=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const moduleUrl=pathToFileURL(path.resolve(__dirname,'../../frontend/src/components/v8/researchLocation.js'));
const model={snapshot:'weights-A',n_layers:36,n_heads:32,d_model:2560,d_ff:9728,vocab_size:151936};
const card=(space,index,extra={})=>({type:'feature',model_snapshot:model.snapshot,body:{layer:18,index:{space,...index},...extra}});
test('research cards preserve native spaces, zero-based blocks and valid edge indices',async()=>{
 const {researchLocation:locate}=await import(moduleUrl);
 for(const [input,expected] of [
  [card('mlp',{unit:9031}),{layer:19,module:'mlp',index:9031}],
  [card('attention',{head:31}),{layer:19,module:'attn',index:31}],
  [card('residual',{coordinate:2559}),{layer:19,module:'hs',index:2559}],
  [card('residual',{coordinate:0},{boundary:'hidden[0]'}),{layer:0,module:'hs',index:0}],
  [card('embedding',{token_id:151935}),{layer:0,module:'embedding',index:151935}],
 ])assert.deepEqual(locate(input,model),expected);
});
test('snapshot mismatch, incomplete identity and out-of-range indices do not create spatial evidence',async()=>{
 const {researchLocation:locate}=await import(moduleUrl);
 for(const input of [
  {...card('mlp',{unit:1}),model_snapshot:'weights-B'},
  card('mlp',{unit:-1}),card('mlp',{unit:9728}),card('mlp',{unit:1},{layer:36}),
  card('attention',{head:32}),card('residual',{coordinate:2560}),
  card('embedding',{token_id:151936}),card('mlp',{unit:'1'}),
  card('mlp',{}),card('sae',{unit:1}),{type:'edge',model_snapshot:model.snapshot,body:{layer:18,index:{space:'mlp',unit:1}}},
 ])assert.equal(locate(input,model),null);
 assert.equal(locate(card('mlp',{unit:0}),null),null);
 assert.equal(locate(card('mlp',{unit:0}),{...model,d_ff:undefined}),null);
});
