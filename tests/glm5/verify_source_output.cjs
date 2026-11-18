const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const out=fs.realpathSync(path.resolve(__dirname,'result/source_output_v1'));
const reg=JSON.parse(fs.readFileSync(path.join(out,'registration.json'),'utf8'));
const result={time:new Date().toISOString(),checks:[],errors:[],gpuRequests:[]};
const record=(name,detail)=>{result.checks.push({name,detail});console.log(JSON.stringify({name,detail}));};
const num=x=>Number.isFinite(x)?x.toPrecision(4):'未检验';
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-sandbox']});let page;
 try{
  page=await browser.newPage({viewport:{width:1440,height:1100}});page.setDefaultTimeout(30000);
  page.on('pageerror',e=>result.errors.push(e.message));page.on('request',r=>{if(r.method()==='POST'&&/\/api\/live\/(load|run)/.test(r.url()))result.gpuRequests.push(r.url());});
  const api='http://localhost:5001/api/research-library/artifacts/';
  const response=await page.request.get(api+reg.report_sha256);assert.equal(response.status(),200);const report=await response.json();
  const mirror=await page.request.get('http://localhost:5032/api/research-library/artifacts/'+reg.report_sha256);assert.equal(mirror.status(),200);assert.deepEqual(await mirror.json(),report);
  assert.equal(report.unique_inputs,5760);assert.equal(report.formal_promotion,false);record('identical_local_reports_and_frozen_input_count',reg.report_sha256);
  const graph=await(await page.request.get(api+reg.graph_sha256)).json();assert.equal(graph.feature_cards.length,report.active_branches.length?2:0);
  for(const card of graph.feature_cards)assert.ok(card.index.group&&!card.index.anchor_only&&card.scope.formal_promotion===false);record('native_group_identity_and_evidence_boundary');
  await page.goto(`http://localhost:5173/workbench?lens=process&station=execution&run=${reg.run_id}&snapshot=${reg.model_snapshot}`);
  const archive=page.getByLabel('链接指定的实验',{exact:true});await archive.getByText('Qwen3-4B · 记录来源与输出符号的独立检验',{exact:true}).waitFor();
  await archive.getByText('研究报告与反例',{exact:true}).click();const panel=page.getByLabel('记录来源与输出符号报告',{exact:true});await panel.waitFor();
  const table=panel.getByLabel('来源编码行为门');assert.equal(await table.locator('tbody tr').count(),6);
  for(const [i,[key,v]] of Object.entries(report.behavior.confirm.branches).entries()){
   const row=table.locator('tbody tr').nth(i);assert.ok((await row.innerText()).includes((v.accuracy*100).toFixed(2)+'%'));
   assert.equal(await row.locator('td').last().innerText(),report.active_branches.includes(key)?'通过':'停止');
  }
  record('all_six_whole_branch_gates_match');
  const diagnostic=panel.getByLabel('来源编码首词诊断');
  for(const [i,d] of report.failure_diagnostics.branches.filter(d=>d.split==='confirm').entries()){
   const values=await diagnostic.locator('tbody tr').nth(i).locator('td').allTextContents();
   assert.deepEqual(values.slice(1),[d.n,d.counts.correct_after_trimming_whitespace,d.counts.correct_code_with_extra_whitespace,d.counts.record_number_in_letter_task].map(String));
  }
  record('format_and_record_number_failures_are_separated');
  if(report.failures.length){await panel.getByText(new RegExp('^'+report.failures[0].sample.id+' ·')).click();assert.ok((await panel.locator('.v8-sample-text:visible').allTextContents()).includes(report.failures[0].sample.prompt));}
  await panel.getByRole('heading',{level:3}).scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'client-behavior.png')});record('real_errors_ties_and_format_failures_retained');
  await panel.getByRole('button',{name:'来源与输出成分',exact:true}).click();
  if(report.active_branches.length){
   const branch=report.active_branches[0];await panel.getByLabel('来源编码分支').selectOption(branch);
   for(const [kind,index,split,pos,factor] of [['hidden',0,1,0,1],['hidden',36,1,1,0],['units',26,0,1,1],['heads',29,1,1,0]]){
    await panel.getByLabel('来源编码对象').selectOption(kind);if(kind==='hidden')await panel.getByLabel('来源编码层').selectOption(String(index));
    await panel.getByLabel('来源编码数据').selectOption(String(split));await panel.getByLabel('来源编码位置').selectOption(String(pos));if(kind!=='heads')await panel.getByLabel('来源编码成分').selectOption(String(factor));
    const n=kind==='hidden'?2560:kind==='units'?9728:32,nf=kind==='heads'?1:2,digest=report.coordinate_artifacts[branch][kind][String(index)];
    await panel.getByLabel(`来源编码完整原生阵列，全部${n}维，未进行TopK截断`,{exact:true}).waitFor();
    const doc=await(await page.request.get(api+digest)).json();assert.deepEqual(doc.array.shape,[2,nf,2,n]);const bytes=Buffer.from(doc.array.data,'base64');
    for(const at of [0,n-1]){
     await panel.getByLabel('来源编码原生索引').fill(String(at));const expected=num(bytes.readFloatLE((((split*nf+factor)*2+pos)*n+at)*4));
     await panel.getByText(`原生索引 ${at} = ${expected}`,{exact:true}).waitFor();
    }
   }
   record('all_layers_and_full_native_index_values_preserved');await panel.screenshot({path:path.join(out,'client-fields.png')});
   await panel.getByLabel('来源编码对象').selectOption('hidden');const missing='**/api/research-library/artifacts/'+report.coordinate_artifacts[branch].hidden['17'];
   await page.route(missing,r=>r.fulfill({status:404,contentType:'application/json',body:JSON.stringify({detail:'验收：缺失数组'})}));
   await panel.getByLabel('来源编码层').selectOption('17');await panel.getByText(/来源编码数组读取失败/).waitFor();await page.unroute(missing);
   await panel.getByLabel('来源编码层').selectOption('18');await panel.getByLabel('来源编码完整原生阵列，全部2560维，未进行TopK截断',{exact:true}).waitFor();
   await panel.getByLabel('来源编码层').selectOption('17');await panel.getByLabel('来源编码完整原生阵列，全部2560维，未进行TopK截断',{exact:true}).waitFor();record('missing_native_artifact_and_recovery');
   const wrongSha=report.coordinate_artifacts[branch].hidden['19'],wrong=await(await page.request.get(api+wrongSha)).json();wrong.model_snapshot='incorrect';
   const route='**/api/research-library/artifacts/'+wrongSha;await page.route(route,r=>r.fulfill({status:200,contentType:'application/json',body:JSON.stringify(wrong)}));
   await panel.getByLabel('来源编码层').selectOption('19');await panel.getByText('数据身份或形状不匹配，未显示坐标。',{exact:true}).waitFor();await page.unroute(route);record('wrong_model_identity_blocked');
  }
  const stopped=Object.keys(report.behavior.confirm.branches).filter(k=>!report.active_branches.includes(k));
  for(const key of stopped){await panel.getByLabel('来源编码分支').selectOption(key);await panel.getByText('这一整支没有通过行为门，未继续机制检验；失败样本保留在“行为与反例”。',{exact:true}).waitFor();}
  record('stopped_branches_explain_missing_mechanism_data',stopped);
  await panel.getByRole('button',{name:'未见样本与基线',exact:true}).click();
  for(const o of report.observations){const rows=panel.getByLabel(`来源编码基线 ${o.branch}`,{exact:true}).locator('tbody tr');for(const [i,v] of Object.values(o.mse).entries())assert.equal(await rows.nth(i).locator('td').last().innerText(),num(v));}
  if(!report.observations.length)await panel.getByText('没有通过行为门的完整分支，未报告内部预测优势。',{exact:true}).waitFor();
  await page.screenshot({path:path.join(out,'client-prediction.png')});record(report.observations.length?'frozen_holdout_baselines_match_without_filtering':'stopped_prediction_explains_no_baseline_comparison');
  await panel.getByRole('button',{name:'作用与负对照',exact:true}).click();
  for(const e of report.effects){const table=panel.getByLabel(`来源编码作用 ${e.branch} ${e.candidate}`,{exact:true});assert.equal(await table.locator('tr').nth(5).locator('td').last().innerText(),String(e.top1_changes));}
  if(!report.effects.length)await panel.getByText('行为门阻止了后续原模型干预，不能把缺失结果写成“没有作用”。',{exact:true}).waitFor();
  await page.screenshot({path:path.join(out,'client-effects.png')});record(report.effects.length?'effect_counts_and_invalid_control_boundaries':'stopped_effects_are_not_mislabeled_zero');
  await page.setViewportSize({width:390,height:844});await panel.getByRole('button',{name:'来源与输出成分',exact:true}).click();
  if(report.active_branches.length){await panel.getByLabel('来源编码分支').selectOption(report.active_branches[0]);await panel.getByLabel('来源编码对象').selectOption('units');
   const input=panel.getByLabel('来源编码原生索引');await input.fill('100');await input.focus();await input.press('ArrowUp');assert.equal(await input.inputValue(),'101');await input.scrollIntoViewIfNeeded();}
  if(!report.active_branches.length){const select=panel.getByLabel('来源编码分支');const keys=await select.locator('option').evaluateAll(nodes=>nodes.map(n=>n.value));await select.selectOption(keys.at(-1));await select.focus();await select.press('ArrowUp');assert.equal(await select.inputValue(),keys.at(-2));await select.scrollIntoViewIfNeeded();}
  assert.ok(await panel.locator('.hs-controls').evaluate(el=>el.scrollWidth<=el.clientWidth));await page.screenshot({path:path.join(out,'client-mobile.png')});record('mobile_controls_and_keyboard_navigation');
  assert.deepEqual(result.errors,[]);assert.deepEqual(result.gpuRequests,[]);result.passed=true;
 }catch(e){result.failure=e.stack;console.error(e);process.exitCode=1;if(page)await page.screenshot({path:path.join(out,'client-failure.png')});}
 finally{fs.writeFileSync(path.join(out,'client-acceptance.json'),JSON.stringify(result,null,2));await browser.close();}
})();
