const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const out=fs.realpathSync(path.resolve(__dirname,'result/record_selection_v3'));
const reg=JSON.parse(fs.readFileSync(path.join(out,'registration.json'),'utf8'));
const result={time:new Date().toISOString(),checks:[],errors:[],gpuRequests:[]};
const record=(name,detail)=>{result.checks.push({name,detail});console.log(JSON.stringify({name,detail}));};
const num=v=>Number.isFinite(v)?v.toPrecision(4):'未定义';
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-sandbox']});
 let page;
 try{
  page=await browser.newPage({viewport:{width:1440,height:1100}});page.setDefaultTimeout(30000);
  page.on('pageerror',e=>result.errors.push(e.message));
  page.on('request',r=>{if(r.method()==='POST'&&/\/api\/live\/(load|run)/.test(r.url()))result.gpuRequests.push(r.url());});
  const api='http://localhost:5001/api/research-library/artifacts/';
  const response=await page.request.get(api+reg.report_sha256);assert.equal(response.status(),200);const report=await response.json();
  assert.equal(report.model_snapshot,reg.model_snapshot);assert.equal(report.unique_inputs,4096);
  const mirror=await page.request.get('http://localhost:5032/api/research-library/artifacts/'+reg.report_sha256);assert.equal(mirror.status(),200);assert.deepEqual(await mirror.json(),report);
  record('two_local_libraries_same_report',reg.report_sha256);
  const graph=await(await page.request.get(api+reg.graph_sha256)).json();assert.equal(graph.feature_cards.length,2);
  assert.deepEqual(graph.feature_cards.map(c=>c.index.indices.length),[2,32]);assert.ok(graph.feature_cards.every(c=>c.index.group&&!c.index.anchor_only&&c.scope.formal_promotion===false));
  record('group_cards_preserve_head_and_unit_indices_without_single_coordinate_anchor');
  await page.goto(`http://localhost:5173/workbench?lens=process&station=execution&run=${reg.run_id}&snapshot=${reg.model_snapshot}`);
  const archive=page.getByLabel('链接指定的实验',{exact:true});await archive.getByText('Qwen3-4B · 有效与过时记录：头与单元组检验',{exact:true}).waitFor();
  await archive.getByText('研究报告与反例',{exact:true}).click();const panel=page.getByLabel('有效与过时记录机制报告',{exact:true});await panel.waitFor();
  await panel.getByText('完整范围的行为门未通过；以下为受限候选与作用诊断，不作正式机制确认。',{exact:true}).waitFor();
  assert.equal(report.diagnostic_only,true);assert.ok(report.effects.every(e=>!e.scope_use_supported));
  const history=panel.getByLabel('记录选择行为校准').locator('tbody tr');assert.equal(await history.count(),5);
  for(let i=0;i<5;i++)assert.ok((await history.nth(i).innerText()).includes(`${(report.calibration_history[i].accuracy*100).toFixed(2)}%`));
  record('failed_calibrations_and_restricted_denominators_visible');
  await panel.getByText('错误集中在哪些条件？',{exact:true}).click();
  const strata=report.behavior_error_strata.filter(r=>['valid_position','language','conflict'].includes(r.field));const sr=panel.getByLabel('记录选择行为失败分组').locator('tbody tr');
  for(let i=0;i<strata.length;i++)assert.equal(await sr.nth(i).locator('td').nth(2).innerText(),String(strata[i].errors));
  record('posthoc_behavior_error_strata_match_and_remain_descriptive');
  if(report.failures.length){await panel.getByText(new RegExp('^'+report.failures[0].sample.id+' ·')).click();assert.ok((await panel.locator('.v8-sample-text:visible').allTextContents()).includes(report.failures[0].sample.prompt));}
  await panel.screenshot({path:path.join(out,'client-behavior.png')});
  record('real_counterexample_prompt_visible');
  await panel.getByRole('button',{name:'注意力头与MLP单元',exact:true}).click();
  for(const [kind,block,split,term] of [['heads',0,1,0],['heads',0,0,1],['units',26,1,1],['units',35,0,0]]){
   await panel.getByLabel('记录机制对象').selectOption(kind);if(kind==='units')await panel.getByLabel('记录机制层').selectOption(String(block));
   await panel.getByLabel('记录机制数据划分').selectOption(String(split));await panel.getByLabel('记录机制度量').selectOption(String(term));
   const count=kind==='heads'?1152:9728;await panel.getByLabel(`记录机制完整原生阵列，全部${count}维，未进行TopK截断`,{exact:true}).waitFor();
   const key=kind==='heads'?'all':String(block),digest=report.coordinate_artifacts[kind][key];const d=await(await page.request.get(api+digest)).json(),bytes=Buffer.from(d.array.data,'base64');
   assert.deepEqual(d.array.shape,[2,2,count]);
   for(const index of [0,count-1]){await panel.getByLabel('记录机制原生索引').fill(String(index));const expected=num(bytes.readFloatLE(((split*2+term)*count+index)*4));await panel.getByText(new RegExp(`^原生索引 ${index} = ${expected.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}`)).waitFor();}
  }
  record('all_1152_heads_and_each_9728_unit_layer_full_native_numeric_identity');await panel.screenshot({path:path.join(out,'client-native.png')});
  const missing='**/api/research-library/artifacts/'+report.coordinate_artifacts.units['17'];
  await page.route(missing,r=>r.fulfill({status:404,contentType:'application/json',body:JSON.stringify({detail:'验收：缺失坐标'})}));
  await panel.getByLabel('记录机制层').selectOption('17');await panel.getByText(/记录机制坐标读取失败/).waitFor();await page.unroute(missing);
  await panel.getByLabel('记录机制层').selectOption('18');await panel.getByLabel('记录机制完整原生阵列，全部9728维，未进行TopK截断',{exact:true}).waitFor();
  await panel.getByLabel('记录机制层').selectOption('17');await panel.getByLabel('记录机制完整原生阵列，全部9728维，未进行TopK截断',{exact:true}).waitFor();
  record('missing_artifact_and_recovery');
  const wrongSha=report.coordinate_artifacts.units['19'],wrong=await(await page.request.get(api+wrongSha)).json();wrong.model_snapshot='wrong';
  const mismatch='**/api/research-library/artifacts/'+wrongSha;await page.route(mismatch,r=>r.fulfill({status:200,contentType:'application/json',body:JSON.stringify(wrong)}));
  await panel.getByLabel('记录机制层').selectOption('19');await panel.getByText('数据身份或形状不匹配，未显示坐标。',{exact:true}).waitFor();await page.unroute(mismatch);record('wrong_snapshot_blocked');
  await panel.getByRole('button',{name:'简单基线比较',exact:true}).click();const baselineRows=panel.getByLabel('记录选择预测基线').locator('tbody tr');
  for(const [i,v] of Object.values(report.baselines.mse).entries())assert.equal(await baselineRows.nth(i).locator('td').last().innerText(),num(v));
  record('same_subset_frozen_baseline_values_match');await panel.screenshot({path:path.join(out,'client-baselines.png')});
  await panel.getByRole('button',{name:'实际作用与对照',exact:true}).click();
  for(const e of report.effects){
   await panel.getByLabel('记录作用候选').selectOption(e.candidate);await panel.getByLabel('记录作用方法').selectOption(e.method);if(e.method==='ablation')await panel.getByLabel('记录作用剂量').selectOption(String(e.dose));
   const rows=panel.getByLabel('记录选择实际作用').locator('tr');assert.equal(await rows.nth(0).locator('td').last().innerText(),num(e.mean_margin_loss));
   assert.equal(await rows.nth(1).locator('td').last().innerText(),num(e.native_group_control_loss));
   for(const s of e.failures)assert.ok((await panel.innerText()).includes(s));
  }
  record('all_six_effects_controls_failures_visible');await panel.screenshot({path:path.join(out,'client-effects.png')});
  for(const e of report.matched_native_controls){
   await panel.getByLabel('记录作用候选').selectOption(e.candidate);await panel.getByLabel('记录作用方法').selectOption('ablation');await panel.getByLabel('记录作用剂量').selectOption('1');
   assert.equal(await panel.getByLabel('记录选择等幅度原生方向对照').locator('tr').first().locator('td').last().innerText(),num(e.control_mean_loss));
  }
  record('supplemental_matched_native_direction_controls_match');
  await page.setViewportSize({width:390,height:844});await panel.getByRole('button',{name:'注意力头与MLP单元',exact:true}).click();
  await panel.getByLabel('记录机制对象').selectOption('units');await panel.getByLabel('记录机制层').selectOption('26');
  const input=panel.getByLabel('记录机制原生索引');await input.fill('100');await input.focus();await input.press('ArrowUp');assert.equal(await input.inputValue(),'101');
  assert.ok(await panel.locator('.hs-controls').evaluate(el=>el.scrollWidth<=el.clientWidth));
  await input.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'client-mobile.png')});record('mobile_controls_fit_and_keyboard_index_works');
  assert.deepEqual(result.errors,[]);assert.deepEqual(result.gpuRequests,[]);result.passed=true;
 }catch(e){result.failure=e.stack;console.error(e);process.exitCode=1;if(page)await page.screenshot({path:path.join(out,'client-failure.png')});}
 finally{fs.writeFileSync(path.join(out,'client-acceptance.json'),JSON.stringify(result,null,2));await browser.close();}
})();
