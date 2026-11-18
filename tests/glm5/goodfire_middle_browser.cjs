const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'tests/glm5/result/goodfire_middle_20261113_v2');
const temp=path.join(root,'tests/glm5_temp/goodfire-middle-browser');fs.mkdirSync(temp,{recursive:true});process.env.TEMP=temp;process.env.TMP=temp;
const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const specs=[['qwen4','Qwen3-4B',9728,2560],['qwen14','Qwen3-14B',17408,5120]];
const only=process.argv[2];if(only&&!specs.some(s=>s[0]===only))throw Error('Unknown model');
const result={checks:[],errors:[],gpuRequests:[]},ok=s=>{result.checks.push(s);console.log(s)};
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});page.setDefaultTimeout(60000);
 page.on('pageerror',e=>result.errors.push(e.message));
 page.on('request',r=>{if(r.method()==='POST'&&/\/api\/live\/(load|run)/.test(r.url()))result.gpuRequests.push(r.url())});
 try{
  for(const [key,name,width,dim] of specs.filter(s=>!only||s[0]===only)){
   const report=JSON.parse(fs.readFileSync(path.join(out,key,'report.json'))),run=report.run_id,snapshot=report.model_snapshot;
   const base='http://127.0.0.1:5035/workbench?'+new URLSearchParams({lens:'spatial',view:'graph',gview:'internal',run,snapshot});
   await page.goto(base);await page.getByRole('button',{name:'成果库',exact:true}).click();
   const lib=page.getByRole('region',{name:'成果库',exact:true});
   await lib.getByLabel('成果研究',{exact:true}).selectOption(run);
   await lib.locator('.aw-library-list > button').filter({hasText:`B18–B28全部${(width*11).toLocaleString('en-US')}`}).click();
   await lib.getByText('完整报告与原始证据',{exact:true}).click();
   const atlas=lib.getByRole('region',{name:'原生全量MLP研究',exact:true});
   await atlas.getByRole('heading',{name:`${name} · B18–B28 神经元档案`,exact:true}).waitFor();
   assert.equal(await atlas.getByLabel('原生研究层').locator('option').count(),11);
   await atlas.locator(`canvas[aria-label*="${width*11}"]`).waitFor();ok(key+'_catalog_and_all_native_points');
   await atlas.getByLabel('原生研究层').selectOption('28');await atlas.getByLabel('原生单元',{exact:true}).fill(String(width-1));
   await atlas.getByText(`B28/U${width-1}`,{exact:true}).waitFor();
   await atlas.getByRole('tab',{name:'完整参数',exact:true}).click();await atlas.getByLabel('原生参数坐标',{exact:true}).fill(String(dim-1));
   const raw=await (await page.request.get(`http://127.0.0.1:5035/api/research/native-mlp/units/28/${width-1}?run=${run}`)).json();
   assert.equal(Number(await atlas.locator('output').textContent()),raw.weights.gate[dim-1]);ok(key+'_last_unit_native_parameter');
   await atlas.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,key+'-parameters.png')});
   await atlas.getByRole('tab',{name:'上下游关联',exact:true}).click();assert.equal(await atlas.getByLabel('原生关联方向').inputValue(),'incoming');
   await atlas.getByText('全部关联原值',{exact:true}).waitFor();ok(key+'_complete_neighbor_relation');
   await atlas.getByRole('tab',{name:'验证与反例',exact:true}).click();await atlas.getByText('标准化预测与完整词身份对照',{exact:true}).waitFor();
   await atlas.getByText('G/V组合能否预测实际乘积',{exact:true}).click();ok(key+'_exact_token_and_composed_prediction');
   const selected=report.intervention.selected_units[0];
   await atlas.getByLabel('原生已测干预单元').selectOption(`${selected.block}:${selected.unit}`);
   await atlas.getByRole('img',{name:'原生真实单元剂量曲线'}).waitFor();
   await atlas.getByRole('button',{name:'查看逐样本干预与失败对照',exact:true}).click();
   await atlas.getByLabel('原生逐样本实验').selectOption('capability');await atlas.getByLabel('原生干预样本').fill('511');
   await atlas.getByText('本样本全部剂量、对照、恢复与身份',{exact:true}).click();await atlas.locator('pre').filter({hasText:'input_tokens'}).waitFor();
   ok(key+'_last_capability_sample_and_failed_controls');
   await lib.getByRole('button',{name:'定位卡与关联',exact:true}).click();await lib.getByLabel('成果检索').fill(`B28:U${width-1}`);
   await lib.locator('.aw-library-list > button').filter({hasText:`B28/U${width-1}`}).click();await lib.getByRole('button',{name:'定位到当前空间',exact:true}).click();
   const query=new URL(page.url()).searchParams;assert.equal(query.get('sblock'),'29');assert.equal(query.get('sindex'),String(width-1));
   await page.getByRole('tab',{name:'对象详情',exact:true}).click();
   const shared=page.locator('section[aria-label="原生全量MLP研究"]:visible');await shared.getByText(`B28/U${width-1}`,{exact:true}).waitFor();
   await page.reload();await page.getByRole('tab',{name:'对象详情',exact:true}).click();await shared.getByText(`B28/U${width-1}`,{exact:true}).waitFor();
   ok(key+'_shared_selection_and_refresh');
   await page.setViewportSize({width:390,height:844});await shared.getByRole('heading').scrollIntoViewIfNeeded();
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   await page.screenshot({path:path.join(out,key+'-mobile.png')});ok(key+'_mobile_no_overflow');await page.setViewportSize({width:1440,height:1000});
   await page.goto('http://127.0.0.1:5035/workbench?'+new URLSearchParams({lens:'spatial',view:'graph',gview:'structure',snapshot,run,sblock:'36',smodule:'mlp',sindex:'0'}));
   await page.getByRole('tab',{name:'对象详情',exact:true}).click();
   await page.getByText('本轮仅采集 B18–B28；当前层没有本轮原生档案。查看其他层的历史结果，请在成果库切换研究版本。',{exact:true}).waitFor();
   assert.equal(await page.locator('section[aria-label="Qwen3-14B全量MLP研究"]:visible').count(),0);ok(key+'_outside_scope_does_not_mix_old_run');
  }
  const old='QWEN14-GOODFIRE-B28-B39-20261113-v2';
  await page.goto('http://127.0.0.1:5035/workbench?'+new URLSearchParams({lens:'spatial',view:'graph',gview:'structure',snapshot:'1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce',run:old,sblock:'40',smodule:'mlp',sindex:'17407'}));
  await page.getByRole('tab',{name:'对象详情',exact:true}).click();await page.getByRole('heading',{name:'Qwen3-14B · B28–B39 神经元档案',exact:true}).waitFor();ok('legacy_tail_atlas_remains_accessible');
  await page.goto('http://127.0.0.1:5035/workbench?'+new URLSearchParams({lens:'spatial',view:'graph',gview:'structure',snapshot:'1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce',run:'QWEN14-MLP-B28-B35-20261109-v1',sblock:'29',smodule:'mlp',sindex:'0'}));
  await page.getByRole('tab',{name:'对象详情',exact:true}).click();await page.getByRole('heading',{name:'Qwen3-14B · B28–B35 神经元档案',exact:true}).waitFor();ok('old_B28_overlap_preserves_original_run');
  assert.deepEqual(result.errors,[]);assert.deepEqual(result.gpuRequests,[]);result.passed=true;
 }catch(error){result.failure=error.stack;process.exitCode=1;await page.screenshot({path:path.join(out,'browser-failure.png')});fs.writeFileSync(path.join(out,'browser-failure-text.txt'),await page.locator('body').innerText());console.error(error)}
 finally{await browser.close();fs.writeFileSync(path.join(out,`browser-acceptance${only?'-'+only:''}.json`),JSON.stringify(result,null,2))}
})();
