const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const out=path.resolve(__dirname,'result/mlp_puzzle_v1');
const reg=JSON.parse(fs.readFileSync(path.join(out,'registration.json'),'utf8'));
const report={time:new Date().toISOString(),checks:[],errors:[],gpuRequests:[]};
function ok(name,detail){report.checks.push({name,detail});console.log(JSON.stringify({name,detail}));}
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-sandbox']});
 let page;
 try {
  page=await browser.newPage({viewport:{width:1440,height:1000}});page.setDefaultTimeout(45000);
  page.on('pageerror',e=>report.errors.push(e.message));
  page.on('request',r=>{if(r.method()==='POST'&&/\/api\/live\/(run|load)/.test(r.url()))report.gpuRequests.push(r.url());});
  await page.goto(`http://localhost:5173/workbench?lens=process&station=execution&run=${reg.run_id}&snapshot=${reg.model_snapshot}`);
  const archive=page.getByLabel('链接指定的实验',{exact:true});
  await archive.getByText('Qwen3-4B · 单元拼图：解释、反例、预测与作用',{exact:true}).waitFor();
  await archive.getByText('研究报告与反例',{exact:true}).click();
  const panel=page.getByLabel('原生单元拼图完整验证报告',{exact:true});await panel.waitFor();
  await panel.getByLabel('候选单元组').selectOption('binding-B34');
  assert.equal(await panel.getByLabel('八条件原生响应数值与热力图').locator('tbody tr').count(),8);
  ok('real_report_and_frozen_response_table');
  await panel.screenshot({path:path.join(out,'client-response.png')});
  const computation=panel.locator('details').filter({hasText:'一个原生单元如何响应八种条件'});
  await computation.locator('summary').click();assert.equal(await computation.locator('tbody tr').count(),8);
  assert.ok((await computation.innerText()).includes('-27.63'));
  await panel.getByRole('button',{name:'反例与失败',exact:true}).click();
  assert.ok(await panel.locator('details').count()>=3);await panel.locator('summary').first().click();
  await panel.locator('.v8-sample-text').first().waitFor();ok('counterexamples_expand_original_input');
  await panel.getByRole('button',{name:'作用与副作用',exact:true}).click();
  assert.equal(await panel.locator('table').first().locator('tbody tr').count(),8);ok('all_four_objects_and_two_doses_with_controls');
  await panel.screenshot({path:path.join(out,'client-effects.png')});
  await panel.getByRole('button',{name:'代表单元',exact:true}).click();
  assert.equal(await panel.locator('tbody tr').count(),48);ok('all_stratified_representatives');
  await panel.getByRole('button',{name:'材料与审计',exact:true}).click();
  const evidence=panel.getByRole('link',{name:'技术与结论审计',exact:true});
  const response=await page.request.get(new URL(await evidence.getAttribute('href'),page.url()).href);assert.equal(response.status(),200);
  const audit=await response.json();assert.equal(audit.model_snapshot,reg.model_snapshot);assert.equal(audit.verified_files,640);
  ok('immutable_audit_readback_and_snapshot');
  await panel.getByRole('button',{name:'能保留的拼图',exact:true}).click();
  const native=panel.getByRole('link',{name:'B34/U2359',exact:true});
  await native.click();
  const detail=page.getByLabel('统一对象详情',{exact:true});
  await detail.getByRole('heading',{name:'B34 · U2359',exact:true}).waitFor();ok('real_native_unit_navigation');
  await detail.getByRole('button',{name:'研究记录',exact:true}).click();
  const linked=detail.locator('a').filter({hasText:'查看研究报告'});
  await linked.first().waitFor();assert.ok((await linked.evaluateAll(es=>es.map(e=>e.href))).some(url=>url.includes(reg.run_id)));
  ok('native_unit_links_back_to_current_evidence');
  await page.goto(`http://localhost:5173/workbench?lens=spatial&view=parameters&pkind=mlp&pblock=34&punit=842&snapshot=${reg.model_snapshot}`);
  await detail.getByRole('heading',{name:'B34 · U842',exact:true}).waitFor();
  await detail.getByRole('button',{name:'研究记录',exact:true}).click();
  await detail.getByText(/组级结果不能当作每个成员单独有效的证据/).first().waitFor();
  ok('non_anchor_group_member_has_group_evidence_and_scope');
  assert.deepEqual(report.gpuRequests,[]);assert.deepEqual(report.errors,[]);report.passed=true;
 } catch(e){report.failure=e.stack;console.error(e);process.exitCode=1;if(page)await page.screenshot({path:path.join(out,'client-failure.png')});}
 finally {fs.writeFileSync(path.join(out,'client-acceptance.json'),JSON.stringify(report,null,2));await browser.close();}
})();
