const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'tests/glm5/result/recursive_branch14_20261113_v6');
const temp=path.join(root,'tests/glm5_temp/recursive_branch14_20261113_v6/browser');
const storage=spawnSync(path.join(root,'.venv/Scripts/python.exe'),['-B','-X','utf8','-c',
 'import sys;from deploy.storage_policy import check_write;check_write(sys.argv[1],100*2**20,operation="offline report browser");check_write(sys.argv[2])',temp,out],{cwd:root,encoding:'utf8'});
assert.equal(storage.status,0,storage.stderr||storage.stdout||'Storage preflight failed');
fs.mkdirSync(temp,{recursive:true});process.env.TEMP=temp;process.env.TMP=temp;
const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
(async()=>{
 const browser=await chromium.launchPersistentContext(path.join(temp,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000}});
 const page=browser.pages()[0],errors=[],checks=[];let completed=false;page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(pathToFileURL(path.join(out,'report.html')).href);
  await page.getByRole('heading',{name:'Qwen3-14B：关系分支验证与末层回溯',exact:true}).waitFor();
  assert.equal(await page.locator('table').count(),8);checks.push('response_joint_covariance_chain_edges_intervention_norm_control_tables');
  const expected=JSON.parse(fs.readFileSync(path.join(out,'counterexamples.json'))).examples.length;
  assert.equal(await page.locator('main > details > pre').count(),expected);checks.push('all_counterexamples_retained');
  for(const img of await page.locator('img').all())assert.ok(await img.evaluate(e=>e.complete&&e.naturalWidth>0));checks.push('scientific_plots_loaded');
  await page.screenshot({path:path.join(out,'report-desktop.png')});
  const first=page.locator('main > details:has(> pre)').first();await first.locator(':scope > summary').click();
  assert.ok((await first.locator(':scope > pre').innerText()).includes('End of passage.'));
  await first.locator(':scope > details > summary').click();
  assert.ok((await first.locator(':scope > details > pre').innerText()).includes('Answer:'));checks.push('causal_prefix_and_future_probe_separated');
  await first.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'report-case.png')});
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));checks.push('mobile_no_page_overflow');
  await page.screenshot({path:path.join(out,'report-mobile.png')});
  for(const a of await page.locator('a').all())assert.ok(fs.existsSync(path.resolve(out,await a.getAttribute('href'))));checks.push('all_evidence_links_exist');
  assert.deepEqual(errors,[]);
  completed=true;
 }finally{
  await browser.close();fs.writeFileSync(path.join(temp,'checks.json'),JSON.stringify({passed:completed&&errors.length===0,checks,errors},null,2));
 }
 console.log(JSON.stringify({checks,errors}));
})().catch(e=>{console.error(e);process.exitCode=1;});
