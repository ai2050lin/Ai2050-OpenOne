const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'tests/glm5/result/branch_reuse14_20261113_v3');
const temp=path.join(root,'tests/glm5_temp/branch_reuse14_20261113_v3/browser');
const gate=spawnSync(path.join(root,'.venv/Scripts/python.exe'),['-B','-X','utf8','-c','import sys;from deploy.storage_policy import check_write;check_write(sys.argv[1],150*2**20,operation="BR14 offline report browser");check_write(sys.argv[2])',temp,out],{cwd:root,encoding:'utf8'});
assert.equal(gate.status,0,gate.stderr||gate.stdout);fs.mkdirSync(temp,{recursive:true});process.env.TEMP=temp;process.env.TMP=temp;
const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
(async()=>{
 const browser=await chromium.launchPersistentContext(path.join(temp,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000},args:['--disable-breakpad','--disable-crash-reporter']});
 const page=browser.pages()[0],errors=[],checks=[];let complete=false;page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(pathToFileURL(path.join(out,'report.html')).href);
  await page.getByRole('heading',{name:'Qwen3-14B：跨分支复用与层间组合',exact:true}).waitFor();
  assert.ok((await page.locator('.notice').innerText()).includes('尚未确认完整'));checks.push('bounded_outcome_and_missing_branch_visible');
  assert.equal(await page.locator('table').count(),7);checks.push('behavior_response_groups_rollout_output_intervention_causal_tables');
  for(const img of await page.locator('img').all())assert.ok(await img.evaluate(e=>e.complete&&e.naturalWidth>0));checks.push('all_scientific_plots_loaded');
  await page.screenshot({path:path.join(out,'report-desktop.png')});
  const first=page.locator('details.case').first();await first.locator(':scope > summary').click();
  assert.ok((await first.locator(':scope > pre').first().innerText()).includes('End of passage.'));
  await first.locator(':scope > details > summary').first().click();assert.ok((await first.locator(':scope > details > pre').first().innerText()).includes('Answer:'));
  await first.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'report-case.png')});checks.push('causal_prefix_and_later_question_separated');
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));checks.push('mobile_page_no_horizontal_overflow');
  await page.screenshot({path:path.join(out,'report-mobile.png')});
  for(const a of await page.locator('a').all()){
   const href=await a.getAttribute('href');
   if(href.startsWith('#'))assert.equal(await page.locator(href).count(),1);
   else assert.ok(fs.existsSync(path.resolve(out,href)),href);
  }
  checks.push('all_local_evidence_and_navigation_links_exist');assert.deepEqual(errors,[]);complete=true;
 }finally{await browser.close();fs.writeFileSync(path.join(temp,'checks.json'),JSON.stringify({passed:complete&&errors.length===0,checks,errors},null,2));}
 console.log(JSON.stringify({checks,errors}));
})().catch(e=>{console.error(e);process.exitCode=1;});
