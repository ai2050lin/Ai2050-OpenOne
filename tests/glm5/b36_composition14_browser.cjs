const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'tests/glm5/result/b36_composition14_20261113_v3');
const temp=path.join(root,'tests/glm5_temp/b36-composition14-browser');
const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
(async()=>{
 const {execFileSync}=require('node:child_process');
 execFileSync(path.join(root,'.venv/Scripts/python.exe'),['-B','-X','utf8','-c',"from deploy.storage_policy import check_write; check_write(r'"+temp+"',100*2**20,operation='B36 report browser profile')"],{cwd:root,stdio:'pipe'});
 fs.mkdirSync(temp,{recursive:true});process.env.TEMP=temp;process.env.TMP=temp;
 const browser=await chromium.launchPersistentContext(path.join(temp,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000}});
 const page=browser.pages()[0],errors=[],checks=[];let passed=false;page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(pathToFileURL(path.join(out,'report.html')).href);
  assert.equal(await page.locator('h2').count(),7);checks.push('seven_research_sections');
  assert.ok((await page.locator('body').innerText()).includes('组合机制尚未获得准入'));checks.push('scope_limitation_visible');
  await page.screenshot({path:path.join(out,'report-desktop.png')});
  const details=page.locator('details').filter({has:page.locator('summary',{hasText:'真实失败题'})});
  await details.locator('summary').click();assert.ok(await details.locator('pre').count()>0);checks.push('actual_counterexamples_expand');
  const map=page.locator('details').filter({has:page.locator('summary',{hasText:'hidden-source-r1-raw'})});
  await map.locator('summary').click();await map.scrollIntoViewIfNeeded();await map.locator('img').waitFor();
  assert.equal(await map.locator('img').evaluate(el=>el.naturalWidth),5120);checks.push('full_native_coordinate_image');
  await page.screenshot({path:path.join(out,'report-field.png')});
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));checks.push('mobile_no_horizontal_overflow');
  await page.screenshot({path:path.join(out,'report-mobile.png')});
  for(const a of await page.locator('a').all()){
   const href=await a.getAttribute('href');assert.ok(fs.existsSync(path.resolve(out,href)),href);
  }checks.push('evidence_links_exist');assert.deepEqual(errors,[]);passed=true;
 }finally{await browser.close();fs.writeFileSync(path.join(out,'browser-check.json'),JSON.stringify({passed,checks,errors},null,2));}
 console.log(JSON.stringify({checks,errors}));
})().catch(e=>{console.error(e);process.exitCode=1;});
