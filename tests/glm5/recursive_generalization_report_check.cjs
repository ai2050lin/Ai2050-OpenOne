const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'tests/glm5/result/recursive_generalization_review_20261113');
const temp=path.join(root,'tests/glm5_temp/recursive-generalization-browser');
fs.mkdirSync(temp,{recursive:true});process.env.TEMP=temp;process.env.TMP=temp;
const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
(async()=>{
 const browser=await chromium.launchPersistentContext(path.join(temp,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000}});
 const page=browser.pages()[0],errors=[],checks=[];page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(pathToFileURL(path.join(out,'report.html')).href);
  assert.equal(await page.locator('section').count(),13);checks.push('all_10_cards_and_3_explanatory_sections');
  assert.equal(await page.locator('table tbody tr').count(),11);checks.push('10_historical_rows_and_header');
  await page.screenshot({path:path.join(out,'report-desktop.png')});
  const candidate=page.locator('section').filter({has:page.getByRole('heading',{name:/qwen14 B26\/U13099/})});
  await candidate.locator(':scope > details > summary').click();
  await candidate.locator(':scope > details > details').filter({hasText:'G positive · holdout · doc 896 token 813'}).locator(':scope > summary').click();
  assert.ok((await candidate.innerText()).includes('A=1.53125'));checks.push('numeric_context_expansion');
  await candidate.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'report-case.png')});
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));checks.push('mobile_no_horizontal_overflow');
  await page.screenshot({path:path.join(out,'report-mobile.png')});
  for(const a of await page.locator('a').all()){
   const href=await a.getAttribute('href');if(href.startsWith('https:'))continue;
   assert.ok(fs.existsSync(path.join(out,href)));
  }checks.push('all_local_evidence_links_exist');assert.deepEqual(errors,[]);
 }finally{
  await browser.close();fs.writeFileSync(path.join(out,'report-browser-check.json'),JSON.stringify({checks,errors},null,2));
 }
 console.log(JSON.stringify({checks,errors}));
})().catch(e=>{console.error(e);process.exitCode=1;});
