/* Browser acceptance of the offline test database against known archive counts. */
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {fileURLToPath,pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_CORE||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root=path.resolve(__dirname,'../..'),out=process.env.AI2050_PREVIEW_RESULTS || path.join(__dirname,'result/client_test_database');
const work=path.join(root,'tests/glm5_temp/client_test_database');
const url=pathToFileURL(path.join(root,'research/glm5/docs/previews/client-test-database.html')).href;
async function main(){
  [out,work].forEach(p=>fs.mkdirSync(p,{recursive:true}));
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||
    path.join(require('node:os').homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe'),
    downloadsPath:work,env:{...process.env,TEMP:work,TMP:work,TMPDIR:work}});
  try{
    const ctx=await browser.newContext({viewport:{width:1440,height:920},acceptDownloads:true}),page=await ctx.newPage();
    page.setDefaultTimeout(5000);
    const errors=[],network=[],files=new Set();
    page.on('pageerror',e=>{errors.push(e.message);console.error(e.message);});
    page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
    await ctx.route(/^https?:/,route=>route.abort());
    const db=page.locator('.wide-evidence'),dialog=db.locator('dialog[open]');
    const btn=(k,v)=>db.locator('[data-db="'+k+'"]'+(v==null?'':'[data-value="'+v+'"]')+':visible').first();
    const click=(k,v)=>btn(k,v).click();
    const set=(k,v)=>db.locator('[data-filter="'+k+'"]').selectOption(v);
    const count=()=>db.locator('.db-hit').allInnerTexts();
    const close=()=>page.keyboard.press('Escape');
    const reset=async()=>{if(await btn('reset').count())await click('reset');};
    const shot=name=>page.screenshot({path:path.join(out,name+'.png')});
    const checkFiles=async()=>{for(const a of await dialog.locator('a:visible').all()){
      const href=await a.getAttribute('href');if(!href||href.startsWith('data:'))continue;
      const f=fileURLToPath(new URL(href,url));assert.ok(fs.existsSync(f),f);files.add(f);
    }};
    await page.goto(url);await db.getByRole('heading',{name:'测试数据库',exact:true}).waitFor();
    assert.equal(await page.locator('.wb-subnav:visible').count(),0);
    assert.deepEqual(await page.locator('.fw-rail button').allInnerTexts(),['测试数据库','语言与模型','实验管理','系统设置']);
    assert.deepEqual(await count(),['1,536 道 ↗','1,536 道 ↗','192 道 ↗','324 道 ↗','3,136 道 ↗']);
    const rawN=await db.locator('.db-number').evaluateAll(cells=>cells.reduce((n,cell)=>n+parseInt((cell.querySelector('small')||cell.querySelector('button')).textContent.replaceAll(',',''),10),0));
    assert.equal(rawN,8656);
    assert.match(await db.locator('.db-resulthead').innerText(),/6,724/);await shot('database');
    await click('preset','head');assert.deepEqual(await count(),['339 道 ↗','303 道 ↗']);
    assert.match(await db.locator('.db-resulthead').innerText(),/642/);await shot('search');
    await click('cases',2990);assert.match(await dialog.innerText(),/303 道题/);
    await click('next');await click('case');assert.match(await dialog.innerText(),/头首选/);
    await dialog.getByText('完整模型输入',{exact:true}).click();assert.match(await dialog.locator('pre').innerText(),/BEGIN/);await checkFiles();await click('back-cases');await close();
    for(const p of [2990,2983])await db.locator('[data-db-pick="'+p+'"]').check();
    await click('compare');assert.match(await dialog.innerText(),/642 道命中题/);await shot('comparison');
    const [download]=await Promise.all([page.waitForEvent('download'),dialog.getByText('导出比较记录 JSON ↓').click()]);
    const json=JSON.parse(fs.readFileSync(await download.path(),'utf8'));await download.delete();
    assert.deepEqual(json.pooled,{questions:642,div:642});await click('compare-cases',2983);assert.match(await dialog.innerText(),/339 道题/);await close();
    assert.equal(json.full_batch_reference[0].h,377);assert.equal(json.full_batch_reference[0].m,1018);
    await page.reload();assert.equal(await db.locator('[data-db-pick="2983"]').isChecked(),true);assert.deepEqual(await count(),['339 道 ↗','303 道 ↗']);
    await click('unpick');await reset();
    await db.locator('#db-search').fill('U17216');assert.deepEqual(await count(),['192 道 ↗']);
    await click('cases',2997);await click('case');assert.match(await dialog.innerText(),/仅恢复 FFN 组/);assert.doesNotMatch(await dialog.locator('table').innerText(),/假操作/);await checkFiles();await close();
    await click('study',2997);await click('object','U17216');assert.match(await dialog.innerText(),/单独的行为效应待测/);
    await click('locate','U17216');await page.locator('.aw-space canvas').waitFor();
    const s=await page.evaluate(()=>Object.fromEntries(new URLSearchParams(window.reviewRoute.search)));
    assert.deepEqual([s.sblock,s.smodule,s.sindex],['29','mlp','17216']);
    await page.getByRole('button',{name:'返回测试数据库',exact:true}).click();await db.locator('#db-search').waitFor();assert.equal(await db.locator('#db-search').inputValue(),'U17216');
    await reset();await click('branch','语法规则');assert.equal(await count().then(a=>a.length),0);
    await reset();await click('branch','知识网络');await click('branch','知识网络/属性');await click('branch','SEM.ATTRIBUTE');await click('design');assert.match(await dialog.innerText(),/设计参考答案/);await close();
    assert.deepEqual(await count(),['324 道 ↗','3,136 道 ↗']);
    await set('lang','中文');await db.locator('#db-search').fill('鲨鱼');assert.deepEqual(await count(),['32 道 ↗']);
    await click('cases',322);await click('case');assert.match(await dialog.innerText(),/候选对数得分/);await checkFiles();await close();
    await reset();if(!await db.locator('[data-filter="arm"]').count())await click('advanced');await set('arm','cut');
    await set('outcome','wrong');await db.locator('#db-search').fill('草莓 番茄');
    assert.deepEqual(await count(),['1 道 ↗']);await click('cases',3008);await click('case','44');assert.match(await dialog.innerText(),/different/);await checkFiles();await close();
    await reset();await set('archive','本稿附中心回执');assert.deepEqual(await count(),['192 道 ↗','324 道 ↗']);
    await click('study',3008);await dialog.getByText('展开查重、运行与八份文件',{exact:true}).click();await checkFiles();await close();
    await reset();await set('outcome','rescued');assert.equal(await db.locator('.db-empty').count(),1);
    await reset();await db.locator('#db-search').fill('confirm-003-08');assert.deepEqual(await count(),['1 道 ↗','1 道 ↗']);
    await reset();await db.locator('#db-search').fill('MA-H0-v5');assert.deepEqual(await count(),['3,136 道 ↗']);await reset();
    await db.locator('#db-search').fill('qwen14b B26/H36 按物理首行 头选错、模型答对');assert.deepEqual(await count(),['339 道 ↗','303 道 ↗']);await reset();
    await page.setViewportSize({width:390,height:844});await page.reload();await db.locator('#db-search').waitFor();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    assert.equal(await db.locator('.db-tree-body').isVisible(),false);await click('tree-toggle');await click('branch','语法规则');await reset();await click('tree-toggle');
    await shot('mobile');await click('preset','head');await click('cases',2990);await click('case');await close();
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const report={passed:true,browser:browser.version(),studies:5,promptsWithinStudies:6724,observations:rawN,
      physicalDisagreements:{2983:339,2990:303},objectRoundTrip:'Qwen3-14B L29/MLP/17216',checkedLocalFiles:files.size,network,errors,viewports:[[1440,920],[390,844]]};
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
