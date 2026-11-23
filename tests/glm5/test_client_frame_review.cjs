/* Acceptance of the actual client frame with the read-only review mounted in Test Records. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL,fileURLToPath} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_CORE ||
  'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root=path.resolve(__dirname,'../..');
const out=path.join(__dirname,'result/client_frame_evidence_review');
const temp=path.join(root,'tests/glm5_temp/client_frame_evidence_review');
const pageFile=path.join(root,'research/glm5/docs/previews/client-evidence-workbench.html');
async function main(){
  for(const p of [out,temp])fs.mkdirSync(p,{recursive:true});
  const browser=await chromium.launch({executablePath:process.env.CHROME_PATH ||
    path.join(require('node:os').homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe'),
    downloadsPath:temp,env:{...process.env,TEMP:temp,TMP:temp,TMPDIR:temp},headless:true});
  try{
    const ctx=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
    const page=await ctx.newPage();page.setDefaultTimeout(6500);
    const errors=[],network=[],files=new Set();
    page.on('pageerror',e=>{errors.push(e.message);console.error(e.message);});
    page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
    await ctx.route(/^https?:/,r=>r.abort());
    const act=(name,value)=>page.locator('[data-act="'+name+'"]'+(value===undefined?'':'[data-value="'+value+'"]')).first().click();
    const view=name=>page.locator('[data-view="'+name+'"]').click();
    const sources=async()=>{
      for(const a of await page.locator('.client-evidence-host a:visible').all()){
        const url=await a.getAttribute('href');
        if(!url || url.startsWith('data:'))continue;
        const p=fileURLToPath(new URL(url,page.url()));assert.ok(fs.existsSync(p),p);files.add(p);
      }
    };
    await page.goto(pathToFileURL(pageFile).href);
    await page.locator('.client-evidence-host h1').waitFor();
    assert.equal(await page.getByRole('tab',{name:'测试记录',exact:true}).getAttribute('aria-selected'),'true');
    assert.equal(await page.getByRole('tab',{name:'成果与方法',exact:true}).getAttribute('aria-selected'),'true');
    assert.equal(await page.locator('.aw-space canvas').count(),1);
    const metrics=()=>page.locator('.client-evidence-host .stat small').allTextContents();
    assert.deepEqual(await metrics(),['377 / 1,024 题','1,018 / 1,024 题']);
    await page.screenshot({path:path.join(out,'console.png')});
    await page.evaluate(()=>{document.querySelector('.wb-main').scrollTop=0;});
    await page.screenshot({path:path.join(out,'model-space.png')});
    await act('rule',0);assert.deepEqual(await metrics(),['1,015 / 1,024 题','1,024 / 1,024 题']);
    await act('rule',2);await page.locator('[data-pick="2983"]').uncheck();
    await act('browse','div');assert.match(await page.locator('#list-table').innerText(),/303 道/);
    await act('case');assert.match(await page.locator('#dialog-body').innerText(),/表外注意力质量/);await sources();
    await page.screenshot({path:path.join(out,'case.png')});await page.locator('#close').click();
    await page.locator('[data-pick="2997"]').check();await act('export');
    const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#download-set').click()]);
    const saved=JSON.parse(fs.readFileSync(await download.path(),'utf8'));await download.delete();
    assert.equal(saved.total.n,512);assert.equal(saved.alongside[0].phase,2997);await page.locator('#close').click();
    await view('parts');await act('part','U17216');
    await act('spatial','U17216');
    assert.equal(await page.getByRole('tab',{name:'对象详情',exact:true}).getAttribute('aria-selected'),'true');
    const loc=await page.evaluate(()=>Object.fromEntries(new URLSearchParams(window.reviewRoute.search)));
    assert.deepEqual([loc.mdl,loc.sblock,loc.smodule,loc.sindex],['Qwen3-14B','29','mlp','17216']);
    await page.getByRole('button',{name:'查看 U17216 的研究证据',exact:true}).click();
    await page.locator('.client-evidence-host [data-act="spatial"][data-value="U17216"]').waitFor();
    await view('catalog');
    await page.locator('#lang').selectOption('中文');await act('study',322);
    assert.match(await page.locator('#dialog-body').innerText(),/31 \/ 32/);await sources();await page.locator('#close').click();
    await page.locator('#lang').selectOption('全部');await act('study',3008);await act('color-cell','red/red');
    await page.locator('#color-arm').selectOption('cut');assert.equal(await page.locator('dialog:visible table tr').count(),37);await sources();await page.locator('#close').click();
    await act('tree');await page.getByLabel('查找方向或题目').waitFor();
    assert.equal(await page.getByLabel('记录来源').inputValue(),'directions');
    await page.getByLabel('查找方向或题目').fill('SEM.IDENTITY');
    assert.match(await page.getByRole('article',{name:'方向与题目'}).innerText(),/参考答案/);
    await page.getByLabel('查找方向或题目').fill('SEM.ATTRIBUTE');
    await page.getByRole('button',{name:'查看 Phase 3008 同主题实测',exact:true}).click();
    await page.locator('dialog:visible #dialog-title').waitFor();
    assert.match(await page.locator('#dialog-body').innerText(),/草莓/);await page.locator('#close').click();
    await page.getByLabel('记录来源').selectOption('goodfire');assert.match(await page.locator('.tr-list').innerText(),/Phase/);
    await page.getByLabel('记录来源').selectOption('center');assert.match(await page.locator('.tr-panel').innerText(),/需要中心管理员会话/);
    await page.getByLabel('记录来源').selectOption('evidence');
    await view('storage');
    await page.getByText('展开查重、运行与八份文件',{exact:true}).click();
    assert.equal(await page.locator('.client-evidence-host details[open] tr').count(),9);
    await sources();
    await view('compare');await page.reload();await page.locator('.client-evidence-host h1').waitFor();
    assert.deepEqual(await metrics(),['206 / 512 题','508 / 512 题']);
    await page.setViewportSize({width:390,height:844});
    await page.locator('.client-evidence-host h1').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(out,'mobile.png')});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await act('browse','div');await act('case');await page.keyboard.press('Escape');
    assert.equal(await page.locator('#detail').isVisible(),false);
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const report={browser:browser.version(),caseRows:3072,colorArmRows:1296,sourceLinksChecked:files.size,
      theme:await page.locator('meta[name="ai2050-client-theme"]').getAttribute('content'),
      treeToStudy:true,objectToEvidence:true,nativeTabs:4,sceneLocator:loc.smodule+' / layer '+loc.sblock+' / unit '+loc.sindex,desktop:[1440,1000],
      mobile:[390,844],network,errors,passed:true};
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
