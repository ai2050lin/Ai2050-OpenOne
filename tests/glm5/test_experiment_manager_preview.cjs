const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_CORE||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root=path.resolve(__dirname,'../..');
const file=pathToFileURL(path.join(root,'research/glm5/docs/previews/client-test-database.html')).href;
const out=path.join(__dirname,'result/experiment_manager_preview'),tmp=path.join(root,'tests/glm5_temp/experiment_manager_preview');

async function main(){
  for(const p of [out,tmp])fs.mkdirSync(p,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||
    path.join(require('node:os').homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe'),
    env:{...process.env,TEMP:tmp,TMP:tmp,TMPDIR:tmp},downloadsPath:tmp});
  try{
    const ctx=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true});
    const page=await ctx.newPage();page.setDefaultTimeout(6000);
    const errors=[],network=[];
    page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
    await ctx.route(/^https?:/,r=>r.abort());
    const manager=page.getByRole('region',{name:'实验管理',exact:true});
    const advanced=page.getByText('更多工具',{exact:true});
    const go=station=>page.goto(file+'#?wb=1&lens=process&station='+station+'&mdl=Qwen3-14B');
    const home=async()=>{await manager.waitFor();assert.equal(await page.locator('.wb-subnav:visible').count(),0);assert.equal(await manager.locator('li').count(),2);};
    await go('execution');await home();
    assert.equal(await manager.locator('details[open]').count(),0);
    assert.equal(await page.getByRole('button',{name:'队列与审计',exact:true}).count(),0);
    assert.equal(await manager.locator('button:visible').count(),3);
    await page.screenshot({path:path.join(out,'desktop.png')});
    for(const phase of [3008,2997]){
      await page.getByRole('button',{name:'查看 Phase '+phase+' 结果'}).click();
      const dialog=page.locator('.wide-evidence dialog[open]');await dialog.waitFor();
      assert.match(await dialog.innerText(),new RegExp('Phase '+phase));assert.match(await dialog.innerText(),/研究问题/);
      await page.keyboard.press('Escape');assert.equal(await page.locator('.db-row').count(),1);
      await page.goBack();await home();
    }
    await page.getByRole('button',{name:'新建实验草稿'}).click();
    const draft=page.getByRole('dialog',{name:'新建实验草稿'});
    assert.equal(await draft.getByRole('button',{name:'下载草稿'}).isDisabled(),true);
    assert.equal(await draft.getByLabel('拟用模型').inputValue(),'Qwen3-14B');
    await draft.getByLabel('想验证什么问题？').fill('换一批姓名后还能选中同一行吗？');
    await draft.getByLabel('拟用模型').selectOption('GLM-4-9B');
    const [download]=await Promise.all([page.waitForEvent('download'),draft.getByRole('button',{name:'下载草稿'}).click()]);
    const content=fs.readFileSync(await download.path(),'utf8');await download.delete();
    assert.match(content,/换一批姓名后还能选中同一行吗/);assert.match(content,/GLM-4-9B/);
    await page.keyboard.press('Escape');await home();
    for(const [tool,heading] of [['队列与审计','实验与队列'],['材料与模板','材料与数据'],['研究计划','当前研发']]){
      if(!await manager.locator('details[open]').count())await advanced.click();
      await page.getByRole('button',{name:tool,exact:true}).click();
      await page.getByRole('heading',{name:heading,exact:true}).first().waitFor();
      await page.reload();await page.getByRole('heading',{name:heading,exact:true}).first().waitFor();
      await page.getByRole('button',{name:'← 返回实验管理'}).click();await home();
    }
    await go('plan');await home();
    await page.locator('.fw-rail').getByRole('button',{name:'测试数据库'}).click();await page.locator('#db-search').waitFor();
    await page.locator('.fw-rail').getByRole('button',{name:'实验管理'}).click();await home();
    await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png')});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.getByRole('button',{name:'新建实验草稿'}).click();await draft.waitFor();
    const box=await draft.boundingBox();assert.ok(box.x>=0&&box.x+box.width<=390);
    assert.deepEqual(network,[]);assert.deepEqual(errors,[]);
    const report={passed:true,browser:browser.version(),homeRoutes:['execution','plan','一级菜单'],archivedPhases:[3008,2997],
      verifiedTools:['队列与审计','材料与模板','研究计划'],downloadedDraft:true,viewports:[[1440,900],[390,844]],errors,network};
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
