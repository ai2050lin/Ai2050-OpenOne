const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {pathToFileURL,fileURLToPath}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_CORE||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root=path.resolve(__dirname,'../..');
const file=pathToFileURL(path.join(root,'research/glm5/docs/previews/client-current-experiments.html')).href;
const out=process.env.AI2050_PREVIEW_RESULTS || path.join(__dirname,'result/current_experiments_preview'),tmp=path.join(root,'tests/glm5_temp/current_experiments_preview');
async function main(){
  for(const dir of [out,tmp])fs.mkdirSync(dir,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||
    path.join(require('node:os').homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe'),
    env:{...process.env,TEMP:tmp,TMP:tmp,TMPDIR:tmp},downloadsPath:tmp});
  try{
    const ctx=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true});
    const page=await ctx.newPage();page.setDefaultTimeout(5500);
    const errors=[],network=[];
    page.on('pageerror',e=>errors.push(e.message));
    ctx.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
    await ctx.route(/^https?:/,r=>r.abort());
    const home=page.locator('.cx-page'),dialog=page.locator('.cx-modal[open]'),dbDialog=page.locator('.wide-evidence dialog[open]');
    const button=(area,name)=>area.getByRole('button',{name,exact:true});
    const close=()=>page.keyboard.press('Escape');
    const navigateHome=async()=>{await button(page.locator('.fw-rail'),'实验管理').click();await home.waitFor();};
    const start=async()=>{await button(home,'新建实验').click();await dialog.waitFor();};
    const snap=name=>page.screenshot({path:path.join(out,name+'.png')});
    async function exportFile(){
      const [d]=await Promise.all([page.waitForEvent('download'),button(dialog,'导出草稿').click()]);
      const content=fs.readFileSync(await d.path(),'utf8');await d.delete();return content;
    }
    await page.goto(file);await home.waitFor();
    assert.deepEqual(await page.locator('.fw-rail button').allInnerTexts(),['测试数据库','语言与模型','实验管理','系统设置']);
    assert.deepEqual(await home.getByRole('columnheader').allInnerTexts(),['实验名称','当前状态','下一步操作']);
    assert.equal(await page.locator('.wb-subnav:visible').count(),0);
    assert.equal(await home.locator('tbody tr').count(),0);
    assert.match(await home.innerText(),/当前实验状态未知/);await snap('desktop');
    await button(home,'预览状态示例').click();assert.deepEqual(await home.locator('.cx-status').allInnerTexts(),['归档失败','运行中','草稿']);await snap('example-states');
    await button(home,'查看原因 →').click();await dialog.getByText('审计与归档回执').click();assert.match(await dialog.innerText(),/上传中断/);await close();
    await button(home,'看进度 →').click();await dialog.getByText('队列与运行日志').click();assert.match(await dialog.innerText(),/正在采集/);
    await button(dialog,'查看完整队列').click();await page.getByRole('heading',{name:'实验与队列',exact:true}).waitFor();
    await button(page,'← 返回实验管理').click();await home.waitFor();
    await start();assert.ok(await button(dialog,'查相似测试').isDisabled());assert.equal(await dialog.getByLabel('拟用模型').inputValue(),'Qwen3-14B');
    await dialog.getByLabel('想验证什么问题？').fill('B26/H36 更换姓名还选首行吗？');await button(dialog,'查相似测试').click();
    assert.match(await dialog.innerText(),/Phase 2983/);assert.match(await dialog.innerText(),/Phase 2990/);
    await button(dialog.locator('article').filter({hasText:'Phase 2990'}),'复用已有结果').click();
    await dbDialog.waitFor();assert.match(await dbDialog.innerText(),/Phase 2990/);await close();await navigateHome();assert.equal(await home.locator('tbody tr').count(),0);
    await start();await dialog.getByLabel('想验证什么问题？').fill('颜色判断可以跨材料复用吗？');await button(dialog,'暂存草稿').click();
    await button(home,'继续设计 →').click();await dialog.getByText('草稿文件').click();assert.equal(JSON.parse(await exportFile()).draft.status,'draft');
    await button(dialog,'查相似测试').click();assert.match(await dialog.innerText(),/Phase 3008/);await snap('wizard');
    await button(dialog,'继续设计新实验').click();assert.ok(await button(dialog,'保存方案').isDisabled());
    const materials=dialog.getByLabel('选择测试材料');
    for(const phase of [2983,2990,2997,3008,322]){
      await materials.selectOption(String(phase));
      assert.ok(fs.existsSync(fileURLToPath(new URL(await dialog.getByRole('link').getAttribute('href'),file))));
    }
    await materials.selectOption('custom');await dialog.getByLabel('新增价值与检验方法').fill('保留同异判断条件，比较未见物体家族。');
    assert.ok(await button(dialog,'保存方案').isDisabled());await dialog.getByLabel('材料来源或设计').fill('新的物体与配对家族');
    await materials.selectOption('3008');await button(dialog,'保存方案').click();await button(home,'核对方案 →').click();await dialog.getByText('审计与归档回执').click();
    assert.match(await dialog.innerText(),/尚未取得/);const content=await exportFile(),saved=JSON.parse(content);assert.equal(saved.preflight_receipt,null);assert.equal(saved.draft.material,'3008');
    await close();await page.reload();await home.waitFor();assert.equal(await home.locator('tbody tr').count(),0);
    async function importFile(value){
      await start();await dialog.getByText('草稿文件').click();
      await dialog.getByLabel('导入草稿 JSON').setInputFiles({name:'draft.json',mimeType:'application/json',buffer:Buffer.from(value)});
    }
    await importFile('{}');await dialog.getByText(/无法读取该草稿/).waitFor();await close();
    for(let i=0;i<2;i++){
      await importFile(content);await dialog.getByText(/已载入草稿/).waitFor();assert.equal(await dialog.getByLabel('想验证什么问题？').inputValue(),saved.draft.question);
      await button(dialog,'查相似测试').click();await button(dialog,'继续设计新实验').click();await button(dialog,'保存方案').click();
      assert.equal(await home.locator('tbody tr').count(),1);
    }
    await button(home,'查看已归档实验 →').click();await page.locator('.db-row').first().waitFor();assert.equal(await page.locator('.db-row').count(),2);
    for(const p of [2997,3008]){
      await page.locator('.db-row').filter({hasText:'Phase '+p}).locator('.db-title').click();await dbDialog.getByText('运行日志、审计与回执').click();
      const links=await dbDialog.locator('details[open]').last().locator('a').evaluateAll(els=>els.map(e=>e.href));
      assert.equal(links.length,2);for(const link of links)assert.ok(fs.existsSync(fileURLToPath(link)));await close();
    }
    const syntax=page.locator('[data-db="branch"][data-value="语法规则"]');await syntax.click();await button(page,'研究计划').click();
    assert.match(await dbDialog.innerText(),/研究计划 · 语法规则/);assert.doesNotMatch(await dbDialog.innerText(),/知识网络/);
    const item=dbDialog.locator('[data-db="design"]').first();await item.click();assert.match(await dbDialog.innerText(),/设计参考答案/);await close();
    await button(page,'研究计划').click();await button(dbDialog,'查看项目研发计划').click();await page.getByRole('heading',{name:'当前研发',exact:true}).waitFor();await navigateHome();
    assert.equal(await home.locator('tbody tr').count(),1);
    await start();await dialog.getByLabel('想验证什么问题？').fill('xyzzz');await button(dialog,'查相似测试').click();assert.match(await dialog.innerText(),/未找到相同的关键词/);await close();
    for(const width of [390,320]){
      await page.setViewportSize({width,height:844});await button(home,'预览状态示例').click();
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      const table=await home.locator('table').boundingBox();assert.ok(table.x>=0&&table.x+table.width<=width+1);
      if(width===390)await snap('mobile');
      await button(home,'继续设计 →').click();const rect=await dialog.boundingBox();assert.ok(rect.x>=0&&rect.x+rect.width<=width);await close();
      await button(home,'返回本地清单').click();
    }
    assert.deepEqual(network,[]);assert.deepEqual(errors,[]);
    const report={passed:true,browser:browser.version(),entry:'实验管理 / 当前实验',desktop:[1440,900],mobileWidths:[390,320],
      similarStudyReuse:2990,materialLinks:[2983,2990,2997,3008,322],archiveReadbackLinks:[2997,3008],directionPlans:'语法规则',
      draftRoundtrip:true,repeatImportDeduplicated:true,incompletePlanBlocked:true,invalidFileHandled:true,emptySearchHandled:true,errors,network};
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
