const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {pathToFileURL}=require('node:url');
const {createHash}=require('node:crypto');
const {checkWrite,checkedDir}=require('../../frontend/electron/storagePolicy');
const {chromium}=require(process.env.PLAYWRIGHT_CORE||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root=path.resolve(__dirname,'../..'),file=pathToFileURL(path.join(root,'research/glm5/docs/previews/client-distributed-testing.html')).href;
const out=process.env.AI2050_PREVIEW_RESULTS || path.join(__dirname,'result/distributed_participation_preview');
const headArchive=require('../../frontend/src/components/v8/threeCueEvidence.json').runs[0].families[0];
async function test(){
  checkWrite(out,300*2**20,'跨页浏览器回归');
  checkedDir(out,40*2**20,'跨页浏览器验收证据');
  const tempParent=checkedDir(path.join(root,'tests/glm5_temp'),256*2**20,'跨页浏览器临时配置');
  const temp=fs.mkdtempSync(path.join(tempParent,'distributed-preview-'));
  for(const key of ['TMP','TEMP','TMPDIR'])process.env[key]=temp;
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||path.join(os.homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe'),
    downloadsPath:temp,env:{...process.env,TEMP:temp,TMP:temp,TMPDIR:temp}});
  try{
    const ctx=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true}),page=await ctx.newPage(),errors=[],network=[];
    ctx.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});page.on('pageerror',e=>errors.push(e.message));
    await ctx.route(/^https?:/,r=>r.abort());page.setDefaultTimeout(9000);
    const home=page.locator('.dp-home'),status=home.locator('[role=status] strong'),graph=page.locator('.dp-atlas'),dlg=page.locator('.dp-dialog[open]');
    const btn=(p,name)=>p.getByRole('button',{name,exact:true});
    const isStatus=label=>status.getByText(label,{exact:true}).waitFor();
    const goHome=async()=>{await btn(page.locator('.fw-rail'),'实验管理').click();await home.waitFor();};
    const shot=name=>page.screenshot({path:path.join(out,name+'.png')});
    async function download(button){
      const [d]=await Promise.all([page.waitForEvent('download'),button.click()]);
      const bytes=fs.readFileSync(await d.path());await d.delete();return bytes;
    }
    const hash=b=>createHash('sha256').update(b).digest('hex');
    const sources=[];
    await page.goto(file);await home.waitFor();await isStatus('尚未参与');await shot('desktop');
    const rail=page.locator('.fw-rail'),tabs=page.getByRole('tablist',{name:'总览视图'});
    const overview=tabs.getByRole('tab',{name:'总览',exact:true}),database=tabs.getByRole('tab',{name:'测试数据库'});
    const search=page.locator('#db-search'),records=page.locator('.db-row');
    const archiveLink=file+'#?wb=1&observeMode=archive&observeTab=run&observePlugin=object&lens=spatial&view=graph&gview=structure';
    await page.goto(archiveLink);await page.reload();
    const spatial=page.getByRole('navigation',{name:'语言与模型导航'});
    await btn(spatial,'模型分析').waitFor();await page.locator('.ow-graph[aria-label="模型分析"]').waitFor();
    await btn(spatial,'语言图谱').click();await page.waitForURL(/gview=language/);
    await page.goBack();await page.locator('.ow-graph[aria-label="模型分析"]').waitFor();
    assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(2)).get('observeMode'),'archive');
    await btn(rail,'总览').click();await page.getByRole('heading',{name:'研究总览'}).waitFor();
    assert.deepEqual(await rail.getByRole('button').allTextContents(),['总览','语言与模型','实验管理','系统设置']);
    assert.deepEqual(await tabs.getByRole('tab').allTextContents(),['总览','测试数据库']);
    assert.deepEqual(await page.locator('.rv-counts dd').allTextContents(),['5 项','2 种','2 项']);
    await shot('overview');await overview.focus();await page.keyboard.press('ArrowRight');await search.waitFor();assert.equal(await records.count(),5);
    await search.fill('2997');await page.locator('[data-db-pick="2997"]').check();
    await overview.click();await page.reload();await page.getByRole('heading',{name:'研究总览'}).waitFor();
    await database.click();await search.waitFor();assert.equal(await search.inputValue(),'2997');assert.ok(await page.locator('[data-db-pick="2997"]').isChecked());
    await page.goBack();await page.getByRole('heading',{name:'研究总览'}).waitFor();await page.goForward();await search.waitFor();
    await btn(rail,'总览').click();await page.locator('.rv-archives button').filter({hasText:'Phase 3008'}).click();
    const detail=page.locator('.wide-evidence dialog[open]');await detail.waitFor();
    assert.match(await detail.innerText(),/Phase 3008/);assert.equal(await database.getAttribute('aria-selected'),'true');await page.keyboard.press('Escape');
    for(const tail of ['rtab=rules&db=q%3D2990','db=q%3D2990']){
      await page.goto(file+'#?wb=1&lens=route&'+tail);await page.reload();await search.waitFor();
      assert.equal(await search.inputValue(),'2990');assert.equal(await records.count(),1);
    }
    await goHome();
    await btn(home,'进入语言与模型 →').click();await graph.getByLabel('查找可载入测试').waitFor();
    const q=graph.getByLabel('查找可载入测试');await q.fill('xyz');await graph.getByText(/没有匹配的研究/).waitFor();await q.fill('');
    for(const phase of [3008,2997]){
      await btn(graph.locator('article').filter({hasText:'Phase '+phase}),'载入图谱').click();await graph.locator('.dp-bars').waitFor();
      await page.reload();await graph.locator('.dp-bars').waitFor();assert.match(page.url(),new RegExp('studyPack='+phase));
      const bytes=await download(btn(graph,'下载逐题结果')),rows=bytes.toString('utf8').trim().split(/\r?\n/).map(l=>JSON.parse(l));
      const receiptFile=path.join(__dirname,phase===3008?'result/color_reuse_v1/greedy/study-receipt.json':'result/routine/run-t7dfn8_7/server-receipt.json');
      const doc=JSON.parse(fs.readFileSync(receiptFile)),r=doc.receipt||doc;
      const query=new URLSearchParams(new URL(page.url()).hash.slice(2));assert.equal(query.get('run'),r.run_id);assert.equal(query.get('sblock'),'29');
      assert.equal(hash(bytes),r.artifacts.find(a=>a.name==='case_results.jsonl').sha256);
      const groups=Map.groupBy(rows,row=>row.arm);
      assert.deepEqual(await graph.locator('.dp-bars b').allTextContents(),[...groups.values()].map(a=>a.filter(row=>row.correct).length+' / '+a.length));
      sources.push({phase,run:r.run_id,rows:rows.length,sha256:hash(bytes)});
      if(phase===3008){await shot('atlas');await btn(graph,'更换测试').click();}
      else{
        await btn(graph,'定位注意力头 B28/H21').click();assert.match(page.url(),/sindex=21/);assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(2)).get('run'),r.run_id);
        await btn(page.getByRole('group',{name:'观察模式'}),'测试记录').click();
        await page.getByRole('tab',{name:'成果与方法',exact:true}).click();
        await btn(graph,'选择测试数据').click();await btn(graph.locator('article').filter({hasText:'Phase 2997'}),'载入图谱').click();
        await btn(graph,'查看题目和原始记录').click();await page.locator('.wide-evidence dialog[open]').waitFor();await page.keyboard.press('Escape');
      }
    }
    const compareLink=file+'#?wb=1&lens=spatial&view=response';
    await page.goto(compareLink);await page.reload();
    const workbench=page.locator('.ow-graph'),scene=workbench.locator('.aw-space'),tools=workbench.getByRole('group',{name:'分析插件'});
    const modes=scene.getByRole('group',{name:'观察模式'});
    const mode=name=>btn(modes,name);
    const graphTabs=workbench.getByRole('tablist',{name:'控制台功能'});
    const verifyTabs=(...names)=>graphTabs.getByRole('tab').allTextContents().then(actual=>assert.deepEqual(actual,names));
    const verifyLiveSections=async()=>{
      const liveArea=workbench.getByRole('region',{name:'实时模式工作区'});
      assert.ok(await liveArea.isVisible() && await tools.isVisible());
      const [a,b]=await Promise.all([liveArea.boundingBox(),tools.boundingBox()]);
      assert.ok(a.y+a.height<=b.y+1);
    };
    await scene.locator('canvas').waitFor();
    assert.deepEqual(await modes.getByRole('button').allTextContents(),['实时模式','测试记录','特征分析']);
    assert.equal(await mode('测试记录').getAttribute('aria-pressed'),'true');
    await verifyTabs('运行与对照','插件分析','成果与方法');
    assert.equal(await workbench.getByLabel('模型',{exact:true}).count(),1);
    assert.equal(await workbench.getByRole('tab',{name:'三线索对照'}).getAttribute('aria-selected'),'true');
    assert.ok((await graphTabs.boundingBox()).y<690);
    await btn(scene,'显示插件').click();
    const display=scene.getByRole('group',{name:'显示插件'});
    const layerToggle=display.getByLabel('HiddenState 网格',{exact:true});
    await layerToggle.uncheck();assert.equal(await layerToggle.isChecked(),false);
    await layerToggle.check();await page.keyboard.press('Escape');await display.waitFor({state:'hidden'});
    await btn(workbench,'在 Head 中观察 B26/H36 →').click();
    const head=workbench.getByRole('region',{name:'B26/H36 归档注意力'});
    await head.waitFor();
    assert.equal(await graphTabs.getByRole('tab',{name:'插件分析'}).getAttribute('aria-selected'),'true');
    assert.match(await scene.locator('.ow-selection strong').innerText(),/B26\/H36/);
    for(const key of ['confirm-000-00','confirm-000-02']){
      await head.getByLabel('归档样本').selectOption(key);
      const sample=headArchive.cases.find(row=>row.id===key);
      const values=await head.locator('progress').evaluateAll(elements=>elements.map(element=>element.value));
      assert.deepEqual(values,sample.order.map(index=>sample.mass[index]));
    }
    await shot('graph-head');
    await btn(scene,'⛶ 全屏').click();
    const floating=workbench.locator('.ow-float');
    await floating.waitFor();
    assert.equal((await scene.boundingBox()).height,900);
    assert.ok(await head.isVisible());
    assert.ok(await btn(tools,'Head').isVisible());
    assert.equal(await floating.getByLabel('全屏层').inputValue(),'27');
    const grip=await floating.locator('.ow-grip').boundingBox();
    await page.mouse.move(grip.x+40,grip.y+12);await page.mouse.down();
    await page.mouse.move(grip.x+970,5,{steps:6});await page.mouse.up();
    assert.ok((await floating.boundingBox()).x>800);
    const header=await scene.locator('.ow-stage-nav').boundingBox();
    assert.ok((await floating.boundingBox()).y>=header.y+header.height);
    await shot('graph-fullscreen');

    // Keep one camera and one canvas as the source of observations changes.
    await scene.locator('canvas').evaluate(element=>{window.previewCanvas=element;});
    const sameCanvas=()=>scene.locator('canvas').evaluate(element=>element===window.previewCanvas);
    await mode('实时模式').click();
    const live=workbench.getByRole('region',{name:'实时模式工作区'});
    await live.waitFor();
    await verifyLiveSections();
    assert.ok(await sameCanvas());
    assert.match(await scene.locator('.ow-selection strong').innerText(),/B26\/H36/);
    await btn(live,'回放 B36 实测样例').click();
    await live.locator('output').waitFor();
    assert.equal(await scene.locator('.ow-selection strong').innerText(),'B36/U2429');
    const trackedValue=live.locator('output');
    await btn(live,'Code 前').click();
    const before=await trackedValue.innerText();
    await btn(live,'输入末尾').click();
    const after=await trackedValue.innerText();
    assert.notEqual(after,before);
    assert.equal(await btn(live,'输入末尾').getAttribute('aria-pressed'),'true');
    await shot('graph-live-fullscreen');
    await live.getByLabel('运行文本').fill('本机下一次观察的输入草稿');
    assert.equal(await btn(live,'开始运行').isDisabled(),true);
    await btn(tools,'MLP').click();
    const mlp=workbench.locator('#console-panel-plugin');
    assert.equal(await trackedValue.count(),0);
    assert.equal(await mlp.locator('output').count(),0);
    assert.ok(await live.getByLabel('运行文本').isVisible());
    const liveBox=await live.boundingBox(),pluginBox=await tools.boundingBox();
    assert.ok(liveBox.y+liveBox.height<=pluginBox.y+1);
    await btn(live,'回放 B36 实测样例').click();
    await btn(live,'输入末尾').click();await btn(tools,'MLP').click();
    assert.equal(await live.getByLabel('运行文本').inputValue(),'本机下一次观察的输入草稿');
    assert.equal(await mlp.locator('output').innerText(),after);
    const renderedActivation=await scene.locator('canvas').evaluate(element=>element.toDataURL());
    await btn(scene,'显示插件').click();
    const colors=display.getByLabel('数值着色');
    await colors.uncheck();
    await page.keyboard.press('Escape');await display.waitFor({state:'hidden'});
    assert.equal(await mlp.locator('output').innerText(),after);
    await page.waitForFunction(previous=>window.previewCanvas.toDataURL()!==previous,renderedActivation);
    await btn(scene,'显示插件').click();
    await colors.check();await page.keyboard.press('Escape');
    await btn(workbench,'管理插件').click();
    const manager=workbench.getByRole('group',{name:'插件管理'});
    await manager.getByLabel('HiddenState',{exact:true}).uncheck();
    await btn(scene,'显示插件').click();
    assert.equal(await layerToggle.isEnabled(),true);
    assert.equal(await layerToggle.isChecked(),true);
    await page.keyboard.press('Escape');
    await manager.getByLabel('HiddenState',{exact:true}).check();
    await btn(workbench,'管理插件').click();

    await mode('特征分析').click();
    const features=workbench.getByRole('region',{name:'特征分析工作区'});
    await features.waitFor();
    await verifyTabs('对象详情','插件分析');
    assert.ok(await sameCanvas());
    assert.match(await features.innerText(),/B36\/U2429/);
    assert.equal(await features.getByRole('row',{name:/输入末尾/}).locator('td').last().innerText(),after);
    await btn(features,'原生单元组').click();
    await btn(features,'在 3D 中标出整组').click();
    assert.equal(await features.getByText('90 / 92 家族正确',{exact:true}).count(),2);
    await shot('graph-features-fullscreen');
    await btn(features,'稀疏字典特征').click();
    assert.match(await features.innerText(),/SAE 字典/);
    await mode('实时模式').click();
    await verifyLiveSections();
    assert.equal(await live.getByLabel('运行文本').inputValue(),'本机下一次观察的输入草稿');
    const modelPicker=scene.getByLabel('模型',{exact:true});
    await modelPicker.selectOption({label:'Qwen3-4B'});
    await live.locator('output').waitFor({state:'hidden'});
    assert.equal(await live.locator('output').count(),0);
    await btn(live,'回放 B36 实测样例').click();
    await live.getByRole('group',{name:'采集位置'}).waitFor();
    assert.ok(await sameCanvas());
    await mode('测试记录').click();
    assert.ok(await sameCanvas());
    assert.ok(await floating.isVisible());
    await btn(scene,'精确定位').click();
    const locator=scene.getByRole('form',{name:'原生索引定位'});
    await locator.getByLabel('定位对象类型').selectOption('attn');
    await locator.locator('[name=layer]').fill('27');
    await locator.locator('[name=index]').fill('36');
    await btn(locator,'定位单元').click();
    await btn(tools,'Head').click();await head.waitFor();
    assert.equal(await head.getByLabel('归档样本').inputValue(),'confirm-000-02');
    await head.getByRole('link',{name:'打开历史测试',exact:true}).click();await detail.waitFor();
    assert.match(await detail.innerText(),/Phase 2983[\s\S]*B26\/H36/);
    assert.equal(await search.inputValue(),'confirm-000-02');
    await btn(detail,'查看当前条件下的命中题目').click();
    await btn(detail,'confirm-000-02').click();assert.match(await detail.innerText(),/confirm-000-02[\s\S]*silver/);
    await page.goBack();await head.waitFor();
    assert.equal(await head.getByLabel('归档样本').inputValue(),'confirm-000-02');
    assert.equal(await scene.evaluate(node=>node.classList.contains('max')),true);
    await page.setViewportSize({width:390,height:844});
    assert.ok(await workbench.evaluate(element=>element.scrollWidth<=element.clientWidth));
    await floating.waitFor();await btn(scene,'退出全屏（Esc）').click();await floating.waitFor({state:'hidden'});
    await btn(scene,'⛶ 全屏').click();await floating.waitFor();
    const exit=await btn(scene,'退出全屏（Esc）').boundingBox();
    assert.ok(exit.x>=0&&exit.x+exit.width<=390);
    await mode('实时模式').click();await live.waitFor();
    await verifyLiveSections();
    await mode('特征分析').click();await features.waitFor();
    await verifyTabs('对象详情','插件分析');
    await mode('测试记录').click();await head.waitFor();
    await verifyTabs('运行与对照','插件分析','成果与方法');
    await page.keyboard.press('Escape');await floating.waitFor({state:'hidden'});
    await page.setViewportSize({width:1440,height:900});

    await goHome();await isStatus('尚未参与');
    await btn(home,'系统测试方案').click();await dlg.waitFor();assert.match(await dlg.innerText(),/24 个分片/);
    await dlg.getByLabel('模型范围').selectOption('both');await dlg.getByLabel('每组家族数').fill('65');
    const plan=JSON.parse(await download(btn(dlg,'导出分片草案')));assert.equal(plan.status,'draft');assert.equal(plan.shards.length,36);
    for(const s of Map.groupBy(plan.shards,x=>[x.topic,x.model,x.language].join('/')).values()){
      assert.deepEqual(s.map(x=>[x.first_family,x.last_family]),[[1,32],[33,64],[65,65]]);
    }
    await dlg.getByLabel('每个分片最多家族数').fill('0');assert.ok(await btn(dlg,'导出分片草案').isDisabled());
    await btn(dlg,'实验设计与草稿').click();await page.locator('.cx-page').waitFor();await goHome();
    await btn(home,'查看完整队列').click();await page.getByRole('heading',{name:'实验与队列',exact:true}).waitFor();await goHome();
    // Run the preview clock deterministically. UI still uses real click, navigation and paint.
    await page.clock.install();
    await btn(home,'一键参与').click();await isStatus('匹配并领取');await page.clock.runFor(1250);await isStatus('正在测试');await shot('running');
    await btn(home,'停止参与').click();await page.clock.runFor(7000);await isStatus('已停止');
    await home.locator('summary').click();
    for(const [id,label] of [['empty','暂无匹配任务'],['model','模型待准备']]){
      await home.getByLabel('模拟情况').selectOption(id);await btn(home,'一键参与').click();await page.clock.runFor(1300);await isStatus(label);
    }
    await home.getByLabel('模拟情况').selectOption('network');await btn(home,'一键参与').click();
    await page.clock.runFor(4800);await isStatus('正在上传');await page.clock.runFor(1700);await isStatus('等待续传');
    await btn(home,'继续上传').click();await isStatus('正在上传');await page.clock.runFor(1700);await isStatus('中心核验');
    assert.match(await home.locator('.dp-count').innerText(),/3 \/ 4/);
    await btn(home,'进入语言与模型 →').click();await page.clock.runFor(1500);await goHome();await isStatus('已汇总');assert.match(await home.locator('.dp-count').innerText(),/4 \/ 4/);
    await page.reload();await isStatus('尚未参与');
    for(const width of [390,320]){
      await page.setViewportSize({width,height:844});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      const rect=await btn(home,'一键参与').boundingBox();assert.ok(rect.x>=0&&rect.x+rect.width<=width);
      if(width===390)await shot('mobile');
      await btn(home,'系统测试方案').click();const d=await dlg.boundingBox();assert.ok(d.x>=0&&d.x+d.width<=width);await btn(dlg,'关闭').click();
      await btn(rail,'总览').click();await overview.waitFor();assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      await database.click();await search.waitFor();const box=await database.boundingBox();assert.ok(box.x>=0&&box.x+box.width<=width);
      await goHome();
    }
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const report={passed:true,browser:browser.version(),desktop:[1440,900],mobileWidths:[390,320],overviewTabs:['总览','测试数据库'],spatialNavigation:['语言与模型','模型分析','语言图谱'],archiveLinkRestored:true,legacyDatabaseLinks:true,filterAndSelectionRestored:true,downloadedAndVerified:sources,planShards:36,
      graph:{singleModelSelector:true,sceneDisplay:true,displayIndependentOfAnalysis:true,head:'B26/H36',archivedPhase:2983,
        liveReadoutPhase:2933,observationValuesChanged:true,featureCard:true,threeModesSharedCanvas:true,fullWindowScene:true,
        floatDragged:true,historyRestored:true,narrowExit:true},
      scenarios:['完成','立即停止','缺少模型','没有匹配任务','中断后继续上传','切换菜单后继续','图谱历史结果独立加载'],errors,network};
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
  }catch(error){
    const page=browser.contexts()[0]?.pages()[0];
    if(page)console.error('Failed route',page.url(),(await page.locator('body').innerText()).slice(-1700));
    throw error;
  }finally{
    await browser.close();
    const actual=fs.realpathSync.native(temp);assert.equal(path.dirname(actual),fs.realpathSync.native(tempParent));
    assert.ok(path.basename(actual).startsWith('distributed-preview-'));fs.rmSync(actual,{recursive:true,force:true,maxRetries:4,retryDelay:150});
  }
}
test().catch(e=>{console.error(e);process.exitCode=1;});
