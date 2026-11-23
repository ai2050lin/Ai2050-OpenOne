const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');

const root = path.resolve(__dirname,'../..');
const preview = pathToFileURL(path.join(root,'research/glm5/docs/previews/graph-exploration-flow-review.html')).href;
const output = path.join(__dirname,'result/graph_exploration_flow');
const temp = path.join(root,'tests/glm5_temp/graph_exploration_flow');
const evidence = require('../../frontend/src/components/v8/threeCueEvidence.json').runs[0];

async function run() {
  for (const dir of [output,temp]) fs.mkdirSync(dir,{recursive:true});
  const executablePath = process.env.CHROME_PATH || path.join(os.homedir(),
    'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
  Object.assign(process.env,{TEMP:temp,TMP:temp,TMPDIR:temp});
  const browser = await chromium.launch({headless:true,executablePath});
  try {
    const geometry=fs.readFileSync(path.join(root,'frontend/src/components/v8/assocLayerGeometry.js'),'utf8');
    const {ASSOC_VIEWS,assocProject}=await import('data:text/javascript;base64,'+Buffer.from(geometry).toString('base64'));
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[], network=[];
    page.on('pageerror',error=>errors.push(error.message));
    page.on('request',request=>{if(/^https?:/.test(request.url()))network.push(request.url());});
    await page.route(/^https?:/,route=>route.abort());
    page.setDefaultTimeout(5000);
    await page.goto(preview);
    const button = (name,scope=page)=>scope.getByRole('button',{name,exact:true});
    const tabs = page.getByRole('tablist',{name:'控制台任务'});
    const tab = name=>tabs.getByRole('tab',{name,exact:true});
    const task = page.locator('#task-panel');
    const tools = page.getByRole('group',{name:'插件系统'});
    const canvas = page.locator('#scene canvas');
    const workspace = page.locator('#model-scene');
    const consolePanel = page.locator('#console');
    const enter = button('⛶ 全屏');
    const exit = button('退出全屏（Esc）');
    const shot = name=>page.screenshot({path:path.join(output,name+'.png'),fullPage:true});
    await task.getByRole('heading',{name:'B23 / H0'}).waitFor();
    await canvas.waitFor();
    assert.deepEqual(await tabs.getByRole('tab').allTextContents(), ['对象详情','运行与对照','插件分析','成果与方法']);
    assert.deepEqual(await tools.getByRole('button').allTextContents(), ['HiddenState','MLP','Head','Logit Lens','管理插件']);
    const scene = await page.getByRole('region',{name:'模型结构'}).boundingBox();
    const dock = await tools.boundingBox();
    assert.ok(dock.y>=scene.y+scene.height);
    assert.equal(await button('运行响应').count(),0);
    assert.equal(await button('条件对照').count(),0);

    const originalCanvasHeight = (await canvas.boundingBox()).height;
    await enter.click();
    await exit.waitFor();
    assert.ok(originalCanvasHeight<1000);
    assert.deepEqual(await canvas.boundingBox(),{x:0,y:0,width:1440,height:1000});
    assert.equal(await page.evaluate(()=>document.fullscreenElement),null);
    assert.equal(await consolePanel.evaluate(element=>element.parentElement.id),'model-scene');
    assert.equal(await page.locator('.rail').evaluate(element=>element.inert),true);

    await button('精确定位').click();
    assert.equal(await page.evaluate(()=>document.activeElement.id),'layer');
    await page.locator('#layer').fill('27');
    await page.locator('#layer').press('Tab');
    await page.locator('#index').fill('36');
    await page.locator('#index').press('Tab');
    await button('完成',page.locator('#scene-locator')).click();
    const status = await page.locator('#context-status').innerText();
    assert.match(status,/Qwen3-14B.*27.*B26 \/ H36.*1e970d119494/);
    assert.equal(await page.locator('#fullscreen-layer').inputValue(),'27');
    await page.locator('#fullscreen-layer').press('ArrowRight');
    assert.equal(await page.locator('#layer').inputValue(),'28');
    await page.locator('#fullscreen-layer').press('ArrowLeft');
    assert.equal(await page.locator('#context-status').innerText(),status);

    // Verify the drawn native grid itself can select a head, outside the floating console.
    const [yaw,pitch]=ASSOC_VIEWS.iso;
    for(const index of [35,36]){
      const x=-8+151*(index%8+.5)/8;
      const z=-147+113*(Math.floor(index/8)+.5)/5;
      const [screenX,screenY]=assocProject({yaw,pitch,W:1440,H:1000,zoom:1},x,20,z);
      await page.mouse.move(screenX,screenY);
      await page.locator('.aw-tip').getByText('B26 / H'+index,{exact:true}).waitFor();
      await page.mouse.click(screenX,screenY);
      assert.equal(await page.locator('#index').inputValue(),String(index));
    }
    const dragBar=page.locator('#console-bar');
    const title=await dragBar.boundingBox();
    const previousX=(await consolePanel.boundingBox()).x;
    await page.mouse.move(title.x+30,title.y+14);
    await page.mouse.down();
    await page.mouse.move(title.x+1000,title.y+14,{steps:8});
    await page.mouse.up();
    assert.equal((await consolePanel.boundingBox()).x,previousX+970);
    await page.screenshot({path:path.join(output,'located.png')});

    await button('Head',tools).click();
    assert.equal(await tab('插件分析').getAttribute('aria-selected'),'true');
    assert.equal(await page.locator('#context-status').innerText(),status);
    assert.equal(await page.locator('#head-sample').inputValue(),'2');

    async function assertCase(id) {
      const family=evidence.families[0], sample=family.cases.find(item=>item.id===id);
      const bars=task.getByRole('region',{name:'归档注意力分布'}).locator('progress');
      assert.deepEqual(await bars.evaluateAll(values=>values.map(item=>item.value)),sample.order.map(item=>sample.mass[item]));
      const strongest=sample.mass.indexOf(Math.max(...sample.mass));
      assert.deepEqual(await task.getByRole('region',{name:'历史测试来源'}).locator('dd').allTextContents(),[
        '第 '+(sample.order.indexOf(strongest)+1)+' 行','第 '+(sample.order.indexOf(sample.gold)+1)+' 行',sample.output,
      ]);
    }
    await assertCase('confirm-000-02');
    await page.locator('#head-sample').selectOption('0');
    await assertCase('confirm-000-00');
    await page.locator('#head-sample').selectOption('2');

    await button('管理插件',tools).click();
    const manager=page.getByRole('dialog',{name:'管理分析插件'});
    await manager.waitFor();
    assert.equal(await manager.getByLabel('Head',{exact:true}).isChecked(),true);
    await button('关闭',manager).click();
    await manager.waitFor({state:'hidden'});
    const archiveLink=task.getByRole('link',{name:'打开历史测试 →'});
    const sceneBeforeScroll=await canvas.boundingBox();
    await archiveLink.scrollIntoViewIfNeeded();
    assert.deepEqual(await canvas.boundingBox(),sceneBeforeScroll);
    await page.screenshot({path:path.join(output,'fullscreen-head.png')});
    await exit.click();
    await enter.waitFor();
    assert.equal(await page.locator('#context-status').innerText(),status);
    assert.equal(await consolePanel.evaluate(element=>element.parentElement.id),'console-dock');
    await assertCase('confirm-000-02');
    await shot('head');
    await enter.click();
    await exit.waitFor();

    await archiveLink.click();
    const record = page.locator('.wide-evidence dialog[open]');
    await record.waitFor();
    assert.match(await record.innerText(), /Phase 2983[\s\S]*B26\/H36/);
    const facets=page.locator('.wide-evidence');
    assert.equal(await facets.locator('[data-filter="model"]').inputValue(),'Qwen3-14B');
    assert.equal(await facets.locator('[data-filter="part"]').inputValue(),'B26/H36');
    assert.equal(await facets.locator('#db-search').inputValue(),'confirm-000-02');
    await button('查看当前条件下的命中题目',record).click();
    await button('confirm-000-02',record).click();
    assert.match(await record.innerText(),/confirm-000-02[\s\S]*silver/);
    await page.screenshot({path:path.join(output,'history.png')});
    await page.goBack();
    await task.getByRole('heading',{name:'Head · B26 / H36'}).waitFor();
    await enter.waitFor();
    await page.reload();
    await task.getByRole('heading',{name:'Head · B26 / H36'}).waitFor();
    assert.equal(await page.locator('#context-status').innerText(),status);
    await assertCase('confirm-000-02');

    await canvas.scrollIntoViewIfNeeded();
    const normalCanvas=await canvas.boundingBox();
    const position={x:normalCanvas.x+normalCanvas.width*.68,y:normalCanvas.y+normalCanvas.height*.45};
    async function assertSceneChanged(action) {
      const tick=Number(await canvas.getAttribute('data-tick'));
      const before=await canvas.evaluate(element=>element.toDataURL());
      await action();
      await page.waitForFunction(last=>Number(document.querySelector('#scene canvas').dataset.tick)>last,tick);
      const after=await canvas.evaluate(element=>element.toDataURL());
      assert.notEqual(after,before);
    }
    for(const mouseButton of ['left','right']){
      await assertSceneChanged(async()=>{
        await page.mouse.move(position.x,position.y);
        await page.mouse.down({button:mouseButton});
        await page.mouse.move(position.x+32,position.y+18,{steps:4});
        await page.mouse.up({button:mouseButton});
      });
    }
    await assertSceneChanged(()=>page.mouse.wheel(0,-120));
    await page.locator('#scene-view').selectOption('top');
    assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(1)).get('angle'),'top');

    await page.locator('#model').selectOption('Qwen3-4B');
    assert.equal(await task.locator('progress').count(),0);
    await button('定位 B26/H36 示例',task).click();
    assert.equal(await page.locator('#context-status').innerText(),status);
    await tab('运行与对照').click();
    await button('条件对照',task).click();
    assert.equal(await task.locator('.pair .result').count(),2);
    await button('Head',tools).click();
    assert.equal(await button('条件对照').count(),0);
    await page.getByRole('tab',{name:'语言图谱',exact:true}).click();
    await page.getByRole('heading',{name:'查询实体指称'}).waitFor();
    await page.getByRole('tab',{name:'模型探索',exact:true}).click();
    assert.equal(await page.locator('#context-status').innerText(),status);
    for (const width of [390,320]) {
      await page.setViewportSize({width,height:844});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      assert.ok(await task.getByRole('link',{name:'打开历史测试 →'}).isVisible());
    }

    await enter.click();
    await exit.waitFor();
    assert.ok(await workspace.evaluate(element=>
      element.classList.contains('max') && element.scrollWidth<=element.clientWidth));
    await button('精确定位').click();
    await page.keyboard.press('Escape');
    await exit.waitFor();
    assert.equal(await button('精确定位').getAttribute('aria-expanded'),'false');
    await button('管理插件',tools).click();
    await manager.waitFor();
    await page.keyboard.press('Escape');
    await manager.waitFor({state:'hidden'});
    await exit.waitFor();
    await page.keyboard.press('Escape');
    await enter.waitFor();
    assert.equal(await page.locator('.rail').evaluate(element=>element.inert),false);
    assert.equal(await page.locator('#context-status').innerText(),status);
    await assertCase('confirm-000-02');

    assert.deepEqual(errors,[]);
    assert.deepEqual(network,[]);
    fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify({
      passed:true,browser:browser.version(),location:{snapshot:evidence.snapshot,layer:27,block:26,head:36},
      archive:{phase:2983,run:evidence.id,caseId:'confirm-000-02'},fiveHistoricValuesVerified:true,backAndReloadRestored:true,
      scene:{panning:true,rotation:true,wheelZoom:true,presetView:true,nativeGridPicking:true},
      fullscreen:{type:'client-window',floatingConsoleDrag:true,layerSlider:true,location:true,headSamples:true,manager:true,
        stationaryScene:true,archiveNavigation:true,exitRetainedContext:true,narrowEscape:true},
      noOtherSnapshotEvidence:true,desktop:[1440,1000],mobileWidths:[390,320],network,errors,
    },null,2)+'\n');
    console.log('Graph exploration flow passed.');
  } finally {
    await browser.close();
  }
}
run().catch(error=>{console.error(error);process.exitCode=1;});
