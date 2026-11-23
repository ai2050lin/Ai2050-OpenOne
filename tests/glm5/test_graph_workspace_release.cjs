const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {fileURLToPath}=require('node:url');
const { chromium, _electron } = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const storage = require('../../frontend/electron/storagePolicy.js');
const head = require('../../frontend/src/components/v8/threeCueEvidence.json').runs[0];

const root=path.resolve(__dirname,'../..');
const dist=path.resolve(process.env.AI2050_GRAPH_WORKSPACE_DIST || path.join(root,'frontend/dist'));
const plugins=process.argv.includes('--plugins');
const fullField=process.argv.includes('--native-field');
const archiveObjects=process.argv.includes('--archive-objects');
const out=process.env.AI2050_GRAPH_WORKSPACE_RESULTS || path.join(__dirname,archiveObjects?'result/workspace_archive_objects':
  fullField?'result/workspace_full_native':plugins?'result/workspace_analysis_plugins':'result/graph_workspace_release');
const scratch=path.join(root,'tests/glm5_temp/graph_workspace_release');
const pkg=require('../../frontend/package.json');
const desktop=process.argv.includes('--desktop');
const overview=process.argv.includes('--overview');
const snapshot='6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad';
const structure={model:'Qwen3-4B',model_snapshot:snapshot,n_layers:36,n_heads:32,d_model:2560,d_ff:9728};
const skeleton={models:[
  {model:'Qwen3-14B',model_snapshot:head.snapshot,n_layers:40,n_heads:40,d_model:5120,d_ff:17408},
  structure,
]};
const tokens=[' Test',' input'];
const traceId='fixture-graph-release';

function array(shape, last=0) {
  const fields=new Float32Array(shape.reduce((product,dimension)=>product*dimension,1));
  fields[fields.length-1]=last;
  return {shape,b64:Buffer.from(fields.buffer).toString('base64')};
}

async function verifyOverview(page, base) {
  await page.goto(base+'?wb=1&lens=route');
  const tabs=page.getByRole('tablist',{name:'总览视图'});
  const group=page.getByRole('group',{name:'总览的更多内容'});
  const panel=page.locator('#overview-more-panel');
  const trigger=name=>group.getByRole('button',{name,exact:true});
  const labels=['研究发现','验证进展','研究资料'];
  await group.waitFor();
  assert.deepEqual(await page.locator('.v8-overview-card strong').allTextContents(),['语言图谱','LLM 编码图谱','关联机制']);
  assert.deepEqual(await group.getByRole('button').allTextContents(),labels);
  assert.equal(await panel.isHidden(),true);
  for(const label of labels) assert.equal(await trigger(label).getAttribute('aria-expanded'),'false');
  if(!desktop) await page.screenshot({path:path.join(out,'overview.png')});
  await trigger('研究发现').focus();
  await page.keyboard.press('Enter');
  await panel.getByRole('region',{name:'最近的发现'}).waitFor();
  await panel.getByText('一个决策点案例',{exact:true}).click();
  await panel.getByRole('button',{name:'打开模型分析',exact:true}).waitFor();
  await panel.getByRole('button',{name:'参与研究',exact:true}).waitFor();
  await trigger('验证进展').click();
  await panel.getByRole('region',{name:'关联机制的四组成绩'}).waitFor();
  assert.equal(await panel.getByRole('region',{name:'最近的发现'}).count(),0);
  await panel.getByText('能力矩阵与证据状态',{exact:true}).click();
  await panel.getByRole('table').waitFor();
  await trigger('研究资料').click();
  await panel.getByText('阶段任务与历史分片',{exact:true}).waitFor();
  await panel.getByText('研究归档与原始来源',{exact:true}).waitFor();
  assert.equal(await panel.getByRole('table').count(),0);
  await trigger('研究资料').click();
  await panel.waitFor({state:'hidden'});
  if(!desktop){
    await page.setViewportSize({width:360,height:800});
    for(const label of labels) assert.ok(await trigger(label).isVisible());
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await trigger('研究发现').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(out,'overview-mobile.png'),fullPage:true});
    await trigger('验证进展').click();
    await panel.getByRole('region',{name:'关联机制的四组成绩'}).waitFor();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.setViewportSize({width:1440,height:900});
  }
  await page.getByRole('button',{name:'打开测试数据库',exact:true}).click();
  assert.equal(await tabs.getByRole('tab',{name:'测试数据库',exact:true}).getAttribute('aria-selected'),'true');
  await page.getByRole('region',{name:'测试记录',exact:true}).waitFor();
}

async function check() {
  storage.checkWrite(scratch,512*2**20,'图谱工作区验收');
  storage.checkWrite(out,8*2**20,'图谱工作区验收结果');
  fs.mkdirSync(scratch,{recursive:true});
  fs.mkdirSync(out,{recursive:true});
  const profile=fs.mkdtempSync(path.join(scratch,'test-profile-'));
  const posts=[], unexpected=[];
  const extraChecks=archiveObjects?require('./workspace_archive_object_checks.cjs'):
    fullField?require('./workspace_full_native_checks.cjs').createChecks({head,structure,checkLive:process.argv.includes('--native-live')}):
    plugins?require('./workspace_analysis_plugin_checks.cjs').createChecks({head,structure}):
    process.argv.includes('--record-links')?require('./workspace_record_link_checks.cjs').createChecks({structure,head}):null;
  const fixture={
    '/api/platform/config':{},
    '/api/model/skeleton':skeleton,
    '/api/live/status':{schema:'live_status.v1',available:true,status:'ready',model_name:'Qwen3-4B',meta:structure},
    '/api/traces/catalog':{traces:[]},
    '/api/research-library/workpackages':{workpackages:[]},
    '/api/traces/research-space':{objects:[]},
  };
  const types={'.js':'text/javascript','.css':'text/css','.json':'application/json','.html':'text/html'};
  const host='http://127.0.0.1:53189';
  const harness=async route=>{
    const request=route.request();
    const url=new URL(request.url());
    if(url.origin!==host){await route.abort();return;}
    const headers={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'Content-Type,Authorization',
      'Access-Control-Allow-Methods':'GET,POST,OPTIONS'};
    const json=(value,status=200)=>route.fulfill({status,headers,json:value});
    if(request.method()==='OPTIONS'){await route.fulfill({status:204,headers});return;}
    if(extraChecks && await extraChecks.handle(request,url,json,array,body=>route.fulfill({status:200,headers,body,contentType:'application/x-ndjson'})))return;
    if(url.pathname==='/api/live/run'&&request.method()==='POST'){
      posts.push(request.postDataJSON());
      await json({trace_id:traceId,model_snapshot:snapshot,n_layers:36,d_model:2560,tokens,readout_positions:[0,1],
        elapsed_ms:5,output_top:[{text:' result',logit:12,id:12}]});
      return;
    }
    if(url.pathname==='/api/traces/'+traceId+'/bundle'){
      const position=Number(url.searchParams.get('position'));
      await json({trace_id:traceId,model_snapshot:snapshot,position,readout_positions:[0,1],
        hs:array([37,2560],position+1),attn:array([36,32,tokens.length]),mlp:array([36,9728],position+1)});
      return;
    }
    if(Object.hasOwn(fixture,url.pathname)){await json(fixture[url.pathname]);return;}
    if(url.pathname.startsWith('/api/')){await json({detail:'Not included in the isolated fixture'},404);return;}
    const name=url.pathname==='/'||url.pathname==='/workbench'?'index.html':decodeURIComponent(url.pathname.slice(1));
    const file=path.resolve(dist,name);
    if(!file.startsWith(dist+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){
      await route.fulfill({status:404});return;
    }
    await route.fulfill({path:file,contentType:types[path.extname(file)]||'application/octet-stream'});
  };
  const env={...process.env,TEMP:profile,TMP:profile,TMPDIR:profile,AI2050_STORAGE_ROOT:path.join(profile,'storage'),
    AI2050_CLIENT_SERVER:host,AI2050_SNAPSHOT_DISABLE:'1',AI2050_DIST_NEWS_OFFLINE:'1'};
  const tempVariables=['TEMP','TMP','TMPDIR'].map(key=>[key,process.env[key]]);
  for(const [key] of tempVariables)process.env[key]=profile;
  const release=process.env.AI2050_GRAPH_WORKSPACE_DESKTOP_DIR || path.join(root,'release','desktop-'+pkg.version);
  let app,browser;
  try{
    let page,context,base,packagePage;
    if(desktop){
      delete env.ELECTRON_RUN_AS_NODE;
      delete env.AI2050_CLIENT_DEV_URL;
      app=await _electron.launch({executablePath:path.join(release,'win-unpacked',pkg.productName+'.exe'),args:['--headless'],env,timeout:60000});
      page=await app.firstWindow();context=app.context();
      await app.evaluate(({BrowserWindow})=>BrowserWindow.getAllWindows().forEach(window=>window.hide()));
      assert.deepEqual(await app.evaluate(({app})=>[app.isPackaged,app.getVersion()]),[true,pkg.version]);
      await page.waitForURL(url=>url.protocol==='file:'&&url.pathname.endsWith('/dist/index.html'));
      base=page.url().split('?')[0];
      const resources=path.join(release,'win-unpacked','resources');
      packagePage=path.relative(resources,fileURLToPath(base)).split(path.sep).join('/');
      assert.equal(packagePage,'app.asar/dist/index.html');
    }else{
      const executablePath=process.env.CHROME_PATH || path.join(os.homedir(),
        'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
      browser=await chromium.launch({headless:true,executablePath,env});
      context=await browser.newContext({viewport:{width:1440,height:900}});
      await context.addInitScript(()=>{window.__AI2050__={mode:'local'};});
      page=await context.newPage();base=host+'/workbench';
    }
    await context.route(/^https?:/,harness);
    page.on('pageerror',error=>unexpected.push(error.message));
    page.setDefaultTimeout(12000);
    if(extraChecks){
      const receipt=path.join(out,(desktop?'desktop':'browser')+'-acceptance.json');
      const build={version:pkg.version,desktop,packagePage};
      try {
        const report=await extraChecks.verify(page,base,out);
        assert.deepEqual(unexpected,[]);
        fs.writeFileSync(receipt,JSON.stringify({passed:true,...build,...report,unexpected},null,2)+'\n');
        console.log(JSON.stringify(report));
      } catch(error) {
        fs.writeFileSync(receipt,JSON.stringify({passed:false,...build,message:error.message,unexpected},null,2)+'\n');
        throw error;
      }
      return;
    }
    if(overview) {
      await verifyOverview(page,base);
      await page.getByRole('navigation',{name:'工作台界面'}).getByRole('button',{name:'分析研究',exact:true}).click();
      const sections=page.getByRole('navigation',{name:'分析研究导航'});
      await sections.getByRole('button',{name:'语言图谱',exact:true}).click();
      await page.getByRole('heading',{name:'语言图谱',level:2,exact:true}).waitFor();
      assert.equal(new URL(page.url()).searchParams.get('gview'),'language');
      await sections.getByRole('button',{name:'模型分析',exact:true}).click();
      await page.getByRole('heading',{name:'模型分析',exact:true}).waitFor();
      assert.equal(new URL(page.url()).searchParams.get('gview'),'structure');
    }
    await page.goto(base+'?wb=1&lens=route');
    await page.getByRole('navigation',{name:'工作台界面'}).getByRole('button',{name:'分析研究',exact:true}).click();
    const workbench=page.locator('.ow-graph'), scene=workbench.locator('.aw-space'), canvas=scene.locator('canvas');
    const tools=workbench.getByRole('group',{name:'分析插件'}), modes=scene.getByRole('group',{name:'观察模式'});
    const consoleTabs=workbench.getByRole('tablist',{name:'控制台功能'});
    const button=(scope,name)=>scope.getByRole('button',{name,exact:true});
    const verifyTabs=(...names)=>consoleTabs.getByRole('tab').allTextContents().then(actual=>assert.deepEqual(actual,names));
    const archive=workbench.getByRole('region',{name:'B26/H36 归档注意力'});
    const objectLocator=workbench.locator('.ow-object-locator');
    const plugin=workbench.locator('#console-panel-plugin');
    const archivedCases=[];
    const archiveCaseId='confirm-000-02';
    async function verifyHistoricalCase(id,{browserBack=false}={}) {
      const row=head.families[0].cases.find(item=>item.id===id);
      const wasFullscreen=await scene.evaluate(element=>element.classList.contains('max'));
      await archive.getByLabel('归档样本').selectOption(id);
      assert.deepEqual(await archive.locator('progress').evaluateAll(values=>values.map(value=>value.value)),row.order.map(index=>row.mass[index]));
      const correct=row.order.indexOf(row.gold)+1;
      const peak=row.order.indexOf(row.mass.indexOf(Math.max(...row.mass)))+1;
      if(id===archiveCaseId) assert.deepEqual([correct,peak,row.output],[1,4,'silver']);
      else assert.deepEqual([correct,peak,row.output],[2,2,'wood']);
      await archive.getByRole('link',{name:'打开历史测试'}).click();
      const record=page.getByRole('region',{name:'历史测试原题'});
      await record.waitFor();
      assert.equal(new URL(page.url()).searchParams.get('case'),id);
      assert.ok((await record.innerText()).includes(id));
      assert.equal(await record.locator('h3 code').innerText(),row.output);
      assert.deepEqual(await record.locator('tbody tr').evaluateAll(rows=>rows.map(tr=>tr.cells[2].textContent)),
        row.order.map(index=>head.families[0].names[index]));
      await record.getByText('原始模型输入',{exact:true}).click();
      assert.ok((await record.locator('pre').innerText()).includes(row.question));
      await record.getByText('运行与快照',{exact:true}).click();
      assert.ok((await record.innerText()).includes(head.snapshot));
      if(browserBack) await page.goBack();
      else await record.getByRole('link',{name:'返回分析'}).click();
      await archive.waitFor();
      assert.equal(await workbench.getByLabel('模型',{exact:true}).inputValue(),'Qwen3-14B');
      assert.equal(await button(modes,'测试记录').getAttribute('aria-pressed'),'true');
      assert.equal(await button(tools,'Head').getAttribute('aria-pressed'),'true');
      assert.equal(await archive.getByLabel('归档样本').inputValue(),id);
      assert.equal(await button(objectLocator,'B26/H36').getAttribute('aria-pressed'),'true');
      assert.equal(await scene.evaluate(element=>element.classList.contains('max')),wasFullscreen);
      archivedCases.push(id);
    }
    await canvas.waitFor();
    assert.deepEqual(await modes.getByRole('button').allTextContents(),['实时模式','测试记录','特征分析']);
    assert.equal(await button(modes,'测试记录').getAttribute('aria-pressed'),'true');
    await verifyTabs('运行与对照','插件分析','成果与方法');
    await archive.waitFor();
    assert.equal(await archive.getByLabel('归档样本').inputValue(),archiveCaseId);
    await button(objectLocator,'B26/H36').click();
    await button(tools,'Head').click();
    await verifyHistoricalCase(archiveCaseId);
    await page.screenshot({path:path.join(out,(desktop?'desktop':'browser')+'-explorer.png')});
    if(!desktop) {
      await page.setViewportSize({width:390,height:844});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await button(scene,'⛶ 全屏').click();
      await button(objectLocator,'B26/H36').click();
      await verifyHistoricalCase(archiveCaseId);
      await button(scene,'退出全屏（Esc）').click();
      await page.setViewportSize({width:1440,height:900});
    }
    await workbench.getByLabel('模型',{exact:true}).selectOption('Qwen3-4B');
    const lookup=objectLocator.getByLabel('输入原生编号',{exact:true});
    await lookup.fill('B26/H36');await lookup.press('Enter');
    await objectLocator.getByRole('status').waitFor();
    assert.equal(await archive.count(),0);
    await workbench.getByLabel('模型',{exact:true}).selectOption('Qwen3-14B');
    await archive.waitFor();
    assert.equal(await workbench.locator('#console-panel-object').count(),0);
    assert.equal(await workbench.getByLabel('模型',{exact:true}).count(),1);
    await canvas.evaluate(element=>{window.testWorkspaceCanvas=element;});
    const isSameCanvas=()=>canvas.evaluate(element=>element===window.testWorkspaceCanvas);
    await button(scene,'⛶ 全屏').click();
    const float=workbench.locator('.ow-float');
    await float.waitFor();
    assert.ok(await scene.evaluate(element=>element.getBoundingClientRect().height===window.innerHeight));
    const handle=float.locator('.ow-grip');
    await handle.hover({position:{x:16,y:8}});
    const grip=await handle.boundingBox();
    const beforeDrag=await float.boundingBox();
    const hit=await page.evaluate(([x,y])=>document.elementsFromPoint(x,y).slice(0,6).map(el=>el.className || el.tagName),[grip.x+16,grip.y+8]);
    const moveRight=await page.evaluate(()=>innerWidth)/2>beforeDrag.x;
    await page.mouse.move(grip.x+16,grip.y+8);
    await page.mouse.down();await page.mouse.move(grip.x+16+(moveRight?72:-72),grip.y+8+40,{steps:6});await page.mouse.up();
    const afterDrag=await float.boundingBox();
    assert.ok(Math.hypot(afterDrag.x-beforeDrag.x,afterDrag.y-beforeDrag.y)>4,JSON.stringify({grip,beforeDrag,afterDrag,hit}));
    const locatorBounds=await objectLocator.boundingBox();
    assert.ok(afterDrag.x>=locatorBounds.x+locatorBounds.width||afterDrag.y>=locatorBounds.y+locatorBounds.height);
    for(const label of ['实时模式','测试记录','特征分析']) assert.ok(await button(modes,label).isVisible());
    await button(modes,'实时模式').click();
    assert.ok(await isSameCanvas());
    const live=workbench.getByRole('region',{name:'实时模式工作区'});
    await live.waitFor();
    await verifyTabs('运行与对照','特征分析','插件分析');
    assert.equal(await workbench.locator('#console-panel-research').count(),0);
    await live.getByLabel('运行文本').fill('Fixture observation');
    assert.equal(await button(live,'开始运行').isDisabled(),true);
    await button(live,'回放 B36 实测样例').click();
    const sample=live.getByRole('group',{name:'采集位置'});
    await button(sample,'输入末尾').click();
    assert.equal(await live.locator('output').innerText(),'2.1094');
    await consoleTabs.getByRole('tab',{name:'插件分析',exact:true}).click();
    await button(tools,'MLP').click();
    assert.equal(await plugin.locator('output').innerText(),'2.1094');
    await button(scene,'显示插件').click();
    const display=scene.getByRole('group',{name:'显示插件'});
    await display.getByLabel('数值着色').uncheck();await page.keyboard.press('Escape');
    assert.equal(await plugin.locator('output').innerText(),'2.1094');
    await button(modes,'特征分析').click();
    const features=workbench.getByRole('region',{name:'特征分析工作区'});
    await features.waitFor();
    await verifyTabs('对象详情','插件分析');
    assert.equal(await workbench.locator('#console-panel-run').count(),0);
    await button(features,'原生单元组').click();
    await button(features,'在 3D 中标出整组').click();
    assert.equal(await features.getByText('90 / 92 家族正确',{exact:true}).count(),2);
    assert.ok(await isSameCanvas());
    await button(modes,'测试记录').click();
    await verifyTabs('运行与对照','插件分析','成果与方法');
    await button(objectLocator,'B36/U2429').click();
    const mlp=workbench.getByRole('region',{name:'MLP 归档激活'});
    await mlp.waitFor();
    assert.deepEqual(await mlp.locator('tbody tr').evaluateAll(rows=>rows.map(row=>Number(row.cells[2].textContent))),
      [2.109375,-0.4765625,-0.447265625,0.79296875]);
    await button(objectLocator,'B26/H36').click();
    await consoleTabs.getByRole('tab',{name:'插件分析',exact:true}).click();
    await button(tools,'Head').click();
    await page.screenshot({path:path.join(out,desktop?'desktop-archive-fullscreen.png':'browser-archive-fullscreen.png')});
    await verifyHistoricalCase('confirm-000-00');
    await verifyHistoricalCase('confirm-000-00',{browserBack:true});
    assert.equal(posts.length,0);

    await page.goto(base+'?wb=1&lens=spatial&view=graph&gview=structure&mdl=Qwen3-4B&observeMode=live');
    await live.getByLabel('运行文本').fill('Fixture observation');
    await button(live,'开始运行').click();
    await live.getByRole('button',{name:'预测结果',exact:true}).waitFor();
    assert.equal(posts.length,1);
    assert.equal(posts[0].text,'Fixture observation');
    await live.getByLabel('当前Token').selectOption('0');
    assert.equal(await live.getByLabel('当前Token').inputValue(),'0');
    await button(scene,'精确定位').click();
    const locator=scene.getByRole('form',{name:'原生索引定位'});
    await locator.getByLabel('定位对象类型').selectOption('hs');
    await locator.locator('[name=layer]').fill('36');
    await locator.locator('[name=index]').fill('2559');
    await button(locator,'定位单元').click();
    await consoleTabs.getByRole('tab',{name:'插件分析',exact:true}).click();
    await button(tools,'HiddenState').click();
    await plugin.getByText('当前对象的完整响应与来源',{exact:true}).click();
    const response=JSON.parse(await plugin.locator('pre.v8-json').innerText());
    assert.deepEqual([response.layer,response.position,response.values.at(-1)],[36,0,1]);
    assert.deepEqual(unexpected,[]);
    fs.writeFileSync(path.join(out,(desktop?'desktop':'browser')+'-acceptance.json'),JSON.stringify({
      passed:true,version:pkg.version,desktop,packagePage,sharedScene:true,threeFullscreenModes:true,contextualTabs:true,displayIndependent:true,
      rawResponseInAnalysis:true,
      ...(overview?{compactOverview:true}:{}),
      recordedSample:2933,archivedCases,liveProtocolFixture:true,livePosts:posts.length,unexpected,
    },null,2)+'\n');
    console.log((desktop?'Desktop':'Browser')+' graph workspace acceptance passed.');
  }finally{
    if(app)await app.close();
    if(browser)await browser.close();
    for(const [key,value] of tempVariables){if(value===undefined)delete process.env[key];else process.env[key]=value;}
    const actual=fs.realpathSync(profile);
    assert.equal(path.dirname(actual),fs.realpathSync(scratch));
    assert.ok(path.basename(actual).startsWith('test-profile-'));
    fs.rmSync(actual,{recursive:true,force:true});
  }
}

check().catch(error=>{console.error(error);process.exitCode=1;});
