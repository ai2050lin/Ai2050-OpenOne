/* Exercise the built workbench and the actual archive router on an isolated loopback center. */
const fs = require('node:fs');
const path = require('node:path');
const {spawn,spawnSync} = require('node:child_process');
const {randomUUID,createHash} = require('node:crypto');
const {once} = require('node:events');
const net = require('node:net');
const assert = require('node:assert/strict');
const {chromium,_electron} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root=path.resolve(__dirname,'../..');
const overview=process.argv.includes('--overview');
const polish=process.argv.includes('--polish');
const compact=process.argv.includes('--compact');
const auditFolder=compact?'workbench_compact_rail':polish?'workbench_polish_20261120':overview?'client_overview_20261120':process.argv.includes('--directions')?'research_directions_20261120':'test_records_20261119';
const result=path.join(root,'tests/glm5/result',auditFolder);
const temp=path.join(root,'tests/glm5_temp',auditFolder);
const python=path.join(root,'.venv/Scripts/python.exe');
const desktop=process.argv.includes('--desktop');
const stage=desktop?'desktop':overview?'overview':process.argv.includes('--directions')?'directions':process.argv.includes('--history')?'history':'pilot';
const checks=[];
async function run(){
  const gate=spawnSync(python,['-B','-c','from deploy.storage_policy import check_write; import sys; check_write(sys.argv[1],400*2**20);check_write(sys.argv[2],8*2**20)',temp,result],{cwd:root});
  assert.equal(gate.status,0,gate.stderr.toString());
  fs.mkdirSync(result,{recursive:true});fs.mkdirSync(temp,{recursive:true});
  const work=fs.mkdtempSync(path.join(temp,'fixture-'));
  const free=net.createServer().listen(0,'127.0.0.1');await once(free,'listening');
  const port=free.address().port; await new Promise(resolve=>free.close(resolve));const url='http://127.0.0.1:'+port;
  const packageRoot=process.env.TEST_RECORDS_CENTER_PACKAGE || (process.argv.includes('--package')?path.join(temp,'package','ai2050-center-'+fs.readFileSync(path.join(root,'VERSION'),'utf8').trim()):root);
  const env={...process.env,TEMP:work,TMP:work,TMPDIR:work,AI2050_STORAGE_ROOT:path.join(work,'storage'),AI2050_CLIENT_SERVER:url,
    AI2050_DIST_DIR:path.join(work,'dist'),AI2050_RESEARCH_DIR:path.join(work,'research'),
    AI2050_NATIVE_DIR:path.join(work,'native'),AI2050_FRONTEND_DIST:process.env.TEST_RECORDS_WEB_DIST || path.join(packageRoot,'frontend/dist'),AI2050_DIST_ADMIN_TOKEN:randomUUID(),
    AI2050_SNAPSHOT_DISABLE:'1',AI2050_DIST_NEWS_OFFLINE:'1',PYTHONDONTWRITEBYTECODE:'1'};
  const log=fs.openSync(path.join(result,stage+'-center.log'),'w');
  const server=spawn(python,['-B','-X','utf8','-m','deploy.center_app','--host','127.0.0.1','--port',String(port)],{cwd:packageRoot,env,stdio:['ignore',log,log],windowsHide:true});
  const done=once(server,'exit');let browser,page,passed=false;
  const errors=[],gpu=[];
  const digest=file=>createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  try{
    for(let i=0;i<100;i++){
      if(server.exitCode!==null)throw Error('fixture center failed; inspect center log');
      try{if((await fetch(url+'/api/health')).ok)break;}catch{}
      await new Promise(r=>setTimeout(r,350));
    }
    const post=(api,body)=>fetch(url+api,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const created=await post('/api/auth/register',{username:'ledgerfixture',password:randomUUID()+'a2'});
    assert.equal(created.status,200,await created.clone().text());
    const session=await created.json();assert.equal(session.user.role,'admin');
    const fixture=spawnSync(python,['-B','-X','utf8','-c','import json,sys;sys.path.insert(0,"tests/glm5");from test_test_archive import plan;print(json.dumps(plan()))'],{cwd:root,env,encoding:'utf8'});
    assert.equal(fixture.status,0,fixture.stderr);
    const spec=JSON.parse(fixture.stdout);spec.title='档案位置与 1111 的跨句式复测';spec.objective='冻结新句式检验请求后移';
    let ctx;
    if(desktop){
      delete env.ELECTRON_RUN_AS_NODE;delete env.AI2050_CLIENT_DEV_URL;
      browser=await _electron.launch({executablePath:path.join(overview?result:temp,'desktop/win-unpacked/AI2050 语言机制研究平台.exe'),args:['--headless'],env});
      page=await browser.firstWindow();ctx=browser.context();
      await browser.evaluate(({BrowserWindow})=>BrowserWindow.getAllWindows().forEach(w=>w.hide()));
      const identity=await browser.evaluate(({app})=>[app.isPackaged,app.getVersion(),app.getPath('userData')]);
      const proof=JSON.parse(fs.readFileSync(path.join(result,'desktop.json'),'utf8'));
      assert.deepEqual(identity.slice(0,2),[true,proof.version]);assert.ok(identity[2].startsWith(work));
    }else{
      browser=await chromium.launch({executablePath:process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',env,headless:true,args:['--disable-background-networking','--disable-component-update']});
      ctx=await browser.newContext({viewport:{width:1360,height:920}});
    }
    await ctx.addInitScript(({token,user,desktop})=>{if(!desktop)window.__AI2050__={mode:'local'};localStorage.setItem('ai2050-session',token);localStorage.setItem('ai2050-user',JSON.stringify(user));},{...session,desktop});
    await ctx.route(/^https?:/,route=>route.request().url().startsWith(url+'/')?route.continue():route.abort());
    page=page || await ctx.newPage();page.setDefaultTimeout(12000);
    page.on('pageerror',e=>errors.push(e.message));
    page.on('request',r=>{if(r.method()==='POST' && /\/api\/live\/(load|run)/.test(r.url()))gpu.push(r.url());});
    const entry=desktop?page.url().split('?')[0]+'?wb=1&':url+'/workbench?';
    if(overview){
      await page.goto(entry+'lens=route');
      await page.getByRole('region',{name:'测试记录概况'}).waitFor();
      if(compact){
        const navigation=page.getByRole('navigation',{name:'工作台界面'});
        assert.equal((await navigation.boundingBox()).width,68);
        assert.equal((await page.locator('.wb-brand .op-logo').boundingBox()).width,24);
        assert.deepEqual(await navigation.getByRole('button').allTextContents(),['总览','语言与模型','实验管理','系统设置']);
        checks.push('左侧 68 像素导航与 24 像素标志');
      }
      const tabs=page.getByRole('tablist',{name:'总览视图'});
      assert.deepEqual(await tabs.getByRole('tab').allTextContents(),['总览','测试数据库']);
      await page.screenshot({path:path.join(result,'overview.png')});
      const summary=tabs.getByRole('tab',{name:'总览',exact:true});
      const database=tabs.getByRole('tab',{name:'测试数据库'});
      await summary.focus();await page.keyboard.press('ArrowRight');
      const nativeLedger=page.getByRole('region',{name:'测试记录',exact:true});
      const nativeSearch=nativeLedger.getByLabel('搜索测试');
      await nativeSearch.fill('gf-2976');
      await nativeLedger.locator('.tr-list>button').first().click();
      await nativeLedger.getByLabel('加入跨测试比较').check();
      await summary.click();await page.reload();await summary.waitFor();await database.click();
      assert.equal(await nativeSearch.inputValue(),'gf-2976');
      assert.ok(await nativeLedger.getByLabel('加入跨测试比较').isChecked());
      await nativeSearch.fill('gf-2979');await nativeLedger.locator('.tr-list>button').first().click();
      await nativeLedger.getByLabel('加入跨测试比较').check();
      await nativeLedger.getByText('跨测试比较（2 项）').click();
      assert.equal(await nativeLedger.locator('.tr-comparison-grid article').count(),2);
      if(polish)await page.screenshot({path:path.join(result,'database.png')});
      await page.goto(entry+'lens=route&rtab=rules&dbStudy=2983');
      await page.getByRole('region',{name:'三线索选源'}).waitFor();
      assert.equal(await database.getAttribute('aria-selected'),'true');
      assert.equal(await page.getByLabel('证据批次').inputValue(),'clt_b26_three_cues_v3');
      if(!desktop){
        await page.setViewportSize({width:390,height:850});
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
        await page.setViewportSize({width:1360,height:920});
      }
      await page.getByRole('navigation',{name:'工作台界面'}).getByRole('button',{name:'实验管理'}).click();
      await page.getByRole('heading',{name:'参与分布式测试'}).waitFor();
      if(polish)await page.screenshot({path:path.join(result,'experiments.png')});
      await page.getByRole('button',{name:'查看测试数据库'}).click();await nativeLedger.waitFor();
      checks.push('正式工作台双标签、键盘切换、小屏、筛选与比较恢复、旧证据链接、实验管理数据库跳转');
    }
    if(polish){
      await page.getByRole('navigation',{name:'工作台界面'}).getByRole('button',{name:'系统设置'}).click();
      await page.getByRole('heading',{name:'连接与诊断',exact:true}).waitFor();
      await page.locator('summary').getByText('连接详情',{exact:true}).click();
      await page.screenshot({path:path.join(result,'settings.png')});
    }
    await page.goto(entry+'lens=spatial&view=graph&gview=internal');
    if(polish){
      const scene=page.getByRole('region',{name:'模型运行 3D 场景'});
      await scene.waitFor();
      const position=await scene.boundingBox();
      const consolePosition=await page.getByRole('tablist',{name:'控制台功能'}).boundingBox();
      assert.ok(consolePosition.y>position.y+position.height);
      await page.screenshot({path:path.join(result,'atlas.png')});
    }
    await page.getByRole('tab',{name:'成果与方法',exact:true}).click();
    await page.getByRole('tab',{name:'测试记录',exact:true}).click();
    const ledger=page.getByRole('region',{name:'测试记录',exact:true});
    const tree=ledger.getByRole('region',{name:'研究方向与设计题'});
    await tree.getByRole('button',{name:/知识网络/}).click();
    await tree.getByRole('button',{name:/实体与类别/}).click();
    await tree.getByRole('button',{name:'实体身份与别名',exact:true}).click();
    const pair=tree.getByRole('article',{name:'方向与题目'});
    assert.match(await pair.innerText(),/钟岚/);
    if(stage==='directions') await ledger.screenshot({path:path.join(result,'direction-tree.png')});
    await tree.getByLabel('查找方向或题目').fill('LOG.CHAIN');
    await pair.getByText('进一步区分两种解法').click();
    assert.match(await pair.innerText(),/灰→紫/);
    await tree.getByLabel('查找方向或题目').fill('不存在的虚构目录项');
    await tree.getByText(/没有匹配项/).waitFor();
    await tree.getByLabel('查找方向或题目').fill('SEM.ATTRIBUTE');
    const familyQuery=page.waitForResponse(r=>r.url().includes('/api/test-archive/runs?') && new URL(r.url()).searchParams.get('family')==='COLOR-NAMING-COMPARISON');
    await pair.getByRole('button',{name:'查询中心同族运行'}).click();
    assert.equal((await familyQuery).status(),200);
    await ledger.getByRole('button',{name:'清除模式族筛选'}).click();
    await ledger.getByLabel('记录来源').selectOption('directions');
    await tree.getByLabel('查找方向或题目').fill('gf-2971');
    await pair.getByText(/相关历史记录/).click();
    await pair.getByRole('button',{name:'打开 Phase 2971'}).click();
    assert.equal(await ledger.getByLabel('记录来源').inputValue(),'goodfire');
    checks.push('方向树展开、A/B 设计题、多步改边对照、空查询和历史跳转');
    const detail=page.getByRole('article',{name:'测试设计与实际运行'});
    const search=ledger.getByLabel('搜索测试');
    async function select(q){await search.fill(q);await ledger.locator('.tr-list>button').first().click();}
    await select('gf-2976');assert.match(await detail.innerText(),/39.8%/);
    await detail.getByRole('button',{name:'Phase 2979',exact:true}).click();assert.match(await detail.innerText(),/76.27%/);checks.push('局部正例与后续推广失败双向追溯');
    await select('gf-2978-v1');assert.match(await detail.innerText(),/0%/);
    await select('gf-2978-v2');assert.match(await detail.innerText(),/256/);
    await select('gf-2980');assert.match(await detail.innerText(),/已有数据复算/);
    await select('gf-2981');assert.match(await detail.innerText(),/4,096/);assert.match(await detail.innerText(),/均未执行/);
    await detail.getByRole('button',{name:'Phase 2971',exact:true}).click();assert.match(await detail.innerText(),/已执行干预/);checks.push('停门版本、旧数据复算与已执行干预分开显示');
    await select('gf-2961');assert.match(await detail.innerText(),/96\/128/);
    await ledger.screenshot({path:path.join(result,'goodfire-record.png')});
    await detail.getByRole('button',{name:'以此记录新建复测'}).click();
    const draft=page.getByRole('region',{name:'新建测试'});
    await draft.getByLabel('先查相似记录').fill('gf-2979');
    await draft.getByRole('button',{name:/Phase 2979/}).first().click();
    await draft.getByLabel('冻结方案内容').fill(JSON.stringify(spec));
    await draft.getByRole('button',{name:'查询中心并冻结方案'}).click();
    await draft.getByRole('heading',{name:'中心查重回执'}).waitFor();
    assert.equal(await draft.getByRole('button',{name:'登记测试',exact:true}).isEnabled(),false);
    await draft.getByLabel('这次多验证了什么').fill('新增两套独立句式，沿用两处占位和原行为门。');
    await draft.getByRole('button',{name:'登记测试',exact:true}).click();
    await detail.getByText('本次复测新增验证',{exact:true}).waitFor();
    assert.match(await detail.innerText(),/新增两套独立句式/);
    assert.match(await detail.innerText(),/Phase 2979/);
    const active=await (await fetch(url+'/api/test-archive/runs',{headers:{Authorization:'Bearer '+session.token}})).json();
    assert.equal(active.total,1);assert.deepEqual(active.items[0].plan.history_refs,['gf-2979']);
    await detail.getByText('原始查重回执',{exact:true}).click();
    await detail.getByRole('button',{name:'读回当时的候选与匹配理由'}).click();
    await detail.locator('pre').filter({hasText:'historical_matches'}).waitFor();
    assert.equal((await fetch(url+'/api/test-archive/preflights/'+active.items[0].preflight_id)).status,401);
    checks.push('真实中心查重、保留历史引用及新增价值、运行登记读回');
    await detail.getByRole('button',{name:'以此记录新建复测'}).click();
    await draft.getByLabel('冻结方案内容').fill(JSON.stringify(spec));await draft.getByRole('button',{name:'查询中心并冻结方案'}).click();
    await draft.getByRole('button',{name:/分析键相同/}).click();await draft.getByText('判定标准',{exact:true}).waitFor();
    await draft.getByLabel('这次多验证了什么').fill('工程验收同采集冲突');
    await draft.getByRole('button',{name:'登记测试',exact:true}).click();await draft.getByRole('alert').waitFor();assert.match(await draft.getByRole('alert').innerText(),/existing acquisition/);
    await draft.getByLabel('冻结方案内容').fill('{}');assert.equal(await draft.getByRole('button',{name:'登记测试',exact:true}).isEnabled(),false);
    checks.push('精确同采集拦截、并排方案比较及修改后旧回执失效');
    await draft.getByRole('button',{name:'返回记录'}).click();
    if(stage!=='pilot'){
      await ledger.getByLabel('记录来源').selectOption('goodfire');await select('gf-lineage-1');
      await detail.locator('summary').filter({hasText:'逐轮历史记录'}).click();
      await detail.getByRole('button',{name:/Claude 261 ·/}).click();
      await detail.getByRole('heading',{name:/第 30 层两个单元的首次运行/}).waitFor();
      await ledger.getByLabel('记录来源').selectOption('history');
      await search.fill('EXP-C001-WP01-001');await ledger.locator('.tr-list>button').first().click();
      assert.match(await detail.innerText(),/RUN-EXP-C001-WP01-001-001/);
      await search.fill('来源绑定');await ledger.locator('.tr-list>button').first().click();await detail.getByText('复核出处与版本').click();
      assert.match(await detail.innerText(),/AGI_.*MEMO|results\//);
      await ledger.screenshot({path:path.join(result,'history-record.png')}); checks.push('逐轮回查早期神经元记录，批量合同关联实跑及来源');
    }
    if(overview){
      const revisit=desktop?page:await ctx.newPage();
      try {
        const auth={Authorization:'Bearer '+session.token,'Content-Type':'application/json'};
        const bytes=Buffer.from('{"case_id":"fixture-case-1","arm":"base","correct":true}\n');
        const sha256=createHash('sha256').update(bytes).digest('hex');
        async function send(route,body,method='POST'){
          const response=await fetch(url+'/api/test-archive'+route,{method,headers:auth,body});
          assert.equal(response.status,200,await response.clone().text());
          return response.json();
        }
        const upload=await send('/uploads',JSON.stringify({sha256,size:bytes.length}));
        await send('/uploads/'+upload.upload_id+'?offset=0',bytes,'PUT');
        await send('/uploads/'+upload.upload_id+'/finish','{}');
        await send('/runs/'+active.items[0].id+'/finalize',JSON.stringify({status:'interrupted',summary:'engineering fixture',
          artifacts:[{name:'case_results.jsonl',role:'case_results',retention:'core',availability:'uploaded',size:bytes.length,sha256}]}));
        await revisit.goto(entry+'lens=route&rtab=database&record='+encodeURIComponent(active.items[0].id));
        const run=revisit.getByRole('article',{name:'测试设计与实际运行'});
        await run.getByRole('heading',{name:spec.title}).waitFor();
        assert.match(await run.innerText(),/新增两套独立句式/);
        await run.getByText('逐题结果与下载',{exact:true}).click();
        const load=run.getByRole('button',{name:'读取逐题结果'});
        await revisit.route('**/objects/'+sha256,r=>r.fulfill({body:bytes.toString().replace('true','null')}),{times:1});
        await load.click();await run.getByText(/长度或 SHA-256 不一致/).waitFor();
        await load.click();await run.getByText('base · 1 / 1 条判对',{exact:true}).waitFor();
        await run.getByText('fixture-case-1 · base',{exact:true}).click();
        assert.equal((await fetch(url+'/api/test-archive/objects/'+sha256)).status,401);
        checks.push('运行链接读回中心实际登记、逐题哈希校验及篡改拒绝、匿名读私有产物被拒');
        if(desktop){
          const target=path.join(work,'verified-case-results.jsonl');
          await browser.evaluate(({dialog},filePath)=>{dialog.showSaveDialog=async()=>({canceled:false,filePath});},target);
          await run.getByRole('button',{name:'选择位置保存逐题文件'}).click();
          await run.getByText('已保存经哈希核验的逐题文件。').waitFor();
          assert.deepEqual(fs.readFileSync(target),bytes);
          checks.push('Electron 保存选择本机 D 盘并逐字节核对');
        }
      } finally {if(!desktop)await revisit.close();}
    }
    assert.equal((await fetch(url+'/api/test-archive/history')).status,401);
    assert.deepEqual(gpu,[]);assert.deepEqual(errors,[]);passed=true;
  }finally{
    if(!passed && page)console.error((await page.locator('body').innerText().catch(()=>'' )).slice(-2100));
    if(browser)await browser.close();if(server.exitCode===null)server.kill();await done;fs.closeSync(log);
    const within=fs.realpathSync(work),parent=fs.realpathSync(temp);assert.equal(path.dirname(within),parent);assert.ok(path.basename(within).startsWith('fixture-'));
    fs.rmSync(within,{recursive:true,force:true});
    const report={passed,stage,checks,pilot_sha256:digest(path.join(root,'shared/dist_protocol/test_records/pilot.json')),models_run:0,
      engineering_fixture_only:true,packaged_center:packageRoot!==root,packaged_desktop:desktop,gpu_requests:gpu.length,page_errors:errors,at:new Date().toISOString()};
    fs.writeFileSync(path.join(result,stage+'-acceptance.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
  }
}
run().catch(e=>{console.error(e);process.exitCode=1;});

