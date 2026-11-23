'use strict';
const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const path=require('node:path');
const ffn=require('../../frontend/src/components/v8/data/graphObservationExample.json');
const qwen4=require('../../frontend/src/components/v8/data/nativeObservationQwen4.json');
const b23=require('../../frontend/src/components/v8/data/nativeObservationB23.json');
const {ASSOC_VIEWS,HEAD_CHANNEL_PLANE,assocProject}=require('../../frontend/src/components/v8/assocLayerGeometry.js');
require('../../frontend/src/components/v8/data/nativeHeadChannels.js');
const heads=globalThis.AI2050HeadArchive;
const digest=values=>createHash('sha256').update(Buffer.from(Float32Array.from(values).buffer)).digest('hex');
function bf16(source,offset=0,width=source.length/2){
  const row=new Float32Array(width), bits=new Uint32Array(row.buffer);
  for(let i=0;i<width;i++)bits[i]=source.readUInt16LE((offset+i)*2)<<16;
  return row;
}
const bfloatSource=Buffer.from(ffn.mlp,'base64');
const ffnRow=site=>bf16(bfloatSource,site*ffn.mlpShape[1],ffn.mlpShape[1]);

function createChecks({head,structure,checkLive=false}){
  const id='native-field-readonly-fixture';
  const requests=[],posts=[];
  let liveConnected=true;
  let liveModelStatus=null;
  async function handle(req,url,json,array){
    if(checkLive&&!liveConnected&&url.pathname==='/api/live/status'){
      await json({detail:'QA: capture service offline'},503);return true;
    }
    if(checkLive&&liveModelStatus&&url.pathname==='/api/live/status'){
      await json({schema:'live_status.v1',available:true,status:liveModelStatus,model_name:'Qwen3-4B',meta:structure});return true;
    }
    if(req.method()==='POST'){
      posts.push({path:url.pathname,body:req.postDataJSON()});
      if(checkLive&&url.pathname==='/api/live/load'){
        liveModelStatus='ready';await json({ok:true,status:'ready'});return true;
      }
      if(checkLive&&url.pathname==='/api/live/run'){
        if(posts.filter(post=>post.path==='/api/live/run').length===2){await json({detail:'QA: capture rejected'},422);return true;}
        await json({trace_id:id,model_snapshot:structure.model_snapshot,n_layers:36,d_model:2560,tokens:[' A',' B',' C'],
          readout_positions:[0,2],output_top:[],elapsed_ms:4});return true;
      }
    }
    const prefix='/api/traces/'+id+'/';
    if(!url.pathname.startsWith(prefix))return false;
    const position=Number(url.searchParams.get('position')),block=Number(url.searchParams.get('block'));
    const head=Number(url.searchParams.get('head')),boundary=url.searchParams.get('boundary'),snap=structure.model_snapshot;
    if(url.pathname.endsWith('/manifest')){
      await json({trace_id:id,model_snapshot:snap,n_layers:36,n_heads:32,n_kv_heads:8,head_dim:128,
        d_model:2560,d_ff:9728,n_tokens:3,tokens:[' A',' B',' C'],readout_positions:[0,2]});return true;
    }
    if(url.pathname.endsWith('/bundle')){
      await json({trace_id:id,model_snapshot:snap,position,readout_positions:[0,2],hs:array([37,2560]),attn:array([36,32,3])});return true;
    }
    if(url.pathname.endsWith('/field')){
      requests.push({block,head,boundary,position});
      const whole=boundary==='head_z'&&!url.searchParams.has('head');
      if(position===1&&(boundary.startsWith('mlp_')||boundary==='head_z')){
        await json({detail:'this position was not captured at module boundaries'},404);return true;
      }
      const size=whole?4096:boundary.startsWith('head_')?128:9728;
      const values=Array(size).fill(0);values[size-1]=block+position+.5;
      await json({schema:boundary.startsWith('head_')&&!whole?'mechanism_head_vector.v1':'mechanism_field.v1',
        kind:'real',source:'native_BF16_bits',trace_id:head===3?'wrong-trace':id,model_snapshot:snap,
        position,block,head,kv_group:Math.floor(head/4),boundary,shape:whole?[32,128]:[size],values});return true;
    }
    return false;
  }
  async function verify(page,base,out){
    // Observe the exact vectors passed into the same native canvas as the
    // user interface. Never replace data or alter selections in this hook.
    await page.addInitScript(()=>{
      let actual;
      Object.defineProperty(window,'AI2050NativeField',{configurable:true,get:()=>actual,set:api=>{
        actual={create:(host,opts)=>{
          const view=api.create(host,opts);
          return {update(data){window.__nativeQAVector=data;view.update(data);},destroy(){view.destroy();}};
        }};
      }});
    });
    const checks=[],hashes={head:{},ffn:[]};
    const work=page.locator('.ow-graph'), scene=work.locator('.aw-space'), three=scene.locator('.aw-canvas canvas');
    const field=scene.getByRole('region',{name:'全屏原生单元',exact:true});
    const canvas=field.locator('[data-field-canvas]'), index=field.locator('[data-field-index]');
    const compareView=field.locator('[data-field-view=positions]'),compareValues=field.locator('[data-field-compare-value]');
    const value=field.locator('[data-field-value]'),measure=field.getByLabel('全场测量量');
    const button=(scope,name)=>scope.getByRole('button',{name,exact:true});
    const exact=async i=>{await index.fill(String(i));await index.press('Enter');};
    const hasValue=async n=>{await value.filter({hasText:new RegExp('^'+String(n).replaceAll('.','\\.')+'$')}).waitFor();};
    const missing=async()=>{await hasValue('未采集');assert.ok(await field.locator('[data-field-missing]').isVisible());};
    async function checkFieldHash(size,hash){
      await page.waitForFunction(n=>window.__nativeQAVector?.values?.length===n,size);
      const actual=await page.evaluate(async()=>{
        const bytes=Float32Array.from(window.__nativeQAVector.values);
        const sha=await crypto.subtle.digest('SHA-256',bytes);
        return Array.from(new Uint8Array(sha),b=>b.toString(16).padStart(2,'0')).join('');
      });
      assert.equal(actual,hash);
      assert.equal(await field.locator('[data-native-inspector]').getAttribute('data-size'),String(size));
      return actual;
    }
    const sameScene=async()=>assert.ok(await three.evaluate(c=>c===window.__nativeQAThree));
    const mapPixels=()=>canvas.evaluate(async c=>{
      await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
      return c.toDataURL();
    });
    const verifyPositions=async unit=>{
      const expected=[0,1,2,3].map(i=>ffnRow(i)[unit]);
      assert.equal(await compareValues.count(),expected.length);
      assert.deepEqual((await compareValues.allTextContents()).map(Number),expected);
    };
    const checkWidth=async()=>assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    const atRecord=async()=>{
      await page.goto(base+'?wb=1&lens=route');
      await button(page.getByRole('navigation',{name:'工作台界面'}),'分析研究').click();
      await button(page.getByRole('navigation',{name:'分析研究导航'}),'模型分析').waitFor();
      await work.getByRole('heading',{name:'模型分析',exact:true}).waitFor();
      await work.getByRole('region',{name:'B26/H36 归档注意力'}).waitFor();
    };
    await atRecord();
    await three.evaluate(c=>{window.__nativeQAThree=c});
    const sceneReadout=scene.getByRole('region',{name:'3D 原生读数'});
    const measure3d=scene.getByLabel('Head 着色量');
    const rawZ=sceneReadout.getByLabel('Head 通道原值');
    const rawHead=sceneReadout.getByLabel('Head 总览原值');
    const channelScale=scene.locator('[data-scene-scale=channels]');
    const lock=button(scene,'锁定色标');
    const headSource=work.getByRole('region',{name:'B26/H36 归档注意力'});
    const realCase=heads.cases['confirm-000-02'];
    await rawZ.filter({hasText:'-1.8671875'}).waitFor();
    assert.equal(await three.getAttribute('data-head-channels'),'128');
    assert.match(await sceneReadout.innerText(),/confirm-000-02 · token 190（ID 510）/);
    const archivedCase=head.families[0].cases.find(row=>row.id==='confirm-000-02');
    assert.equal(Number(await rawHead.innerText()),archivedCase.mass[archivedCase.gold]);
    await measure3d.selectOption('strength');
    const rms=Math.sqrt(realCase.values.reduce((sum,z)=>sum+z*z,0)/128);
    assert.equal(Number(await rawHead.innerText()),rms);
    const channelPicker=sceneReadout.getByLabel('3D Head 通道');
    const expectedChannel=15;
    const channelPosition=async index=>{
      const [W,H]=await three.evaluate(c=>[c.getBoundingClientRect().width,c.getBoundingClientRect().height]);
      const [yaw,pitch]=ASSOC_VIEWS.iso;
      const {x0,x1,z0,z1,y}=HEAD_CHANNEL_PLANE;
      const [x,screenY]=assocProject({W,H,yaw,pitch,zoom:1,expandedHead:true},
        x0+(x1-x0)*(index%16+.5)/16,y,z0+(z1-z0)*(Math.floor(index/16)+.5)/8);
      return {x,y:screenY};
    };
    const pixelAt=position=>three.evaluate(async (c,p)=>{
      await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
      const r=c.getBoundingClientRect();
      return [...c.getContext('2d').getImageData(Math.floor(p.x*c.width/r.width),Math.floor(p.y*c.height/r.height),1,1).data];
    },position);
    const negativePixel=await pixelAt(await channelPosition(0));
    const positivePixel=await pixelAt(await channelPosition(expectedChannel));
    assert.ok(negativePixel[0]>negativePixel[2] && positivePixel[0]<positivePixel[2]);
    await three.click({position:await channelPosition(expectedChannel)});
    assert.equal(await channelPicker.inputValue(),String(expectedChannel));
    assert.equal(Number(await rawZ.innerText()),realCase.values[expectedChannel]);
    for (const [i,value] of realCase.values.entries()){
      await channelPicker.selectOption(String(i));
      assert.equal(Number(await rawZ.innerText()),value,`3D channel D${i}`);
    }
    await channelPicker.selectOption('0');
    await lock.click();
    const scale=await channelScale.innerText();
    const overviewScale=await scene.locator('[data-scene-scale=heads]').innerText();
    await headSource.getByLabel('归档样本').selectOption('confirm-000-00');
    assert.equal(await channelScale.innerText(),scale);
    assert.equal(await scene.locator('[data-scene-scale=heads]').innerText(),overviewScale);
    assert.equal(Number(await rawZ.innerText()),heads.cases['confirm-000-00'].values[0]);
    await headSource.getByLabel('归档样本').selectOption('confirm-000-02');
    await page.screenshot({path:path.join(out,'3d-head-channels.png')});
    const jump3d=async address=>{
      await scene.getByLabel('输入原生编号').fill(address);
      await button(scene,'定位对象').click();
    };
    await jump3d('B26/H35');
    assert.equal(await rawZ.innerText(),'未采集');
    await sceneReadout.getByRole('status').filter({hasText:'只保存 H36'}).waitFor();
    assert.deepEqual((await pixelAt(await channelPosition(0))).slice(0,3),[211,220,229]);
    await scene.getByRole('group',{name:'观察模式'}).getByRole('button',{name:'实时模式',exact:true}).click();
    assert.equal(await rawZ.innerText(),'未采集');
    assert.equal(await lock.getAttribute('aria-pressed'),'true');
    await scene.getByRole('group',{name:'观察模式'}).getByRole('button',{name:'测试记录',exact:true}).click();
    await jump3d('B36/U2429');
    const mlpSite=scene.getByLabel('3D 归档读出位置');
    await mlpSite.selectOption('0');
    const rawA=sceneReadout.getByLabel('MLP 单元原值');
    await rawA.filter({hasText:'2.109375'}).waitFor();
    const ffnScale=await scene.locator('[data-scene-scale=mlp]').innerText();
    await mlpSite.selectOption('1');
    assert.equal(Number(await rawA.innerText()),-.4765625);
    assert.equal(await scene.locator('[data-scene-scale=mlp]').innerText(),ffnScale);
    await jump3d('B36/U17407');
    assert.equal(Number(await rawA.innerText()),ffnRow(1).at(-1));
    await page.screenshot({path:path.join(out,'3d-mlp-layer.png')});
    await jump3d('B26/H36/D0');
    await measure3d.selectOption('probability');
    await lock.click();
    checks.push('真实 3D 归档：B26/H36 的 128 个 Z 原值、实际画布点选及蓝橙色、按目标记录概率与 RMS 切换、锁定后换条件；H35 不补值；B36 整层任意编号及同题 token 换位保留色标');
    await button(scene,'原生单元').click();await field.waitFor();
    assert.ok(await scene.evaluate(el=>el.classList.contains('max')));
    await canvas.focus();await canvas.press('End');
    for(const [caseId,sample] of Object.entries(heads.cases)){
      await field.getByLabel('归档样本').selectOption(caseId);
      await hasValue(sample.values.at(-1));
      hashes.head[caseId]=await checkFieldHash(128,digest(sample.values));
    }
    assert.ok(!(await compareView.isVisible()));
    await page.screenshot({path:path.join(out,'head-channels.png')});
    await measure.selectOption('head_q');await missing();
    await measure.selectOption('head_z');await hasValue(heads.cases['confirm-000-02'].values.at(-1));
    await field.getByLabel('头编号').selectOption('35');await missing();
    await field.getByLabel('头编号').selectOption('36');await hasValue(heads.cases['confirm-000-02'].values.at(-1));
    await button(scene,'3D 与控制台').click();await three.waitFor();await sameScene();
    checks.push('全屏与原 3D 画布往返，三道历史题同一 B26/H36 的全部 128 个 Z 通道按单题逐字节核对，Q 和邻头测量缺失');
    await button(scene.locator('.ow-object-locator'),'B36/U2429').click();
    await button(scene,'原生单元').click();await field.waitFor();
    for(let i=0;i<4;i++){
      const row=ffnRow(i);
      await field.getByLabel('全屏归档读出位置').selectOption(String(i));
      await hasValue(row[2429]);
      hashes.ffn.push(await checkFieldHash(row.length,digest(row)));
    }
    await field.getByLabel('全屏归档读出位置').selectOption('0');await hasValue(2.109375);
    await canvas.focus();await canvas.press('End');await hasValue(ffnRow(0).at(-1));
    assert.equal(await index.inputValue(),'17407');
    await exact(2429);await hasValue(2.109375);
    await field.locator('[data-field-scale]').selectOption('absolute');
    const initialPixels=await mapPixels();
    const contrast=field.locator('[data-field-contrast]');
    await contrast.check();await hasValue(2.109375);
    assert.notEqual(await mapPixels(),initialPixels,'增强显示应改变热图着色');
    assert.equal(await checkFieldHash(17408,digest(ffnRow(0))),hashes.ffn[0]);
    await field.locator('[data-field-scale]').selectOption('signed');
    await mapPixels();
    await field.evaluate(el=>{el.scrollTop=0});
    await page.screenshot({path:path.join(out,'ffn-heatmap.png')});
    await compareView.click();
    await verifyPositions(2429);
    await page.screenshot({path:path.join(out,'ffn-position-comparison.png')});
    await exact(17407);await verifyPositions(17407);
    await exact(2429);
    await field.locator('[data-field-view=distribution]').click();
    assert.equal(await field.locator('[data-field-bars] button').count(),20);
    await field.locator('[data-field-bars] button').nth(5).click();
    assert.match(await field.locator('[data-field-bin]').innerText(),/项/);
    await field.locator('[data-field-view=table]').click();
    await field.locator('[data-field-row="2429"]').waitFor();assert.equal(await field.locator('tbody tr').count(),12);
    await field.locator('[data-field-page-next]').click();assert.equal(await field.locator('[data-field-row="2429"]').count(),0);
    await field.locator('[data-field-view=heat]').click();
    const grid=await canvas.evaluate(c=>({columns:+c.dataset.columns,rows:+c.dataset.rows,width:c.clientWidth,height:c.clientHeight}));
    const cell=13;
    await canvas.click({position:{x:(cell%grid.columns+.5)*grid.width/grid.columns,y:(Math.floor(cell/grid.columns)+.5)*grid.height/grid.rows}});
    await hasValue(ffnRow(0)[cell]);assert.equal(await index.inputValue(),String(cell));
    await measure.selectOption('mlp_gate');await missing();
    assert.ok(!(await compareView.isVisible()));
    await measure.selectOption('mlp_activation');await hasValue(ffnRow(0)[cell]);
    checks.push('B36 的四个采集位置各 17,408 个 FFN 原值与原场逐字节相同；验证编号 17407、热图点选、绝对值色标、20 桶直方图与分页');
    checks.push('增强弱响应改变画布像素并保留完整原值；四位置视图逐值核对 U2429 与 U17407 的有符号原始激活，单位置和缺场隐藏比较入口');
    await page.setViewportSize({width:390,height:844});await checkWidth();
    assert.ok(await index.isVisible());
    await exact(17407);await hasValue(ffnRow(0).at(-1));
    await field.evaluate(el=>{el.scrollTop=0});
    await page.screenshot({path:path.join(out,'mobile.png')});
    await work.getByLabel('模型',{exact:true}).selectOption('Qwen3-4B');
    await field.getByLabel('原生对象类型').selectOption('ffn');
    await missing();
    assert.equal(await field.locator('[data-native-inspector]').getAttribute('data-size'),'9728');
    await checkWidth();await page.setViewportSize({width:1440,height:900});
    await button(scene,'退出全屏（Esc）').click();await sameScene();
    checks.push('390 像素窄屏定位末单元；切换模型只展示 4B 原生维数，未沿用前一个模型的 FFN 值');
    // Replay is explicit. A new draft must clear both replay evidence and its
    // position comparison, even before the user starts another forward.
    await button(scene.getByRole('group',{name:'观察模式'}),'实时模式').click();
    const live=work.getByRole('region',{name:'实时模式工作区'});
    await button(live,'回放 B36 实测样例').click();
    await live.getByRole('group',{name:'采集位置'}).waitFor();
    await button(scene,'原生单元').click();await field.waitFor();
    await field.getByLabel('全屏归档读出位置').selectOption('0');
    await exact(2429);await hasValue(2.109375);
    await button(scene,'退出全屏（Esc）').click();await sameScene();
    await live.getByLabel('运行文本').fill('一条尚未运行的新输入');
    await button(scene,'原生单元').click();await field.waitFor();await missing();
    assert.ok(!(await compareView.isVisible()));
    assert.equal(await page.evaluate(()=>window.__nativeQAVector.values),null);
    checks.push('实时模式回放样例后修改输入，尚未运行的新题清空归档读数与四位置比较');
    const traceUrl=base+'?wb=1&lens=spatial&view=graph&gview=structure&mdl=Qwen3-4B&observeMode=archive&observeFullscreen=1&observeField=1&snapshot='+structure.model_snapshot+'&trace='+id+'&tpos=2';
    await page.goto(traceUrl);
    await field.getByLabel('原生对象类型').selectOption('head');
    await field.getByLabel('原生单元模型块').selectOption('4');
    await field.getByLabel('头编号').selectOption('2');
    await measure.selectOption('head_k');await exact(127);await hasValue(6.5);
    await field.getByLabel('头编号').selectOption('3');
    await field.locator('[data-field-missing]').filter({hasText:'不一致'}).waitFor();
    await missing();await field.getByLabel('头编号').selectOption('2');await hasValue(6.5);
    assert.ok(requests.some(r=>r.boundary==='head_k'&&r.head===2&&r.block===4&&r.position===2));
    await page.goto(base+'?wb=1&lens=spatial&view=graph&gview=structure&observeMode=archive&observeTab=research');
    await work.getByRole('tab',{name:'测试记录',exact:true}).click();
    const records=work.getByRole('region',{name:'测试记录',exact:true});
    await records.getByText('回放已归档的原生读数',{exact:true}).click();
    await button(records,'Qwen3-4B · 全层实测轨迹').click();
    const jump=async address=>{const input=scene.getByLabel('输入原生编号');await input.fill(address);await input.press('Enter');};
    const scope=field.getByLabel('Head 范围');
    const sourcePicker=field.getByLabel('原生读数来源');
    await scope.selectOption('layer');
    const z0=bf16(Buffer.from(qwen4.blocks[0].fields.head_z,'base64'));
    await hasValue(z0[0]);
    hashes.qwen4_b0z=await checkFieldHash(4096,digest(z0));
    await jump('B35/H31/D127');
    const z35=bf16(Buffer.from(qwen4.blocks[35].fields.head_z,'base64'));
    await hasValue(z35.at(-1));
    hashes.qwen4_b35z=await checkFieldHash(4096,digest(z35));
    assert.equal(await field.locator('[data-field-selection]').innerText(),'H31/D127');
    await canvas.evaluate(()=>new Promise(requestAnimationFrame));
    assert.equal(await canvas.getAttribute('data-columns'),'128');
    assert.equal(await canvas.getAttribute('data-rows'),'32');
    await jump('B35/H32/D127');await scene.locator('#ow-index-error').waitFor();
    await hasValue(z35.at(-1));
    await scope.selectOption('head');await measure.selectOption('head_k');
    const k35=bf16(Buffer.from(qwen4.blocks[35].fields.head_k,'base64'),7*128,128);
    await hasValue(k35.at(-1));
    await field.getByLabel('头编号').selectOption('30');await hasValue(k35.at(-1));
    hashes.qwen4_b35k=await checkFieldHash(128,digest(k35));
    await jump('B35/U9727');
    for(const boundary of ['mlp_activation','mlp_gate','mlp_value']){
      await measure.selectOption(boundary);
      const row=bf16(Buffer.from(qwen4.blocks[35].fields[boundary],'base64'));
      await hasValue(row.at(-1));hashes[boundary]=await checkFieldHash(9728,digest(row));
    }
    checks.push('从测试记录进入全层 Qwen3-4B 实测轨迹，手动定位 B0 和 B35，整层 32×128 Z、GQA 共用 K 以及整层 G/U/A 逐位核对');
    await sourcePicker.selectOption('b23');
    await measure.selectOption('mlp_activation');
    for(const record of b23.rows){
      await field.getByLabel('全屏归档读出位置').selectOption(record.key);
      const row=bf16(Buffer.from(record.bits,'base64'));
      await hasValue(row[7708]);hashes[record.key]=await checkFieldHash(17408,digest(row));
    }
    await jump('B22/U7708');await missing();
    assert.match(await field.locator('[data-field-missing]').innerText(),/仅包含 B23/);
    await jump('B23/U17407');
    await hasValue(bf16(Buffer.from(b23.rows[1].bits,'base64')).at(-1));
    await sourcePicker.selectOption('head');await measure.selectOption('head_z');await scope.selectOption('layer');
    await field.locator('[data-field-stats]').filter({hasText:/已采集 128 \/ 5,120/}).waitFor();
    await jump('B26/H35/D127');await missing();
    await jump('B26/H36/D127');await hasValue(heads.cases['confirm-000-02'].values.at(-1));
    await work.getByLabel('模型',{exact:true}).selectOption('GLM-4-9B');
    await jump('B39/U13695');await missing();
    assert.equal(await field.locator('[data-native-inspector]').getAttribute('data-size'),'13696');
    checks.push('切到 B23 的两个独立文档，核对全向量；邻层不串值。B26 只给已归档 H36 着色；GLM 末层与末单元可定位并明确缺场');
    assert.deepEqual(posts,[]);
    checks.push('只读轨迹接口夹具检验 Head K、GQA 组编号、错误轨迹拒绝和恢复；全流程零 POST');
    if(checkLive){
      liveConnected=false;
      liveModelStatus='idle';
      await page.goto(base+'?wb=1&lens=spatial&view=graph&gview=structure&mdl=Qwen3-14B&observeMode=live');
      await live.getByLabel('运行文本').fill('本机采集工作区工程验收');
      const reconnect=button(live,'重新查找本机服务');
      await reconnect.waitFor();assert.equal(await button(live,'开始运行').isEnabled(),false);
      liveConnected=true;
      await reconnect.click();
      const modelStatus=live.getByRole('group',{name:'本机模型与运行状态'});
      await live.getByText('工作台当前选中 Qwen3-14B，本机采集服务使用 Qwen3-4B。',{exact:true}).waitFor();
      assert.deepEqual(posts,[]);
      await button(modelStatus,'切换到 Qwen3-4B 并加载').click();
      await modelStatus.getByText('Qwen3-4B · 已加载',{exact:true}).waitFor();
      assert.equal(await work.getByLabel('模型',{exact:true}).inputValue(),'Qwen3-4B');
      assert.deepEqual(posts.map(post=>post.path),['/api/live/load']);
      const consoleTabs=work.getByRole('tablist',{name:'控制台功能'});
      const featureTab=consoleTabs.getByRole('tab',{name:'特征分析',exact:true});
      const runTab=consoleTabs.getByRole('tab',{name:'运行与对照',exact:true});
      await button(live,'任务模式').click();
      await live.getByLabel('观察任务类型').selectOption('negation');
      const raw=live.getByRole('region',{name:'输入、轨迹与原始分析记录'});
      await raw.waitFor();
      const original=await raw.locator('pre').innerText();
      assert.equal(JSON.parse(original).kind,'negation');
      await featureTab.click();
      await work.getByRole('region',{name:'特征分析工作区'}).waitFor();
      assert.equal(await scene.getByRole('button',{name:'实时模式',exact:true}).getAttribute('aria-pressed'),'true');
      await runTab.click();
      assert.equal(await raw.locator('pre').innerText(),original);
      assert.deepEqual(posts.map(post=>post.path),['/api/live/load']);
      checks.push('实时模式可切换并加载服务配置的 Qwen3-4B；进入特征分析不会离开实时模式、自动运行模型或丢失任务；原始记录持续可见');
      await button(live,'自由输入').click();
      await live.getByLabel('运行文本').fill('本机采集工作区工程验收');
      await button(live,'开始运行').click();
      const openLive=button(live,'查看本次原生读数');
      await openLive.waitFor();
      const runtime=scene.getByRole('region',{name:'3D 运行状态'});
      await runtime.waitFor();
      const sceneRuntimeText=await runtime.innerText();
      await jump3d('B35/H31/D127');
      await measure3d.selectOption('strength');
      await rawZ.filter({hasText:'37.5'}).waitFor();
      await lock.click();
      const liveScale=await channelScale.innerText();
      await live.getByLabel('当前Token').selectOption('1');
      await sceneReadout.getByRole('status').filter({hasText:'未采集 Head 输出'}).waitFor();
      assert.equal(await rawZ.innerText(),'未采集');
      assert.equal(await channelScale.innerText(),liveScale);
      await live.getByLabel('当前Token').selectOption('0');
      await rawZ.filter({hasText:'35.5'}).waitFor();
      assert.equal(await channelScale.innerText(),liveScale);
      await live.getByLabel('当前Token').selectOption('2');
      checks.push('实时协议夹具：同一 3D Head 在捕获与未捕获 token 间切换，原值和缺测原因同步，通道色标固定');
      await button(scene,'查相关测试').click();
      await work.getByRole('region',{name:'测试记录',exact:true}).waitFor();
      assert.equal(await runtime.innerText(),sceneRuntimeText);
      await featureTab.click();
      await button(work.getByRole('tabpanel',{name:'特征分析'}),'查看本次原生读数').click();
      await jump('B35/H31/D127');await scope.selectOption('layer');
      await hasValue(37.5);
      assert.equal(await field.locator('[data-native-inspector]').getAttribute('data-size'),'4096');
      assert.match(await field.locator('[data-field-source]').innerText(),new RegExp(id));
      await jump('B35/U9727');await hasValue(37.5);
      await field.getByLabel('当前Token').selectOption('1');await missing();
      await field.locator('[data-field-missing]').filter({hasText:'未在此 token 采集模块边界'}).waitFor();
      assert.ok(!requests.some(r=>r.position===1&&r.boundary==='mlp_activation'));
      await field.getByLabel('当前Token').selectOption('0');await hasValue(35.5);
      assert.equal(posts.length,2);
      assert.deepEqual(posts[1],{path:'/api/live/run',body:{text:'本机采集工作区工程验收',max_tokens:1024}});
      await button(scene,'退出全屏（Esc）').click();
      await runTab.click();
      await button(live,'开始运行').click();
      await live.getByRole('alert').filter({hasText:'QA: capture rejected'}).waitFor();
      assert.equal(await openLive.count(),0);
      await button(scene,'原生单元').click();await missing();
      await field.locator('[data-field-missing]').filter({hasText:'QA: capture rejected'}).waitFor();
      assert.equal(posts.length,3);
      checks.push('本机服务协议夹具：离线可重新查找并保留输入，显式采集一次，整层 Head、末 MLP 单元与各 token 共用同一轨迹；缺测有原因；同文本重跑失败后清空旧值');
    }
    return {checks,headShape:[3,128],ffnShape:[4,17408],sourceHashes:hashes,posts};
  }
  return {handle,verify};
}
module.exports={createChecks};
