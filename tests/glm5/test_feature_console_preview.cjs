'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {checkedDir,checkWrite} = require('../../frontend/electron/storagePolicy');
const observations = require('../../frontend/src/components/v8/data/graphObservationExample.json');
require('../../frontend/src/components/v8/data/nativeHeadChannels.js');

const guiHash = '#?wb=1&studyPane=tools&observeMode=features&observeTab=object&observePlugin=object&sblock=37&smodule=mlp&sindex=2429&lens=spatial&view=graph&gview=langmeasure';
const root = path.resolve(__dirname,'../..');
const url = pathToFileURL(path.join(root,'research/glm5/docs/previews/client-distributed-testing.html')).href+guiHash;
const out = path.join(__dirname,'result/feature_console_preview');
const raw = Buffer.from(observations.mlp,'base64');
function mlp(site,index) {
  const high = raw.readUInt16LE(2*(site*observations.mlpShape[1]+index));
  const b = Buffer.alloc(4);
  b.writeUInt16LE(high,2);
  return b.readFloatLE();
}
function rms(v) {return Math.sqrt(v.reduce((sum,x)=>sum+x*x,0)/v.length);}
function signedPixel(value,limit) {
  const endpoint=value>=0?[31,95,191]:[212,98,43],mix=Math.min(1,Math.abs(value)/(limit||1))**.6;
  return Array.from(Uint8ClampedArray.from(endpoint,c=>255+(c-255)*mix));
}
function close(a,b,message){assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=Math.max(1e-7,Math.abs(b)*1e-5),message+' — '+a+' / '+b);}

async function verify() {
  checkWrite(out,280*2**20,'离线控制台浏览器验收');
  checkedDir(out,24*2**20,'控制台验收截图与回执');
  const tmp=checkedDir(path.join(root,'tests/glm5_temp'),256*2**20,'控制台临时浏览器数据');
  const scratch=fs.mkdtempSync(path.join(tmp,'feature-console-'));
  for (const k of ['TMP','TEMP','TMPDIR'])process.env[k]=scratch;
  const report={passed:false,entry:guiHash,checks:[],screenshots:[],values:{},network:[],errors:[]};
  let browser;
  try{
    const {chromium}=require(process.env.PLAYWRIGHT_CORE||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
    const executablePath=process.env.CHROME_PATH||path.join(os.homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
    browser=await chromium.launch({headless:true,executablePath});
    report.browser=browser.version();
    const context=await browser.newContext({viewport:{width:1440,height:1000},locale:'zh-CN'});
    await context.route(/^https?:/,r=>{report.network.push(r.request().url());return r.abort();});
    // Read the actual source textures drawn on the visible 3D canvas. Capturing the current
    // frame distinguishes an old texture left in memory from one still applied to the scene.
    await context.addInitScript(()=>{
      let textures={};
      const proto=CanvasRenderingContext2D.prototype,draw=proto.drawImage,clear=proto.clearRect;
      proto.clearRect=function(...args){if(this.canvas.closest('.aw-space'))textures={};return clear.apply(this,args);};
      proto.drawImage=function(image,...args){
        if(this.canvas.closest('.aw-space') && image.dataset?.featureGrid)textures[image.dataset.featureGrid]=image;
        return draw.call(this,image,...args);
      };
      window.scenePixel=(grid,index)=>{
        const img=textures[grid];if(!img || index<0 || index>=img.width*img.height)return null;
        return Array.from(img.getContext('2d').getImageData(index%img.width,Math.floor(index/img.width),1,1).data).slice(0,3);
      };
    });
    const page=await context.newPage();page.setDefaultTimeout(7500);
    page.on('pageerror',e=>report.errors.push(e.message));
    page.on('requestfailed',r=>{if(r.url().startsWith('file:'))report.errors.push(r.url());});
    const bench=page.locator('.ow-graph'),scene=bench.locator('.aw-space'),consolePanel=bench.getByRole('region',{name:'控制台',exact:true});
    const modes=scene.getByRole('group',{name:'观察模式'});
    const button=(scope,name)=>scope.getByRole('button',{name,exact:true});
    const mode=name=>button(modes,name).click();
    const model=scene.getByLabel('模型',{exact:true});
    const live=bench.getByRole('region',{name:'实时模式工作区'});
    const plugins=bench.getByRole('group',{name:'分析插件'});
    const card=bench.getByTestId('goodfire-profile');
    const legend=scene.locator('[data-feature-colors]');
    const valueFor=kind=>legend.locator('[data-feature-value="'+kind+'"]');
    const scaleFor=kind=>legend.locator('[data-feature-scale="'+kind+'"]');
    const clock=()=>page.evaluate(()=>new Promise(ok=>requestAnimationFrame(()=>requestAnimationFrame(ok))));
    async function shot(name,locator) {
      await clock();
      await (locator||page).screenshot({path:path.join(out,name+'.png'),animations:'disabled'});
      report.screenshots.push(name+'.png');
    }
    async function check(name,run){try{await run();report.checks.push(name);console.log(name);}catch(err){err.message=name+' — '+err.message;throw err;}}
    async function locate(kind,block,index) {
      await button(scene,'精确定位').click();
      const form=scene.locator('form[aria-label="原生索引定位"]');
      await form.getByLabel('定位对象类型').selectOption(kind);
      await form.locator('[name=layer]').fill(String(block+1));await form.locator('[name=index]').fill(String(index));
      await button(form,'定位单元').click();await clock();
    }
    async function above(first,second){
      const a=await first.boundingBox(),b=await second.boundingBox();
      assert.ok(a&&b&&a.y+a.height<=b.y+2,'内容与插件应上下排列，实际 '+JSON.stringify([a,b]));
    }
    async function measured(kind,index){
      const out=valueFor(kind);
      await out.waitFor();assert.equal(await out.getAttribute('data-index'),String(index));
      const value=await out.getAttribute('data-value');assert.ok(value!==null && value!=='','缺少选中读数');return Number(value);
    }
    const pixel=(kind,index)=>page.evaluate(([k,i])=>window.scenePixel(k,i),[kind,index]);
    const colorChecking=(v,rgb)=>{assert.ok(rgb,'画布未使用该坐标的纹理');assert.ok(v>0?rgb[2]>rgb[0]:rgb[0]>rgb[2],'符号与 RGB 不匹配 '+JSON.stringify([v,rgb]));};
    await page.goto(url);await bench.waitFor();await mode('实时模式');
    await check('实时文本与运行操作位于插件上方，展开插件后仍保留输入',async()=>{
      await live.waitFor();await plugins.waitFor();await above(live,plugins);
      const draft='布袋是蓝色，木盒是红色。木盒是什么颜色？';
      const input=live.getByLabel('运行文本');await input.fill(draft);
      await button(plugins,'MLP').click();await live.waitFor();await above(live,plugins);
      assert.equal(await input.inputValue(),draft);
      await bench.locator('#console-panel-plugin').waitFor();
      await shot('live-console',consolePanel);
      await button(plugins,'收起结果').click();await bench.locator('#console-panel-plugin').waitFor({state:'hidden'});
      assert.equal(await input.inputValue(),draft);await above(live,plugins);
      await button(live,'回放 B36 实测样例').click();await above(live,plugins);
      assert.ok((await consolePanel.innerText()).includes('2429'));
    });
    await mode('特征分析');await locate('mlp',36,2429);
    const chooseGoodfire=()=>button(bench.getByRole('group',{name:'特征类型'}),'Goodfire 分析').click();
    await check('同一 MLP 在两个读出位置按实测值换色，负值、正值与缺测可辨',async()=>{
      await legend.waitFor();
      const site=legend.getByLabel('着色读出位置');
      await site.selectOption({index:0});await clock();
      const observed=await measured('mlp',2429);close(observed,mlp(0,2429),'输入末尾 BF16');
      const positive=await pixel('mlp',2429);colorChecking(observed,positive);
      let max=0;for(let s=0;s<observations.mlpShape[0];s++)for(let i=0;i<observations.mlpShape[1];i++)max=Math.max(max,Math.abs(mlp(s,i)));
      close(Number(await scaleFor('mlp').getAttribute('data-max')),max,'所有位置使用同一绝对值范围');
      assert.deepEqual(positive,signedPixel(observed,max));
      assert.equal(Number(await scaleFor('mlp').getAttribute('data-count')),17408);
      assert.equal(Number(await scaleFor('heads').getAttribute('data-count')),0);
      await site.selectOption({index:1});await clock();
      const next=await measured('mlp',2429),negative=await pixel('mlp',2429);
      close(next,mlp(1,2429),'第二处 BF16');colorChecking(next,negative);assert.notDeepEqual(positive,negative);
      assert.deepEqual(negative,signedPixel(next,max));
      report.values.mlp14={unit:2429,atFirst:observed,atSecond:next,positiveRGB:positive,negativeRGB:negative};
    });
    await check('Goodfire 卡跟随当前神经元，候选语境、复核数和普查数据可以与归档对照',async()=>{
      await chooseGoodfire();await card.waitFor();
      const records=require('./result/qwen14_client_v2/native/B36-cues.json').units;
      async function cue(unit){
        await card.getByText('B36/U'+unit,{exact:false}).first().waitFor();
        assert.equal(await card.getAttribute('data-state'),'atlas');
        const text=await card.innerText();
        for(const word of records[unit].positive_keywords.slice(0,3))assert.ok(text.includes(word),'无候选词 '+word);
        return text;
      }
      const initial=await cue(2429);assert.match(initial,/留出/);
      assert.match(initial,/0\s*\/\s*4/);assert.match(initial,/1\.5803/);
      await shot('goodfire-unit',card);
      await locate('mlp',36,2430);const changed=await cue(2430);assert.notEqual(initial,changed);
      await locate('mlp',26,2429);assert.equal(await card.getAttribute('data-state'),'missing');
      assert.ok(!(await card.innerText()).includes(records[2429].positive_keywords[1]));
      report.values.goodfire14=records[2429].positive_keywords.slice(0,3);
    });
    await check('Head 着色取 128 个原生 Z 通道的 RMS，相邻未测头保持缺测颜色',async()=>{
      await locate('attn',26,36);
      const selector=legend.getByLabel('着色读出位置');
      await selector.selectOption('confirm-000-02');await clock();
      const expected=rms(globalThis.AI2050HeadArchive.cases['confirm-000-02'].values);
      close(await measured('heads',36),expected,'14B 单头 RMS');
      assert.equal(Number(await scaleFor('heads').getAttribute('data-count')),1);
      const observedPixel=await pixel('heads',36),missingPixel=await pixel('heads',35);
      assert.deepEqual(missingPixel,[198,205,214]);assert.notDeepEqual(observedPixel,missingPixel);
      await locate('attn',26,35);assert.equal(await valueFor('heads').getAttribute('data-value'),'');
      assert.equal(await valueFor('heads').getAttribute('data-index'),'35');
      report.values.head14={head:36,rms:expected,observedRGB:observedPixel,missingRGB:missingPixel};
    });
    await check('4B 的 Head 和 MLP 复用同一个真实 token，数值来自所选模型',async()=>{
      const vm=require('node:vm'),globals={window:{},atob,TextDecoder};
      vm.runInNewContext(fs.readFileSync(path.join(root,'research/glm5/docs/previews/free-model-data.js'),'utf8'),globals);
      const fields=globals.window.FreeModelData;
      await model.selectOption('Qwen3-4B');await locate('attn',30,6);
      const z=fields.fields('Qwen3-4B','head',30,6).find(f=>f.measure==='head_z');
      assert.equal(z.position,201);close(await measured('heads',6),rms(z.values),'4B 实测 Head RMS');
      assert.equal(Number(await scaleFor('heads').getAttribute('data-count')),32);
      assert.equal(Number(await scaleFor('mlp').getAttribute('data-count')),9728);
      await shot('head-color',scene);
      await locate('mlp',30,9517);
      const field=fields.fields('Qwen3-4B','ffn',30,9517)[0];
      assert.equal(field.position,z.position);close(await measured('mlp',9517),field.values[9517],'4B MLP');
      await chooseGoodfire();assert.equal(await card.getAttribute('data-state'),'probe');
      await card.getByText('三组原句与逐句结果',{exact:true}).click();
      assert.match(await card.innerText(),/5\.219/);assert.match(await card.innerText(),/9517/);
      report.values.head4={block:30,head:6,position:z.position,rms:rms(z.values),mlp9517:field.values[9517]};
    });
    await check('全屏中可切换神经元、查看同一张特征卡与图例，再返回实时控制台',async()=>{
      await button(scene,'⛶ 全屏').click();await clock();
      assert.ok(await scene.evaluate(e=>e.classList.contains('max')));
      await locate('mlp',30,80);await card.waitFor();assert.ok((await card.innerText()).includes('U80'));
      assert.equal(await valueFor('mlp').getAttribute('data-index'),'80');
      await shot('fullscreen-features');
      await mode('实时模式');await live.waitFor();await above(live,plugins);
      await button(scene,'退出全屏（Esc）').click();
      assert.ok(!await scene.evaluate(e=>e.classList.contains('max')));
      await above(live,plugins);
    });
    await check('从自由观察的 FFN 地址直达对应报告，并可切换另一模型的归档实例',async()=>{
      const free=new URL('encoding-workbench.html',url);
      free.hash=new URLSearchParams({model:'Qwen3-14B',at:'B36/U2429',panel:'locator'});
      await page.goto(free.href);
      await page.getByRole('link',{name:'查看 Goodfire 分析 →'}).click();
      await card.waitFor();assert.match(await card.locator('h3').innerText(),/B36\/U2429/);
      assert.equal(await card.getAttribute('data-state'),'atlas');
      await button(card,'Qwen3-4B · B30/U9517').click();
      await card.getByRole('heading',{name:/B30\/U9517/}).waitFor();
      assert.equal(await model.inputValue(),'Qwen3-4B');assert.equal(await card.getAttribute('data-state'),'probe');
      await page.reload();await card.getByRole('heading',{name:/B30\/U9517/}).waitFor();
    });
    assert.deepEqual(report.network,[]);assert.deepEqual(report.errors,[]);
    report.passed=true;
  }catch(err){report.failure=err.message;throw err;}
  finally {
    await browser?.close();
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2)+'\n');
    const target=fs.realpathSync.native(scratch);assert.equal(path.dirname(target),fs.realpathSync.native(tmp));assert.ok(path.basename(target).startsWith('feature-console-'));
    fs.rmSync(target,{recursive:true,force:true,maxRetries:4,retryDelay:150});
  }
}
verify().catch(err=>{console.error(err);process.exitCode=1;});
