'use strict';

const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {pathToFileURL} = require('node:url');
const {checkedDir} = require('../../frontend/electron/storagePolicy');

const root = path.resolve(__dirname,'../..');
const out = checkedDir(path.join(__dirname,'result/model_analysis_native_review'),32*2**20);
const scratch = checkedDir(path.join(root,'tests/glm5_temp'),320*2**20);
const temp = fs.mkdtempSync(path.join(scratch,'model-analysis-native-review-'));
for(const key of ['TEMP','TMP','TMPDIR']) process.env[key]=temp;
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const executablePath = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const file = path.join(root,'research/glm5/docs/previews/model-analysis-native-review.html');
const report = {passed:false,checks:[],errors:[],http:[],screenshots:[]};

async function verify(){
  let browser;
  try{
    browser=await chromium.launch({headless:true,executablePath});
    const page=await browser.newPage({viewport:{width:1440,height:1000},locale:'zh-CN'});
    page.setDefaultTimeout(7000);
    await page.route(/^https?:/,r=>{report.http.push(r.request().url());return r.abort();});
    page.on('pageerror',e=>report.errors.push(e.message));
    page.on('requestfailed',r=>{if(!r.url().startsWith('http'))report.errors.push(r.url()+' '+r.failure()?.errorText);});
    await page.goto(pathToFileURL(file).href);
    const form=page.getByRole('form',{name:'模型与原生地址'});
    const input=page.getByLabel('原生地址',{exact:true});
    const label=page.locator('#scene-address');
    const field=page.locator('[data-native-inspector]');
    const size=()=>field.getAttribute('data-size');
    const raw=()=>field.locator('[data-field-value]').innerText();
    const stats=()=>field.locator('[data-field-stats]').innerText();
    const address=async value=>{
      await input.fill(value);
      await form.getByRole('button',{name:'定位',exact:true}).click();
    };
    const shot=async name=>{await page.screenshot({path:path.join(out,name),fullPage:true});report.screenshots.push(name);};
    assert.equal(await label.innerText(),'B35/H31/D127');
    assert.equal(await size(),'128');
    const totals=await page.evaluate(()=>{
      const f=window.FreeModelData;
      const m=f.model('Qwen3-4B');
      let heads=0, units=0, attention=0;
      for(let b=0;b<m.layers;b++){
        units+=f.fields(m.id,'ffn',b)[0].values.length;
        for(let h=0;h<m.heads;h++){
          const r=f.fields(m.id,'head',b,h);
          if(r.length!==4 || r.some(v=>v.values.length!==m.headDim)) throw Error('头通道覆盖不完整');
          heads++; attention+=r[0].attention.weights.length;
        }
      }
      return {heads,units,attention};
    });
    assert.deepEqual(totals,{heads:1152,units:350208,attention:1152*202});
    report.qwen4b=totals;
    await page.getByRole('button',{name:'本层全部 Head',exact:true}).click();
    assert.equal(await page.locator('#head-list > button').count(),32);
    await page.locator('#head-list [data-head="12"]').click();
    assert.equal(await label.innerText(),'B35/H12/D127');
    for(const key of ['head_q','head_k','head_v','head_z']){
      await page.getByLabel('数值字段',{exact:true}).selectOption(key);
      const expect=await page.evaluate(k=>window.FreeModelData.fields('Qwen3-4B','head',35,12).find(r=>r.measure===k).values[127],key);
      assert.equal(Number(await raw()),expect);
    }
    await shot('01-全部Head与头内通道.png');
    await page.getByLabel('数值字段',{exact:true}).selectOption('attention');
    assert.equal(await size(),'202');
    await field.getByLabel('原生编号',{exact:true}).fill('201');
    await field.getByRole('button',{name:'定位',exact:true}).click();
    assert.match(await page.locator('#source-index').innerText(),/来源 token 201/);
    report.checks.push('4B 同一轨迹全层 1,152 头与 MLP 覆盖，手动切换单头 Q/K/V/Z 及来源 token');

    await address('B0/U0');
    assert.equal(await label.innerText(),'B0/U0');
    assert.equal(await size(),'9728');
    await address('B35/U9727');
    assert.equal(await label.innerText(),'B35/U9727');
    const last4b=await page.evaluate(()=>window.FreeModelData.fields('Qwen3-4B','ffn',35)[0].values[9727]);
    assert.equal(Number(await raw()),last4b);
    await address('B35/U9728');
    assert.match(await page.getByRole('alert').innerText(),/0–9727/);
    assert.equal(await label.innerText(),'B35/U9727');
    await field.getByRole('button',{name:'数值表',exact:true}).click();
    await field.getByRole('button',{name:'U9727',exact:true}).waitFor();
    await field.getByRole('button',{name:'热力图',exact:true}).click();
    report.checks.push('合法首末编号、越界保护和原生值分页');

    await page.getByRole('button',{name:'测试记录',exact:true}).click();
    await page.locator('article:has-text("Phase 3053")').getByRole('button',{name:'在同一工作区打开'}).click();
    assert.equal(await input.inputValue(),'B23/U7708');
    assert.equal(await size(),'17408');
    assert.equal(Number(await raw()),0.58984375);
    const b23=await page.evaluate(()=>window.NativeReviewB23);
    const original=fs.readFileSync(path.join(root,b23.previewSource.path));
    assert.equal(createHash('sha256').update(original).digest('hex'),b23.previewSource.sha256);
    const origRows=JSON.parse(original).rows;
    for(let i=0;i<2;i++) assert.ok(Buffer.from(origRows[i].bits,'base64').equals(Buffer.from(b23.rows[i].bits,'base64')));
    await page.getByLabel('观察位置',{exact:true}).selectOption('1');
    assert.equal(Number(await raw()),-0.6328125);
    await address('B23/U17407');
    assert.equal(Number(await raw()),await page.evaluate(()=>{
      const o=window.NativeReviewB23;
      const bytes=Uint8Array.from(atob(o.rows[1].bits),c=>c.charCodeAt(0));
      const read=new DataView(bytes.buffer), val=new DataView(new ArrayBuffer(4));
      val.setUint32(0,read.getUint16(2*17407,true)*65536,true); return val.getFloat32(0,true);
    }));
    assert.match(await stats(),/已采集 17,408 \/ 17,408/);
    await shot('02-整层MLP全量热力图.png');
    await address('B22/U17407');
    assert.equal(await size(),'17408');
    assert.match(await stats(),/已采集 0 \/ 17,408/);
    assert.equal(await raw(),'未采集');
    report.checks.push('测试记录恢复原始地址；B23 两条 BF16 全向量逐字节匹配，邻块缺测不串用');

    await page.getByRole('button',{name:'测试记录',exact:true}).click();
    await page.locator('article:has-text("Phase 2983")').getByRole('button',{name:'在同一工作区打开'}).click();
    assert.equal(await input.inputValue(),'B26/H36/D0');
    assert.equal(await size(),'128');
    await page.getByLabel('数值字段',{exact:true}).selectOption('head_q');
    assert.equal(await raw(),'未采集');
    assert.match(await field.locator('[data-field-missing]').innerText(),/当前部件或字段没有对应向量/);
    await page.getByRole('button',{name:'测试记录',exact:true}).click();
    await page.locator('article:has-text("Phase 2933")').getByRole('button',{name:'在同一工作区打开'}).click();
    await field.getByRole('button',{name:'位置对照',exact:true}).click();
    assert.equal(await field.locator('[data-field-compare-value]').count(),4);
    await page.getByLabel('模型快照',{exact:true}).selectOption('GLM-4-9B');
    const last=await page.evaluate(()=>{const m=FreeModelData.model('GLM-4-9B');return 'B'+(m.layers-1)+'/U'+(m.ffn-1);});
    await address(last);
    assert.equal(await label.innerText(),last);
    assert.match(await field.locator('[data-field-missing]').innerText(),/模型快照不匹配/);
    assert.equal(await raw(),'未采集');
    await shot('03-可定位但无匹配响应.png');
    report.checks.push('14B 头部归档只开实采字段；位置对照四值；GLM 自由寻址与身份不匹配');

    await page.getByRole('button',{name:'实时模式',exact:true}).click();
    await page.getByLabel('输入',{exact:true}).fill('待采集材料');
    await page.getByRole('button',{name:'特征分析',exact:true}).click();
    await page.getByRole('button',{name:'实时模式',exact:true}).click();
    assert.equal(await page.getByLabel('输入',{exact:true}).inputValue(),'待采集材料');
    await page.getByRole('button',{name:'⛶ 全屏观察',exact:true}).click();
    assert.ok(await page.locator('#workspace').evaluate(el=>el.classList.contains('full-screen')));
    await page.keyboard.press('Escape');
    assert.ok(!(await page.locator('#workspace').evaluate(el=>el.classList.contains('full-screen'))));
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.getByRole('button',{name:'特征分析',exact:true}).click();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    report.checks.push('采集入口、模式之间上下文、全屏与 390 像素窄屏');
    assert.deepEqual(report.errors,[]);
    assert.deepEqual(report.http,[]);
    report.passed=true;
  }finally{
    if(browser)await browser.close();
    const resolved=fs.realpathSync(temp);
    assert.equal(path.dirname(resolved),fs.realpathSync(scratch));
    assert.ok(path.basename(resolved).startsWith('model-analysis-native-review-'));
    const bytes=dir=>fs.readdirSync(dir,{withFileTypes:true}).reduce((sum,e)=>{
      const f=path.join(dir,e.name);assert.ok(!e.isSymbolicLink());
      return sum+(e.isDirectory()?bytes(f):fs.statSync(f).size);
    },0);
    report.removedTransientBytes=bytes(resolved);
    fs.rmSync(resolved,{recursive:true,force:true});
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2));
  }
}
verify().then(()=>console.log(JSON.stringify(report))).catch(e=>{console.error(e);process.exitCode=1;});

