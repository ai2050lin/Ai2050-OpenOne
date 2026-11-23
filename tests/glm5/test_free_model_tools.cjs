'use strict';

// Offline engineering acceptance for the mathematical instruments in the design preview.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {checkedDir,checkWrite} = require('../../frontend/electron/storagePolicy.js');

async function main() {
  const root=path.resolve(__dirname,'../..');
  const temp=checkedDir(path.join(root,'tests/glm5_temp'),180*2**20,'工具验收临时目录');
  const output=checkedDir(path.join(__dirname,'result/free_model_tools_design'),2*2**20,'工具验收回执');
  const profile=fs.mkdtempSync(path.join(temp,'free-model-tools-'));
  for (const name of ['TEMP','TMP','TMPDIR']) process.env[name]=profile;
  const receipt={kind:'offline_ui_acceptance',created:new Date().toISOString(),passed:false,checks:[],errors:[],requests:[]};
  let browser;
  try {
    const {chromium}=require(process.env.PLAYWRIGHT_CORE||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
    const executablePath=process.env.CHROME_PATH||path.join(os.homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
    browser=await chromium.launch({headless:true,executablePath});
    receipt.browser=browser.version();
    const page=await browser.newPage({viewport:{width:1440,height:960},locale:'zh-CN'});
    page.setDefaultTimeout(5000);
    page.on('pageerror',e=>receipt.errors.push(e.message));
    await page.route(/^https?:/,r=>{receipt.requests.push(r.request().url());return r.abort();});
    await page.goto(pathToFileURL(path.join(root,'research/glm5/docs/previews/encoding-workbench.html')).href);
    const panel=page.locator('#tools-host');
    const choose=name=>panel.locator('[data-fmt=tool]').selectOption(name);
    const dock=id=>page.locator('[data-open-window="'+id+'"]').first().click();
    async function jump(addr) {
      await dock('locator');await page.locator('#address').fill(addr);await page.locator('#address').press('Enter');await dock('tools');
    }
    async function execute(id) {
      await choose(id);await panel.locator('[data-fmt-run]').click();
    }
    async function metrics() {
      await panel.locator('.fmt-metrics dd').first().waitFor();
      return panel.locator('.fmt-metrics dd').allTextContents();
    }
    async function check(name,run) {const facts=await run();receipt.checks.push({name,...facts});}

    await check('14B B26/H36 单头 128 维 Z 的 RMS 与条件向量余弦',async()=>{
      await jump('B26/H36');await execute('summary');const summary=await metrics();
      await execute('compare');const compared=await metrics();
      assert.equal(summary[1],'0.662609');assert.equal(compared[0],'0.720029');
      return {rms:summary[1],cosine:compared[0]};
    });
    await check('14B 正文行及其余 token 的六组分布',async()=>{
      await execute('attention');const measures=await metrics();assert.equal(await panel.locator('.fmt-masses div').count(),6);
      return {entropy:measures[0],groups:6};
    });
    await check('14B 四个位置的残差层间 CKA / RSA',async()=>{
      await jump('hs[37]/D0');await execute('geometry');const vals=await metrics();
      assert.ok(vals.every(v=>Number.isFinite(Number(v))));return {cka:vals[0],rsa:vals[1]};
    });
    await check('14B B26 查询投影真实 8×8 切片的奇异值与平方和',async()=>{
      await jump('model.layers.26.self_attn.q_proj.weight[4608,0]');await execute('spectrum');
      const vals=await metrics();assert.equal(vals[0],'0.133192');
      const s=await panel.locator('.fmt-spectrum output').allTextContents();assert.equal(s.length,8);
      const energy=await page.evaluate(()=>FreeModelData.tile('Qwen3-14B','model.layers.26.self_attn.q_proj.weight',4608,0).values.flat().reduce((n,x)=>n+x*x,0));
      assert.ok(Math.abs(s.reduce((n,x)=>n+x*x,0)-energy)<1e-6);
      return {largest:Number(s[0]),frobeniusSquared:energy};
    });
    await check('权重切片有缺值时拒绝计算',async()=>{
      await jump('model.layers.26.self_attn.q_proj.weight[999,999]');await execute('spectrum');
      await panel.getByText(/权重切片含缺值/).waitFor();return {};
    });
    await check('两个模型的最终归一化边界和中间层透镜，与独立点积对齐',async()=>{
      const records=[];
      for (const [id,layer,alreadyNormalized] of [['Qwen3-14B',40,false],['Qwen3-4B',36,true],['Qwen3-4B',1,false]]) {
        await page.locator('#model-select').selectOption(id);await jump('hs['+layer+']/D0');await execute('lens');
        const shown=(await metrics()).map(Number);
        const expected=await page.evaluate(async({id,layer,alreadyNormalized})=>{
          const model=FreeModelData.model(id),h=FreeModelData.fields(id,'hidden',layer,0)[0].values;
          const scale=Math.sqrt(h.reduce((s,x)=>s+x*x/h.length,0)+model.rmsNormEps);
          const gamma=(await FreeModelData.readWeightVector(id,'model.norm.weight')).values;
          const normalized=alreadyNormalized?h:Array.from(h,(v,i)=>v*gamma[i]/scale);
          const logits=[];
          for (const token of [32,33]) {
            const row=(await FreeModelData.readWeightVector(id,'lm_head.weight',token)).values;
            logits.push(normalized.reduce((n,v,i)=>n+v*row[i],0));
          }
          return [logits[0],logits[1],logits[0]-logits[1]];
        },{id,layer,alreadyNormalized});
        assert.ok(shown.every((value,i)=>Math.abs(value-expected[i])<0.0001));
        records.push({id,layer,alreadyNormalized,shown,expected});
      }
      return {records};
    });
    await check('未附输出权重行和错误目标对象分别给出所需输入',async()=>{
      await panel.locator('[data-fmt=tokenB]').fill('11111');await panel.locator('[data-fmt=tokenB]').press('Tab');await panel.locator('[data-fmt-run]').click();
      await panel.getByText(/缺少完整输出权重行/).waitFor();
      await jump('B9/H6');await panel.getByText(/需要有实测向量的 HiddenState/).waitFor();return {};
    });
    await check('4B B9/H6 实际共享 KV 组的 Q/K 分数',async()=>{
      await execute('qk');const vals=await metrics();assert.equal(vals[0],'5.0778');
      return {scaledQk:Number(vals[0]),width:Number(vals[1])};
    });
    await check('4B 逐 token 熵、前五项质量及 202 个原生位置',async()=>{
      await execute('attention');const vals=await metrics();
      const n=await panel.locator('.fmt-table-scroll tbody tr').count();
      const h=await page.evaluate(()=>{
        const w=FreeModelData.fields('Qwen3-4B','head',9,6)[0].attention.weights,t=w.reduce((a,b)=>a+b);
        return -w.reduce((s,v)=>s+(v?v/t*Math.log2(v/t):0),0);
      });
      assert.equal(n,202);assert.ok(Math.abs(h-parseFloat(vals[0]))<1e-5);return {entropy:h,sources:n,topFive:vals[2]};
    });
    await check('GLM-4-9B 独立参数目录、融合权重 SVD 和激活缺件',async()=>{
      await page.locator('#model-select').selectOption('GLM-4-9B');
      await jump('B0/H0');await choose('summary');await panel.getByText(/缺少这个模型和位置的完整激活向量/).waitFor();
      await jump('model.layers.0.mlp.gate_up_proj.weight[0,0]');await execute('spectrum');
      const vals=await metrics();assert.ok(Number(vals[0])>0);
      const shape=await page.evaluate(()=>FreeModelData.tensor('GLM-4-9B','model.layers.0.mlp.gate_up_proj.weight').shape);
      assert.equal(shape[0],2*13696);return {shape,largest:Number(vals[0])};
    });
    await check('原生 FFN 尾号、残差网格与 Head 热图的读数范围',async()=>{
      await page.locator('#model-select').selectOption('Qwen3-14B');await jump('B36/U17407');await dock('readout');
      const r=page.locator('#readout-host');assert.equal(await r.locator('.native-field').getAttribute('data-size'),'17408');
      assert.equal(await r.locator('.native-field').getAttribute('data-selected'),'17407');
      await jump('hs[37]/D5119');await dock('readout');assert.ok(await r.locator('[data-fmt-grid]').isVisible());
      assert.match(await r.locator('.fmt-coordinate').textContent(),/d5119/);
      await jump('B26/H36/D127');await dock('readout');assert.equal(await r.locator('.native-field').getAttribute('data-selected'),'127');
      return {ffn:17408,hidden:5120,head:128};
    });
    assert.deepEqual(receipt.requests,[]);assert.deepEqual(receipt.errors,[]);receipt.passed=true;
  } catch (error) {receipt.errors.push(error.message);throw error;}
  finally {
    await browser?.close();
    const resolved=fs.realpathSync(profile);
    if (path.dirname(resolved)!==fs.realpathSync(temp)||!path.basename(resolved).startsWith('free-model-tools-')) throw Error('浏览器临时路径与预期不符。');
    fs.rmSync(resolved,{recursive:true,force:true});
    checkWrite(output,2*2**20,'工具验收回执');fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify(receipt,null,2)+'\n');
  }
  console.log(receipt.checks.length+' 项工具与数学核验通过 · '+output);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
