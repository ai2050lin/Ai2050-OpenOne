'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {createHash} = require('node:crypto');
const {checkedDir,checkWrite} = require('../../frontend/electron/storagePolicy.js');
const rawField = require('../../frontend/src/components/v8/data/graphObservationExample.json');
const headRun = require('../../frontend/src/components/v8/threeCueEvidence.json').runs[0];
const prompts = require('../../frontend/src/components/v8/data/archiveHeadInputs.json').prompts;

const root = path.resolve(__dirname,'../..');
const output = path.join(__dirname,'result/encoding_workbench_design');
const preview = path.join(root,'research/glm5/docs/previews/encoding-cases.html');

function sourceDigest(base64) {
  // Render data is float32. Expanding BF16 by copying its high word is exact.
  const original = Buffer.from(base64,'base64');
  const decoded = Buffer.alloc(original.length * 2);
  for (let i=0;i<original.length;i+=2) decoded.writeUInt16LE(original.readUInt16LE(i),2*i+2);
  return createHash('sha256').update(decoded).digest('hex');
}

async function verify() {
  checkWrite(output,280*2**20,'模型分析静态稿浏览器检查');
  checkedDir(output,24*2**20,'模型分析浏览器截图和回执');
  const tempRoot = checkedDir(path.join(root,'tests/glm5_temp'),256*2**20,'浏览器临时数据');
  const scratch = fs.mkdtempSync(path.join(tempRoot,'encoding-ui-'));
  for(const key of ['TEMP','TMP','TMPDIR'])process.env[key]=scratch;
  const report = {passed:false,checks:[],screenshots:[],network:[],errors:[]};
  let browser;
  try {
    const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
    const executablePath = process.env.CHROME_PATH || path.join(os.homedir(),
      'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
    browser = await chromium.launch({headless:true,executablePath});
    report.browser = browser.version();
    const context = await browser.newContext({viewport:{width:1440,height:960},locale:'zh-CN'});
    await context.route(/^https?:/,route=>{report.network.push(route.request().url());return route.abort();});
    const page = await context.newPage();
    page.setDefaultTimeout(5000);
    page.on('pageerror',e=>report.errors.push(e.message));
    page.on('requestfailed',r=>{if(r.url().startsWith('file:'))report.errors.push(r.url());});
    const id = name=>page.locator('#'+name);
    const tab = name=>page.locator('[data-tab="'+name+'"]');
    const mode = name=>page.locator('[data-mode="'+name+'"]');
    const settle = ()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
    async function check(title,act){
      try { await act(); report.checks.push(title); }
      catch(error){ error.message=title+' — '+error.message; throw error; }
    }
    async function shot(name){
      await settle();
      await page.screenshot({path:path.join(output,name+'.png'),animations:'disabled'});
      report.screenshots.push(name+'.png');
    }
    async function fits() {
      await settle();
      const [pageWidth,device] = await page.evaluate(()=>[document.documentElement.scrollWidth,innerWidth]);
      assert.ok(pageWidth<=device+1,'页面宽度 '+pageWidth+' 超出 '+device);
    }

    await page.goto(pathToFileURL(preview).href);
    await page.evaluate(()=>document.fonts.ready);
    await id('scene-canvas').waitFor();
    const canvas=await id('scene-canvas').elementHandle();
    await check('同族三道历史题的物理行、目标、模型输出及原题逐项匹配归档',async()=>{
      const rows=await page.evaluate(()=>EncodingEvidence.headCases.map(x=>({
        id:x.id,output:x.output,gold:x.gold,winner:x.winner,rows:x.rows.map(r=>r.mass),prompt:x.prompt
      })));
      const cases=headRun.families[0].cases.filter(x=>!x.labelSwap&&!x.positionSwap);
      assert.equal(cases.length,3);
      for(const c of cases){
        const actual=rows.find(x=>x.id===c.id);
        assert.ok(actual,c.id);
        assert.equal(actual.output,c.output);
        assert.equal(actual.gold,c.order.indexOf(c.gold));
        assert.equal(actual.winner,c.order.indexOf(c.mass.indexOf(Math.max(...c.mass))));
        assert.deepEqual(actual.rows,c.order.map(i=>c.mass[i]));
        assert.equal(actual.prompt,prompts[c.id]);
      }
    });
    await check('全部 FFN 与 HiddenState 的位置、层、坐标顺序及原值精度与归档的字节一致',async()=>{
      const digests=await page.evaluate(async()=>{
        async function digest(vectors) {
          const combined=new Float32Array(vectors.reduce((n,v)=>n+v.length,0));
          let at=0;
          for(const v of vectors){combined.set(v,at);at+=v.length;}
          const bytes=await crypto.subtle.digest('SHA-256',combined);
          return Array.from(new Uint8Array(bytes),x=>x.toString(16).padStart(2,'0')).join('');
        }
        const e=EncodingEvidence, hidden=[];
        for(let l=0;l<e.ffn.hiddenLayers;l++)for(let p=0;p<e.ffn.positions.length;p++)hidden.push(e.hiddenValues(l,p));
        return {ffn:await digest(e.ffn.positions.map((_,site)=>e.ffnValues(site))),hidden:await digest(hidden)};
      });
      assert.equal(digests.ffn,sourceDigest(rawField.mlp));
      assert.equal(digests.hidden,sourceDigest(rawField.hidden));
      report.sourceDigests=digests;
    });
    await fits();
    await shot('desktop');
    // Further checks exercise only user-visible controls and their associated evidence.
    const figure=id('measure-figure'), panel=id('work-panel');
    async function locate(coord){
      await id('locate').fill(coord);
      await id('locate').press('Enter');
      await settle();
    }
    await check('同一份五行材料可切换姓名、标签和首行，原题可关闭返回原对照',async()=>{
      assert.equal(await id('question').inputValue(),'head');
      await id('select-a').selectOption('0');
      await id('select-b').selectOption('2');
      await tab('contrast').click();
      const text=await page.locator('.surface').innerText();
      assert.ok(text.includes('wood') && text.includes('silver'));
      const columns=await panel.locator('tbody tr').evaluateAll(list=>list.map(r=>Array.from(r.cells,c=>c.textContent)));
      assert.match(columns[0][1],/第\s*2\s*行/);assert.match(columns[0][2],/第\s*2\s*行/);
      assert.match(columns[1][1],/第\s*1\s*行/);assert.match(columns[1][2],/第\s*4\s*行/);
      assert.equal(columns[1][3],'silver');
      await panel.getByText(/对照 128 个确认家族/, {exact:false}).click();
      assert.match(await panel.innerText(),/33\.4%/);
      await id('material').click();
      assert.ok(await id('material-dialog').isVisible());
      const dialogText=await id('material-dialog').textContent();
      assert.ok(dialogText.includes('Tebaka'));
      await id('close-material').click();
      assert.equal(await id('select-b').inputValue(),'2');
      const initial=await figure.innerText();
      await id('select-a').selectOption('1');
      assert.notEqual(await figure.innerText(),initial);
      assert.ok((await page.locator('.surface').innerText()).includes('lion'));
    });
    await check('显示插件能调整结构网格并复位视角',async()=>{
      await id('display').click();
      await id('grid').uncheck();assert.equal(await id('grid').isChecked(),false);
      await id('grid').check();
      await id('reset-camera').click();
      if(await id('grid').isVisible())await id('display').click();
    });
    const plugin=name=>page.locator('[data-et-tool="'+name+'"]');
    await check('头的 Z 向量与原值表相连，未归档的 Q、未定位的相邻头均有缺件提示',async()=>{
      await tab('tools').click();await plugin('Head').click();
      assert.deepEqual((await page.locator('[data-et-tool]').allTextContents()).sort(),['Head','HiddenState','Logit Lens','MLP']);
      await page.locator('[data-et="boundary"]').selectOption('Z');
      const real=await page.locator('[data-field-value]').innerText();
      assert.ok(Number.isFinite(Number(real)) && real.trim()!=='');
      await page.locator('[data-et="boundary"]').selectOption('Q');
      assert.notEqual(await page.locator('[data-field-value]').innerText(),real);
      assert.match(await panel.innerText(),/未采集|缺/);
      await page.locator('[data-et="boundary"]').selectOption('Z');
      await locate('B26/H35');assert.match(await panel.innerText(),/缺少|未采集|没有/);
      await locate('B26/H36');await page.locator('[data-field-value]').waitFor();
    });
    await check('三个工作模式复用同一模型画布',async()=>{
      for(const key of ['features','live','archive']){
        await mode(key).click();
        assert.ok(await canvas.evaluate(node=>node===document.querySelector('#scene-canvas')));
      }
    });
    await check('全屏内可切换问题并回到相同画布',async()=>{
      await id('screen').click();
      await page.locator('#analysis.full').waitFor();
      await id('question').selectOption('ffn');
      assert.ok(await canvas.evaluate(node=>node===document.querySelector('#scene-canvas')));
      await id('screen').click();
      await page.locator('#analysis.full').waitFor({state:'hidden'});
      assert.equal(await id('question').inputValue(),'ffn');
    });
    await check('单元位置对照、单坐标跨层和同层几何比较采用不同的原生测量',async()=>{
      await locate('B36/U2429');await tab('tools').click();await plugin('MLP').click();
      await page.locator('[data-field-view="positions"]').click();
      assert.deepEqual((await page.locator('[data-field-compare-value]').allTextContents()).map(Number),
        [2.109375,-.4765625,-.447265625,.79296875]);
      await plugin('HiddenState').click();
      await page.locator('[data-et="layer"]').selectOption('0');
      const expected=await page.evaluate(()=>{
        const e=EncodingEvidence;
        return ['select-a','select-b'].map(id=>e.hiddenValues(0,Number(document.getElementById(id).value))[0]);
      });
      assert.deepEqual((await page.locator('[data-et-hidden] .et-reading').allTextContents()).map(Number),expected);
      await panel.locator('.et-manage summary').click();
      await page.locator('[data-et-enable="geometry"]').check();await plugin('geometry').click();
      await page.locator('[data-et="layer"]').selectOption('36');
      await page.locator('[data-et="geometry-layer"]').selectOption('36');
      assert.deepEqual((await page.locator('[data-et-geometry] .et-reading').allTextContents()).map(Number),[1,1]);
      await plugin('lens').click();
      assert.match(await panel.innerText(),/权重|缺少/);
      await tab('contrast').click();await fits();await shot('ffn');
    });
    await check('已存笔记在切换题目后恢复到相同单元、两处读出及其原值',async()=>{
      await tab('evidence').click();
      const pair=[await id('select-a').inputValue(),await id('select-b').inputValue()];
      const note='复核 U2429 在两个读出位置的响应';
      await panel.locator('[name="memo"]').fill(note);
      await panel.locator('[data-et-note] button[type="submit"]').click();
      assert.ok((await panel.innerText()).includes(note));
      await id('question').selectOption('head');await tab('evidence').click();
      await panel.locator('[data-et-restore]').first().click();
      assert.equal(await id('question').inputValue(),'ffn');
      assert.equal(await id('locate').inputValue(),'B36/U2429');
      assert.deepEqual([await id('select-a').inputValue(),await id('select-b').inputValue()],pair);
      await shot('evidence');
    });
    await check('观察笔记保存主图视角和原生索引，行注意力、头 Z、FFN 激活和 HiddenState 互不混用',async()=>{
      await id('angle').selectOption('hidden');
      await id('metric-index').fill('411');
      await id('metric-index').press('Tab');
      const hiddenView=await figure.innerText();
      await panel.locator('[name="memo"]').fill('残差流原生坐标 411');
      await panel.locator('[data-et-note] button[type="submit"]').click();
      await id('question').selectOption('head');
      await id('angle').selectOption('rows');
      await tab('evidence').click();
      await panel.locator('[name="memo"]').fill('行注意力读数');
      await panel.locator('[data-et-note] button[type="submit"]').click();
      const rowNote=panel.locator('.et-notes li').filter({hasText:'行注意力读数'});
      assert.match(await rowNote.innerText(),/行|注意力/);
      assert.doesNotMatch(await rowNote.locator('b').innerText(),/\bZ\b|D\d/);
      await id('angle').selectOption('z');
      await rowNote.locator('[data-et-restore]').click();
      assert.equal(await id('angle').inputValue(),'rows');
      await panel.locator('.et-notes li').filter({hasText:'残差流原生坐标 411'}).locator('[data-et-restore]').click();
      assert.equal(await id('question').inputValue(),'ffn');
      assert.equal(await id('angle').inputValue(),'hidden');
      assert.equal(await id('metric-index').inputValue(),'411');
      assert.equal(await figure.innerText(),hiddenView);
    });
    await check('自定义实时输入清除原题数值，可以显式回放恢复',async()=>{
      await mode('live').click();
      await id('live-input').fill('新的输入，尚未有对应采集');
      const pending=await figure.innerText();
      assert.match(pending,/未采集|未保存|待采集|缺少|尚无|没有/);
      await id('replay').click();
      assert.notEqual(await figure.innerText(),pending);
      await mode('archive').click();
    });
    await check('中文句对切换到 4B 原始量表；未测句对与单元组合保持缺件',async()=>{
      await id('question').selectOption('zh');
      await id('select-a').selectOption('0');
      await id('select-b').selectOption('1');
      await tab('contrast').click();
      assert.match(await page.locator('.model-label').innerText(),/Qwen3-4B/);
      const source=await page.locator('.surface').innerText();
      assert.match(source,/香港/);assert.match(source,/巴黎/);
      assert.match(source,/6\.381/);assert.match(source,/1\.162/);assert.match(source,/5\.219/);assert.match(source,/标准分/);
      await id('select-b').selectOption('0');
      assert.match(await figure.innerText(),/1\.165/);
      assert.match(await panel.innerText(),/\+0\.240/);
      await shot('chinese-pair');
      await id('select-a').selectOption('2');
      await locate('B30/U80');
      const unavailable=await figure.innerText();
      assert.match(unavailable,/缺|尚无|没有|未采|未收|未保|未归|未测|待补/);
      assert.doesNotMatch(unavailable,/6\.381|1\.162|5\.219|5\.033/);
      await id('question').selectOption('head');
      assert.match(await page.locator('.model-label').innerText(),/Qwen3-14B/);
      assert.doesNotMatch(await figure.innerText(),/香港|标准差/);
    });
    await check('窄屏可选条件、开原题并返回',async()=>{
      await page.setViewportSize({width:390,height:844});
      await id('question').selectOption('head');await tab('contrast').click();
      await fits();
      await id('material').click();await fits();
      await id('close-material').click();
      await id('question').scrollIntoViewIfNeeded();
      await page.evaluate(()=>window.scrollTo(0,0));
      await shot('mobile');
    });
    await check('离线页面没有网络请求或脚本异常',async()=>{
      assert.deepEqual(report.network,[]);assert.deepEqual(report.errors,[]);
    });
    report.passed=true;
  } catch(error) {
    report.failure=error.message;
    throw error;
  } finally {
    if(browser)await browser.close();
    fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify(report,null,2)+'\n');
    const actual=fs.realpathSync.native(scratch);
    assert.equal(path.dirname(actual),fs.realpathSync.native(tempRoot));
    assert.ok(path.basename(actual).startsWith('encoding-ui-'));
    fs.rmSync(actual,{recursive:true,force:true,maxRetries:4,retryDelay:200});
  }
  console.log(JSON.stringify(report));
}
verify().catch(error=>{console.error(error);process.exitCode=1;});
