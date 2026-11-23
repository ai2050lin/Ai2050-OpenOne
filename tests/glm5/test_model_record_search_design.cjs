'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {checkedDir, checkWrite} = require('../../frontend/electron/storagePolicy.js');

const root = path.resolve(__dirname, '../..');
const preview = path.join(root, 'research/glm5/docs/previews/model-records-design.html');
const resultDir = path.join(__dirname, 'result/model_record_search_design');
const scratch = path.join(root, 'tests/glm5_temp');

async function verify() {
  checkWrite(resultDir, 280 * 2 ** 20, '测试记录搜索静态稿验收');
  checkedDir(resultDir, 24 * 2 ** 20, '浏览器验收截图与记录');
  const parent = checkedDir(scratch, 256 * 2 ** 20, '隔离浏览器资料');
  const profile = fs.mkdtempSync(path.join(parent, 'record-search-preview-'));
  for (const key of ['TEMP','TMP','TMPDIR']) process.env[key] = profile;
  const report = {passed:false, checks:[], errors:[], network:[]};
  let browser;
  try {
    const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
    const executablePath = process.env.CHROME_PATH || path.join(os.homedir(),
      'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
    browser = await chromium.launch({headless:true, executablePath});
    const context = await browser.newContext({viewport:{width:1440,height:900}});
    await context.route(/^https?:/, async route => {
      report.network.push(route.request().url());
      await route.abort();
    });
    const page = await context.newPage();
    page.on('pageerror',e=>report.errors.push(e.message));
    page.setDefaultTimeout(6500);
    const list = page.locator('#record-list');
    const rows = list.locator('[data-record]');
    const view = page.locator('#record-view');
    const canvas = page.locator('canvas#scene, #scene canvas');
    const model = page.locator('#model-filter');
    const question = page.locator('#question-filter');
    const kind = page.locator('#kind-filter');
    const object = page.locator('#object-filter');
    const clear = page.locator('#clear-filters');
    const receipt = page.locator('#receipt-filter');
    const zero = page.locator('#zero-results');
    const back = page.getByRole('button',{name:/返回搜索结果$/});
    const frame = ()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    const shot = name=>page.screenshot({path:path.join(resultDir,name+'.png'),animations:'disabled'});
    const readFilters = ()=>Promise.all([model.inputValue(), question.inputValue(), kind.inputValue(), object.inputValue(), receipt.isChecked()]);
    const ids = ()=>rows.evaluateAll(nodes=>nodes.map(node=>node.dataset.record));
    async function fits() {
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'页面横向溢出');
    }
    async function roundTrip(row,{browserBack=false, sceneIndex='', noLocation=false, screen=''}={}) {
      const recordId = await row.getAttribute('data-record');
      const filters = await readFilters();
      await row.scrollIntoViewIfNeeded();
      const scroll = await list.evaluate(el=>el.scrollTop);
      await row.click();
      await view.waitFor();
      const source=byId.get(recordId);
      assert.equal(await page.locator('#record-result').textContent(),source.result);
      assert.equal(await page.locator('#record-question').textContent(),source.question);
      assert.equal(await page.locator('#record-title').textContent(),source.title);
      for(const metric of source.metrics || []){
        const metrics=await page.locator('#metrics').innerText();
        assert.ok(metrics.includes(String(metric.value)));
      }
      if (sceneIndex) {
        await canvas.waitFor();
        await frame();
        const [block,index] = sceneIndex.split('/');
        const label = await canvas.getAttribute('aria-label');
        assert.ok(label.includes(block) && label.includes(index), '3D 对象错误 '+label+' / '+sceneIndex);
      }
      if(noLocation) {
        await page.locator('#no-location').waitFor();
        assert.ok(await canvas.isHidden(),'缺少原生定位的记录显示了上一题的场景');
      }
      if(screen) await shot(screen);
      if(browserBack) await page.goBack(); else await back.click();
      await list.waitFor();
      await frame();
      assert.deepEqual(await readFilters(),filters);
      const activeId = await page.evaluate(()=>document.activeElement.closest('[data-record]')?.dataset.record);
      assert.equal(activeId,recordId,'列表焦点未恢复');
      assert.ok(Math.abs(await list.evaluate(el=>el.scrollTop)-scroll)<=1, '列表滚动位置未恢复');
      return {id:recordId,scroll};
    }
    await page.goto(pathToFileURL(preview).href);
    await list.waitFor();
    await page.evaluate(()=>document.fonts.ready);
    await frame();
    const records = await page.evaluate(()=>window.modelRecords);
    assert.ok(Array.isArray(records));
    const byId = new Map(records.map(r=>[String(r.id),r]));
    const target = records.find(r=>r.caseId==='confirm-000-02') ||
      records.find(r=>r.objects?.some(o=>o.kind==='Head' && o.block===26 && o.index===36));
    assert.ok(target,'预览缺少 B26/H36');
    await fits();
    await shot('search');
    await model.selectOption({label:target.model});
    await kind.selectOption('Head');
    await object.fill('B26/H36');
    await question.fill(target.question || target.title);
    await frame();
    const found = await ids();
    assert.ok(found.includes(String(target.id)),'四项交叉筛选漏掉实际题目');
    for(const id of found){
      const input = byId.get(id);
      assert.equal(input.model,target.model);
      assert.ok(input.objects.some(o=>o.kind==='Head' && o.block===26 && o.index===36));
    }
    const targetRow=rows.nth(found.indexOf(String(target.id)));
    await roundTrip(targetRow,{sceneIndex:'B26/H36',screen:'head'});
    report.checks.push('模型、题目、Head 类型、原生索引交叉筛选，3D 定位 B26/H36，显式返回同一行');
    await clear.click();
    await kind.selectOption('MLP');await object.fill('B36/U2429');
    await frame();
    const units=await ids();
    assert.ok(units.length);
    for(const id of units) assert.ok(byId.get(id).objects.some(o=>o.kind==='MLP'&&o.index===2429&&o.block===36));
    await roundTrip(rows.first(),{browserBack:true,sceneIndex:'B36/U2429',screen:'mlp'});
    report.checks.push('MLP 单元精确检索、3D 定位与浏览器后退');
    await clear.click();
    const otherModel=records.find(r=>r.model && r.model!==target.model)?.model;
    assert.ok(otherModel,'缺少对照模型');
    await model.selectOption({label:otherModel});await frame();
    const other=await ids();
    assert.ok(other.length);
    assert.ok(other.every(id=>byId.get(id).model===otherModel));
    await question.fill('没有归档的题目-qa-72e');await zero.waitFor();
    assert.equal(await rows.count(),0);
    await clear.click();await frame();
    assert.ok(await rows.count()>other.length);
    report.checks.push('第二模型独立筛选，空结果可清除并恢复');
    await receipt.check();await frame();
    const receiptIds=await ids();
    const expectedReceipts=records.filter(r=>r.status==='中心已读回').map(r=>String(r.id));
    assert.deepEqual(receiptIds.sort(),expectedReceipts.sort());
    if(receiptIds.length) await roundTrip(rows.first());
    report.receipts=receiptIds.length;
    report.checks.push('仅有中心回执的筛选与逐条标记一致');
    await clear.click();
    const multi=records.find(r=>r.snapshot?.length===64 && r.objects?.length>1);
    if(multi) {
      const hit=multi.objects.at(-1);
      const targetIndex='B'+hit.block+'/'+(hit.kind==='MLP'?'U':'H')+hit.index;
      await kind.selectOption(hit.kind);await object.fill(targetIndex);await frame();
      const resultIds=await ids();
      assert.ok(resultIds.includes(String(multi.id)));
      await roundTrip(rows.nth(resultIds.indexOf(String(multi.id))),{sceneIndex:targetIndex});
      report.multiObject={record:multi.id,targetIndex};
      report.checks.push('同条记录的多个原生对象中按精确命中定位');
      await clear.click();
    } else {
      report.multiObject='当前归档摘录缺少可定位的多对象记录';
    }
    const missing = records.find(r=>!(r.objects || []).length);
    if(missing) {
      const visibleIds=await ids();
      await roundTrip(rows.nth(visibleIds.indexOf(String(missing.id))),{noLocation:true});
      report.checks.push('无原生对象的归档保留结果并说明定位缺件');
    }
    await page.setViewportSize({width:1100,height:620});
    await frame();
    const last=rows.last();
    await last.scrollIntoViewIfNeeded();
    const before=await list.evaluate(el=>el.scrollTop);
    assert.ok(before>0,'需有非零滚动位置以检验还原');
    report.scrollReturn=await roundTrip(last);
    report.checks.push('列表非零滚动位置和原行焦点还原');
    await page.setViewportSize({width:390,height:844});
    await model.selectOption({label:target.model});await kind.selectOption('Head');await object.fill('B26/H36');
    await frame();await fits();await shot('mobile');
    await roundTrip(rows.first(),{sceneIndex:'B26/H36'});
    await fits();
    report.checks.push('390 像素窄屏筛选、定位、返回，无横向溢出');
    const previous=path.join(path.dirname(preview),'model-explorer-design.html');
    await page.goto(pathToFileURL(previous).href);
    await page.getByRole('link',{name:/查找历史测试/}).click();
    await list.waitFor();
    await page.getByRole('link',{name:'实时模式',exact:true}).click();
    assert.equal(await page.getByRole('button',{name:'实时模式',exact:true}).getAttribute('aria-pressed'),'true');
    report.checks.push('从已确认稿进入历史检索，切换实时模式仍可达');
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.network,[]);
    report.checks.push('直接打开本地 HTML，零外部请求与页面异常');
    report.passed=true;
  } catch(error) {
    report.failure=error.message;
    throw error;
  } finally {
    if(browser) await browser.close();
    fs.writeFileSync(path.join(resultDir,'acceptance.json'),JSON.stringify(report,null,2)+'\n');
    const actual=fs.realpathSync.native(profile);
    assert.equal(path.dirname(actual),fs.realpathSync.native(parent));
    assert.ok(path.basename(actual).startsWith('record-search-preview-'));
    fs.rmSync(actual,{recursive:true,force:true,maxRetries:4,retryDelay:150});
  }
  console.log(JSON.stringify(report));
}
verify().catch(error=>{console.error(error);process.exitCode=1;});
