/* Verify the offline review page against its published historical examples. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL, fileURLToPath} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_CORE ||
  'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root = path.resolve(__dirname, '../..');
const preview = path.join(root, 'research/glm5/docs/previews/research-evidence-workbench.html');
const result = path.join(root, 'tests/glm5/result/research_evidence_workbench_preview');
const temp = path.join(root, 'tests/glm5_temp/research_evidence_workbench_preview');
const systemChrome = process.argv.includes('--system-chrome');

async function main() {
  fs.mkdirSync(result, {recursive:true});
  fs.mkdirSync(temp, {recursive:true});
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || (systemChrome ?
      'C:/Program Files/Google/Chrome/Application/chrome.exe' :
      path.join(require('node:os').homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe')),
    headless: true, downloadsPath: temp,
    env: {...process.env, TEMP:temp, TMP:temp, TMPDIR:temp},
    args: ['--disable-background-networking', '--disable-component-update']
  });
  const errors = [], remote = [], links = new Set();
  try {
    const ctx = await browser.newContext({viewport:{width:1440,height:960}, acceptDownloads:true});
    await ctx.route(/^https?:/, route=>route.abort('internetdisconnected'));
    const page = await ctx.newPage();
    page.setDefaultTimeout(4000);
    page.on('pageerror', e=>errors.push(e.message));
    page.on('request', req=>{if (/^https?:/.test(req.url())) remote.push(req.url());});
    const click = (act, value) => page.locator('[data-act="'+act+'"]'+
      (value === undefined ? '' : '[data-value="'+value+'"]')).first().click();
    const nav = key => page.locator('[data-view="'+key+'"]').click();
    const close = () => page.locator('#close').click();
    async function checkLinks() {
      for (const url of await page.locator('a[href]').evaluateAll(nodes=>nodes.map(n=>n.href))) {
        assert.equal(new URL(url).protocol, 'file:');
        assert.ok(fs.existsSync(fileURLToPath(url)), url);
        links.add(url);
      }
    }
    await page.goto(pathToFileURL(preview).href);
    const totals = () => page.locator('.stat small').allTextContents();
    assert.deepEqual(await totals(), ['377 / 1,024 题','1,018 / 1,024 题']);
    if(!systemChrome) await page.screenshot({path:path.join(result,'desktop.png')});
    await click('rule','0');assert.deepEqual(await totals(),['1,015 / 1,024 题','1,024 / 1,024 题']);
    await click('rule','1');assert.deepEqual(await totals(),['1,022 / 1,024 题','1,007 / 1,024 题']);
    await click('rule','2');
    await page.locator('[data-pick="2983"]').uncheck();
    assert.deepEqual(await totals(), ['206 / 512 题','508 / 512 题']);
    await click('browse','div');
    assert.match(await page.locator('#list-table').innerText(),/303 道/);
    await click('next');assert.match(await page.locator('#list-table').innerText(),/第 2 \/ 38 页/);
    await click('prev');
    await page.locator('#case-search').fill('no-case-found');assert.match(await page.locator('#list-table').innerText(),/没有题目/);
    await page.locator('#case-search').fill('');await click('case');
    assert.match(await page.locator('#dialog-body').innerText(),/表外注意力质量/);
    await checkLinks();await click('back-list');assert.match(await page.locator('#list-table').innerText(),/303 道/);
    await close();
    await page.locator('[data-pick="2997"]').check();
    await click('study','2997');assert.match(await page.locator('#dialog-body').innerText(),/301 次/);
    assert.match(await page.locator('#dialog-body').innerText(),/12.7702/);await checkLinks();
    await click('openpart','U17216');assert.match(await page.locator('main').innerText(),/17216,1843/);
    await nav('compare');
    await click('export');
    await click('select-json');
    const selected=await page.locator('#export-json').evaluate(t=>t.value.slice(t.selectionStart,t.selectionEnd));
    const report=JSON.parse(selected);
    if(!systemChrome){
      const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#download-set').click()]);
      assert.deepEqual(JSON.parse(fs.readFileSync(await download.path(),'utf8')),report);
      await download.delete();
    }
    assert.equal(report.total.n,512);assert.equal(report.included[0].phase,2990);
    assert.equal(report.alongside[0].phase,2997);
    await close();
    await page.locator('[data-pick="2990"]').uncheck();assert.match(await page.locator('main').innerText(),/选择 Phase 2983/);
    await nav('catalog');
    await click('tree');
    await page.locator('dialog summary').filter({hasText:'知识网络'}).click();
    await page.locator('dialog summary').filter({hasText:'实体与类别'}).click();
    await click('leaf','SEM.IDENTITY');
    assert.match(await page.locator('#dialog-body').innerText(),/设计参考答案/);
    await close();
    await page.locator('#lang').selectOption('中文');
    assert.match(await page.locator('#catalog-list').innerText(),/1 项测试/);
    await click('study','322');assert.match(await page.locator('#dialog-body').innerText(),/31 \/ 32/);await checkLinks();await close();
    await page.locator('#model').selectOption('Qwen3-14B');assert.match(await page.locator('#catalog-list').innerText(),/没有已收录/);
    await click('clear');await page.locator('#search').fill('U17216');
    assert.match(await page.locator('#catalog-list').innerText(),/1 项测试/);
    await page.reload();assert.equal(await page.locator('#search').inputValue(),'U17216');
    await page.locator('#search').fill('');await page.locator('#area').selectOption('语法');
    assert.match(await page.locator('#catalog-list').innerText(),/没有已收录/);await click('clear');
    for(const p of [2983,2990]){await click('study',p);await checkLinks();
      await click('study-rows',p+',2');await click('case');await checkLinks();await close();}
    await click('study','3008');assert.match(await page.locator('#dialog-body').innerText(),/57.6%/);
    await checkLinks();await click('color-cell','red/red');assert.match(await page.locator('#dialog-body').innerText(),/7 \/ 36 正确/);
    assert.equal(await page.locator('dialog table tr').count(),37);
    await page.locator('#color-arm').selectOption('restore');assert.equal(await page.locator('dialog table tr').count(),37);
    await page.keyboard.press('Escape');assert.equal(await page.locator('dialog').isVisible(),false);
    await nav('storage');await checkLinks();await nav('parts');await click('part','B26/H36');await click('jump-h36');
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    if(!systemChrome) await page.screenshot({path:path.join(result,'mobile.png'),fullPage:true});
    await click('browse','div');await click('case');assert.equal(await page.locator('dialog').isVisible(),true);
    assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
    const evidence={scope:'offline HTML only',browser:browser.version(),fileDownloadVerified:!systemChrome,
      jsonTextSelectionVerified:true,caseCount:3072,colorArmRows:1296,
      checkedLocalLinks:links.size,networkRequests:remote,scriptErrors:errors,
      checked:['three rules','pooled and single batch','case disagreement and source files',
      'pagination and empty search','mixed metrics and JSON export','multilingual filters','direction tree and planned questions',
      'refresh state','color matrix and four-arm source data','component and group limits',
      'keyboard Escape','desktop and narrow viewport']};
    fs.writeFileSync(path.join(result,systemChrome?'chrome-check.json':'browser-check.json'),JSON.stringify(evidence,null,2)+'\n');
    console.log(JSON.stringify(evidence));
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
