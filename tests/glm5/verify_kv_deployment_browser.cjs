const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.AI2050_PLAYWRIGHT_MODULE ||
  'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');

const url = 'http://127.0.0.1:5048';
const run = 'KV-RECEIPT-20261117-v1';
const snapshot = '1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce';
const out = path.resolve('tests/glm5/result/deployment_20261117/browser.json');

async function openReport(page, runId, title, region) {
  await page.addInitScript(() => { window.__AI2050__ = { mode: 'local' }; });
  await page.goto(url + '/workbench?lens=spatial&view=graph&gview=internal&run=' + runId + '&snapshot=' + snapshot);
  await page.getByRole('button', { name: '成果库', exact: true }).click();
  const library = page.getByRole('region', { name: '成果库', exact: true });
  await library.getByLabel('成果研究', { exact: true }).selectOption(runId);
  await library.locator('.aw-library-list > button').filter({ hasText: title }).click();
  await library.getByText('完整报告与原始证据', { exact: true }).click();
  const result = library.getByRole('region', { name: region, exact: true });
  await result.waitFor();
  return result;
}

async function test() {
  const browser = await chromium.launch({
    executablePath: process.env.AI2050_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disk-cache-size=262144', '--disable-gpu-shader-disk-cache']
  });
  const checks = [], errors = [], modelRequests = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 980 } });
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (request.method() === 'POST' && /\/api\/live\/(load|run)/.test(request.url())) modelRequests.push(request.url());
    });
    const result = await openReport(page, run, '固定实发 State 的等长插入与真实改码', 'Phase 2955 固定 State 的真实改码');
    assert.match(await result.innerText(), /828 个合格插入视图/);
    assert.match(await result.innerText(), /32\/576/);
    assert.match(await result.innerText(), /17\/17/);
    await result.getByText('阻断、恢复与反例', { exact: true }).click();
    assert.match(await result.innerText(), /sham 与干净分支对齐失败/);
    await result.getByText('准入、预测与证据来源', { exact: true }).click();
    assert.match(await result.innerText(), /反角色 State 行为门存在未过格/);
    checks.push('Phase 2955 public report, recovery, failures and evidence open');
    const old = await openReport(page, 'KV-LENGTH-20261116-v1', '实发 State 的距离、寻址与写入', 'Phase 2952 实发 State 证据');
    assert.match(await old.innerText(), /0\.863 logit/);
    checks.push('Phase 2952 remains accessible in the same build');
    const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true });
    mobile.on('pageerror', error => errors.push(error.message));
    await openReport(mobile, run, '固定实发 State 的等长插入与真实改码', 'Phase 2955 固定 State 的真实改码');
    assert.equal(await mobile.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    checks.push('Phase 2955 mobile without page overflow');
    assert.deepEqual(errors, []);
    assert.deepEqual(modelRequests, []);
  } finally {
    await browser.close();
    fs.writeFileSync(out, JSON.stringify({ checks, errors, modelRequests }, null, 2));
  }
}

test().catch(error => { console.error(error); process.exitCode = 1; });
