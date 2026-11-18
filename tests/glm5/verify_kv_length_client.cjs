const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');

const out = path.resolve('tests/glm5/result/kv_length_client_20261117_v1');
const temp = path.resolve('tests/glm5_temp/kv_length_client_20261117_v1');
const run = 'KV-LENGTH-20261116-v1';
const snapshot = '1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce';
const key = `${run}|${snapshot}`;
const url = `http://127.0.0.1:5041/workbench?lens=spatial&view=graph&gview=internal&run=${run}&snapshot=${snapshot}`;

async function report(page) {
  await page.addInitScript(() => { window.__AI2050__ = { mode: 'local' }; });
  await page.goto(url);
  await page.getByRole('button', { name: '成果库', exact: true }).click();
  const library = page.getByRole('region', { name: '成果库', exact: true });
  await library.getByLabel('成果研究', { exact: true }).selectOption(run);
  await library.locator('.aw-library-list > button').filter({ hasText: '实发 State 的距离、寻址与写入' }).click();
  await library.getByText('完整报告与原始证据', { exact: true }).click();
  const result = library.getByRole('region', { name: 'Phase 2952 实发 State 证据', exact: true });
  assert.match(await result.innerText(), /0\.863 logit/);
  assert.match(await result.innerText(), /342.*38[\s\S]*336.*44/);
  return { library, result };
}

(async () => {
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true, args: ['--no-sandbox'], env: { ...process.env, TEMP: temp, TMP: temp } });
  const seen = { checks: [], errors: [], modelRequests: [] };
  try {
    const desktop = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    desktop.on('pageerror', e => seen.errors.push(e.message));
    desktop.on('request', r => { if (r.method() === 'POST' && /\/api\/live\/(load|run)/.test(r.url())) seen.modelRequests.push(r.url()); });
    const index = await (await desktop.request.get(`http://127.0.0.1:5041/api/traces/research-space?run_id=${run}`)).json();
    assert.equal(index.studies.filter(x => x.study_key === key).length, 1);
    seen.checks.push('production research-space index resolves the real study');
    const { library, result } = await report(desktop);
    seen.checks.push('catalog renders checked held-out State and prompt-fact counts');
    await result.getByRole('button', { name: '阻断与恢复', exact: true }).click();
    assert.match(await result.innerText(), /182 条实际答码/);
    assert.match(await result.innerText(), /码本头阻断[\s\S]*未过门/);
    await result.getByText('等幅随机对照的实际剂量', { exact: true }).click();
    assert.match(await result.innerText(), /3\.4%/);
    seen.checks.push('interventions, full restoration, and dose mismatch are visible');
    await result.getByRole('button', { name: '反例与来源', exact: true }).click();
    await result.getByText(/长段改为跟随原事实/).click();
    assert.match(await result.innerText(), /独立物理节点复算待完成/);
    seen.checks.push('actual contrary pair can be opened');
    await result.scrollIntoViewIfNeeded();
    await desktop.screenshot({ path: path.join(out, 'client-desktop.png') });
    await library.getByRole('button', { name: '定位卡与关联', exact: true }).click();
    await library.locator('.aw-library-list > button').filter({ hasText: 'L25H18 · State 的 Q/K 寻址' }).click();
    await library.getByRole('button', { name: '定位到当前空间' }).click();
    await desktop.locator('.aw-big').getByText('第 26 层', { exact: true }).waitFor();
    seen.checks.push('snapshot-bound L25H18 feature links to current space');
    const small = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, deviceScaleFactor: 1 });
    small.on('pageerror', e => seen.errors.push(e.message));
    const mobile = await report(small);
    assert.equal(await small.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await mobile.result.scrollIntoViewIfNeeded();
    await small.screenshot({ path: path.join(out, 'client-mobile.png') });
    seen.checks.push('mobile report renders without document overflow');
    assert.deepEqual(seen.errors, []);
    assert.deepEqual(seen.modelRequests, []);
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(out, 'client-browser-verification.json'), JSON.stringify(seen, null, 2));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
