import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.AI2050_PLAYWRIGHT || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const out = path.resolve('tests/glm5/result/workbench_evidence_rules_20261105');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: process.env.AI2050_CHROMIUM || 'C:/Users/Admin/AppData/Local/ms-playwright/chromium-1247/chrome-win64/chrome.exe' });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
page.setDefaultTimeout(25000);

const url = 'http://127.0.0.1:5173/workbench?' + new URLSearchParams({
  lens: 'route', rtab: 'rules', mdl: 'qwen3-4b|cfg-8ba006f74fec|bf16', ds: 'C-Eval val',
  snapshot: '6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad', run: 'P2794-ATOMIC-ARBITRATION-v4',
});
const requests = [];
const errors = [];
const report = { kind: 'engineering', passed: false, checks: [] };
page.on('request', request => requests.push(new URL(request.url()).pathname));
page.on('pageerror', error => errors.push(String(error)));
const summary = title => page.locator('summary').filter({ hasText: title });
const read = endpoint => page.waitForResponse(response => new URL(response.url()).pathname === endpoint && response.ok());
const negative = page.getByRole('button', { name: /^负面规则/ });
const positive = page.getByRole('button', { name: /^有效规则/ });
const check = page.getByRole('button', { name: '查重', exact: true });
const query = page.getByRole('textbox', { name: '实验查重', exact: true });
const results = page.locator('.v8-rules-results');
const ruleHead = id => page.locator('#rule-head-' + id);
const archiveTitle = '原始报告与版本记录';

try {
  const overview = read('/api/rules/overview');
  await page.goto(url);
  const data = await (await overview).json();
  await page.getByRole('heading', { name: '成果与证据', exact: true }).waitFor();
  await query.waitFor();
  const totals = data.counts.by_status;
  for (const [head, n] of [[negative, totals.refuted], [positive, totals.confirmed]]) {
    await head.getByText(`${n} 条`, { exact: true }).waitFor();
    assert.equal(await head.getAttribute('aria-expanded'), 'false');
  }
  assert.equal(await page.locator('.v8-rules-head').count(), 0);
  assert.equal(await page.locator('main table').count(), 0);
  assert.equal(await page.locator('main details[open]').count(), 0);
  const detailedPaths = ['/api/research-library/artifacts', '/api/research-library/fragments', '/api/research-library/graph', '/api/research-library/overview'];
  assert(detailedPaths.every(endpoint => !requests.includes(endpoint)));
  assert(!requests.includes('/api/lang/tree'));
  await page.screenshot({ path: path.join(out, 'evidence-desktop.png') });
  report.metrics = totals;
  report.checks.push('first load shows search and two collapsed category headers from the real ledger');

  await negative.click();
  await ruleHead('R-MLP-18').click();
  const detail = page.locator('#rule-detail-R-MLP-18');
  for (const name of ['适用范围', '实测结果', '反例与判读', '复测条件']) await detail.getByRole('heading', { name }).waitFor();
  await detail.getByText('-0.74', { exact: true }).waitFor();
  await detail.getByRole('button', { name: /^R-MLP-01/ }).click();
  assert.equal(await positive.getAttribute('aria-expanded'), 'true');
  assert.equal(await negative.getAttribute('aria-expanded'), 'false');
  await page.locator('#rule-detail-R-MLP-01').getByText('65/97', { exact: true }).waitFor();
  await positive.click();
  assert.equal(await page.locator('.v8-rules-head').count(), 0);
  await positive.click();
  await ruleHead('R-MLP-01').waitFor();
  await negative.click();
  const auditedId = 'R-MLP-P2794-H2';
  await ruleHead(auditedId).click();
  const audited = page.locator('#rule-detail-' + auditedId);
  await audited.getByText('各 128/128', { exact: true }).waitFor();
  await audited.locator('summary').filter({ hasText: '原始证据来源' }).click();
  const reportLink = new URL(await audited.getByRole('link', { name: '打开研究报告' }).getAttribute('href'), url);
  assert.equal(reportLink.searchParams.get('run'), 'P2794-ATOMIC-ARBITRATION-v4');
  report.checks.push('both headers toggle, rule details show real values, related rules and source link');
  await page.screenshot({ path: path.join(out, 'evidence-rule-detail.png') });

  await audited.getByRole('button', { name: '带入查重' }).click();
  await check.click();
  await results.getByText('同范围已测', { exact: true }).waitFor();
  await results.getByText('查看核对项').click();
  for (const label of ['模型快照', '数据与划分', '实验口径']) await results.getByText(`${label} · 一致`, { exact: true }).waitFor();
  const scope = data.rules.find(rule => rule.id === auditedId).scope;
  await page.screenshot({ path: path.join(out, 'evidence-check.png') });
  for (const [label, original, changed] of [
    ['模型快照', scope.model_snapshot, 'f'.repeat(64)],
    ['数据与划分', scope.data_sha256, 'a'.repeat(64)],
    ['实验口径', `${scope.protocol_id}@${scope.design_sha256}`, `other-step@${scope.design_sha256}`],
  ]) {
    const field = page.getByRole('textbox', { name: label, exact: true });
    await field.fill(changed);
    await check.click();
    await results.getByText('部分重合', { exact: true }).waitFor();
    assert.equal(await results.getByText('同范围已测', { exact: true }).count(), 0);
    await results.getByText('查看核对项').click();
    await results.getByText(`${label} · 不同`, { exact: true }).waitFor();
    await field.fill(original);
  }
  await query.fill('R-MLP-18');
  await check.click();
  await results.getByText('证据不足', { exact: true }).waitFor();
  await results.getByText('查看核对项').click();
  await results.getByText('模型快照 · 账本缺少可核对身份', { exact: true }).waitFor();
  await audited.getByRole('button', { name: '带入查重' }).click();
  await page.waitForFunction(id => document.getElementById('rule-query').value === id, auditedId);
  await query.fill('归因排序');
  await query.press('Enter');
  await results.getByRole('button', { name: /^R-MLP-18/ }).click();
  await page.locator('#rule-detail-R-MLP-18').waitFor();
  report.checks.push('scope checks actual P2794 hashes, separate model/data/protocol mismatches, legacy unknown and text search');

  const archive = read(detailedPaths[3]);
  await summary(archiveTitle).click();
  await archive;
  const artifactList = read(detailedPaths[0]);
  await summary('研究产物与来源').click();
  const artifacts = await (await artifactList).json();
  assert(artifacts.artifacts.length > 0);
  const { sha } = artifacts.artifacts[0];
  await page.getByRole('button', { name: sha.slice(0, 12), exact: true }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByText(`SHA256 ${sha}`, { exact: true }).waitFor();
  await dialog.locator('pre').filter({ hasText: '{' }).waitFor();
  await page.keyboard.press('Escape');
  await dialog.waitFor({ state: 'detached' });
  await page.getByRole('button', { name: sha.slice(0, 12), exact: true }).waitFor();
  if (artifacts.total > 50) {
    const nextPage = current => current.pathname === detailedPaths[0] && current.searchParams.get('offset') === '50';
    await page.route(nextPage, route => route.fulfill({ status: 503, headers: { 'access-control-allow-origin': '*' }, json: { detail: '该页暂不可用' } }));
    await page.getByRole('button', { name: '较早版本' }).click();
    await page.getByText('该页暂不可用', { exact: true }).waitFor();
    await page.getByRole('button', { name: '较新版本' }).click();
    await page.getByRole('button', { name: sha.slice(0, 12), exact: true }).waitFor();
    await page.unroute(nextPage);
    report.checks.push('failed archive page can return to the last available page');
  }
  await summary('研究产物与来源').click();
  report.checks.push('archive loads on demand and opens real artifact by hash');

  await summary('图谱证据').click();
  assert(!requests.includes(detailedPaths[1]) && !requests.includes(detailedPaths[2]));
  const fragments = read(detailedPaths[1]);
  await summary('特征卡与反例').click();
  assert((await (await fragments).json()).features.length > 0);
  await page.getByRole('button', { name: '查看原始证据' }).first().waitFor();
  await summary('特征卡与反例').click();
  const graph = read(detailedPaths[2]);
  await summary('特征与关系的版本记录').click();
  assert((await (await graph).json()).objects.length > 0);
  await page.getByRole('columnheader', { name: '证据声明' }).waitFor();
  await summary('图谱证据').click();
  report.checks.push('feature cards and graph versions load independently inside the archive');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(url);
  await negative.getByText(`${totals.refuted} 条`, { exact: true }).waitFor();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.screenshot({ path: path.join(out, 'evidence-mobile.png') });
  await negative.click();
  await ruleHead('R-MLP-18').click();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await summary(archiveTitle).click();
  await summary('研究产物与来源').click();
  await page.getByRole('button', { name: sha.slice(0, 12), exact: true }).waitFor();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  report.checks.push('390px default and archive layouts fit viewport');
  assert.deepEqual(errors, []);
  report.passed = true;
} catch (error) {
  report.failure = String(error.stack || error);
  await page.screenshot({ path: path.join(out, 'failure.png') }).catch(() => {});
  throw error;
} finally {
  report.page_errors = errors;
  await fs.writeFile(path.join(out, 'playwright.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify(report));
