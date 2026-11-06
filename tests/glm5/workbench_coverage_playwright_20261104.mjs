import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.AI2050_PLAYWRIGHT || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const out = path.resolve('tests/glm5/result/workbench_overview_ratios_20261105/coverage');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: process.env.AI2050_CHROMIUM || 'C:/Users/Admin/AppData/Local/ms-playwright/chromium-1247/chrome-win64/chrome.exe' });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
page.setDefaultTimeout(25000);
const model = 'qwen3-4b|cfg-8ba006f74fec|bf16';
const url = 'http://127.0.0.1:5173/workbench?' + new URLSearchParams({ lens: 'route', rtab: 'lang', mdl: model, ds: 'C-Eval val', snapshot: '6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad' });
const errors = [], requests = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('request', request => requests.push({ path: new URL(request.url()).pathname, method: request.method() }));
const report = { kind: 'engineering', passed: false, checks: [] };
const metric = name => page.getByRole('button', { name, exact: true });
const summary = title => page.locator('summary').filter({ hasText: title }).first();

try {
  const catalog = JSON.parse(await fs.readFile('shared/dist_protocol/lang_catalog.json', 'utf8'));
  assert.deepEqual(catalog.sections.flatMap(section => section.domains).sort(),
    catalog.domains.map(domain => domain.id).sort());
  const items = catalog.domains.flatMap(domain => domain.groups.flatMap(group => group.items));
  assert.equal(new Set(items.map(item => item.id)).size, items.length);
  assert(items.every(item => item.name && item.example));
  const treeResponse = await page.request.get('http://127.0.0.1:5001/api/lang/tree');
  assert(treeResponse.ok());
  const tree = await treeResponse.json();
  assert.deepEqual(items.map(item => item.measurement).filter(Boolean).sort(), tree.items.map(item => item.id).sort());
  const lv = (item, key) => item.status?.by_model?.[key] ?? 0;
  const groups = [
    ['已有数据', level => level >= 1], ['留出复现', level => level >= 3], ['待补数据', level => level === 0],
  ];
  async function checkCounts(key) {
    for (const [name, matches] of groups) {
      const count = tree.items.filter(item => matches(lv(item, key))).length;
      await metric(name).getByText(`${count} / ${tree.items.length} 项`, { exact: true }).waitFor();
    }
  }
  await page.goto(url);
  const heading = page.getByRole('heading', { name: '外部语言图谱', exact: true });
  await heading.waitFor();
  assert.equal(await page.locator('.v8-catalog-card').count(), 3);
  assert.deepEqual(await page.locator('.v8-catalog-card strong').allTextContents(), ['知识网络', '推理与功能', '语法体系']);
  assert.equal(await page.locator('.v8-catalog-card[aria-expanded=true]').count(), 0);
  assert.equal(await page.locator('.v8-coverage-number').count(), 0);
  await page.screenshot({ path: path.join(out, 'catalog-desktop.png') });
  for (const group of catalog.sections) {
    await metric(group.name).click();
    const section = page.getByRole('region', { name: `${group.name}条目` });
    for (const id of group.domains) {
      const domain = catalog.domains.find(entry => entry.id === id);
      if (group.domains.length > 1) await section.getByRole('button', { name: domain.name, exact: true }).click();
      assert.deepEqual(await section.locator('.v8-catalog-item > summary').allTextContents(), domain.groups.flatMap(group => group.items.map(item => item.name)));
    }
  }
  await metric('推理与功能').click();
  await metric('语言应用任务').click();
  await summary(/^翻译$/).click();
  await page.getByText(/表达原句事件、角色和语气/).waitFor();
  await metric('知识网络').click();
  await summary(/^上下位关系$/).click();
  await page.getByText(/表达类别归属/).waitFor();
  await page.screenshot({ path: path.join(out, 'catalog-detail.png') });
  await metric('实验覆盖').click();
  await checkCounts(model);
  assert(!await page.getByText(/表达类别归属/).isVisible());
  await metric('语言目录').click();
  await page.getByText(/表达类别归属/).waitFor();
  await page.getByRole('button', { name: '查看对应实验条目' }).click();
  await page.waitForURL(current => current.searchParams.get('gview') === 'language' && current.searchParams.get('item') === 'KN.ISA');
  report.catalog = { sections: catalog.sections.length, domains: catalog.domains.length, groups: catalog.domains.reduce((sum, domain) => sum + domain.groups.length, 0), items: items.length, measurement_links: items.filter(item => item.measurement).length };
  report.checks.push('three catalog sections, applications under reasoning and functions, view switching and legacy item links');
  requests.length = 0;
  await page.goto(url);
  await metric('实验覆盖').click();
  await checkCounts(model);
  assert.equal(await page.locator('.v8-coverage details[open]').count(), 0);
  assert.equal(await page.locator('.v8-coverage table').count(), 0);
  for (const path of ['/api/lang/atlas/datasets', '/api/lang/bench/overview', '/api/platform/snapshot/scoreboard', '/api/neurons/tower']) {
    assert(!requests.some(request => request.path === path), `${path} should wait for disclosure`);
  }
  report.metrics = await page.locator('.v8-coverage-number').allTextContents();
  await page.screenshot({ path: path.join(out, 'coverage-desktop.png') });
  report.checks.push('three real coverage counts, expensive detail deferred');

  const other = tree.models.find(key => key !== model);
  await page.getByLabel('覆盖模型').selectOption(other);
  await page.waitForURL(current => current.searchParams.get('mdl') === other);
  await checkCounts(other);
  await page.goBack();
  await checkCounts(model);
  report.checks.push('model change and browser history preserve coverage identity');

  for (const [name, matches] of groups) {
    await metric(name).click();
    const region = page.getByRole('region', { name: `${name}条目`, exact: true });
    await region.waitFor();
    assert.deepEqual(await region.getByRole('button').allTextContents(), tree.items.filter(item => matches(lv(item, model))).map(item => item.name));
    assert((await region.locator('.v8-chip').allTextContents()).every(label => label.includes('未研究') === (name === '待补数据')));
  }
  await metric('待补数据').click();
  await summary(/^数据集实测/).click();
  const dataset = page.getByLabel('评测数据集');
  assert.equal(await dataset.inputValue(), 'C-Eval val');
  await page.getByRole('heading', { name: /^图谱测试题/ }).waitFor();
  await dataset.selectOption('TPL');
  await page.waitForURL(current => current.searchParams.get('ds') === 'TPL');
  await checkCounts(model);
  await dataset.selectOption('C-Eval val');
  await summary(/^数据集实测/).click();
  await summary(/^跨模型对照$/).click();
  await page.locator('.v8-matrix').waitFor();
  assert.equal(await page.locator('.v8-matrix tbody tr').count(), tree.items.length);
  await summary(/^跨模型对照$/).click();
  await summary(/^更多研究细节$/).click();
  const score = page.waitForResponse(response => new URL(response.url()).pathname === '/api/platform/snapshot/scoreboard' && response.ok());
  await summary(/^证据门链与预测记分板$/).click();
  await score;
  report.checks.push('item details, dataset isolation, matrix and scoreboard retained');

  await page.goto(url);
  await metric('实验覆盖').click();
  const item = tree.items.find(entry => lv(entry, model) >= 3);
  await metric('留出复现').click();
  await page.getByRole('region', { name: '留出复现条目' }).getByRole('button', { name: item.name, exact: true }).click();
  await page.waitForURL(current => current.searchParams.get('gview') === 'language' && current.searchParams.get('item') === item.id);
  assert.equal(new URL(page.url()).searchParams.get('mdl'), model);
  report.checks.push('real item opens its existing language graph');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(url);
  await heading.waitFor();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.screenshot({ path: path.join(out, 'catalog-mobile.png') });
  await metric('知识网络').click();
  const branch = await page.getByRole('region', { name: '知识网络条目' }).boundingBox();
  assert(branch.y < (await metric('推理与功能').boundingBox()).y);
  await metric('实验覆盖').click();
  await checkCounts(model);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.screenshot({ path: path.join(out, 'coverage-mobile.png') });
  await metric('已有数据').click();
  const detail = await page.locator('#coverage-items').boundingBox();
  const next = await metric('留出复现').boundingBox();
  assert(detail.y < next.y);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.screenshot({ path: path.join(out, 'coverage-mobile-detail.png') });
  report.checks.push('390px detail opens beside its own metric without horizontal overflow');
  assert(!requests.some(request => request.path === '/api/lang/bench/run' && request.method === 'POST'));
  assert.deepEqual(errors, []);
  report.passed = true;
} catch (error) {
  report.failure = String(error.stack || error);
  await page.screenshot({ path: path.join(out, 'coverage-failure.png') }).catch(() => {});
  throw error;
} finally {
  report.page_errors = errors;
  await fs.writeFile(path.join(out, 'playwright.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify(report));
