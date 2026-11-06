/* Visual verification against the local client and current real research archive. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.AI2050_PLAYWRIGHT || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const out = path.resolve('tests/glm5/result/workbench_overview_ratios_20261105');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: process.env.AI2050_CHROMIUM || 'C:/Users/Admin/AppData/Local/ms-playwright/chromium-1247/chrome-win64/chrome.exe' });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
page.setDefaultTimeout(25000);
const errors = [], requests = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('request', request => requests.push(new URL(request.url()).pathname));
const snapshot = '6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad';
const run = 'P2794-ATOMIC-ARBITRATION-v4';
const base = 'http://127.0.0.1:5173/workbench?lens=route';
const model = 'qwen3-4b|cfg-8ba006f74fec|bf16';
const url = base + '&' + new URLSearchParams({ snapshot, ds: 'C-Eval val', mdl: model });
const names = ['外部语言图谱', '内部模型图谱', '两图谱的关联与验证'];
const card = name => page.getByRole('button', { name, exact: true });
const metrics = page.locator('.v8-overview-metric');
const report = { kind: 'engineering', passed: false, checks: [] };

try {
  const load = async suffix => {
    const response = await page.request.get(`http://127.0.0.1:5001/api/${suffix}`);
    assert(response.ok(), suffix);
    return response.json();
  };
  const [tree, skeletons, catalog] = await Promise.all([load('lang/tree'), load('model/skeleton'),
    fs.readFile('shared/dist_protocol/lang_catalog.json', 'utf8').then(JSON.parse)]);
  const skeleton = skeletons.models.find(entry => entry.model === model && entry.model_snapshot === snapshot);
  assert(skeleton?.census_key);
  const checks = await load('neurons/explained?model_key=' + encodeURIComponent(skeleton.census_key));
  assert.equal(checks.model_key, skeleton.census_key);
  const items = catalog.domains.flatMap(domain => domain.groups.flatMap(group => group.items));
  const measured = tree.items.filter(item => item.status?.by_model?.[model] >= 1);
  const covered = items.filter(entry => measured.some(item => item.id === entry.measurement)).length;
  const tested = [...new Map(checks.results.map(row => [`${row.block}:${row.unit}`, row])).values()];
  const unitTotal = skeleton.n_layers * skeleton.d_ff;
  const label = (n, total, unit) => `${n.toLocaleString('zh-CN')} / ${total.toLocaleString('zh-CN')} ${unit} · ${(100 * n / total).toFixed(2)}%`;
  const languageLabel = label(covered, items.length, '项');
  const neuronLabel = label(tested.length, unitTotal, '个');
  await page.goto(base);
  await metrics.getByText(languageLabel, { exact: true }).waitFor();
  await metrics.getByText(neuronLabel, { exact: true }).waitFor();
  assert.deepEqual(await page.getByRole('navigation', { name: '进展视图' }).locator('button:not(.wb-help-btn)').allTextContents(), ['总览', '成果与证据']);
  for (const name of names) assert.equal(await card(name).getAttribute('aria-expanded'), 'false');
  assert.equal(await page.locator('#overview-detail').count(), 0);
  assert(!requests.includes('/api/traces/research-space'));
  assert(!requests.includes('/api/research-library/progress'));
  assert(!requests.includes('/api/research-library/overview'));
  report.metrics = await metrics.allTextContents();
  await page.screenshot({ path: path.join(out, 'overview-desktop.png') });
  report.checks.push('three cards, actual metrics, heavy archive deferred');

  await page.goto(url + '&rtab=lang');
  await page.waitForURL(current => current.searchParams.get('opane') === 'language' && !current.searchParams.has('rtab'));
  await page.getByRole('button', { name: '语言目录', exact: true }).waitFor();
  assert.equal(await card(names[0]).getAttribute('aria-expanded'), 'true');
  await card(names[0]).click();
  await page.goBack();
  await page.getByRole('button', { name: '语言目录', exact: true }).waitFor();
  await page.getByRole('button', { name: '实验覆盖', exact: true }).click();
  await card('已有数据').click();
  assert.deepEqual(await page.getByRole('region', { name: '已有数据条目', exact: true }).getByRole('button').allTextContents(), measured.map(item => item.name));
  report.checks.push('legacy URL and browser history open integrated catalog with actual measured items');

  await page.goto(url);
  await card(names[1]).click();
  for (const grade of new Set(tested.map(row => row.grade))) {
    await page.getByText(`${grade} ${tested.filter(row => row.grade === grade).length} 个`, { exact: true }).waitFor();
  }
  await page.getByText('查看已检验神经元', { exact: true }).click();
  const row = tested[0];
  await page.getByLabel('已检验神经元所在块').selectOption(String(row.block));
  const unit = page.getByRole('button', { name: `块 ${row.block} · 单元 ${row.unit}`, exact: true });
  await unit.waitFor();
  await page.screenshot({ path: path.join(out, 'overview-neurons.png') });
  await unit.click();
  await page.waitForURL(current => current.searchParams.get('view') === 'parameters' && current.searchParams.get('ptab') === 'neurons');
  const target = new URL(page.url()).searchParams;
  assert.equal(target.get('snapshot'), snapshot);
  assert.equal(target.get('mdl'), model);
  assert.equal(target.get('block'), String(row.block));
  assert.equal(target.get('unit'), String(row.unit));
  report.checks.push('neuron outcomes and exact model, block and unit navigation');

  await page.goto(url);
  await page.getByText('阶段任务与历史分片', { exact: true }).click();
  await page.getByLabel('分片来源').selectOption({ label: 'qwen3-4b · cfg-8ba006f74fec · bf16 · 历史配置' });
  assert((await page.getByRole('table').innerText()).includes('待核验 · 计划'));
  await page.getByText('阶段任务与历史分片', { exact: true }).click();
  assert.equal(await page.getByRole('table').count(), 0);
  await page.getByText('研究归档与原始来源', { exact: true }).click();
  await page.locator('.v8-library-stats').waitFor();
  await page.getByText('研究归档与原始来源', { exact: true }).click();
  report.checks.push('legacy shards and archive visible on demand');

  for (const name of names) {
    await page.goto(url + '&run=' + run);
    await card(name).click();
    if (name === names[1]) {
      await page.getByText('图谱卡片与真实参数', { exact: true }).click();
      await page.getByText(/当前范围已登记 \d+ 张特征卡/).waitFor();
    }
    if (name === names[2]) await page.getByRole('list', { name: '跨图谱证据类型' }).waitFor();
    await page.getByText('相关工作包', { exact: true }).click();
    const link = page.locator('.v8-overview-worklist a').first();
    await link.waitFor();
    const href = new URL(await link.getAttribute('href'), url);
    assert.equal(href.searchParams.get('snapshot'), snapshot);
    assert.equal(href.searchParams.get('run'), run);
    assert.equal(href.searchParams.get('ds'), 'C-Eval val');
    assert((await link.locator('..').innerText()).includes('知识网络 · 推理与功能'));
    if (name === names[2]) await page.screenshot({ path: path.join(out, 'overview-evidence.png') });
    await link.click();
    if (name === names[1]) {
      await page.getByLabel('成果图研究工作包').waitFor();
      assert.equal(await page.getByLabel('成果图研究工作包').inputValue(), run);
    } else {
      await page.getByRole('heading', { name: name === names[0] ? '研究计划' : '实验与队列', exact: true }).waitFor();
      assert.equal(new URL(page.url()).searchParams.get('run'), run);
      if (name === names[2]) {
        const evidence = page.getByRole('region', { name: '原子序数仲裁与未见组合真实报告' });
        await evidence.waitFor();
        await evidence.getByRole('button', { name: /^反例（/ }).click();
        assert(await evidence.locator('summary').count() > 0);
      }
    }
  }
  report.checks.push('same real P2794 package linked from all three views');

  await page.goto(url.replace(snapshot, 'f'.repeat(64)));
  await metrics.getByText('— / — 个', { exact: true }).waitFor();
  await metrics.getByText(`— / ${items.length} 项`, { exact: true }).waitFor();
  await page.locator('.v8-overview-scope summary').click();
  await page.getByLabel('总览模型快照', { exact: true }).selectOption(snapshot);
  await page.waitForURL(current => current.searchParams.get('snapshot') === snapshot);
  await metrics.getByText(neuronLabel, { exact: true }).waitFor();
  report.checks.push('unknown snapshot never falls back, selection updates shared URL');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(url);
  await metrics.getByText(neuronLabel, { exact: true }).waitFor();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.screenshot({ path: path.join(out, 'overview-mobile.png') });
  await card(names[0]).click();
  await page.getByRole('button', { name: '语言目录', exact: true }).waitFor();
  const detail = await page.locator('#overview-detail').boundingBox();
  const next = await card(names[1]).boundingBox();
  assert(detail.y < next.y, 'on narrow screens the detail follows its own card');
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  report.checks.push('390px overview and expanded content fit the viewport');
  assert.deepEqual(errors, []);
  report.passed = true;
} finally {
  report.page_errors = errors;
  await fs.writeFile(path.join(out, 'playwright.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify(report));
