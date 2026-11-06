import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { once } from 'node:events';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.AI2050_PLAYWRIGHT || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const out = path.resolve('tests/glm5/result/workbench_live_research');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: process.env.AI2050_CHROMIUM ||
  'C:/Users/Admin/AppData/Local/ms-playwright/chromium-1247/chrome-win64/chrome.exe' });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.setDefaultTimeout(20000);
const report = { passed: false, checks: [], page_errors: [] };
const requests = [];
page.on('request', request => requests.push(new URL(request.url()).pathname));
page.on('pageerror', error => report.page_errors.push(String(error)));
const root = process.env.AI2050_WORKBENCH_TEST_ORIGIN || 'http://127.0.0.1:5173';
const url = (station, extras = {}) => root + '/workbench?' + new URLSearchParams({ lens: 'process', station, ...extras });
const tab = name => page.getByRole('navigation', { name: '研发工位' }).getByRole('button', { name, exact: true });
const main = page.getByRole('main');
const fold = (parent, name) => parent.locator('summary').filter({ hasText: name });
const exactButton = name => main.getByRole('button', { name, exact: true });
const run = 'P2794-ATOMIC-ARBITRATION-v4';
const snapshot = '6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad';
const noArtifacts = () => assert(requests.every(endpoint => !endpoint.startsWith('/api/research-library/artifacts/')));
const screenshot = async name => {
  await main.evaluate(element => element.scrollTo(0, 0));
  await page.screenshot({ path: path.join(out, name + '.png') });
};
let fixture;
let desktop;
let readNode;
let fixtureErrors = '';
const fixtureLines = [];
async function fixtureLine(prefix) {
  for (let i = 0; i < 300; i++) {
    const match = fixtureLines.findIndex(line => line.startsWith(prefix));
    if (match >= 0) return fixtureLines.splice(match, 1)[0].slice(prefix.length);
    if (fixture.exitCode !== null) throw new Error(fixtureErrors || `Node fixture exited ${fixture.exitCode}`);
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for isolated fixture (${prefix})`);
}

try {
  await page.goto(root + '/workbench?lens=process');
  const focus = main.getByRole('region', { name: '研究进度' });
  await focus.getByRole('heading').waitFor();
  await tab('当前研发').waitFor();
  assert.equal(await focus.count(), 1);
  const censusProgress = focus.getByRole('progressbar', { name: /qwen3-4b · 神经元普查已完成分片/i });
  await censusProgress.waitFor();
  const jobs = await page.request.get('http://127.0.0.1:5001/api/census/jobs').then(response => response.json());
  const pilot = jobs.jobs.find(job => job.job_id === 'CJ-qwen3-4b-pilot');
  assert.equal(await censusProgress.getAttribute('value'), String(pilot.counts.done ?? 0));
  assert.equal(await censusProgress.getAttribute('max'), String(pilot.n_shards));
  await main.getByText(/开放作业 · qwen3-4b · 神经元普查/i).waitFor();
  await exactButton('下载参与工具').waitFor();
  assert.equal(await main.locator('details[open]').count(), 0);
  assert.equal(await fold(main, '研究历史').count(), 1);
  assert.equal(await main.getByText(/在短链确认和长链留出上/).count(), 0);
  assert.equal(await main.locator('pre').count(), 0);
  noArtifacts();
  await screenshot('01-plan');
  await fold(main, '研究历史').click();
  await main.locator('.v8-rnd-list summary').first().waitFor();
  const download = page.waitForEvent('download');
  await exactButton('下载参与工具').click();
  assert.equal((await download).suggestedFilename(), 'ai2050-node.zip');
  report.checks.push('default Current Research shows real public census totals and open work, with real web kit download');

  requests.length = 0;
  await page.goto(url('plan', { run, snapshot }));
  const archived = fold(main, '研究历史 · 原子序数来源选择');
  await archived.waitFor();
  await censusProgress.waitFor();
  assert.equal(await focus.getByText('原子序数来源选择', { exact: true }).count(), 0);
  assert.equal(await main.locator('pre').count(), 0);
  assert.equal(await main.locator('details[open]').count(), 0);
  noArtifacts();
  await screenshot('02-archived-research');
  await archived.click();
  await fold(main, '查看一条真实样本').click();
  await main.getByText(/Scandium/).waitFor();
  const original = await main.getByRole('link', { name: '查看原文与模型响应' }).getAttribute('href');
  assert.match(original, /trace=[a-f0-9]{64}/);
  requests.length = 0;
  await exactButton('查看所用材料').click();
  assert.equal(new URL(page.url()).searchParams.get('mrun'), run);
  const materials = main.getByLabel('材料清单', { exact: true });
  const dataset = fold(materials, 'reference-assisted-atomic-arbitration');
  await dataset.waitFor();
  assert.equal(await materials.locator(':scope > details').count(), 1);
  await dataset.click();
  await main.getByText('704 条', { exact: false }).first().waitFor();
  await fold(materials, '关联主张与实验').click();
  await fold(materials, 'joint32_MLP').click();
  noArtifacts();
  await fold(materials, '原始主张证据').click();
  await main.locator('pre').first().waitFor();
  report.checks.push('legacy plan URL keeps the archived study in history while live work stays on the dashboard, with real sample and material evidence');

  await tab('材料与数据').click();
  await materials.locator('summary').first().waitFor();
  const search = main.getByRole('searchbox', { name: '搜索材料', exact: true });
  const type = main.getByRole('combobox', { name: '材料类型' });
  await search.fill('TG-KN-ATTR');
  await type.selectOption('template');
  await fold(materials, '模板').first().waitFor();
  assert.equal(await main.locator('table').count(), 0);
  await fold(materials, '模板').first().click();
  await materials.locator('.v6-cat-item').waitFor();
  await page.goto(url('catalog'));
  await fold(materials, '数据集').first().waitFor();
  await fold(materials, '模板').first().waitFor();
  assert(await materials.locator(':scope > details').count() <= 8);
  await screenshot('02-materials');
  report.checks.push('materials: one compact searchable list, type filter and template detail on demand');

  requests.length = 0;
  await page.goto(url('execution'));
  const queue = main.getByLabel('当前任务', { exact: true });
  await fold(queue, /异常任务|失败|租约过期/).first().waitFor();
  await fold(queue, '待审').first().waitFor();
  assert.match(await queue.locator('summary').first().innerText(), /异常任务|失败|租约过期/);
  assert(!requests.includes('/api/research-library/workpackages'));
  assert.equal(await main.locator('details[open]').count(), 0);
  await screenshot('03-queue');
  await fold(main, '历史任务').click();
  await main.getByText('已完成', { exact: true }).first().waitFor();
  await fold(main, '已完成').first().click();
  await main.getByText('节点', { exact: true }).first().waitFor();
  const resultLink = await main.getByRole('link', { name: '查看原始结果', exact: true }).first().getAttribute('href');
  assert.match(resultLink, /\/api\/results\/[a-f0-9]{64}$/);
  assert(!requests.some(endpoint => endpoint.startsWith('/api/results/')));
  await fold(main, '专业工具').click();
  await exactButton('模板流水线').waitFor();
  report.checks.push('execution: actual failure and audit entries precede the queue, history and tools are folded');

  requests.length = 0;
  await page.goto(url('execution', { run, snapshot }));
  const experiment = main.getByRole('region', { name: '链接指定的实验' });
  await experiment.getByText(run, { exact: true }).waitFor();
  noArtifacts();
  await fold(experiment, '研究报告与反例').click();
  await experiment.getByLabel('原子序数仲裁与未见组合真实报告').waitFor();
  await experiment.getByRole('button', { name: /^反例（/ }).click();
  assert(requests.some(endpoint => endpoint.startsWith('/api/research-library/artifacts/')));
  await exactButton('返回当前研发').click();
  await archived.waitFor();
  await page.goBack();
  await experiment.getByText(run, { exact: true }).waitFor();
  assert.equal(await main.locator('details[open]').count(), 0);
  await page.goto(url('audit'));
  await exactButton('返回实验与队列').waitFor();
  const migrated = new URL(page.url());
  assert.equal(migrated.searchParams.get('station'), 'pipeline');
  assert.equal(migrated.searchParams.get('pstep'), 'audit');
  report.checks.push('legacy execution and audit URLs, real archived report/counterexamples, browser Back preserves run');

  await page.setViewportSize({ width: 390, height: 844 });
  for (const station of ['plan', 'catalog', 'execution']) {
    await page.goto(url(station));
    await main.locator('summary').first().waitFor();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), station);
    if (station === 'plan') await screenshot('04-mobile-progress');
  }
  await fold(queue, '待审').first().waitFor();
  await screenshot('04-mobile');
  assert.deepEqual(report.page_errors, []);
  report.checks.push('390px layouts without horizontal overflow and no browser exceptions');

  fixture = spawn(process.env.AI2050_TEST_PYTHON || '.venv/Scripts/python.exe',
    ['-u', 'tests/glm5/current_research_browser_fixture.py'], { stdio: ['pipe', 'pipe', 'pipe'] });
  fixture.stderr.on('data', buffer => { fixtureErrors += String(buffer); });
  createInterface({ input: fixture.stdout }).on('line', line => fixtureLines.push(line));
  const service = JSON.parse(await fixtureLine('BROWSER_FIXTURE '));
  desktop = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  desktop.setDefaultTimeout(24000);
  desktop.on('pageerror', error => report.page_errors.push(String(error)));
  const calls = [];
  readNode = async route => fetch(`http://127.0.0.1:${service.port}${route}`,
    { headers: { 'X-Local-Token': service.token } }).then(response => response.json());
  await desktop.exposeFunction('callLocalNode', async (method, route, body) => {
    calls.push([method, route]);
    const response = await fetch(`http://127.0.0.1:${service.port}${route}`, {
      method, headers: { 'X-Local-Token': service.token, 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return { status: response.status, data: await response.json() };
  });
  await desktop.addInitScript(port => {
    window.ai2050Desktop = { node: { request: window.callLocalNode, info: async () => ({ state: 'attached', port }), onInfo: () => () => {} } };
  }, service.port);
  const desktopMain = desktop.getByRole('main');
  const desktopButton = name => desktopMain.getByRole('button', { name, exact: true });
  await desktop.route('**/api/census/jobs', async route => {
    const response = await fetch(service.center + '/api/census/jobs');
    await route.fulfill({ status: response.status, json: await response.json(), headers: { 'access-control-allow-origin': '*' } });
  });
  await desktop.goto(url('plan'));
  await desktopButton('注册或绑定节点').click();
  assert.equal(new URL(desktop.url()).searchParams.get('cstep'), 'account');
  const step = desktopMain.getByLabel('参与准备步骤');
  await step.getByRole('heading', { name: '账号与节点' }).waitFor();
  assert.equal(await step.getByRole('heading', { name: '设备', exact: true }).count(), 0);
  await desktopButton('匿名注册').click();
  await desktopMain.getByText(/已注册 NODE-/).waitFor();
  await desktopButton('返回当前研发').click();
  await desktopButton('一键参与').waitFor();
  const participation = desktopMain.getByRole('region', { name: '参与研发' });
  await participation.getByText(/可领取 · 神经元普查 · S0-G0/).waitFor();
  const sandboxProgress = desktopMain.getByRole('progressbar', { name: 'browser-test · 神经元普查已完成分片' });
  assert.equal(await sandboxProgress.getAttribute('value'), '0');
  assert.equal(await sandboxProgress.getAttribute('max'), '2');
  const openJob = await fetch(service.center + '/api/census/jobs/CJ-browser').then(response => response.json());
  assert.deepEqual(openJob.counts, { queued: 2 });
  await desktop.screenshot({ path: path.join(out, '05-desktop-ready.png') });
  await desktopButton('一键参与').click();
  await participation.locator('progress[value]').waitFor();
  const measured = await participation.getByRole('progressbar', { name: '本机任务进度' }).evaluate(el => ({ done: el.value, total: el.max }));
  assert(measured.done > 0 && measured.done <= measured.total);
  await desktopMain.getByText('1 运行中 · 1 待领取', { exact: true }).waitFor();
  await desktop.screenshot({ path: path.join(out, '06-desktop-running.png') });
  await fold(desktopMain, /最近回执 · (中心校验通过|执行失败|中心拒收)/).waitFor({ timeout: 90000 });
  const runReceipts = await readNode('/history?limit=5');
  assert.equal(runReceipts.items[0]?.status, 'accepted', JSON.stringify(runReceipts));
  assert.equal(calls.filter(([method, route]) => method === 'POST' && route === '/start').length, 1);
  await fold(desktopMain, '最近回执 · 中心校验通过').click();
  await desktopMain.getByText(/结果校验值 [a-f0-9]{64}/).first().waitFor();
  await desktop.screenshot({ path: path.join(out, '06-desktop-receipt.png') });
  let doneJob;
  for (let i = 0; i < 50; i++) {
    doneJob = await fetch(service.center + '/api/census/jobs/CJ-browser').then(response => response.json());
    if (doneJob.counts?.done === 2) break;
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  assert.deepEqual(doneJob.counts, { done: 2 });
  await sandboxProgress.waitFor({ state: 'hidden' });

  const node = await readNode('/status');
  fixture.stdin.write(JSON.stringify({ action: 'seed_audit', node_id: node.node_id }) + '\n');
  await fixtureLine('BROWSER_ACK');
  await fold(desktopMain, '最近回执 · 已上传，待审核').waitFor();
  fixture.stdin.write(JSON.stringify({ action: 'audit' }) + '\n');
  await fixtureLine('BROWSER_ACK');
  await fold(desktopMain, '最近回执 · 需要复测').waitFor();
  const reviewNote = desktopMain.getByText('浏览器验收：要求复测', { exact: true });
  if (!(await reviewNote.isVisible())) await fold(desktopMain, '最近回执 · 需要复测').click();
  await reviewNote.waitFor();
  await desktop.screenshot({ path: path.join(out, '07-desktop-audit.png') });
  await desktop.close();
  assert.deepEqual(report.page_errors, []);
  report.checks.push('desktop bridge and isolated center: actual claim preview, one-click lease, measured execution, uploaded result and audit receipt');
  report.passed = true;
} catch (error) {
  report.failure = error.stack || String(error);
  if (readNode) {
    report.node_history = await readNode('/history?limit=5').catch(() => null);
    report.node_log = await readNode('/logs?after=0').catch(() => null);
  }
  await (desktop || page).screenshot({ path: path.join(out, 'failure.png') }).catch(() => {});
  throw error;
} finally {
  if (fixture && fixture.exitCode === null) {
    fixture.stdin.end();
    await once(fixture, 'exit');
  }
  await fs.writeFile(path.join(out, 'playwright.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify(report));
