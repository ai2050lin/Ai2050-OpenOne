// Isolated UI contribution regression adapted from workbench_research_tabs_playwright.mjs.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { once } from 'node:events';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.AI2050_PLAYWRIGHT || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const out = path.resolve('tests/glm5/result/research_layout_20261109/contribution');
const tempRoot = path.resolve('tests/glm5_temp/research_layout_node');
await fs.mkdir(tempRoot, { recursive: true });
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: process.env.AI2050_CHROMIUM ||
  'C:/Program Files/Google/Chrome/Application/chrome.exe' });
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
  fixture = spawn(process.env.AI2050_TEST_PYTHON || '.venv/Scripts/python.exe',
    ['-u', 'tests/glm5/current_research_browser_fixture.py'], { stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, TMP:tempRoot, TEMP:tempRoot, PYTHONDONTWRITEBYTECODE:'1' } });
  fixture.stderr.on('data', buffer => { fixtureErrors += String(buffer); });
  createInterface({ input: fixture.stdout }).on('line', line => fixtureLines.push(line));
  const service = JSON.parse(await fixtureLine('BROWSER_FIXTURE '));
  console.log('isolated fixture ready');
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
  console.log('preparation and task preview passed');
  await desktopButton('一键参与').click();
  await participation.locator('progress[value]').waitFor();
  const measured = await participation.getByRole('progressbar', { name: '本机任务进度' }).evaluate(el => ({ done: el.value, total: el.max }));
  assert(measured.done > 0 && measured.done <= measured.total);
  await desktopMain.getByText('1 运行中 · 1 待领取', { exact: true }).waitFor();
  await desktop.screenshot({ path: path.join(out, '06-desktop-running.png') });
  await fold(desktopMain, /最近回执 · (中心校验通过|执行失败|中心拒收)/).waitFor({ timeout: 90000 });
  console.log('claim, measured progress and upload receipt observed');
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
  console.log('audit receipt update observed');
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
