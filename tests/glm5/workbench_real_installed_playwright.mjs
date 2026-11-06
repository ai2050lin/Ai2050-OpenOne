import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { createInterface } from 'node:readline';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { _electron: electron } = require(process.env.AI2050_PLAYWRIGHT ||
  'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root = fileURLToPath(new URL('../../', import.meta.url));
const out = path.join(root, 'tests/glm5/result/desktop_release_1_0_7');
const packaged = JSON.parse(await fs.readFile(path.join(out, 'package-verification.json'), 'utf8'));
const install = JSON.parse(await fs.readFile(path.join(out, 'install.json'), 'utf8'));
const archive = path.join(path.dirname(install.executable), 'resources/app.asar');
assert.equal(install.exit_code, 0);
assert.equal(createHash('sha256').update(await fs.readFile(archive)).digest('hex'), packaged.archive_sha256);
const python = path.join(root, '.venv/Scripts/python.exe');
const model = process.env.AI2050_ACCEPTANCE_MODEL || 'D:/develop/model/Qwen/Qwen3-4B';
const area = await fs.mkdtemp(path.join(os.tmpdir(), 'ai2050-1.0.7-real-'));
const report = { schema: 'installed_research_acceptance.v1', version: packaged.version,
  archive_sha256: packaged.archive_sha256, passed: false, kind: 'real', checks: [], page_errors: [] };
const fixture = spawn(python, ['-u', path.join(root, 'tests/glm5/desktop_real_release_center.py'), area, model],
  { cwd: root, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
let fixtureError = '';
const lines = [];
fixture.stderr.on('data', chunk => { fixtureError += String(chunk); });
createInterface({ input: fixture.stdout }).on('line', line => lines.push(line));
async function line(prefix) {
  for (let i = 0; i < 600; i++) {
    const index = lines.findIndex(item => item.startsWith(prefix));
    if (index !== -1) return JSON.parse(lines.splice(index, 1)[0].slice(prefix.length));
    if (fixture.exitCode !== null) throw new Error(fixtureError || `Center exit ${fixture.exitCode}`);
    await delay(100);
  }
  throw new Error(`Center timed out before ${prefix}`);
}
async function waitUntil(check, message, seconds = 60) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    const result = await check();
    if (result) return result;
    await delay(500);
  }
  throw new Error(message);
}

let app, page, callNode;
try {
  const center = await line('CENTER_READY ');
  const readJob = async () => {
    const response = await fetch(`${center.url}/api/census/jobs/${center.job}`, { signal: AbortSignal.timeout(8000) });
    assert(response.ok);
    return response.json();
  };
  const profile = path.join(area, 'profile');
  const env = { ...process.env, AI2050_CLIENT_SERVER: center.url, AI2050_NODE_HOME: path.join(area, 'node'),
    AI2050_PYTHON: python, HF_HUB_OFFLINE: '1' };
  for (const key of ['ELECTRON_RUN_AS_NODE', 'AI2050_NODE_SYNTHETIC', 'AI2050_CLIENT_DEV_URL']) delete env[key];
  app = await electron.launch({ executablePath: install.executable, args: ['--user-data-dir=' + profile], env, timeout: 90000 });
  const identity = await app.evaluate(({ app }) => ({ version: app.getVersion(), packaged: app.isPackaged, archive: app.getAppPath() }));
  assert.equal(identity.version, packaged.version);
  assert.equal(identity.packaged, true);
  assert.equal(path.resolve(identity.archive).toLowerCase(), archive.toLowerCase());
  report.checks.push({ installed_version: identity.version });
  page = await app.firstWindow({ timeout: 90000 });
  page.setDefaultTimeout(45000);
  page.on('pageerror', error => report.page_errors.push(String(error)));
  await page.waitForFunction(() => Boolean(window.ai2050Desktop?.node));
  callNode = (method, endpoint, body) => page.evaluate(({ method, endpoint, body }) =>
    window.ai2050Desktop.node.request(method, endpoint, body), { method, endpoint, body });
  const node = await waitUntil(async () => {
    const response = await callNode('GET', '/status');
    return response.status === 200 && response.data;
  }, 'Installed node did not start');
  assert.equal(node.synthetic, false);
  assert.equal(node.server, center.url);
  const config = await callNode('PUT', '/config', { task_kinds: ['census_shard'], node_name: 'desktop-1.0.7-real-acceptance' });
  assert.equal(config.status, 200);
  const check = await callNode('POST', '/models/check', { snapshot: center.snapshot, dir: model });
  assert.equal(check.status, 200, JSON.stringify(check.data));

  const pageUrl = page.url().split('?')[0];
  assert(pageUrl.startsWith('file:') && pageUrl.includes('app.asar/'));
  await page.goto(pageUrl + '?wb=1&lens=process&station=plan');
  const main = page.getByRole('main');
  const progress = main.getByRole('region', { name: '研究进度' });
  const participation = main.getByRole('region', { name: '参与研发' });
  const total = progress.getByRole('progressbar', { name: /qwen3-4b · 神经元普查已完成分片/i });
  await total.waitFor();
  assert.equal(await total.getAttribute('value'), '0');
  assert.equal(await total.getAttribute('max'), '2');
  assert.deepEqual((await readJob()).counts, { queued: 2 });
  await main.getByRole('button', { name: '注册或绑定节点' }).click();
  await main.getByRole('button', { name: '匿名注册' }).click();
  await main.getByText(/已注册 NODE-/).waitFor();
  await main.getByRole('button', { name: '返回当前研发' }).click();
  const start = participation.getByRole('button', { name: '一键参与', exact: true });
  await start.waitFor({ timeout: 120000 });
  const prep = await callNode('GET', '/participation');
  assert.equal(prep.data.available.job, center.job);
  assert.equal(prep.data.available.shard, 'S0-G0');
  await participation.getByText(/可领取 · 神经元普查 · S0-G0/).waitFor();
  await main.screenshot({ path: path.join(out, '01-installed-ready.png') });
  report.checks.push({ published_shards: 2, ready: prep.data.available });
  await start.click();

  const measured = await waitUntil(async () => {
    const data = (await callNode('GET', '/status')).data?.current;
    return data?.task && data.progress?.done > 0 && data.progress?.total > 0 && data;
  }, 'No measured real progress from the installed node', 240);
  const bar = participation.getByRole('progressbar', { name: '本机任务进度' });
  await bar.waitFor();
  assert((await readJob()).counts.running > 0);
  await main.screenshot({ path: path.join(out, '02-installed-running.png') });
  report.checks.push({ task: measured.task, model: measured.model, shard: measured.shard, progress: measured.progress });

  const receipt = participation.locator('summary').filter({ hasText: '最近回执 · 中心校验通过' });
  await receipt.waitFor({ timeout: 180000 });
  await receipt.click();
  await participation.locator('details[open] .v8-rnd-receipts li').first().waitFor();

  const complete = await waitUntil(async () => {
    const job = await readJob();
    return job.counts.done === 2 && job;
  }, 'Center did not accept both real shards', 300);
  fixture.stdin.write('inspect\n');
  const validated = await line('CENTER_RESULTS ');
  assert.equal(validated.length, 2);
  for (const row of validated) {
    assert.equal(row.kind, 'real');
    assert.equal(row.status, 'accepted');
    assert.equal(row.snapshot, center.snapshot);
    assert.equal(row.validation.ok, true);
    assert(row.n_tokens > 0);
    assert.match(row.env.device, /^cuda/);
  }
  for (const row of validated) await participation.getByText(`结果校验值 ${row.sha}`, { exact: true }).waitFor();
  await progress.getByText('暂无进行中的研究或开放作业', { exact: true }).waitFor();
  await main.screenshot({ path: path.join(out, '03-installed-receipt.png') });
  report.checks.push({ job: center.job, corpus: center.corpus, counts: complete.counts,
    receipt_remained_open: true, validation: validated });
  assert.deepEqual(report.page_errors, []);
  report.passed = true;
} catch (error) {
  report.failure = String(error.stack || error);
  if (page) await page.screenshot({ path: path.join(out, 'installed-failure.png') }).catch(() => {});
  if (callNode) {
    report.node_status = await callNode('GET', '/status').catch(() => null);
    report.node_log = await callNode('GET', '/logs?after=0').catch(() => null);
  }
  throw error;
} finally {
  if (app) await app.close().catch(() => {});
  if (fixture.exitCode === null) {
    fixture.stdin.end();
    await once(fixture, 'exit');
  }
  await fs.writeFile(path.join(out, 'installed-acceptance.json'), JSON.stringify(report, null, 2));
}
console.log(JSON.stringify(report));
