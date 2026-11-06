import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium, _electron: electron } = require(process.env.AI2050_PLAYWRIGHT ||
  'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root = fileURLToPath(new URL('../../', import.meta.url));
const out = path.join(root, 'tests/glm5/result/official_download_1_0_7');
const release = JSON.parse(await fs.readFile(path.join(root, 'tests/glm5/result/desktop_release_1_0_7/github-release.json'), 'utf8'));
const installed = JSON.parse(await fs.readFile(process.env.AI2050_INSTALL_PROOF || path.join(out, 'download-install.json'), 'utf8'));
const archive = path.join(path.dirname(installed.executable), 'resources/app.asar');
assert.equal(createHash('sha256').update(await fs.readFile(archive)).digest('hex'), installed.archive_sha256);
const center = 'https://server.ai2050.com';
const area = process.env.AI2050_OFFICIAL_PROFILE_AREA || await fs.mkdtemp(path.join(os.tmpdir(), 'ai2050-production-client-'));
const profile = path.join(area, 'profile');
await fs.mkdir(profile, { recursive: true });
await fs.writeFile(path.join(profile, 'config.json'), JSON.stringify({ server: center, verified: true }));
const env = { ...process.env, AI2050_NODE_HOME: path.join(area, 'node'), HF_HUB_OFFLINE: '1',
  AI2050_PYTHON: path.join(root, '.venv/Scripts/python.exe') };
for (const key of ['ELECTRON_RUN_AS_NODE', 'AI2050_NODE_SYNTHETIC', 'AI2050_CLIENT_DEV_URL', 'AI2050_CLIENT_SERVER']) delete env[key];
const report = { version: release.version, center, archive_sha256: installed.archive_sha256,
  checks: [], page_errors: [], passed: false };

async function until(get, message, seconds = 60) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    const result = await get();
    if (result) return result;
    await delay(350);
  }
  throw new Error(message);
}
async function api(endpoint) {
  const response = await fetch(center + endpoint, { signal: AbortSignal.timeout(12000) });
  assert(response.ok, `${endpoint}: ${response.status}`);
  return response.json();
}

let browser, app, page, callNode;
try {
  browser = await chromium.launch({ headless: true, executablePath: process.env.AI2050_CHROMIUM ||
    'C:/Users/Admin/AppData/Local/ms-playwright/chromium-1247/chrome-win64/chrome.exe' });
  const website = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const websiteUrl = process.env.AI2050_WEBSITE_ACCEPTANCE_URL || pathToFileURL(path.join(root, 'frontend/website/index.html')).href;
  await website.goto(websiteUrl);
  const download = website.getByRole('link', { name: '下载 Windows x64 客户端 1.0.7' });
  await download.waitFor();
  assert.equal(await download.getAttribute('href'), release.download_url);
  await website.screenshot({ path: path.join(out, 'website-download.png') });
  report.checks.push({ website: websiteUrl, href: release.download_url });
  await browser.close();
  browser = null;

  const { models } = await api('/api/platform/config');
  const qwen = models.find(model => model.name.toLowerCase() === 'qwen3-4b');
  assert(qwen?.snapshot_sha256);
  app = await electron.launch({ executablePath: installed.executable, args: [`--user-data-dir=${profile}`], env, timeout: 90000 });
  const identity = await app.evaluate(({ app }) => ({ version: app.getVersion(), packaged: app.isPackaged }));
  assert.equal(identity.version, release.version);
  assert.equal(identity.packaged, true);
  page = await app.firstWindow({ timeout: 90000 });
  page.setDefaultTimeout(60000);
  page.on('pageerror', error => report.page_errors.push(String(error)));
  await page.waitForFunction(server => window.__AI2050__?.apiBase === server, center);
  callNode = (method, endpoint, body) => page.evaluate(({ method, endpoint, body }) =>
    window.ai2050Desktop.node.request(method, endpoint, body), { method, endpoint, body });
  const node = await until(async () => {
    const reply = await callNode('GET', '/status');
    return reply.status === 200 && reply.data;
  }, 'Installed node did not start');
  assert.equal(node.server, center);
  assert.equal(node.synthetic, false);
  const config = await callNode('PUT', '/config', { node_name: 'AI2050 Desktop 1.0.7 verification', participation_mode: 'self' });
  assert.equal(config.status, 200);
  const model = process.env.AI2050_ACCEPTANCE_MODEL || 'D:/develop/model/Qwen/Qwen3-4B';
  const verification = await callNode('POST', '/models/check', { snapshot: qwen.snapshot_sha256, dir: model });
  assert.equal(verification.status, 200, JSON.stringify(verification.data));
  const base = page.url().split('?')[0];
  assert(base.startsWith('file:') && base.includes('app.asar/'));
  await page.goto(base + '?wb=1&lens=process&station=plan');
  const main = page.getByRole('main');
  const participation = main.getByRole('region', { name: '参与研发' });
  if (!node.bound) {
    await main.getByRole('button', { name: '注册或绑定节点' }).click();
    await main.getByRole('button', { name: '匿名注册' }).click();
    await main.getByText(/已注册 NODE-/).waitFor();
    await main.getByRole('button', { name: '返回当前研发' }).click();
  }
  const ready = participation.getByRole('button', { name: '一键参与' });
  await ready.waitFor({ timeout: 150000 });
  const preview = await callNode('GET', '/participation');
  assert.equal(preview.status, 200);
  assert.equal(preview.data.center_current, true);
  assert(preview.data.available);
  await participation.getByText(/可领取/).waitFor();
  await main.screenshot({ path: path.join(out, 'production-ready.png') });
  const available = preview.data.available;
  report.checks.push({ installed: identity.version, server: node.server, model_snapshot: qwen.snapshot_sha256,
    node_registered: true, claim_preview: available });

  if (process.env.AI2050_VERIFY_REAL_CLAIM === '1') {
    assert.equal(available.kind, 'family_test', 'This check is bounded to one existing family test');
    const template = await api(`/api/templates/${encodeURIComponent(available.tm_id)}`);
    const shardCount = Number(available.shard?.match(/^\d+\/(\d+)$/)?.[1]) || 1;
    assert(template.meta.items > 0 && Math.ceil(template.meta.items / shardCount) <= 2000, 'Unexpected shard size');
    await ready.click();
    const active = await until(async () => {
      const current = (await callNode('GET', '/status')).data?.current;
      return current?.task && current;
    }, 'Production center did not grant a lease', 120);
    const paused = await callNode('POST', '/pause');
    assert.equal(paused.status, 200);
    assert.equal(paused.data.state, 'paused');
    assert.equal(active.tm_id, available.tm_id);
    report.checks.push({ task: active.task, template: active.tm_id, shard: active.shard, mode: 'real' });
    const completion = await until(async () => {
      const items = (await callNode('GET', '/history?limit=5')).data?.items;
      return items?.find(item => item.task === active.task);
    }, 'Production lease did not return a completion receipt', 720);
    assert.equal(completion.status, 'done', JSON.stringify(completion));
    const results = await api(`/api/results?tm_id=${encodeURIComponent(active.tm_id)}&limit=20`);
    const result = results.results.find(item => item.sha === completion.sha);
    assert.equal(result?.kind, 'real');
    const receipt = participation.locator('summary').filter({ hasText: '最近回执' });
    await receipt.waitFor();
    await receipt.click();
    await participation.getByText(`结果校验值 ${completion.sha}`, { exact: true }).waitFor();
    await main.screenshot({ path: path.join(out, 'production-receipt.png') });
    report.checks.push({ production_result: completion.sha, result_kind: result.kind, state: 'uploaded' });
  }
  assert.deepEqual(report.page_errors, []);
  report.passed = true;
} catch (error) {
  report.failure = String(error.stack || error);
  if (page) await page.screenshot({ path: path.join(out, 'production-failure.png') }).catch(() => {});
  if (callNode) report.node_status = (await callNode('GET', '/status').catch(() => null))?.data;
  throw error;
} finally {
  if (browser) await browser.close().catch(() => {});
  if (app) await app.close().catch(() => {});
  await fs.writeFile(path.join(out, 'production-acceptance.json'), JSON.stringify(report, null, 2));
}
console.log(JSON.stringify(report));
