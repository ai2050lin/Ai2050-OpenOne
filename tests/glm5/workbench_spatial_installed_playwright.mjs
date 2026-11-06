import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { _electron: electron } = require(process.env.AI2050_PLAYWRIGHT ||
  'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root = fileURLToPath(new URL('../../', import.meta.url));
const out = path.join(root, 'tests/glm5/result/workbench_spatial_desktop');
const packaged = JSON.parse(await fs.readFile(path.join(out, 'package-verification.json'), 'utf8'));
const installed = path.resolve(process.env.AI2050_INSTALLED_EXE);
const installedArchive = path.join(path.dirname(installed), 'resources/app.asar');
const hash = createHash('sha256');
for await (const chunk of createReadStream(installedArchive)) hash.update(chunk);
assert.equal(hash.digest('hex'), packaged.archive_sha256);
assert(!installed.includes('win-unpacked'), 'Acceptance must run the NSIS-installed executable');

const base = await fs.mkdtemp(path.join(os.tmpdir(), 'ai2050-installed-acceptance-'));
const profile = path.join(base, 'profile');
const center = process.env.AI2050_ACCEPTANCE_CENTER || 'http://127.0.0.1:5001';
const modelDir = process.env.AI2050_ACCEPTANCE_MODEL || 'D:/develop/model/Qwen/Qwen3-4B';
const env = {
  ...process.env, AI2050_CLIENT_SERVER: center, AI2050_NODE_HOME: path.join(base, 'node'),
  AI2050_PYTHON: path.join(root, '.venv/Scripts/python.exe'),
};
for (const key of ['ELECTRON_RUN_AS_NODE', 'AI2050_NODE_SYNTHETIC', 'AI2050_CLIENT_DEV_URL']) delete env[key];

const report = { schema: 'installed_workbench_acceptance.v1', version: packaged.version, installer_executed: true,
  passed: false, checks: [], page_errors: [] };
let app, page;
const check = (name, details) => {
  report.checks.push({ name, ...details });
  console.log(name);
};
try {
  app = await electron.launch({ executablePath: installed, args: ['--user-data-dir=' + profile], env, timeout: 90000 });
  const identity = await app.evaluate(({ app }) => ({
    version: app.getVersion(), packaged: app.isPackaged, archive: app.getAppPath(), profile: app.getPath('userData'),
  }));
  assert.equal(identity.version, packaged.version);
  assert.equal(identity.packaged, true);
  assert.equal(path.resolve(identity.archive).toLowerCase(), installedArchive.toLowerCase());
  assert.equal(path.resolve(identity.profile).toLowerCase(), profile.toLowerCase());
  check('installed_application_identity', { ...identity, archive_sha256: packaged.archive_sha256 });

  page = await app.firstWindow({ timeout: 90000 });
  page.setDefaultTimeout(60000);
  page.on('pageerror', error => report.page_errors.push(String(error)));
  await page.waitForFunction(() => Boolean(window.ai2050Desktop?.node));
  const callNode = (method, endpoint, body) => page.evaluate(({ method, endpoint, body }) =>
    window.ai2050Desktop.node.request(method, endpoint, body), { method, endpoint, body });
  let status;
  const deadline = Date.now() + 45000;
  while (Date.now() < deadline) {
    status = await callNode('GET', '/status');
    if (status.status === 200) break;
    await delay(350);
  }
  if (status?.status !== 200) {
    const { state, error } = await page.evaluate(() => window.ai2050Desktop.node.info());
    throw new Error(`Packaged local node did not start: ${state} · ${error}`);
  }
  assert.equal(status.data.state, 'stopped');
  assert.equal(status.data.bound, false);
  const config = await callNode('PUT', '/config', { models_root: modelDir });
  assert.equal(config.status, 200, JSON.stringify(config.data));
  const models = await callNode('GET', '/parameters/models');
  const local = models.data.models.find(model => model.name.toLowerCase() === 'qwen3-4b');
  assert(local, 'No real local Qwen3-4B weights available through desktop IPC');
  check('packaged_node_service', { source: models.data.source, model: local.name });

  const read = async endpoint => {
    const response = await fetch(center + endpoint, { signal: AbortSignal.timeout(30000) });
    assert(response.ok, `${endpoint}: ${response.status}`);
    return response.json();
  };
  const qwen = (await read('/api/model/skeleton')).models.find(model => model.model_id === 'qwen3-4b' && model.census_run);
  const run = 'P2794-ATOMIC-ARBITRATION-v4';
  const { traces } = await read('/api/traces/catalog?' + new URLSearchParams({ run_id: run }));
  const a = traces.find(trace => trace.sample_id === 'Pr:F0:L2048:world');
  const b = traces.find(trace => trace.sample_id === 'Pr:F0:L2048:doc');
  assert(a && b);
  const position = a.readout_positions.at(-1);
  const basePage = page.url().split('?')[0];
  assert(basePage.startsWith('file:') && basePage.includes('app.asar/'));
  const spatial = (params = {}) => basePage + '?' + new URLSearchParams({
    wb: '1', lens: 'spatial', mdl: qwen.model, snapshot: qwen.model_snapshot, ...params,
  });
  const modes = page.getByRole('navigation', { name: '空间模式' });
  const detail = page.getByRole('region', { name: '选中索引' });
  const layerPane = page.getByRole('region', { name: '当前层详情' });
  const field = (trace, boundary) => read(`/api/traces/${trace.trace_id}/field?` +
    new URLSearchParams({ block: 34, position, boundary }));
  const pickMode = mode => modes.getByRole('button', { name: mode, exact: true }).click();
  const weight = detail.locator('.v8-weight-scroll td').first();
  const selected = name => detail.getByRole('heading', { name, exact: true }).waitFor();
  const validSlice = () => detail.getByText('当前原权重与研究来源的完整快照一致。', { exact: true }).waitFor();

  // This is the old parameter URL form, including the historical language-layer pointer.
  await page.goto(spatial({ view: 'parameters', pblock: '0', pkind: 'head', phead: '0', prow: '0',
    pmodel: local.id, lblock: '30', run, trace: a.trace_id, tpos: position }));
  await selected('B0 · H0');
  await validSlice();
  await weight.waitFor();
  const slice = await callNode('GET', '/parameters/slice?' + new URLSearchParams({
    model: local.id, name: 'model.layers.0.self_attn.q_proj.weight', row: 0, col: 0, rows: 8, cols: 8,
  }));
  assert.equal(slice.status, 200);
  assert.equal(await weight.innerText(), slice.data.values[0][0].toPrecision(5));
  const hashCheck = await callNode('GET', '/parameters/identity?' + new URLSearchParams({ model: local.id }));
  assert.equal(hashCheck.data.sha256, qwen.model_snapshot);
  check('legacy_parameters_and_native_weight', { object: 'B0 · H0', source: slice.data.source, snapshot: hashCheck.data.sha256 });
  await detail.screenshot({ path: path.join(out, 'installed-real-parameters.png') });

  await page.getByLabel('定位模型层').selectOption('35');
  await layerPane.getByRole('button', { name: /^MLP 单元/ }).click();
  await page.getByLabel('定位 MLP 单元').fill('1995');
  await selected('B34 · U1995');
  await detail.getByRole('button', { name: '研究证据', exact: true }).click();
  const paper = detail.getByRole('link', { name: '查看研究报告', exact: true }).first();
  await paper.waitFor();
  assert((await paper.getAttribute('href')).includes('run='));
  assert.equal(await detail.locator('pre').count(), 0);
  await detail.getByText('原始证据 JSON', { exact: true }).first().click();
  await detail.locator('pre').first().waitFor();
  const evidence = await detail.locator('pre').first().innerText();
  assert(Object.keys(JSON.parse(evidence)).length > 0);
  await detail.screenshot({ path: path.join(out, 'installed-research-evidence.png') });
  check('published_research_card_and_source', { object: 'B34 · U1995', raw_json_loaded: true });

  await pickMode('运行与路径');
  await selected('B34 · U1995');
  const [aValue, bValue] = (await Promise.all([field(a, 'mlp_activation'), field(b, 'mlp_activation')]))
    .map(response => response.values[1995]);
  const fmt = value => value.toPrecision(6);
  await detail.getByText(new RegExp(`^G .+ · V .+ · A ${fmt(aValue)}$`)).waitFor();
  await pickMode('响应对照');
  await selected('B34 · U1995');
  await page.getByLabel('样本 B', { exact: true }).selectOption(b.trace_id);
  const comparison = `A ${fmt(aValue)} · B ${fmt(bValue)} · B − A ${fmt(bValue - aValue)}`;
  await detail.getByText(comparison, { exact: true }).waitFor();
  await page.locator('.wb-main').evaluate(main => main.scrollTo(0, 0));
  await page.screenshot({ path: path.join(out, 'installed-response-comparison.png') });
  await page.reload();
  await selected('B34 · U1995');
  await detail.getByText(comparison, { exact: true }).waitFor();
  await pickMode('结构');
  await selected('B34 · U1995');
  await validSlice();
  await weight.waitFor();
  await page.goBack();
  await detail.getByText(comparison, { exact: true }).waitFor();
  check('three_modes_reload_and_back', { object: 'B34 · U1995', paired_input_tokens: [a.n_tokens, b.n_tokens] });

  await page.goto(spatial({ view: 'live', run, trace: a.trace_id, tpos: position, tblock: '34',
    tunit: '1995', thead: '31', ttab: 'connections' }));
  await selected('B34 · U1995');
  const other = detail.getByLabel('关联 Head', { exact: true });
  await other.waitFor();
  assert.equal(await other.inputValue(), '31');
  assert.equal(await page.getByLabel('真实query位置').inputValue(), String(position));
  await page.goto(spatial({ view: 'parameters', pkind: 'embedding', prow: '42', pcol: '11', pmodel: local.id }));
  await selected('Token #42');
  await layerPane.getByLabel(/词嵌入原生坐标全部.*已载入完整采集响应/).waitFor();
  assert.equal(await detail.getByLabel('矩阵列').inputValue(), '11');
  await page.goto(spatial({ view: 'response', rlayer: '35', tm: 'TG-KN-ATTR' }));
  await layerPane.getByRole('heading', { name: 'B34 · 层结构' }).waitFor();
  const responsePanel = page.locator('.v6-response');
  const template = responsePanel.locator('select').filter({ has: page.locator('option[value="TG-KN-ATTR"]') });
  await template.waitFor();
  assert.equal(await template.inputValue(), 'TG-KN-ATTR');
  const templatePack = page.locator('.op-extra').filter({ hasText: '实验响应 · TG-KN-ATTR' });
  await templatePack.getByRole('button', { name: '下载', exact: true }).click();
  await page.getByText('该模板未采集当前层。', { exact: true }).waitFor();
  await page.getByLabel('定位模型层').selectOption('34');
  const templateLayer = responsePanel.locator('select').filter({ has: page.locator('option[value$="@L34"]') });
  assert.match(await templateLayer.inputValue(), /@L34$/);
  await page.goto(spatial({ view: 'graph', gview: 'language', lblock: '26' }));
  await page.getByRole('group', { name: '图谱图层' }).getByRole('button', { name: '外部语言模式' }).waitFor();
  assert.equal(new URL(page.url()).searchParams.get('lblock'), '26');
  check('legacy_run_embedding_response_and_language_routes', { template_pack_downloaded: true });
  assert.deepEqual(report.page_errors, []);
  report.passed = true;
} catch (error) {
  report.failure = error.stack || String(error);
  if (page) {
    await page.screenshot({ path: path.join(out, 'installed-failure.png') }).catch(() => {});
    await fs.writeFile(path.join(out, 'installed-failure-body.txt'), await page.locator('body').innerText().catch(() => ''));
  }
  throw error;
} finally {
  if (app) await app.close();
  await fs.writeFile(path.join(out, 'installed-acceptance.json'), JSON.stringify(report, null, 2));
}
