import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.AI2050_PLAYWRIGHT || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const out = path.resolve('tests/glm5/result/workbench_spatial_unified');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: process.env.AI2050_CHROMIUM || 'C:/Users/Admin/AppData/Local/ms-playwright/chromium-1247/chrome-win64/chrome.exe' });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
page.setDefaultTimeout(30000);
const base = 'http://127.0.0.1:5173/workbench?lens=spatial';
const api = 'http://127.0.0.1:5001';
const requests = [], errors = [];
page.on('request', request => requests.push(new URL(request.url()).pathname));
page.on('pageerror', error => errors.push(String(error)));
const report = { kind: 'engineering', passed: false, checks: [] };
const scene = page.locator('.v8-structure-scene canvas');
const block = page.getByLabel('定位模型层', { exact: true });
const detail = page.getByRole('region', { name: '当前层详情' });
const object = page.getByRole('region', { name: '选中索引' });
const btn = name => page.getByRole('button', { name, exact: true });
const read = async endpoint => {
  const response = await page.request.get(api + endpoint);
  assert(response.ok(), endpoint);
  return response.json();
};
const frame = () => scene.evaluate(canvas => canvas.toDataURL());
const waitFrame = async before => page.waitForFunction(previous => document.querySelector('.v8-structure-scene canvas')?.toDataURL() !== previous, before);
const scrollTop = async () => page.locator('.wb-main').evaluate(node => node.scrollTo(0, 0));

try {
  const { models } = await read('/api/model/skeleton');
  const qwen = models.find(m => m.model_id === 'qwen3-4b' && m.census_run);
  const [arch, explanation] = await Promise.all([
    read('/api/model/arch?' + new URLSearchParams({ model_key: qwen.model })),
    read('/api/neurons/explained?' + new URLSearchParams({ model_key: qwen.census_key })),
  ]);
  const recursive = explanation.results.find(row => row.cause === '递归泛化');
  assert(recursive);
  await page.goto(base);
  await page.waitForURL(current => current.searchParams.get('gview') === 'structure');
  await page.getByRole('img', { name: `LLM 全部 ${arch.n_layers} 层的 3D 层塔`, exact: true }).waitFor();
  assert.equal(await block.locator('option').count(), arch.n_layers + 2);
  assert.equal(await detail.count(), 0);
  assert.equal(await object.count(), 0);
  assert(!requests.includes('/api/parameters/slice'));
  assert(!requests.some(value => value.startsWith('/api/neurons/') || value === '/api/traces/research-space'));
  await page.waitForFunction(() => {
    const canvas = document.querySelector('.v8-structure-scene canvas');
    return canvas && Math.abs(canvas.clientWidth - canvas.parentElement.clientWidth) <= 1;
  });
  await page.screenshot({ path: path.join(out, 'structure-desktop.png') });

  let before = await frame();
  const rect = await scene.boundingBox();
  await page.mouse.move(rect.x + rect.width * .48, rect.y + rect.height * .42);
  await page.mouse.down();
  await page.mouse.move(rect.x + rect.width * .57, rect.y + rect.height * .49, { steps: 9 });
  await page.mouse.up();
  await waitFrame(before);
  before = await frame();
  await btn('平移').click();
  await page.mouse.move(rect.x + rect.width * .5, rect.y + rect.height * .5);
  await page.mouse.down();
  await page.mouse.move(rect.x + rect.width * .56, rect.y + rect.height * .52, { steps: 6 });
  await page.mouse.up();
  await waitFrame(before);
  before = await frame();
  await btn('放大').click();
  await waitFrame(before);
  before = await frame();
  await page.mouse.move(rect.x + rect.width * .5, rect.y + rect.height * .5);
  await page.mouse.wheel(0, 150);
  await waitFrame(before);
  await btn('重置视角').click();

  let picked = false;
  for (const offset of [0.69, 0.73, 0.78, 0.82]) {
    const point = { x: rect.x + rect.width * .51, y: rect.y + rect.height * offset };
    await page.mouse.move(point.x, point.y);
    try {
      await page.locator('.v8-structure-scene .v8-tip').filter({ hasText: /块 \d+（输出/ }).waitFor({ timeout: 700 });
    } catch {
      continue;
    }
    await page.mouse.click(point.x, point.y);
    picked = true;
    break;
  }
  assert(picked, '3D layer board is directly clickable');
  await detail.getByRole('heading', { name: /B\d+ · 层结构/ }).waitFor();
  report.checks.push('default complete tower, direct layer hit, rotation, pan, zoom and reset');

  await block.selectOption('35');
  await detail.getByRole('button', { name: new RegExp(`^注意力 Head · ${arch.n_heads}$`) }).click();
  assert.equal(await detail.locator('.v8-head-grid button').count(), arch.n_heads);
  await scene.scrollIntoViewIfNeeded();
  const headPoint = await scene.evaluate((canvas, layers) => {
    const r = canvas.getBoundingClientRect();
    const x = -.94 + 1.88 * 7.5 / 8, z = -.53 + .4 * 3.5 / 4;
    const y = -2.1 + 4.2 * 35 / (layers + 1);
    const yaw = -Math.PI / 6, pitch = 22 * Math.PI / 180;
    const s = Math.min(r.width * .3, r.height * .22) * .92;
    return { x: r.x + r.width / 2 + (x * Math.cos(yaw) - z * Math.sin(yaw)) * s,
      y: r.y + r.height / 2 - (y * Math.cos(pitch) - (x * Math.sin(yaw) + z * Math.cos(yaw)) * Math.sin(pitch)) * s };
  }, arch.n_layers);
  await page.mouse.click(headPoint.x, headPoint.y);
  await object.getByRole('heading', { name: 'B34 · H31', exact: true }).waitFor();
  const selectedHead = new URL(page.url());
  assert.equal(selectedHead.searchParams.get('sindex'), '31');
  assert.equal(selectedHead.searchParams.get('smodule'), 'attn');
  await object.locator('.v8-weight-scroll td').first().waitFor();
  await object.getByText('当前原权重与研究来源的完整快照一致。', { exact: true }).waitFor();
  const [left, right, lower] = await Promise.all([
    page.locator('.v8-structure-scene').boundingBox(), detail.boundingBox(), object.boundingBox(),
  ]);
  assert(Math.abs(left.width - right.width) < 2 && Math.abs(left.y - right.y) < 2 && right.x > left.x);
  assert(lower.y > Math.max(left.y + left.height, right.y + right.height));
  await object.screenshot({path:path.join(out,'selected-head-parameters.png')});

  await detail.getByRole('button', { name: /^MLP 单元/ }).click();
  const array = detail.getByLabel(new RegExp(`MLP 单元全部${arch.d_ff}个索引`));
  await array.waitFor();
  const last = arch.d_ff - 1;
  await array.scrollIntoViewIfNeeded();
  const grid = await array.boundingBox();
  const nativeHeight = await array.evaluate(element => element.height);
  await page.mouse.click(grid.x + ((last % 128) * 5 + 2) / 640 * grid.width,
    grid.y + (Math.floor(last / 128) * 5 + 2) / nativeHeight * grid.height);
  await object.getByRole('heading', { name: `B34 · U${last}`, exact: true }).waitFor();
  const index = page.getByLabel('定位 MLP 单元');
  await index.fill('1995');
  await object.getByRole('heading', { name: 'B34 · U1995', exact: true }).waitFor();
  await btn('研究证据').click();
  await object.getByRole('link', { name: '查看研究报告' }).first().waitFor();
  assert.equal(await object.locator('pre').count(), 0);
  await object.locator('summary').filter({ hasText: '定位记录 JSON' }).first().click();
  await object.locator('pre').first().waitFor();
  await scrollTop();
  await page.screenshot({ path: path.join(out, 'structure-object.png') });
  await btn('历史测试').click();
  await object.getByRole('link', { name: '查看研究报告' }).first().waitFor();
  report.checks.push('equal columns and full-width detail, all native Head/MLP indices, real weight slice, exact research sources and collapsed JSON');

  await detail.locator('summary').filter({ hasText: /^递归泛化线索$/ }).click();
  await detail.getByRole('button', { name: `B${recursive.block} · U${recursive.unit}`, exact: true }).click();
  await object.getByText(/该单元的解释检验记录了这一原因分类/).waitFor();
  const archiveDownload = object.getByRole('button', { name: '下载', exact: true });
  const upstream = object.getByText('上游 · 前一层写入', { exact: true });
  await Promise.race([upstream.waitFor(), archiveDownload.waitFor()]);
  if (await archiveDownload.isVisible()) await archiveDownload.click();
  await upstream.waitFor();
  await object.getByText('下游 · 后一层读取', { exact: true }).waitFor();
  await object.getByText(/虚线表示预计算/).waitFor();
  const neighbor = object.locator('.v8-structure-neighbors button').first();
  const neighborLabel = (await neighbor.locator('strong').innerText()).trim();
  await scrollTop();
  await page.screenshot({ path: path.join(out, 'structure-relations.png') });
  await neighbor.click();
  await object.getByRole('heading', { name: neighborLabel, exact: true }).waitFor();
  assert.equal(await btn('关联模式').getAttribute('aria-pressed'), 'true');
  const deepUrl = page.url();
  await page.reload();
  await object.getByRole('heading', { name: neighborLabel, exact: true }).waitFor();
  assert.equal(new URL(deepUrl).searchParams.get('smode'), 'relations');
  report.checks.push('actual recursive-generalization annotation, historical candidate pack, adjacent-node follow and reload');

  const run = 'P2794-ATOMIC-ARBITRATION-v4';
  const { traces } = await read('/api/traces/catalog?' + new URLSearchParams({ run_id: run }));
  const a = traces.find(t => t.sample_id === 'Pr:F0:L2048:world');
  const b = traces.find(t => t.sample_id === 'Pr:F0:L2048:doc');
  const legacy = new URL(base + '&view=parameters');
  Object.entries({ snapshot:qwen.model_snapshot, mdl:qwen.model, run, trace:a.trace_id, tpos:2047,
    pblock:34, pkind:'head', phead:31, prow:0, pmodel:'9a784350de7f8d690b04ffef', lblock:30 }).forEach(([k,v])=>legacy.searchParams.set(k,v));
  await page.goto(legacy.href);
  await object.getByRole('heading', { name: 'B34 · H31', exact: true }).waitFor();
  await object.locator('.v8-weight-scroll td').first().waitFor();
  const modes=page.getByRole('navigation', {name:'空间模式'});
  assert.deepEqual(await modes.locator('button:not(.wb-help-btn)').allTextContents(), ['结构','运行与路径','响应对照']);
  await modes.getByRole('button',{name:'运行与路径',exact:true}).click();
  await object.getByRole('heading', {name:'B34 · H31',exact:true}).waitFor();
  assert.equal(await page.getByLabel('真实query位置').inputValue(), '2047');
  await object.getByText(/Head 输出范数 .*KV .*重算误差/).waitFor();
  await object.getByText('完整注意力来源', {exact:true}).click();
  await object.getByLabel(/原生来源权重全部2048个索引/).waitFor();
  await detail.getByRole('button', { name: /^MLP 单元/ }).click();
  await index.fill('1995');
  const field = (trace, boundary, position=2047) => read(`/api/traces/${trace.trace_id}/field?` + new URLSearchParams({block:34,position,boundary}));
  const [aAct,bAct,gate,value] = await Promise.all([field(a,'mlp_activation'),field(b,'mlp_activation'),field(a,'mlp_gate'),field(a,'mlp_value')]);
  const exact = v => v.toPrecision(6);
  await object.getByText(`G ${exact(gate.values[1995])} · V ${exact(value.values[1995])} · A ${exact(aAct.values[1995])}`, {exact:true}).waitFor();
  await modes.getByRole('button',{name:'响应对照',exact:true}).click();
  await object.getByRole('heading', {name:'B34 · U1995',exact:true}).waitFor();
  await page.getByLabel('样本 B',{exact:true}).selectOption(b.trace_id);
  const compareText = `A ${exact(aAct.values[1995])} · B ${exact(bAct.values[1995])} · B − A ${exact(bAct.values[1995]-aAct.values[1995])}`;
  await object.getByText(compareText,{exact:true}).waitFor();
  await detail.getByLabel(/已载入完整采集响应/).waitFor();
  const coloured = await array.evaluate(canvas=>canvas.toDataURL());
  await detail.getByRole('button',{name:'样本 A',exact:true}).click();
  assert.notEqual(await array.evaluate(canvas=>canvas.toDataURL()),coloured);
  await scrollTop();
  await page.screenshot({ path:path.join(out,'response-compare.png') });
  await page.reload();
  await object.getByText(compareText,{exact:true}).waitFor();
  await modes.getByRole('button',{name:'结构',exact:true}).click();
  await object.getByRole('heading', {name:'B34 · U1995',exact:true}).waitFor();
  await object.locator('.v8-weight-scroll td').first().waitFor();
  await page.goBack();
  await object.getByText(compareText,{exact:true}).waitFor();
  await btn('回到来源样本').click();
  await object.getByText(`G ${exact(gate.values[1995])} · V ${exact(value.values[1995])} · A ${exact(aAct.values[1995])}`, {exact:true}).waitFor();
  report.checks.push('old parameter URL, archived Head and MLP values, paired native response values verified against API, object survives mode switch, reload and Back');

  await page.goto(base + `&view=live&trace=${a.trace_id}&tpos=2047&tblock=34&tunit=1995&thead=31&ttab=connections`);
  await object.getByRole('heading',{name:'B34 · U1995',exact:true}).waitFor();
  await object.getByLabel('关联 Head',{exact:true}).waitFor();
  assert.equal(await object.getByLabel('关联 Head',{exact:true}).inputValue(),'31');
  const nativePath = await read(`/api/traces/${a.trace_id}/connections?` + new URLSearchParams({block:34,position:2047,head:31,unit:1995,coordinate:0}));
  assert.equal(nativePath.verified_full_snapshot, true);
  await object.getByRole('button',{name:'核验权重并读取计算路径',exact:true}).click();
  await object.getByText(`H31 → U1995 · 局部 Gate 贡献 ${exact(nativePath.head_gate_terms[31])} · 残差贡献 ${exact(nativePath.residual_gate_term)}`,{exact:true}).waitFor();
  await page.goto(base + '&view=response&rlayer=35&tm=TG-KN-ATTR&mdl=' + encodeURIComponent(qwen.model));
  await detail.getByRole('heading',{name:'B34 · 层结构',exact:true}).waitFor();
  await page.getByRole('combobox',{name:'模型桶',exact:true}).locator('option:checked').filter({hasText:qwen.model}).waitFor({state:'attached'});
  await page.getByText('该模板未采集当前层。',{exact:true}).waitFor();
  assert.equal(await page.getByRole('combobox',{name:'层',exact:true}).inputValue(), '');
  await block.selectOption('34');
  await detail.getByRole('heading',{name:'B33 · 层结构',exact:true}).waitFor();
  assert.match(await page.getByRole('combobox',{name:'层',exact:true}).inputValue(), /@L34$/);
  report.checks.push('old run/response links restore their selections and exact weight/native-state path; missing template layers are not replaced and collected layers stay synchronized');

  await page.goto(base + '&view=parameters&pkind=embedding&prow=42&pcol=11&pmodel=9a784350de7f8d690b04ffef&snapshot=' + qwen.model_snapshot);
  await object.getByRole('heading',{name:'Token #42',exact:true}).waitFor();
  await detail.getByLabel(/词嵌入原生坐标全部.*已载入完整采集响应/).waitFor();
  assert.equal(await object.getByLabel('矩阵列').inputValue(),'11');
  const selectedScalar = await object.locator('.v8-weight-scroll td').first().innerText();
  const eValue = await detail.getByText(/^E\[11\]/).innerText();
  assert(Math.abs(Number(eValue.split(' ').at(-1)) - Number(selectedScalar)) < 1e-5);
  await object.getByLabel('矩阵列').fill('12');
  await detail.getByText(/^E\[12\]/).waitFor();
  report.checks.push('old embedding link, full real embedding row, index and scalar coordinate kept in sync');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(base);
  await scene.waitFor();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.screenshot({ path: path.join(out, 'structure-mobile.png') });
  await block.selectOption('35');
  await detail.getByRole('button', { name: /^MLP 单元/ }).click();
  await array.waitFor();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.goto(base + '&view=graph&gview=language&lblock=26');
  await page.waitForURL(current => current.searchParams.get('gview') === 'language' && current.searchParams.get('lblock') === '26');
  await btn('模型结构').click();
  await scene.waitFor();
  assert.equal(await detail.count(), 0);
  report.checks.push('390px default and full array fit; published language deep link still routes correctly');
  report.dimensions = { layers: arch.n_layers, heads: arch.n_heads, mlp_per_layer: arch.d_ff };
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
