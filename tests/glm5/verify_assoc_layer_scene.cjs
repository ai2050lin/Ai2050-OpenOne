/* Real local-client acceptance. Negative routes below are engineering fixtures only. */
const { chromium } = require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const output = path.resolve(__dirname, 'result/assoc_layer_client_20261109');
fs.mkdirSync(output, { recursive: true });
const report = { time: new Date().toISOString(), checks: [], errors: [], screenshots: [] };
function check(name, detail = {}) { report.checks.push({ name, ...detail }); console.log(JSON.stringify({ step: name, completed: report.checks.length, ...detail })); }
function f16(o, index) {
  const h = Buffer.from(o.b64, 'base64').readUInt16LE(2 * index), sign = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  return e === 0 ? sign * m * 2 ** -24 : e === 31 ? m ? NaN : sign * Infinity : sign * (1 + m / 1024) * 2 ** (e - 15);
}
const main = async () => {
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: ['--no-sandbox'] });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage(); page.setDefaultTimeout(30000);
    page.on('pageerror', e => report.errors.push(e.message));
    const list = await (await context.request.get('http://127.0.0.1:5001/api/traces/mainline/ma-h2/samples')).json();
    const raw = await (await context.request.get('http://127.0.0.1:5001/api/traces/mainline/ma-h2/sample/0')).json();
    const p = list.positions.indexOf('ans'), [P, NL, DM] = raw.hs.shape, [NB, NF] = raw.mlp.shape, [, NH, T] = raw.attn.shape;
    report.source = { model_snapshot: raw.model_snapshot, sample: 0, hs: raw.hs.shape, mlp: raw.mlp.shape, attn: raw.attn.shape, token_position: raw.tokpos[p] };
    async function ready(target = page) {
      await target.goto('http://127.0.0.1:5173/workbench?lens=spatial&view=live');
      await target.locator('.aw-cin').getByText(/现在显示的是冻结样本 #0/).waitFor();
      await target.locator('.aw-canvas canvas').waitFor();
    }
    await ready();
    async function shot(name, target = page) { const file = path.join(output, name + '.png'); await target.screenshot({ path: file, fullPage: true }); report.screenshots.push(file); }
    const boxes = {};
    for (const selector of ['.aw-space', '.aw-cons', '.aw-models', '.aw-cin', '.aw-body', '.fw-rail']) boxes[selector] = await page.locator(selector).boundingBox();
    assert.equal(boxes['.aw-space'].height, 562); assert.equal(boxes['.aw-space'].width, 1348);
    assert.equal(boxes['.aw-cons'].y, boxes['.aw-space'].y + boxes['.aw-space'].height + 10);
    assert.equal(boxes['.aw-body'].x, boxes['.aw-models'].x);
    check('real_data_and_existing_outer_layout', { boxes, count: { P, NL, DM, NB, NF, NH, T } });
    await shot('desktop-default');
    const canvas = page.locator('.aw-canvas canvas');
    async function click3d(x, y, z) {
      await canvas.scrollIntoViewIfNeeded(); const b = await canvas.boundingBox();
      const yaw = -.26, pitch = .71, scale = Math.min(b.width / 1060, b.height / 500), x1 = x * Math.cos(yaw) - z * Math.sin(yaw), z1 = x * Math.sin(yaw) + z * Math.cos(yaw);
      await canvas.click({ position: { x: b.width * .455 + x1 * scale, y: b.height * .52 - (y * Math.cos(pitch) - z1 * Math.sin(pitch)) * scale } });
    }
    await click3d(-378, (35 - 18) * 9.2 + 20, 44.5);
    await page.waitForFunction(() => document.querySelector('.aw-big b')?.textContent === '第 35 层');
    check('click_tower_selects_previously_transparent_layer');
    async function clickCell(kind, index) {
      const zone = kind === 'heads' ? [-8, 143, -147, -34] : kind === 'hs' ? [181, 491, -147, -34] : [-8, 491, 31, 203];
      const n = kind === 'heads' ? NH : kind === 'hs' ? DM : NF, w = kind === 'heads' ? 8 : Math.ceil(Math.sqrt(n * 1.6)), h = Math.ceil(n / w);
      await click3d(zone[0] + (zone[1] - zone[0]) * (index % w + .5) / w, 20, zone[2] + (zone[3] - zone[2]) * (Math.floor(index / w) + .5) / h);
    }
    const popup = page.getByRole('dialog', { name: '单元详情' });
    for (const [kind, idx, expected] of [
      ['heads', 7, f16(raw.attn, (34 * NH + 7) * T + raw.tokpos[p])],
      ['hs', 2049, f16(raw.hs, (p * NL + 35) * DM + 2049)],
      ['mlp', 2663, f16(raw.mlp, 34 * NF + 2663)],
    ]) {
      await clickCell(kind, idx); await popup.waitFor();
      assert.ok((await popup.innerText()).includes(expected.toFixed(4)), `${kind} native value`);
      check(`click_${kind}_native_value`, { layer: 35, index: idx, expected });
      await popup.getByRole('button', { name: '关闭', exact: true }).click();
    }
    async function locate(target, kind, layer, index) {
      await target.getByRole('button', { name: '精确定位', exact: true }).click();
      const f = target.getByRole('form', { name: '原生索引定位' });
      await f.getByLabel('定位对象类型').selectOption(kind);
      await f.locator('[name=layer]').fill(String(layer)); await f.locator('[name=index]').fill(String(index));
      await f.getByRole('button', { name: '定位单元', exact: true }).click();
      await target.getByRole('dialog', { name: '单元详情' }).waitFor();
    }
    await locate(page, 'mlp', 35, 2663);
    await popup.getByText('特征与前后层关联', { exact: true }).click();
    await popup.getByText(/激活 A 均值/).first().waitFor();
    await popup.getByRole('button', { name: /定位 B33 · U/ }).first().waitFor();
    await popup.getByRole('button', { name: /定位 B35 · U/ }).first().waitFor();
    await popup.getByText('读取已登记的研究卡…', { exact: true }).waitFor({ state: 'hidden' });
    const info = await popup.innerText(); assert.ok(info.includes('留出共同响应')); assert.ok(info.includes('共同响应不等于因果连接'));
    check('real_feature_cards_and_both_neighbors', { text: info.slice(0, 2200) });
    await shot('mlp-neighborhood');
    const nextButton = popup.getByRole('button', { name: /定位 B33 · U/ }).first();
    const nextIndex = Number((await nextButton.innerText()).match(/U(\d+)/)[1]);
    await nextButton.click();
    await page.waitForFunction(() => document.querySelector('.aw-big b')?.textContent === '第 34 层');
    assert.ok((await popup.innerText()).includes(f16(raw.mlp, 33 * NF + nextIndex).toFixed(4)));
    check('neighbor_locates_actual_previous_layer', { block: 33, unit: nextIndex });
    await popup.getByRole('button', { name: '关闭', exact: true }).click();
    await shot('mlp-links');
    await page.getByRole('tab', { name: 'HiddenState', exact: true }).click();
    assert.ok((await page.locator('.aw-kv').first().innerText()).includes(`B33·U${nextIndex}`));
    await page.getByRole('tab', { name: 'MLP 神经元', exact: true }).click();
    assert.ok((await page.locator('.aw-pbody').innerText()).includes('特征与前后层关联'));
    check('switch_analysis_plugin_keeps_native_selection');
    await page.locator('.aw-toks button').filter({ hasText: String(raw.tokpos[list.positions.indexOf('qname')]) }).click();
    await locate(page, 'mlp', 35, 2663);
    assert.ok((await popup.innerText()).includes('这个位置没有采集 MLP 激活'));
    check('missing_token_mlp_not_filled_with_answer_values');
    await page.locator('.aw-toks button').filter({ hasText: String(raw.tokpos[p]) }).click();
    assert.equal(await popup.count(), 0); check('readout_change_closes_old_popup');
    for (const [kind, l, index] of [['mlp', 36, NF - 1], ['attn', 36, NH - 1], ['hs', 0, DM - 1]]) {
      await locate(page, kind, l, index); check('full_native_index_access', { kind, l, index });
      await popup.getByRole('button', { name: '关闭', exact: true }).click();
    }
    await locate(page, 'attn', 1, 2);
    await popup.getByText('特征与前后层关联', { exact: true }).click();
    await popup.getByText('自然样本Gate来源 B0/H2', { exact: true }).waitFor();
    assert.ok(await popup.getByRole('link', { name: '查看研究报告', exact: true }).count());
    check('registered_head_feature_and_original_research_link');
    await popup.getByRole('button', { name: '关闭', exact: true }).click();
    const headPlugin = page.locator('.aw-pl').filter({ has: page.locator('b', { hasText: '注意力头' }) });
    await headPlugin.locator('div').first().click();
    await headPlugin.locator('select').first().selectOption('tcol');
    await locate(page, 'attn', 1, 2);
    const target = raw.tokpos[list.positions.indexOf('tcol')], expectedHead = f16(raw.attn, 2 * T + target);
    assert.ok((await popup.innerText()).includes(expectedHead.toFixed(4)));
    assert.ok((await popup.innerText()).includes(`目标 token ${target}`));
    check('head_color_and_detail_use_same_target', { target, expectedHead });
    await popup.getByRole('button', { name: '关闭', exact: true }).click();
    const hash = () => canvas.evaluate(c => c.toDataURL());
    async function drag(button) {
      const b = await canvas.boundingBox(); const old = await hash();
      await page.mouse.move(b.x + b.width * .5, b.y + 70); await page.mouse.down({ button });
      await page.mouse.move(b.x + b.width * .5 + 42, b.y + 88, { steps: 6 }); await page.mouse.up({ button });
      assert.notEqual(await hash(), old); assert.equal(await popup.count(), 0);
    }
    await canvas.scrollIntoViewIfNeeded(); await drag('left'); check('left_drag_pan'); await drag('right'); check('right_drag_rotate');
    const old = await hash(); await page.mouse.wheel(0, -240); await page.waitForTimeout(150); assert.notEqual(await hash(), old); check('wheel_zoom');
    await page.getByRole('button', { name: '⛶ 全屏', exact: true }).click();
    assert.equal((await canvas.boundingBox()).height, 1000); await page.keyboard.press('Escape'); check('fullscreen_and_escape');
    await page.reload(); await page.locator('.aw-cin').getByText(/现在显示的是冻结样本 #0/).waitFor();
    await page.setViewportSize({ width: 390, height: 844 });
    await locate(page, 'mlp', 35, 2663);
    const cb = await canvas.boundingBox(), pb = await popup.boundingBox();
    assert.ok(pb.x >= cb.x && pb.x + pb.width <= cb.x + cb.width + 1); assert.ok(pb.y + pb.height <= cb.y + cb.height + 1);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await shot('mobile-unit'); check('mobile_popup_and_no_horizontal_overflow', { canvas: cb, popup: pb });
    await popup.getByRole('button', { name: '关闭', exact: true }).click(); await shot('mobile-default');
    // Explicitly isolated failure fixtures. Never submitted to the research center.
    const first = list.items[0], twin = list.items.find(x => x.entity === first.entity && x.cell[0] !== first.cell[0] && x.cell.slice(1).join() === first.cell.slice(1).join());
    const mismatch = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    await mismatch.route(`**/api/traces/mainline/ma-h2/sample/${twin.i}`, async route => {
      const response = await route.fetch(), body = await response.json(); body.model_snapshot = 'engineering-mismatched-snapshot';
      await route.fulfill({ json: body });
    });
    const mp = await mismatch.newPage(); mp.on('pageerror', e => report.errors.push(e.message)); await ready(mp);
    await locate(mp, 'mlp', 35, 2663);
    assert.ok(!(await mp.getByRole('dialog', { name: '单元详情' }).innerText()).includes('对照运行'));
    await mp.getByRole('dialog', { name: '单元详情' }).getByRole('button', { name: '关闭', exact: true }).click();
    await mp.locator('.aw-pl').filter({ has: mp.locator('b', { hasText: '条件对照' }) }).getByRole('checkbox').check();
    await mp.getByText('对照模型快照或数组维度不匹配，已停止比较。', { exact: true }).waitFor();
    check('engineering_negative_mismatched_comparison_rejected'); await mismatch.close();
    const stale = await browser.newContext(); let relationsRequested = 0;
    await stale.route('**/api/research/mlp-tail', async route => {
      const response = await route.fetch(), body = await response.json(); body.model_snapshot = 'engineering-wrong-atlas'; await route.fulfill({ json: body });
    });
    stale.on('request', r => { if (r.url().includes('/mlp-tail/relations/')) relationsRequested++; });
    const sp = await stale.newPage(); sp.on('pageerror', e => report.errors.push(e.message)); await ready(sp);
    const atlasResponse = sp.waitForResponse(r => r.url().endsWith('/api/research/mlp-tail'));
    await locate(sp, 'mlp', 35, 2663); await atlasResponse;
    await sp.getByRole('dialog', { name: '单元详情' }).getByText('特征与前后层关联', { exact: true }).click();
    await sp.getByRole('dialog', { name: '单元详情' }).getByText('当前快照暂无可显示的关联记录。', { exact: true }).first().waitFor();
    assert.equal(relationsRequested, 0); check('engineering_negative_wrong_atlas_snapshot_not_joined'); await stale.close();
    const missing = await browser.newContext();
    await missing.route('**/api/traces/mainline/ma-h2/samples', route => route.fulfill({ status: 503, json: { detail: 'engineering missing native data' } }));
    const np = await missing.newPage(); np.on('pageerror', e => report.errors.push(e.message));
    await np.goto('http://127.0.0.1:5173/workbench?lens=spatial&view=live');
    await np.locator('.aw-cin').getByText(/网页版不公开原生全场/).waitFor();
    await locate(np, 'hs', 24, 11);
    assert.ok((await np.getByRole('dialog', { name: '单元详情' }).innerText()).includes('未采集'));
    check('engineering_negative_missing_native_data_explicit'); await missing.close();
    assert.deepEqual(report.errors, []);
    check('no_runtime_page_errors');
  } finally { await browser.close(); }
};
main().then(() => { report.passed = true; }).catch(e => { report.passed = false; report.failure = e.stack; console.error(e.stack); process.exitCode = 1; }).finally(() => fs.writeFileSync(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2)));
