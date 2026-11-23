'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {checkedDir, checkWrite} = require('../../frontend/electron/storagePolicy');

async function verify() {
  const root = path.resolve(__dirname, '../..');
  const preview = path.join(root, 'research/glm5/docs/previews/encoding-workbench.html');
  const out = path.join(__dirname, 'result/free_model_design');
  checkWrite(out, 280 * 2 ** 20, '模型分析设计稿浏览器与截图');
  checkedDir(out, 24 * 2 ** 20, '模型分析三视图及回执');
  const scratch = checkedDir(path.join(root, 'tests/glm5_temp'), 256 * 2 ** 20, '浏览器临时配置');
  const profile = fs.mkdtempSync(path.join(scratch, 'free-model-design-'));
  for (const key of ['TEMP', 'TMP', 'TMPDIR']) process.env[key] = profile;
  const report = {passed: false, checks: [], screenshots: [], http: [], errors: []};
  let browser;
  try {
    const {chromium} = require(process.env.PLAYWRIGHT_CORE ||
      'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
    const executablePath = process.env.CHROME_PATH || path.join(os.homedir(),
      'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
    browser = await chromium.launch({headless: true, executablePath});
    report.browser = browser.version();
    const context = await browser.newContext({viewport: {width: 1440, height: 980}, locale: 'zh-CN'});
    await context.route(/^https?:/, r => { report.http.push(r.request().url()); return r.abort(); });
    const page = await context.newPage();
    page.setDefaultTimeout(6500);
    page.on('pageerror', e => report.errors.push(e.message));
    page.on('requestfailed', r => { if (r.url().startsWith('file:')) report.errors.push(r.url()); });
    const el = id => page.locator('#' + id);
    const panel = id => page.locator('[data-window="' + id + '"]');
    const launch = id => page.locator('[data-open-window="' + id + '"]:visible').first().click();
    const settle = () => page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    const go = async address => {
      await el('address').fill(address);
      await el('address').press('Enter');
      await settle();
    };
    async function check(name, run) {
      try { await run(); report.checks.push(name); }
      catch (e) { e.message = name + ' — ' + e.message; throw e; }
    }
    async function fits() {
      await settle();
      const [content, width] = await page.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
      assert.ok(content <= width + 1, '横向溢出 ' + content + ' / ' + width);
    }
    async function shot(name) {
      await fits();
      await page.screenshot({path: path.join(out, name + '.png'), animations: 'disabled'});
      report.screenshots.push(name + '.png');
    }
    await page.goto(pathToFileURL(preview).href);
    await page.evaluate(() => document.fonts.ready);
    await el('space-canvas').waitFor();
    await page.evaluate(() => { window.qaCanvas = document.querySelector('#space-canvas'); });

    await check('默认 Qwen3-14B 与 B26/H36；桌面工作区及替换模型入口', async () => {
      assert.match(await el('model-select').inputValue(), /Qwen3-14B/i);
      assert.match(await el('selection-name').innerText(), /B26\s*\/\s*H36/i);
      const options = await el('model-select').locator('option').allTextContents();
      assert.ok(options.some(s => /Qwen3-4B/i.test(s)) && options.some(s => /Qwen3-14B/i.test(s)));
      report.models = options;
      assert.equal(await page.locator('.free-window:visible').count(), 1);
      await shot('desktop');
    });
    await check('显示设置作用于同一画布，可点击模型顶层切换原生地址', async () => {
      await launch('display');
      const pixels = () => page.evaluate(() => qaCanvas.toDataURL());
      const before = await pixels();
      await el('show-grid').uncheck();
      await settle();
      assert.notEqual(await pixels(), before);
      await el('show-grid').check();
      await settle();
      assert.equal(await pixels(), before);
      await launch('locator');
      const box = await el('space-canvas').boundingBox();
      await page.mouse.click(box.x + box.width * .325, box.y + box.height * .22);
      assert.match(await el('selection-name').innerText(), /B39\s*\/\s*H36/i);
      await go('B26/H36');
    });
    await check('同一画布绘出一个 Head 的全部 128 个通道，可高亮 D127 并在全屏读数', async () => {
      const original = await el('model-select').inputValue();
      await el('model-select').selectOption('Qwen3-4B');
      await go('B26/H12');
      // Count actual filled top faces for the scene, and record the label rendered beside its selected cell.
      await page.evaluate(() => {
        const proto = CanvasRenderingContext2D.prototype;
        const originals = {clearRect: proto.clearRect, fill: proto.fill, fillText: proto.fillText};
        window.qaPaint = {faces: {}, labels: []};
        proto.clearRect = function (...args) {
          if (this.canvas === qaCanvas) window.qaPaint = {faces: {}, labels: []};
          return originals.clearRect.apply(this, args);
        };
        proto.fill = function (...args) {
          if (this.canvas === qaCanvas) qaPaint.faces[this.fillStyle] = (qaPaint.faces[this.fillStyle] || 0) + 1;
          return originals.fill.apply(this, args);
        };
        proto.fillText = function (text, ...args) {
          if (this.canvas === qaCanvas) qaPaint.labels.push(text);
          return originals.fillText.call(this, text, ...args);
        };
        window.qaRestorePainter = () => Object.assign(proto, originals);
      });
      await el('channel-view').click();
      await go('B26/H12/D127');
      const label = await el('space-canvas').getAttribute('aria-label');
      assert.match(label, /128/);
      assert.match(label, /通道/);
      assert.match(label, /D127/);
      assert.match(await el('selection-name').innerText(), /B26\/H12\/D127/);
      const painted = await page.evaluate(() => window.qaPaint);
      assert.equal(painted.faces['#dbe8f9'], 127);
      assert.equal(painted.faces['#4380ee'], 1);
      assert.ok(painted.labels.includes('D127'));
      await page.evaluate(() => qaRestorePainter());
      await el('view-screen').click();
      await page.waitForFunction(() => !!document.fullscreenElement);
      await launch('readout');
      assert.ok(await panel('readout').isVisible());
      assert.equal(await page.evaluate(() => qaCanvas === document.querySelector('#space-canvas')), true);
      assert.match(await el('space-canvas').getAttribute('aria-label'), /D127/);
      await panel('locator').locator('[data-fw-action="close"]').click();
      const readoutBar = await panel('readout').locator('.window-bar').boundingBox();
      await page.mouse.move(readoutBar.x + 50, readoutBar.y + 15);
      await page.mouse.down();
      await page.mouse.move(68, readoutBar.y + 15);
      await page.mouse.up();
      const readoutGrip = await panel('readout').locator('.fw-resize').boundingBox();
      await page.mouse.move(readoutGrip.x + 9, readoutGrip.y + 9);
      await page.mouse.down();
      await page.mouse.move(readoutGrip.x + 35, readoutGrip.y + 90);
      await page.mouse.up();
      await shot('channels');
      await panel('readout').locator('[data-fw-action="close"]').click();
      await el('view-screen').click();
      await page.waitForFunction(() => !document.fullscreenElement);
      await launch('locator');
      await go('B0/U0');
      assert.match(await el('space-canvas').getAttribute('aria-label'), /U0/);
      assert.equal(await el('channel-view').isVisible() && await el('channel-view').isEnabled(), false);
      await go('B26/H12');
      await launch('display');
      await el('head-detail').selectOption('heads');
      await settle();
      assert.match(await el('space-canvas').getAttribute('aria-label'), /H12/);
      assert.doesNotMatch(await el('space-canvas').getAttribute('aria-label'), /128.*通道|通道.*128/);
      await el('head-detail').selectOption('channels');
      await settle();
      assert.match(await el('space-canvas').getAttribute('aria-label'), /通道/);
      await el('head-detail').selectOption('heads');
      await launch('locator');
      await el('model-select').selectOption(original);
      await go('B26/H36');
    });
    await check('Head 历史读数可以计算六组注意力熵', async () => {
      await launch('tools');
      const tool = panel('tools');
      await tool.locator('[data-fmt="tool"]').selectOption('attention');
      await tool.locator('[data-fmt-run]').click();
      await tool.getByText('六组熵', {exact: true}).waitFor();
      assert.match(await tool.innerText(), /其余 token/);
      const reference = await page.evaluate(() => {
        const id = document.querySelector('[data-window="tools"] [data-fmt="a"]').value;
        const obs = FreeModelData.fields('Qwen3-14B', 'head', 26, 36).find(f => f.id === id);
        const weights = obs.rows.map(r => r.mass);
        weights.push(1 - weights.reduce((a, b) => a + b, 0));
        return -weights.filter(x => x > 0).reduce((a, b) => a + b * Math.log2(b), 0);
      });
      const rendered = parseFloat(await tool.locator('.fmt-metrics dd').first().innerText());
      assert.ok(Math.abs(reference - rendered) < 1e-5);
    });
    await check('模型切换后按新架构定位，原有 Head 读数不跨模型复用', async () => {
      const models = el('model-select');
      const other = await models.locator('option').filter({hasText: /Qwen3-4B/i}).first().getAttribute('value');
      await models.selectOption(other);
      await launch('locator');
      await go('B0/H0');
      assert.match(await el('selection-name').innerText(), /B0\s*\/\s*H0/i);
      await launch('tools');
      await panel('tools').locator('[data-fmt="tool"]').selectOption('attention');
      assert.equal(await panel('tools').getByText('六组熵', {exact: true}).count(), 0);
      const qwen = await models.locator('option').filter({hasText: /Qwen3-14B/i}).first().getAttribute('value');
      await models.selectOption(qwen);
      await launch('locator');
      await go('B35/U9727');
      assert.match(await el('selection-name').innerText(), /B35\s*\/\s*U9727/i);
      await launch('readout');
      assert.ok(await panel('readout').isVisible());
      await launch('locator');
      await go('hs[0]/D0');
      assert.match(await el('selection-name').innerText(), /hs\[0\]\s*\/\s*D0/i);
      await go('B26/H36');
      assert.match(await el('selection-name').innerText(), /B26\s*\/\s*H36/i);
    });
    await check('GLM 的 gate 与 up 可追到融合矩阵的不同行，权重可读，缺激活有说明', async () => {
      const models = el('model-select'), original = await models.inputValue();
      const glm = await models.locator('option').filter({hasText: /glm/i}).first().getAttribute('value');
      await models.selectOption(glm);
      const width = await page.evaluate(id => FreeModelData.model(id).ffn, glm);
      const object = async () => { await launch('locator'); await go('B0/U3'); };
      await object();
      await launch('readout');
      assert.match(await panel('readout').innerText(), /缺少|暂无|尚无|未采集|未归档|没有/);
      await object();
      await el('related').getByRole('button', {name: /gate/i}).click();
      const fused = await el('tensor-select').inputValue();
      assert.match(fused, /gate_up|dense_h_to_4h/i);
      assert.equal(await el('weight-row').inputValue(), '3');
      assert.equal(await el('weight-col').inputValue(), '0');
      await el('weight-table').getByRole('cell').first().waitFor();
      assert.match(await el('weight-table').innerText(), /[-+]?\d+\.\d+/);
      await object();
      await el('related').getByRole('button', {name: /^up|上投影/i}).click();
      assert.equal(await el('tensor-select').inputValue(), fused);
      assert.equal(await el('weight-row').inputValue(), String(width + 3));
      await models.selectOption(original);
      await launch('locator');
      await go('B26/H36');
      assert.match(await el('selection-name').innerText(), /B26\s*\/\s*H36/i);
      report.glm = {name: glm, fused, gateRow: 3, upRow: width + 3};
    });
    await check('参数目录、原生 BF16 切片与权重工具使用同一选区', async () => {
      await launch('parameters');
      const sample = await page.evaluate(() => {
        const model = document.querySelector('#model-select').value;
        const available = [...document.querySelector('#tensor-select').options].map(o => o.value);
        const t = available.map(n => FreeModelData.tensor(model, n)).filter(Boolean)
          .find(t => t.name.includes('q_proj') && t.samples.some(a => a[2] >= 8 && a[3] >= 8));
        if (!t) return null;
        const [r, c] = t.samples.find(a => a[2] >= 8 && a[3] >= 8);
        return {name: t.name, r, c};
      });
      assert.ok(sample, '当前目录中需要一块可核对的二维权重区域');
      await el('tensor-select').selectOption(sample.name);
      await el('weight-row').fill(String(sample.r));
      await el('weight-col').fill(String(sample.c));
      await el('weight-col').press('Enter');
      await el('weight-table').locator('td').first().waitFor();
      assert.match(await el('weight-table').innerText(), /[-+]?\d+\.\d+/);
      await el('analyze-weight').click();
      const tool = panel('tools');
      await tool.locator('[data-fmt-run]').click();
      await tool.getByText('最大奇异值', {exact: true}).waitFor();
      assert.ok((await tool.innerText()).includes(sample.name));
      report.weightSample = sample;
    });
    await check('导入非法 JSON 显示失败且原有数据入口仍可用', async () => {
      await launch('evidence');
      const file = el('experiment-field');
      await file.setInputFiles({name: 'invalid-field.json', mimeType: 'application/json', buffer: Buffer.from('{')});
      await page.waitForFunction(() => document.querySelector('#field-load-note')?.textContent?.match(/错误|失败|无效|JSON|无法|解析/i));
      await launch('locator');
      await go('B26/H36');
    });
    await check('显式标注的工程输入可读取，切换快照拒绝错配，移除后恢复归档', async () => {
      await launch('evidence');
      const fixture = await page.evaluate(() => {
        const model = document.querySelector('#model-select').value, meta = FreeModelData.model(model);
        return {schema: 'ai2050.activation-vector.v1', model, snapshot: meta.snapshot, kind: 'head',
          block: 26, head: 36, measure: 'head_z', boundary: 'Z · o_proj 输入', position: 1, tokenId: 20,
          run: 'ui-test', caseId: 'engineering-fixture', values: Array.from({length: meta.headDim}, (_, i) => i)};
      });
      const upload = value => el('experiment-field').setInputFiles({name: 'engineering-fixture.json',
        mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(value))});
      await upload({...fixture, snapshot: 'engineering-mismatched-snapshot'});
      await page.waitForFunction(() => /快照/.test(document.querySelector('#field-load-note').textContent));
      await upload(fixture);
      await el('clear-field').waitFor();
      await launch('tools');
      await panel('tools').locator('[data-fmt="tool"]').selectOption('summary');
      await panel('tools').locator('[data-fmt-run]').click();
      await panel('tools').getByText('63.5', {exact: true}).waitFor();
      await launch('evidence');
      await el('clear-field').click();
      await launch('locator');
    });
    await check('原生全屏共享同一画布，两个窗口同时观察，退出回原地址', async () => {
      await el('view-screen').click();
      await page.waitForFunction(() => !!document.fullscreenElement && document.querySelector('main.fw-floating'));
      for (const selector of ['.desk-toolbar', '.tool-dock']) {
        const bar = page.locator(selector);
        const before = await bar.boundingBox();
        await bar.locator('.palette-grip').focus();
        await page.keyboard.press('ArrowRight');
        const after = await bar.boundingBox();
        assert.ok(Math.abs(after.x - before.x - 12) <= 1);
      }
      await el('layout-reset').click();
      const models = el('model-select');
      const original = await models.inputValue();
      const small = await models.locator('option').filter({hasText: /Qwen3-4B/i}).getAttribute('value');
      await models.selectOption(small);
      assert.equal(await models.inputValue(), small);
      await models.selectOption(original);
      await launch('locator');
      await go('B26/H36');
      await launch('tools');
      await panel('tools').locator('[data-fmt="tool"]').selectOption('attention');
      await panel('tools').locator('[data-fmt-run]').click();
      await panel('tools').getByText('六组熵', {exact: true}).waitFor();
      assert.ok(await panel('locator').isVisible());
      assert.ok(await panel('tools').isVisible());
      assert.equal(await page.evaluate(() => qaCanvas === document.querySelector('#space-canvas')), true);
      await shot('fullscreen');
      await el('view-screen').click();
      await page.waitForFunction(() => !document.fullscreenElement && !document.querySelector('main.fw-floating'));
      assert.equal(await page.locator('.free-window:visible').count(), 1);
      assert.match(await el('selection-name').innerText(), /B26\s*\/\s*H36/i);
      assert.equal(await page.evaluate(() => qaCanvas === document.querySelector('#space-canvas')), true);
    });
    await check('手机竖屏可定位模型、打开工具，页面无横向溢出', async () => {
      await page.setViewportSize({width: 390, height: 844});
      await launch('locator');
      await go('B26/H36');
      await page.evaluate(() => scrollTo(0, 0));
      await shot('mobile');
      await launch('tools');
      await fits();
      assert.equal(await panel('tools').locator('[data-fmt="tool"]').isVisible(), true);
      await el('view-screen').click();
      await page.waitForFunction(() => !!document.fullscreenElement);
      for (const id of ['locator', 'readout', 'parameters', 'tools', 'evidence', 'display']) {
        await launch(id);
        const box = await panel(id).boundingBox();
        assert.ok(box && box.x >= 0 && box.x + box.width <= 391, '窗口超出手机宽度 ' + id);
      }
      await fits();
      await el('view-screen').click();
    });
    await check('浏览器限制原生全屏时可用页面全屏，Escape 恢复停靠', async () => {
      await page.waitForFunction(() => !document.fullscreenElement);
      await page.evaluate(() => {
        document.querySelector('#free-workbench').requestFullscreen = async () => {
          throw new DOMException('浏览器验收所设置的全屏限制', 'NotAllowedError');
        };
      });
      await el('view-screen').click();
      await page.waitForFunction(() => !document.fullscreenElement && document.querySelector('main.fw-floating'));
      assert.equal(await page.locator('main').evaluate(n => getComputedStyle(n).position), 'fixed');
      await launch('parameters');
      assert.ok(await panel('parameters').isVisible());
      await page.keyboard.press('Escape');
      await page.waitForFunction(() => !document.querySelector('main.fw-floating'));
      assert.ok(await panel('parameters').isVisible());
    });
    await check('地址栏随选择更新，强制刷新后恢复模型、原生地址与活动窗口', async () => {
      await el('model-select').selectOption('GLM-4-9B');
      await launch('locator');
      await go('B2/U27');
      await launch('tools');
      const route = new URLSearchParams(new URL(page.url()).hash.slice(1));
      assert.deepEqual(Object.fromEntries(route), {model: 'GLM-4-9B', at: 'B2/U27', panel: 'tools'});
      const cdp = await context.newCDPSession(page);
      await Promise.all([page.waitForEvent('load'), cdp.send('Page.reload', {ignoreCache: true})]);
      await cdp.detach();
      await panel('tools').waitFor();
      assert.equal(await el('model-select').inputValue(), 'GLM-4-9B');
      assert.equal(await el('selection-name').innerText(), 'B2/U27');
      assert.equal(await page.locator('.free-window:visible').count(), 1);
    });
    assert.deepEqual(report.http, []);
    assert.deepEqual(report.errors, []);
    report.passed = true;
  } catch (e) {
    report.failure = e.message;
    throw e;
  } finally {
    if (browser) await browser.close();
    fs.writeFileSync(path.join(out, 'acceptance.json'), JSON.stringify(report, null, 2) + '\n');
    const actual = fs.realpathSync.native(profile);
    assert.equal(path.dirname(actual), fs.realpathSync.native(scratch));
    assert.ok(path.basename(actual).startsWith('free-model-design-'));
    fs.rmSync(actual, {recursive: true, force: true, maxRetries: 4, retryDelay: 150});
  }
  console.log(report.checks.join('\n'));
}
verify().catch(e => { console.error(e); process.exitCode = 1; });
