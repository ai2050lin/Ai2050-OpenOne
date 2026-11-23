'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {checkedDir, checkWrite} = require('../../frontend/electron/storagePolicy');
const archive = require('../../frontend/src/components/v8/data/graphObservationExample.json');
const cues = require('./result/qwen14_client_v2/native/B36-cues.json').units;
const b23Dir = path.join(__dirname, 'result/goodfire_middle_20261113_v2/qwen14');
const b23Cues = require(path.join(b23Dir, 'B23-cues.json')).units;

const root = path.resolve(__dirname, '../..');
const preview = path.join(root, 'research/glm5/docs/previews/client-distributed-testing.html');
const entry = pathToFileURL(preview);
entry.hash = '?wb=1&observeMode=live&sblock=24&smodule=mlp&sindex=7708&lens=spatial&view=graph&gview=structure&observeTab=run&observePlugin=object';
const out = path.join(__dirname, 'result/neuron_study_preview');
const raw = Buffer.from(archive.mlp, 'base64');

function expectedValue(site, unit) {
  const offset = 2 * (site * archive.mlpShape[1] + unit);
  const decoded = Buffer.alloc(4);
  decoded.writeUInt16LE(raw.readUInt16LE(offset), 2);
  return decoded.readFloatLE();
}

function corpusValue(document, position, unit) {
  const fd = fs.openSync(path.join(b23Dir, 'B23-A.npy'), 'r');
  try {
    const header = Buffer.alloc(128);
    assert.equal(fs.readSync(fd, header, 0, header.length, 0), header.length);
    assert.equal(header.toString('ascii', 1, 6), 'NUMPY');
    assert.equal(header[6], 1);
    const headerSize = header.readUInt16LE(8);
    assert.match(header.toString('ascii', 10), /'descr': '<u2', 'fortran_order': False, 'shape': \(1024, 8, 17408\)/);
    const offset = 10 + headerSize + 2 * ((document * 8 + position) * 17408 + unit);
    const value = Buffer.alloc(4);
    assert.equal(fs.readSync(fd, value, 2, 2, offset), 2);
    return value.readFloatLE();
  } finally {
    fs.closeSync(fd);
  }
}

async function inFirstViewport(locator) {
  await locator.waitFor();
  const visible = await locator.evaluate(node => {
    const rect = node.getBoundingClientRect();
    const left = Math.max(0, rect.left), right = Math.min(innerWidth, rect.right);
    const top = Math.max(0, rect.top), bottom = Math.min(innerHeight, rect.bottom);
    if (right <= left || bottom - top < 10) return false;
    const hit = document.elementFromPoint((left + right) / 2, (top + bottom) / 2);
    return !!hit && node.contains(hit);
  });
  assert.ok(visible, '首屏被遮挡或需要滚动才能到达 ' + locator);
}

async function chooseCell(canvas, unit) {
  const bounds = await canvas.boundingBox();
  assert.ok(bounds, '应能通过鼠标选取热图中的单元');
  const columns = Number(await canvas.getAttribute('width')), rows = Number(await canvas.getAttribute('height'));
  await canvas.click({position: {
    x: (unit % columns + .5) * bounds.width / columns,
    y: (Math.floor(unit / columns) + .5) * bounds.height / rows,
  }});
}

async function pixelAt(canvas, index, positive) {
  const rgb = await canvas.evaluate((node, i) => Array.from(
    node.getContext('2d').getImageData(i % node.width, Math.floor(i / node.width), 1, 1).data
  ), index);
  assert.equal(rgb[3], 255);
  assert.ok(positive ? rgb[2] > rgb[0] : rgb[0] > rgb[2], '可见热图颜色与原始激活的符号不符 ' + rgb);
  return rgb;
}

async function assertSelection(study, report, block, unit, expected) {
  const reading = study.locator('output.nhm-value[data-index="' + unit + '"]');
  await reading.waitFor();
  const value = await reading.getAttribute('data-value');
  assert.ok(value !== null && value !== '', '缺值不能视为零');
  assert.equal(Number(value), expected, '热图与归档张量中的原始值不符');
  assert.equal(Number(await reading.innerText()), expected);
  const profile = study.getByTestId('goodfire-profile');
  await profile.getByRole('heading', {name: new RegExp('B' + block + '/U' + unit + '\\b')}).waitFor();
  const text = await profile.innerText();
  if ([23, 36].includes(block)) {
    const record = (block === 23 ? b23Cues : cues)[unit];
    for (const keyword of record.positive_keywords.slice(0, 3)) assert.ok(text.includes(keyword), '缺少归档候选词 ' + keyword);
    const holdout = record.repeat.holdout;
    assert.ok(text.includes(holdout.hits + '/' + holdout.n), '缺少归档留出集命中数');
  }
  report.values.push({block, unit, value: Number(value), profile: await profile.getAttribute('data-state')});
  return text;
}

async function runJourneys(page, report, shot) {
  const study = page.getByRole('region', {name: '神经元研究', exact: true});
  const canvas = study.locator('canvas.nhm-raster'), site = study.locator('#study-site');
  const address = study.locator('#study-address'), profile = study.getByTestId('goodfire-profile');
  const go = async unitAddress => {
    await address.fill(unitAddress);
    await study.locator('#study-locate').click();
  };
  const byName = (scope, name) => scope.getByRole('button', {name, exact: true});
  await page.goto(entry.href);
  await inFirstViewport(study.getByRole('heading', {name: '神经元热力图'}));
  await inFirstViewport(canvas);
  assert.equal(await canvas.evaluate(node => node.width * node.height), 17408, '应完整绘制每个原生单元');
  await inFirstViewport(profile.getByRole('heading', {name: /B23\/U7708.*Goodfire/}));
  assert.equal(await profile.getAttribute('data-state'), 'atlas');
  const locations = await site.locator('option').allTextContents();
  assert.equal(locations.length, 2);
  const first = locations.findIndex(label => /945/.test(label));
  const second = locations.findIndex(label => /598/.test(label));
  assert.ok(first >= 0 && second >= 0 && first !== second, '应显示两个语料位置的 token 索引');
  await site.selectOption({index: first});
  const profileText = await assertSelection(study, report, 23, 7708, corpusValue(952, 6, 7708));
  assert.match(profileText, /1\s*\/\s*4/);
  assert.match(profileText, /wikipedia-zh:18501/);
  assert.match(profileText, /wikipedia-en:4073941/);
  await pixelAt(canvas, 7708, true);
  report.layout = await study.evaluate(node => {
    const selectors = ['.ns-study', '.ns-title', '.ns-object', '.ns-columns', '.ns-card'];
    return Object.fromEntries(selectors.map(selector => {
      const element = node.matches(selector) ? node : node.querySelector(selector);
      const style = getComputedStyle(element);
      return [selector, {padding: style.padding, margin: style.margin}];
    }));
  });
  assert.ok(parseFloat(report.layout['.ns-card'].padding) >= 12, '项目全局重置覆盖了证据卡内边距');
  await shot('first-look');
  report.checked.push('原实时深链首屏可直接读到 B23/U7708 热图和独立语料报告，数值与原始 NPY 一致');
  await site.selectOption({index: second});
  await assertSelection(study, report, 23, 7708, corpusValue(897, 1, 7708));
  await pixelAt(canvas, 7708, false);
  await chooseCell(canvas, 7709);
  await assertSelection(study, report, 23, 7709, corpusValue(897, 1, 7709));
  await study.getByText(/沿用 U7708 选出的两个文本位置/).waitFor();
  await go('B23/U17408');
  await study.getByRole('alert').waitFor();
  await assertSelection(study, report, 23, 7709, corpusValue(897, 1, 7709));
  await canvas.press('ArrowLeft');
  await assertSelection(study, report, 23, 7708, corpusValue(897, 1, 7708));
  report.checked.push('同一色标下切 token、鼠标/键盘选格均同步报告；相邻单元注明沿用读出位置，越界地址保留原证据');
  await page.reload();
  await assertSelection(study, report, 23, 7708, corpusValue(897, 1, 7708));
  assert.equal(await site.inputValue(), 'negative');
  report.checked.push('刷新后仍打开同一模型、同一单元和同一个负端语料位置');

  await go('B1/U7708');
  await profile.getByRole('heading', {name: /B1\/U7708/}).waitFor();
  assert.equal(await profile.getAttribute('data-state'), 'missing');
  assert.equal(await canvas.count(), 0, '缺件位置不能显示以前的热图');
  await study.getByText('暂无可显示的热图', {exact: true}).waitFor();
  await shot('missing');
  await study.getByRole('button', {name: /B36\/U2429/}).first().click();
  await assertSelection(study, report, 36, 2429, expectedValue(0, 2429));
  await chooseCell(canvas, 2430);
  await assertSelection(study, report, 36, 2430, expectedValue(0, 2430));
  await study.getByRole('button', {name: /返回原对象/}).click();
  await profile.getByRole('heading', {name: /B1\/U7708/}).waitFor();
  assert.equal(await profile.getAttribute('data-state'), 'missing');
  assert.equal(await canvas.count(), 0);
  report.checked.push('缺件位置没有图案或他人报告，一键看跨题归档、改选相邻单元后仍能返回原对象');

  await checkViews({page, entry, study, profile, byName, go, shot, report});
}

async function checkViews({page, entry, study, profile, byName, go, shot, report}) {
  await go('B23/U7708');
  await byName(study, '实时输入').click();
  const bench = page.locator('.ow-graph'), scene = bench.locator('.aw-space');
  await bench.getByRole('region', {name: '实时模式工作区'}).waitFor();
  const modes = scene.getByRole('group', {name: '观察模式'});
  await byName(modes, '特征分析').click();
  await byName(bench, '神经元研究').click();
  await assertSelection(study, report, 23, 7708, corpusValue(952, 6, 7708));
  await study.getByRole('button', {name: /全屏/}).click();
  const canvas = study.locator('canvas.nhm-raster');
  const fullscreenSize = await study.evaluate(node => {
    const bounds = node.getBoundingClientRect();
    return {width: bounds.width, height: bounds.height, viewport: [innerWidth, innerHeight],
      fullscreen: !!document.fullscreenElement || (bounds.left < 3 && bounds.top < 3 && bounds.right >= innerWidth - 3 && bounds.bottom >= innerHeight - 3)};
  });
  assert.ok(fullscreenSize.fullscreen, '研究区应贴合全屏区域');
  await chooseCell(canvas, 7710);
  await assertSelection(study, report, 23, 7710, corpusValue(952, 6, 7710));
  await shot('fullscreen');
  await byName(study, '退出全屏').click();
  await byName(study, '全屏观察').waitFor();
  report.checked.push('实时输入、特征模式和研究视图带入同一对象，全屏可继续点选并读取相应语料报告');
  await byName(study, '全屏观察').click();
  await study.locator('#study-site').selectOption('negative');
  await byName(study, '实时输入').click();
  await bench.locator('.aw-space.max').waitFor();
  await bench.getByRole('region', {name: '实时模式工作区'}).waitFor();
  await byName(modes, '特征分析').click();
  await bench.locator('.aw-space.max').waitFor();
  await byName(bench, '神经元研究').click();
  await byName(study, '退出全屏').waitFor();
  await assertSelection(study, report, 23, 7710, corpusValue(897, 1, 7710));
  assert.equal(await study.locator('#study-site').inputValue(), 'negative');
  await byName(study, '退出全屏').click();
  report.checked.push('研究页全屏与原 3D 全屏往返，可切实时/特征，再回到相同负端位置与单元');

  const old = new URL(entry.href);
  old.hash += '&studyPane=tools';
  await page.goto('about:blank');
  await page.goto(old.href);
  await bench.locator('.aw-space').waitFor();
  await byName(bench, '神经元研究').click();
  await assertSelection(study, report, 23, 7708, corpusValue(952, 6, 7708));
  report.checked.push('旧 3D 工具页有直接进入神经元研究的入口并带入原对象');

  const sample = new URL(entry.href);
  sample.hash = '?wb=1&view=graph&lens=spatial&observeMode=features&studyPane=study&mdl=Qwen3-14B&sblock=37&smodule=mlp&sindex=2429';
  await page.goto('about:blank');
  await page.goto(sample.href);
  await assertSelection(study, report, 36, 2429, expectedValue(0, 2429));
  await shot('archive-example');
  report.checked.push('显式归档深链定位 B36/U2429 并读到真实完整矩阵和相同编号的报告');

  await study.getByRole('button', {name: /B30\/U9517/}).first().click();
  // Row 2, unit 9517 of the original BF16 block_30.npz activation
  // in live_traces/fcf90469ce80cc33091526080da6c8b11b03da3d9c36a9b9b3031d31f4b1876e.
  const fourB = await assertSelection(study, report, 30, 9517, -0.3125);
  assert.equal(await profile.getAttribute('data-state'), 'probe');
  assert.match(fourB, /香港/);
  assert.match(fourB, /SiLU/);
  assert.match(fourB, /gate_proj/);
  report.checked.push('跨模型的 4B 报告显示 gate 语料测量，木盒题热图显示门控乘积，与原始块中的 −0.3125 一致');
  const model = study.getByRole('combobox', {name: '研究模型'});
  assert.equal(await model.inputValue(), 'Qwen3-4B');
  await model.selectOption('Qwen3-14B');
  await assertSelection(study, report, 36, 2429, expectedValue(0, 2429));
  await model.selectOption('Qwen3-4B');
  await assertSelection(study, report, 30, 9517, -0.3125);
  report.checked.push('研究模型选择框切换 4B/14B 时分别恢复上次对象、热图与报告');

  await page.setViewportSize({width: 390, height: 844});
  await page.goto('about:blank');
  await page.goto(entry.href);
  await inFirstViewport(study.getByRole('heading', {name: '神经元热力图'}));
  await inFirstViewport(canvas);
  await shot('narrow');
  const size = await page.evaluate(() => ({body: document.documentElement.scrollWidth, viewport: innerWidth}));
  assert.ok(size.body <= size.viewport + 1, '窄屏出现横向溢出 ' + JSON.stringify(size));
  await study.getByRole('navigation', {name: '查看分析结果'}).getByRole('button', {name: 'Goodfire 报告'}).click();
  await inFirstViewport(profile.getByRole('heading', {name: /B23\/U7708.*Goodfire/}));
  report.checked.push('390×844 窄屏可触达热图和相同单元的 Goodfire 报告且没有横向溢出');
}

async function run() {
  checkWrite(out, 280 * 2 ** 20, '神经元观察静态稿离线验收');
  checkedDir(out, 24 * 2 ** 20, '神经元观察验收截图与回执');
  const temporary = checkedDir(path.join(root, 'tests/glm5_temp'), 256 * 2 ** 20, '浏览器临时数据');
  const scratch = fs.mkdtempSync(path.join(temporary, 'neuron-study-'));
  for (const name of ['TEMP', 'TMP', 'TMPDIR']) process.env[name] = scratch;
  const report = {passed: false, entry: entry.hash, checked: [], screenshots: [], values: [], external: [], errors: []};
  report.sources = {
    corpus: {path: path.relative(root, path.join(b23Dir, 'B23-A.npy')), shape: [1024, 8, 17408], positions: [945, 598]},
    question: {run: archive.run, snapshot: archive.snapshot, case: archive.caseId, shape: archive.mlpShape},
  };
  let browser;
  try {
    const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
    const executablePath = process.env.CHROME_PATH || path.join(os.homedir(), 'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
    browser = await chromium.launch({executablePath, headless: true});
    report.browser = browser.version();
    const context = await browser.newContext({locale: 'zh-CN', viewport: {width: 1440, height: 900}});
    await context.route(/^https?:/, route => {
      report.external.push(route.request().url());
      return route.abort();
    });
    const page = await context.newPage();
    page.setDefaultTimeout(6500);
    page.on('pageerror', err => report.errors.push(err.stack ?? err.message));
    page.on('requestfailed', req => {
      if (req.url().startsWith('file:')) report.errors.push(req.url());
    });
    const shot = async name => {
      await page.screenshot({path: path.join(out, name + '.png'), animations: 'disabled'});
      report.screenshots.push(name + '.png');
    };
    await runJourneys(page, report, shot);
    assert.deepEqual(report.external, [], '离线页面请求了网络');
    assert.deepEqual(report.errors, [], '存在文件加载或页面异常');
    report.passed = true;
    console.log('已通过 ' + report.checked.length + ' 组静态稿验收');
  } catch (err) {
    report.failure = err.stack;
    throw err;
  } finally {
    await browser?.close();
    fs.writeFileSync(path.join(out, 'acceptance.json'), JSON.stringify(report, null, 2) + '\n');
    const target = fs.realpathSync.native(scratch);
    assert.equal(path.dirname(target), fs.realpathSync.native(temporary));
    assert.ok(path.basename(target).startsWith('neuron-study-'));
    fs.rmSync(target, {recursive: true, force: true, maxRetries: 5, retryDelay: 170});
  }
}

run().catch(err => {console.error(err); process.exitCode = 1;});
