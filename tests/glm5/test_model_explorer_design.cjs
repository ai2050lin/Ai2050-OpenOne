'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { checkedDir, checkWrite } = require('../../frontend/electron/storagePolicy.js');

const root = path.resolve(__dirname, '../..');
const preview = path.join(root, 'research/glm5/docs/previews/model-explorer-design.html');
const results = path.join(__dirname, 'result/model_explorer_design');
const scratch = path.join(root, 'tests/glm5_temp');
const modes = ['实时模式', '测试记录', '特征分析'];
const cueArchive = require('../../frontend/src/components/v8/threeCueEvidence.json').runs.find(run => run.id === 'clt_b26_three_cues_v3');
const mlpArchive = require('../../frontend/src/components/v8/data/graphObservationExample.json');

function archivedUnitValues(unit) {
  const bytes = Buffer.from(mlpArchive.mlp, 'base64');
  const fp32 = Buffer.alloc(4);
  return mlpArchive.positions.map((_, row) => {
    // The archive stores the high 16 bits of each float (bfloat16), in little endian.
    const offset = (row * mlpArchive.mlpShape[1] + unit) * 2;
    fp32.writeUInt16LE(bytes.readUInt16LE(offset), 2);
    return String(fp32.readFloatLE());
  });
}

async function verify() {
  assert.ok(fs.statSync(preview).isFile(), '静态稿缺失');
  checkWrite(results, 272 * 2 ** 20, '模型分析静态稿验收');
  checkedDir(results, 16 * 2 ** 20, '界面验收截图');
  const parent = checkedDir(scratch, 256 * 2 ** 20, '隔离浏览器缓存');
  const temp = fs.mkdtempSync(path.join(parent, 'model-explorer-design-'));
  for (const key of ['TEMP', 'TMP', 'TMPDIR']) process.env[key] = temp;
  let browser;
  const report = { passed: false, preview: path.basename(preview), checks: [], errors: [], network: [] };
  try {
    const { chromium } = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
    const executablePath = process.env.CHROME_PATH || path.join(os.homedir(),
      'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
    browser = await chromium.launch({ headless: true, executablePath });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 } });
    await context.route(/^https?:/, async route => {
      report.network.push(route.request().url());
      await route.abort();
    });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    page.setDefaultTimeout(9000);
    const button = name => page.getByRole('button', { name, exact: true });
    const analysis = page.locator('#analysis-panel');
    const record = page.locator('#record-detail');
    const screenshot = name => page.screenshot({ path: path.join(results, name + '.png'), animations: 'disabled' });
    const modeIs = async selected => {
      for (const name of modes) {
        const control = button(name);
        await control.waitFor({ state: 'visible' });
        assert.equal(await control.getAttribute('aria-pressed'), String(name === selected), name);
      }
    };
    const fitsViewport = async () => {
      const sizes = await page.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
      assert.ok(sizes[0] <= sizes[1] + 1, '页面横向溢出 ' + sizes.join(' / '));
    };

    await page.goto(pathToFileURL(preview).href);
    await modeIs('测试记录');
    await fitsViewport();
    await page.evaluate(() => document.fonts.ready);
    await screenshot('desktop');
    await button('B26/H36').click();
    await button('Head').click();
    await analysis.waitFor({ state: 'visible' });
    assert.match(await analysis.innerText(), /B26\s*\/\s*H36/i);
    await screenshot('head-analysis');
    const family = cueArchive.families[0];
    const picker = analysis.getByRole('combobox', { name: '测试条件' });
    async function verifyCase(label, id, output) {
      const sample = family.cases.find(item => item.id === id);
      assert.equal(sample.output, output);
      await picker.selectOption({ label });
      assert.ok((await analysis.innerText()).includes(sample.id));
      assert.equal(await analysis.locator('.reading code').innerText(), sample.output);
      assert.deepEqual(await analysis.locator('.barrow b').allTextContents(),
        sample.order.map(index => (100 * sample.mass[index]).toFixed(1) + '%'));
      await button('打开历史测试').click();
      await record.waitFor();
      assert.ok((await record.innerText()).includes(sample.id));
      await record.getByRole('heading', { name: '本题输出 ' + output, exact: true }).waitFor();
      await record.getByText('原始模型输入', { exact: true }).click();
      assert.ok((await record.innerText()).includes(sample.question));
      await record.getByText('运行与快照', { exact: true }).click();
      assert.ok((await record.innerText()).includes(cueArchive.id));
      assert.ok((await record.innerText()).includes(cueArchive.snapshot));
      await button('返回分析').click();
      assert.equal(await picker.locator('option:checked').innerText(), label);
    }
    await verifyCase('按姓名', 'confirm-000-00', 'wood');
    await verifyCase('按物理首行', 'confirm-000-02', 'silver');
    report.checks.push('两种条件的真实输出、五行注意力、原题、运行与快照和归档一致');
    await button('打开历史测试').click();
    await record.waitFor({ state: 'visible' });
    assert.match(await record.innerText(), /Phase\s*\d+/i);
    await screenshot('historical-record');
    await button('返回分析').click();
    await record.waitFor({ state: 'hidden' });
    assert.match(await analysis.innerText(), /B26\s*\/\s*H36/i);
    report.checks.push('B26/H36 定位、Head 分析、历史测试与返回');

    await button('全屏').click();
    await button('退出全屏').waitFor();
    for (const name of ['实时模式', '特征分析', '测试记录']) {
      await button(name).click();
      await modeIs(name);
      assert.ok(await button('退出全屏').isVisible());
    }
    await button('显示插件').click();
    for (const name of ['空间网格', '对象标注']) {
      const toggle = page.getByRole('checkbox', { name });
      await toggle.uncheck();
      assert.ok(!(await toggle.isChecked()));
      await toggle.check();
    }
    await button('显示插件').click();
    await button('Head').click();
    await verifyCase('按姓名', 'confirm-000-00', 'wood');
    assert.ok(await button('退出全屏').isVisible());
    await picker.selectOption({label:'按物理首行'});
    await screenshot('fullscreen');
    await button('退出全屏').click();
    await button('全屏').waitFor();
    await modeIs('测试记录');
    report.checks.push('全屏切换三种模式、显示插件勾选、Head 同题往返及退出');

    const model = page.getByRole('combobox', { name: '选择模型' });
    await model.selectOption('Qwen3-4B');
    const locator = page.getByRole('textbox', { name: '输入原生编号' });
    await locator.fill('B26/H36');
    await locator.press('Enter');
    assert.match(await page.locator('#loc-error').innerText(), /H0–H31/);
    assert.notEqual(await page.locator('#object-id').innerText(), 'B26/H36');
    assert.ok(await button('打开历史测试').isHidden());
    assert.ok(await picker.isHidden());
    await model.selectOption('Qwen3-14B');
    assert.equal(await page.locator('#object-id').innerText(), 'B26/H36');
    await picker.waitFor();
    report.checks.push('4B 越界 H36 无旧题证据，切回 14B 恢复已选对象');

    const values = archivedUnitValues(2429);
    const tasks = page.locator('#task-panel');
    await button('实时模式').click();
    assert.ok(await button('开始本机运行').isDisabled());
    await button('回放归档样例').click();
    const position = tasks.getByRole('combobox', {name:/回放位置/});
    for (const [i, p] of mlpArchive.positions.entries()) {
      await position.selectOption(String(i));
      assert.match(await tasks.locator('.replay').innerText(), new RegExp('token ' + p.index + '\\b'));
      assert.equal(await tasks.locator('.replay code').innerText(), values[i]);
    }
    await button('特征分析').click();
    await button('B36/U2429').click();
    assert.match(await tasks.innerText(), /Phase 2933/);
    assert.match(await tasks.innerText(), /B36\/U2429/);
    assert.deepEqual(await tasks.locator('.feature-values b').allTextContents(), values);
    await screenshot('feature');
    report.checks.push('实时模式归档回放及特征详情四处激活与 Phase 2933 的原生 BF16 记录一致');
    await button('测试记录').click();

    await page.setViewportSize({ width: 390, height: 844 });
    await fitsViewport();
    await button('B26/H36').click();
    await button('Head').click();
    await analysis.waitFor();
    await fitsViewport();
    await page.evaluate(() => window.scrollTo(0, 0));
    await screenshot('mobile');
    await button('打开历史测试').click();
    await record.waitFor();
    await fitsViewport();
    await button('返回分析').click();
    report.checks.push('390 像素下定位、阅读和返回，无页面横向溢出');
    assert.deepEqual(report.errors, []);
    assert.deepEqual(report.network, []);
    report.checks.push('本地文件离线打开，无外部请求或页面异常');
    report.passed = true;
  } catch (error) {
    report.failure = error.message;
    throw error;
  } finally {
    fs.writeFileSync(path.join(results, 'acceptance.json'), JSON.stringify(report, null, 2) + '\n');
    if (browser) await browser.close();
    const actual = fs.realpathSync.native(temp);
    assert.equal(path.dirname(actual), fs.realpathSync.native(parent));
    assert.ok(path.basename(actual).startsWith('model-explorer-design-'));
    fs.rmSync(actual, { recursive: true, force: true, maxRetries: 4, retryDelay: 150 });
  }
  console.log(JSON.stringify(report));
}

verify().catch(error => { console.error(error); process.exitCode = 1; });
