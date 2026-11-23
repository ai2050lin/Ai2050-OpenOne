'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {checkedDir} = require('../../frontend/electron/storagePolicy');

async function verify() {
  const root = path.resolve(__dirname, '../..');
  const source = path.join(root, 'research/glm5/docs/previews');
  const out = checkedDir(path.join(__dirname, 'result/free_model_design'), 4 * 2 ** 20, '窗口验收报告');
  const scratch = checkedDir(path.join(root, 'tests/glm5_temp'), 250 * 2 ** 20, '窗口验收浏览器');
  const profile = fs.mkdtempSync(path.join(scratch, 'free-model-windows-'));
  for (const key of ['TMP', 'TEMP', 'TMPDIR']) process.env[key] = profile;
  const {chromium} = require(process.env.PLAYWRIGHT_CORE ||
    'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
  const executablePath = process.env.CHROME_PATH || path.join(os.homedir(),
    'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
  const report = {passed: false, checks: [], pageErrors: []};
  let browser;
  try {
    browser = await chromium.launch({headless: true, executablePath});
    const page = await browser.newPage({viewport: {width: 1280, height: 800}});
    page.on('pageerror', e => report.pageErrors.push(e.message));
    const ids = ['locator', 'parameters', 'tools', 'evidence'];
    const launchers = ids.map(id => '<button data-open-window="' + id + '">' + id + '</button>').join('');
    const windows = ids.map(id => '<section class="free-window" data-window="' + id +
      '"><header class="window-bar"><h2>' + id + '</h2></header>' +
      '<div class="window-content"><input value="' + id + '"></div></section>').join('');
    await page.setContent('<style>*{box-sizing:border-box}body{margin:0}main{height:100vh;width:100%}</style>' +
      '<main id="free-workbench"><nav>' + launchers + '</nav><div id="free-windows">' + windows + '</div></main>');
    await page.addStyleTag({path: path.join(source, 'free-model-windows.css')});
    await page.addScriptTag({path: path.join(source, 'free-model-windows.js')});
    await page.evaluate(() => { window.wm = FreeWindowManager.create(document.querySelector('main')); });
    const panel = id => page.locator('[data-window="' + id + '"]');
    const launch = id => page.locator('[data-open-window="' + id + '"]').click();
    const flip = on => page.evaluate(x => wm.setFullscreen(x), on);
    const action = (id, type) => panel(id).locator('[data-fw-action="' + type + '"]').click();
    const geometry = id => panel(id).boundingBox();
    const count = () => page.locator('.free-window:visible').count();
    const frame = () => page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
    const check = async (name, fn) => { await fn(); report.checks.push(name); };

    await check('普通页面只停靠一个模块，并保留输入', async () => {
      assert.equal(await count(), 1);
      await launch('parameters');
      assert.equal(await panel('parameters').evaluate(el => getComputedStyle(el).position), 'static');
      await panel('parameters').locator('input').fill('row 27, col 95');
      await page.evaluate(() => { window.sameInput = document.querySelector('[data-window="parameters"] input'); });
    });
    await check('全屏多窗、拖动、边界和键盘调整', async () => {
      await flip(true);
      await launch('tools');
      assert.equal(await count(), 2);
      await launch('parameters');
      const bar = panel('parameters').locator('.window-bar');
      const start = await geometry('parameters');
      await page.mouse.move(start.x + 65, start.y + 16);
      await page.mouse.down();
      await page.mouse.move(start.x + 5, start.y - 36);
      await page.mouse.up();
      const moved = await geometry('parameters');
      assert.equal(Math.round(moved.x - start.x), -60);
      assert.equal(Math.round(moved.y - start.y), -52);
      await bar.focus();
      await page.keyboard.press('ArrowLeft');
      await page.keyboard.press('Shift+ArrowLeft');
      const key = await geometry('parameters');
      assert.equal(key.x, moved.x - 12);
      assert.equal(key.width, moved.width - 12);
      const grip = await panel('parameters').locator('.fw-resize').boundingBox();
      await page.mouse.move(grip.x + grip.width / 2, grip.y + grip.height / 2);
      await page.mouse.down();
      await page.mouse.move(grip.x + grip.width / 2 - 20, grip.y + grip.height / 2 + 25);
      await page.mouse.up();
      const resized = await geometry('parameters');
      assert.equal(resized.width, key.width - 20);
      assert.equal(resized.height, key.height + 25);
    });
    await check('置顶可取消，收起、最大化和还原均保留原矩形', async () => {
      const original = await geometry('parameters');
      await action('parameters', 'pin');
      await launch('tools');
      const z = id => panel(id).evaluate(el => Number(el.style.zIndex));
      assert.ok(await z('parameters') > await z('tools'));
      await action('parameters', 'pin');
      await action('parameters', 'collapse');
      assert.ok((await geometry('parameters')).height < 70);
      await action('parameters', 'collapse');
      assert.deepEqual(await geometry('parameters'), original);
      await action('parameters', 'maximize');
      assert.deepEqual(await geometry('parameters'), await page.locator('#free-windows').boundingBox());
      await action('parameters', 'maximize');
      assert.deepEqual(await geometry('parameters'), original);
    });
    await check('关闭后可从栏中重开，同一输入保留选中内容', async () => {
      await action('parameters', 'close');
      assert.equal(await count(), 1);
      await launch('parameters');
      assert.equal(await panel('parameters').locator('input').inputValue(), 'row 27, col 95');
      await flip(false);
      assert.equal(await count(), 1);
      assert.equal(await page.evaluate(() =>
        sameInput === document.querySelector('[data-window="parameters"]:not([hidden]) input')), true);
    });
    await check('小屏所有窗口可恢复、复位、重新打开，始终在屏内', async () => {
      await flip(true);
      for (const id of ids) await launch(id);
      await page.setViewportSize({width: 360, height: 590});
      await frame();
      const box = await page.locator('#free-windows').boundingBox();
      async function bounded() {
        for (const id of ids) {
          const win = await geometry(id);
          assert.ok(win.x >= box.x && win.y >= box.y);
          assert.ok(win.x + win.width <= box.x + box.width + 1 && win.y + win.height <= box.y + box.height + 1);
        }
      }
      await bounded();
      await page.evaluate(() => wm.reset());
      await bounded();
      await launch('evidence');
      await action('evidence', 'close');
      await launch('evidence');
      assert.equal(await count(), 4);
    });
    assert.deepEqual(report.pageErrors, []);
    report.passed = true;
  } catch (e) {
    report.failure = e.message;
    throw e;
  } finally {
    if (browser) await browser.close();
    fs.writeFileSync(path.join(out, 'windows-acceptance.json'), JSON.stringify(report, null, 2) + '\n');
    const actual = fs.realpathSync.native(profile);
    assert.equal(path.dirname(actual), fs.realpathSync.native(scratch));
    assert.ok(path.basename(actual).startsWith('free-model-windows-'));
    fs.rmSync(actual, {recursive: true, force: true, maxRetries: 4, retryDelay: 100});
  }
  console.log(report.checks.join('\n'));
}
verify().catch(e => { console.error(e); process.exitCode = 1; });
