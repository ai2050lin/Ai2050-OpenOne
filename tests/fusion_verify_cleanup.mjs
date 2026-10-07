/* 系统梳理后全透镜验证：左轨导航（tabs 已删）+ 空间四模式 + 总览 */
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Users\\Admin\\AppData\\Local\\ms-playwright\\chromium-1247\\chrome-win64\\chrome.exe';
const errors = [];
const S = 'D:/AI2050/Ai2050-OpenOne/tests/';

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

await page.goto('http://localhost:5173/rdc-fusion', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2500);

/* 1) 默认=空间透镜层平铺；确认旧 tabs 已不存在、左轨空间高亮 */
console.log('TABS_REMOVED=' + ((await page.locator('.fw-lens-tabs').count()) === 0));
console.log('RAIL_SPATIAL_ON=' + ((await page.locator('.fw-rbtn.on', { hasText: '空间' }).count()) > 0));
await page.screenshot({ path: S + 'fusion2_stack.png' });

/* 2) 左轨切研发/脉络 */
await page.click('.fw-rbtn:has-text("研发")');
await page.waitForTimeout(1200);
console.log('PROCESS_VIEW=' + ((await page.locator('.fw-pr-left').count()) > 0));
await page.screenshot({ path: S + 'fusion2_process.png' });
await page.click('.fw-rbtn:has-text("路线")');
await page.waitForTimeout(1200);
console.log('PROGRESS_VIEW=' + ((await page.locator('.fw-tl-node').count()) > 0));
await page.screenshot({ path: S + 'fusion2_progress.png' });

/* 3) 回空间：模式切换 层平铺→神经元级（经层详情下钻）→点云→热图 */
await page.click('.fw-rbtn:has-text("空间")');
await page.waitForTimeout(1000);
const box = await page.locator('.fw-sp-canvas').first().boundingBox();
for (const t of [0.5, 0.42, 0.58, 0.35, 0.65, 0.48]) {
  await page.mouse.click(box.x + box.width * t, box.y + box.height * 0.48);
  await page.waitForTimeout(250);
  if (await page.locator('.fw-lp-hd', { hasText: 'TransformerBlock' }).count() > 0) break;
}
await page.click('button:has-text("进入神经元空间")');
await page.waitForTimeout(1600);
console.log('NEURON_ON=' + ((await page.locator('.fw-lp-hd', { hasText: '神经元总览' }).count()) > 0));
await page.screenshot({ path: S + 'fusion2_neuron.png' });

/* 模式分段第 3/4 个（点云/热图） */
await page.click('.fw-mode-seg button:has-text("特征点云")');
await page.waitForTimeout(1200);
console.log('CLOUD_ON=' + ((await page.locator('.fw-sp-focus').count()) > 0));
await page.screenshot({ path: S + 'fusion2_cloud.png' });
await page.click('.fw-mode-seg button:has-text("参数热图")');
await page.waitForTimeout(1000);
console.log('PARAM_NOTE=' + ((await page.locator('.fw-param-note').count()) > 0));
await page.screenshot({ path: S + 'fusion2_param.png' });

/* 4) 总览 */
await page.click('.fw-rbtn:has-text("总览")');
await page.waitForTimeout(1000);
console.log('HOME_ON=' + ((await page.locator('.fw-hero').count()) > 0));
await page.screenshot({ path: S + 'fusion2_home.png' });

console.log('CONSOLE_ERRORS=' + errors.length);
errors.slice(0, 8).forEach(e => console.log('ERR: ' + e.slice(0, 150)));
await browser.close();
