/* 空间透镜三模式验证：默认层栈 → 点击层板出详情卡 → 点云 → 参数热图 */
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Users\\Admin\\AppData\\Local\\ms-playwright\\chromium-1247\\chrome-win64\\chrome.exe';
const OUT = 'D:\\AI2050\\Ai2050-OpenOne\\tests\\';
const errors = [];

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

await page.goto('http://localhost:5173/rdc-fusion', { waitUntil: 'networkidle' });
await page.waitForTimeout(1800);
await page.screenshot({ path: OUT + 'fusion_spatial_stack.png' });

/* 点击层板（画布中上部 → 选中某中间偏上层） */
const cv = page.locator('.fw-sp-canvas');
const box = await cv.boundingBox();
await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.38);
await page.waitForTimeout(700);
await page.screenshot({ path: OUT + 'fusion_spatial_stack_sel.png' });

/* 切特征点云 */
await page.locator('.fw-mode-seg button').nth(1).click();
await page.waitForTimeout(800);
await page.screenshot({ path: OUT + 'fusion_spatial_cloud.png' });

/* 切参数热图 */
await page.locator('.fw-mode-seg button').nth(2).click();
await page.waitForTimeout(800);
await page.screenshot({ path: OUT + 'fusion_spatial_param.png' });

/* 热图上悬停读数 */
await page.mouse.move(box.x + box.width * 0.42, box.y + box.height * 0.5);
await page.waitForTimeout(400);
await page.screenshot({ path: OUT + 'fusion_spatial_param_hover.png' });

await browser.close();
console.log('CONSOLE_ERRORS=' + errors.length);
errors.slice(0, 8).forEach(e => console.log('ERR: ' + e.slice(0, 160)));
