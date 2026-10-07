/* 融合页验证脚本：单进程完成 打开→三透镜截图→console 收集 */
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Users\\Admin\\AppData\\Local\\ms-playwright\\chromium-1247\\chrome-win64\\chrome.exe';
const OUT = 'D:\\AI2050\\Ai2050-OpenOne\\tests\\';
const errors = [];

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

await page.goto('http://localhost:5173/rdc-fusion', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);
await page.screenshot({ path: OUT + 'fusion_spatial.png' });

await page.locator('.fw-ltab').nth(1).click();
await page.waitForTimeout(600);
await page.screenshot({ path: OUT + 'fusion_process.png' });

await page.locator('.fw-ltab').nth(2).click();
await page.waitForTimeout(600);
await page.screenshot({ path: OUT + 'fusion_progress.png' });

await page.locator('.fw-rail .fw-rbtn').first().click();
await page.waitForTimeout(600);
await page.screenshot({ path: OUT + 'fusion_home.png' });

/* 首页默认旧版 App 是否不受影响 */
await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.waitForTimeout(4000);
await page.screenshot({ path: OUT + 'legacy_home.png' });

await browser.close();
console.log('CONSOLE_ERRORS=' + errors.length);
errors.slice(0, 10).forEach(e => console.log('ERR: ' + e.slice(0, 200)));
