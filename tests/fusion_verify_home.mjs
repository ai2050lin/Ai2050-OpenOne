/* 默认首页切换验证：/ = 融合页，/legacy = 旧版，子页不受影响 */
import { chromium } from 'playwright-core';
const CHROME = 'C:\\Users\\Admin\\AppData\\Local\\ms-playwright\\chromium-1247\\chrome-win64\\chrome.exe';
const OUT = 'D:\\AI2050\\Ai2050-OpenOne\\tests\\';
const errors = [];
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));
const results = [];

// 断言1: / 渲染融合驾驶舱（.fw-root 存在）
await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.waitForTimeout(1800);
const rootFusion = await page.evaluate(() => !!document.querySelector('.fw-root'));
results.push(['root / renders fusion (.fw-root)', rootFusion]);

// 断言2: /rdc-fusion 仍渲染融合页
await page.goto('http://localhost:5173/rdc-fusion', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(1200);
results.push(['/rdc-fusion still fusion', await page.evaluate(() => !!document.querySelector('.fw-root'))]);

// 断言3: /legacy 渲染旧版（无 .fw-root 且有内容）
await page.goto('http://localhost:5173/legacy', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2500);
const legacyNoFusion = await page.evaluate(() => !document.querySelector('.fw-root'));
const legacyHasContent = await page.evaluate(() => {
  const el = document.getElementById('root');
  return !!el && el.children.length > 0 && document.body.innerText.length > 20;
});
results.push(['/legacy is old page (no .fw-root, has content)', legacyNoFusion && legacyHasContent]);

// 断言4: /rdc-query 子页不受影响
await page.goto('http://localhost:5173/rdc-query', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(1500);
results.push(['/rdc-query unaffected', await page.evaluate(() => !document.querySelector('.fw-root') && document.body.innerText.length > 20)]);

// 首页截图留档
await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2000);
await page.screenshot({ path: OUT + 'fusion_home_default.png' });

let pass = 0, fail = 0;
for (const [name, ok] of results) {
  console.log((ok ? 'PASS' : 'FAIL') + ' | ' + name);
  ok ? pass++ : fail++;
}
console.log('TOTAL ' + pass + ' PASS ' + fail + ' FAIL');
console.log('CONSOLE_ERRORS=' + errors.length);
errors.slice(0, 5).forEach(e => console.log('ERR: ' + e.slice(0, 150)));
await browser.close();
process.exit(fail === 0 ? 0 : 1);
