/* 检查总览入口卡 h3 是否渲染 */
import { chromium } from 'playwright-core';
const CHROME = 'C:\\Users\\Admin\\AppData\\Local\\ms-playwright\\chromium-1247\\chrome-win64\\chrome.exe';
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
await page.goto('http://localhost:5173/rdc-fusion', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(1500);
await page.locator('.fw-rail .fw-rbtn').first().click();
await page.waitForTimeout(500);
const info = await page.evaluate(() => {
  const out = [];
  document.querySelectorAll('.fw-entry').forEach((el, i) => {
    const h3 = el.querySelector('h3');
    const r = h3 ? h3.getBoundingClientRect() : null;
    const cs = h3 ? getComputedStyle(h3) : null;
    out.push({
      i,
      h3text: h3 ? h3.textContent : null,
      rect: r ? { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) } : null,
      display: cs ? cs.display : null, color: cs ? cs.color : null, fontSize: cs ? cs.fontSize : null,
    });
  });
  return out;
});
console.log(JSON.stringify(info, null, 1));
await browser.close();
