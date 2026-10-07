/* 递归遍历全部 cssRules（含 @media/@layer 嵌套）找染白 h3 的规则 */
import { chromium } from 'playwright-core';
const CHROME = 'C:\\Users\\Admin\\AppData\\Local\\ms-playwright\\chromium-1247\\chrome-win64\\chrome.exe';
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
await page.goto('http://localhost:5173/rdc-fusion', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(1500);
await page.locator('.fw-rail .fw-rbtn').first().click();
await page.waitForTimeout(500);
const hits = await page.evaluate(() => {
  const h3 = document.querySelector('.fw-entry h3');
  const out = [];
  function walk(rules, sheet, ctx) {
    for (const r of rules) {
      if (r.cssRules) { walk(r.cssRules, sheet, ctx + '>' + (r.conditionText || r.name || r.cssText.slice(0, 30))); continue; }
      if (r.selectorText && r.style && r.style.color) {
        try {
          if (h3.matches(r.selectorText)) out.push({ sel: r.selectorText, color: r.style.color, ctx, sheet: sheet.href ? sheet.href.split('/').pop() : 'inline#' + (sheet.ownerNode && sheet.ownerNode.id || '') });
        } catch {}
      }
    }
  }
  for (const ss of document.styleSheets) {
    let rules; try { rules = ss.cssRules; } catch { continue; }
    walk(rules, ss, '');
  }
  /* 同时给出 h3 的祖先链 color */
  let el = h3, chain = [];
  while (el && el !== document.documentElement) {
    chain.push((el.className || el.tagName) + '=' + getComputedStyle(el).color);
    el = el.parentElement;
  }
  return { rules: out, chain: chain.slice(0, 8) };
});
console.log(JSON.stringify(hits, null, 1));
await browser.close();
