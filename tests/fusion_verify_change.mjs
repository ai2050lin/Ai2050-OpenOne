/* 神经元级「参数变化双轴」验证：时间轴 / 编码模式 / 时变曲线 / 注意力分布 / 层扫描 */
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Users\\Admin\\AppData\\Local\\ms-playwright\\chromium-1247\\chrome-win64\\chrome.exe';
const OUT = 'D:/AI2050/Ai2050-OpenOne/tests/';

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
const errs = [];
page.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 120)); });
page.on('pageerror', e => errs.push('PAGEERR ' + String(e).slice(0, 160)));

await page.goto('http://localhost:5173/rdc-fusion', { waitUntil: 'networkidle' });
await page.waitForTimeout(800);

/* 左轨 → 空间透镜（默认已在）→ 层平铺中点选一个层盒（多点候选）→ 进入神经元空间 */
const cv = page.locator('.fw-sp-canvas');
const box = await cv.boundingBox();
let entered = false;
for (const [fx, fy] of [[0.40, 0.42], [0.46, 0.46], [0.52, 0.50], [0.58, 0.44], [0.36, 0.50], [0.62, 0.48]]) {
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
  await page.waitForTimeout(300);
  const btn = page.locator('button:has-text("进入神经元空间")');
  if (await btn.count() > 0 && await btn.first().isVisible()) { await btn.first().click(); entered = true; break; }
}
const NEU = await page.evaluate(() => !!document.querySelector('.fw-neu-timeline'));
console.log('NEURON_ENTERED', entered, 'TIMELINE', NEU);

/* 时间轴：8 token chips + 编码 3 按钮 */
const tl = await page.evaluate(() => {
  const chips = [...document.querySelectorAll('.fw-neu-timeline .chip')];
  const encs = [...document.querySelectorAll('.fw-neu-enc button')];
  return { nChips: chips.length, toks: chips.map(c => c.textContent), encs: encs.map(e => e.textContent) };
});
console.log('TIMELINE_CHIPS', tl.nChips, JSON.stringify(tl.toks), JSON.stringify(tl.encs));

/* 点 chip[4] → 高亮跟随 */
await page.locator('.fw-neu-timeline .chip').nth(4).click();
await page.waitForTimeout(200);
const chipOn = await page.evaluate(() => [...document.querySelectorAll('.fw-neu-timeline .chip')].findIndex(c => c.classList.contains('on')));
console.log('CHIP_CLICK_ON', chipOn);
await page.screenshot({ path: OUT + 'fusion_change_t4.png' });

/* 播放 → t 前进 */
await page.locator('.fw-neu-timeline .fw-tbtn').first().click();
await page.waitForTimeout(1600);
const tAfter = await page.evaluate(() => document.querySelector('.fw-neu-timeline .tl-pos').textContent);
console.log('PLAY_ADVANCE', tAfter);
await page.locator('.fw-neu-timeline .fw-tbtn').first().click(); // 暂停

/* 编码模式 → write */
await page.locator('.fw-neu-enc button:has-text("写入")').click();
await page.waitForTimeout(300);
const encOn = await page.evaluate(() => [...document.querySelectorAll('.fw-neu-enc button')].find(b => b.classList.contains('on'))?.textContent);
console.log('ENC_WRITE_ON', encOn);
await page.screenshot({ path: OUT + 'fusion_change_write.png' });
await page.locator('.fw-neu-enc button:has-text("变化")').click();
await page.waitForTimeout(300);
await page.screenshot({ path: OUT + 'fusion_change_diff.png' });

/* 点画布选 MLP unit（多点候选直到 spark 出现）→ 断言 t 轴曲线 + ℓ 轴跨层基座 */
let picked = '';
for (const [fx, fy] of [[0.50, 0.50], [0.46, 0.52], [0.54, 0.48], [0.48, 0.44], [0.52, 0.56], [0.44, 0.48], [0.56, 0.52], [0.50, 0.40]]) {
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
  await page.waitForTimeout(250);
  const hd = await page.evaluate(() => document.querySelector('.fw-lp .fw-lp-hd span')?.textContent || '');
  if (/MLP unit/.test(hd)) { picked = hd; break; }
  if (/Q-dim/.test(hd)) { picked = hd; break; }
  if (/Residual/.test(hd)) { picked = hd; break; }
}
const sparks = await page.evaluate(() => [...document.querySelectorAll('.fw-lp .fw-neu-spark')].length);
console.log('PICKED', picked, 'SPARKS', sparks);
await page.screenshot({ path: OUT + 'fusion_change_unit.png' });

/* 若选中的不是 MLP，点 × 后继续找 MLP（跨层曲线断言） */
if (!/MLP unit/.test(picked)) {
  for (let round = 0; round < 6; round++) {
    const xbtn = page.locator('.fw-lp .fw-lp-hd button');
    if (await xbtn.count() > 0) await xbtn.first().click();
    await page.waitForTimeout(150);
    const [fx, fy] = [[0.50 + round * 0.01, 0.50 + (round % 2 ? 0.03 : -0.03)], [0.47, 0.55], [0.53, 0.45], [0.49, 0.47], [0.51, 0.53], [0.48, 0.50]][round];
    await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
    await page.waitForTimeout(250);
    const hd = await page.evaluate(() => document.querySelector('.fw-lp .fw-lp-hd span')?.textContent || '');
    if (/MLP unit/.test(hd)) { picked = hd; break; }
  }
  const lCurves = await page.evaluate(() => [...document.querySelectorAll('.fw-lp .fw-neu-spark')].length);
  console.log('RETRY_PICKED', picked, 'SPARKS', lCurves);
  await page.screenshot({ path: OUT + 'fusion_change_unit.png' });
}

/* MLP 面板断言：t 轴三联曲线 + ℓ 轴跨层基座两个 spark */
const mlpPanel = await page.evaluate(() => {
  const secs = [...document.querySelectorAll('.fw-lp .sec')].map(s => s.textContent);
  return {
    hasT: secs.some(s => /时变曲线 · t 轴/.test(s)),
    hasL: secs.some(s => /跨层基座 · ℓ 轴/.test(s)),
  };
});
console.log('MLP_PANEL_CURVES', JSON.stringify(mlpPanel));

/* 选 Q-dim → 注意力分布条 */
const xbtn2 = page.locator('.fw-lp .fw-lp-hd button');
if (await xbtn2.count() > 0) await xbtn2.first().click();
await page.waitForTimeout(150);
let qPicked = false;
for (const [fx, fy] of [[0.80, 0.50], [0.84, 0.46], [0.78, 0.54], [0.86, 0.52], [0.76, 0.48], [0.82, 0.40]]) {
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
  await page.waitForTimeout(250);
  const hd = await page.evaluate(() => document.querySelector('.fw-lp .fw-lp-hd span')?.textContent || '');
  if (/Q-dim/.test(hd)) { qPicked = true; break; }
  if (/MLP unit|Residual/.test(hd)) {
    const xb = page.locator('.fw-lp .fw-lp-hd button');
    if (await xb.count() > 0) await xb.first().click();
    await page.waitForTimeout(150);
  }
}
const attnRows = await page.evaluate(() => document.querySelectorAll('.fw-neu-attn .bar-row').length);
console.log('QDIM_PICKED', qPicked, 'ATTN_BARS', attnRows);
if (qPicked) await page.screenshot({ path: OUT + 'fusion_change_attn.png' });

/* 层扫描 ▶L：1.6s 后层号应变化 */
await page.keyboard.press('Escape');
const layerBefore = await page.evaluate(() => document.querySelector('.fw-neu-cur')?.textContent);
await page.locator('.fw-neu-lsel button:has-text("▶L")').click();
await page.waitForTimeout(1700);
const layerAfter = await page.evaluate(() => document.querySelector('.fw-neu-cur')?.textContent);
await page.locator('.fw-neu-lsel button:has-text("⏸")').click();
console.log('L_SCAN', layerBefore, '→', layerAfter);

console.log('CONSOLE_ERRORS', errs.length);
errs.slice(0, 8).forEach(e => console.log('ERR', e));
await browser.close();
