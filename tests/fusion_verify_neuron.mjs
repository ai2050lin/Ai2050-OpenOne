/* 神经元级模式浏览器验证：层平铺→下钻→点选单个神经元→层切换 */
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Users\\Admin\\AppData\\Local\\ms-playwright\\chromium-1247\\chrome-win64\\chrome.exe';
const errors = [];

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

await page.goto('http://localhost:5173/rdc-fusion', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2500);

/* 1) 层平铺：点击层盒选中（沿层带 10 点候选） */
const box = await page.locator('.fw-sp-canvas').first().boundingBox();
let selLayer = false;
for (const t of [0.5, 0.42, 0.58, 0.35, 0.65, 0.48, 0.55, 0.3, 0.7, 0.45]) {
  await page.mouse.click(box.x + box.width * t, box.y + box.height * 0.48);
  await page.waitForTimeout(300);
  if (await page.locator('.fw-lp-hd', { hasText: 'TransformerBlock' }).count() > 0) { selLayer = true; break; }
}
console.log('LAYER_SELECTED=' + selLayer);

/* 2) 点击「进入神经元空间 →」 */
await page.click('button:has-text("进入神经元空间")');
await page.waitForTimeout(1800);
const neuCanvas = await page.locator('.fw-sp-canvas').count();
console.log('NEURON_CANVAS=' + neuCanvas);
console.log('PANEL_OVERVIEW=' + ((await page.locator('.fw-lp-hd', { hasText: '神经元总览' }).count()) > 0));
await page.screenshot({ path: 'D:/AI2050/Ai2050-OpenOne/tests/fusion_neuron_default.png' });

/* 3) 点选单个神经元：画布中心区域（MLP 簇云最密处）多点候选 */
const nbox = await page.locator('.fw-sp-canvas').first().boundingBox();
let picked = '';
for (const [fx, fy] of [[0.5, 0.5], [0.48, 0.52], [0.52, 0.48], [0.46, 0.55], [0.54, 0.45], [0.45, 0.5], [0.55, 0.52]]) {
  await page.mouse.click(nbox.x + nbox.width * fx, nbox.y + nbox.height * fy);
  await page.waitForTimeout(250);
  const hd = await page.locator('.fw-lp-hd span').first().textContent().catch(() => '');
  if (hd && /MLP unit|Q-dim|Residual dim/.test(hd)) { picked = hd; break; }
}
console.log('PICKED=' + picked);
await page.screenshot({ path: 'D:/AI2050/Ai2050-OpenOne/tests/fusion_neuron_sel.png' });

/* 4) 选中详情断言（若选中的是 MLP unit，应有三段参数切片 + 激活示例） */
if (/MLP unit/.test(picked)) {
  const secs = await page.locator('.fw-lp .sec').allTextContents();
  console.log('MLP_SECTIONS=' + secs.filter(s => /up_proj|gate_proj|down_proj|激活示例/.test(s)).length);
}

/* 5) 层切换 > 一次 */
await page.click('.fw-neu-lsel button:nth-child(3)');
await page.waitForTimeout(1200);
const cur = await page.locator('.fw-neu-cur').textContent();
console.log('AFTER_NEXT=' + cur);
await page.screenshot({ path: 'D:/AI2050/Ai2050-OpenOne/tests/fusion_neuron_next.png' });

/* 6) 返回层平铺 */
await page.click('button:has-text("← 层平铺")');
await page.waitForTimeout(800);
console.log('BACK_STACK=' + ((await page.locator('.fw-lp-hd', { hasText: '神经元总览' }).count()) === 0));

console.log('CONSOLE_ERRORS=' + errors.length);
errors.slice(0, 8).forEach(e => console.log('ERR: ' + e.slice(0, 160)));
await browser.close();
