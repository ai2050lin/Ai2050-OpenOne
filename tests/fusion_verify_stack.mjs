/* 层平铺模式验证 v2：多点候选点击层板 → 面板断言 → 滚底 → Esc回总览 → 点云/热图回归 */
import { chromium } from 'playwright-core';

const CHROME = 'C:\\Users\\Admin\\AppData\\Local\\ms-playwright\\chromium-1247\\chrome-win64\\chrome.exe';
const OUT = 'D:\\AI2050\\Ai2050-OpenOne\\tests\\';
const errors = [];

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

await page.goto('http://localhost:5173/rdc-fusion', { waitUntil: 'networkidle' });
await page.waitForTimeout(1600);

const lpW = await page.locator('.fw-lp').evaluate(el => el.offsetWidth);
const cvW = await page.locator('.fw-sp-canvasbox').evaluate(el => el.offsetWidth);
console.log('PANEL_WIDTH=' + lpW, 'CANVASBOX_WIDTH=' + cvW);
await page.screenshot({ path: OUT + 'fusion_stack_default.png' });

/* 沿层带多点候选点击（自动旋转会平移层带，逐点尝试直到选中） */
const cv = page.locator('.fw-sp-canvas');
const box = await cv.boundingBox();
const cands = [-150, -110, -70, -30, 14, 55, 95, 135, 175, 215];
let picked = null;
for (const dx of cands) {
  await page.mouse.click(box.x + box.width * 0.5 + dx, box.y + box.height * 0.52 - 4);
  await page.waitForTimeout(350);
  const txt = await page.locator('.fw-lp').innerText();
  const m = txt.match(/L(\d+) · TransformerBlock/);
  if (m) { picked = m[1]; break; }
}
console.log('PANEL_LAYER=' + (picked ?? 'NONE'));
if (picked !== null) {
  const txt = await page.locator('.fw-lp').innerText();
  console.log('HAS_QPROJ=' + /q_proj/.test(txt), 'HAS_DOWN=' + /down_proj/.test(txt),
    'HAS_TOTAL=' + /100\.9M/.test(txt), 'HAS_HEADS=' + /32 heads/.test(txt));
  await page.screenshot({ path: OUT + 'fusion_stack_sel.png' });
  await page.locator('.fw-lp').evaluate(el => { el.scrollTop = el.scrollHeight; });
  await page.waitForTimeout(300);
  await page.screenshot({ path: OUT + 'fusion_stack_panel_bottom.png' });
}

/* Esc → 模型总览 */
await page.keyboard.press('Escape');
await page.waitForTimeout(400);
const ov = await page.locator('.fw-lp').innerText();
console.log('PANEL_OVERVIEW=' + /模型总览/.test(ov), 'PARAM_TREE=' + /参数树/.test(ov), 'TOTAL_402B=' + /4\.02B/.test(ov));
await page.screenshot({ path: OUT + 'fusion_stack_overview.png' });

/* 点云 / 热图回归 */
await page.locator('.fw-mode-seg button').nth(1).click();
await page.waitForTimeout(700);
await page.screenshot({ path: OUT + 'fusion_spatial_cloud.png' });
await page.locator('.fw-mode-seg button').nth(2).click();
await page.waitForTimeout(700);
await page.screenshot({ path: OUT + 'fusion_spatial_param.png' });

await browser.close();
console.log('CONSOLE_ERRORS=' + errors.length);
errors.slice(0, 8).forEach(e => console.log('ERR: ' + e.slice(0, 160)));
