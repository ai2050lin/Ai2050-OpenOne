/* 鼠标交互验证：左键=水平旋转(th)、右键=垂直俯仰(ph)、右键菜单被阻止、左键点选不受影响 */
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

/* 读 StackMode readout 的 θ/φ */
async function readCam() {
  const txt = await page.evaluate(() => document.querySelector('.fw-sp-readout')?.textContent || '');
  const m = txt.match(/θ (-?\d+)° · φ (-?\d+)°/);
  return m ? { th: +m[1], ph: +m[2] } : null;
}

const cv = page.locator('.fw-sp-canvas');
const box = await cv.boundingBox();
const cx = box.x + box.width * 0.45, cy = box.y + box.height * 0.5;

/* 1) 左键水平拖拽 → th 变，ph 不变（mousedown 先停 auto-rotate，读数干净） */
await page.mouse.move(cx, cy);
await page.mouse.down();
await page.waitForTimeout(650);
const s0 = await readCam();
await page.mouse.move(cx + 120, cy, { steps: 6 });
await page.mouse.up();
await page.waitForTimeout(650);
const s1 = await readCam();
console.log('LEFT_DRAG', JSON.stringify(s0), '→', JSON.stringify(s1),
  'Δth=' + (s1.th - s0.th), 'Δph=' + (s1.ph - s0.ph),
  'PASS', Math.abs(s1.th - s0.th) >= 5 && Math.abs(s1.ph - s0.ph) <= 1);

/* 2) 右键垂直拖拽 → ph 变，th 不变 */
await page.mouse.move(cx, cy);
await page.mouse.down({ button: 'right' });
await page.waitForTimeout(650);
const s2 = await readCam();
await page.mouse.move(cx, cy - 110, { steps: 6 });
await page.mouse.up({ button: 'right' });
await page.waitForTimeout(650);
const s3 = await readCam();
console.log('RIGHT_DRAG', JSON.stringify(s2), '→', JSON.stringify(s3),
  'Δth=' + (s3.th - s2.th), 'Δph=' + (s3.ph - s2.ph),
  'PASS', Math.abs(s3.ph - s2.ph) >= 5 && Math.abs(s3.th - s2.th) <= 1);

/* 3) 右键菜单被阻止（canvas 上合成 contextmenu 检查 defaultPrevented） */
const ctxBlocked = await page.evaluate(() => {
  const c = document.querySelector('.fw-sp-canvas');
  const ev = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
  c.dispatchEvent(ev);
  return ev.defaultPrevented;
});
console.log('CONTEXTMENU_BLOCKED', ctxBlocked);

/* 4) 左键点选层盒仍工作（多点候选 → 面板出现层参数） */
let picked = false;
for (const [fx, fy] of [[0.40, 0.42], [0.46, 0.46], [0.52, 0.50], [0.58, 0.44], [0.36, 0.50]]) {
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
  await page.waitForTimeout(300);
  const hd = await page.evaluate(() => document.querySelector('.fw-lp .fw-lp-hd')?.textContent || '');
  if (/L\d+/.test(hd) && !/总览/.test(hd)) { picked = true; break; }
}
console.log('LEFT_CLICK_PICK_LAYER', picked);

/* 5) 神经元级：同样验证左/右键拆分 */
const btnNeu = page.locator('button:has-text("进入神经元空间")');
if (picked && await btnNeu.count() > 0) {
  await btnNeu.first().click();
  await page.waitForTimeout(600);
  const box2 = await page.locator('.fw-sp-canvas').boundingBox();
  const cx2 = box2.x + box2.width * 0.45, cy2 = box2.y + box2.height * 0.5;
  await page.mouse.move(cx2, cy2);
  await page.mouse.down();
  await page.waitForTimeout(650);
  const n0 = await readCam();
  await page.mouse.move(cx2 + 120, cy2, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(650);
  const n1 = await readCam();
  console.log('NEU_LEFT', 'Δth=' + (n1.th - n0.th), 'Δph=' + (n1.ph - n0.ph),
    'PASS', Math.abs(n1.th - n0.th) >= 5 && Math.abs(n1.ph - n0.ph) <= 1);
  await page.mouse.move(cx2, cy2);
  await page.mouse.down({ button: 'right' });
  await page.waitForTimeout(650);
  const n2 = await readCam();
  await page.mouse.move(cx2, cy2 - 110, { steps: 6 });
  await page.mouse.up({ button: 'right' });
  await page.waitForTimeout(650);
  const n3 = await readCam();
  console.log('NEU_RIGHT', 'Δth=' + (n3.th - n2.th), 'Δph=' + (n3.ph - n2.ph),
    'PASS', Math.abs(n3.ph - n2.ph) >= 5 && Math.abs(n3.th - n2.th) <= 1);
  await page.screenshot({ path: OUT + 'fusion_mouse_neuron.png' });
}

await page.screenshot({ path: OUT + 'fusion_mouse_stack.png' });
console.log('CONSOLE_ERRORS', errs.length);
errs.slice(0, 6).forEach(e => console.log('ERR', e));
await browser.close();
