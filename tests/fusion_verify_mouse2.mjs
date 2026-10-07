/* 鼠标交互验证 v2：左键=完整旋转(θ+φ 上下左右都变)、右键=仅上下平移视角位置(pan, 角度零变化) */
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

/* 读 readout 的 θ/φ/pan */
async function readCam() {
  const txt = await page.evaluate(() => document.querySelector('.fw-sp-readout')?.textContent || '');
  const m = txt.match(/θ (-?\d+)° · φ (-?\d+)° · 平移 ([+-]?\d+)px/);
  return m ? { th: +m[1], ph: +m[2], pan: +m[3] } : null;
}
async function shot(name) { await page.screenshot({ path: OUT + name }); }

const cv = page.locator('.fw-sp-canvas');
const box = await cv.boundingBox();
const cx = box.x + box.width * 0.45, cy = box.y + box.height * 0.5;

/* 1) 左键斜向拖拽（下右）→ θ 与 φ 都变化（完整旋转） */
await page.mouse.move(cx, cy);
await page.mouse.down();
await page.waitForTimeout(650);
const s0 = await readCam();
await page.mouse.move(cx + 100, cy + 70, { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(650);
const s1 = await readCam();
console.log('LEFT_DRAG_FULL_ROTATE', JSON.stringify(s0), '→', JSON.stringify(s1),
  'PASS', Math.abs(s1.th - s0.th) >= 5 && Math.abs(s1.ph - s0.ph) >= 2);

/* 2) 右键垂直拖拽（上移 110px）→ θ/φ 零变化，pan 变化 ≈ +110（内容跟随下移） */
await page.mouse.move(cx, cy);
await page.mouse.down({ button: 'right' });
await page.waitForTimeout(650);
const s2 = await readCam();
await page.mouse.move(cx, cy - 110, { steps: 8 });
await page.mouse.up({ button: 'right' });
await page.waitForTimeout(650);
const s3 = await readCam();
const angleFrozen = s3.th === s2.th && s3.ph === s2.ph;
const panMoved = Math.abs(s3.pan - s2.pan) >= 90;
console.log('RIGHT_DRAG_PAN', JSON.stringify(s2), '→', JSON.stringify(s3),
  'ANGLE_FROZEN', angleFrozen, 'PAN_MOVED', panMoved, 'PASS', angleFrozen && panMoved);

/* 3) 右键菜单被阻止 */
const ctxBlocked = await page.evaluate(() => {
  const c = document.querySelector('.fw-sp-canvas');
  const ev = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
  c.dispatchEvent(ev);
  return ev.defaultPrevented;
});
console.log('CONTEXTMENU_BLOCKED', ctxBlocked);

/* 4) 左键点选层盒仍工作 */
let picked = false;
for (const [fx, fy] of [[0.40, 0.42], [0.46, 0.46], [0.52, 0.50], [0.58, 0.44], [0.36, 0.50]]) {
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
  await page.waitForTimeout(300);
  const hd = await page.evaluate(() => document.querySelector('.fw-lp .fw-lp-hd')?.textContent || '');
  if (/L\d+/.test(hd) && !/总览/.test(hd)) { picked = true; break; }
}
console.log('LEFT_CLICK_PICK_LAYER', picked);
await shot('fusion_mouse2_stack.png');

/* 5) 神经元级：左键斜拖旋转 + 右键竖拖平移 */
const btnNeu = page.locator('button:has-text("进入神经元空间")');
if (picked && await btnNeu.count() > 0) {
  await btnNeu.first().click();
  await page.waitForTimeout(700);
  const box2 = await page.locator('.fw-sp-canvas').boundingBox();
  const cx2 = box2.x + box2.width * 0.45, cy2 = box2.y + box2.height * 0.5;
  await page.mouse.move(cx2, cy2);
  await page.mouse.down();
  await page.waitForTimeout(650);
  const n0 = await readCam();
  await page.mouse.move(cx2 - 90, cy2 + 60, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(650);
  const n1 = await readCam();
  console.log('NEU_LEFT_FULL_ROTATE', JSON.stringify(n0), '→', JSON.stringify(n1),
    'PASS', Math.abs(n1.th - n0.th) >= 5 && Math.abs(n1.ph - n0.ph) >= 2);
  await page.mouse.move(cx2, cy2);
  await page.mouse.down({ button: 'right' });
  await page.waitForTimeout(650);
  const n2 = await readCam();
  await page.mouse.move(cx2, cy2 - 80, { steps: 8 });
  await page.mouse.up({ button: 'right' });
  await page.waitForTimeout(650);
  const n3 = await readCam();
  const frozen2 = n3.th === n2.th && n3.ph === n2.ph;
  const pan2 = Math.abs(n3.pan - n2.pan) >= 60;
  console.log('NEU_RIGHT_PAN', JSON.stringify(n2), '→', JSON.stringify(n3),
    'ANGLE_FROZEN', frozen2, 'PAN_MOVED', pan2, 'PASS', frozen2 && pan2);
  await shot('fusion_mouse2_neuron.png');
}

/* 6) 特征点云模式：右键 pan 同样生效 */
const cloudBtn = page.locator('.fw-mode-seg button', { hasText: '特征点云' });
if (await cloudBtn.count() > 0) {
  await cloudBtn.first().click();
  await page.waitForTimeout(700);
  const box3 = await page.locator('.fw-sp-canvas').boundingBox();
  const cx3 = box3.x + box3.width * 0.5, cy3 = box3.y + box3.height * 0.5;
  await page.mouse.move(cx3, cy3);
  await page.mouse.down({ button: 'right' });
  await page.waitForTimeout(650);
  const c0 = await readCam();
  await page.mouse.move(cx3, cy3 - 70, { steps: 8 });
  await page.mouse.up({ button: 'right' });
  await page.waitForTimeout(650);
  const c1 = await readCam();
  const frozen3 = c1.th === c0.th && c1.ph === c0.ph;
  const pan3 = Math.abs(c1.pan - c0.pan) >= 50;
  console.log('CLOUD_RIGHT_PAN', JSON.stringify(c0), '→', JSON.stringify(c1),
    'ANGLE_FROZEN', frozen3, 'PAN_MOVED', pan3, 'PASS', frozen3 && pan3);
  await shot('fusion_mouse2_cloud.png');
}

console.log('CONSOLE_ERRORS', errs.length);
errs.slice(0, 6).forEach(e => console.log('ERR', e));
await browser.close();
