import fs from 'node:fs';

const port = process.argv[2] || '9223';
const screenshotPath = process.argv[3];
const resultPath = process.argv[4];
const viewportWidth = Number(process.argv[5] || 0);
const viewportHeight = Number(process.argv[6] || 0);
const screenMode = process.argv[7] || 'workspace';
if (!screenshotPath || !resultPath) {
  throw new Error('usage: node phase385_client_cdp_check.mjs PORT SCREENSHOT RESULT');
}

const pages = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
const page = pages.find((item) => item.type === 'page' && item.url.includes('127.0.0.1:5173'));
if (!page) throw new Error('Vite page not found in Chromium targets');

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', reject, { once: true });
});
let nextId = 1;
const pending = new Map();
socket.addEventListener('message', (event) => {
  const payload = JSON.parse(event.data);
  if (!payload.id || !pending.has(payload.id)) return;
  const { resolve, reject } = pending.get(payload.id);
  pending.delete(payload.id);
  if (payload.error) reject(new Error(JSON.stringify(payload.error)));
  else resolve(payload.result);
});

function command(method, params = {}) {
  const id = nextId++;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  return result.result.value;
}

await command('Runtime.enable');
await command('Page.enable');
if (viewportWidth > 0 && viewportHeight > 0) {
  await command('Emulation.setDeviceMetricsOverride', {
    width: viewportWidth,
    height: viewportHeight,
    deviceScaleFactor: 1,
    mobile: viewportWidth <= 768,
  });
  await command('Page.reload', { ignoreCache: true });
}
await new Promise((resolve) => setTimeout(resolve, 2500));
if (screenMode === 'atlas-dashboard') {
  await evaluate(`(() => {
    const button = document.querySelector('button[aria-label="Global Strategy"]');
    if (!button) return false;
    button.click();
    return true;
  })()`);
  await new Promise((resolve) => setTimeout(resolve, 500));
  await evaluate(`(() => {
    const button = [...document.querySelectorAll('button')]
      .find((item) => item.textContent.trim() === '语言分析');
    if (!button) return false;
    button.click();
    return true;
  })()`);
  await new Promise((resolve) => setTimeout(resolve, 2500));
  await evaluate(`(() => {
    const heading = [...document.querySelectorAll('body *')]
      .find((item) => item.children.length === 0 && item.textContent.trim() === '最新成果');
    if (!heading) return false;
    heading.scrollIntoView({ block: 'start' });
    return true;
  })()`);
  await new Promise((resolve) => setTimeout(resolve, 500));
}
const before = await evaluate(`(() => {
  const buttons = [...document.querySelectorAll('button')].map((button) => ({
    text: button.textContent.trim(), disabled: button.disabled,
  }));
  const selects = [...document.querySelectorAll('select')].map((select) => ({
    value: select.value,
    options: [...select.options].map((option) => ({ value: option.value, text: option.textContent.trim() })),
  }));
  return { buttons, selects, canvasCount: document.querySelectorAll('canvas').length };
})()`);
const click = screenMode === 'atlas-dashboard' ? { clicked: false, reason: 'dashboard_mode' } : await evaluate(`(() => {
  const candidates = [...document.querySelectorAll('button')].filter((button) =>
    button.textContent.trim() === '运行' && !button.disabled
  );
  if (!candidates.length) return { clicked: false };
  candidates[0].click();
  return { clicked: true, text: candidates[0].textContent.trim() };
})()`);
await new Promise((resolve) => setTimeout(resolve, 5500));
const after = await evaluate(`(() => {
  const canvases = [...document.querySelectorAll('canvas')];
  const panels = [...document.querySelectorAll('.workspace-panel')];
  const panelLayout = panels.map((panel) => {
    const rect = panel.getBoundingClientRect();
    const content = panel.lastElementChild;
    return {
      className: panel.className,
      left: rect.left,
      top: rect.top,
      right: rect.right,
      bottom: rect.bottom,
      width: rect.width,
      height: rect.height,
      contentClientHeight: content?.clientHeight ?? null,
      contentScrollHeight: content?.scrollHeight ?? null,
      contentScrollable: Boolean(content && content.scrollHeight > content.clientHeight),
    };
  });
  const panelOverlaps = [];
  for (let left = 0; left < panelLayout.length; left += 1) {
    for (let right = left + 1; right < panelLayout.length; right += 1) {
      const width = Math.max(0, Math.min(panelLayout[left].right, panelLayout[right].right)
        - Math.max(panelLayout[left].left, panelLayout[right].left));
      const height = Math.max(0, Math.min(panelLayout[left].bottom, panelLayout[right].bottom)
        - Math.max(panelLayout[left].top, panelLayout[right].top));
      if (width > 0 && height > 0) {
        panelOverlaps.push({
          left: panelLayout[left].className,
          right: panelLayout[right].className,
          area: width * height,
        });
      }
    }
  }
  const canvasPixels = canvases.map((canvas) => {
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return { available: false };
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let sampled = 0;
    let nonBlack = 0;
    let colored = 0;
    let luminanceTotal = 0;
    const pixelStride = Math.max(1, Math.floor((canvas.width * canvas.height) / 100000));
    for (let pixel = 0; pixel < canvas.width * canvas.height; pixel += pixelStride) {
      const offset = pixel * 4;
      const r = pixels[offset];
      const g = pixels[offset + 1];
      const b = pixels[offset + 2];
      sampled += 1;
      luminanceTotal += (r + g + b) / 3;
      if (r + g + b > 15) nonBlack += 1;
      if (Math.max(r, g, b) - Math.min(r, g, b) > 10) colored += 1;
    }
    return {
      available: true,
      sampledPixels: sampled,
      nonBlackPixels: nonBlack,
      coloredPixels: colored,
      nonBlackRatio: nonBlack / Math.max(sampled, 1),
      coloredRatio: colored / Math.max(sampled, 1),
      meanLuminance: luminanceTotal / Math.max(sampled, 1),
      glError: gl.getError(),
    };
  });
  return {
    canvasCount: canvases.length,
    canvases: canvases.map((canvas) => ({
      width: canvas.width,
      height: canvas.height,
      clientWidth: canvas.clientWidth,
      clientHeight: canvas.clientHeight,
    })),
    canvasPixels,
    panelLayout,
    panelOverlaps,
    bodyTextHasPhase385: document.body.innerText.includes('Phase385'),
    bodyTextHasStrictPathCount: document.body.innerText.includes('0/72 complete paths'),
    bodyTextHasExactEventScope: document.body.innerText.includes('精确事件覆盖模式族'),
    bodyTextHasRunning: document.body.innerText.includes('运行中'),
    bodyTextHasGenerated: document.body.innerText.includes('已生成概念集'),
  };
})()`);
const screenshot = await command('Page.captureScreenshot', {
  format: 'png',
  captureBeyondViewport: false,
});
fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, 'base64'));

const canvasOnlyPath = screenshotPath.replace(/\.png$/i, '.canvas-only.png');
const canvasClip = await evaluate(`(() => {
  const canvas = document.querySelector('canvas');
  if (!canvas) return null;
  const keep = new Set();
  let node = canvas;
  while (node) {
    keep.add(node);
    node = node.parentElement;
  }
  window.__phase385HiddenElements = [];
  for (const element of document.querySelectorAll('body *')) {
    if (keep.has(element) || element.contains(canvas)) continue;
    window.__phase385HiddenElements.push([element, element.style.visibility]);
    element.style.visibility = 'hidden';
  }
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, rect.left),
    y: Math.max(0, rect.top),
    width: Math.max(1, Math.min(rect.width, window.innerWidth - Math.max(0, rect.left))),
    height: Math.max(1, Math.min(rect.height, window.innerHeight - Math.max(0, rect.top))),
  };
})()`);
await new Promise((resolve) => setTimeout(resolve, 250));
const canvasOnlyScreenshot = await command('Page.captureScreenshot', {
  format: 'png',
  captureBeyondViewport: false,
  clip: { ...canvasClip, scale: 1 },
});
fs.writeFileSync(canvasOnlyPath, Buffer.from(canvasOnlyScreenshot.data, 'base64'));
await evaluate(`(() => {
  for (const [element, visibility] of window.__phase385HiddenElements || []) {
    element.style.visibility = visibility;
  }
  delete window.__phase385HiddenElements;
})()`);

const compositorPixels = await evaluate(`(async () => {
  const image = new Image();
  image.src = ${JSON.stringify(`data:image/png;base64,${canvasOnlyScreenshot.data}`)};
  await image.decode();
  const probe = document.createElement('canvas');
  probe.width = image.naturalWidth;
  probe.height = image.naturalHeight;
  const context = probe.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, probe.width, probe.height).data;
  const pixelCount = probe.width * probe.height;
  const stride = Math.max(1, Math.floor(pixelCount / 100000));
  const colors = new Set();
  let sampled = 0;
  let nonBlack = 0;
  let colored = 0;
  let luminanceTotal = 0;
  for (let pixel = 0; pixel < pixelCount; pixel += stride) {
    const offset = pixel * 4;
    const r = pixels[offset];
    const g = pixels[offset + 1];
    const b = pixels[offset + 2];
    const luminance = (r + g + b) / 3;
    sampled += 1;
    luminanceTotal += luminance;
    if (r + g + b > 15) nonBlack += 1;
    if (Math.max(r, g, b) - Math.min(r, g, b) > 10) colored += 1;
    colors.add(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4));
  }
  const nonBlackRatio = nonBlack / Math.max(sampled, 1);
  const coloredRatio = colored / Math.max(sampled, 1);
  return {
    width: probe.width,
    height: probe.height,
    sampledPixels: sampled,
    nonBlackPixels: nonBlack,
    coloredPixels: colored,
    nonBlackRatio,
    coloredRatio,
    meanLuminance: luminanceTotal / Math.max(sampled, 1),
    quantizedColorCount: colors.size,
    nonBlank: nonBlack > 100 && colors.size > 8,
  };
})()`);

const result = {
  before,
  click,
  after,
  screenshotPath,
  canvasOnlyPath,
  canvasClip,
  compositorPixels,
};
fs.writeFileSync(resultPath, `${JSON.stringify(result, null, 2)}\n`);
socket.close();
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
