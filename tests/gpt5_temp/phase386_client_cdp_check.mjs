import fs from 'node:fs';
import path from 'node:path';

const port = process.argv[2] || '9223';
const screenshotPath = process.argv[3];
const resultPath = process.argv[4];
const width = Number(process.argv[5] || 1440);
const height = Number(process.argv[6] || 1000);
if (!screenshotPath || !resultPath) {
  throw new Error('usage: node phase386_client_cdp_check.mjs PORT SCREENSHOT RESULT [WIDTH HEIGHT]');
}

const pages = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
const page = pages.find((item) => item.type === 'page' && item.url.includes('127.0.0.1:5173'));
if (!page) throw new Error('Vite page not found');
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
  const handlers = pending.get(payload.id);
  pending.delete(payload.id);
  if (payload.error) handlers.reject(new Error(JSON.stringify(payload.error)));
  else handlers.resolve(payload.result);
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
await command('Emulation.setDeviceMetricsOverride', {
  width,
  height,
  deviceScaleFactor: 1,
  mobile: width <= 768,
});
await command('Page.reload', { ignoreCache: true });
await new Promise((resolve) => setTimeout(resolve, 3000));
await evaluate(`(() => {
  const global = document.querySelector('button[aria-label="Global Strategy"]');
  if (global) global.click();
  return Boolean(global);
})()`);
await new Promise((resolve) => setTimeout(resolve, 500));
await evaluate(`(() => {
  const button = [...document.querySelectorAll('button')]
    .find((item) => item.textContent.trim() === '语言分析');
  if (button) button.click();
  return Boolean(button);
})()`);
await new Promise((resolve) => setTimeout(resolve, 2500));
await evaluate(`(() => {
  const card = [...document.querySelectorAll('button')]
    .find((item) => item.innerText.includes('10 physical relations / 0 causal paths'));
  if (card) card.click();
  const heading = [...document.querySelectorAll('body *')]
    .find((item) => item.children.length === 0 && item.textContent.trim() === '最新成果');
  if (heading) heading.scrollIntoView({ block: 'start' });
  return { card: Boolean(card), heading: Boolean(heading) };
})()`);
await new Promise((resolve) => setTimeout(resolve, 1000));

const result = await evaluate(`(() => {
  const text = document.body.innerText;
  const panels = [...document.querySelectorAll('.workspace-panel')].map((panel) => {
    const rect = panel.getBoundingClientRect();
    return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom };
  });
  const overlaps = [];
  for (let i = 0; i < panels.length; i += 1) {
    for (let j = i + 1; j < panels.length; j += 1) {
      const w = Math.max(0, Math.min(panels[i].right, panels[j].right) - Math.max(panels[i].left, panels[j].left));
      const h = Math.max(0, Math.min(panels[i].bottom, panels[j].bottom) - Math.max(panels[i].top, panels[j].top));
      if (w > 0 && h > 0) overlaps.push({ i, j, area: w * h });
    }
  }
  return {
    viewport: { width: window.innerWidth, height: window.innerHeight },
    bodyTextHasPhase386: text.includes('Phase386-StageMerge'),
    bodyTextHasLatestValue: text.includes('10 physical relations / 0 causal paths'),
    bodyTextHasPhysicalRelations: text.includes('物理预测关系：10/12'),
    bodyTextHasUpstreamRelations: text.includes('上游描述预测关系：1/10'),
    bodyTextHasMlpBoundary: text.includes('物理 MLP 通道关系：0/5'),
    bodyTextHasNeuronBoundary: text.includes('单神经元因果路径：0/72'),
    bodyTextHasNoCausalClaim: text.includes('当前没有 MLP 神经元通道关系，也没有因果必要性或完整语言路径'),
    panelOverlaps: overlaps,
    bodyScrollWidth: document.body.scrollWidth,
    viewportWidth: window.innerWidth,
  };
})()`);
const screenshot = await command('Page.captureScreenshot', {
  format: 'png',
  captureBeyondViewport: false,
});
fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, 'base64'));
fs.writeFileSync(resultPath, `${JSON.stringify(result, null, 2)}\n`);
socket.close();
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
