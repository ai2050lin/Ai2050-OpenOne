#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const cdpPort = process.env.CDP_PORT || process.argv[2] || '9224';
const appPort = process.env.APP_PORT || process.argv[3] || '5179';
const root = process.cwd();
const outputDir = path.resolve(
  root,
  process.env.VIS_CHECK_OUTPUT_DIR
    || 'tests/gpt5/result/phase550_matched_route_identity_atlas/client_visual_check',
);
const resultPath = path.join(
  outputDir,
  process.env.VIS_CHECK_RESULT_FILE || 'phase550_client_visual_check.json',
);
const sourceId = process.env.VIS_CHECK_SOURCE_ID || 'gpt5_phase549_route_answer_factorial';
const datasets = JSON.parse(process.env.VIS_CHECK_DATASETS || JSON.stringify([
  { id: 'phase549_qwen3', model: 'qwen3' },
  { id: 'phase549_glm4', model: 'glm4' },
  { id: 'phase549_deepseek7b', model: 'deepseek7b' },
]));
const phaseText = process.env.VIS_CHECK_PHASE_TEXT || 'Phase549';
const evidenceText = process.env.VIS_CHECK_EVIDENCE_TEXT || '答案身份';
const screenshotPrefix = process.env.VIS_CHECK_SCREENSHOT_PREFIX || 'phase550';
const schemaVersion = process.env.VIS_CHECK_SCHEMA_VERSION || 'phase550_client_visual_check.v1';
const modelKeys = { qwen3: 'qwen3-4b', glm4: 'glm4-9b', deepseek7b: 'ds7b' };
const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

fs.mkdirSync(outputDir, { recursive: true });

const pages = await fetch(`http://127.0.0.1:${cdpPort}/json/list`).then((response) => response.json());
const page = pages.find(
  (entry) => entry.type === 'page' && entry.url.includes(`127.0.0.1:${appPort}`),
);
if (!page) throw new Error(`No client page target found for port ${appPort}`);

const socket = new WebSocket(page.webSocketDebuggerUrl);
const pending = new Map();
const browserEvents = { consoleErrors: [], exceptions: [], failedRequests: [], httpErrors: [] };
let requestId = 0;

socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const callback = pending.get(message.id);
    if (!callback) return;
    pending.delete(message.id);
    if (message.error) callback.reject(new Error(JSON.stringify(message.error)));
    else callback.resolve(message.result);
    return;
  }
  if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') {
    browserEvents.consoleErrors.push(message.params.args.map((item) => item.value ?? item.description).join(' '));
  }
  if (message.method === 'Runtime.exceptionThrown') {
    browserEvents.exceptions.push(message.params.exceptionDetails?.text ?? 'unknown exception');
  }
  if (message.method === 'Network.loadingFailed') {
    const errorText = message.params.errorText || '';
    if (!errorText.includes('ERR_ABORTED')) browserEvents.failedRequests.push(errorText);
  }
  if (message.method === 'Network.responseReceived' && message.params.response.status >= 400) {
    browserEvents.httpErrors.push({
      status: message.params.response.status,
      url: message.params.response.url,
    });
  }
};

await new Promise((resolve, reject) => {
  socket.onopen = resolve;
  socket.onerror = reject;
});

function send(method, params = {}) {
  requestId += 1;
  const id = requestId;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

async function evaluate(expression) {
  const result = await send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}

async function selectValue(ariaLabel, value) {
  return evaluate(`(() => {
    const select = document.querySelector(${JSON.stringify(`select[aria-label="${ariaLabel}"]`)});
    if (!select) return { ok: false, reason: 'select missing' };
    const option = Array.from(select.options).find((row) => row.value === ${JSON.stringify(value)});
    if (!option) {
      return { ok: false, reason: 'option missing', options: Array.from(select.options).map((row) => row.value) };
    }
    select.value = option.value;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    return { ok: true, value: select.value, label: option.textContent.trim() };
  })()`);
}

async function canvasClip() {
  return evaluate(`(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, rect.left);
    const y = Math.max(0, rect.top);
    return {
      x,
      y,
      width: Math.max(1, Math.min(rect.right, window.innerWidth) - x),
      height: Math.max(1, Math.min(rect.bottom, window.innerHeight) - y),
      scale: 1,
    };
  })()`);
}

async function captureCanvas() {
  const clip = await canvasClip();
  if (!clip) throw new Error('Canvas missing');
  const screenshot = await send('Page.captureScreenshot', {
    format: 'png',
    fromSurface: true,
    captureBeyondViewport: false,
    clip,
  });
  return { clip, base64: screenshot.data };
}

async function analyzePng(base64) {
  return evaluate(`(async () => {
    const image = new Image();
    image.src = ${JSON.stringify(`data:image/png;base64,${base64}`)};
    await image.decode();
    const probe = document.createElement('canvas');
    probe.width = image.naturalWidth;
    probe.height = image.naturalHeight;
    const context = probe.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, probe.width, probe.height).data;
    const pixelCount = probe.width * probe.height;
    const stride = Math.max(1, Math.floor(pixelCount / 120000));
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
      sampled += 1;
      luminanceTotal += (r + g + b) / 3;
      if (r + g + b > 15) nonBlack += 1;
      if (Math.max(r, g, b) - Math.min(r, g, b) > 10) colored += 1;
      colors.add(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4));
    }
    return {
      width: probe.width,
      height: probe.height,
      sampledPixels: sampled,
      nonBlackRatio: nonBlack / Math.max(sampled, 1),
      coloredRatio: colored / Math.max(sampled, 1),
      meanLuminance: luminanceTotal / Math.max(sampled, 1),
      quantizedColorCount: colors.size,
      nonBlank: nonBlack > 100 && colors.size > 12,
    };
  })()`);
}

async function pixelDifference(leftBase64, rightBase64) {
  return evaluate(`(async () => {
    async function loadImage(base64) {
      const image = new Image();
      image.src = 'data:image/png;base64,' + base64;
      await image.decode();
      return image;
    }
    const [left, right] = await Promise.all([
      loadImage(${JSON.stringify(leftBase64)}),
      loadImage(${JSON.stringify(rightBase64)}),
    ]);
    const width = Math.min(left.naturalWidth, right.naturalWidth);
    const height = Math.min(left.naturalHeight, right.naturalHeight);
    const probe = document.createElement('canvas');
    probe.width = width;
    probe.height = height;
    const context = probe.getContext('2d', { willReadFrequently: true });
    context.drawImage(left, 0, 0, width, height);
    const a = context.getImageData(0, 0, width, height).data;
    context.clearRect(0, 0, width, height);
    context.drawImage(right, 0, 0, width, height);
    const b = context.getImageData(0, 0, width, height).data;
    const pixelCount = width * height;
    const stride = Math.max(1, Math.floor(pixelCount / 120000));
    let sampled = 0;
    let changed = 0;
    for (let pixel = 0; pixel < pixelCount; pixel += stride) {
      const offset = pixel * 4;
      const delta = Math.abs(a[offset] - b[offset])
        + Math.abs(a[offset + 1] - b[offset + 1])
        + Math.abs(a[offset + 2] - b[offset + 2]);
      sampled += 1;
      if (delta > 24) changed += 1;
    }
    return { sampledPixels: sampled, changedPixels: changed, changedRatio: changed / Math.max(sampled, 1) };
  })()`);
}

async function pageState(expectedDataset) {
  return evaluate(`(() => {
    const source = document.querySelector('select[aria-label="主工作台测试路线数据源"]');
    const dataset = document.querySelector('select[aria-label="主工作台测试数据集"]');
    const architectureModel = document.querySelector('select[aria-label="主工作台模型架构"]');
    const canvas = document.querySelector('canvas');
    const canvasRect = canvas?.getBoundingClientRect();
    const failureText = Array.from(document.querySelectorAll('body *'))
      .map((node) => node.children.length === 0 ? (node.textContent || '').trim() : '')
      .find((text) => text.includes('Unsupported schema') || text.includes('加载失败')) || null;
    return {
      source: source?.value,
      dataset: dataset?.value,
      architectureModel: architectureModel?.value,
      expectedDataset: ${JSON.stringify(expectedDataset)},
      phaseTextPresent: document.body.innerText.includes(${JSON.stringify(phaseText)}),
      evidenceTextPresent: document.body.innerText.includes(${JSON.stringify(evidenceText)}),
      failureText,
      canvasCount: document.querySelectorAll('canvas').length,
      canvasRect: canvasRect ? {
        left: canvasRect.left, top: canvasRect.top, right: canvasRect.right,
        bottom: canvasRect.bottom, width: canvasRect.width, height: canvasRect.height,
      } : null,
      viewport: { width: window.innerWidth, height: window.innerHeight },
      horizontalOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    };
  })()`);
}

async function dragCanvas() {
  const rect = await evaluate(`(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return null;
    const bounds = canvas.getBoundingClientRect();
    return { x: bounds.left + bounds.width * 0.5, y: bounds.top + bounds.height * 0.5 };
  })()`);
  if (!rect) throw new Error('Cannot drag missing canvas');
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: rect.x, y: rect.y });
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: rect.x, y: rect.y, button: 'left', clickCount: 1 });
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: rect.x + 96, y: rect.y + 38, button: 'left' });
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: rect.x + 96, y: rect.y + 38, button: 'left', clickCount: 1 });
}

async function captureFull(filename) {
  const screenshot = await send('Page.captureScreenshot', {
    format: 'png', fromSurface: true, captureBeyondViewport: false,
  });
  fs.writeFileSync(path.join(outputDir, filename), Buffer.from(screenshot.data, 'base64'));
}

async function verifyDataset(dataset, viewportName) {
  const selected = await selectValue('主工作台测试数据集', dataset.id);
  if (!selected.ok) throw new Error(JSON.stringify(selected));
  await wait(4500);
  const state = await pageState(dataset.id);
  const steadyA = await captureCanvas();
  await wait(350);
  const steadyB = await captureCanvas();
  await dragCanvas();
  await wait(650);
  const interacted = await captureCanvas();
  const [pixels, steadyDifference, interactionDifference] = await Promise.all([
    analyzePng(interacted.base64),
    pixelDifference(steadyA.base64, steadyB.base64),
    pixelDifference(steadyB.base64, interacted.base64),
  ]);
  await captureFull(`${screenshotPrefix}_${dataset.model}_${viewportName}.png`);
  const interactionChanged = interactionDifference.changedRatio > Math.max(
    0.002,
    steadyDifference.changedRatio * 1.25,
  );
  return {
    model: dataset.model,
    dataset: dataset.id,
    selected,
    state,
    canvasPixels: pixels,
    steadyDifference,
    interactionDifference,
    interactionChanged,
    canvasSha256: crypto.createHash('sha256').update(interacted.base64).digest('hex'),
  };
}

await send('Page.enable');
await send('Runtime.enable');
await send('Network.enable');
await send('Emulation.setDeviceMetricsOverride', {
  width: 1440, height: 900, deviceScaleFactor: 1, mobile: false,
});
await send('Page.reload', { ignoreCache: true });
await wait(3500);

const sourceSelected = await selectValue('主工作台测试路线数据源', sourceId);
if (!sourceSelected.ok) throw new Error(JSON.stringify(sourceSelected));
await wait(2500);

const desktop = [];
for (const dataset of datasets) desktop.push(await verifyDataset(dataset, 'desktop'));

await send('Emulation.setDeviceMetricsOverride', {
  width: 390, height: 844, deviceScaleFactor: 1, mobile: true,
});
await wait(1500);
const mobile = await verifyDataset(datasets[0], 'mobile');

const checks = [...desktop, mobile];
const passed = checks.every((check) => (
  check.state.source === sourceId
  && check.state.dataset === check.dataset
  && check.state.architectureModel === modelKeys[check.model]
  && check.state.phaseTextPresent
  && check.state.evidenceTextPresent
  && !check.state.failureText
  && check.state.canvasCount > 0
  && check.state.horizontalOverflow <= 1
  && check.canvasPixels.nonBlank
  && check.interactionChanged
)) && Object.values(browserEvents).every((items) => items.length === 0);

const result = {
  schema_version: schemaVersion,
  source_id: sourceId,
  desktop,
  mobile,
  browser_events: browserEvents,
  passed,
};
fs.writeFileSync(resultPath, `${JSON.stringify(result, null, 2)}\n`);
socket.close();
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
if (!passed) process.exitCode = 1;
