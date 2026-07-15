#!/usr/bin/env node
import fs from 'node:fs';

const sourceId = process.env.ATLAS_SOURCE_ID || 'gpt5_phase431_position_time';
const datasetId = process.env.ATLAS_DATASET_ID || 'phase431_qwen3_position_time';
const phaseText = process.env.ATLAS_PHASE_TEXT || 'Phase431';
const screenshotPrefix = process.env.ATLAS_SCREENSHOT_PREFIX || 'phase431_loaded';
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const pages = await fetch('http://127.0.0.1:9222/json/list').then((response) => response.json());
const page = pages.find((entry) => entry.type === 'page');
if (!page) throw new Error('No Chromium page target');

const socket = new WebSocket(page.webSocketDebuggerUrl);
const pending = new Map();
let requestId = 0;
socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (!message.id) return;
  const callback = pending.get(message.id);
  if (!callback) return;
  pending.delete(message.id);
  if (message.error) callback.reject(new Error(JSON.stringify(message.error)));
  else callback.resolve(message.result);
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

await send('Page.enable');
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', {
  width: 1440,
  height: 900,
  deviceScaleFactor: 1,
  mobile: false,
});
await wait(3000);

const sourceSelected = await evaluate(`(() => {
  const select = document.querySelector('select[aria-label="主工作台测试路线数据源"]');
  if (!select) return { ok: false, reason: 'source select missing' };
  const option = Array.from(select.options).find((row) => row.value === ${JSON.stringify(sourceId)});
  if (!option) return { ok: false, reason: 'requested source option missing', options: Array.from(select.options).map((row) => row.value) };
  select.value = option.value;
  select.dispatchEvent(new Event('change', { bubbles: true }));
  return { ok: true, value: select.value, label: option.textContent };
})()`);
if (!sourceSelected.ok) throw new Error(JSON.stringify(sourceSelected));
await wait(2500);

const datasetSelected = await evaluate(`(() => {
  const select = document.querySelector('select[aria-label="主工作台测试数据集"]');
  if (!select) return { ok: false, reason: 'dataset select missing' };
  const option = Array.from(select.options).find((row) => row.value === ${JSON.stringify(datasetId)});
  if (!option) return { ok: false, reason: 'requested dataset missing', options: Array.from(select.options).map((row) => ({ value: row.value, text: row.textContent })) };
  select.value = option.value;
  select.dispatchEvent(new Event('change', { bubbles: true }));
  return { ok: true, value: select.value, label: option.textContent };
})()`);
if (!datasetSelected.ok) throw new Error(JSON.stringify(datasetSelected));
await wait(5000);

const state = await evaluate(`(() => {
  const source = document.querySelector('select[aria-label="主工作台测试路线数据源"]');
  const dataset = document.querySelector('select[aria-label="主工作台测试数据集"]');
  const canvases = Array.from(document.querySelectorAll('canvas'));
  return {
    source: source?.value,
    dataset: dataset?.value,
    loadedText: document.body.innerText.includes('已加载'),
    phaseText: document.body.innerText.includes(${JSON.stringify(phaseText)}),
    errorText: Array.from(document.querySelectorAll('div')).map((node) => node.textContent || '').find((text) => text.includes('Unsupported schema') || text.includes('加载失败')) || null,
    canvasCount: canvases.length,
    canvases: canvases.map((canvas) => ({ width: canvas.width, height: canvas.height, dataUrlLength: canvas.toDataURL().length })),
  };
})()`);
if (
  state.source !== sourceId
  || state.dataset !== datasetId
  || !state.loadedText
  || !state.phaseText
  || state.errorText
  || state.canvasCount === 0
  || !state.canvases.some((canvas) => canvas.width >= 800 && canvas.height >= 500 && canvas.dataUrlLength > 1000)
) {
  throw new Error(`Atlas client state invalid: ${JSON.stringify(state)}`);
}

const desktop = await send('Page.captureScreenshot', { format: 'png', fromSurface: true });
fs.writeFileSync(
  `/home/rankrank/Documents/OpenOne/Ai2050-OpenOne/frontend/logs/${screenshotPrefix}_desktop.png`,
  Buffer.from(desktop.data, 'base64'),
);
await send('Emulation.setDeviceMetricsOverride', {
  width: 390,
  height: 844,
  deviceScaleFactor: 1,
  mobile: true,
});
await wait(1000);
const mobile = await send('Page.captureScreenshot', { format: 'png', fromSurface: true });
fs.writeFileSync(
  `/home/rankrank/Documents/OpenOne/Ai2050-OpenOne/frontend/logs/${screenshotPrefix}_mobile.png`,
  Buffer.from(mobile.data, 'base64'),
);

console.log(JSON.stringify({ sourceSelected, datasetSelected, state }, null, 2));
socket.close();
