#!/usr/bin/env node

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { chromium } = require('/home/rankrank/.hermes/hermes-agent/node_modules/playwright');

const ROOT = path.resolve(__dirname, '../..');
const OUT = process.env.VIS_CHECK_OUT
  ? path.resolve(ROOT, process.env.VIS_CHECK_OUT)
  : path.join(ROOT, 'tests/gpt5/result/phase525_world_query_stage_audit/screenshots');
const URL = process.env.VIS_CHECK_URL || process.env.PHASE525_CLIENT_URL || 'http://127.0.0.1:5175/';
const SOURCE_ID = process.env.VIS_CHECK_SOURCE_ID || 'gpt5_phase524_world_query_platform_atlas';
const PHASE_ID = process.env.VIS_CHECK_PHASE_ID || 'Phase525-WorldQueryClientVisualCheck';
const FILE_PREFIX = process.env.VIS_CHECK_FILE_PREFIX || 'phase525';
const OUTPUT_NAME = process.env.VIS_CHECK_OUTPUT_NAME || 'phase525_world_query_client_visual_check.json';

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

function readPngStats(buffer) {
  const signature = buffer.subarray(0, 8).toString('hex');
  if (signature !== '89504e470d0a1a0a') throw new Error('Unexpected PNG signature');
  let offset = 8;
  let width;
  let height;
  let bitDepth;
  let colorType;
  const idat = [];
  while (offset < buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.subarray(offset + 4, offset + 8).toString('ascii');
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
    } else if (type === 'IDAT') {
      idat.push(data);
    }
    offset += length + 12;
    if (type === 'IEND') break;
  }
  const channels = colorType === 2 ? 3 : colorType === 6 ? 4 : 0;
  if (bitDepth !== 8 || !channels) throw new Error(`Unsupported PNG format: ${bitDepth}/${colorType}`);
  const packed = zlib.inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const pixels = Buffer.alloc(stride * height);
  let sourceOffset = 0;
  for (let y = 0; y < height; y += 1) {
    const filter = packed[sourceOffset];
    sourceOffset += 1;
    const rowOffset = y * stride;
    const previousOffset = (y - 1) * stride;
    for (let x = 0; x < stride; x += 1) {
      const raw = packed[sourceOffset + x];
      const left = x >= channels ? pixels[rowOffset + x - channels] : 0;
      const up = y > 0 ? pixels[previousOffset + x] : 0;
      const upperLeft = y > 0 && x >= channels ? pixels[previousOffset + x - channels] : 0;
      let value;
      if (filter === 0) value = raw;
      else if (filter === 1) value = raw + left;
      else if (filter === 2) value = raw + up;
      else if (filter === 3) value = raw + Math.floor((left + up) / 2);
      else if (filter === 4) value = raw + paeth(left, up, upperLeft);
      else throw new Error(`Unsupported PNG filter: ${filter}`);
      pixels[rowOffset + x] = value & 0xff;
    }
    sourceOffset += stride;
  }

  const step = Math.max(1, Math.floor((width * height) / 100000));
  const bins = new Set();
  let count = 0;
  let mean = 0;
  let m2 = 0;
  let opaque = 0;
  for (let pixel = 0; pixel < width * height; pixel += step) {
    const index = pixel * channels;
    const r = pixels[index];
    const g = pixels[index + 1];
    const b = pixels[index + 2];
    const alpha = channels === 4 ? pixels[index + 3] : 255;
    const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    count += 1;
    const delta = luminance - mean;
    mean += delta / count;
    m2 += delta * (luminance - mean);
    if (alpha > 0) opaque += 1;
    bins.add(`${r >> 3}:${g >> 3}:${b >> 3}:${alpha >> 5}`);
  }
  return {
    width,
    height,
    sampled_pixels: count,
    opaque_fraction: opaque / count,
    luminance_mean: mean,
    luminance_stddev: Math.sqrt(m2 / Math.max(1, count - 1)),
    quantized_color_count: bins.size,
    sha256: crypto.createHash('sha256').update(buffer).digest('hex'),
  };
}

async function inspect(browser, name, viewport, inspectAllModels) {
  const page = await browser.newPage({ viewport });
  const consoleErrors = [];
  const failedResponses = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('response', (response) => {
    if (response.status() >= 400) failedResponses.push({ status: response.status(), url: response.url() });
  });
  await page.addInitScript(() => localStorage.setItem('fpMode', 'demo'));
  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(3500);

  const route = page.getByLabel('主工作台测试路线数据源');
  const dataset = page.getByLabel('主工作台测试数据集');
  await route.scrollIntoViewIfNeeded();
  await route.selectOption(SOURCE_ID);
  await page.waitForFunction(() => {
    const select = document.querySelector('[aria-label="主工作台测试数据集"]');
    return [...(select?.options || [])].filter((option) => option.value && !option.value.startsWith('__')).length === 3;
  });
  const datasetIds = await dataset.locator('option').evaluateAll((options) => options
    .map((option) => option.value)
    .filter((value) => value && !value.startsWith('__')));
  const idsToInspect = inspectAllModels ? datasetIds : datasetIds.slice(0, 1);
  const modelChecks = [];
  for (const datasetId of idsToInspect) {
    await dataset.selectOption(datasetId);
    await page.getByText('✓ 已加载', { exact: true }).waitFor({ state: 'visible', timeout: 15000 });
    await page.waitForTimeout(1200);
    const canvas = page.locator('canvas').first();
    await canvas.waitFor({ state: 'visible' });
    const beforePath = path.join(OUT, `${FILE_PREFIX}_${name}_${datasetId}_canvas.png`);
    const afterPath = path.join(OUT, `${FILE_PREFIX}_${name}_${datasetId}_canvas_interaction.png`);
    const beforeBuffer = await canvas.screenshot({ path: beforePath });
    await canvas.dispatchEvent('wheel', { deltaY: 420, bubbles: true, cancelable: true });
    await page.waitForTimeout(700);
    const afterBuffer = await canvas.screenshot({ path: afterPath });
    const before = readPngStats(beforeBuffer);
    const after = readPngStats(afterBuffer);
    modelChecks.push({
      dataset_id: datasetId,
      before,
      after,
      interaction_changed_pixels: before.sha256 !== after.sha256,
      canvas_nonblank: before.luminance_stddev > 4 && before.quantized_color_count > 32,
      canvas_screenshot: path.relative(ROOT, beforePath),
      interaction_canvas_screenshot: path.relative(ROOT, afterPath),
    });
  }
  const layout = await page.evaluate(() => ({
    document_client_width: document.documentElement.clientWidth,
    document_scroll_width: document.documentElement.scrollWidth,
    body_scroll_width: document.body.scrollWidth,
    canvas_count: document.querySelectorAll('canvas').length,
  }));
  const screenshotPath = path.join(OUT, `${FILE_PREFIX}_${name}_workspace.png`);
  await page.screenshot({ path: screenshotPath, fullPage: false });
  await page.close();
  return {
    name,
    viewport,
    dataset_ids: datasetIds,
    model_checks: modelChecks,
    layout,
    horizontal_overflow: layout.document_scroll_width > layout.document_client_width,
    console_errors: consoleErrors,
    failed_responses: failedResponses,
    screenshot: path.relative(ROOT, screenshotPath),
  };
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({
    executablePath: '/snap/bin/chromium',
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  try {
    const checks = [
      await inspect(browser, 'desktop_1440x900', { width: 1440, height: 900 }, true),
      await inspect(browser, 'mobile_390x844', { width: 390, height: 844 }, false),
    ];
    const allowedConsoleError = (message) => (
      message.includes('ERR_CONNECTION_REFUSED') || message.includes('Failed to load resource')
    );
    const result = {
      phase_id: PHASE_ID,
      source_id: SOURCE_ID,
      url: URL,
      checks,
      pass: checks.every((check) => (
        check.dataset_ids.length === 3
        && check.model_checks.every((model) => model.canvas_nonblank && model.interaction_changed_pixels)
        && !check.horizontal_overflow
        && check.layout.canvas_count > 0
        && check.console_errors.every(allowedConsoleError)
        && check.failed_responses.length === 0
      )),
    };
    const output = path.join(OUT, OUTPUT_NAME);
    fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    if (!result.pass) process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
