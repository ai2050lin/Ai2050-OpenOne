#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const ROOT = path.resolve(__dirname, '../..');
const PHASE_SLUG = process.env.PHASE_VIS_SLUG || 'phase426_exact_position_role_validation';
const CHECK_ID = process.env.PHASE_CHECK_ID || 'Phase426-CanvasPixelCheck';
const OUT = path.join(ROOT, 'tests/gpt5/result', PHASE_SLUG, 'screenshots');
const RESULT = path.join(OUT, `${PHASE_SLUG}_canvas_pixel_check.json`);
const URL = process.env.PHASE_VIS_URL || 'http://127.0.0.1:5175/';
const SOURCE_ID = process.env.PHASE_SOURCE_ID || 'gpt5_phase426_exact_position_role_validation';
const EXPECTED_DATASET_COUNT = Number(process.env.PHASE_EXPECTED_DATASET_COUNT || 3);

async function imagePairStats(page, before, after) {
  return page.evaluate(async ([beforeBase64, afterBase64]) => {
    const decode = (base64) => new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        const probe = document.createElement('canvas');
        probe.width = image.naturalWidth;
        probe.height = image.naturalHeight;
        const context = probe.getContext('2d', { willReadFrequently: true });
        context.drawImage(image, 0, 0);
        resolve({
          width: probe.width,
          height: probe.height,
          pixels: context.getImageData(0, 0, probe.width, probe.height).data,
        });
      };
      image.onerror = reject;
      image.src = `data:image/png;base64,${base64}`;
    });
    const first = await decode(beforeBase64);
    const second = await decode(afterBase64);
    const pixelCount = first.width * first.height;
    const stride = Math.max(1, Math.floor(pixelCount / 120000));
    const colors = new Set();
    let minLuminance = 255;
    let maxLuminance = 0;
    let sampledPixels = 0;
    let opaquePixels = 0;
    let changedPixels = 0;
    let absoluteDifference = 0;
    for (let pixel = 0; pixel < pixelCount; pixel += stride) {
      const offset = pixel * 4;
      const r = first.pixels[offset];
      const g = first.pixels[offset + 1];
      const b = first.pixels[offset + 2];
      const a = first.pixels[offset + 3];
      const luminance = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
      minLuminance = Math.min(minLuminance, luminance);
      maxLuminance = Math.max(maxLuminance, luminance);
      colors.add(`${r >> 3}:${g >> 3}:${b >> 3}:${a >> 5}`);
      if (a > 0) opaquePixels += 1;
      let difference = 0;
      for (let channel = 0; channel < 4; channel += 1) {
        difference += Math.abs(first.pixels[offset + channel] - second.pixels[offset + channel]);
      }
      if (difference > 8) changedPixels += 1;
      absoluteDifference += difference;
      sampledPixels += 1;
    }
    return {
      width: first.width,
      height: first.height,
      sampled_pixels: sampledPixels,
      opaque_fraction: opaquePixels / sampledPixels,
      quantized_unique_color_count: colors.size,
      luminance_range: maxLuminance - minLuminance,
      interaction_changed_pixel_fraction: changedPixels / sampledPixels,
      interaction_mean_absolute_rgba_difference: absoluteDifference / (sampledPixels * 4),
    };
  }, [before.toString('base64'), after.toString('base64')]);
}

async function inspect(browser, viewportName, viewport) {
  const page = await browser.newPage({ viewport });
  const consoleErrors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  await page.addInitScript(() => localStorage.setItem('fpMode', 'demo'));
  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(3500);
  const source = page.getByLabel('主工作台测试路线数据源');
  const dataset = page.getByLabel('主工作台测试数据集');
  await source.selectOption(SOURCE_ID);
  await page.waitForFunction(({ sourceId, expectedCount }) => {
    const sourceElement = document.querySelector('[aria-label="主工作台测试路线数据源"]');
    const datasetElement = document.querySelector('[aria-label="主工作台测试数据集"]');
    const count = [...(datasetElement?.options || [])]
      .filter((option) => option.value && !option.value.startsWith('__')).length;
    return sourceElement?.value === sourceId && count === expectedCount;
  }, { sourceId: SOURCE_ID, expectedCount: EXPECTED_DATASET_COUNT }, { timeout: 12000 });
  const datasetIds = await dataset.locator('option').evaluateAll((options) => options
    .map((option) => option.value)
    .filter((value) => value && !value.startsWith('__')));
  const datasetChecks = [];
  for (const datasetId of datasetIds) {
    await dataset.selectOption(datasetId);
    await page.getByText('✓ 已加载', { exact: true }).waitFor({ state: 'visible', timeout: 12000 });
    await page.waitForTimeout(900);
    const canvas = page.locator('canvas').first();
    await canvas.waitFor({ state: 'visible' });
    const before = await canvas.screenshot();
    await canvas.dispatchEvent('wheel', { deltaY: 360, bubbles: true, cancelable: true });
    await page.waitForTimeout(650);
    const after = await canvas.screenshot();
    const safeDatasetId = datasetId.replace(/[^a-zA-Z0-9_.-]+/g, '_');
    fs.writeFileSync(path.join(OUT, `${viewportName}_${safeDatasetId}_before.png`), before);
    fs.writeFileSync(path.join(OUT, `${viewportName}_${safeDatasetId}_after.png`), after);
    const stats = await imagePairStats(page, before, after);
    datasetChecks.push({
      dataset_id: datasetId,
      ...stats,
      nonblank: stats.quantized_unique_color_count >= 16 && stats.luminance_range >= 12,
      interaction_changed: stats.interaction_changed_pixel_fraction > 0.0001,
    });
    await canvas.dispatchEvent('wheel', { deltaY: -360, bubbles: true, cancelable: true });
  }
  const layout = await page.evaluate(() => ({
    client_width: document.documentElement.clientWidth,
    scroll_width: document.documentElement.scrollWidth,
    canvas_count: document.querySelectorAll('canvas').length,
  }));
  await page.close();
  return {
    viewport: viewportName,
    dimensions: viewport,
    dataset_checks: datasetChecks,
    horizontal_overflow: layout.scroll_width > layout.client_width,
    canvas_count: layout.canvas_count,
    console_errors: consoleErrors,
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
      await inspect(browser, 'desktop_1440x900', { width: 1440, height: 900 }),
      await inspect(browser, 'mobile_390x844', { width: 390, height: 844 }),
    ];
    const optionalBackend = (message) => message.includes('ERR_CONNECTION_REFUSED')
      || message.includes('Failed to load resource');
    const result = {
      phase_id: CHECK_ID,
      url: URL,
      checks,
      pass: checks.every((check) => (
        check.dataset_checks.length === EXPECTED_DATASET_COUNT
        && check.dataset_checks.every((dataset) => dataset.nonblank && dataset.interaction_changed)
        && !check.horizontal_overflow
        && check.canvas_count > 0
        && check.console_errors.every(optionalBackend)
      )),
      evidence_boundary: 'Rendering and interaction pixels only; no mechanism or causal evidence is added.',
    };
    fs.writeFileSync(RESULT, `${JSON.stringify(result, null, 2)}\n`);
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
