#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const ROOT = path.resolve(__dirname, '../..');
const PREFIX = process.env.ATLAS_PREFIX || 'phase418';
const OUT = path.join(ROOT, process.env.ATLAS_SCREENSHOT_DIR || 'tests/gpt5/result/phase418_interface_history_atlas/screenshots');
const URL = 'http://127.0.0.1:5175/';
const SOURCE_ID = process.env.ATLAS_SOURCE_ID || 'gpt5_phase418_interface_history_atlas';
const CHECK_PHASE_ID = process.env.ATLAS_CHECK_PHASE_ID || 'Phase418-InterfaceHistoryAtlasClientVisualCheck';
const CHECK_FILENAME = process.env.ATLAS_CHECK_FILENAME || 'phase418_interface_history_client_visual_check.json';
const EXPECTED_DATASET_COUNT = Number(process.env.ATLAS_EXPECTED_DATASET_COUNT || '3');

async function values(select) {
  return select.locator('option').evaluateAll((options) => options
    .map((option) => option.value)
    .filter((value) => value && !value.startsWith('__')));
}

async function inspect(browser, name, viewport, allModels) {
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
  await page.waitForTimeout(4000);
  const sourceSelect = page.getByLabel('主工作台测试路线数据源');
  const datasetSelect = page.getByLabel('主工作台测试数据集');
  await sourceSelect.waitFor({ state: 'visible' });
  const sourceOptions = await sourceSelect.locator('option').evaluateAll((options) => options.map((option) => option.value));
  await sourceSelect.selectOption(SOURCE_ID);
  await page.waitForFunction(
    ([sourceId, expectedCount]) => {
      const source = document.querySelector('[aria-label="主工作台测试路线数据源"]');
      const dataset = document.querySelector('[aria-label="主工作台测试数据集"]');
      const items = [...(dataset?.options || [])].filter((option) => option.value && !option.value.startsWith('__'));
      return source?.value === sourceId && items.length === expectedCount;
    },
    [SOURCE_ID, EXPECTED_DATASET_COUNT],
    { timeout: 12000 }
  );
  const datasetIds = await values(datasetSelect);
  const checks = [];
  for (const datasetId of allModels ? datasetIds : datasetIds.slice(0, 1)) {
    await datasetSelect.selectOption(datasetId);
    await page.getByText('✓ 已加载', { exact: true }).waitFor({ state: 'visible', timeout: 12000 });
    await page.waitForTimeout(1200);
    const canvas = page.locator('canvas').first();
    await canvas.waitFor({ state: 'visible' });
    const before = path.join(OUT, `${PREFIX}_${name}_${datasetId}_canvas.png`);
    const after = path.join(OUT, `${PREFIX}_${name}_${datasetId}_canvas_interaction.png`);
    await canvas.screenshot({ path: before });
    await canvas.dispatchEvent('wheel', { deltaY: 360, bubbles: true, cancelable: true });
    await page.waitForTimeout(650);
    await canvas.screenshot({ path: after });
    await canvas.dispatchEvent('wheel', { deltaY: -360, bubbles: true, cancelable: true });
    checks.push({
      dataset_id: datasetId,
      canvas_screenshot: path.relative(ROOT, before),
      interaction_canvas_screenshot: path.relative(ROOT, after),
    });
  }
  const bounds = await sourceSelect.boundingBox();
  const layout = await page.evaluate(() => ({
    width: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    canvasCount: document.querySelectorAll('canvas').length,
  }));
  const screenshot = path.join(OUT, `${PREFIX}_atlas_${name}.png`);
  await page.screenshot({ path: screenshot, fullPage: false });
  await page.close();
  return {
    name,
    viewport,
    source_present: sourceOptions.includes(SOURCE_ID),
    dataset_count: datasetIds.length,
    model_checks: checks,
    source_selector_inside_viewport: Boolean(bounds && bounds.x >= 0 && bounds.x + bounds.width <= viewport.width),
    horizontal_overflow: layout.scrollWidth > layout.width,
    canvas_count: layout.canvasCount,
    console_errors: consoleErrors,
    failed_responses: failedResponses,
    screenshot: path.relative(ROOT, screenshot),
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
    const optionalBackend = (message) => message.includes('ERR_CONNECTION_REFUSED') || message.includes('Failed to load resource');
    const result = {
      phase_id: CHECK_PHASE_ID,
      url: URL,
      checks,
      pass: checks.every((check) => (
        check.source_present
        && check.dataset_count === EXPECTED_DATASET_COUNT
        && check.source_selector_inside_viewport
        && !check.horizontal_overflow
        && check.canvas_count > 0
        && check.console_errors.every(optionalBackend)
        && check.failed_responses.length === 0
      )),
      evidence_boundary: 'Rendering only; registered contrast and depth-order edges remain observational and non-causal.',
    };
    fs.writeFileSync(
      path.join(OUT, CHECK_FILENAME),
      `${JSON.stringify(result, null, 2)}\n`
    );
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
