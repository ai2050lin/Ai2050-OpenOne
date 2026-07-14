#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'tests/gpt5/result/phase417_native_generation_physical_atlas/screenshots');
const URL = 'http://127.0.0.1:5175/';
const SOURCE_ID = 'gpt5_phase417_native_generation_atlas';

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
    (sourceId) => {
      const source = document.querySelector('[aria-label="主工作台测试路线数据源"]');
      const dataset = document.querySelector('[aria-label="主工作台测试数据集"]');
      const items = [...(dataset?.options || [])].filter((option) => option.value && !option.value.startsWith('__'));
      return source?.value === sourceId && items.length === 3;
    },
    SOURCE_ID,
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
    const before = path.join(OUT, `phase417_${name}_${datasetId}_canvas.png`);
    const after = path.join(OUT, `phase417_${name}_${datasetId}_canvas_interaction.png`);
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
  const screenshot = path.join(OUT, `phase417_generation_atlas_${name}.png`);
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
      phase_id: 'Phase417-GenerationAtlasClientVisualCheck',
      url: URL,
      checks,
      pass: checks.every((check) => (
        check.source_present
        && check.dataset_count === 3
        && check.source_selector_inside_viewport
        && !check.horizontal_overflow
        && check.canvas_count > 0
        && check.console_errors.every(optionalBackend)
        && check.failed_responses.length === 0
      )),
      evidence_boundary: 'Rendering only; depth and call-order edges remain observational and non-causal.',
    };
    fs.writeFileSync(
      path.join(OUT, 'phase417_generation_atlas_client_visual_check.json'),
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
