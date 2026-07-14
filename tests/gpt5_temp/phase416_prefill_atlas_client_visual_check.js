#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'tests/gpt5/result/phase416_formal_world_physical_atlas/screenshots');
const URL = 'http://127.0.0.1:5175/';
const SOURCE_ID = 'gpt5_phase416_formal_prefill_atlas';

async function datasetValues(select) {
  return select.locator('option').evaluateAll((options) => options
    .map((option) => option.value)
    .filter((value) => value && !value.startsWith('__')));
}

async function inspectViewport(browser, name, viewport, loadAllModels) {
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

  const routeSelect = page.getByLabel('主工作台测试路线数据源');
  const datasetSelect = page.getByLabel('主工作台测试数据集');
  await routeSelect.waitFor({ state: 'visible' });
  const sourceOptions = await routeSelect.locator('option').evaluateAll((options) => options.map((option) => option.value));
  await routeSelect.selectOption(SOURCE_ID);
  await page.waitForFunction(
    (sourceId) => {
      const route = document.querySelector('[aria-label="主工作台测试路线数据源"]');
      const dataset = document.querySelector('[aria-label="主工作台测试数据集"]');
      const values = [...(dataset?.options || [])]
        .map((option) => option.value)
        .filter((value) => value && !value.startsWith('__'));
      return route?.value === sourceId && values.length === 3;
    },
    SOURCE_ID,
    { timeout: 12000 }
  );

  const values = await datasetValues(datasetSelect);
  const selectedValues = loadAllModels ? values : values.slice(0, 1);
  const modelChecks = [];
  for (const value of selectedValues) {
    await datasetSelect.selectOption(value);
    await page.getByText('✓ 已加载', { exact: true }).waitFor({ state: 'visible', timeout: 12000 });
    await page.waitForTimeout(1200);
    const canvas = page.locator('canvas').first();
    await canvas.waitFor({ state: 'visible' });
    const before = path.join(OUT, `phase416_${name}_${value}_canvas.png`);
    const after = path.join(OUT, `phase416_${name}_${value}_canvas_interaction.png`);
    await canvas.screenshot({ path: before });
    await canvas.dispatchEvent('wheel', { deltaY: 360, bubbles: true, cancelable: true });
    await page.waitForTimeout(650);
    await canvas.screenshot({ path: after });
    await canvas.dispatchEvent('wheel', { deltaY: -360, bubbles: true, cancelable: true });
    modelChecks.push({
      dataset_id: value,
      canvas_screenshot: path.relative(ROOT, before),
      interaction_canvas_screenshot: path.relative(ROOT, after),
    });
  }

  const routeBounds = await routeSelect.boundingBox();
  const layout = await page.evaluate(() => ({
    documentClientWidth: document.documentElement.clientWidth,
    documentScrollWidth: document.documentElement.scrollWidth,
    canvasCount: document.querySelectorAll('canvas').length,
  }));
  const screenshot = path.join(OUT, `phase416_prefill_atlas_${name}.png`);
  await page.screenshot({ path: screenshot, fullPage: false });
  await page.close();
  return {
    name,
    viewport,
    source_present: sourceOptions.includes(SOURCE_ID),
    observed_dataset_count: values.length,
    model_checks: modelChecks,
    route_selector_inside_viewport: Boolean(
      routeBounds && routeBounds.x >= 0 && routeBounds.x + routeBounds.width <= viewport.width
    ),
    horizontal_document_overflow: layout.documentScrollWidth > layout.documentClientWidth,
    canvas_count: layout.canvasCount,
    screenshot: path.relative(ROOT, screenshot),
    console_errors: consoleErrors,
    failed_responses: failedResponses,
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
      await inspectViewport(browser, 'desktop_1440x900', { width: 1440, height: 900 }, true),
      await inspectViewport(browser, 'mobile_390x844', { width: 390, height: 844 }, false),
    ];
    const optionalBackendWarning = (message) => (
      message.includes('ERR_CONNECTION_REFUSED') || message.includes('Failed to load resource')
    );
    const result = {
      phase_id: 'Phase416-PrefillAtlasClientVisualCheck',
      url: URL,
      checks,
      pass: checks.every((check) => (
        check.source_present
        && check.observed_dataset_count === 3
        && check.route_selector_inside_viewport
        && !check.horizontal_document_overflow
        && check.canvas_count > 0
        && check.console_errors.every(optionalBackendWarning)
        && check.failed_responses.length === 0
      )),
      evidence_boundary: 'Rendering validation only; a visible edge is still observational and non-causal.',
    };
    fs.writeFileSync(
      path.join(OUT, 'phase416_prefill_atlas_client_visual_check.json'),
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
