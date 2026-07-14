#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'tests/gpt5/result/phase415_multi_route_vis_sources/screenshots');
const URL = 'http://127.0.0.1:5175/';
const SOURCES = [
  { id: 'gpt5_pattern_family_neuron_atlas', expected: 27 },
  { id: 'gpt5_real_component_trace', expected: 3 },
  { id: 'gpt5_mechanism_trace', expected: 3 },
  { id: 'glm5_causal_fiber_atlas', expected: 67 },
];

async function datasetValues(select) {
  return select.locator('option').evaluateAll((options) => options
    .map((option) => option.value)
    .filter((value) => value && !value.startsWith('__')));
}

async function inspectViewport(browser, name, viewport, loadEverySource) {
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
  await page.waitForTimeout(4500);

  const routeSelect = page.getByLabel('主工作台测试路线数据源');
  const datasetSelect = page.getByLabel('主工作台测试数据集');
  await routeSelect.waitFor({ state: 'attached' });
  await routeSelect.scrollIntoViewIfNeeded();
  await routeSelect.waitFor({ state: 'visible' });

  const sourceOptions = await routeSelect.locator('option').evaluateAll((options) => options.map((option) => ({
    value: option.value,
    text: option.textContent,
  })));
  const sourceChecks = [];
  const sourcesToLoad = loadEverySource ? SOURCES : SOURCES.slice(0, 1);
  for (const source of sourcesToLoad) {
    await routeSelect.selectOption(source.id);
    await page.waitForTimeout(900);
    await page.waitForFunction(
      ({ id, expected }) => {
        const route = document.querySelector('[aria-label="主工作台测试路线数据源"]');
        const dataset = document.querySelector('[aria-label="主工作台测试数据集"]');
        const values = [...(dataset?.options || [])]
          .map((option) => option.value)
          .filter((value) => value && !value.startsWith('__'));
        return route?.value === id && values.length === expected;
      },
      source,
      { timeout: 12000 }
    );
    const values = await datasetValues(datasetSelect);
    await datasetSelect.selectOption(values[0]);
    await page.getByText('✓ 已加载', { exact: true }).waitFor({ state: 'visible', timeout: 12000 });
    await page.waitForTimeout(1200);
    const loadedSummary = await page.getByText('✓ 已加载', { exact: true }).locator('..').innerText();
    const canvas = page.locator('canvas').first();
    await canvas.waitFor({ state: 'visible' });
    const canvasPath = path.join(OUT, `phase415_${name}_${source.id}_canvas.png`);
    await canvas.screenshot({ path: canvasPath });
    await canvas.dispatchEvent('wheel', { deltaY: 360, bubbles: true, cancelable: true });
    await page.waitForTimeout(650);
    const interactionCanvasPath = path.join(OUT, `phase415_${name}_${source.id}_canvas_interaction.png`);
    await canvas.screenshot({ path: interactionCanvasPath });
    await canvas.dispatchEvent('wheel', { deltaY: -360, bubbles: true, cancelable: true });
    await page.waitForTimeout(350);
    sourceChecks.push({
      source_id: source.id,
      expected_dataset_count: source.expected,
      observed_dataset_count: values.length,
      loaded_dataset_id: values[0],
      loaded_summary: loadedSummary,
      canvas_screenshot: path.relative(ROOT, canvasPath),
      interaction_canvas_screenshot: path.relative(ROOT, interactionCanvasPath),
    });
  }

  const routeBounds = await routeSelect.boundingBox();
  const layout = await page.evaluate(() => ({
    documentClientWidth: document.documentElement.clientWidth,
    documentScrollWidth: document.documentElement.scrollWidth,
    bodyClientWidth: document.body.clientWidth,
    bodyScrollWidth: document.body.scrollWidth,
    canvasCount: document.querySelectorAll('canvas').length,
  }));
  const screenshot = path.join(OUT, `phase415_multi_route_${name}.png`);
  await page.screenshot({ path: screenshot, fullPage: false });
  await page.close();
  return {
    name,
    viewport,
    source_options: sourceOptions,
    source_checks: sourceChecks,
    route_selector_bounds: routeBounds,
    route_selector_inside_viewport: Boolean(
      routeBounds
      && routeBounds.x >= 0
      && routeBounds.x + routeBounds.width <= viewport.width
    ),
    screenshot: path.relative(ROOT, screenshot),
    layout,
    horizontal_document_overflow: layout.documentScrollWidth > layout.documentClientWidth,
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
    const allowedConsoleError = (message) => (
      message.includes('ERR_CONNECTION_REFUSED')
      || message.includes('Failed to load resource')
    );
    const result = {
      phase_id: 'Phase415-MultiRouteClientVisualCheck',
      url: URL,
      checks,
      pass: checks.every((check) => (
        check.source_options.length === 4
        && check.source_checks.every((source) => source.expected_dataset_count === source.observed_dataset_count)
        && check.route_selector_inside_viewport
        && !check.horizontal_document_overflow
        && check.layout.canvasCount > 0
        && check.console_errors.every(allowedConsoleError)
        && check.failed_responses.length === 0
      )),
      known_optional_backend_warning: 'The optional localhost:5001 research server may be absent during the static client check.',
    };
    const output = path.join(OUT, 'phase415_multi_route_client_visual_check.json');
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
