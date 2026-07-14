#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'tests/gpt5/result/phase412_typed_quotient_preflight/screenshots');
const CLIENT_PROGRESS = path.join(
  ROOT,
  'frontend/public/vis_data/pattern_family_atlas/v2/progress.json',
);
const URL = 'http://127.0.0.1:5175/';

function rowText(stage, key, label) {
  const row = stage[key];
  if (!row) throw new Error(`Missing Phase412 progress row: ${key}`);
  return `${label}：${row.numerator}/${row.denominator}。`;
}

async function inspectViewport(browser, name, viewport, stage) {
  const page = await browser.newPage({ viewport });
  const consoleErrors = [];
  const failedResponses = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('response', (response) => {
    if (response.status() >= 400) failedResponses.push({ status: response.status(), url: response.url() });
  });

  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(4000);

  const canvas = page.locator('canvas').first();
  await canvas.waitFor({ state: 'visible' });
  const canvasPath = path.join(OUT, `phase412_atlas_${name}_canvas.png`);
  await canvas.screenshot({ path: canvasPath });
  await canvas.dispatchEvent('wheel', { deltaY: 420, bubbles: true, cancelable: true });
  await page.waitForTimeout(900);
  const interactionCanvasPath = path.join(OUT, `phase412_atlas_${name}_canvas_interaction.png`);
  await canvas.screenshot({ path: interactionCanvasPath });

  await page.getByRole('button', { name: 'Global Strategy' }).click();
  await page.getByRole('button', { name: '语言分析', exact: true }).click();
  const latestValue = page.getByText('223 partitions / 0 model', { exact: true });
  await latestValue.waitFor({ state: 'visible' });
  await latestValue.scrollIntoViewIfNeeded();
  await page.getByText('总体最新判断', { exact: true }).click();

  const requiredTexts = [
    'Phase412-TypedObserverQuotientPreflightStage：类型化观察者协变与非平凡状态商穷举',
    rowText(stage, 'typed_observer_covariance_cells', 'Phase412 类型化观察者协变'),
    rowText(stage, 'fixed_failures_explained_by_role_transport', '固定角色失败的角色运输解释'),
    rowText(stage, 'observer_action_compositions', '观察者作用组合审计'),
    rowText(stage, 'finite_partitions_exhausted', '有限状态分区穷举'),
    rowText(stage, 'global_nontrivial_qualifying_partitions', '全局合格非平凡商'),
    rowText(stage, 'external_role_indexed_partition_bundles', '外部角色索引分区束'),
    rowText(stage, 'registered_irreversible_operations', '已注册不可逆操作'),
    rowText(stage, 'registered_cross_family_bridges', '已注册跨族桥'),
    rowText(stage, 'independent_external_reviewers', '独立外部规则审阅者'),
    rowText(stage, 'sealed_model_collector_equivalence', '密封模型采集器等价'),
    rowText(stage, 'model_cases_consumed', '已使用模型案例'),
    rowText(stage, 'physical_cases_consumed', '已使用物理案例'),
    rowText(stage, 'new_physical_paths', '新增物理路径'),
    rowText(stage, 'new_single_neuron_causal_paths', '新增单神经元因果路径'),
  ];
  const assertions = {};
  for (const text of requiredTexts) {
    assertions[text] = await page.getByText(text, { exact: true }).isVisible();
  }
  const layout = await page.evaluate(() => ({
    documentClientWidth: document.documentElement.clientWidth,
    documentScrollWidth: document.documentElement.scrollWidth,
    bodyClientWidth: document.body.clientWidth,
    bodyScrollWidth: document.body.scrollWidth,
    canvasCount: document.querySelectorAll('canvas').length,
  }));
  const screenshot = path.join(OUT, `phase412_atlas_${name}.png`);
  await page.screenshot({ path: screenshot, fullPage: false });
  await page.close();
  return {
    name,
    viewport,
    screenshot: path.relative(ROOT, screenshot),
    canvasScreenshot: path.relative(ROOT, canvasPath),
    interactionCanvasScreenshot: path.relative(ROOT, interactionCanvasPath),
    assertions,
    allAssertionsPass: Object.values(assertions).every(Boolean),
    layout,
    horizontalDocumentOverflow: layout.documentScrollWidth > layout.documentClientWidth,
    consoleErrors,
    failedResponses,
  };
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const progress = JSON.parse(fs.readFileSync(CLIENT_PROGRESS, 'utf8'));
  const stage = progress.typed_observer_quotient_preflight_stage;
  if (!stage) throw new Error('Phase412 client progress has not been synchronized');
  const browser = await chromium.launch({
    executablePath: '/snap/bin/chromium',
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  try {
    const checks = [
      await inspectViewport(browser, 'desktop_1440x900', { width: 1440, height: 900 }, stage),
      await inspectViewport(browser, 'mobile_390x844', { width: 390, height: 844 }, stage),
    ];
    const result = {
      phase_id: 'Phase412-ClientVisualCheck',
      url: URL,
      checks,
      pass: checks.every((item) => (
        item.allAssertionsPass
        && !item.horizontalDocumentOverflow
        && item.layout.canvasCount > 0
        && item.consoleErrors.every((message) => message === 'Failed to load resource: net::ERR_CONNECTION_REFUSED')
        && item.failedResponses.length === 0
      )),
      knownOptionalBackendWarning: 'The client probes http://localhost:5001/api/ai-rnd/session/status when the optional research server is not running.',
    };
    const output = path.join(OUT, 'phase412_client_visual_check.json');
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
