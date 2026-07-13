#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'tests/gpt5/result/phase407_event_horizon_kernel/screenshots');
const URL = 'http://127.0.0.1:5173/';

async function inspectViewport(browser, name, viewport) {
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
  const canvasPath = path.join(OUT, `phase407_atlas_${name}_canvas.png`);
  await canvas.screenshot({ path: canvasPath });

  await page.getByRole('button', { name: 'Global Strategy' }).click();
  await page.getByRole('button', { name: '语言分析', exact: true }).click();
  const latestValue = page.getByText('3,935/5,760 semantic / 0/3 cross-model', { exact: true });
  await latestValue.waitFor({ state: 'visible' });
  await latestValue.scrollIntoViewIfNeeded();
  await page.getByText('总体最新判断', { exact: true }).click();

  const requiredTexts = [
    'Phase407-EventHorizonConditionResponseStage：事件地平线条件响应核与接口迁移审计',
    'Phase407 正式发现案例：5760/5760。',
    '注册语义正确案例：3935/5760。',
    '完整响应案例：3933/5760。',
    '表面/接口/历史/序列四门完整组：10/108。',
    '跨模型条件状态族：0/3。',
    '新增物理路径：0/72。',
    '新增单神经元因果路径：0/72。',
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
  const screenshot = path.join(OUT, `phase407_atlas_${name}.png`);
  await page.screenshot({ path: screenshot, fullPage: false });
  await page.close();
  return {
    name,
    viewport,
    screenshot: path.relative(ROOT, screenshot),
    canvasScreenshot: path.relative(ROOT, canvasPath),
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
  const browser = await chromium.launch({
    executablePath: '/snap/bin/chromium',
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  try {
    const checks = [
      await inspectViewport(browser, 'desktop_1440x900', { width: 1440, height: 900 }),
      await inspectViewport(browser, 'mobile_390x844', { width: 390, height: 844 }),
    ];
  const result = {
      phase_id: 'Phase407-ClientVisualCheck',
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
    const output = path.join(OUT, 'phase407_client_visual_check.json');
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
