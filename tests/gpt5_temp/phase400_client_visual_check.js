#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'tests/gpt5/result/phase400_partial_order/screenshots');
const URL = 'http://127.0.0.1:5173/';

async function inspectViewport(browser, name, viewport) {
  const page = await browser.newPage({ viewport });
  const consoleErrors = [];
  const failedResponses = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('response', (response) => {
    if (response.status() >= 400) {
      failedResponses.push({ status: response.status(), url: response.url() });
    }
  });

  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Global Strategy' }).waitFor({ state: 'visible' });
  await page.getByRole('button', { name: 'Global Strategy' }).click();
  await page.getByRole('button', { name: '语言分析', exact: true }).click();

  const latestValue = page.getByText('5/6 discovery graphs / 0/6 prediction cells', { exact: true });
  await latestValue.waitFor({ state: 'visible' });
  await latestValue.scrollIntoViewIfNeeded();
  await page.getByText('总体最新判断', { exact: true }).click();

  const requiredTexts = [
    'Phase400-DynamicPartialOrderStage：区间部分序候选与严格门控结果',
    '发现集预测门合格单元：0/6。',
    '校准集守恒合格组模型单元：23/24。',
    '已使用物理留出案例：0/384。',
    '单神经元因果路径：0/72。',
  ];
  const assertions = {};
  for (const text of requiredTexts) {
    const locator = page.getByText(text, { exact: true });
    assertions[text] = await locator.isVisible();
  }

  const layout = await page.evaluate(() => ({
    documentClientWidth: document.documentElement.clientWidth,
    documentScrollWidth: document.documentElement.scrollWidth,
    bodyClientWidth: document.body.clientWidth,
    bodyScrollWidth: document.body.scrollWidth,
  }));
  const screenshot = path.join(OUT, `phase400_atlas_${name}.png`);
  await page.screenshot({ path: screenshot, fullPage: false });
  await page.close();
  return {
    name,
    viewport,
    screenshot: path.relative(ROOT, screenshot),
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
    const checks = [];
    checks.push(await inspectViewport(browser, 'desktop_1440x900', { width: 1440, height: 900 }));
    checks.push(await inspectViewport(browser, 'mobile_390x844', { width: 390, height: 844 }));
    const result = {
      phase_id: 'Phase400-ClientVisualCheck',
      url: URL,
      checks,
      pass: checks.every((item) => item.allAssertionsPass && !item.horizontalDocumentOverflow),
    };
    const output = path.join(OUT, 'phase400_client_visual_check.json');
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
