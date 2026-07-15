#!/usr/bin/env node

const { chromium } = require('playwright-core');

const URL = process.env.PHASE_VIS_URL || 'http://127.0.0.1:5176/';
const SOURCE_ID = process.env.PHASE_SOURCE_ID || 'gpt5_phase429_typed_route';

(async () => {
  const browser = await chromium.launch({
    executablePath: '/snap/bin/chromium',
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const consoleRows = [];
  page.on('console', (message) => consoleRows.push(`${message.type()}: ${message.text()}`));
  page.on('pageerror', (error) => consoleRows.push(`pageerror: ${error.stack || error.message}`));
  await page.addInitScript(() => localStorage.setItem('fpMode', 'demo'));
  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(3000);
  const source = page.getByLabel('主工作台测试路线数据源');
  const dataset = page.getByLabel('主工作台测试数据集');
  await source.selectOption(SOURCE_ID);
  await page.waitForTimeout(2500);
  const options = await dataset.locator('option').evaluateAll((rows) => rows.map((row) => ({ value: row.value, text: row.textContent })));
  const datasetId = options.find((row) => row.value && !row.value.startsWith('__'))?.value;
  if (datasetId) await dataset.selectOption(datasetId);
  await page.waitForTimeout(5000);
  const state = await page.evaluate(() => ({
    body_text: document.body.innerText.slice(0, 12000),
    canvas_count: document.querySelectorAll('canvas').length,
    source_value: document.querySelector('[aria-label="主工作台测试路线数据源"]')?.value,
    dataset_value: document.querySelector('[aria-label="主工作台测试数据集"]')?.value,
  }));
  process.stdout.write(`${JSON.stringify({ options, state, consoleRows }, null, 2)}\n`);
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
