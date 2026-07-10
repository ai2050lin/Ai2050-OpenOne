const { test, expect } = require('@playwright/test');

test('Phase326 physical atlas family and evidence filters', async ({ page }) => {
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForTimeout(4000);

  const panel = page.getByRole('region', { name: '模式族物理叠层控制' });
  await expect(panel).toBeVisible();
  await expect(panel).toContainText('336 个唯一候选');
  await expect(panel).toContainText('12 个扩大确认候选');

  const familySelect = panel.locator('select').first();
  await familySelect.selectOption('reasoning_constraint');
  await page.waitForTimeout(1200);
  await expect(panel).toContainText('48 个唯一候选');
  await expect(panel).toContainText('0 个扩大确认候选');

  const confirmedButton = panel.getByRole('button', { name: '扩大确认' });
  await confirmedButton.click();
  await page.waitForTimeout(500);
  await expect(confirmedButton).toHaveClass(/is-active/);
  await expect(panel).toContainText('48 个唯一候选');
  expect(errors).toEqual([]);
});

test('Phase326 physical atlas remains usable at 390px', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForTimeout(4000);
  const panel = page.getByRole('region', { name: '模式族物理叠层控制' });
  await expect(panel).toBeVisible();
  const box = await panel.boundingBox();
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(390);
  expect(box.y + box.height).toBeLessThanOrEqual(844);
  const pageWidth = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(pageWidth).toBeLessThanOrEqual(390);
});

test('base DNN shape remains when the language overlay is hidden', async ({ page }) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForTimeout(4000);
  const canvas = page.locator('canvas').first();
  await expect(canvas).toBeVisible();
  const withOverlay = await canvas.screenshot({ path: '/tmp/phase327-base-with-overlay.png' });
  await page.getByRole('button', { name: '隐藏模式族叠层' }).click();
  await page.waitForTimeout(1000);
  await expect(page.getByRole('button', { name: '打开模式族物理叠层' })).toBeVisible();
  await expect(canvas).toBeVisible();
  const withoutOverlay = await canvas.screenshot({ path: '/tmp/phase327-base-without-overlay.png' });
  expect(withOverlay.length).toBeGreaterThan(10000);
  expect(withoutOverlay.length).toBeGreaterThan(10000);
  expect(Buffer.compare(withOverlay, withoutOverlay)).not.toBe(0);
});
