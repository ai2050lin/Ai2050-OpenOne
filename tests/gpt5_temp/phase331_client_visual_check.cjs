const fs = require('fs');
const path = require('path');
const { chromium } = require('/home/rankrank/.hermes/hermes-agent/node_modules/playwright');

const output = path.resolve('tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit/screenshots');
fs.mkdirSync(output, { recursive: true });

async function inspectViewport(browser, name, viewport) {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(3000);
  const controls = page.locator('.pattern-atlas-controls');
  await controls.waitFor({ state: 'visible', timeout: 30000 });
  const familySelect = controls.locator('select').first();
  const modelSelect = controls.locator('select').nth(1);
  const familyOptions = await familySelect.locator('option').count();
  await familySelect.selectOption('content_knowledge');
  await modelSelect.selectOption('qwen3-4b');
  await page.waitForTimeout(2000);
  await controls.getByRole('button', { name: '扩展审计' }).click();
  await page.waitForTimeout(1200);
  const result = await page.evaluate(() => {
    const controlsNode = document.querySelector('.pattern-atlas-controls');
    const canvases = Array.from(document.querySelectorAll('canvas'));
    const bounds = controlsNode?.getBoundingClientRect();
    const overflowing = Array.from(controlsNode?.querySelectorAll('button, span, select') || []).filter((node) => (
      node.scrollWidth > node.clientWidth + 2 || node.scrollHeight > node.clientHeight + 2
    )).map((node) => node.textContent?.trim()).filter(Boolean);
    const footerText = controlsNode?.querySelector('.pattern-atlas-controls__footer')?.textContent || '';
    return {
      controlsVisible: Boolean(controlsNode),
      controlsWithinViewport: Boolean(bounds && bounds.left >= 0 && bounds.right <= window.innerWidth && bounds.top >= 0),
      canvasCount: canvases.length,
      nonzeroCanvasCount: canvases.filter((canvas) => canvas.width > 0 && canvas.height > 0).length,
      overflowingLabels: overflowing,
      footerText,
      phase331MetricsVisible: footerText.includes('扩展审计机制') && footerText.includes('完整门槛通过'),
      refinedFocusActive: Array.from(controlsNode?.querySelectorAll('button') || []).some((node) => (
        node.textContent?.trim() === '扩展审计' && node.classList.contains('is-active')
      )),
    };
  });
  await page.screenshot({ path: path.join(output, `${name}.png`), fullPage: false });
  await page.close();
  return { name, viewport, familyOptions, errors, ...result };
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/snap/bin/chromium',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--use-gl=swiftshader'],
  });
  try {
    const results = [];
    results.push(await inspectViewport(browser, 'phase331_desktop_1440x900', { width: 1440, height: 900 }));
    results.push(await inspectViewport(browser, 'phase331_mobile_390x844', { width: 390, height: 844 }));
    const valid = results.every((row) => (
      row.familyOptions === 9 && row.controlsVisible && row.controlsWithinViewport &&
      row.canvasCount > 0 && row.nonzeroCanvasCount === row.canvasCount &&
      row.phase331MetricsVisible && row.refinedFocusActive && row.errors.length === 0
    ));
    const payload = { phase_id: 'Phase331', created_at: new Date().toISOString(), valid, results };
    fs.writeFileSync(path.join(output, 'visual_check.json'), JSON.stringify(payload, null, 2) + '\n');
    process.stdout.write(JSON.stringify(payload, null, 2) + '\n');
    if (!valid) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
