const fs = require('fs');
const path = require('path');
const { chromium } = require('/home/rankrank/.hermes/hermes-agent/node_modules/playwright');

const output = path.resolve('tests/gpt5/result/phase396_field_binding_physical/screenshots');
fs.mkdirSync(output, { recursive: true });

async function inspectViewport(browser, name, viewport) {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(3500);

  const controls = page.locator('.pattern-atlas-controls');
  await controls.waitFor({ state: 'visible', timeout: 30000 });
  const familySelect = controls.locator('select').first();
  const familyOptions = await familySelect.locator('option').count();
  await familySelect.selectOption('content_knowledge');
  const modelSelect = controls.locator('select[aria-label="模型"]');
  if (await modelSelect.count()) await modelSelect.first().selectOption('qwen3-4b');
  const evidenceSelect = controls.locator('select[aria-label="证据范围"]');
  await evidenceSelect.selectOption('binding_context');
  await page.waitForTimeout(1800);

  const result = await page.evaluate(async () => {
    const controlsNode = document.querySelector('.pattern-atlas-controls');
    const canvases = Array.from(document.querySelectorAll('canvas'));
    const bounds = controlsNode?.getBoundingClientRect();
    const overflowing = Array.from(controlsNode?.querySelectorAll('button, span, select') || []).filter((node) => (
      node.scrollWidth > node.clientWidth + 2 || node.scrollHeight > node.clientHeight + 2
    )).map((node) => node.textContent?.trim()).filter(Boolean);
    const latestText = controlsNode?.querySelector('.pattern-atlas-controls__latest')?.textContent || '';
    const stage = await fetch('/vis_data/pattern_family_atlas/v2/phase396_binding_separation_stage_summary.json', { cache: 'no-store' }).then((response) => response.json());
    const neuron = await fetch('/vis_data/pattern_family_neuron_atlas/v1/manifest.json', { cache: 'no-store' }).then((response) => response.json());
    const partition = await fetch('/vis_data/pattern_family_neuron_atlas/v1/partitions/content_knowledge/qwen3.json', { cache: 'no-store' }).then((response) => response.json());
    const anchors = (partition.nodes || []).filter((node) => node.phase396_tested);
    return {
      controlsVisible: Boolean(controlsNode),
      controlsWithinViewport: Boolean(bounds && bounds.left >= 0 && bounds.right <= window.innerWidth && bounds.top >= 0),
      canvasCount: canvases.length,
      nonzeroCanvasCount: canvases.filter((canvas) => canvas.width > 0 && canvas.height > 0).length,
      overflowingLabels: overflowing,
      latestHasContext: latestText.includes('P396') && latestText.includes('上下文搬运 46/72'),
      latestHasControls: latestText.includes('内容对照 71/72') && latestText.includes('物理复现 3/3'),
      latestHasBoundary: latestText.includes('跨任务共享 0/1') && latestText.includes('神经元路径 0'),
      stageMatches: stage?.results?.phase396_same_literal_answer_switches === 46
        && stage?.results?.phase395_crosssurface_shared_state_count === 0,
      neuronBoundaryMatches: neuron?.phase === 396
        && neuron?.phase396_audit?.new_aggregate_state_anchor_count === 6
        && neuron?.phase396_audit?.new_neuron_path_nodes_promoted === 0,
      aggregateAnchorCount: anchors.length,
      aggregateAnchorsAreNotNeurons: anchors.length === 2
        && anchors.every((node) => node.node_type === 'aggregate_token_state_anchor'
          && node.is_real_unit === false && node.single_neuron_claim === false),
    };
  });

  const pagePath = path.join(output, `${name}.png`);
  const canvasPath = path.join(output, `${name}_canvas.png`);
  await page.screenshot({ path: pagePath, fullPage: false });
  await page.locator('canvas').first().screenshot({ path: canvasPath });
  const canvasDataUrl = `data:image/png;base64,${fs.readFileSync(canvasPath).toString('base64')}`;
  const canvasPixelStats = await page.evaluate(async (dataUrl) => {
    const image = new Image();
    image.src = dataUrl;
    await image.decode();
    const sampleCanvas = document.createElement('canvas');
    sampleCanvas.width = image.naturalWidth;
    sampleCanvas.height = image.naturalHeight;
    const context = sampleCanvas.getContext('2d');
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height).data;
    const stride = Math.max(1, Math.floor((sampleCanvas.width * sampleCanvas.height) / 20000));
    const buckets = new Set();
    let minimumLuma = 255;
    let maximumLuma = 0;
    let sampledPixels = 0;
    for (let index = 0; index < pixels.length; index += 4 * stride) {
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      const luma = (red * 0.2126) + (green * 0.7152) + (blue * 0.0722);
      minimumLuma = Math.min(minimumLuma, luma);
      maximumLuma = Math.max(maximumLuma, luma);
      buckets.add(`${red >> 4}:${green >> 4}:${blue >> 4}`);
      sampledPixels += 1;
    }
    return {
      sampledPixels,
      quantizedColorCount: buckets.size,
      lumaRange: maximumLuma - minimumLuma,
      nonblank: buckets.size > 16 && maximumLuma - minimumLuma > 24,
    };
  }, canvasDataUrl);
  await page.close();
  return { name, viewport, familyOptions, errors, pagePath, canvasPath, canvasPixelStats, ...result };
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/snap/bin/chromium',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--use-gl=swiftshader'],
  });
  try {
    const results = [];
    results.push(await inspectViewport(browser, 'phase396_desktop_1440x900', { width: 1440, height: 900 }));
    results.push(await inspectViewport(browser, 'phase396_mobile_390x844', { width: 390, height: 844 }));
    const valid = results.every((row) => (
      row.familyOptions === 9 && row.controlsVisible && row.controlsWithinViewport
      && row.canvasCount > 0 && row.nonzeroCanvasCount === row.canvasCount
      && row.latestHasContext && row.latestHasControls && row.latestHasBoundary
      && row.stageMatches && row.neuronBoundaryMatches
      && row.aggregateAnchorCount === 2 && row.aggregateAnchorsAreNotNeurons
      && row.errors.length === 0 && row.overflowingLabels.length === 0
      && row.canvasPixelStats.nonblank
    ));
    const payload = { phase_id: 'Phase396', created_at: new Date().toISOString(), valid, results };
    fs.writeFileSync(path.join(output, 'visual_check.json'), `${JSON.stringify(payload, null, 2)}\n`);
    process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
    if (!valid) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
