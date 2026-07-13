const fs = require('fs');
const path = require('path');
const { chromium } = require('/home/rankrank/.hermes/hermes-agent/node_modules/playwright');

const output = path.resolve('tests/gpt5/result/phase399_dynamic_binding/screenshots');
fs.mkdirSync(output, { recursive: true });

async function inspectViewport(browser, name, viewport) {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(3500);
  const controls = page.locator('.pattern-atlas-controls');
  await controls.waitFor({ state: 'visible', timeout: 30000 });
  await controls.locator('select').first().selectOption('language_action');
  const modelSelect = controls.locator('select[aria-label="模型"]');
  if (await modelSelect.count()) await modelSelect.first().selectOption('ds7b');
  await controls.locator('select[aria-label="证据范围"]').selectOption('binding_context');
  await page.waitForTimeout(1800);

  const result = await page.evaluate(async () => {
    const controlsNode = document.querySelector('.pattern-atlas-controls');
    const canvases = Array.from(document.querySelectorAll('canvas'));
    const bounds = controlsNode?.getBoundingClientRect();
    const overflowing = Array.from(controlsNode?.querySelectorAll('button, span, select') || []).filter((node) => (
      node.scrollWidth > node.clientWidth + 2 || node.scrollHeight > node.clientHeight + 2
    )).map((node) => node.textContent?.trim()).filter(Boolean);
    const latestText = controlsNode?.querySelector('.pattern-atlas-controls__latest')?.textContent || '';
    const stage = await fetch('/vis_data/pattern_family_atlas/v2/phase399_dynamic_binding_stage_summary.json', { cache: 'no-store' }).then((response) => response.json());
    const neuron = await fetch('/vis_data/pattern_family_neuron_atlas/v1/manifest.json', { cache: 'no-store' }).then((response) => response.json());
    const partition = await fetch('/vis_data/pattern_family_neuron_atlas/v1/partitions/language_action/deepseek7b.json', { cache: 'no-store' }).then((response) => response.json());
    const anchors = (partition.nodes || []).filter((node) => node.phase399_tested);
    return {
      controlsVisible: Boolean(controlsNode),
      controlsWithinViewport: Boolean(bounds && bounds.left >= 0 && bounds.right <= window.innerWidth && bounds.top >= 0),
      canvasCount: canvases.length,
      nonzeroCanvasCount: canvases.filter((canvas) => canvas.width > 0 && canvas.height > 0).length,
      overflowingLabels: overflowing,
      latestMatches: latestText.includes('P399')
        && latestText.includes('完整组 82/112')
        && latestText.includes('任务面 3/4')
        && latestText.includes('三分割必需事件 27/27')
        && latestText.includes('三分割有序链 3/27')
        && latestText.includes('跨模型链 0/3')
        && latestText.includes('因果未授权'),
      stageMatches: stage?.results?.required_event_physical_cell_count === 9
        && stage?.results?.ordered_chain_physical_cell_count === 1
        && stage?.results?.ordered_chain_crossmodel_surface_count === 0
        && stage?.results?.model_specific_chain_model === 'deepseek7b'
        && stage?.results?.model_specific_chain_surface === 'role_filling'
        && JSON.stringify(stage?.results?.model_specific_chain_layers) === JSON.stringify([10, 10, 20])
        && stage?.authorization?.run_joint_causal_intervention === false,
      neuronBoundaryMatches: neuron?.phase === 399
        && neuron?.phase399_audit?.new_aggregate_dynamic_event_count === 3
        && neuron?.phase399_audit?.new_neuron_path_nodes_promoted === 0,
      aggregateAnchorCount: anchors.length,
      aggregateAnchorsAreNotNeurons: anchors.length === 3
        && anchors.every((node) => node.node_type === 'aggregate_dynamic_route_event'
          && node.is_real_unit === false && node.single_neuron_claim === false
          && node.phase399_crossmodel_chain_pass === false
          && node.phase399_causal_gate_open === false),
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
    for (let index = 0; index < pixels.length; index += 4 * stride) {
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      const luma = (red * 0.2126) + (green * 0.7152) + (blue * 0.0722);
      minimumLuma = Math.min(minimumLuma, luma);
      maximumLuma = Math.max(maximumLuma, luma);
      buckets.add(`${red >> 4}:${green >> 4}:${blue >> 4}`);
    }
    return { quantizedColorCount: buckets.size, lumaRange: maximumLuma - minimumLuma, nonblank: buckets.size > 16 && maximumLuma - minimumLuma > 24 };
  }, canvasDataUrl);
  await page.close();
  return { name, viewport, errors, pagePath, canvasPath, canvasPixelStats, ...result };
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/snap/bin/chromium',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--use-gl=swiftshader'],
  });
  try {
    const results = [
      await inspectViewport(browser, 'phase399_desktop_1440x900', { width: 1440, height: 900 }),
      await inspectViewport(browser, 'phase399_mobile_390x844', { width: 390, height: 844 }),
    ];
    const valid = results.every((row) => (
      row.controlsVisible && row.controlsWithinViewport
      && row.canvasCount > 0 && row.nonzeroCanvasCount === row.canvasCount
      && row.latestMatches && row.stageMatches && row.neuronBoundaryMatches
      && row.aggregateAnchorCount === 3 && row.aggregateAnchorsAreNotNeurons
      && row.errors.length === 0 && row.overflowingLabels.length === 0
      && row.canvasPixelStats.nonblank
    ));
    const payload = { phase_id: 'Phase399', created_at: new Date().toISOString(), valid, results };
    fs.writeFileSync(path.join(output, 'visual_check.json'), `${JSON.stringify(payload, null, 2)}\n`);
    process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
    if (!valid) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
