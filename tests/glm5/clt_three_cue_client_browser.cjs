// Verify the compiled client and audited static data in the real workbench.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '../..');
const work = path.join(root, 'tests/glm5_temp/clt_three_cue_client_qa');
const dist = path.join(work, 'dist');
const desktopArchive = process.argv.includes('--desktop-archive');
const awaitingAuth = desktopArchive || process.argv.includes('--archive-auth');
const recheck = awaitingAuth || process.argv.includes('--center-recheck');
const followup = recheck || process.argv.includes('--first-record-followup');
const resultFolder = desktopArchive ? 'clt_first_record_desktop' :
  awaitingAuth ? 'clt_first_record_archive_auth' : recheck ? 'clt_first_record_center_recheck' : followup ? 'clt_first_record_path_v1' : 'clt_b26_competition_v1';
const out = path.join(__dirname, 'result', resultFolder);
const gate = spawnSync(path.join(root, '.venv/Scripts/python.exe'), ['-B', '-X', 'utf8', '-c',
  'import sys; from deploy.storage_policy import check_write; check_write(sys.argv[1], 256*2**20); check_write(sys.argv[2], 16*2**20)',
  work, out], { cwd: root, encoding: 'utf8' });
assert.equal(gate.status, 0, gate.stderr || gate.stdout);
process.env.TEMP = process.env.TMP = work;
const { chromium } = require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const evidence = require(path.join(root, 'frontend/src/components/v8/threeCueEvidence.json'));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.svg': 'image/svg+xml' };

async function exercise() {
  const origin = 'http://ai2050-client.test';
  const browser = await chromium.launchPersistentContext(path.join(work, 'browser'), {
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true, viewport: { width: 1440, height: 1000 },
    args: ['--disable-background-networking', '--disable-component-update', '--enable-unsafe-swiftshader'],
  });
  try {
    const [page] = browser.pages();
    const errors = [];
    const computeRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => {
      const request = route.request();
      const url = new URL(request.url());
      if (['fetch', 'xhr'].includes(request.resourceType()) && request.method() === 'POST') computeRequests.push(url.pathname);
      if (url.origin !== origin) return route.abort();
      const target = url.pathname === '/' ? '/index.html' : url.pathname;
      const file = path.resolve(dist, '.' + target);
      if (path.relative(dist, file).startsWith('..') || !fs.statSync(file, { throwIfNoEntry: false })?.isFile()) {
        return route.fulfill({ status: 404, body: '' });
      }
      return route.fulfill({ path: file, contentType: types[path.extname(file)] || 'application/octet-stream' });
    });
    await page.goto(origin + '/?wb=1&lens=spatial&view=graph&gview=structure');
    const features = page.getByRole('group', { name: '功能插件' });
    await features.getByRole('button', { name: /条件对照/ }).click();
    const panel = page.getByRole('region', { name: '三线索选源', exact: true });
    await panel.waitFor();
    assert.match(page.url(), /view=response/);
    const outputs = () => panel.locator('.cue-output dd code').allTextContents();
    const firstRow = () => panel.locator('tbody tr').first().locator('td code').textContent();
    assert.equal(await panel.getByLabel('证据批次').inputValue(), evidence.runs.at(-1).id);
    await panel.getByLabel('证据批次').selectOption('clt_b26_three_cues_v3');
    assert.deepEqual(await outputs(), ['lion', 'lion', 'lion']);
    await panel.getByRole('checkbox', { name: '交换印刷标签' }).check();
    assert.deepEqual(await outputs(), ['glass', 'glass', 'glass']);
    assert.equal(await firstRow(), 'silver');
    await panel.getByRole('checkbox', { name: '交换印刷标签' }).uncheck();
    await panel.getByRole('button', { name: '物理首行', exact: true }).click();
    assert.deepEqual(await outputs(), ['silver', 'silver', 'glass']);
    await panel.getByRole('checkbox', { name: '交换首尾记录' }).check();
    assert.equal(await firstRow(), 'fire');
    assert.deepEqual(await outputs(), ['fire', 'fire', 'glass']);
    await panel.getByRole('checkbox', { name: '交换印刷标签' }).check();
    assert.deepEqual((await outputs()).slice(0, 2), ['fire', 'fire']);
    await panel.getByRole('button', { name: '姓名', exact: true }).click();
    assert.deepEqual(await outputs(), ['wood', 'wood', 'wood']);
    const provenance = panel.getByText('实验范围、坐标与复核来源', { exact: true });
    await provenance.click();
    assert.match(await provenance.locator('..').innerText(), /confirm-000-09/);
    assert.match(await provenance.locator('..').innerText(), /1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce/);
    await provenance.click();
    const views = page.getByRole('group', { name: '条件对照类型' });
    await views.getByRole('button', { name: '冻结样本对', exact: true }).click();
    await page.getByLabel('条件对照样本').waitFor();
    await views.getByRole('button', { name: '三线索选源', exact: true }).click();
    const current = evidence.runs.at(-1);
    const family = current.families[1];
    await panel.getByLabel('冻结家族').selectOption(String(family.id));
    const expected = family.cases.find(item => item.rule === 1 && !item.labelSwap && !item.positionSwap);
    assert.deepEqual(await outputs(), [family.codes[expected.gold], expected.output.trim(), family.codes[expected.mass.indexOf(Math.max(...expected.mass))]]);
    assert.equal(await panel.getByRole('heading', { name: /^(可读出|可预测|因果作用)$/ }).count(), 3);
    assert.match(await panel.getByRole('heading', { name: '因果作用', exact: true }).locator('..').innerText(),
      current.causal ? /匹配干预/ : current.admission.length ? /对照校准未通过/ : /Q\/K 预测均未过门/);
    await panel.screenshot({ path: path.join(out, 'client-three-cues.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    const nameRule = panel.getByRole('button', { name: '姓名', exact: true });
    await nameRule.focus();
    await page.keyboard.press('Space');
    assert.equal(await nameRule.getAttribute('aria-pressed'), 'true');
    const columns = await panel.locator('.cue-evidence').evaluate(node => getComputedStyle(node).gridTemplateColumns.trim().split(/\s+/).length);
    assert.equal(columns, 1);
    const widths = await panel.getByRole('region', { name: '记录与注意力表' }).evaluate(node => [node.clientWidth, node.scrollWidth, getComputedStyle(node).overflowX]);
    assert.equal(widths[2], 'auto');
    assert.ok(widths[0] <= 390, widths.join(','));
    const checks = ['actual client navigation', 'two independent swaps', 'three rules and immutable records', 'actual outputs and H36 ranking',
      'per-sample hashes', 'workbench compare switching', 'latest locally verified evidence', 'mobile keyboard and layout', 'no compute requests'];
    if (followup) {
      const filter = panel.getByRole('checkbox', { name: '只看首行反例' });
      await filter.check();
      const cases = panel.getByRole('combobox', { name: '首行反例', exact: true });
      await cases.waitFor();
      assert.equal(await cases.locator('option').count(), 303);
      const [gold, model, head] = await outputs();
      assert.equal(model, gold);
      assert.notEqual(head, gold);
      const previous = await cases.inputValue();
      const last = await cases.locator('option').last().getAttribute('value');
      await cases.selectOption(last);
      assert.notEqual(last, previous);
      await panel.getByLabel('证据批次').selectOption('clt_b26_three_cues_v3');
      assert.equal(await filter.isChecked(), false);
      await filter.check();
      await cases.waitFor();
      assert.equal(await cases.locator('option').count(), 339);
      await panel.getByLabel('证据批次').selectOption(current.id);
      await filter.check();
      await filter.uncheck();
      assert.equal(await panel.getByLabel('冻结家族').isVisible(), true);
      const archiveStatus = require(path.join(root, 'frontend/src/components/v8/firstRecordPathStatus.json'));
      const expectedStatus = archiveStatus.preflightId ? '归档与运行状态' : '未取得中心预检回执';
      const experiment = panel.getByText(`首行并行路径实验 · ${expectedStatus}`, { exact: true });
      await experiment.click();
      if (archiveStatus.archiveState === 'server_verified') {
        const result = experiment.locator('..');
        const report = archiveStatus.conclusion;
        const statusText = await result.innerText();
        assert.ok(statusText.includes(`B${report.block}/H${report.head}`));
        assert.match(statusText, /中心已读回核验.+可读出.+可预测.+因果作用/s);
        if (report.prediction && !report.prediction.gate_passed) assert.match(statusText, /未通过预注册门槛.+未执行匹配干预/s);
        if (report.causal) assert.ok(statusText.includes(`${report.causal.causal_gate_passed ? '达到' : '未达到'}因果验收条件`));
        await result.getByText('核对口径与正式回执', { exact: true }).click();
        assert.equal(await result.getByText(archiveStatus.preflightId, { exact: true }).isVisible(), true);
        if (report.prediction) assert.equal(await result.getByRole('table', { name: '原生坐标预测较对照的误差下降' }).isVisible(), true);
      }
      if (desktopArchive) assert.equal(await panel.getByRole('button', { name: '查重、登记并运行首行实验' }).count(), 0);
      await page.setViewportSize({ width: 1440, height: 1000 });
      await filter.check();
      await panel.screenshot({ path: path.join(out, 'client-first-record-counterexample.png') });
      checks.push('stored first-record counterexamples', 'batch isolation and filter recovery', 'first-record archive status');
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(computeRequests, []);
    fs.writeFileSync(path.join(out, 'client-browser.json'), JSON.stringify({ passed: true, checks, evidenceBytes: fs.statSync(require.resolve(path.join(root, 'frontend/src/components/v8/threeCueEvidence.json'))).size }, null, 2));
    console.log('Browser checks passed:', checks.join(', '));
  } finally {
    await browser.close();
  }
}

exercise().catch(error => { console.error(error); process.exitCode = 1; });
