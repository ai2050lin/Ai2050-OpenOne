const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');

const output = path.resolve('tests/glm5/result/task_mode_real');
const temp = path.resolve('tests/glm5_temp/task_mode_real');
process.env.TEMP = process.env.TMP = temp;
const { chromium } = require(process.env.AI2050_PLAYWRIGHT || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const api = 'http://127.0.0.1:5001';
const frontend = 'http://127.0.0.1:5175';
const families = (process.env.AI2050_ACCEPTANCE_TASKS || 'binding').split(',');
const started = Date.now();
const report = { runMode: 'real local CUDA', checks: [], runs: [], browserErrors: [] };
let stage = '启动预览';
let sharedPeak = null;
const progress = () => {
  const gpu = spawnSync('nvidia-smi', ['--query-gpu=memory.used', '--format=csv,noheader'], { encoding: 'utf8', windowsHide: true });
  const elapsed = (Date.now()-started)/1000;
  const completed = report.runs.length;
  const shared = process.env.AI2050_GPU_SHARED_REPORT;
  if (shared && fs.existsSync(shared)) {
    const bytes = Number(fs.readFileSync(shared,'utf8').trim());
    if (Number.isFinite(bytes)) sharedPeak = Math.max(sharedPeak ?? 0, bytes);
  }
  console.log(JSON.stringify({ at: new Date().toISOString(), stage, done: completed, total: 2 * families.length, elapsed_s: elapsed, eta_s: completed ? Math.round(elapsed * (2 * families.length - completed) / completed) : null, gpu: gpu.stdout?.trim() || 'unavailable', shared_peak_bytes: sharedPeak }));
};
const step = value => { stage = value; progress(); };
const read = async route => {
  const res = await fetch(api + route);
  assert.equal(res.status, 200, route);
  return res.json();
};
const responseFor = (page, route, method = 'GET') => page.waitForResponse(res => {
  const url = new URL(res.url());
  return url.origin === api && url.pathname === route && res.request().method() === method;
}, { timeout: 180000 });

async function main() {
  const modelStatus = await read('/api/live/status');
  assert.equal(modelStatus.status, 'idle', 'Start with a dedicated idle live service so this test owns the model load');
  const models = (await read('/api/model/skeleton')).models;
  const model = models.find(row => row.model.startsWith('qwen3-4b|'));
  assert.ok(model?.model_snapshot);
  const preview = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--host', '127.0.0.1', '--port', '5175', '--strictPort', '--outDir', path.join(temp,'build')], { cwd: path.resolve('frontend'), stdio: 'ignore', windowsHide: true });
  const timer = setInterval(progress, 20000);
  let browser, page, loadRequested = false;
  try {
    progress();
    for (let i=0; i<30; i++) {
      assert.equal(preview.exitCode, null);
      if (await fetch(frontend).then(r=>r.ok).catch(()=>false)) break;
      await new Promise(resolve=>setTimeout(resolve,200));
    }
    browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
    page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.setDefaultTimeout(90000);
    const requests=[];
    page.on('pageerror', e=>report.browserErrors.push(e.message));
    page.on('request', req=>requests.push({path:new URL(req.url()).pathname,method:req.method()}));
    await page.addInitScript(base=>{window.__AI2050__={apiBase:base,mode:'local'};},api);
    await page.route('**/*',route=>[api,frontend].includes(new URL(route.request().url()).origin)?route.continue():route.abort());
    await page.goto(`${frontend}/workbench?lens=spatial&view=graph&gview=structure&snapshot=${model.model_snapshot}`);
    await page.getByRole('button',{name:'任务模式',exact:true}).click();
    const task=page.getByRole('region',{name:'自动文本任务'});
    const input=page.getByLabel('运行文本',{exact:true});
    const taskRows=task.locator('.aw-task-row');
    const canvas=page.locator('.aw-space canvas');
    await task.getByText(/请先在上方加载本机模型/).waitFor();
    assert.equal(await taskRows.first().getByRole('button',{name:'运行并分析'}).isDisabled(),true);
    const loading=responseFor(page,'/api/live/load','POST');
    step('从控制台加载 Qwen3-4B');
    await page.getByRole('button',{name:'加载',exact:true}).click();
    loadRequested = true;
    assert.equal((await loading).status(),200);
    await page.getByLabel('本机模型',{exact:true}).getByText(/已加载/).first().waitFor({timeout:120000});
    const ready=await read('/api/live/status');
    assert.equal(ready.device,'cuda');
    assert.equal(ready.meta.n_layers,36);
    assert.match(ready.model_name,/^qwen3-4b$/i);
    report.device=ready;
    report.checks.push('idle state blocks task, explicit model load reaches CUDA');
    for (const kind of families) {
      await task.getByLabel('观察任务类型').selectOption(kind);
      const pair=[];
      for (const i of [0,1]) {
        const condition=i?'B':'A';
        const row=taskRows.nth(i);
        await row.getByRole('button',{name:'查看文本'}).click();
        const text=await input.inputValue();
        assert.ok(text.length>300);
        const expected=(await row.innerText()).match(/目标颜色：([^\s])/)[1];
        step(`${kind} ${condition} 原生前向及轨迹读取`);
        const sent=responseFor(page,'/api/live/run','POST');
        await row.getByRole('button',{name:'运行并分析',exact:true}).click();
        const actual=await sent;
        if(!actual.ok()) throw Error(`/api/live/run ${actual.status()} ${await actual.text()}`);
        const result=await actual.json();
        report.runs.push({kind,condition,text,expected,trace:result.trace_id,snapshot:result.model_snapshot,tokens:result.n_tokens,top:result.output_top,elapsed_ms:result.elapsed_ms,readout:result.readout_positions,files:result.files,attention_max:Math.max(...Object.values(result.reconstruction).map(v=>v.attention_relative_l2))});
        assert.equal(actual.request().postDataJSON().text,text);
        assert.equal(result.text,text);
        assert.equal(result.kind,'real');
        assert.equal(result.model_snapshot,model.model_snapshot);
        assert.equal(result.truncated,false);
        assert.ok(result.n_tokens>100);
        assert.equal(result.readout_positions.at(-1),result.n_tokens-1);
        await row.getByRole('button',{name:'在3D中观察'}).waitFor();
        await page.locator('.aw-space .aw-big').getByText(/‖h‖/).waitFor();
        assert.ok((await row.innerText()).includes(JSON.stringify(result.output_top[0].text)));
        pair.push(result);
      }
      assert.notEqual(pair[0].text,pair[1].text);
      await task.getByText('2 / 2 已运行',{exact:true}).waitFor();
      const layers=task.locator('.aw-task-layers button');
      assert.equal(await layers.count(),37);
      await layers.nth(12).click();
      assert.match(await canvas.getAttribute('aria-label'),/当前第 12 层/);
      assert.equal(await page.evaluate(()=>document.activeElement?.getAttribute('aria-label')),'模型运行 3D 场景');
      await page.locator('.aw-space .aw-big').getByText(/任务条件 B.*‖h‖/).waitFor();
      assert.equal(await page.getByLabel('当前层',{exact:true}).inputValue(),'12');
      assert.equal(await page.getByLabel('当前Token',{exact:true}).inputValue(),String(pair[1].n_tokens-1));
      const manifest=await read(`/api/traces/${pair[1].trace_id}/manifest`);
      assert.equal(manifest.text,pair[1].text);
      const bundle=await read(`/api/traces/${pair[1].trace_id}/bundle?position=${pair[1].n_tokens-1}`);
      assert.equal(bundle.trace_id,pair[1].trace_id);
      assert.deepEqual(bundle.hs.shape,[37,2560]);
      assert.deepEqual(bundle.attn.shape,[36,32,pair[1].n_tokens]);
      assert.deepEqual(bundle.mlp.shape,[36,9728]);
      const [a,b]=await Promise.all(pair.map(r=>read(`/api/traces/${r.trace_id}/field?block=12&position=${r.n_tokens-1}&boundary=hidden_state`)));
      const norm=v=>Math.hypot(...v);
      const ratio=norm(b.values.map((v,i)=>v-a.values[i]))/norm(a.values);
      assert.match(await layers.nth(12).innerText(),new RegExp(ratio.toFixed(3).replace('.','\\.')));
      const layerRatioText = await layers.nth(12).getAttribute('title');
      report.checks.push(`${kind}: generated A/B, real predictions, 37 layer comparisons, native B bundle and L12 metric`);
      report[`${kind}_L12`]={relativeL2:ratio,label:layerRatioText};
      await canvas.scrollIntoViewIfNeeded();
      await page.screenshot({path:path.join(output,`${kind}-scene.png`)});
      await task.scrollIntoViewIfNeeded();
      await page.screenshot({path:path.join(output,`${kind}-analysis.png`)});
      await taskRows.first().getByRole('button',{name:'在3D中观察'}).click();
      const sceneA=await page.locator('.aw-space .aw-big').getByText(/任务条件 A.*‖h‖/).innerText();
      assert.ok(sceneA.endsWith(norm(a.values).toFixed(1)));
      assert.equal(await page.getByLabel('当前层',{exact:true}).inputValue(),'12');
      for (const r of pair) {
        const path=`/api/traces/${r.trace_id}`;
        assert.equal(requests.filter(req=>req.path===path+'/bundle').length,1,'Reuse the same final-position bundle when switching A / B');
        assert.equal(requests.filter(req=>req.path===path+'/field').length,0,'No per-layer network loop');
      }
      report.checks.push(`${kind}: focuses the scene, switches to A and reuses each verified bundle`);
    }
    assert.deepEqual(report.browserErrors,[]);
    assert.equal(requests.filter(req=>req.method==='POST'&&req.path==='/api/live/run').length,2*families.length);
    step('完成真实验收');
    report.passed=true;
  } catch(error) {
    report.error=error.stack;
    if(page)await page.screenshot({path:path.join(output,'failure.png')}).catch(()=>{});
    throw error;
  } finally {
    clearInterval(timer);
    if(loadRequested && (await read('/api/live/status').catch(()=>({}))).status==='ready') {
      const response=await fetch(api+'/api/live/unload',{method:'POST'}).catch(()=>null);
      report.unloaded=Boolean(response?.ok) && (await read('/api/live/status')).status==='idle';
    }
    if(browser)await browser.close();
    preview.kill();
    fs.writeFileSync(path.join(output,'real-browser.json'),JSON.stringify(report,null,2));
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
