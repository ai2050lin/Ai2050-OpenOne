'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {checkedDir} = require('../../frontend/electron/storagePolicy');
const {chromium,_electron} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');

const out = checkedDir(path.join(__dirname,'result/qwen14_full_mlp_3d'), 40*2**20, '普查 3D 验收');
const scratch = checkedDir(path.join(__dirname,'../glm5_temp/qwen14_full_mlp_3d'), 160*2**20, '浏览器缓存');
const source = 'http://127.0.0.1:5098';
const snapshot = '1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce';
const report = {passed:false, checks:[], errors:[], blockedExternal:[]};
const executable = process.env.AI2050_TEST_ELECTRON;

async function run() {
  const profile = fs.mkdtempSync(path.join(scratch,'chromium-'));
  process.env.TEMP = process.env.TMP = profile;
  let browser, electron;
  try {
    let page;
    if(executable){
      electron=await _electron.launch({executablePath:executable,args:['--user-data-dir='+path.join(profile,'profile')],
        env:{...process.env,AI2050_CLIENT_SERVER:source,AI2050_NODE_HOME:path.join(profile,'node'),AI2050_STORAGE_ROOT:profile}});
      const app=await electron.evaluate(({app})=>({packaged:app.isPackaged,version:app.getVersion(),archive:app.getAppPath()}));
      assert.equal(app.packaged,true);
      assert.match(app.archive,/app\.asar$/);
      report.desktop=app;
      page=await electron.firstWindow();
      await page.waitForURL(/^file:/);
    }else {
      browser = await chromium.launch({executablePath:process.env.CHROME_PATH||path.join(os.homedir(),
        'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe'),
        headless:true});
      page = await browser.newPage({viewport:{width:1440,height:980},locale:'zh-CN'});
    }
    const entry=executable?page.url().split('?')[0]:source+'/workbench';
    const openUnit=(block,unit)=>page.goto(entry+'?wb=1&lens=spatial&view=graph&observeMode=features&observeTab=object&snapshot='+snapshot+
      '&sblock='+(block+1)+'&smodule=mlp&sindex='+unit);
    await page.route('**/*',route=>{
      const url = new URL(route.request().url());
      if(url.origin===source||!url.protocol.startsWith('http'))return route.continue();
      report.blockedExternal.push(url.origin);return route.abort();
    });
    page.on('pageerror',error=>report.errors.push(error.message));
    page.setDefaultTimeout(25000);
    await openUnit(23,7708);
    const controls = page.getByRole('group',{name:'MLP 神经元普查'});
    const reportPanel = page.getByRole('region',{name:'Goodfire 神经元分析'});
    const heading = name=>reportPanel.getByRole('heading',{name:new RegExp(name)});
    const raw = controls.getByLabel('普查神经元原值');
    const blocks = controls.getByLabel('普查模型块');
    const metric = controls.getByLabel('普查着色量');
    const scene = page.locator('.aw-canvas canvas').first();
    const waitText = (locator,text)=>locator.getByText(text,{exact:true}).waitFor();
    await heading('B23/U7708').waitFor();
    await waitText(controls,'17,408 个单元');
    const options = await blocks.locator('option').evaluateAll(items=>items.map(item=>item.value));
    assert.deepEqual(options,Array.from({length:22},(_,i)=>String(i+18)));
    assert.ok(await scene.isVisible());
    const standard = await raw.innerText();
    const image = await scene.evaluate(node=>node.toDataURL());
    assert.equal(Number(standard),0.13476703412881927);
    await metric.selectOption('mean');
    await waitText(controls,'-0.010448965011164546');
    await page.waitForFunction(initial=>document.querySelector('.aw-canvas canvas').toDataURL()!==initial,image);
    report.checks.push('B18–B39 目录覆盖全部 22 块，3D 图面与 B23 均值、标准差切换同步');
    await reportPanel.getByRole('button',{name:'在 3D 显示此位置'}).first().click();
    await waitText(controls,'0.58984375');
    assert.match(await controls.innerText(),/wikipedia-zh:18501.*945/);
    const suffix=executable?'-desktop':'';
    await page.getByRole('region',{name:'模型运行 3D 场景'}).screenshot({path:path.join(out,'b23-field'+suffix+'.png')});
    await reportPanel.screenshot({path:path.join(out,'b23-report'+suffix+'.png')});
    report.checks.push('从报告打开同一文本位置，3D 指向原始 BF16 激活');
    await blocks.selectOption('28');
    await heading('B28/U7708').waitFor();
    const versions = controls.getByLabel('普查归档运行');
    assert.equal(await versions.locator('option').count(),2);
    await versions.selectOption('QWEN14-GOODFIRE-B28-B39-20261113-v2');
    await reportPanel.getByText(/原始采集/).locator('..').evaluate(details=>{details.open=true;});
    await reportPanel.getByText('QWEN14-MLP-B28-B35-20261109-v1',{exact:true}).waitFor();
    await page.getByText('完整参数、跨层关联与验证记录',{exact:true}).click();
    const advanced = page.getByRole('region',{name:'Qwen3-14B全量MLP研究'});
    await advanced.getByRole('tab',{name:'完整参数'}).click();
    await advanced.getByText(/均保留全部 5,120 个参数/).waitFor();
    assert.equal(await advanced.getByLabel('14B研究层').count(),0);
    await blocks.selectOption('39');
    await heading('B39/U7708').waitFor();
    const address = page.getByLabel('输入原生编号');
    await address.fill('B39/U17407');
    await page.getByRole('button',{name:'定位对象',exact:true}).click();
    await heading('B39/U17407').waitFor();
    report.checks.push('B28 两份版本显式切换，B39 末位神经元可定位');
    await page.getByRole('button',{name:'原生单元',exact:true}).click();
    const full = page.getByRole('region',{name:'全屏原生单元'});
    await full.getByRole('region',{name:'原生全场读数'}).waitFor();
    await heading('B39/U17407').waitFor();
    await reportPanel.getByText(/留出集命中/).waitFor();
    await page.screenshot({path:path.join(out,'b39-fullscreen'+suffix+'.png')});
    await page.getByRole('button',{name:'3D 与控制台',exact:true}).click();
    await heading('B39/U17407').waitFor();
    await page.getByRole('button',{name:'退出全屏（Esc）',exact:true}).click();
    await waitText(controls,'17,408 个单元');
    report.checks.push('全屏原生网格与报告共用 B39/U17407，返回同一 3D 对象');
    await openUnit(17,0);
    await controls.getByText('当前模型块没有可读取的 MLP 普查。',{exact:true}).waitFor();
    assert.equal(await controls.getByLabel('普查神经元原值').count(),0);
    assert.deepEqual(report.errors,[]);
    report.checks.push('B17 显示无档案且没有借用其他层的数值');
    report.passed=true;
  } finally {
    await electron?.close();
    await browser?.close();
    fs.writeFileSync(path.join(out,executable?'desktop-check.json':'browser-check.json'),JSON.stringify(report,null,2)+'\n');
    const target=fs.realpathSync.native(profile);
    assert.equal(path.dirname(target),fs.realpathSync.native(scratch));
    fs.rmSync(target,{recursive:true,force:true});
  }
}
run().then(()=>console.log(report.checks.join('\n'))).catch(error=>{console.error(error);process.exitCode=1;});
