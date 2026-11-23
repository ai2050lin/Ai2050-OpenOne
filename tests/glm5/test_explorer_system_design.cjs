'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {pathToFileURL} = require('node:url');
const {checkedDir, checkWrite} = require('../../frontend/electron/storagePolicy.js');

const root = path.resolve(__dirname,'../..');
const preview = path.join(root,'research/glm5/docs/previews/explorer-system-design.html');
const checkPlugins = process.argv.includes('--plugins');
const checkNative = process.argv.includes('--native-field');
const resultDir = path.join(__dirname,'result',checkNative?'explorer_system_native_field':checkPlugins?'explorer_system_plugins':'explorer_system_design');
const scratch = path.join(root,'tests/glm5_temp');
const nativeFile = path.join(root,'frontend/src/components/v8/data/graphObservationExample.json');

async function verify() {
  assert.ok(fs.existsSync(preview),'缺少静态稿入口 '+preview);
  checkWrite(resultDir,280 * 2 ** 20,'模型分析系统静态稿验收');
  checkedDir(resultDir,24 * 2 ** 20,'截图与验收结果');
  const parent = checkedDir(scratch,256 * 2 ** 20,'隔离浏览器目录');
  const profile = fs.mkdtempSync(path.join(parent,'explorer-system-qa-'));
  for(const key of ['TEMP','TMP','TMPDIR'])process.env[key]=profile;
  const report = {passed:false,checks:[],network:[],errors:[],screenshots:[]};
  let browser;
  try {
    const {chromium}=require(process.env.PLAYWRIGHT_CORE ||
      'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
    const executablePath=process.env.CHROME_PATH || path.join(os.homedir(),
      'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe');
    browser=await chromium.launch({headless:true,executablePath});
    const context=await browser.newContext({viewport:{width:1440,height:1000},locale:'zh-CN'});
    await context.route(/^https?:/,async route=>{
      report.network.push(route.request().url());await route.abort();
    });
    const page=await context.newPage();
    page.on('pageerror',e=>report.errors.push(e.message));
    page.on('requestfailed',req=>{
      if(req.url().startsWith('file:'))report.errors.push(req.url()+': '+req.failure()?.errorText);
    });
    page.setDefaultTimeout(4500);
    const frame=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    async function fits() {
      await frame();
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1),'页面出现横向溢出');
    }
    async function shot(name) {
      await frame();
      await page.screenshot({path:path.join(resultDir,name+'.png'),animations:'disabled'});
      report.screenshots.push(name+'.png');
    }
    async function check(title,action){
      try{await action();report.checks.push(title);}
      catch(error){error.message=title+' — '+error.message;throw error;}
    }
    await page.goto(pathToFileURL(preview).href);
    await page.evaluate(()=>document.fonts.ready);
    await frame();
    const canvas=page.locator('#model-canvas');
    const originCanvas=await canvas.elementHandle();
    async function objectLabel(block,index) {
      await frame();
      assert.equal(await page.locator('.sys-scene canvas').count(),1,'三个模式应共用一份场景');
      assert.ok(await originCanvas.evaluate(el=>el===document.querySelector('#model-canvas')),'切换时场景被重建');
      const label=await canvas.getAttribute('aria-label');
      assert.match(label,new RegExp('\\b'+block+'(?!\\d).*\\b'+index+'(?!\\d)'),'当前画布位置 '+label+'，预期 '+block+'/'+index);
    }
    // Workflows below use records, labels and source values from the public preview.
    const records=await page.evaluate(()=>window.modelRecords);
    assert.ok(Array.isArray(records)&&records.length>=2);
    const live=page.locator('#mode-live'),archive=page.locator('#mode-archive'),features=page.locator('#mode-features');
    const field=id=>page.locator('#'+id);
    const sceneModel=field('model-select'),coordinate=field('coord-input');
    const rows=page.locator('#archive-list [data-record]');
    const archiveFields=['archive-model','archive-kind','archive-index','archive-q'];
    const filterValues=()=>Promise.all(archiveFields.map(id=>field(id).inputValue()));
    const rowBy=id=>rows.and(page.locator('[data-record="'+id+'"]'));
    async function enterCoordinate(model,value) {
      await sceneModel.selectOption({label:model});
      await coordinate.fill(value);await coordinate.press('Enter');
      await frame();
    }
    async function archiveFilters({model='',kind='',index='',q=''}={}) {
      await field('archive-model').selectOption(model);
      await field('archive-kind').selectOption(kind);
      await field('archive-index').fill(index);
      await field('archive-q').fill(q);
      await frame();
    }
    async function roundTrip(id,viaKeys=false) {
      const row=rowBy(id),list=field('archive-list');
      await row.scrollIntoViewIfNeeded();
      const previous=await list.evaluate(el=>el.scrollTop);
      const filters=await filterValues();
      if(viaKeys){await row.focus();await row.press('Enter');}else{await row.click();}
      const record=records.find(r=>r.id===id);
      assert.ok(record);
      await field('archive-back').waitFor();
      const visible=await page.locator('body').innerText();
      assert.ok(visible.includes(record.result),'详情的测试结论应与原始摘录一致');
      await field('archive-back').click();
      await frame();
      assert.deepEqual(await filterValues(),filters);
      assert.equal(await page.evaluate(()=>document.activeElement.closest('[data-record]')?.dataset.record),id);
      const restored=await list.evaluate(el=>el.scrollTop);
      assert.ok(Math.abs(restored-previous)<=1,'位置从 '+previous+' 恢复为 '+restored);
      return previous;
    }
    await fits();
    if(checkNative){
      const grid=field('native-host'),val=grid.locator('[data-field-value]'),index=grid.locator('[data-field-index]');
      const positionView=grid.locator('[data-field-view="positions"]');
      const comparedValues=()=>grid.locator('[data-field-compare-value]').allTextContents().then(values=>values.map(Number));
      const changeIndex=async value=>{await index.fill(String(value));await index.press('Enter');await frame();};
      const openSample=async id=>{await page.locator('.sys-native-examples > summary').click();await field('native-sample-'+id).click();};
      await check('全屏保留 3D，原生单元入口可查看 B36 全 17,408 个 FFN 单元与四个真实读出位置',async()=>{
        await field('fullscreen').click();
        assert.equal(await field('full-native').isVisible(),false);
        await field('full-units').click();await grid.waitFor();
        assert.equal(await val.innerText(),'2.109375');
        assert.equal(await grid.locator('[data-field-index]').getAttribute('max'),'17407');
        await positionView.click();
        assert.deepEqual(await comparedValues(),[2.109375,-0.4765625,-0.447265625,0.79296875]);
        await changeIndex(17407);
        assert.deepEqual(await comparedValues(),[0.2890625,0.318359375,0.00051116943359375,-0.017822265625]);
        await changeIndex(2429);await shot('ffn-positions');
        await grid.locator('[data-field-view="heat"]').click();
        await field('native-site').selectOption('1');assert.equal(await val.innerText(),'-0.4765625');
        await field('native-site').selectOption('0');await changeIndex(17407);assert.equal(await val.innerText(),'0.2890625');
        await objectLabel('B36','U17407');
        await grid.locator('[data-field-scale]').selectOption('absolute');
        await grid.locator('[data-field-contrast]').check();
        assert.equal(await val.innerText(),'0.2890625');
        await shot('ffn-heatmap');
        await grid.locator('[data-field-view="distribution"]').click();assert.ok(await grid.locator('[data-field-bars] button').count()>0);
        await grid.locator('[data-field-view="table"]').click();assert.ok(await grid.locator('[data-field-row="17407"]').isVisible());
        await field('native-measure').selectOption('gate');assert.equal(await val.innerText(),'未采集');
        assert.equal(await positionView.isVisible(),false);
        await field('native-measure').selectOption('post_gate');
        await field('full-workspace').click();await field('fullscreen').click();await objectLabel('B36','U17407');
        assert.equal(await field('lv-value').innerText(),'0.2890625');
      });
      await check('显式切换到同题 B26/H36 的 128 维输出，切 Head 通道始终保留同一个 Head；未采集的 K 栏标出共享 KV 头',async()=>{
        await field('native-open').click();await openSample('head');
        await grid.locator('[data-field-view="heat"]').click();
        assert.equal(await positionView.isVisible(),false);
        assert.equal(await index.getAttribute('max'),'127');
        assert.equal(await grid.locator('[data-field-source]').innerText().then(t=>t.includes('confirm-000-02')&&t.includes('token 190')),true);
        assert.equal(await val.innerText(),'-1.8671875');
        await changeIndex(1);assert.equal(await val.innerText(),'-0.48046875');await objectLabel('B26','H36');
        await grid.locator('[data-field-canvas]').focus();await grid.locator('[data-field-canvas]').press('End');
        assert.equal(await index.inputValue(),'127');await objectLabel('B26','H36');await shot('head-channels');
        await field('native-measure').selectOption('k');assert.equal(await val.innerText(),'未采集');
        assert.match(await grid.innerText(),/共享 KV 头 7/);
        await field('native-measure').selectOption('z');
        await field('full-workspace').click();await field('lv-custom').click();
        await field('lv-text').fill('用于核对空场的自定义输入');await field('full-units').click();
        assert.equal(await val.innerText(),'未采集');assert.match(await grid.innerText(),/自定义文本/);assert.equal(await positionView.isVisible(),false);
        await field('fullscreen').click();
      });
      await check('历史题的单元图保留原筛选和样本关联；窄屏可操作并返回 3D',async()=>{
        await archive.click();await archiveFilters({model:'Qwen3-14B',index:'B36/U2429'});
        await rowBy('p2933-counted0-u2429').click();
        await page.setViewportSize({width:390,height:844});await frame();
        await field('native-open').click();await field('native-site').selectOption('3');await changeIndex(17407);
        await field('display-open').click();await field('grid').uncheck();await field('grid').check();await field('display-open').click();
        await objectLabel('B36','U17407');
        assert.ok(await field('workbench').evaluate(el=>el.scrollWidth<=el.clientWidth+1),'全屏窄屏出现横向溢出');
        await shot('mobile-native');
        await field('fullscreen').click();await objectLabel('B36','U2429');await field('archive-back').click();
        assert.equal(await field('archive-index').inputValue(),'B36/U2429');
        await page.setViewportSize({width:1440,height:1000});await frame();
        await live.click();
      });
    }
    await check('离线原生场与随包归档的模型快照、位置和全部张量字节一致',async()=>{
      const original=fs.readFileSync(nativeFile);
      const archiveData=JSON.parse(original);
      const expected=await page.evaluate(()=>{
        const f=window.explorerSystemField;
        return {model:f.snapshot,mlp:f.mlp,hidden:f.hidden,pos:f.positions,source:f.sourceFileSha};
      });
      assert.equal(expected.source,createHash('sha256').update(original).digest('hex'));
      assert.equal(expected.model,archiveData.snapshot);
      assert.deepEqual(expected.pos,archiveData.positions);
      assert.equal(expected.hidden,archiveData.hidden);
      assert.equal(expected.mlp,archiveData.mlp);
      report.native={run:archiveData.run,caseId:archiveData.caseId,mlp:archiveData.mlpShape,hidden:archiveData.hiddenShape};
    });
    await check('实时模式使用真实归档注意力和单元测量，按 token/原生坐标读取结果',async()=>{
      await live.click();
      await field('lv-history').click();
      await field('lv-record').selectOption('p2983-confirm-000-02');
      await objectLabel('B26','H36');
      await field('lv-tool').selectOption('native');
      await field('lv-source').selectOption('3');
      assert.match(await field('lv-value').innerText(),/34\.138%/);
      await field('lv-measure').selectOption('q');
      assert.match(await page.locator('.lv-result').innerText(),/待采集/);
      assert.equal(await field('lv-value').count(),0);
      await field('lv-record').selectOption('p2933-counted0-u2429');
      await objectLabel('B36','U2429');
      const position=field('lv-position');
      await position.selectOption({index:0});
      assert.match(await field('lv-value').innerText(),/2\.109375/);
      await position.selectOption({index:1});
      assert.match(await field('lv-value').innerText(),/-0\.4765625/);
      await position.selectOption({index:0});
      await enterCoordinate('Qwen3-14B','B36/U17407');
      await objectLabel('B36','U17407');
      assert.match(await field('lv-value').innerText(),/0\.28906/);
      await enterCoordinate('Qwen3-14B','B36/U2429');
      await fits();await shot('native-live');
    });
    await check('在同一 token 的原生归档上复算数学工具，缺权重的透镜显示所缺数据',async()=>{
      await field('lv-tool').selectOption('hidden');
      await field('lv-hs-layer').selectOption('36');
      await field('lv-position').selectOption('0');
      assert.match(await page.locator('.lv-result').innerText(),/0\.96076/);
      assert.ok(Number(await field('lv-value').innerText())>0);
      await field('lv-tool').selectOption('projection');
      await field('lv-position').selectOption('1');
      assert.equal(await field('lv-value').innerText(),'0.55868');
      await field('lv-position').selectOption('3');
      assert.equal(await field('lv-value').innerText(),'1.00000');
      await field('lv-tool').selectOption('hidden');
      await field('lv-hs-layer').selectOption('0');
      assert.match(await page.locator('.lv-result').innerText(),/无前一层/);
      await field('lv-tool').selectOption('logit');
      assert.match(await page.locator('.lv-result').innerText(),/词表读出权重/);
      assert.equal(await field('lv-value').count(),0);
      await field('lv-tool').selectOption('tuned');
      assert.match(await page.locator('.lv-result').innerText(),/校准映射/);
      assert.equal(await field('lv-value').count(),0);
      await field('lv-tool').selectOption('native');
    });
    await check('选择特征档案并区分正例、反例、缺少分析的原生对象',async()=>{
      await features.click();
      await objectLabel('B36','U2429');
      await page.locator('[data-feature-example="u9517"]').click();
      await objectLabel('B30','U9517');
      let dossier=await field('feature-dossier').innerText();
      for(const value of ['5.219','4.564','5.033','0.333'])assert.ok(dossier.includes(value),'U9517 缺少 '+value);
      await page.locator('[data-feature-tab="sources"]').click();
      assert.ok(await page.locator('[data-feature-record]').count()>=3);
      await page.locator('[data-feature-tab="overview"]').click();
      await shot('feature');
      await page.locator('[data-feature-example="h36"]').click();
      await objectLabel('B26','H36');
      dossier=await field('feature-dossier').innerText();
      for(const rate of ['99.0%','99.8%','33.4%'])assert.ok(dossier.includes(rate),'B26/H36 缺少 '+rate);
      await enterCoordinate('Qwen3-14B','B1/U10');
      await objectLabel('B1','U10');
      assert.match(await field('feature-dossier').innerText(),/该对象尚无特征档案/);
    });
    await check('自定义输入后隐藏上一题的实测值，跨模式回来仍保留原文',async()=>{
      await live.click();await field('lv-custom').click();
      const novel='雨停后，她将 blue cup 放回窗台。';
      await field('lv-text').fill(novel);
      assert.doesNotMatch((await field('lv-value').allInnerTexts()).join(' '),/2\.109375|-0\.4765625|34\.1381/);
      await features.click();await live.click();
      assert.equal(await field('lv-text').inputValue(),novel);
      assert.ok(!(await field('lv-text').isDisabled()));
    });
    await check('从规律索引按领域查到反例原题，读回结果后回到原规律',async()=>{
      await features.click();
      await page.locator('[data-feature-laws]').click();
      await field('law-index').waitFor();
      await field('law-domain').selectOption('lookup');
      const law=page.locator('[data-law-id="lookup"]');
      if(!await law.evaluate(el=>el.open))await law.locator('summary').click();
      await page.locator('[data-law-record="p2983-confirm-000-02"]').click();
      await objectLabel('B26','H36');
      assert.equal(await field('archive-result').textContent(),records.find(r=>r.id==='p2983-confirm-000-02').result);
      await page.getByRole('button',{name:/返回规律索引/}).click();
      assert.equal(await field('law-domain').inputValue(),'lookup');
      await field('law-domain').selectOption('grammar');
      const gap=page.locator('[data-law-id="grammar"]');
      if(!await gap.evaluate(el=>el.open))await gap.locator('summary').click();
      assert.ok(await page.locator('[data-law-stage="grammar"]').isVisible());
      await field('plan-close').click();await live.click();
    });
    await check('补测任务可切换本机与分布式路径并保存纯草案',async()=>{
      await field('lv-config').click();
      const dialog=field('plan-dialog');
      await dialog.waitFor();
      await field('plan-local').click();
      assert.ok(await field('plan-local').isChecked());
      assert.match(await field('route-description').innerText(),/同一模型快照/);
      await field('plan-distributed').click();
      assert.ok(await field('plan-distributed').isChecked());
      assert.match(await field('route-description').innerText(),/冻结家族|分片/);
      await field('plan-save').click();
      if(await dialog.isHidden())await field('plan-open').click();
      assert.match(await field('plan-list').innerText(),/分布式/);
      await field('plan-list').getByText('核对采集配置',{exact:true}).first().click();
      assert.match(await field('plan-list').innerText(),/雨停后，她将 blue cup 放回窗台/);
      await field('plan-questions').click();
      for (const topic of ['知识网络','组合推理','语法系统'])assert.ok((await dialog.innerText()).includes(topic));
      await field('plan-close').click();
    });
    await check('按模型、题目、Head / MLP 和原生编号查找，区分同条记录中多个对象',async()=>{
      await archive.click();
      const target=records.find(r=>r.id==='p2997-first-line-cohort');
      assert.ok(target);
      await archiveFilters({model:target.model,kind:'MLP',index:'B28/U17216',q:target.title});
      assert.equal(await rows.count(),1);
      await rowBy(target.id).click();
      await objectLabel('B28','U17216');
      assert.ok((await page.locator('body').innerText()).includes(target.result));
      await field('archive-back').click();
      assert.deepEqual(await filterValues(),[target.model,'MLP','B28/U17216',target.title]);
      report.multiObject={record:target.id,index:'B28/U17216'};
    });
    await check('整模型题仅显示原有结果，回执状态和对象索引可分别核对',async()=>{
      await archiveFilters({q:'草莓'});
      await field('archive-receipt').check();
      assert.equal(await rows.count(),2);
      const id=await rows.first().getAttribute('data-record');
      await rowBy(id).click();
      assert.match(await field('archive-panel').innerText(),/缺少对象索引/);
      assert.equal(await field('archive-features').count(),0);
      assert.equal(await field('archive-result').textContent(),records.find(r=>r.id===id).result);
      await page.goBack();await rowBy(id).waitFor();await frame();
      assert.ok(await field('archive-receipt').isChecked());
      assert.equal(await field('archive-q').inputValue(),'草莓');
      await field('archive-receipt').uncheck();
    });
    await check('扩展到第二个模型，使用键盘打开归档后回到原行及非零滚动位置',async()=>{
      await archiveFilters({model:'Qwen3-4B',kind:'MLP',index:'B30/U9517'});
      assert.equal(await rows.count(),records.filter(r=>r.model==='Qwen3-4B'&&r.objects.some(o=>o.index===9517)).length);
      await roundTrip('claude261-u9517-s9-2',true);
      await archiveFilters();
      await page.setViewportSize({width:1100,height:660});
      const last=await rows.last().getAttribute('data-record');
      const offset=await roundTrip(last,true);
      assert.ok(offset>0,'最后一条记录应在滚动后可见');
      report.returnTo={id:last,scroll:offset};
      await page.setViewportSize({width:1440,height:1000});
    });
    await check('归档单题可回用为实时输入，也可打开该原生对象的特征档案',async()=>{
      const record=records.find(r=>r.id==='p2983-confirm-000-02');
      await archiveFilters({model:record.model,kind:'Head',index:'B26/H36'});
      await rowBy(record.id).click();
      await field('archive-live').click();
      assert.equal(await field('lv-record').inputValue(),record.id);
      await objectLabel('B26','H36');
      await archive.click();
      if(await field('archive-back').isHidden())await rowBy(record.id).click();
      await field('archive-features').click();
      await objectLabel('B26','H36');await field('feature-dossier').waitFor();
    });
    await check('全屏内三模式共用同一画布、键盘定位和视角',async()=>{
      await field('fullscreen').click();await frame();
      assert.ok(await field('workbench').evaluate(el=>document.fullscreenElement===el||
        [...el.classList].some(c=>/full/i.test(c))),'未进入全屏布局');
      await archive.click();await live.click();await features.click();
      await enterCoordinate('Qwen3-14B','B28/H21');
      await objectLabel('B28','H21');
      await field('fullscreen').click();await frame();
      assert.equal(await page.evaluate(()=>!!document.fullscreenElement),false);
    });
    await check('390 像素页面可在三个模式间切换、定位并返回归档题目',async()=>{
      await page.setViewportSize({width:390,height:844});
      await live.click();await fits();
      await features.click();await fits();
      await archive.click();
      if(await field('archive-back').isVisible())await field('archive-back').click();
      await archiveFilters({model:'Qwen3-4B',index:'B30/U9517'});
      await page.evaluate(()=>window.scrollTo(0,0));
      await fits();await shot('mobile');
      await roundTrip('claude261-u9517-s9-2');await fits();
    });
    if(checkPlugins){
      await page.setViewportSize({width:1440,height:900});await page.reload();await live.click();
      const tool=field('lv-tool');
      await check('三个新插件按需启用，表征几何可核对距离、自比较与缺少输入',async()=>{
        assert.equal(await tool.locator('option[value="geometry"]').count(),0);
        await field('lv-manager').locator('summary').click();
        for(const id of ['geometry','attn-stats','sae'])await field('lv-enable-'+id).check();
        await field('lv-record').selectOption('p2933-counted0-u2429');
        await tool.selectOption('geometry');
        await field('lv-hs-layer').selectOption('37');await field('lv-compare-layer').selectOption('36');
        const result=page.locator('.lv-result');
        assert.match(await result.innerText(),/0\.9918/);assert.match(await result.innerText(),/0\.9429/);
        await result.locator('summary').click();
        assert.equal(await result.locator('tbody tr').count(),6);
        assert.match(await result.locator('table').innerText(),/0\.3435/);
        await shot('geometry-plugin');
        await field('lv-compare-layer').selectOption('37');
        assert.deepEqual(await result.locator('.lv-archived-metrics strong').allTextContents(),['1.0000','1.0000']);
        await field('lv-custom').click();assert.equal(await result.locator('table').count(),0);await field('lv-history').click();
        await field('lv-enable-geometry').uncheck();assert.equal(await tool.inputValue(),'native');
      });
      await check('五行加其余 token 的六组质量完整，排名与 Top-k 累计可核对',async()=>{
        await field('lv-record').selectOption('p2983-confirm-000-02');await tool.selectOption('attn-stats');
        const result=page.locator('.lv-result');
        await field('lv-top-groups').selectOption('6');
        assert.match(await result.innerText(),/100\.00%/);
        await result.locator('summary').click();assert.equal(await result.locator('tbody tr').count(),6);
        assert.match(await result.locator('table').innerText(),/其余 token 合计/);
        await shot('attention-plugin');
        await enterCoordinate('Qwen3-14B','B26/H35');assert.match(await result.innerText(),/所选 Head/);
      });
      await check('SAE 特征独立于原生单元编号，显式转到快照和块后读归档',async()=>{
        await tool.selectOption('sae');
        const result=page.locator('.lv-result');
        assert.match(await result.innerText(),/没有配套 SAE 字典/);
        await result.locator('[data-lv-dictionary-block="8"]').click();
        assert.match(await field('lv-dictionary-feature').locator('option:checked').innerText(),/F1884/);
        assert.equal(await field('lv-dictionary-feature').locator('option').count(),16);
        const coord=await field('current-object').innerText();
        await field('lv-dictionary-feature').selectOption({index:2});
        assert.equal(await field('current-object').innerText(),coord);
        await field('lv-custom').click();await field('lv-text').fill('独立的自定义输入。');
        assert.equal(await field('lv-dictionary-feature').locator('option').count(),16);
        await result.locator('[data-lv-dictionary-block="24"]').click();
        assert.match(await field('lv-dictionary-feature').locator('option:checked').innerText(),/F2269/);
        await result.locator('summary').click();assert.match(await result.innerText(),/重建门槛未通过/);
        await result.locator('[data-lv-dictionary-block="35"]').click();
        await result.locator('summary').click();assert.match(await result.innerText(),/重建门槛通过/);
        await shot('dictionary-plugin');await page.setViewportSize({width:390,height:844});await fits();
      });
    }
    await check('整条流程只读取本地静态数据',async()=>{
      assert.deepEqual(report.network,[]);assert.deepEqual(report.errors,[]);
    });
    report.passed=true;
  } catch(error) {
    report.failure=error.message;throw error;
  } finally {
    if(browser)await browser.close();
    fs.writeFileSync(path.join(resultDir,'acceptance.json'),JSON.stringify(report,null,2)+'\n');
    const actual=fs.realpathSync.native(profile);
    assert.equal(path.dirname(actual),fs.realpathSync.native(parent));
    assert.ok(path.basename(actual).startsWith('explorer-system-qa-'));
    fs.rmSync(actual,{recursive:true,force:true,maxRetries:4,retryDelay:150});
  }
  console.log(JSON.stringify(report));
}
verify().catch(error=>{console.error(error);process.exitCode=1;});
