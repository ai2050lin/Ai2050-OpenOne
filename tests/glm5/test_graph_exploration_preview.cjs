const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {homedir} = require('node:os');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');

const root = path.resolve(__dirname, '../..');
const preview = path.join(root, 'research/glm5/docs/previews/graph-exploration-review.html');
const output = path.join(__dirname, 'result/graph_exploration_preview');
const temp = path.join(root, 'tests/glm5_temp/graph_exploration_preview');
const archive = require('../../frontend/src/components/v8/threeCueEvidence.json');

async function run() {
  for (const dir of [output, temp]) fs.mkdirSync(dir, {recursive:true});
  process.env.TEMP = temp;
  process.env.TMP = temp;
  process.env.TMPDIR = temp;
  const browser = await chromium.launch({
    headless:true,
    executablePath:process.env.CHROME_PATH || path.join(homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe'),
    env:{...process.env,TEMP:temp,TMP:temp,TMPDIR:temp},
  });
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    page.setDefaultTimeout(5000);
    const network = [], errors = [];
    page.on('request', request => {if (/^https?:/.test(request.url())) network.push(request.url());});
    page.on('pageerror', error => errors.push(error.message));
    await page.route(/^https?:/, route=>route.abort());
    await page.goto(pathToFileURL(preview).href);
    const tasks = page.getByRole('tablist',{name:'控制台任务'});
    const view = page.locator('#task-panel');
    const button = name => page.getByRole('button',{name,exact:true});
    const tab = name => page.getByRole('tab',{name,exact:true});
    const shot = name => page.screenshot({path:path.join(output,name+'.png'),fullPage:true});
    await view.getByRole('heading',{name:'B26 / H36'}).waitFor();
    assert.deepEqual(await tasks.getByRole('tab').allTextContents(), ['对象详情','运行与对照','插件分析','成果与方法']);
    const scene = await page.getByRole('region',{name:'模型结构'}).boundingBox();
    const consoleBox = await page.getByRole('region',{name:'控制台'}).boundingBox();
    assert.ok(consoleBox.y >= scene.y + scene.height);
    assert.equal(await page.locator('#model').count(),1);
    assert.deepEqual(await page.locator('#analysis button').allTextContents(),['HiddenState','MLP','Head','Logit Lens']);
    await shot('model');
    const layerTower=page.locator('#scene');
    const canvas=await layerTower.boundingBox();
    await layerTower.click({position:{x:Math.round(canvas.width*.22),y:Math.round(canvas.height-28-10*(canvas.height-65)/40)}});
    assert.equal(await page.locator('#layer').inputValue(),'10');
    await page.locator('#layer').fill('27');
    await page.locator('#layer').press('Tab');

    await button('精确定位 ↓').click();
    assert.equal(await page.evaluate(()=>document.activeElement.id),'layer');
    await button('查看相关测试').click();
    assert.equal(await view.getByRole('link',{name:'在测试数据库查看 →'}).getAttribute('href'),
      './client-distributed-testing.html#?wb=1&lens=route&rtab=database&dbStudy=2983');
    await button('运行响应').click();
    assert.equal(await view.locator('progress').count(), 0);
    await button('回放样本').click();
    const source = archive.runs[0].families[0].cases.find(item=>item.id==='confirm-000-00');
    assert.deepEqual(await view.locator('progress').evaluateAll(items=>items.map(item=>item.value)),source.order.map(id=>source.mass[id]));
    await view.getByText(source.output,{exact:true}).first().waitFor();
    await button('条件对照').click();
    assert.equal(await view.locator('.pair .result').count(),2);
    assert.match(await view.innerText(), /按姓名[\s\S]*按物理首行/);
    const second=archive.runs[0].families[0].cases.find(item=>item.id==='confirm-000-02');
    const secondCard=view.locator('.pair .result').last();
    const top=second.mass.indexOf(Math.max(...second.mass));
    assert.match(await secondCard.innerText(), new RegExp('目标第 '+(second.order.indexOf(second.gold)+1)+' 行'));
    assert.deepEqual(await secondCard.locator('dd').allTextContents(),
      ['第 '+(second.order.indexOf(top)+1)+' 行',second.output,archive.runs[0].families[0].codes[second.gold]]);
    await shot('comparison');

    await button('运行响应').click();
    await page.locator('#run-source').selectOption('manual');
    await page.getByRole('textbox',{name:'输入文本'}).fill('一条可保留的草稿 <>&');
    await tab('对象详情').click();
    await tab('运行与对照').click();
    assert.equal(await page.getByRole('textbox',{name:'输入文本'}).inputValue(),'一条可保留的草稿 <>&');
    assert.ok(await button('执行模型').isDisabled());
    await page.locator('#model').selectOption('Qwen3-4B');
    assert.equal(await page.locator('#index').inputValue(),'31');
    assert.equal(await page.locator('#layer').getAttribute('max'),'36');
    await button('条件对照').click();
    assert.match(await view.innerText(), /没有配对读数/);
    await button('定位 B26 / H36').click();
    assert.equal(await view.locator('.pair .result').count(),2);
    assert.equal(await page.locator('#index').inputValue(),'36');

    await button('管理插件').click();
    await page.getByText('其他已接入工具',{exact:true}).click();
    await page.getByRole('checkbox',{name:'三类检验'}).check();
    await button('关闭').click();
    await button('三类检验').click();
    assert.match(await view.innerText(),/原生机制检验/);
    await tab('语言图谱').click();
    await page.getByRole('heading',{name:'查询实体指称'}).waitFor();
    await tab('推理功能').focus();
    await tab('推理功能').click();
    await page.getByRole('heading',{name:'记录与选源'}).waitFor();
    await page.getByRole('searchbox',{name:'筛选语言条目'}).fill('unmatched');
    await page.getByRole('heading',{name:'没有匹配条目'}).waitFor();
    await page.getByRole('searchbox',{name:'筛选语言条目'}).fill('RS.RECORD');
    await page.getByRole('heading',{name:'记录与选源'}).waitFor();
    await shot('language');
    await tab('语言图谱').focus();
    await page.keyboard.press('ArrowLeft');
    assert.equal(await tab('模型探索').getAttribute('aria-selected'),'true');
    await page.setViewportSize({width:390,height:844});
    await tab('对象详情').click();
    await shot('mobile');
    for (const width of [390,320]) {
      await page.setViewportSize({width,height:844});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    }
    await page.setViewportSize({width:1440,height:1000});
    await tab('语言图谱').click();
    await page.getByRole('link',{name:'查看相关测试 →'}).click();
    await page.locator('.wide-evidence dialog[open]').waitFor();
    assert.match(await page.locator('.wide-evidence dialog[open]').innerText(),/Phase 2983/);
    assert.deepEqual(network,[]);
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify({passed:true,browser:browser.version(),
      desktop:[1440,1000],mobileWidths:[390,320],historicalCase:source.id,comparedRules:2,
      sharedContext:true,armedExecution:false,languageDirectory:true,archiveLink:true,pluginsUnderScene:true,network,errors},null,2)+'\n');
    console.log('Graph review passed.');
  } finally {
    await browser.close();
  }
}
run().catch(error=>{console.error(error);process.exitCode=1;});
