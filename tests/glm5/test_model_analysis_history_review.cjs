'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {pathToFileURL} = require('node:url');
const {checkedDir} = require('../../frontend/electron/storagePolicy');

const root = path.resolve(__dirname, '../..');
const source = path.join(root,'research/glm5/docs/previews/model-analysis-history-review.html');
const out = checkedDir(path.join(__dirname,'result/model_analysis_history_review'),32*2**20);
const scratch = checkedDir(path.join(root,'tests/glm5_temp'),300*2**20);
const profile = fs.mkdtempSync(path.join(scratch,'model-analysis-history-review-'));
for (const key of ['TEMP','TMP','TMPDIR']) process.env[key]=profile;
const report = {passed:false,checks:[],errors:[],http:[],screenshots:[]};
const {chromium} = require(process.env.PLAYWRIGHT_CORE ||
  'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const executablePath = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

async function verify() {
  let browser;
  try {
    browser = await chromium.launch({headless:true,executablePath});
    const page = await browser.newPage({viewport:{width:1440,height:1000},locale:'zh-CN'});
    await page.route(/^https?:/, r=>{report.http.push(r.request().url());return r.abort();});
    page.on('pageerror',e=>report.errors.push(e.message));
    page.on('requestfailed',r=>{if(!r.url().startsWith('http')) report.errors.push(r.url()+': '+r.failure()?.errorText);});
    const shot=async name=>{
      await page.screenshot({path:path.join(out,name),fullPage:true});
      report.screenshots.push(name);
    };
    await page.goto(pathToFileURL(source).href);
    const evidence=page.getByRole('complementary',{name:'当前对象读数'});
    const detail=page.getByRole('article',{name:'所选实验详情'});
    await evidence.getByText('silver',{exact:true}).waitFor();
    await evidence.getByText('第 4 行',{exact:true}).first().waitFor();
    assert.match(await detail.innerText(),/Q\/K 候选未进正式原模型干预/);
    await shot('01-模型分析与相关测试.png');
    report.checks.push('默认打开 B26/H36 的原生结构、同题读数和已有历史');

    const saved=await page.evaluate(()=>window.historyReviewData);
    const sha=f=>createHash('sha256').update(fs.readFileSync(f)).digest('hex');
    for(const {path:file,sha256} of Object.values(saved.sources)) assert.equal(sha(path.join(root,file)),sha256,file);
    const original=JSON.parse(fs.readFileSync(path.join(root,saved.sources.head.path),'utf8')).runs[0].families[0];
    for(const c of saved.head.cases) assert.deepEqual(Object.fromEntries(Object.entries(c).filter(([k])=>k!=='prompt')),original.cases.find(x=>x.id===c.id));
    const old=fs.readFileSync(path.join(root,saved.sources.old.path),'utf8').trim().split(/\r?\n/).map(JSON.parse);
    for (const example of saved.old) {
      const actual=old.find(r=>r.id===example.id);
      assert.equal(example.top_token,actual.top_token);
      assert.equal(example.gold_token,actual.gold_token);
      assert.equal(example.run_id,actual.run_id);
    }
    report.checks.push('四份来源文件完整哈希及头部、结案逐题摘录一致');

    await evidence.getByRole('button',{name:'按姓名',exact:true}).click();
    await evidence.getByText('wood',{exact:true}).waitFor();
    await evidence.getByRole('button',{name:'按印刷 ROW'}).click();
    await evidence.getByText('lion',{exact:true}).waitFor();
    await evidence.getByRole('button',{name:'按物理首行'}).click();
    await evidence.getByRole('button',{name:/查看原题和行表/}).click();
    const caseDialog=page.getByRole('dialog',{name:'原题与模型实答'});
    assert.match(await caseDialog.getByRole('row',{name:/1 C Kifota silver/}).innerText(),/10.4% · 目标/);
    assert.match(await caseDialog.getByRole('row',{name:/4 A Loneru glass/}).innerText(),/34.1% · 峰值/);
    await caseDialog.getByText('实际模型输入与来源').click();
    assert.match(await caseDialog.innerText(),/Active Archive Orion/);
    assert.match(await caseDialog.innerText(),/SHA-256/);
    await caseDialog.getByRole('button',{name:'返回当前视角'}).click();
    assert.match(await evidence.innerText(),/silver/);
    report.checks.push('同表三问只换问法，真实输出、峰值、目标及实际模型输入可回查');

    await page.getByRole('button',{name:/核对复测方案/}).click();
    const review=page.getByRole('dialog',{name:'开始之前，先读旧结果'});
    assert.match(await review.innerText(),/同一命题 · 沿用结案/);
    assert.match(await review.innerText(),/27\/256/);
    await shot('02-只换人名的复测核对.png');
    await review.getByText('查看供 AI 参考的历史证据',{exact:true}).click();
    assert.match(await review.locator('.ai-context').innerText(),/仅换人名或题量沿用结案/);
    assert.match(await review.locator('.ai-context').innerText(),/record-72bac7a5/);
    await review.getByLabel('复用原运行').check();
    await review.getByText('可先复用归档',{exact:true}).waitFor();
    await review.getByLabel('带新证据复核').check();
    await review.getByLabel(/改变原判断的证据/).fill('已有运行引用与判据');
    await review.getByText('待核对新增证据',{exact:true}).waitFor();
    await review.getByRole('button',{name:'打开相匹配的旧记录'}).click();
    assert.match(await detail.innerText(),/候选交接干预未执行/);
    await detail.getByRole('button',{name:'查原题 / 原文'}).click();
    const oldDialog=page.getByRole('dialog',{name:'旧行号路线的真实反例'});
    const negative=oldDialog.getByRole('row',{name:/000-full_old-01/});
    assert.match(await negative.innerText(),/metal\s+wolf/);
    await oldDialog.getByRole('button',{name:'返回当前视角'}).click();
    assert.equal(await page.getByRole('searchbox').inputValue(),'3004');
    report.checks.push('原场复用、结案重提与新增证据入口可区分，供 AI 参考的证据包含结案与运行');

    await page.getByRole('searchbox').fill('无此记录');
    await page.getByText('当前筛选没有匹配的摘录。',{exact:true}).waitFor();
    await page.getByRole('button',{name:'清除筛选'}).click();
    assert.equal(await page.locator('#history-list .record-item').count(),5);
    await page.getByRole('button',{name:'实时模式',exact:true}).click();
    await page.getByLabel('输入文本').fill('历史回查状态验证');
    await page.getByRole('button',{name:'特征分析',exact:true}).click();
    await page.getByLabel('原生对象').selectOption('unit');
    await evidence.getByText('0.58984375',{exact:true}).waitFor();
    await evidence.getByRole('button',{name:'负端语境'}).click();
    await evidence.getByText('-0.6328125',{exact:true}).waitFor();
    await page.getByRole('button',{name:'插件分析',exact:true}).click();
    assert.equal(await page.locator('.heat-line span').count(),32);
    assert.match(await page.locator('.heat-line .chosen').getAttribute('title'),/U7708 · -0.6328125/);
    assert.match(await page.locator('#scene-address').innerText(),/B23\/U7708/);
    await shot('03-U7708特征与实测数值.png');
    await page.getByRole('button',{name:'⛶ 全屏'}).click();
    assert.ok(await page.locator('#workspace').evaluate(e=>e.classList.contains('full-screen')));
    await page.keyboard.press('Escape');
    assert.ok(!(await page.locator('#workspace').evaluate(e=>e.classList.contains('full-screen'))));
    await page.getByRole('button',{name:'实时模式',exact:true}).click();
    assert.equal(await page.getByLabel('输入文本').inputValue(),'历史回查状态验证');
    report.checks.push('三模式与原生对象、实测 MLP 局部热图及全屏状态联动');

    await page.setViewportSize({width:390,height:844});
    await page.getByRole('button',{name:'测试记录',exact:true}).click();
    await page.getByLabel('原生对象').selectOption('head');
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.getByRole('button',{name:/核对复测方案/}).click();
    await review.getByText('同一命题 · 沿用结案',{exact:true}).waitFor();
    assert.ok(await review.evaluate(e=>e.scrollWidth<=e.clientWidth));
    await page.keyboard.press('Escape');
    await shot('04-窄屏历史回查.png');
    report.checks.push('390 像素窄屏可读、可筛选、弹窗可用');
    assert.deepEqual(report.errors,[]);
    assert.deepEqual(report.http,[]);
    report.passed=true;
  } finally {
    if (browser) await browser.close();
    const real=fs.realpathSync(profile);
    assert.equal(path.dirname(real),fs.realpathSync(scratch));
    assert.ok(path.basename(real).startsWith('model-analysis-history-review-'));
    const bytes=dir=>fs.readdirSync(dir,{withFileTypes:true}).reduce((n,e)=>{
      const f=path.join(dir,e.name);
      assert.ok(!e.isSymbolicLink());
      return n+(e.isDirectory()?bytes(f):fs.statSync(f).size);
    },0);
    report.removedTransientBytes=bytes(real);
    fs.rmSync(real,{recursive:true,force:true});
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2));
  }
}
verify().then(()=>console.log(JSON.stringify(report))).catch(e=>{console.error(e);process.exitCode=1;});

