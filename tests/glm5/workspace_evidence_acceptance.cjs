const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');

const [host, , output] = process.argv.slice(2);
const database = host+'/workbench?wb=1&lens=route&rtab=database';
const executablePath = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const caseId = 'confirm-000-02';

async function readout(page,label,value) {
  await page.waitForFunction(({label,value})=>
    document.querySelector(`output[aria-label="${label}"]`)?.textContent===String(value),{label,value});
}

async function guestChecks(page) {
  await page.goto(database);
  const lab = page.locator('[aria-label="统一证据检索"]');
  await lab.getByRole('button',{name:'学习案例 · 读一题选源实验'}).click();
  await lab.getByLabel('搜索证据').fill(caseId);
  await lab.getByRole('status').filter({hasText:'1 条匹配'}).waitFor();
  await lab.getByRole('button',{name:new RegExp(caseId)}).first().click();
  const detail = lab.getByRole('article',{name:'证据链详情'});
  assert.equal(await detail.getByText('实际输出',{exact:false}).count(),0);
  await detail.getByRole('button',{name:'查看模型输出与反例'}).click();
  await detail.getByText('实际输出',{exact:false}).waitFor();
  await detail.getByRole('link',{name:'3D 原值 · token 190（ID 510）'}).click();
  await readout(page,'Head 通道原值',-1.8671875);
  await page.screenshot({path:path.join(output,'public-raw.png')});
}

async function adminChecks(page) {
  await page.goto(database);
  const lab = page.locator('[aria-label="统一证据检索"]');
  await lab.getByRole('status').filter({hasText:'条匹配'}).waitFor();
  await lab.getByText('本次可查范围',{exact:true}).click();
  await lab.getByRole('table',{name:'证据覆盖统计'}).getByText('4861',{exact:true}).waitFor();
  await lab.getByLabel('搜索证据').fill(caseId);
  await lab.getByLabel('记录类型').selectOption('case');
  await lab.getByLabel('证据覆盖').selectOption('native');
  await lab.getByText('按快照、家族、部件和 Token 精确筛选',{exact:true}).click();
  await lab.getByLabel('原生部件').fill('B26/U17407');
  await lab.getByLabel('Token 位置',{exact:true}).fill('95');
  await lab.getByRole('status').filter({hasText:'1 条匹配'}).waitFor();
  await lab.getByRole('button',{name:new RegExp(caseId)}).first().click();
  const detail = lab.getByRole('article',{name:'证据链详情'});
  await detail.getByRole('button',{name:'查看同题原档'}).waitFor();
  await page.screenshot({path:path.join(output,'search-raw-link.png')});
  await detail.getByRole('link',{name:/3D 原值 · token 95/}).click();
  await readout(page,'MLP 单元原值',0.412109375);
  const scene = page.getByRole('region',{name:'模型运行 3D 场景'});
  assert.equal(await scene.getByLabel('3D 归档读出位置').inputValue(),'95');
  await page.screenshot({path:path.join(output,'private-exact-token.png')});

  await page.goto(database);
  await lab.getByLabel('搜索证据').fill('gf-2961');
  await lab.getByLabel('记录类型').selectOption('study');
  await lab.getByRole('status').filter({hasText:'条匹配'}).waitFor();
  await lab.getByRole('button',{name:/同义词|同义措辞/}).first().click();
  await detail.getByText('相对前轮新增验证',{exact:true}).waitFor();
  await detail.getByText('行为门',{exact:true}).waitFor();
  await detail.getByText('反例与限制',{exact:true}).waitFor();

  await lab.getByLabel('搜索证据').fill(caseId);
  await lab.getByLabel('记录类型').selectOption('case');
  await lab.getByLabel('证据覆盖').selectOption('native');
  await lab.getByRole('status').filter({hasText:'2 条匹配'}).waitFor();
  await lab.getByRole('button',{name:/私有归档/}).click();
  await detail.getByRole('button',{name:'基于此原场新建分析'}).click();
  const draft = page.getByRole('region',{name:'新建测试'});
  await draft.getByLabel('冻结方案内容').waitFor();
  assert.match(await draft.getByLabel('冻结方案内容').inputValue(),/research_test_plan\.v1/);
}

async function main() {
  const browser = await chromium.launch({headless:true,executablePath});
  const exceptions = [], writes = [];
  try {
    const guest = await browser.newContext({viewport:{width:1450,height:920}});
    const user = await browser.newContext({viewport:{width:1450,height:920}});
    const session = await user.request.post(host+'/api/auth/login',{data:{username:'workspace-qa',password:process.env.AI2050_WORKSPACE_QA_PASSWORD}});
    assert.equal(session.status(),200);
    await user.addInitScript(({host,token,user})=>{
      window.__AI2050__={apiBase:host,mode:'remote'};
      if(!sessionStorage.getItem('evidence-qa')){
        localStorage.setItem('ai2050-session:'+host,token);
        localStorage.setItem('ai2050-user:'+host,JSON.stringify(user));
        sessionStorage.setItem('evidence-qa','yes');
      }
    },{host,...await session.json()});
    for(const [context,checks] of [[guest,guestChecks],[user,adminChecks]]){
      const page = await context.newPage();
      page.setDefaultTimeout(16000);
      page.on('pageerror',error=>exceptions.push(error.message));
      page.on('request',request=>{
        if(!['GET','HEAD','OPTIONS'].includes(request.method())&&/\/api\/(live|test-archive)\//.test(request.url()))writes.push(request.url());
      });
      await checks(page);
    }
    assert.deepEqual(exceptions,[]);
    assert.deepEqual(writes,[]);
    fs.writeFileSync(path.join(output,'evidence-browser.json'),JSON.stringify({passed:true,guest_case_to_3d:true,
      authorized_exact_token_to_3d:true,history_coverage_display:true,retest_delta_and_failed_gate:true,
      reuse_draft:true,original_D0:-1.8671875,rerun_token_95_U17407:0.412109375,no_model_or_archive_writes:true},null,2)+'\n');
    console.log('Unified search, guest lesson, exact native replay and reuse draft verified.');
  } finally {
    await browser.close();
  }
}

main().catch(error=>{console.error(error);process.exitCode=1;});
