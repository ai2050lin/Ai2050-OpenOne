const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');

const [host, work, out] = process.argv.slice(2);
const seed = require('../../deploy/workspace_data/catalog.json');
const q14 = seed.models.find(model => model.model === 'Qwen3-14B');
const q4 = seed.models.find(model => model.model === 'Qwen3-4B');
const alt = 'a'.repeat(64);
const original = seed.archives.find(entry => entry.id === 'head');
const privateRun = seed.archives.find(entry => entry.id === 'b26').run;
const route = snapshot => host + '/workbench?' + new URLSearchParams({ wb:'1', lens:'spatial', view:'graph',
  gview:'structure', mdl:'Qwen3-14B', snapshot, observeMode:'live' });
const api = '/api/research/workspace';

async function main() {
  const browser = await chromium.launch({ headless:true, executablePath:process.env.CHROME_PATH ||
    'C:/Program Files/Google/Chrome/Application/chrome.exe' });
  try {
    const context = await browser.newContext({ viewport:{ width:1440, height:900 } });
    const login = await context.request.post(host+'/api/auth/login',{data:{username:'workspace-qa',password:process.env.AI2050_WORKSPACE_QA_PASSWORD}});
    assert.equal(login.status(),200);
    const session = await login.json();
    await context.addInitScript(({host, token, user})=>{
      const center=new URLSearchParams(location.search).has('qa_other_center')?host.replace('127.0.0.1','localhost'):host;
      window.__AI2050__={apiBase:center,mode:'remote'};
      if (!sessionStorage.getItem('workspace-qa-initialized')) {
        localStorage.setItem('ai2050-session:'+host,token);
        localStorage.setItem('ai2050-user:'+host,JSON.stringify(user));
        sessionStorage.setItem('workspace-qa-initialized','yes');
      }
    },{host,...session});
    const page=await context.newPage();
    page.setDefaultTimeout(14000);
    const errors=[], live=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.route(/\/api\/live\//,route=>{
      if (route.request().method()!=='GET') live.push(route.request().url());
      return route.abort();
    });
    await page.goto(route(q14.model_snapshot));
    const scene=page.getByRole('region',{name:'模型运行 3D 场景'});
    const objects=scene.getByRole('group',{name:'有归档的对象'});
    const reading=scene.getByRole('region',{name:'3D 原生读数'});
    const button=(group,name)=>group.getByRole('button',{name,exact:true});
    const open=name=>button(objects,name).click();
    const raw=(label,expected)=>page.waitForFunction(({label,expected})=>
      document.querySelector(`output[aria-label="${label}"]`)?.textContent===String(expected),{label,expected});
    const head=expected=>raw('Head 通道原值',expected);
    const mlp=expected=>raw('MLP 单元原值',expected);
    const model=scene.getByLabel('模型',{exact:true});
    await model.locator(`option[value="${q14.model_snapshot}"]`).waitFor({state:'attached'});
    assert.equal(await model.inputValue(),q14.model_snapshot);
    await open('B23/U7708');
    await reading.getByText('wikipedia-zh:18501',{exact:false}).waitFor();
    await open('B36/U3929');
    await scene.getByLabel('3D 归档读出位置').waitFor();
    await page.waitForFunction(()=>{
      const value=document.querySelector('output[aria-label="MLP 单元原值"]')?.textContent;
      return value && Number.isFinite(Number(value));
    });
    await open('B26/H36');
    await head(-1.8671875);
    assert.equal(await reading.getByLabel('3D Head 通道').locator('option').count(),128);
    assert.match(await reading.innerText(),/confirm-000-02 · token 190（ID 510）/);
    await page.screenshot({path:path.join(out,'original-head.png')});
    await open('B26/H39');
    const lookup=scene.getByLabel('输入原生编号',{exact:true});
    await lookup.fill('B26/H36');
    await lookup.press('Enter');
    await head(-1.859375);
    await open('B26/U17407');
    await mlp(0.0299072265625);
    const positions=scene.getByLabel('3D 归档读出位置');
    await positions.selectOption('95');
    await mlp(0.412109375);
    await button(scene,'原生单元').click();
    const full=scene.getByRole('region',{name:'全屏原生单元'});
    await full.locator('[data-field-value]').filter({hasText:'0.412109375'}).waitFor();
    assert.equal(await full.locator('[data-native-inspector]').getAttribute('data-size'),'17408');
    assert.equal(await full.getByLabel('全屏归档读出位置').inputValue(),'95');
    await page.screenshot({path:path.join(out,'private-mlp.png')});
    await button(scene,'3D 与控制台').click();
    await lookup.fill('B25/U17407');
    await lookup.press('Enter');
    await mlp('未采集');
    await reading.getByRole('status').filter({hasText:'未随该归档发布'}).waitFor();

    await page.goto(host+'/workbench?'+new URLSearchParams({wb:'1',lens:'route',rtab:'database',record:privateRun}));
    const records=page.getByRole('region',{name:'测试记录'});
    await records.getByText('逐题结果与下载',{exact:true}).click();
    await button(records,'读取逐题结果').click();
    await records.getByText('confirm-000-02',{exact:true}).click();
    await records.getByRole('link',{name:'在 3D 中回放本题'}).click();
    await scene.getByRole('group',{name:'有归档的对象'}).waitFor();
    await scene.getByLabel('3D 归档读出位置').waitFor();
    assert.equal(await scene.getByLabel('3D 归档读出位置').inputValue(),'190');
    await open('B26/U17407');
    await mlp(0.0299072265625);

    await page.goto(host+'/workbench?wb=1&lens=route&rtab=database');
    await records.getByLabel('记录来源').selectOption('goodfire');
    await records.getByLabel('搜索测试').fill('1111');
    await button(records,'新建测试').click();
    const newTest=page.getByRole('region',{name:'新建测试'});
    await newTest.getByLabel('导入冻结方案（JSON）').waitFor();
    await newTest.getByLabel('先查相似记录').fill('1111');
    await newTest.getByRole('button',{name:/Phase \d/}).first().waitFor();
    await page.goto(route(q14.model_snapshot));

    await model.selectOption(q4.model_snapshot);
    await open('B35/U9727');
    await reading.getByText('LIVE-NATIVE-OBSERVATION',{exact:false}).waitFor();
    await model.selectOption(q14.model_snapshot);

    const published=path.join(work,'publish');
    fs.copyFileSync(path.resolve(__dirname,'../../deploy/workspace_data/head.json'),path.join(published,'qa-head.json'));
    fs.writeFileSync(path.join(published,'catalog.json'),JSON.stringify({schema:'workspace_publication.v1',
      models:[{...q14,model_snapshot:alt}],archives:[{...original,id:'qa-added',file:'qa-head.json',title:'新增中心入口',
        location:{layer:27,module:'attn',index:36}}]}));
    await button(objects,'同步中心目录').click();
    await objects.getByTitle('新增中心入口').waitFor();
    await model.selectOption(alt);
    assert.equal(await objects.getByTitle('新增中心入口').count(),0);
    await model.selectOption(q14.model_snapshot);

    await objects.getByTitle(original.title).click();
    await head(-1.8671875);
    await page.route('**'+api+'/**',request=>request.abort());
    await button(objects,'同步中心目录').click();
    await objects.getByText('上次同步的公开目录',{exact:false}).waitFor();
    await head(-1.8671875);
    await page.evaluate(host=>{localStorage.removeItem('ai2050-session:'+host);localStorage.removeItem('ai2050-user:'+host);},host);
    await page.reload();
    await button(objects,'B26/H36').first().waitFor();
    assert.equal(await button(objects,'B26/H39').count(),0);
    await button(objects,'B26/H36').first().click();
    await head(-1.8671875);

    const other=await context.newPage();
    await other.route('**/api/**',route=>route.abort());
    await other.goto(route(q14.model_snapshot)+'&qa_other_center=1');
    const otherScene=other.getByRole('region',{name:'模型运行 3D 场景'});
    await otherScene.getByText('当前快照没有可访问的归档入口。').waitFor();
    assert.equal(await button(otherScene,'B26/H36').count(),0);
    assert.deepEqual(errors,[]);
    assert.deepEqual(live,[]);
    fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({passed:true,oldD0:-1.8671875,rerunD0:-1.859375,
      mlpAt95:0.412109375,fullMLP:17408,headChannels:128,missingBlock:25,publicationAppearsWithoutBuild:true,
      sameNameDifferentSnapshot:true,recordToReplay:true,newTestPanel:true,publicOffline:true,privateOfflineHidden:true,
      centerCacheIsolation:true,noLiveRequests:true},null,2)+'\n');
    console.log('Browser archive, permissions, dynamic publication and offline checks passed.');
  } finally {
    await browser.close();
  }
}

main().catch(error=>{ console.error(error); process.exitCode=1; });
