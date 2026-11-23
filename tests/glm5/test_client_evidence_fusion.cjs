/* File-only acceptance of the progress, archive and model-space paths in the fused client preview. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL, fileURLToPath} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root = path.resolve(__dirname,'../..');
const output = path.join(__dirname,'result/client_evidence_fusion');
const scratch = path.join(root,'tests/glm5_temp/client_evidence_fusion');
const url = pathToFileURL(path.join(root,'research/glm5/docs/previews/client-evidence-fusion.html')).href;
const progress = '#?wb=1&lens=route&rtab=rules&mdl=Qwen3-14B';
const analytic = '#view=compare&pick=2983%2C2990&rule=2&area=全部&lang=全部&model=全部&q=&part=B26%2FH36';
async function main(){
  [output,scratch].forEach(p=>fs.mkdirSync(p,{recursive:true}));
  const browser = await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH ||
    path.join(require('node:os').homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe'),
    downloadsPath:scratch,env:{...process.env,TEMP:scratch,TMP:scratch,TMPDIR:scratch}});
  try{
    const ctx = await browser.newContext({viewport:{width:1440,height:920},acceptDownloads:true});
    const page = await ctx.newPage();
    page.setDefaultTimeout(6000);
    const errors=[], network=[], links=new Set();
    page.on('pageerror',e=>{errors.push(e.message);console.error(e.message);});
    page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
    await ctx.route(/^https?:/,route=>route.abort());
    const wide=page.locator('.wide-evidence');
    const act=(key,value)=>wide.locator('[data-act="'+key+'"]'+(value===undefined?'':'[data-value="'+value+'"]')+':visible').first().click();
    const view=id=>wide.locator('[data-view="'+id+'"]').click();
    const stats=()=>wide.locator('.stat small').allTextContents();
    const checkLinks=async()=>{
      for(const a of await wide.locator('a:visible').all()){
        const href=await a.getAttribute('href');
        if(!href||href.startsWith('data:'))continue;
        const p=fileURLToPath(new URL(href,page.url()));
        assert.ok(fs.existsSync(p),p);links.add(p);
      }
    };
    const ready=()=>wide.getByRole('heading',{level:1}).waitFor();
    for(const hash of [progress,analytic,'']){
      await page.goto('about:blank');await page.goto(url+hash);await ready();
      assert.equal(await page.locator('.wb-subnav .on').innerText(),'成果与证据');
      assert.deepEqual(await stats(),['377 / 1,024 题','1,018 / 1,024 题']);
      assert.match(await wide.locator('.case-total').innerText(),/642/);
      assert.match(await wide.locator('.chart-foot').innerText(),/256 个确认家族/);
    }
    const shot=name=>page.screenshot({path:path.join(output,name+'.png')});
    await shot('comparison');
    await act('study',2997);assert.match(await wide.locator('#dialog-body').innerText(),/B28\/H21/);await page.keyboard.press('Escape');
    for(const [kind,n] of Object.entries({both:376,headOnly:1,div:642,neither:5})){
      await wide.locator('.case-matrix [data-value="'+kind+'"]').click();
      assert.equal(await wide.locator('#list-kind').inputValue(),kind);
      assert.match(await wide.locator('#list-table .caption').innerText(),new RegExp('^'+n+' 道'));
      await page.keyboard.press('Escape');
    }
    await act('browse','div');assert.match(await wide.locator('#list-table').innerText(),/642 道/);
    await act('next');await act('case');assert.match(await wide.locator('#dialog-body').innerText(),/表外注意力质量/);
    await checkLinks();await shot('case');await page.keyboard.press('Escape');
    await act('rule',0);assert.deepEqual(await stats(),['1,015 / 1,024 题','1,024 / 1,024 题']);
    await act('rule',1);assert.match(await wide.locator('.chart-foot').innerText(),/按印刷标签/);
    await act('rule',2);await wide.locator('[data-pick="2983"]').uncheck();
    assert.deepEqual(await stats(),['206 / 512 题','508 / 512 题']);
    await wide.locator('[data-pick="2997"]').check();assert.match(await wide.locator('#main').innerText(),/相关测试并读/);
    await act('export');
    const [dl]=await Promise.all([page.waitForEvent('download'),wide.locator('#download-set').click()]);
    const exported=JSON.parse(fs.readFileSync(await dl.path(),'utf8'));await dl.delete();
    assert.deepEqual([exported.total.n,exported.total.div,exported.alongside[0].phase],[512,303,2997]);await page.keyboard.press('Escape');
    assert.deepEqual(exported.outcomes,{both:205,headOnly:1,div:303,neither:3});
    await act('ledger');await page.locator('.ck-waffle button').first().waitFor();
    const ruleCount=await page.locator('.ck-waffle button').count();assert.ok(ruleCount>0);
    await page.locator('.ck-waffle button').first().click();
    await page.getByRole('button',{name:'← 返回证据分析'}).click();await ready();
    assert.deepEqual(await stats(),['206 / 512 题','508 / 512 题']);
    await act('openpart','B26/H36');
    const partToScene=async(part,expected)=>{
      await act('part',part);await act('spatial',part);await page.locator('.aw-space canvas').waitFor();
      const query=await page.evaluate(()=>Object.fromEntries(new URLSearchParams(window.reviewRoute.search)));
      assert.deepEqual([query.mdl,query.sblock,query.smodule,query.sindex],expected);
      assert.equal(await page.getByRole('tab',{name:'对象详情',exact:true}).getAttribute('aria-selected'),'true');
      await page.getByRole('button',{name:'返回成果与证据',exact:true}).click();await ready();
      assert.equal(await wide.locator('[data-act="part"][aria-pressed="true"]').innerText(),part);
    };
    await partToScene('B26/H36',['Qwen3-14B','27','attn','36']);
    await partToScene('B28/H21',['Qwen3-14B','29','attn','21']);
    await partToScene('U17216',['Qwen3-14B','29','mlp','17216']);
    await view('compare');assert.deepEqual(await stats(),['206 / 512 题','508 / 512 题']);
    await wide.locator('[data-pick="2990"]').uncheck();
    assert.deepEqual(await stats(),['0 / 0 题','0 / 0 题']);
    assert.equal(await wide.locator('.case-card .primary').isDisabled(),true);
    await view('catalog');await wide.locator('#lang').selectOption('中文');await act('study',322);
    assert.match(await wide.locator('#dialog-body').innerText(),/31 \/ 32/);await checkLinks();await page.keyboard.press('Escape');
    await wide.locator('#lang').selectOption('全部');await wide.locator('#search').fill('草莓');await act('study',3008);
    await act('color-cell','red/red');await wide.locator('#color-arm').selectOption('cut');
    assert.equal(await wide.locator('dialog:visible tr').count(),37);await checkLinks();await page.keyboard.press('Escape');
    await act('tree');
    const leaf=wide.locator('[data-act="leaf"][data-value="SEM.ATTRIBUTE"]');
    await leaf.evaluate(el=>{for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});
    await leaf.click();await wide.locator('dialog [data-act="study"][data-value="3008"]').click();
    assert.match(await wide.locator('#dialog-body').innerText(),/草莓/);await page.keyboard.press('Escape');
    await view('storage');await wide.getByText('展开查重、运行与八份文件',{exact:true}).click();
    assert.equal(await wide.locator('details[open] tr').count(),9);await checkLinks();
    await view('compare');await wide.locator('[data-pick="2990"]').check();
    await page.reload();await ready();assert.deepEqual(await stats(),['206 / 512 题','508 / 512 题']);
    await act('ledger');await page.goBack();await ready();assert.deepEqual(await stats(),['206 / 512 题','508 / 512 题']);
    await page.goto('about:blank');await page.goto(url+analytic.replace('2983%2C2990','2983').replace('rule=2','rule=0'));await ready();
    assert.equal(await wide.locator('[data-act="rule"][aria-pressed="true"]').innerText(),'按姓名');
    assert.equal(await wide.locator('[data-pick="2990"]').isChecked(),false);
    await page.setViewportSize({width:390,height:844});await page.goto('about:blank');await page.goto(url);await ready();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    const bounds=await wide.locator('.chartcard').boundingBox();assert.ok(bounds.x>=0&&bounds.x+bounds.width<=391);
    await shot('mobile');await act('browse','div');await act('case');await page.keyboard.press('Escape');assert.equal(await wide.locator('dialog').isVisible(),false);
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const report={passed:true,browser:browser.version(),sizes:[[1440,920],[390,844]],ruleCount,sourceLinks:links.size,
      entryPoints:['default','client hash','comparison hash','edited comparison hash'],pooled:{questions:1024,families:256,head:377,model:1018,disagreements:642},
      outcomes:{both:376,headOnly:1,div:642,neither:5},reduced:exported.total,sceneRoundTrips:['B26/H36','B28/H21','U17216'],archiveFiles:8,network,errors};
    fs.writeFileSync(path.join(output,'acceptance.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
  }finally{await browser.close();}
}
main().catch(err=>{console.error(err);process.exitCode=1;});
