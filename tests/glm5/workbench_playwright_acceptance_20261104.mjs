/* Real local Chromium acceptance. Research numbers come from the running service. */
import {createRequire} from 'node:module';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=process.cwd(),out=path.join(root,'tests/glm5/result/workbench_playwright_20261104');
await fs.mkdir(out,{recursive:true});
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.AI2050_PLAYWRIGHT||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const browser=await chromium.launch({headless:true,executablePath:'C:/Users/Admin/AppData/Local/ms-playwright/chromium-1247/chrome-win64/chrome.exe'});
const context=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1,acceptDownloads:true});
const page=await context.newPage();page.setDefaultTimeout(20000);
const errors=[],checks=[],screens=[];page.on('pageerror',e=>errors.push(String(e)));
const source=JSON.parse(await fs.readFile(path.join(root,'tests/glm5/result/phase2794_arbitration_v4/client-registration.json'),'utf8'));
const trace=new URL(source.trace_url).searchParams.get('trace'),run='P2794-ATOMIC-ARBITRATION-v4';
const base='http://127.0.0.1:5173',api='http://127.0.0.1:5001';
const get=async p=>{const r=await context.request.get(api+p,{timeout:90000});assert(r.ok(),p+' '+r.status());return r.json()};
const primary=()=>page.getByRole('navigation',{name:'工作台界面',exact:true});
const nav=name=>page.getByRole('navigation',{name,exact:true});
const button=name=>page.getByRole('button',{name,exact:true});
async function shot(name){const file=path.join(out,name+'.png');await page.screenshot({path:file,fullPage:false});screens.push({name,path:path.relative(root,file),url:page.url()});}
async function goto(query){await page.goto(base+'/workbench'+query,{waitUntil:'domcontentloaded'});await primary().waitFor();}
async function visible(text){await page.getByText(text,{exact:false}).first().waitFor({state:'visible'});}
async function check(name,fn){if(process.env.AI2050_CASE&&process.env.AI2050_CASE!==name)return;const t=Date.now();await fn();checks.push({name,passed:true,milliseconds:Date.now()-t});console.log(JSON.stringify({name,passed:true,seconds:((Date.now()-t)/1000).toFixed(1)}));}
let failure;
try{
 await check('navigation',async()=>{
   await goto('?lens=route');assert.deepEqual(await primary().getByRole('button').allTextContents(),['进展','空间','研发','配置']);
   await page.locator('.v8-overview-ratio').filter({hasText:/\d+ \/ [\d,]+ 个 ·/}).waitFor({timeout:90000});await shot('01-overview-ready');
   const pages=[['进展','进展视图',['总览','成果与证据']],['空间','空间视图',['图谱浏览','结构与参数','运行与路径','响应对照']],['研发','研发工位',['研究计划','材料与数据','实验与队列']],['配置','配置分段',['连接与诊断','账号与权限','本机算力','模型与AI']]];
   for(const [main,aria,labels] of pages){await primary().getByRole('button',{name:main,exact:true}).click();assert.deepEqual(await nav(aria).locator('button:not(.wb-help-btn)').allTextContents(),labels);for(const label of labels){await nav(aria).getByRole('button',{name:label,exact:true}).click();await page.waitForFunction(()=>document.querySelector('.wb-main')?.textContent.length>25);}}
   await shot('02-config');
 });
 await check('legacy-links',async()=>{
   const cases=[['?lens=spatial&view=research','view','graph','gview','internal'],['?lens=spatial&view=atlas','view','graph','gview','relations'],['?lens=spatial&view=neuron&block=34&unit=9727','view','parameters','ptab','neurons'],['?station=freeze','station','pipeline','pstep','freeze'],['?station=contribute','lens','config','csec','node'],['?station=ai','lens','config','csec','model'],['?lens=local','lens','config','csec','legacy']];
   for(const [url,k,v,k2,v2] of cases){await goto(url);await page.waitForFunction(([k,v,k2,v2])=>{const q=new URLSearchParams(location.search);return q.get(k)===v&&q.get(k2)===v2},[k,v,k2,v2]);}
   await page.getByRole('heading',{name:'旧版兼容与教学工具',exact:true}).waitFor();await shot('03-legacy-compatibility');
   await goto('?lens=spatial&view=local');await page.getByRole('dialog',{name:'数据与同步',exact:true}).waitFor();await page.keyboard.press('Escape');await page.getByRole('dialog',{name:'数据与同步',exact:true}).waitFor({state:'detached'});
 });
 await check('graph-runtime-parameters',async()=>{
   const m=await get(`/api/traces/${trace}/manifest`);
   await goto('?lens=spatial&view=research&run='+run+'&snapshot='+m.model_snapshot);
   await page.getByLabel('成果图研究工作包',{exact:true}).waitFor({timeout:90000});
   await page.getByLabel('成果图研究工作包',{exact:true}).selectOption(run);await visible('已绑定真实原生轨迹');
   const link=page.getByRole('link',{name:/原始输入、完整数组与参数路径/}).first();await link.waitFor();await shot('04-research-graph');await link.click();
   await page.getByLabel('轨迹块',{exact:true}).waitFor({timeout:90000});
   await page.getByLabel('轨迹块',{exact:true}).fill('34');await page.getByLabel('轨迹Head',{exact:true}).fill('10');await page.getByLabel('轨迹MLP单元',{exact:true}).fill('1995');
   await button('全部Head').click();assert.equal(await page.locator('.v8-head-grid button').count(),32);
   const pos=page.getByLabel('真实query位置',{exact:true}),options=await pos.locator('option').evaluateAll(o=>o.map(x=>x.value));
   const earlier=options[0],last=options.at(-1);
   const earlierResponse=page.waitForResponse(r=>r.url().includes('/attention-row?')&&new URL(r.url()).searchParams.get('query')===earlier&&r.ok());await pos.selectOption(earlier);const ea=await (await earlierResponse).json();assert.equal(ea.query_position,Number(earlier));
   await pos.selectOption(last);await button('完整MLP阵列').click();await page.locator('canvas[aria-label*="完整MLP中间单元阵列，全部9728维"]').waitFor();
   await shot('05-real-mlp-array');await button('参数与关联').click();await button('核验参数并显示计算关联').click();await visible('完整快照一致');
   await shot('06-real-parameter-contributions');const parameterLink=page.getByRole('link',{name:'打开同一模型的完整参数切片',exact:true});const href=await parameterLink.getAttribute('href');assert.equal(new URL(href,base).searchParams.get('punit'),'1995');
   await parameterLink.click();await page.getByLabel('参数单元',{exact:true}).waitFor({timeout:90000});assert.equal(await page.getByLabel('参数块',{exact:true}).inputValue(),'34');assert.equal(await page.getByLabel('参数单元',{exact:true}).inputValue(),'1995');await visible('当前原权重与研究来源的完整快照一致');await page.locator('.v8-weight-scroll td').first().waitFor();
   await shot('07-actual-weight-slices');await page.goBack();await page.getByLabel('轨迹MLP单元',{exact:true}).waitFor();assert.equal(await page.getByLabel('轨迹MLP单元',{exact:true}).inputValue(),'1995');assert.equal(await page.getByLabel('轨迹Head',{exact:true}).inputValue(),'10');
   await page.reload();await page.getByLabel('轨迹MLP单元',{exact:true}).waitFor();assert.equal(await page.getByLabel('轨迹MLP单元',{exact:true}).inputValue(),'1995');
 });
 await check('contribution-web',async()=>{
   await goto('?lens=route');await button('贡献算力').click();await visible('浏览器里的网页不能调用显卡');assert.equal(new URL(page.url()).searchParams.get('csec'),'node');
   await visible('本人账号的远程托管节点');await visible('登录后查看本人绑定节点');
   const pending=page.waitForEvent('download');await page.getByRole('button',{name:'下载命令行一键包（匿名）',exact:true}).click();const download=await pending;assert.equal(await download.failure(),null);await download.saveAs(path.join(out,'anonymous-node-kit.zip'));
   await visible('已下载 ai2050-node.zip');await shot('08-contribution-browser');
   await nav('配置分段').getByRole('button',{name:'账号与权限',exact:true}).click();assert.equal(await page.getByRole('heading',{name:'本人账号的远程托管节点',exact:true}).count(),0);
 });
 await check('data-sync',async()=>{
   await goto('?lens=route');await button('数据与同步').click();const dialog=page.getByRole('dialog',{name:'数据与同步',exact:true});await dialog.waitFor();
   await button('下载或更新主数据包').waitFor();await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='下载或更新主数据包'&&!b.disabled));
   const response=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/platform/datapack'&&r.ok());await button('下载或更新主数据包').click();assert((await response).ok());await visible('数据包已更新');
   const version=await page.evaluate(()=>new Promise((resolve,reject)=>{const open=indexedDB.open('ai2050-data');open.onerror=()=>reject(open.error);open.onsuccess=()=>{const db=open.result;const get=db.transaction('kv','readonly').objectStore('kv').get('pack');get.onerror=()=>reject(get.error);get.onsuccess=()=>{resolve(get.result?.version);db.close()}}}));assert(version);await shot('09-data-sync-success');
   await page.route('**/api/platform/datapack',route=>route.fulfill({status:503,contentType:'application/json',body:'{"detail":"acceptance injected offline response"}'}));await button('下载或更新主数据包').click();await page.waitForFunction(()=>[...document.querySelectorAll('[role=status]')].some(e=>e.textContent&& !e.textContent.includes('数据包已更新')&&!e.textContent.startsWith('下载 ')));assert(await button('下载或更新主数据包').isEnabled());await shot('10-data-sync-failure');await page.unroute('**/api/platform/datapack');
   await page.keyboard.press('Escape');await dialog.waitFor({state:'detached'});
   await page.reload();await button('数据与同步').click();await visible('本机版本：'+version);await page.keyboard.press('Escape');
 });
 await check('narrow-layout',async()=>{
   await page.setViewportSize({width:390,height:844});await goto('?lens=route');await page.locator('.v8-library-stats').first().waitFor({timeout:90000});await shot('11-mobile-overview');
   const width=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,viewport:innerWidth}));assert(width.scroll<=width.viewport+2,JSON.stringify(width));
   await button('贡献算力').click();await visible('浏览器里的网页不能调用显卡');await shot('12-mobile-contribution');
   await button('数据与同步').click();await page.getByRole('dialog',{name:'数据与同步'}).waitFor();await shot('13-mobile-sync');
   assert(await page.getByRole('button',{name:'关闭数据面板',exact:true}).isVisible());await page.keyboard.press('Escape');await page.setViewportSize({width:1440,height:1000});
 });
 assert.deepEqual(errors,[],'browser page errors');
}catch(e){failure=String(e.stack||e);await shot('FAILED').catch(()=>{});await fs.writeFile(path.join(out,'failed-body.txt'),await page.locator('body').innerText().catch(()=>''));console.error(failure)}
finally{
 await fs.writeFile(path.join(out,process.env.AI2050_CASE?'case-'+process.env.AI2050_CASE+'.json':'browser-acceptance.json'),JSON.stringify({schema:'browser_visual_acceptance.v1',kind:'engineering',passed:!failure,actual_browser:'local Playwright Chromium',mocked_research_data:false,cases:checks,page_errors:errors,screenshots:screens,failure},null,2));
 await browser.close();
}
if(failure)process.exitCode=1;
