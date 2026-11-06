/* Launch the actual Electron main/preload with an isolated profile and real idle node. */
import {createRequire} from 'node:module';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {_electron:electron}=require(process.env.AI2050_PLAYWRIGHT||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root=process.cwd(),out=path.join(root,'tests/glm5/result/workbench_playwright_20261104'),temp=path.join(root,'tests/glm5_temp/workbench-electron-acceptance');
await fs.mkdir(temp,{recursive:true});await fs.mkdir(out,{recursive:true});
const profile=process.env.AI2050_REUSE_TEST_PROFILE||await fs.mkdtemp(path.join(temp,'profile-')),home=await fs.mkdtemp(path.join(temp,'node-'));
const entry=path.join(temp,'entry.cjs');
await fs.writeFile(entry,`const {app}=require('electron');app.setName('AI2050 acceptance');app.setPath('userData',${JSON.stringify(profile)});require(${JSON.stringify(path.join(root,'frontend/electron/main.js'))});`);
const env={...process.env,AI2050_CLIENT_SERVER:'http://127.0.0.1:5001',AI2050_PYTHON:path.join(root,'.venv/Scripts/python.exe'),AI2050_NODE_HOME:home,TEMP:path.join(root,'tests/glm5_temp'),TMP:path.join(root,'tests/glm5_temp')};
delete env.ELECTRON_RUN_AS_NODE;delete env.AI2050_NODE_SYNTHETIC;delete env.AI2050_CLIENT_DEV_URL;
const executablePath=process.env.AI2050_ELECTRON_EXE||path.join(root,'frontend/node_modules/electron/dist/electron.exe');
const app=await electron.launch({executablePath,args:[entry],env,timeout:90000});
const errors=[],checks=[];let failure;
try{
 const page=await app.firstWindow({timeout:90000});page.setDefaultTimeout(60000);page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(path.join(root,'frontend/dist/index.html')).href+'?wb=1&lens=config&csec=node');
 await page.getByRole('navigation',{name:'配置分段',exact:true}).waitFor();
 await page.getByRole('button',{name:'请先绑定节点',exact:true}).waitFor({timeout:90000});assert(await page.getByRole('button',{name:'请先绑定节点',exact:true}).isDisabled());
 const status=await page.evaluate(async()=>{const r=await window.ai2050Desktop.node.request('GET','/status');return {http:r.status,state:r.data.state,bound:r.data.bound,synthetic:r.data.synthetic}});
 assert.equal(status.http,200);assert.equal(status.state,'stopped');assert.equal(status.bound,false);
 checks.push({name:'actual_electron_preload_and_idle_node',passed:true,guard:'unbound start disabled',synthetic:false});
 await page.getByText('参与方式与资源预算',{exact:true}).click();
 // The disclosure may initially be open for unbound nodes. Ensure its controls are visible.
 if(!await page.getByLabel('允许的时间段',{exact:true}).isVisible())await page.getByText('参与方式与资源预算',{exact:true}).click();
 await page.getByLabel('允许的时间段',{exact:true}).fill('22:00-08:00');await page.getByRole('button',{name:'保存',exact:true}).click();
 await page.getByText('已保存；托管授权在本机点击开始或继续后生效',{exact:true}).waitFor();
 const saved=await page.evaluate(async()=>{const r=await window.ai2050Desktop.node.request('GET','/config');return {http:r.status,hours:r.data.limits.hours,mode:r.data.participation_mode}});
 assert.equal(saved.hours,'22:00-08:00');assert.equal(saved.mode,'self');checks.push({name:'resource_budget_saved_via_real_ipc',passed:true,remote_authority_granted:false});
 await page.screenshot({path:path.join(out,'14-desktop-real-node.png')});
 await page.getByRole('button',{name:'数据与同步',exact:true}).click();const dialog=page.getByRole('dialog',{name:'数据与同步',exact:true});await dialog.waitFor();
 await page.getByRole('button',{name:'下载或更新主数据包',exact:true}).click();await page.getByText('数据包已更新，相关页面将读取新版本。',{exact:true}).waitFor();
 checks.push({name:'packaged_ui_primary_pack_real_download',passed:true});
 if(process.env.AI2050_CHECK_FULL_EXTRA==='1'&&!process.env.AI2050_REUSE_TEST_PROFILE){
   const extra=dialog.locator('.op-extra');await extra.getByRole('button',{name:'下载',exact:true}).waitFor();
   await extra.getByRole('button',{name:'下载',exact:true}).click();
   const deadline=Date.now()+600000;let committed=false;
   while(Date.now()<deadline){
     committed=await page.evaluate(()=>new Promise(resolve=>{const r=indexedDB.open('ai2050-data',1);r.onsuccess=()=>{const db=r.result,q=db.transaction('kv').objectStore('kv').get('xmeta');q.onsuccess=()=>{const yes=Boolean(q.result?.['research-library']?.key_prefix);db.close();resolve(yes)}}}));
     if(committed)break;
     assert(await dialog.isVisible(),'Data dialog must stay open while downloading');
     const alert=await extra.locator('[role=alert]').allTextContents();assert.deepEqual(alert,[]);
     await new Promise(r=>setTimeout(r,500));
   }
   assert(committed,'Research stream must commit real IndexedDB metadata');
   checks.push({name:'complete_research_library_stream_real_download',passed:true});
   if(!await dialog.isVisible())await page.getByRole('button',{name:'数据与同步',exact:true}).click();
 }
 console.log('acceptance-cache '+JSON.stringify(await page.evaluate(()=>new Promise(resolve=>{const r=indexedDB.open('ai2050-data',1);r.onsuccess=()=>{const db=r.result,q=db.transaction('kv').objectStore('kv').get('xmeta');q.onsuccess=()=>{const m=q.result?.['research-library'];if(!m){db.close();resolve({exists:false});return}const p=db.transaction('kv').objectStore('kv').get(m.key_prefix+'/api/traces/research-space');p.onsuccess=()=>{db.close();resolve({meta:m,index:!!p.result,counts:p.result?.counts})}}}}))));
 await page.screenshot({path:path.join(out,'15-desktop-data-sync.png')});await page.keyboard.press('Escape');
 await page.getByRole('navigation',{name:'工作台界面',exact:true}).getByRole('button',{name:'空间',exact:true}).click();
 await page.getByRole('button',{name:'内部特征与研究证据',exact:true}).click();
 if(process.env.AI2050_CHECK_FULL_EXTRA==='1'){
   await page.getByLabel('成果图研究工作包',{exact:true}).waitFor({timeout:90000});checks.push({name:'research_graph_reads_actual_downloaded_extra',passed:true});
   await page.getByLabel('成果图研究工作包',{exact:true}).selectOption('P2794-ATOMIC-ARBITRATION-v4');
   await page.getByText('已绑定真实原生轨迹',{exact:true}).waitFor();
   await page.screenshot({path:path.join(out,'16-desktop-offline-graph.png')});
   await page.reload();await page.getByLabel('成果图研究工作包',{exact:true}).waitFor({timeout:90000});
   assert.equal(await page.getByLabel('成果图研究工作包',{exact:true}).inputValue(),'P2794-ATOMIC-ARBITRATION-v4');
   checks.push({name:'downloaded_extra_and_graph_selection_survive_reload',passed:true});
 }
 assert.deepEqual(errors,[]);
}catch(e){failure=String(e.stack||e);const windows=app.windows();if(windows[0]){await windows[0].screenshot({path:path.join(out,'DESKTOP-FAILED.png')}).catch(()=>{});await fs.writeFile(path.join(out,'desktop-failed-body.txt'),await windows[0].locator('body').innerText().catch(()=>''));}console.error(failure)}
finally{await app.close();await fs.writeFile(path.join(out,process.env.AI2050_ELECTRON_REPORT||'electron-acceptance.json'),JSON.stringify({schema:'electron_visual_acceptance.v1',kind:'engineering',passed:!failure,actual_main_and_preload:true,isolated_profile:true,actual_node_service:true,model_inference_executed:false,checks,page_errors:errors,failure},null,2));}
console.log(JSON.stringify({desktop_passed:!failure,checks:checks.length}));if(failure)process.exitCode=1;
