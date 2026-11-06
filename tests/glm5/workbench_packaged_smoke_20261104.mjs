/* Launch the shipped unpacked executable with isolated Windows app data. No installation. */
import {createRequire} from 'node:module';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{_electron}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root=process.cwd(),out=path.join(root,'tests/glm5/result/workbench_playwright_20261104');
const base=await fs.mkdtemp(path.join(root,'tests/glm5_temp/packaged-smoke-'));
for(const p of ['Roaming','Local','node'])await fs.mkdir(path.join(base,p));
const pkg=JSON.parse(await fs.readFile(path.join(root,'frontend/package.json'),'utf8'));
const env={...process.env,APPDATA:path.join(base,'Roaming'),LOCALAPPDATA:path.join(base,'Local'),AI2050_NODE_HOME:path.join(base,'node'),AI2050_CLIENT_SERVER:'http://127.0.0.1:5001',AI2050_PYTHON:path.join(root,'.venv/Scripts/python.exe'),TEMP:path.join(root,'tests/glm5_temp'),TMP:path.join(root,'tests/glm5_temp')};
for(const k of ['ELECTRON_RUN_AS_NODE','AI2050_NODE_SYNTHETIC','AI2050_CLIENT_DEV_URL'])delete env[k];
const app=await _electron.launch({executablePath:path.join(root,'frontend/dist-installer',pkg.version,'win-unpacked',pkg.productName+'.exe'),args:['--user-data-dir='+path.join(base,'profile')],env,timeout:90000});
const errors=[];let failure,identity,status;
try{
 identity=await app.evaluate(({app})=>({version:app.getVersion(),packaged:app.isPackaged,userData:app.getPath('userData'),appPath:app.getAppPath()}));
 assert.equal(identity.version,pkg.version);assert.equal(identity.packaged,true);assert(identity.userData.toLowerCase().startsWith(base.toLowerCase()+path.sep));
 const page=await app.firstWindow({timeout:90000});page.setDefaultTimeout(90000);page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(page.url().split('?')[0]+'?wb=1&lens=config&csec=node');
 await page.getByRole('button',{name:'请先绑定节点',exact:true}).waitFor();assert(await page.getByRole('button',{name:'请先绑定节点',exact:true}).isDisabled());
 status=await page.evaluate(async()=>{const r=await window.ai2050Desktop.node.request('GET','/status');return {http:r.status,state:r.data.state,bound:r.data.bound}});
 assert.equal(status.http,200);assert.equal(status.bound,false);assert.equal(status.state,'stopped');
 await page.getByRole('button',{name:'数据与同步',exact:true}).click();await page.getByRole('dialog',{name:'数据与同步',exact:true}).waitFor();
 await page.screenshot({path:path.join(out,'17-packaged-client.png')});assert.deepEqual(errors,[]);
}catch(e){failure=String(e.stack||e);console.error(failure)}finally{await app.close();await fs.writeFile(path.join(out,'packaged-smoke.json'),JSON.stringify({schema:'packaged_smoke.v1',kind:'engineering',passed:!failure,identity:identity&&{version:identity.version,packaged:identity.packaged,asar:identity.appPath.endsWith('app.asar'),isolated_profile:true},node:status,page_errors:errors,installer_executed:false,failure},null,2));}
console.log(JSON.stringify({passed:!failure,version:identity?.version}));if(failure)process.exitCode=1;
