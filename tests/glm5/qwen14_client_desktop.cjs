const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'tests/glm5/result/qwen14_client_v2');
const dir=path.join(out,'desktop/win-unpacked'),archive=path.join(dir,'resources/app.asar');
const asar=require(path.join(root,'frontend/node_modules/@electron/asar'));
const {_electron}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const walk=d=>fs.readdirSync(d,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(d,e.name)):[path.join(d,e.name)]);
const report={checks:[],errors:[],gpuRequests:[],installationWizardRun:false};
const ok=s=>{report.checks.push(s);console.log(s)};
(async()=>{
 let app;
 try{
  const pkg=JSON.parse(asar.extractFile(archive,'package.json'));assert.equal(pkg.version,'1.0.14');let count=0;
  for(const folder of ['dist','electron'])for(const p of walk(path.join(root,'frontend',folder))){
   const rel=path.relative(path.join(root,'frontend'),p).replaceAll('\\','/');if(/\.test\.[cm]?js$/.test(rel))continue;
   assert.equal(sha(asar.extractFile(archive,path.normalize(rel))),sha(fs.readFileSync(p)),rel);count++;
  }
  const resources=JSON.parse(fs.readFileSync(path.join(root,'frontend/package.json'))).build.extraResources;
  for(const r of resources)assert.equal(sha(fs.readFileSync(path.resolve(root,'frontend',r.from))),sha(fs.readFileSync(path.join(dir,'resources',r.to))));
  report.payloadFiles=count;report.payloadSha256=sha(fs.readFileSync(archive));ok('packaged_renderer_main_process_and_node_resources_match_sources');
  const storage=path.join(root,'tests/glm5_temp/qwen14-client-electron'),profile=path.join(storage,'client-cache/profile');fs.mkdirSync(profile,{recursive:true});
  fs.writeFileSync(path.join(profile,'config.json'),JSON.stringify({server:'http://127.0.0.1:5034',verified:true}));
  const env={...process.env,AI2050_STORAGE_ROOT:storage,AI2050_CLIENT_SERVER:'http://127.0.0.1:5034',TEMP:storage,TMP:storage,PATH:'C:\\Windows\\System32'};
  delete env.ELECTRON_RUN_AS_NODE;delete env.AI2050_PYTHON;delete env.AI2050_CLIENT_DEV_URL;
  console.log('Launching isolated packaged Electron client');
  app=await _electron.launch({executablePath:path.join(dir,'AI2050 语言机制研究平台.exe'),args:['--headless'],env,timeout:60000});
  await app.evaluate(({BrowserWindow})=>BrowserWindow.getAllWindows().forEach(w=>w.hide()));
  const identity=await app.evaluate(({app})=>({version:app.getVersion(),packaged:app.isPackaged,userData:app.getPath('userData')}));
  assert.equal(identity.version,'1.0.14');assert.equal(identity.packaged,true);assert.ok(identity.userData.startsWith(storage));report.identity=identity;
  const page=await app.firstWindow();page.setDefaultTimeout(60000);page.on('pageerror',e=>report.errors.push(e.message));
  page.on('request',r=>{if(r.method()==='POST'&&/\/api\/live\/(load|run)/.test(r.url()))report.gpuRequests.push(r.url())});
  const base=page.url().split('?')[0];await page.goto(base+'?'+new URLSearchParams({wb:'1',lens:'spatial',view:'graph',gview:'structure',snapshot:'1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce',run:'QWEN14-GOODFIRE-B28-B39-20261113-v2',sblock:'40',smodule:'mlp',sindex:'17407'}));
  await page.getByRole('tab',{name:'对象详情',exact:true}).click();
  const atlas=page.locator('section[aria-label="Qwen3-14B全量MLP研究"]:visible');
  await atlas.getByText('B39/U17407',{exact:true}).waitFor();assert.equal(await atlas.getByLabel('14B研究层').locator('option').count(),12);ok('packaged_client_reads_B39_last_unit_and_all_layer_options');
  await atlas.getByRole('tab',{name:'完整参数',exact:true}).click();await atlas.getByLabel('14B参数坐标').fill('5119');assert.equal(await atlas.locator('output').textContent(),'0.03759765625');ok('packaged_client_reads_native_coordinate');
  await atlas.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'desktop-packaged.png')});
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.gpuRequests,[]);report.passed=true;
 }catch(e){report.failure=e.stack;process.exitCode=1;console.error(e)}
 finally{if(app)await app.close();fs.writeFileSync(path.join(out,'desktop-acceptance.json'),JSON.stringify(report,null,2));}
})();
