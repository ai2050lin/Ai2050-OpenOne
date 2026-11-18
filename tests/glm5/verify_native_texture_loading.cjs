const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const out=path.resolve(__dirname,process.env.AI2050_ACCEPTANCE_OUT||'result/native_texture_loading_20261109');fs.mkdirSync(out,{recursive:true});
const report={time:new Date().toISOString(),checks:[],errors:[],gpuRequests:[]};
const base='http://localhost:5173',snapshot='6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad';
const url=view=>`${base}/workbench?lens=spatial&view=${view}&gview=structure&snapshot=${snapshot}&sblock=35&smodule=mlp&sindex=2663`;
function ok(name,detail){report.checks.push({name,detail});console.log(JSON.stringify({step:name,detail}));}
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-sandbox']});
 let page;
 try{
  page=await browser.newPage({viewport:{width:1440,height:1000}});page.setDefaultTimeout(35000);
  page.on('pageerror',e=>report.errors.push(e.message));
  page.on('request',r=>{if(r.method()==='POST'&&/\/api\/live\/(run|load)/.test(r.url()))report.gpuRequests.push(r.url());});
  await page.addInitScript(()=>{window.textureAudit={valid:0,invalid:[]};const original=CanvasRenderingContext2D.prototype.createImageData;
    CanvasRenderingContext2D.prototype.createImageData=function(w,h,...rest){if(typeof w==='number'){if(!Number.isSafeInteger(w)||!Number.isSafeInteger(h)||w<=0||h<=0)window.textureAudit.invalid.push([String(w),String(h)]);else window.textureAudit.valid++;}return original.call(this,w,h,...rest);};});
  for(const view of ['live','graph','parameters']){
    let release;const gate=new Promise(resolve=>release=resolve);
    await page.route('**/api/model/skeleton',async route=>{await gate;await route.continue();});
    const sample=page.waitForResponse(r=>r.url().includes('/api/traces/mainline/ma-h2/sample/0')&&r.ok());
    await page.goto(url(view));
    await page.getByText('正在读取模型结构尺寸；尺寸就绪后显示单元网格。',{exact:true}).waitFor();
    await sample;assert.equal(await page.getByText('⚠️ Something went wrong.',{exact:true}).count(),0);
    assert.deepEqual(await page.evaluate(()=>window.textureAudit.invalid),[]);
    release();await page.getByRole('heading',{name:'B34 · U2663',exact:true}).waitFor();
    if(view==='live')await page.locator('.aw-cin').getByText(/现在显示的是冻结样本 #0/).waitFor();
    await page.waitForFunction(()=>window.textureAudit.valid>0);
    assert.deepEqual(await page.evaluate(()=>window.textureAudit.invalid),[]);
    ok(`response_before_model_${view}`,await page.evaluate(()=>window.textureAudit));
    await page.unroute('**/api/model/skeleton');
  }
  for(const invalid of [0,'not-ready']){
    await page.route('**/api/model/skeleton',async route=>{const response=await route.fetch();const data=await response.json();data.models=data.models.map(m=>({...m,d_model:invalid,n_heads:invalid,d_ff:invalid}));await route.fulfill({response,json:data});});
    const sample=page.waitForResponse(r=>r.url().includes('/api/traces/mainline/ma-h2/sample/0')&&r.ok());
    await page.goto(url('live'));await sample;
    await page.getByText('正在读取模型结构尺寸；尺寸就绪后显示单元网格。',{exact:true}).waitFor();
    assert.equal(await page.getByText('⚠️ Something went wrong.',{exact:true}).count(),0);
    assert.deepEqual(await page.evaluate(()=>window.textureAudit.invalid),[]);ok(`invalid_metadata_${invalid}`);
    await page.unroute('**/api/model/skeleton');
  }
  await page.goto(url('live'));await page.locator('.aw-cin').getByText(/现在显示的是冻结样本 #0/).waitFor();
  await page.getByLabel('统一对象详情',{exact:true}).getByRole('button',{name:'参数与响应',exact:true}).click();
  await page.getByText(/MLP · 块 34 · 单元 2663 · A -0.0654/).waitFor();ok('real_response_recovers_after_metadata_restored');
  const helper=await page.evaluate(async()=>{const {nativeTexture}=await import('/src/components/v8/nativeTexture.js');const before=window.textureAudit.valid;const empty=[0,NaN,undefined,-1,Infinity,'x',.2].map(h=>nativeTexture(8,h,()=>{throw Error('invalid fill called');}));const valid=nativeTexture(2,2,p=>{p[0]=91;p[3]=255;});return {allAbsent:empty.every(x=>x===null),newAllocations:window.textureAudit.valid-before,pixel:valid.getContext('2d').getImageData(0,0,1,1).data[0],invalid:window.textureAudit.invalid};});
  assert.deepEqual(helper,{allAbsent:true,newAllocations:1,pixel:91,invalid:[]});ok('allocation_guard_in_real_browser',helper);
  const code=await (await page.request.get(`${base}/src/components/v8/AssocWorkbench.jsx?t=1794210849065`)).text();assert.ok(code.includes('nativeTexture'));ok('old_module_url_serves_current_fix');
  // Fault injection exercises the existing reload recovery; no source or research result is changed.
  await page.addInitScript(()=>{const shouldFail=!sessionStorage.getItem('texture-recovery-injected');const original=CanvasRenderingContext2D.prototype.createImageData;CanvasRenderingContext2D.prototype.createImageData=function(...args){if(shouldFail){sessionStorage.setItem('texture-recovery-injected','1');throw new DOMException('Isolated texture recovery test','IndexSizeError');}return original.apply(this,args);};});
  await page.goto(url('live'));await page.getByRole('button',{name:'Reload Application',exact:true}).waitFor();
  await page.getByRole('button',{name:'Reload Application',exact:true}).click();
  await page.locator('.aw-cin').getByText(/现在显示的是冻结样本 #0/).waitFor();
  assert.equal(await page.getByRole('button',{name:'Reload Application',exact:true}).count(),0);ok('error_boundary_reload_recovers');
  await page.locator('.aw-space').screenshot({path:path.join(out,'recovered-space.png')});
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.gpuRequests,[]);report.passed=true;
 }catch(error){report.passed=false;report.failure=error.stack;console.error(error.stack);process.exitCode=1;if(page){await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});fs.writeFileSync(path.join(out,'failure.txt'),await page.locator('body').innerText());}}
 finally{await browser.close();fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2));}
})();
