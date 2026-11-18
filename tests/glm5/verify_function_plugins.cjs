const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const temp=path.resolve('tests/glm5_temp/function_plugins_20261112');fs.mkdirSync(temp,{recursive:true});
process.env.TEMP=temp;process.env.TMP=temp;
const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const out=path.resolve('tests/glm5/result/function_plugins_20261112');fs.mkdirSync(out,{recursive:true});
const report={checks:[],errors:[],gpuRequests:[]};
const ok=s=>{report.checks.push(s);console.log(s);};
(async()=>{const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const page=await browser.newPage({viewport:{width:1440,height:1000}});page.setDefaultTimeout(45000);
page.on('pageerror',e=>report.errors.push(e.message));page.on('request',r=>{if(r.method()==='POST'&&/\/api\/live\/(load|run)/.test(r.url()))report.gpuRequests.push(r.url());});
try{
const root='http://localhost:5173/workbench?',snapshot='6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad';
await page.goto(root+new URLSearchParams({lens:'spatial',view:'graph',gview:'structure',snapshot,sblock:'35',smodule:'mlp',sindex:'2663'}));
const ops=page.getByLabel('操作面板',{exact:true}),func=ops.getByRole('group',{name:'功能插件',exact:true});
const activate=name=>func.getByRole('button',{name:new RegExp('^'+name)}).click();
await page.getByRole('heading',{name:'B34 · U2663',exact:true}).waitFor();
assert.equal(await func.getByRole('button').count(),3);assert.equal(await page.getByRole('group',{name:'观察方式',exact:true}).count(),0);ok('three_function_plugins_replace_top_modes');
assert.deepEqual(await ops.locator('.aw-pl > div:first-child > b').allTextContents(),['HiddenState','MLP 神经元','注意力头','Logit Lens']);ok('four_default_analysis_plugins_retained');
await page.getByLabel('运行文本',{exact:true}).fill('功能插件之间保留输入，不执行模型。');
for(const [name,tab] of [['运行响应','运行与对照'],['条件对照','运行与对照'],['结构与档案','对象详情']]){
 await activate(name);assert.equal(await func.getByRole('button',{name:new RegExp('^'+name)}).getAttribute('aria-pressed'),'true');
 assert.equal(await page.getByRole('tab',{name:tab,exact:true}).getAttribute('aria-selected'),'true');
 const q=new URL(page.url()).searchParams;assert.equal(q.get('sindex'),'2663');assert.equal(q.get('sblock'),'35');
 assert.equal(await page.getByLabel('运行文本',{exact:true}).inputValue(),'功能插件之间保留输入，不执行模型。');
 if(name==='条件对照')await page.getByLabel('条件对照样本',{exact:true}).waitFor();
 ok(name+'_activates_real_workspace_preserves_object_and_input');
}
await ops.getByRole('button',{name:'管理插件',exact:true}).click();await ops.getByText('基础功能 · 常驻',{exact:true}).waitFor();await ops.getByRole('button',{name:'恢复默认',exact:true}).click();assert.equal(await func.getByRole('button').count(),3);await ops.getByRole('button',{name:'完成',exact:true}).click();ok('reset_analysis_does_not_remove_core_functions');
await ops.getByRole('button',{name:'查看Logit Lens插件',exact:true}).click();assert.equal(await page.getByRole('tab',{name:'插件分析',exact:true}).getAttribute('aria-selected'),'true');await activate('结构与档案');await page.getByRole('heading',{name:'B34 · U2663',exact:true}).waitFor();ok('analysis_and_structure_share_object');
await ops.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'desktop.png')});
await activate('条件对照');await page.reload();await func.getByRole('button',{name:/^条件对照/}).waitFor();assert.equal(await func.getByRole('button',{name:/^条件对照/}).getAttribute('aria-pressed'),'true');assert.equal(await page.getByRole('tab',{name:'运行与对照',exact:true}).getAttribute('aria-selected'),'true');ok('old_response_route_and_refresh_restore_mode');
await page.setViewportSize({width:390,height:844});await ops.scrollIntoViewIfNeeded();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.screenshot({path:path.join(out,'mobile.png')});ok('mobile_no_horizontal_overflow');
assert.deepEqual(report.errors,[]);assert.deepEqual(report.gpuRequests,[]);report.passed=true;
}catch(e){report.failure=e.stack;process.exitCode=1;await page.screenshot({path:path.join(out,'failure.png')});console.error(e);}finally{await browser.close();fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2));}})();
