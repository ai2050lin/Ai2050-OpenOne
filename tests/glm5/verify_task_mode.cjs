const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const temp=path.resolve('tests/glm5_temp/task_mode_20261117');process.env.TEMP=temp;process.env.TMP=temp;
const {chromium}=require('C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const out=path.resolve('tests/glm5/result/task_mode_20261117');
(async()=>{const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const page=await browser.newPage({viewport:{width:1440,height:1000}});const report={scope:'built client, isolated unavailable-service path; no GPU execution',checks:[],errors:[],writes:[]};
page.on('pageerror',e=>report.errors.push(e.message));page.on('request',r=>{if(r.method()==='POST')report.writes.push(r.url());});
// No real service mutations or background model work; test missing prerequisites.
await page.route('**/api/**',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'浏览器验收：本机服务未连接'})}));
try{await page.goto('http://127.0.0.1:5175/workbench?lens=spatial&view=graph&gview=structure');await page.getByRole('button',{name:'任务模式',exact:true}).click();
const task=page.getByRole('region',{name:'自动文本任务'});await task.waitFor();assert.equal(await task.getByRole('button',{name:'运行并分析',exact:true}).count(),2);assert.equal(await task.getByRole('button',{name:'运行并分析',exact:true}).first().isDisabled(),true);report.checks.push('task_mode_visible_and_unavailable_execution_blocked');
await task.getByRole('button',{name:'查看文本',exact:true}).first().click();const input=page.getByLabel('运行文本',{exact:true});assert.ok((await input.inputValue()).length>300);assert.equal(await input.getAttribute('readonly'),'');report.checks.push('long_generated_text_preview_is_readonly');
await task.getByLabel('观察任务类型').selectOption('negation');assert.match(await input.inputValue(),/不是/);await task.getByLabel('观察任务类型').selectOption('order');await task.getByRole('button',{name:'生成另一组文本'}).click();assert.match(await input.inputValue(),/陶杯/);report.checks.push('three_tasks_and_variant_generation');
await task.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'desktop.png')});
await page.getByRole('button',{name:'自由输入',exact:true}).click();assert.equal(await task.isVisible(),false);await page.getByRole('button',{name:'任务模式',exact:true}).click();assert.equal(await task.getByLabel('观察任务类型').inputValue(),'order');report.checks.push('mode_switch_preserves_task');
await page.setViewportSize({width:390,height:844});await task.scrollIntoViewIfNeeded();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.screenshot({path:path.join(out,'mobile.png')});report.checks.push('mobile_no_overflow');assert.deepEqual(report.errors,[]);assert.deepEqual(report.writes,[]);report.passed=true;
}catch(e){report.failure=e.stack;process.exitCode=1;await page.screenshot({path:path.join(out,'failure.png')});console.error(e);}finally{fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(report,null,2));await browser.close();}})();
