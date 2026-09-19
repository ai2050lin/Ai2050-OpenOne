/* Local development regression, isolated headless browser; no user browser/session access. */
const {chromium}=require('C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const root='D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_continuity_campaign_20260909';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1500,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto('http://localhost:5173/rdc');await page.getByLabel('实验范围').selectOption('e_confirmation');
  await page.getByLabel('真实样本').selectOption('e-taxonomy_chain-0-0-0-en');
  await page.getByTestId('field-coverage').filter({hasText:'native_bfloat16'}).waitFor();
  await page.getByRole('button',{name:'自然 / 同形全坐标',exact:true}).click();
  await page.getByTestId('continuity-value').filter({hasText:'最大数值差'}).waitFor();
  await page.screenshot({path:root+'/client_equalshape.png',fullPage:true});
  await page.getByRole('button',{name:'新材料单参数路径',exact:true}).click();
  await page.getByLabel('MLP单元').fill('9727');await page.getByLabel('链输入坐标').fill('2559');await page.getByLabel('链写回坐标').fill('2559');
  await page.getByTestId('scalar-chain').filter({hasText:'"output_coordinate": 2559'}).waitFor();
  await page.screenshot({path:root+'/client_scalar_chain.png',fullPage:true});
  const scalar=await page.getByTestId('scalar-chain').innerText();
  await page.getByRole('button',{name:'H12 → H24 预测',exact:true}).click();
  await page.getByTestId('continuity-value').filter({hasText:'预测MSE'}).waitFor();
  assert.match(await page.getByLabel('真实样本').inputValue(),/-12-/);
  await page.getByLabel('原生坐标起点').fill('2559');
  await page.getByTestId('field-coverage').filter({hasText:'3 / 7,680'}).waitFor();
  await page.screenshot({path:root+'/client_heldout_forecast.png',fullPage:true});
  await page.getByRole('button',{name:'全单元条件门',exact:true}).click();
  await page.getByLabel('原生坐标起点').fill('9727');
  await page.getByTestId('field-coverage').filter({hasText:'4 / 38,912'}).waitFor();
  await page.getByLabel('实验范围').selectOption('g_generation');
  await page.getByLabel('真实样本').selectOption('g-e-taxonomy_chain-0-0-0-en-s0');
  await page.getByLabel('生成步').selectOption('g-e-taxonomy_chain-0-0-0-en-s2');
  await page.getByRole('button',{name:'原生输出账本',exact:true}).click();
  await page.getByTestId('mechanism-ledger').filter({hasText:'151645'}).waitFor();
  await page.screenshot({path:root+'/client_current_eos_ledger.png',fullPage:true});
  await page.getByRole('button',{name:'输出MLP单元',exact:true}).click();
  await page.getByLabel('原生坐标起点').fill('9727');
  await page.getByTestId('field-coverage').filter({hasText:'3 / 29,184'}).waitFor();
  await page.getByRole('button',{name:'全部历史来源',exact:true}).click();
  await page.getByTestId('mechanism-ledger').waitFor();
  await page.getByRole('button',{name:'后16来源 →',exact:true}).click();
  await page.getByLabel('起始token').evaluate(e=>{if(e.value!=='16')throw new Error('source paging failed')});
  await page.screenshot({path:root+'/client_all_source_paging.png',fullPage:true});
  for(const [run,width,units] of (process.argv.includes('--without-scale')?[]:[['scale_qwen14',5120,17408],['scale_glm4',4096,13696]])){
   await page.getByLabel('实验范围').selectOption(run);
   await page.getByLabel('真实样本').selectOption('e-taxonomy_chain-0-0-0-en');
   await page.getByLabel('起始层').fill('40');
   await page.getByLabel('原生坐标起点').fill(String(width-1));
   await page.getByTestId('field-coverage').filter({hasText:'native_bfloat16'}).waitFor();
   await page.screenshot({path:root+'/client_'+run+'_last_layer.png',fullPage:true});
   await page.getByLabel('起始层').fill('39');
   await page.getByLabel('坐标域').selectOption('a');
   await page.getByLabel('原生坐标起点').fill(String(units-1));
   await page.getByTestId('field-coverage').filter({hasText:'1 / '+units.toLocaleString()}).waitFor();
   await page.getByText('真实参数 / 提取器坐标交互账本',{exact:true}).click();
   await page.getByLabel('参数组件').selectOption('up');
   await page.getByLabel('参数行').fill(String(units-1));
   await page.getByLabel('参数列').fill(String(width-1));
   await page.getByRole('button',{name:'读取真实权重',exact:true}).click();
   await page.getByTestId('native-parameter').filter({hasText:run==='scale_glm4'?'gate_up_proj':'up_proj'}).waitFor();
   await page.screenshot({path:root+'/client_'+run+'_last_unit.png',fullPage:true});
  }
  assert.equal(errors.length,0,errors.join('\n'));
  const report={timestamp:new Date().toISOString(),passed:true,scope:process.argv.includes('--without-scale')?'E/F/G only':'E/F/G/Q14/GLM4',errors,finalCoverage:await page.getByTestId('field-coverage').innerText(),scalar};
  fs.writeFileSync(root+'/client_browser_audit'+(process.argv.includes('--without-scale')?'_partial':'')+'.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify(report));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
