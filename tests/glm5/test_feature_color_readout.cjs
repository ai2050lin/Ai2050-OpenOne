'use strict';
// UI regression: a coloured cell and the full native vector must refer to the same archive.
const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const {checkWrite,checkedDir}=require('../../frontend/electron/storagePolicy.js');
const {chromium}=require(process.env.PLAYWRIGHT_CORE||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const root=path.resolve(__dirname,'../..'),preview=path.join(root,'research/glm5/docs/previews');
const raw=require('../../frontend/src/components/v8/data/graphObservationExample.json');
require('../../frontend/src/components/v8/data/nativeHeadChannels.js');
global.window=global;
require(path.join(preview,'free-model-data.js'));

async function main(){
  const out=checkedDir(path.join(__dirname,'result/feature_console_readout'),1024**2,'特征读数回执');
  const tmp=checkedDir(path.join(root,'tests/glm5_temp'),256*1024**2,'特征读数浏览器临时目录');
  checkWrite(tmp,256*1024**2,'检查浏览器临时目录容量');
  const scratch=fs.mkdtempSync(path.join(tmp,'feature-readout-'));
  for(const variable of ['TMP','TEMP','TMPDIR'])process.env[variable]=scratch;
  let browser;
  const receipt={passed:false,checks:[],errors:[],network:[]};
  try {
    browser=await chromium.launch({executablePath:process.env.CHROME_PATH||path.join(os.homedir(),'AppData/Local/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-win64/chrome-headless-shell.exe')});
    const page=await browser.newPage({viewport:{width:1440,height:960},locale:'zh-CN'});page.setDefaultTimeout(8000);
    page.on('pageerror',e=>receipt.errors.push(e.message));await page.route(/^https?:/,r=>{receipt.network.push(r.request().url());return r.abort();});
    const field=page.getByRole('region',{name:'原生全场读数'}),value=field.locator('[data-field-value]');
    const paint=page.locator('[data-feature-colors]');
    async function open(id,b,kind,index){
      const url=pathToFileURL(path.join(preview,'client-distributed-testing.html'));
      url.hash='?'+new URLSearchParams({wb:'1',studyPane:'tools',observeMode:'features',observeTab:'object',observePlugin:'object',
        mdl:id,sblock:String(b+1),smodule:kind,sindex:String(index),lens:'spatial',view:'graph',gview:'langmeasure'});
      await page.goto('about:blank');await page.goto(url.href);
      await paint.waitFor();await page.waitForFunction(b=>document.querySelector('[data-feature-colors]')?.dataset.block===String(b),b);
    }
    async function enter(){await page.getByRole('button',{name:'原生单元',exact:true}).click();await field.waitFor();}
    async function hasValue(expected){
      await page.waitForFunction(x=>{const el=document.querySelector('.ow-native-inspector [data-field-value]');return el && el.textContent.trim()!=='' && Number(el.textContent)===x;},expected);
    }
    await open('Qwen3-14B',36,'mlp',2429);
    const bits=Buffer.from(raw.mlp,'base64'),f=Buffer.alloc(4);
    const at=(site,idx)=>{f.writeUInt16LE(bits.readUInt16LE(2*(site*raw.mlpShape[1]+idx)),2);return f.readFloatLE(0);};
    await page.waitForFunction(x=>{const o=document.querySelector('[data-feature-value="mlp"]');return o?.dataset.value!=='' && Number(o?.dataset.value)===x;},at(3,2429));
    await enter();await hasValue(at(3,2429));
    await page.getByLabel('全屏特征读出位置').selectOption('0');
    await hasValue(at(0,2429));
    receipt.checks.push('14B B36/U2429 同一份原场的两个 token 读数与场景色值对齐');
    await open('Qwen3-14B',26,'attn',36);await enter();
    const head=global.AI2050HeadArchive;
    await hasValue(head.cases['confirm-000-02'].values[0]);
    await page.getByLabel('全屏特征读出位置').selectOption('confirm-000-00');await hasValue(head.cases['confirm-000-00'].values[0]);
    await page.getByLabel('全场测量量').selectOption('head_q');
    await page.waitForFunction(()=>document.querySelector('.ow-native-inspector [data-field-value]').textContent==='未采集');
    receipt.checks.push('14B B26/H36 的 Z 随所选题更新，缺测 Q 显示为空');
    await open('Qwen3-4B',26,'attn',14);await enter();
    const measurements=global.FreeModelData.fields('Qwen3-4B','head',26,14);
    for(const vector of measurements){
      await page.getByLabel('全场测量量').selectOption(vector.measure);
      await hasValue(vector.values[0]);
      if(['head_k','head_v'].includes(vector.measure))assert.match(await field.innerText(),/共享 KV3/);
    }
    receipt.checks.push('4B 的 128 通道 Z/Q/K/V 对齐同一题与共享 KV3');
    await page.getByLabel('原生对象类型').selectOption('ffn');
    const ff=global.FreeModelData.fields('Qwen3-4B','ffn',26,0)[0];
    await hasValue(ff.values[0]);
    const form=field.locator('[data-field-form]');
    await form.locator('input').fill(String(ff.values.length-1));await form.getByRole('button',{name:'定位',exact:true}).click();
    await hasValue(ff.values.at(-1));
    receipt.checks.push('4B 同题从 Head 切到 FFN 可定位末位 U9727');
    await open('Qwen3-14B',26,'attn',35);await enter();
    assert.equal(await value.innerText(),'未采集');
    assert.match(await field.innerText(),/没有保存这项测量/);
    receipt.checks.push('相邻未测头保持缺测');
    assert.deepEqual(receipt.network,[]);assert.deepEqual(receipt.errors,[]);
    receipt.passed=true;
  }finally{
    await browser?.close();const checked=fs.realpathSync(scratch);
    if(path.dirname(checked)!==fs.realpathSync(tmp)||!path.basename(checked).startsWith('feature-readout-'))throw Error('Temporary path changed');
    fs.rmSync(checked,{recursive:true,force:true});
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(receipt,null,2)+'\n');
  }
  console.log(receipt.checks.join('\n'));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
