const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const {checkWrite} = require('../../frontend/electron/storagePolicy.js');

const [service,dist,work,questions] = process.argv.slice(2);
const [original] = JSON.parse(fs.readFileSync(questions,'utf8'));
const cfg = require('../../models/hf/Qwen3-14B/config.json');
const model = {model:'Qwen3-14B',model_snapshot:original.snapshot,n_layers:cfg.num_hidden_layers,n_heads:cfg.num_attention_heads,
  n_kv_heads:cfg.num_key_value_heads,head_dim:cfg.head_dim,d_model:cfg.hidden_size,d_ff:cfg.intermediate_size};
const shell = 'http://127.0.0.1:53191';
const root = path.resolve(dist);
const executablePath = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

async function main(){
  checkWrite(work,150*2**20,'同题实时 3D 浏览器验收');
  const browser = await chromium.launch({headless:true,executablePath});
  try {
    const context = await browser.newContext({viewport:{width:1440,height:900}});
    await context.addInitScript(()=>{window.__AI2050__={mode:'local'};});
    const apiCalls=[];
    await context.route(/^https?:/, async route=>{
      const url = new URL(route.request().url());
      if(url.origin!==shell)return route.abort();
      if(url.pathname.startsWith('/api/live/')||url.pathname.startsWith('/api/traces/')){
        apiCalls.push({method:route.request().method(),path:url.pathname});
        const response=await route.fetch({url:service+url.pathname+url.search,timeout:240000});
        return route.fulfill({response});
      }
      if(url.pathname==='/api/model/skeleton')return route.fulfill({json:{models:[model]}});
      if(url.pathname==='/api/platform/config')return route.fulfill({json:{}});
      if(url.pathname.startsWith('/api/'))return route.fulfill({status:404,json:{detail:'not part of live acceptance'}});
      const file=path.resolve(root,url.pathname==='/'||url.pathname==='/workbench'?'index.html':'.'+decodeURIComponent(url.pathname));
      if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile())return route.fulfill({status:404});
      return route.fulfill({path:file});
    });
    const page=await context.newPage();
    page.setDefaultTimeout(20000);
    const errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    const params=new URLSearchParams({wb:'1',lens:'spatial',view:'graph',gview:'structure',mdl:model.model,snapshot:model.model_snapshot,
      sblock:'27',smodule:'attn',sindex:'36',observeMode:'live',observeTab:'run',liveCase:original.id});
    await page.goto(shell+'/workbench?'+params);
    const scene=page.getByRole('region',{name:'模型运行 3D 场景'}), live=page.getByRole('region',{name:'实时模式工作区'});
    const readout=scene.getByRole('region',{name:'3D 原生读数'});
    const btn=(el,name)=>el.getByRole('button',{name,exact:true});
    await live.getByText('归档原始输入 · '+original.id,{exact:true}).waitFor();
    assert.equal(await live.getByLabel('运行文本').inputValue(),original.prompt);
    await btn(live,'加载模型').click();
    await live.getByText('Qwen3-14B · 已加载',{exact:true}).waitFor({timeout:240000});
    const response=page.waitForResponse(r=>r.url().endsWith('/api/live/run')&&r.request().method()==='POST',{timeout:240000});
    await btn(live,'开始运行').click();
    const capture=await response;
    assert.ok(capture.ok(),await capture.text());
    const trace=await capture.json();
    assert.deepEqual(trace.token_ids,original.tokens);
    assert.equal(trace.model_snapshot,original.snapshot);
    assert.equal(trace.output_top[0].token_id,original.gold_id);
    assert.equal(trace.prompt,original.prompt);
    assert.equal(trace.provenance.prompt_format,'rendered');
    const traceURL=`${service}/api/traces/${trace.trace_id}`;
    const field=async (position,boundary)=>{
      const r=await fetch(`${traceURL}/field?block=26&position=${position}&boundary=${boundary}`);
      assert.ok(r.ok,await (r.ok?Promise.resolve(''):r.text()));
      return r.json();
    };
    const rawZ=readout.getByLabel('Head 通道原值');
    const rawA=readout.getByLabel('MLP 单元原值');
    const scale=()=>scene.locator('[data-scene-scale=channels]').innerText();
    const token=live.getByLabel('当前Token');
    const last=trace.token_ids.length-1;
    await token.waitFor();
    assert.equal(await token.inputValue(),String(last));
    await scene.getByLabel('Head 着色量').selectOption('strength');
    const hop=readout.getByLabel('3D Head 通道');
    const assertValue=async(el,n)=>{
      await page.waitForFunction(({label,expected})=>document.querySelector(`output[aria-label="${label}"]`)?.textContent===expected,
        {label:await el.getAttribute('aria-label'),expected:String(n)});
    };
    const originalVectors=[];
    for(const position of [last,...trace.readout_positions.filter(p=>p!==last)]){
      const data=await field(position,'head_z');
      assert.deepEqual(data.shape,[40,128]);
      originalVectors.push({position,head:data.values.slice(36*128,37*128)});
    }
    const end=originalVectors[0].head;
    await assertValue(rawZ,end[0]);
    await btn(scene,'锁定色标').click();
    const locked=await scale();
    for(const {position,head} of originalVectors){
      await token.selectOption(String(position));
      for(const [channel,n] of head.entries()){
        await hop.selectOption(String(channel));
        await assertValue(rawZ,n);
      }
      const strength=Math.sqrt(head.reduce((s,x)=>s+x*x,0)/128);
      assert.equal(Number(await readout.getByLabel('Head 总览原值').innerText()),strength);
      assert.equal(await scale(),locked);
    }
    const missing=trace.token_ids.findIndex((_,i)=>!trace.readout_positions.includes(i));
    await token.selectOption(String(missing));
    await assertValue(rawZ,'未采集');
    await readout.getByRole('status').filter({hasText:'未采集 Head 输出'}).waitFor();
    assert.equal(await scale(),locked);
    await page.screenshot({path:path.join(work,'live-missing.png')});
    await token.selectOption(String(last));
    await hop.selectOption('0'); await assertValue(rawZ,end[0]);
    await page.screenshot({path:path.join(work,'live-head.png')});

    const jump=async address=>{await scene.getByLabel('输入原生编号').fill(address);await btn(scene,'定位对象').click();};
    await jump('B26/U17407');
    const lastMlp=await field(last,'mlp_activation');
    assert.deepEqual(lastMlp.shape,[17408]);
    await assertValue(rawA,lastMlp.values.at(-1));
    const mlpScale=await scene.locator('[data-scene-scale=mlp]').innerText();
    for(const position of trace.readout_positions){
      await token.selectOption(String(position));
      const data=await field(position,'mlp_activation');
      await assertValue(rawA,data.values.at(-1));
      assert.equal(await scene.locator('[data-scene-scale=mlp]').innerText(),mlpScale);
    }
    await token.selectOption(String(missing));await assertValue(rawA,'未采集');
    await readout.getByRole('status').filter({hasText:'未采集 MLP 激活'}).waitFor();
    await token.selectOption(String(last));await assertValue(rawA,lastMlp.values.at(-1));
    await page.screenshot({path:path.join(work,'live-mlp.png')});

    params.set('observeMode','archive');params.delete('liveCase');params.set('tpos',String(last));params.set('trace',trace.trace_id);
    await page.goto(shell+'/workbench?'+params);
    await assertValue(rawZ,end[0]);
    await page.screenshot({path:path.join(work,'live-trace-replay.png')});
    params.delete('trace');params.delete('tpos');params.set('headCase',original.id);params.set('observeArchive','head');
    params.set('observeTab','plugin');params.set('observePlugin','heads');
    await page.goto(shell+'/workbench?'+params);
    await assertValue(rawZ,original.archived_z[0]);
    assert.match(await readout.innerText(),/confirm-000-02 · token 190（ID 510）/);
    await jump('B26/H35');await assertValue(rawZ,'未采集');
    await readout.getByRole('status').filter({hasText:'只保存 H36'}).waitFor();
    assert.deepEqual(errors,[]);
    assert.equal(apiCalls.filter(x=>x.method==='POST'&&x.path==='/api/live/run').length,1);
    const proof={passed:true,trace_id:trace.trace_id,token_ids:trace.token_ids,readout_positions:trace.readout_positions,
      checked_head_values:originalVectors.length*128,mlp_width:17408,head_scales_locked:true,mlp_scales_locked:true,
      missing_position:missing,live_trace_replay:true,original_archive_replay:true,live_posts:1,errors};
    fs.writeFileSync(path.join(work,'browser-acceptance.json'),JSON.stringify(proof,null,2)+'\n');
    console.log(JSON.stringify({...proof,token_ids:trace.token_ids.length}));
  } finally {await browser.close();}
}

main().catch(error=>{console.error(error);process.exitCode=1;});
