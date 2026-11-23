'use strict';
const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const path=require('node:path');

function createChecks({structure,head}){
  const traceId='9'.repeat(64),runId='qa-record-replay',nativeRun='qa-trace-run';
  const catalog14='qwen3-14b|cfg-e73c3664ca09|bf16';
  const catalog4='qwen3-4b|cfg-8ba006f74fec|bf16';
  const user={user_id:'qa-admin',role:'admin',username:'qa-admin'};
  const makeCase=(id,sample)=>({case_id:id,correct:true,prompt:'QA only',native_trace:{schema:'native_trace_ref.v1',
    run_id:runId,case_id:id,model_snapshot:structure.model_snapshot,trace_id:traceId,trace_run_id:nativeRun,
    sample_id:sample,position:1,layer:5,module:'attn',index:2}});
  const data=Buffer.from([makeCase('qa-good','qa-sample'),makeCase('qa-mismatch','wrong-sample'),{case_id:'qa-summary-only'}].map(row=>JSON.stringify(row)).join('\n')+'\n');
  const sha=createHash('sha256').update(data).digest('hex');
  const fixture={id:runId,state:'completed',archive_state:'server_verified',decision:'new_acquisition',
    plan:{title:'QA center observation',family_ids:['KN-qa'],domains:['KN'],objective:'Software contract',principle:'QA fixture',
      acquisition:{model_id:'Qwen3-4B',model_snapshot:structure.model_snapshot}},final:{summary:'QA only'},
    receipt:{artifacts:[{role:'case_results',availability:'uploaded',name:'case_results.jsonl',size:data.length,sha256:sha}]}};
  const calls=[],posts=[];
  async function handle(req,url,json,array,raw){
    const p=url.pathname, token=req.headers().authorization;
    if(req.method()==='POST')posts.push(p);
    if(p==='/api/auth/status'){await json({user:token==='Bearer qa-session'?user:null});return true;}
    if(p==='/api/model/skeleton'){
      await json({models:[{model:catalog14,model_snapshot:head.snapshot,n_layers:40,n_heads:null,d_model:5120,d_ff:null},
        {...structure,model:catalog4}]});
      return true;
    }
    if(p==='/api/test-archive/runs'){
      if(token!=='Bearer qa-session'){await json({detail:'QA session required'},401);return true;}
      const filters=Object.fromEntries([...url.searchParams].filter(([key])=>!['limit','offset'].includes(key)));
      calls.push(filters);
      await json({items:[fixture],total:1,applied_filters:filters});return true;
    }
    if(p==='/api/test-archive/objects/'+sha){
      if(token!=='Bearer qa-session')await json({detail:'QA session required'},401);else await raw(data);
      return true;
    }
    if(p==='/api/traces/'+traceId+'/manifest'){
      await json({...structure,kind:'real',trace_id:traceId,run_id:nativeRun,sample_id:'qa-sample',tokens:[' A',' B',' C'],readout_positions:[0,1,2]});return true;
    }
    if(p==='/api/traces/'+traceId+'/bundle'){
      const position=Number(url.searchParams.get('position'));
      await json({trace_id:traceId,model_snapshot:structure.model_snapshot,position,readout_positions:[0,1,2],
        hs:array([37,2560],3),attn:array([36,32,3]),mlp:array([36,9728])});return true;
    }
    if(p==='/api/traces/'+traceId+'/field'){
      const values=Array(128).fill(0);values[0]=3.25;
      await json({schema:'mechanism_head_vector.v1',kind:'real',trace_id:traceId,model_snapshot:structure.model_snapshot,
        block:4,head:2,kv_group:0,position:1,boundary:'head_z',shape:[128],values,source:'native_BF16_bits'});return true;
    }
    return false;
  }
  async function verify(page,base,out){
    const work=page.locator('.ow-graph'),scene=work.getByRole('region',{name:'模型运行 3D 场景'});
    const button=(scope,name)=>scope.getByRole('button',{name,exact:true});
    const records=work.getByRole('region',{name:'测试记录',exact:true});
    const reading=scene.getByRole('region',{name:'3D 测试读数'});
    const canvas=scene.locator('.aw-canvas canvas');
    async function jump(label){const search=scene.getByLabel('输入原生编号');await search.fill(label);await search.press('Enter');}
    await page.goto(base+'?wb=1&lens=spatial&view=graph&gview=structure&mdl='+encodeURIComponent(catalog14)+'&observeMode=features');
    await jump('B26/H36');
    await canvas.evaluate(el=>{window.__qa3dCanvas=el;});
    await button(scene,'查相关测试').click();
    assert.equal(await button(scene,'特征分析').getAttribute('aria-pressed'),'true');
    assert.equal(await records.getByLabel('搜索测试').inputValue(),'B26/H36');
    await button(records,'查看 B26/H36 已发布原题').click();
    const cues=records.getByRole('region',{name:'三线索选源'});
    await button(cues,'物理首行').click();
    await button(cues,'在 3D 中观察本题').click();
    await reading.getByText('归档原题 confirm-000-02',{exact:false}).waitFor({timeout:4000}).catch(async error=>{
      throw new Error(`${error.message}\n${page.url()}\n${(await page.locator('body').innerText()).slice(0,1500)}`);
    });
    assert.equal(await canvas.getAttribute('data-archive-count'),'1');
    assert.ok(await canvas.evaluate(el=>el===window.__qa3dCanvas));
    const toolbox=work.getByRole('group',{name:'分析插件'});
    await button(toolbox,'管理插件').click();
    await work.getByRole('group',{name:'插件管理'}).getByRole('checkbox',{name:'注意力分布',exact:true}).check();
    await button(toolbox,'注意力分布').click();
    await button(work.getByRole('region',{name:'注意力分布',exact:true}),'将注意力结果标在 3D').click();
    const overlay=scene.getByRole('region',{name:'3D 分析结果'});
    await overlay.getByText(/熵/).waitFor();
    await page.waitForFunction(()=>{
      const box=document.querySelector('.ow-graph .aw-canvas canvas').getBoundingClientRect();
      return box.top>=0&&box.bottom<=innerHeight;
    });
    await page.screenshot({path:path.join(out,'published-3d.png')});
    await jump('B26/H35');await overlay.waitFor({state:'hidden'});
    assert.equal(await canvas.getAttribute('data-archive-count'),'1');
    await button(scene,'原生单元').click();
    await scene.locator('[data-field-missing]').filter({hasText:'当前对象未随本题保存'}).waitFor();

    await page.evaluate(account=>{
      localStorage.setItem('ai2050-session','qa-session');localStorage.setItem('ai2050-user',JSON.stringify(account));
    },user);
    await page.goto(base+'?wb=1&lens=spatial&view=graph&gview=structure&mdl='+encodeURIComponent(catalog4)+'&observeMode=live');
    await jump('B4/H2');await button(scene,'查相关测试').click();
    assert.equal(await button(scene,'实时模式').getAttribute('aria-pressed'),'true');
    await records.getByLabel('模型标识').waitFor();
    assert.equal(await records.getByLabel('模型标识').inputValue(),'');
    await records.getByLabel('研究领域').selectOption('KN');
    await records.getByText('高级筛选',{exact:true}).click();
    assert.equal(await records.getByLabel('模型快照').inputValue(),structure.model_snapshot);
    await records.getByLabel('原生地址').fill('B4/H2');
    await records.getByLabel('有已上传产物').selectOption('case_results');
    await records.getByRole('button',{name:/QA center observation/}).click();
    await records.getByText('逐题结果与下载',{exact:true}).click();
    await button(records,'读取逐题结果').click();
    await records.getByText('qa-summary-only',{exact:true}).click();
    await records.getByText('归档没有可绑定到此题的原生轨迹。',{exact:false}).waitFor();
    await records.getByText('qa-good',{exact:true}).click();
    await button(records,'在 3D 中回放本题').click();
    await reading.getByText('归档原题 qa-good · token 1',{exact:true}).waitFor();
    await work.getByLabel('当前Token').selectOption('1');
    await button(scene,'原生单元').click();
    await scene.locator('[data-field-value]').filter({hasText:/^3.25$/}).waitFor();
    await button(scene,'3D 与控制台').click();
    await button(reading,'查看测试记录').click();
    await records.getByText('qa-mismatch',{exact:true}).click();
    await button(records.getByText('qa-mismatch',{exact:true}).locator('..'),'在 3D 中回放本题').click();
    await reading.getByText('逐题引用与轨迹清单的模型、运行、样本或位置不一致。',{exact:true}).waitFor();
    assert.equal(await canvas.getAttribute('data-archive-count'),'0');
    assert.ok(calls.some(q=>!q.model&&q.snapshot===structure.model_snapshot&&q.domain==='KN'&&q.target==='B4/H2'&&q.artifact_role==='case_results'));
    assert.deepEqual(posts,[]);
    return {checks:['3D 选中部件带入检索，不替换场景或实时模式','发布原题、注意力标记和缺测邻头使用真实归档',
      '中心筛选与认证、逐题 SHA 核验、按原位置读取原值','无轨迹记录保留原摘要，错样本清单拒绝叠加，零 POST'],filters:calls};
  }
  return {handle,verify};
}
module.exports={createChecks};
