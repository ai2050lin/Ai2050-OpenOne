const assert=require('node:assert/strict');
const path=require('node:path');
require('../../frontend/src/components/v8/sparseDictionaryArchive.js');
const dictionaries=globalThis.AI2050SparseDictionary;

function createChecks({head,structure}) {
  const id='engineering-plugins-readonly';
  const tokenText=[' red',' is',' here'];
  const ro=[0,1,2];
  const {model_snapshot:snapshot,d_model:width}=structure;
  const requests={fields:[],attention:[],posts:[]};
  const reject={geometry:false,attention:false};
  async function handle(request,url,json,array){
    if(request.method()==='POST') requests.posts.push(url.pathname);
    const prefix='/api/traces/'+id+'/';
    if(!url.pathname.startsWith(prefix))return false;
    const getInt=key=>Number(url.searchParams.get(key));
    const layer=getInt('block'),pos=getInt('position');
    let data;
    switch(url.pathname.slice(prefix.length)){
      case 'manifest':
        data={trace_id:id,model_snapshot:snapshot,readout_positions:ro,tokens:tokenText,n_tokens:3,n_layers:36,d_model:width,n_heads:32,d_ff:9728};break;
      case 'bundle':
        data={trace_id:id,model_snapshot:snapshot,position:pos,readout_positions:ro,
          hs:array([37,width],1),attn:array([36,32,3]),mlp:array([36,9728],1)};break;
      case 'field': {
        requests.fields.push({layer,pos,rejected:reject.geometry});
        const values=new Array(width).fill(0);values[pos]=1;values[3]=layer/20;
        data={schema:'mechanism_field.v1',trace_id:reject.geometry?'unrelated-trace':id,model_snapshot:snapshot,
          position:pos,layer,boundary:layer===0?'embedding':'hidden_state',shape:[width],values};break;
      }
      case 'attention-row': {
        const query=getInt('query'),h=getInt('head');
        requests.attention.push({block:layer,head:h,query,rejected:reject.attention});
        data={schema:'mechanism_attention_row.v1',trace_id:reject.attention?'unrelated-trace':id,
          block:layer,head:h,query_position:query,validation:.001,weights:[.5,.25,.25]};break;
      }
      default:return false;
    }
    await json(data);return true;
  }

  async function verify(page,base,out) {
    const checked=[];
    const work=page.locator('.ow-graph'),scene=work.locator('.aw-space');
    const tools=work.getByRole('group',{name:'分析插件'});
    const mgmt=work.getByRole('group',{name:'插件管理'});
    const loc=work.locator('.ow-object-locator');
    const button=(scope,name)=>scope.getByRole('button',{name,exact:true});
    const tabs=work.getByRole('tablist',{name:'控制台功能'});
    const names=['注意力分布','表征几何','SAE 特征字典'];
    const panel=name=>work.getByRole('region',{name,exact:true});
    const manager=async open=>{
      if((await button(tools,'管理插件').getAttribute('aria-expanded'))!==String(open))await button(tools,'管理插件').click();
    };
    const activate=name=>button(tools,name).click();
    const locate=async coord=>{const input=loc.getByLabel('输入原生编号',{exact:true});await input.fill(coord);await input.press('Enter');};
    await page.goto(base+'?wb=1&lens=route');
    await button(page.getByRole('navigation',{name:'工作台界面'}),'语言与模型').click();
    await work.getByRole('region',{name:'B26/H36 归档注意力'}).waitFor();
    for(const name of names)assert.equal(await button(tools,name).count(),0);
    await manager(true);
    for(const name of names)await mgmt.getByRole('checkbox',{name,exact:true}).check();
    await mgmt.getByRole('checkbox',{name:names[1],exact:true}).uncheck();
    assert.equal(await button(tools,names[1]).count(),0);
    await mgmt.getByRole('checkbox',{name:names[1],exact:true}).check();
    await manager(false);
    await page.reload();
    await tools.waitFor();
    for(const name of names)assert.ok(await button(tools,name).isVisible());
    checked.push('默认四项与可选三项、逐项添加与移除、刷新后保持');

    const attention=panel(names[0]), geometry=panel(names[1]), sae=panel(names[2]);
    await button(loc,'B26/H36').click();await activate(names[0]);
    const source=head.families[0].cases.find(r=>r.id==='confirm-000-02');
    const mass=[...source.mass,1-source.mass.reduce((a,b)=>a+b,0)];
    const entropy=-mass.reduce((sum,x)=>x>0?sum+x*Math.log2(x):sum,0);
    await attention.getByText(entropy.toFixed(3)+' bit',{exact:true}).waitFor();
    await attention.getByLabel('前 K 个来源').selectOption('1');
    assert.equal(await attention.locator('tbody tr').count(),1);
    const top=Math.max(...mass);
    assert.ok((await attention.locator('tbody tr').innerText()).includes((100*top).toFixed(2)+'%'));
    assert.ok((await attention.innerText()).includes('组内 token 的分布未知'));
    await button(scene,'⛶ 全屏').click();
    assert.ok(await scene.evaluate(el=>el.classList.contains('max')));
    await attention.getByLabel('前 K 个来源').selectOption('6');
    assert.equal(await attention.locator('tbody tr').count(),6);
    await page.screenshot({path:path.join(out,'head-fullscreen.png')});
    await button(scene,'退出全屏（Esc）').click();
    checked.push('真实 B26/H36 五条记录及其余来源的分组熵和累计质量，全屏可操作');

    await button(loc,'B36/U2429').click();await activate(names[1]);
    await geometry.getByLabel('表征几何对照层').selectOption('36');
    await geometry.getByText('0.9918',{exact:true}).waitFor();
    await geometry.getByText('0.9429',{exact:true}).waitFor();
    const matrix=geometry.getByRole('table');
    assert.equal(await matrix.locator('tbody tr').count(),4);
    assert.equal(await matrix.locator('tbody tr').first().locator('td').nth(1).innerText(),'0.344');
    await button(geometry,'对照层 L36 距离矩阵').click();
    assert.ok((await matrix.locator('caption').innerText()).includes('L36'));
    await page.screenshot({path:path.join(out,'geometry.png')});
    checked.push('B36 真实 4 × 5120 跨层原场、L37 / L36 CKA 和 RSA、距离矩阵');

    await activate(names[2]);await sae.getByRole('status').waitFor();assert.equal(await sae.getByLabel('字典特征 F').count(),0);
    await work.getByLabel('模型',{exact:true}).selectOption('Qwen3-4B');
    await tabs.getByRole('tab',{name:'插件分析',exact:true}).click();await activate(names[2]);
    for(const entry of dictionaries.layers){
      const selector=button(sae,'B'+entry.block+' 字典');
      if(!await selector.isDisabled())await selector.click();
      const choose=sae.getByLabel('字典特征 F');
      const f=entry.features.at(-1);await choose.selectOption(String(f.id));
      assert.match(await choose.locator('option:checked').innerText(),new RegExp('F'+f.id+' '));
      assert.ok((await sae.innerText()).includes(f.orientedDelta.toFixed(5)));
      const details=sae.locator('details');if(!await details.evaluate(el=>el.open))await details.locator('summary').click();
      assert.match(await details.innerText(),/模型本身的 FFN 神经元使用 U 编号/);
      assert.ok((await details.innerText()).includes(entry.dictionarySha));
    }
    await page.screenshot({path:path.join(out,'sae.png')});
    checked.push('Qwen3-4B 三块 SAE 真实字典、F 与 U 分开、快照错配不出卡片');

    const traceUrl=base+'?wb=1&lens=spatial&view=graph&gview=structure&mdl=Qwen3-4B&observeMode=archive&observeTab=plugin&trace='+id+'&tpos=2&snapshot='+snapshot;
    await page.goto(traceUrl);
    await locate('B4/H2');await activate(names[0]);
    await attention.getByText('1.500 bit',{exact:true}).waitFor();
    assert.match(await attention.locator('tbody tr').first().innerText(),/#0\s+red/);
    reject.attention=true;await locate('B4/H3');await activate(names[0]);
    await attention.getByRole('status').filter({hasText:'不符'}).waitFor();
    assert.equal(await attention.getByRole('table').count(),0);
    await activate(names[1]);await geometry.getByRole('table').waitFor();
    assert.ok(requests.fields.some(call=>call.layer===5&&call.pos===2));
    reject.geometry=true;
    await geometry.getByLabel('表征几何对照层').selectOption('3');
    await geometry.getByRole('status').filter({hasText:'不一致'}).waitFor();
    assert.equal(await geometry.getByRole('table').count(),0);
    reject.geometry=false;await button(work,'重新读取归档').click();
    await geometry.getByRole('table').waitFor();
    checked.push('只读轨迹节点夹具的逐 token 熵、几何字段、身份错误拒绝和重试恢复');

    await manager(true);await button(mgmt,'恢复默认').click();
    for(const name of names)assert.equal(await button(tools,name).count(),0);
    assert.deepEqual(await tools.getByRole('button').allTextContents(),['Head','HiddenState','MLP','Logit Lens','管理插件']);
    await page.reload();await tools.waitFor();for(const name of names)assert.equal(await button(tools,name).count(),0);
    assert.deepEqual(requests.posts,[]);
    checked.push('恢复默认及刷新保持，所有检查零 POST');
    return {checked,archiveEntropyBits:entropy,archiveCKA:.9918,archiveRSA:.9429,saeBlocks:dictionaries.blocks,requests};
  }
  return {handle,verify};
}
module.exports={createChecks};
