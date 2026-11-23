(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const catalog = window.FreeModelData, evidence = window.EncodingEvidence;
  const b23 = window.NativeReviewB23;
  const wood = catalog.fields('Qwen3-4B','ffn',0)[0];
  if (!wood || b23.width !== catalog.model(b23.model).ffn || b23.snapshot !== catalog.model(b23.model).snapshot)
    throw Error('随稿原场和模型结构身份不一致。');
  const nativeRows = b23.rows.map(record => {
    const bytes = Uint8Array.from(atob(record.bits), c => c.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    if (bytes.length !== 2*b23.width) throw Error('B23 原生记录长度不一致。');
    const storage = new ArrayBuffer(4);
    const word = new DataView(storage);
    return Float32Array.from({length:b23.width}, (_,i)=>{
      word.setUint32(0,view.getUint16(2*i,true)*65536,true);
      return word.getFloat32(0,true);
    });
  });
  const option = (v,t) => new Option(t,String(v));
  const esc = v => String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = n => n.toLocaleString('en-US');
  const sources = {
    wood: {label:'木盒颜色 · 4B 全层响应',model:wood.model,snapshot:wood.snapshot,sample:wood.caseId,
      address:'B35/H31/D127', sites:[{label:'输入末位 · token '+wood.position}]},
    head: {label:'Phase 2983 · 14B 按规则选行',model:evidence.head.model,snapshot:evidence.head.snapshot,
      address:'B26/H36/D0', sites:evidence.headCases.map(c=>({label:c.rule+' · token '+c.position}))},
    ffn: {label:'Phase 2933 · 14B 算术接续',model:evidence.ffn.record.model,snapshot:evidence.ffn.record.snapshot,
      address:'B36/U2429', sites:evidence.ffn.positions.map(p=>({label:p.label+' · token '+p.index}))},
    b23: {label:'Phase 3053 · 14B B23 语料',model:b23.model,snapshot:b23.snapshot,
      address:'B23/U7708', sites:b23.rows.map((r,i)=>({label:(i?'留出负端':'留出正端')+' · token '+r.position}))},
    structure: {label:'结构入口 · 查看任意编号',sites:[]},
    live: {label:'本机输入 · 待采集',sites:[]},
  };
  const fields = {head_z:'Z · O 投影前',head_q:'Q · 查询通道',head_k:'K · 共享键通道',
    head_v:'V · 共享值通道',attention:'注意力来源 · token',mlp_activation:'A · 门控乘积',hidden:'残差坐标'};
  const kinds={head:'Head',ffn:'MLP',hidden:'HiddenState'};
  const state={model:catalog.model('Qwen3-4B'),source:'wood',site:0,object:{kind:'head',block:35,head:31,index:127},
    measure:'head_z',mode:'features',tab:'plugin',expanded:false,fullscreen:false,input:''};
  const fieldModes={head:['head_z','head_q','head_k','head_v','attention'],ffn:['mlp_activation'],hidden:['hidden']};
  const address = (o=state.object) => o.kind==='head' ? 'B'+o.block+'/H'+o.head+'/D'+o.index :
    o.kind==='ffn' ? 'B'+o.block+'/U'+o.index : 'hs['+o.block+']/D'+o.index;
  const dimension = () => ({head:state.model.headDim,ffn:state.model.ffn,hidden:state.model.hidden}[state.object.kind]);
  const scene = window.createModelScene($('native-scene'),{
    onSelectBlock: block=>choose({...state.object,block:state.object.kind==='hidden'?block+1:block}),
    onSelect:(block,head)=>choose({kind:'head',block,head,index:0}),
    onSelectUnit:(_block,index)=>choose({...state.object,index}),
  });
  const meter = window.AI2050NativeField.create($('native-field-host'),{onPick:index=>{
    if(state.measure==='attention'){
      const r=readout();
      $('source-index').textContent = '来源 token '+index+' · '+(r?.tokens?.[index] ?? '该位置无原词元摘录');
    } else choose({...state.object,index},false);
  }});
  $('model').replaceChildren(...catalog.models.map(m=>option(m.id,m.id)));
  $('source').replaceChildren(...Object.entries(sources).map(([key,item])=>option(key,item.label)));

  function parse(input) {
    let m;
    const raw=input.trim().replace(/\s+/g,'');
    if ((m=/^B(\d+)\/H(\d+)(?:\/D(\d+))?$/i.exec(raw))) return {kind:'head',block:+m[1],head:+m[2],index:+(m[3]||0)};
    if ((m=/^B(\d+)\/U(\d+)$/i.exec(raw))) return {kind:'ffn',block:+m[1],index:+m[2]};
    if ((m=/^hs\[(\d+)\]\/D(\d+)$/i.exec(raw))) return {kind:'hidden',block:+m[1],index:+m[2]};
    throw Error('请输入 B35/H31/D127、B35/U9727 或 hs[0]/D0 形式的原生地址。');
  }
  function validate(o,m) {
    const last = m.layers - (o.kind==='hidden'?0:1);
    if (o.block<0 || o.block>last) throw Error('该模型的此类层编号为 0–'+last+'。');
    if (o.kind==='head' && (o.head<0 || o.head>=m.heads)) throw Error('该模型的 Head 编号为 H0–H'+(m.heads-1)+'。');
    const width = {head:m.headDim,ffn:m.ffn,hidden:m.hidden}[o.kind];
    if (o.index<0 || o.index>=width) throw Error('该对象的原生范围是 0–'+(width-1)+'。');
  }
  function choose(next, resetError=true) {
    validate(next,state.model);
    const changedKind = next.kind!==state.object.kind;
    state.object=next;
    if (changedKind) state.measure=fieldModes[next.kind][0];
    if (resetError) $('address-error').hidden=true;
    render();
  }
  function dataMatches() {
    const source = sources[state.source];
    return source.model===state.model.id && source.snapshot===state.model.snapshot;
  }
  function vectors(o=state.object) {
    if (!dataMatches()) return [];
    if (state.source==='b23') {
      if (o.kind!=='ffn' || o.block!==b23.block) return [];
      return b23.rows.map((r,i)=>({values:nativeRows[i],measure:'mlp_activation',site:i,position:r.position,
        caseId:r.doc,run:b23.run,source:b23.source.manifest,sha256:b23.source.sha256,
        boundary:'down_proj 输入 · SiLU(gate) × up',question:r.text}));
    }
    if ((state.source==='head' && o.kind!=='head') || (state.source==='ffn' && o.kind==='head')) return [];
    return catalog.fields(state.model.id,o.kind,o.block,o.kind==='head'?o.head:o.index);
  }
  function readout() {
    const set=vectors();
    if (state.measure==='attention') {
      const c = set.find(r=>r.measure==='head_z' && r.attention);
      return c ? {...c,values:c.attention.weights,tokens:c.attention.tokens,
        boundary:'查询位置对每个来源 token 的注意力权重',source:c.attention.source} : null;
    }
    const matches=set.filter(r=>r.measure===state.measure);
    return state.source==='head'?matches[state.site]:matches.find(r=>r.site===state.site);
  }
  function missingReason() {
    if (state.source==='structure') return '结构范围可浏览。请选择实测归档或准备本机采集，以查看响应。';
    if (state.source==='live') return '待采集。本稿演示采集入口，请查看「运行与对照」中的配置。';
    if (!dataMatches()) return '该数据来源与当前模型快照不匹配。可保留当前地址，切换到同快照实测来源。';
    if (state.source==='head') return '该归档包含 B26/H36 的 Z 通道与归档行级摘要。当前部件或字段没有对应向量。';
    if (state.source==='b23') return '该归档包含 B23 的整层 MLP 两条实测行。当前块或字段没有采集。';
    return '所选来源没有这一层、位置或字段的原生数值。';
  }
  function selectSource(key) {
    state.source=key; state.site=0;
    if (key==='live') {state.mode='live';state.tab='run';}
    render();
  }
  function bindArchive(key) {
    const record=sources[key];
    const next=catalog.model(record.model);
    if (next.snapshot!==record.snapshot) throw Error('归档模型快照不匹配。');
    const o=parse(record.address);
    validate(o,next);
    state.model=next;state.source=key;state.site=key==='head'?2:0;state.object=o;
    state.measure=fieldModes[o.kind][0];state.tab='plugin';
    $('address-error').hidden=true;
    render();
    $('address').focus({preventScroll:true});
  }

  function renderHeadList() {
    const tray=$('head-list');
    tray.hidden=!state.expanded || state.object.kind!=='head' || state.tab!=='plugin';
    $('head-legend').hidden=tray.hidden;
    $('all-heads').hidden=state.object.kind!=='head';
    $('all-heads').setAttribute('aria-expanded',String(!tray.hidden));
    if(tray.hidden)return;
    const {object:o,model:m}=state;
    const rms=Array.from({length:m.heads},(_,head)=>{
      const z=vectors({...o,head}).filter(r=>r.measure==='head_z')[state.source==='head'?state.site:0];
      return z ? Math.sqrt(z.values.reduce((sum,v)=>sum+v*v,0)/z.values.length) : null;
    });
    const max=Math.max(0,...rms);
    tray.replaceChildren(...rms.map((value,head)=>{
      const b=document.createElement('button');
      b.type='button';b.dataset.head=String(head);b.setAttribute('aria-pressed',String(head===o.head));
      b.textContent='H'+head;
      const small=document.createElement('small');
      small.textContent=value===null?'未采':value.toPrecision(3);
      if(max && value!==null)b.style.backgroundColor='rgba(100,153,229,'+(0.08+0.28*value/max)+')';
      b.append(small);return b;
    }));
  }
  function renderField(r) {
    const o=state.object, attn=state.measure==='attention';
    const kind=o.kind==='ffn'?'ffn':'head';
    const size=attn?(r?.values.length||0):dimension();
    const title=attn ? 'B'+o.block+'/H'+o.head+' · 注意力来源' :
      o.kind==='head'?'B'+o.block+'/H'+o.head+' · 全部通道':
        o.kind==='ffn'?'B'+o.block+' · 整层 MLP':'hs['+o.block+'] · 全部坐标';
    const comparisons=state.source==='ffn' && r ? vectors().map(f=>({label:f.label,value:f.values[o.index],active:f.site===state.site})):[];
    meter.update({kind,size,values:r?.values,selected:attn?0:o.index,vectorLabel:title,
      measure:fields[state.measure],source:r?(r.position!=null?'token '+r.position+' · ':'')+r.boundary:'',
      identity:JSON.stringify([state.model.snapshot,state.source,state.site,o.block,o.head,state.measure]),comparisons,
      missing:r?'':missingReason(),dataLabel:attn?'来源权重':'BF16 原值'});
    $('source-index').hidden=!attn || !r;
    if(attn && r) $('source-index').textContent='图中 D 编号对应来源 token 的位置。选择一格查看该位置的原词元。';
    const details = $('origin-data');
    details.innerHTML='<p>模型快照 <code>'+state.model.snapshot+'</code></p>'+
      (r?'<p>运行 <code>'+esc(r.run)+'</code> · 样本 '+esc(r.caseId)+' · 读出位置 '+r.position+
        '</p><p>原始来源 <code>'+esc(r.source)+'</code></p><p>SHA-256 <code>'+esc(r.sha256 || '随原始归档')+'</code></p>':
        '<p>'+esc(missingReason())+'</p>');
  }
  function renderOther() {
    const out=$('other-panel'), r=readout();
    if(state.tab==='record') {
      out.innerHTML='<h2>测试记录 · 载入原始地址</h2><div class="records">'+
        ['wood','head','ffn','b23'].map(key=>'<article><strong>'+sources[key].label+'</strong><p>'+sources[key].model+' · '+sources[key].address+
          '</p><button type="button" class="button compact" data-archive="'+key+'">在同一工作区打开</button></article>').join('')+
        '</div><p><a href="model-analysis-history-review.html">查看更早的相关结果、反例与复测核对</a></p>';
    } else if (state.tab==='run') {
      out.innerHTML='<h2>本机输入与采集范围</h2><p>当前地址 '+address()+'。正式接入从同一输入保存全层所选位置的全部 Head、MLP 和残差；长文本可指定更多位置。</p>'+
        '<label>输入<textarea id="local-input" aria-label="输入" placeholder="输入文字或一对需要比较的句子">'+esc(state.input)+'</textarea></label>'+
        '<p class="caption">这里展示采集入口。静态评审用已有轨迹，其中未采字段会显示具体原因。</p>'+
        '<button class="button compact" data-tab="plugin">查看当前地址与覆盖</button>';
    } else {
      out.innerHTML='<h2>原生对象 · '+address()+'</h2><p>结构可以定位到该地址。'+
        (r?'此地址有 '+esc(r.boundary)+' 的实测值。':esc(missingReason()))+'</p>'+
        '<p><a class="button compact" href="'+$('parameter-link').getAttribute('href')+'">在参数工作台打开同一地址</a></p>';
    }
  }
  function render() {
    const {model:m,object:o}=state;
    $('model').value=m.id; $('source').value=state.source; $('address').value=address();
    const opts=sources[state.source].sites;
    $('site-label').hidden=opts.length===0;
    $('site').replaceChildren(...opts.map((s,i)=>option(i,s.label)));$('site').value=String(state.site);
    const n=m.layers+(o.kind==='hidden'?1:0);
    $('block').replaceChildren(...Array.from({length:n},(_,i)=>option(i,o.kind==='hidden'?'hs['+i+']':'B'+i)));
    $('block').value=String(o.block);
    for(const b of document.querySelectorAll('[data-mode]')) b.setAttribute('aria-pressed',String(b.dataset.mode===state.mode));
    for(const b of document.querySelectorAll('[data-kind]')) if(b.tagName==='BUTTON') b.setAttribute('aria-pressed',String(b.dataset.kind===o.kind));
    for(const b of $('console-tabs').querySelectorAll('button')) b.setAttribute('aria-pressed',String(b.dataset.tab===state.tab));
    const mOpts=fieldModes[o.kind];
    $('measure').replaceChildren(...mOpts.map(f=>option(f,fields[f])));
    $('measure').value=state.measure;$('measure-label').hidden=mOpts.length<2;
    const r=readout();
    $('scene-address').textContent=address();
    $('scene-context').textContent=r?'token '+r.position+' · 已采响应':'结构定位';
    $('structure-coverage').innerHTML='<p><strong>'+fmt(m.layers)+'</strong> 层 · <strong>'+fmt(m.layers*m.heads)+'</strong> 个查询头</p>'+
      '<p><strong>'+fmt(m.ffn)+'</strong> 个 MLP 单元/层 · 共 <strong>'+fmt(m.layers*m.ffn)+'</strong> 个层内单元</p>';
    const coverage=$('response-coverage');
    coverage.classList.toggle('missing',!r);
    coverage.textContent=r?state.source==='wood'?'此轨迹归档同一位置全部 36 层的 Head 和 MLP。':
      r.values.length===dimension()?'当前字段有完整原生向量，范围以本归档为准。':'当前记录有 '+r.values.length+' 个来源读数。':missingReason();
    $('parameter-link').href='encoding-workbench.html#'+new URLSearchParams({model:m.id,at:address(),panel:'parameters'});
    scene.update({model:m.id,layers:m.layers,heads:m.heads,block:o.kind==='hidden'?Math.max(0,o.block-1):o.block,
      mode:o.kind,stateIndex:o.block,head:o.kind==='head'?o.head:null,unit:o.kind==='head'?null:o.index,width:dimension()});
    $('plugin-panel').hidden=state.tab!=='plugin';
    $('other-panel').hidden=state.tab==='plugin';
    if(state.tab==='plugin'){renderHeadList();renderField(r);}
    else renderOther();
  }
  function fullscreen(on) {
    state.fullscreen=on;$('workspace').classList.toggle('full-screen',on);
    $('fullscreen').textContent=on?'退出全屏（Esc）':'⛶ 全屏观察';
    $('fullscreen').setAttribute('aria-pressed',String(on));
    document.body.style.overflow=on?'hidden':'';
  }

  $('context').addEventListener('submit',event=>{
    event.preventDefault();
    try {choose(parse($('address').value));}
    catch(e){$('address-error').textContent=e.message;$('address-error').hidden=false;}
  });
  $('model').addEventListener('change',e=>{
    state.model=catalog.model(e.target.value);
    try{validate(state.object,state.model);}
    catch {state.object={kind:'head',block:0,head:0,index:0};state.measure='head_z';}
    $('address-error').hidden=true;render();
  });
  $('source').addEventListener('change',e=>selectSource(e.target.value));
  $('site').addEventListener('change',e=>{state.site=Number(e.target.value);render();});
  $('block').addEventListener('change',e=>choose({...state.object,block:Number(e.target.value)}));
  $('measure').addEventListener('change',e=>{state.measure=e.target.value;render();});
  $('fullscreen').onclick=()=>fullscreen(!state.fullscreen);
  $('zoom-in').onclick=()=>scene.zoom(1.12);$('zoom-out').onclick=()=>scene.zoom(1/1.12);$('zoom-reset').onclick=()=>scene.reset();
  $('jump').onclick=()=>{state.tab='plugin';render();$('plugin-panel').scrollIntoView({block:'start'});};
  $('all-heads').onclick=()=>{state.expanded=!state.expanded;renderHeadList();};
  document.addEventListener('input',e=>{if(e.target.id==='local-input')state.input=e.target.value;});
  document.addEventListener('keydown',e=>{if(e.key==='Escape' && state.fullscreen) fullscreen(false);});
  document.addEventListener('click',event=>{
    const b=event.target.closest('button');
    if(!b)return;
    const {kind,head,tab,mode,archive}=b.dataset;
    if(kind){
      const previous=state.object.kind==='hidden'?Math.max(0,state.object.block-1):state.object.block;
      choose({kind,block:kind==='hidden'?previous+1:previous,head:0,index:0});
    }
    if(head!=null){choose({...state.object,head:Number(head)});$('head-list').querySelector('[aria-pressed="true"]')?.focus({preventScroll:true});}
    if(tab){state.tab=tab;render();}
    if(mode){state.mode=mode;state.tab={live:'run',records:'record',features:'plugin'}[mode];render();}
    if(archive)bindArchive(archive);
  });
  render();
})();

