(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const data = window.FreeModelData, desk = $('free-workbench');
  const icons = {
    overview:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
    layers:'m12 3 9 5-9 5-9-5z M3 12l9 5 9-5 M3 16l9 5 9-5',
    flask:'M9 3h6 M10 3v7l-6 9q-1 2 2 2h12q3 0 2-2l-6-9V3 M8 15h8',
    sliders:'M3 7h7m5 0h6 M3 17h11m5 0h2 M12 4v6 M16 14v6',
    target:'M4 9V4h5m6 0h5v5m0 6v5h-5m-6 0H4v-5 M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0z',
    screen:'M3 9V3h6 M15 3h6v6 M21 15v6h-6 M9 21H3v-6',
    book:'M4 3h14a2 2 0 0 1 2 2v15H6a2 2 0 0 1-2-2z M4 17h16 M8 7h8 M8 10h5',
    matrix:'M6 3H3v18h3 M18 3h3v18h-3 M8 8h1m5 0h1 M8 12h1m5 0h1 M8 16h1m5 0h1',
    wave:'M2 13h4l3-9 4 16 3-11 2 4h4',math:'M19 4H6l7 8-7 8h13',
    reset:'M4 10a8 8 0 1 1 1 8 M3 4v6h6',arrow:'M5 12h14 M14 7l5 5-5 5'
  };
  for (const tag of document.querySelectorAll('[data-ico]')) {
    tag.innerHTML='<svg aria-hidden="true" viewBox="0 0 24 24"><path d="'+icons[tag.dataset.ico]+'"/></svg>';
  }
  const windows = window.FreeWindowManager.create(desk);
  const initial=new URLSearchParams(location.hash.slice(1));
  let activePanel='locator', restoring=true;
  const snapshots = new Map(), loaded = new Map(), archivePreferred = new Set();
  let model = data.model('Qwen3-14B') || data.models[0], state, tensorName, row, col, tile = null;
  let tileVersion = 0, importVersion = 0, bindingVersion = 0, headDetail='heads';
  const fmt = n => Number(n).toLocaleString('en-US');
  const current = () => data.tensor(model.id,tensorName);
  const text = (id, value) => { $(id).textContent=value; };
  function modelBlock() { return state.kind==='hidden'?Math.max(0,state.block-1):state.block; }
  function addr() {
    if(state.kind==='parameter')return tensorName+'['+row+(current()?.shape.length===2?','+col:'')+']';
    if(state.kind==='hidden')return 'hs['+state.block+']/D'+state.index;
    if(state.kind==='ffn')return 'B'+state.block+'/U'+state.index;
    return 'B'+state.block+'/H'+state.head+(state.index?'/D'+state.index:'');
  }
  function rememberAddress(){
    if(restoring||!state)return;
    const fragment=new URLSearchParams({model:model.id,at:addr(),panel:activePanel}).toString();
    if(location.hash.slice(1)===fragment)return;
    try{history.replaceState(null,'','#'+fragment);}catch{/* Embedded file viewers can disable history updates. */}
  }
  desk.addEventListener('freewindowchange',e=>{activePanel=e.detail.active||'locator';rememberAddress();});
  function validate(kind,b,i,h=0) {
    const limit=kind==='hidden'?model.layers+1:model.layers;
    if(!Number.isInteger(b)||b<0||b>=limit)throw Error('该模型的'+(kind==='hidden'?' HiddenState 编号':'块编号')+'范围为 0–'+(limit-1)+'。');
    const width={head:model.headDim,ffn:model.ffn,hidden:model.hidden}[kind];
    if(!width||!Number.isInteger(i)||i<0||i>=width)throw Error('该对象的坐标范围为 0–'+(width-1)+'。');
    if(kind==='head'&&(!Number.isInteger(h)||h<0||h>=model.heads))throw Error('该模型的 Head 编号范围为 0–'+(model.heads-1)+'。');
  }
  const vectorKey = s=>[model.id,s.kind,s.block,s.kind==='head'?s.head:''].join('|');
  function fields() {
    if(state.kind==='parameter')return [];
    return data.fields(model.id,state.kind,state.block,state.kind==='head'?state.head:state.index);
  }
  function ctx() {
    return {model:model.id,...state,unit:state.index,coord:state.index,tensor:tensorName,row,col,
      parameterTile:tile,importedFields:archivePreferred.has(vectorKey(state))?[]:(loaded.get(vectorKey(state))||[]),revision:bindingVersion};
  }
  function pick(i) {
    if(state.kind==='parameter')return;
    selectObject(state.kind,state.block,i,state.head);
  }
  const readout=window.FreeModelTools.mount($('readout-host'),{mode:'readout',onPick:pick});
  const tools=window.FreeModelTools.mount($('tools-host'),{mode:'tools',onPick:pick});
  function refreshTools() { const c=ctx();readout.update(c);tools.update(c); }
  const scene=window.createModelScene($('space-canvas'),{
    onSelect:(b,h)=>selectObject('head',b,0,h),
    onSelectChannel:i=>selectObject('head',state.block,i,state.head),
    onSelectUnit:(_b,i)=>selectObject(state.kind,state.block,i,state.head),
    onSelectParameter:(r,c)=>selectTensor(tensorName,r,c),
    onSelectBlock:b=>{
      if(state.kind==='parameter'){
        const t=current(), sibling=t.name.replace(/^model\.layers\.\d+\./,'model.layers.'+b+'.');
        if(t.block!=null && data.tensor(model.id,sibling)){selectTensor(sibling,row,col);return;}
        selectObject('head',b,0,0);return;
      }
      selectObject(state.kind,state.kind==='hidden'?b+1:b,state.index,state.head);
    }
  });
  function option(value,label) { return new Option(label,value); }
  function tensorDefault(kind,b) {
    if(kind==='hidden'&&b===0)return 'model.embed_tokens.weight';
    if(kind==='hidden')return 'lm_head.weight';
    return 'model.layers.'+b+'.'+(kind==='ffn'?(model.fusedGateUp?'mlp.gate_up_proj.weight':'mlp.gate_proj.weight'):'self_attn.q_proj.weight');
  }
  function parameterOrigin(kind,i,h){
    if(kind==='head')return [h*model.headDim+i,0];
    if(kind==='hidden')return [0,i];
    return [i,0];
  }
  function options() {
    const filter=$('tensor-filter').value.trim().toLowerCase();
    const found=model.tensors.filter(t=>t.name.toLowerCase().includes(filter));
    const select=$('tensor-select');
    select.replaceChildren(...found.map(t=>option(t.name,t.name)));
    select.value=tensorName;
    text('tensor-count',fmt(found.length)+' 个名称');
    if(!found.length)select.replaceChildren(option('','没有匹配的张量'));
    if(found.length&&!select.value){const hint=option('','选择匹配的张量');select.prepend(hint);select.value='';}
  }
  function selectObject(kind,b,i,h=0,resetParameter=true) {
    validate(kind,b,i,h);
    if(kind!=='head')headDetail='heads';
    state={kind,block:b,head:kind==='head'?h:0,index:i};
    if(resetParameter){
      tensorName=tensorDefault(kind,b);[row,col]=parameterOrigin(kind,i,h);
      $('tensor-filter').value='';options();tile=null;
    }
    text('loc-note','');render();
    if(resetParameter) readWeights();
  }
  function selectTensor(name,r=0,c=0,open=true) {
    const t=data.tensor(model.id,name);
    if(!t)throw Error('没有找到该张量。可以在「模型参数」中按名称片段检索。');
    const [height,width=1]=t.shape;
    if(!Number.isInteger(r)||!Number.isInteger(c)||r<0||c<0||r>=height||c>=width)throw Error('坐标超出张量形状 '+t.shape.join(' × ')+'。');
    tensorName=name;row=r;col=c;tile=null;
    state={kind:'parameter',block:t.block??0,head:0,index:0};
    if(!name.toLowerCase().includes($('tensor-filter').value.trim().toLowerCase()))$('tensor-filter').value='';
    options();text('loc-note','');render();readWeights();
    if(open) windows.open('parameters');
  }
  function jump(raw) {
    const s=raw.trim();
    let m=/^B(\d+)\s*\/\s*H(\d+)(?:\s*\/\s*D(\d+))?$/i.exec(s);
    if(m){selectObject('head',+m[1],+(m[3]||0),+m[2]);return;}
    m=/^B(\d+)\s*\/\s*U(\d+)$/i.exec(s);
    if(m){selectObject('ffn',+m[1],+m[2]);return;}
    m=/^hs\[(\d+)\]\s*\/\s*D(\d+)$/i.exec(s);
    if(m){selectObject('hidden',+m[1],+m[2]);return;}
    m=/^([\w.]+)(?:\[(\d+)(?:,\s*(\d+))?\])?$/.exec(s);
    if(m&&data.tensor(model.id,m[1])){selectTensor(m[1],+(m[2]||0),+(m[3]||0));return;}
    throw Error('地址格式如 B26/H36、B0/U0 或 hs[0]/D0。张量支持全名和 [行,列]。');
  }
  function button(label,fn,tip) {
    const btn=document.createElement('button');btn.type='button';btn.textContent=label;
    if(tip)btn.title=tip;btn.addEventListener('click',fn);return btn;
  }
  function related() {
    const {kind,head,index}=state;
    const base='model.layers.'+modelBlock()+'.', group=Math.floor(head/(model.heads/model.kvHeads));
    const line=$('related');line.replaceChildren();
    function link(label,name,r=0,c=0) {
      line.append(button(label,()=>selectTensor(name,r,c),name+' ['+r+', '+c+']'));
    }
    if(kind==='head'){
      const hd=head*model.headDim+index, kv=group*model.headDim+index;
      link('Q 投影',base+'self_attn.q_proj.weight',hd);link('K 投影',base+'self_attn.k_proj.weight',kv);
      link('V 投影',base+'self_attn.v_proj.weight',kv);link('O 写回',base+'self_attn.o_proj.weight',0,hd);
      if(data.tensor(model.id,base+'self_attn.q_proj.bias')){
        link('Q 偏置',base+'self_attn.q_proj.bias',hd);link('K 偏置',base+'self_attn.k_proj.bias',kv);link('V 偏置',base+'self_attn.v_proj.bias',kv);
      }
    }else if(kind==='ffn'){
      if(model.fusedGateUp){
        link('gate 输入行',base+'mlp.gate_up_proj.weight',index);link('up 输入行',base+'mlp.gate_up_proj.weight',model.ffn+index);
      }else{
        link('gate 输入行',base+'mlp.gate_proj.weight',index);link('up 输入行',base+'mlp.up_proj.weight',index);
      }
      link('down 写回列',base+'mlp.down_proj.weight',0,index);
      if(['Qwen3-14B','Qwen3-4B'].includes(model.id)){
        const profile=document.createElement('a');
        profile.className='text-btn';profile.textContent='查看 Goodfire 分析 →';
        const route=new URLSearchParams({wb:'1',lens:'spatial',view:'graph',gview:'internal',observeMode:'features',
          observeTab:'object',featureView:'goodfire',mdl:model.id,snapshot:model.snapshot,
          sblock:String(state.block+1),smodule:'mlp',sindex:String(index)});
        profile.href='client-distributed-testing.html#?'+route.toString();line.append(profile);
      }
    }else if(kind==='hidden'){
      link('词嵌入列','model.embed_tokens.weight',0,index);link('最终归一化','model.norm.weight',index);
      link('词表读出列','lm_head.weight',0,index);
    }else{line.append(button('观察当前参数',()=>windows.open('parameters')));}
  }
  function render(){
    const {kind,block,head,index}=state, target=current();
    $('channel-view').hidden=kind!=='head';
    text('channel-view',headDetail==='heads'?'展开头内 '+model.headDim+' 个通道':'返回本块全部 Head');
    $('channel-view').setAttribute('aria-pressed',String(headDetail==='channels'));
    $('head-detail').disabled=kind!=='head';$('head-detail').value=headDetail;
    const asParameter=kind==='parameter',b=modelBlock();
    $('address').value=addr();text('selection-name',addr());text('object-title',addr());
    const locKind=$('loc-kind');
    if(asParameter&&!locKind.querySelector('[value=parameter]'))locKind.add(option('parameter','原生参数 · 张量'));
    if(!asParameter)locKind.querySelector('[value=parameter]')?.remove();
    locKind.value=kind;
    text('block-label',kind==='hidden'?'HiddenState 层':'模型块');$('loc-block').value=block;
    $('loc-block').max=kind==='hidden'?model.layers:model.layers-1;$('loc-block').disabled=asParameter;
    text('index-label',kind==='head'?'Head 编号':kind==='ffn'?'FFN 单元':'坐标编号');
    $('loc-index').value=kind==='head'?head:index;$('loc-index').max=({head:model.heads,ffn:model.ffn,hidden:model.hidden}[kind]||1)-1;
    $('loc-index').disabled=asParameter;
    const dim=model.headDim,group=Math.floor(head/(model.heads/model.kvHeads));
    text('object-description',kind==='head'?'这个查询头有 '+dim+' 个原生通道，与同组查询头共享 KV'+group+'。可沿 Q/K 查看选择，沿 V/O 查看读出和写回。':
      kind==='ffn'?'本块有 '+fmt(model.ffn)+' 个前馈单元。'+(model.fusedGateUp?'U'+index+' 读取融合矩阵的第 '+index+' 行（gate）和第 '+(model.ffn+index)+' 行（up），写回 down 矩阵的第 '+index+' 列。':'单元 U'+index+' 对应 gate 和 up 投影的同号行，以及 down 投影的同号列。'):
      kind==='hidden'?'这层状态有 '+fmt(model.hidden)+' 个原生坐标。每个值对应具体材料、token 位置和采集边界。':
      '固定权重张量 · '+target.shape.join(' × ')+' · '+target.dtype+(target.alias?' · 与 '+target.alias+' 共享存储':''));
    text('model-info',model.layers+' 块  ·  '+model.heads+' 个查询头/块  ·  '+fmt(model.ffn)+' 个 FFN 单元/块');
    text('snapshot',model.id+' · '+model.snapshot);
    text('source-summary',model.id==='Qwen3-4B'?'中文木盒颜色题，输入末尾一个 token。全部 36 块的 Q、K、V、Z、FFN 和 37 个 HiddenState 读出可查。':
      model.id==='Qwen3-14B'?'五行查找题保存了 B26/H36 的三个 Z 向量。算术题保存了四个 token 位置的 B36 FFN 与逐层残差。':
      '本快照附有权重目录与原生切片。要看特定题目的激活，请导入该模型保存的向量。');
    const archive=fields();
    const sourceHost=$('source-fields');sourceHost.replaceChildren();
    if(!archive.length)sourceHost.textContent=asParameter?'当前对象为固定权重。':'当前对象尚无随稿激活。';
    const seen=new Set();
    for(const f of archive){
      const key=JSON.stringify([f.run,f.caseId,f.position]);if(seen.has(key))continue;seen.add(key);
      const details=document.createElement('details'),summary=document.createElement('summary'),body=document.createElement('pre');
      summary.textContent=f.caseId+' · token '+f.position;
      body.textContent=[f.question,f.output?'模型输出  '+f.output:'','token ID  '+f.tokenId,'运行  '+f.run,'来源  '+f.source].filter(Boolean).join('\n');
      details.append(summary,body);sourceHost.append(details);
    }
    const files=data.boundShards(model.id);
    $('clear-files').hidden=!files.length;
    $('clear-field').hidden=!(loaded.get(vectorKey(state))||[]).length;
    $('field-source-control').hidden=$('clear-field').hidden;
    $('source-switch').value=archivePreferred.has(vectorKey(state))?'archive':'import';
    $('field-example').disabled=asParameter;
    text('tensor-info',target.shape.join(' × ')+' · '+target.dtype+' · '+fmt(target.count)+' 个标量'+(target.alias?' · 共享 '+target.alias:''));
    $('weight-row').max=target.shape[0]-1;$('weight-col').max=(target.shape[1]??1)-1;
    $('weight-row').value=row;$('weight-col').value=col;
    related();
    scene.update({model:model.id,layers:model.layers,heads:model.heads,block:b,head:kind==='head'?head:null,
      unit:kind==='ffn'||kind==='hidden'?index:null,width:kind==='hidden'?model.hidden:model.ffn,
      mode:kind,stateIndex:block,channelView:kind==='head'&&headDetail==='channels',channel:index,channels:model.headDim,
      tensorLabel:asParameter?target.name:'',tensorShape:asParameter?target.shape:null,
      tensorBlock:asParameter?target.block:null,tensorRow:row,tensorCol:col});
    refreshTools();rememberAddress();
  }
  async function readWeights(){
    const ticket=++tileVersion,name=tensorName, id=model.id;
    text('weight-message','正在读取切片…');$('weight-table').replaceChildren();
    try{
      const next=await data.readTile(id,name,row,col,8,8);
      if(ticket!==tileVersion)return;
      tile=next;
      text('weight-message',next.source+' · '+next.available+' / '+(next.actualShape[0]*next.actualShape[1])+' 个标量'+
        (next.available<next.actualShape[0]*next.actualShape[1]?'。空格需在「材料与来源」接入 '+next.file:''));
      const t=$('weight-table'), header=t.createTHead().insertRow();
      function th(tr,v){const cell=document.createElement('th');cell.textContent=v;tr.append(cell);}
      th(header,'行 / 列');for(let j=0;j<next.actualShape[1];j++)th(header,String(col+j));
      const body=t.createTBody();
      next.values.forEach((values,i)=>{
        const tr=body.insertRow();th(tr,String(row+i));
        values.forEach((v,j)=>{const cell=tr.insertCell();
          cell.title='['+(row+i)+','+(col+j)+'] '+(v??'未载入');
          cell.textContent=v==null?'—':String(v);
          if(v!=null)cell.style.background=v<0?'#e9f0fb':v>0?'#e1f0ed':'#fff';
        });
      });
      const [height,width=1]=current().shape;
      $('rows-prev').disabled=row===0;$('rows-next').disabled=row+8>=height;
      $('cols-prev').disabled=col===0;$('cols-next').disabled=col+8>=width;
      if(state.kind==='parameter')refreshTools();
    }catch(err){if(ticket===tileVersion){tile=null;text('weight-message',err.message);refreshTools();}}
  }
  $('model-select').replaceChildren(...data.models.map(m=>option(m.id,m.name)));
  function chooseModel(id) {
    if(state)snapshots.set(model.id,{state:{...state},tensorName,row,col,filter:$('tensor-filter').value});
    model=data.model(id);$('model-select').value=id;headDetail='heads';
    const saved=snapshots.get(id);
    state=saved?.state||(id==='Qwen3-14B'?{kind:'head',block:26,head:36,index:0}:{kind:'head',block:0,head:0,index:0});
    tensorName=saved?.tensorName||tensorDefault(state.kind,state.block);
    [row,col]=saved?[saved.row,saved.col]:parameterOrigin(state.kind,state.index,state.head);
    tile=null;importVersion++; $('tensor-filter').value=saved?.filter||'';options();
    for (const id of ['loc-note','field-load-note','weight-load-note'])text(id,'');
    const files=data.boundShards(model.id);if(files.length)text('weight-load-note','已连接 '+files.length+' 个本机分片。');
    render();readWeights();
  }
  $('model-select').onchange=e=>chooseModel(e.target.value);
  $('jump-form').onsubmit=e=>{e.preventDefault();try{jump($('address').value);}catch(err){text('loc-note',err.message);}};
  $('loc-kind').onchange=e=>{
    const kind=e.target.value;
    if(kind==='parameter')return;
    const b=state.kind==='hidden'&&kind!=='hidden'?modelBlock():state.block;
    selectObject(kind,b,0,0);
  };
  function manual(){
    try{
      const kind=state.kind,b=$('loc-block').valueAsNumber,n=$('loc-index').valueAsNumber;
      selectObject(kind,b,kind==='head'?state.index:n,kind==='head'?n:0);
    }catch(err){text('loc-note',err.message);}
  }
  $('loc-block').onchange=manual;$('loc-index').onchange=manual;
  $('tensor-filter').oninput=options;
  $('tensor-select').onchange=e=>{if(e.target.value)selectTensor(e.target.value,0,0,false);};
  $('weight-form').onsubmit=e=>{e.preventDefault();try{selectTensor(tensorName,$('weight-row').valueAsNumber,$('weight-col').valueAsNumber,false);}catch(err){text('weight-message',err.message);}};
  for(const [id,dr,dc] of [['rows-prev',-8,0],['rows-next',8,0],['cols-prev',0,-8],['cols-next',0,8]]){
    $(id).onclick=()=>{const [height,width=1]=current().shape;
      selectTensor(tensorName,Math.min(height-1,Math.max(0,row+dr)),Math.min(width-1,Math.max(0,col+dc)),false);
    };
  }
  $('analyze-weight').onclick=()=>{if(state.kind!=='parameter')selectTensor(tensorName,row,col,false);
    tools.selectTool(current().shape.length===2?'spectrum':'summary');windows.open('tools');};
  $('weight-files').onchange=async e=>{
    const id=model.id, files=Array.from(e.target.files);e.target.value='';if(!files.length)return;
    try{
      text('weight-load-note','正在核对模型分片…');const result=await data.bindShards(id,files);
      if(model.id!==id)return;
      bindingVersion++;text('weight-load-note','已连接 '+result.files.length+' 个分片，覆盖 '+result.tensors+' 个张量名称。已核对头部指纹。');
      tile=null;render();readWeights();
    }catch(err){if(model.id===id)text('weight-load-note',err.message);}
  };
  $('clear-files').onclick=()=>{data.clearShards(model.id);bindingVersion++;tile=null;text('weight-load-note','已移除本机分片。');render();readWeights();};
  const schema='ai2050.activation-vector.v1', measures={head:['head_q','head_k','head_v','head_z'],ffn:['mlp_activation'],hidden:['hidden']};
  function normalized(record,file){
    if(!record||record.schema!==schema)throw Error('数据格式不符。可下载当前对象的格式模板。');
    if(record.model!==model.id||record.snapshot!==model.snapshot)throw Error('导入向量的模型或快照与当前选择不符。');
    const {kind,block,position,tokenId,measure,values}=record;
    if(!Object.hasOwn(measures,kind)||!measures[kind].includes(measure))throw Error('未知对象类型或测量边界。');
    const head=kind==='head'?record.head:0;
    validate(kind,block,0,head);
    if(!Array.isArray(values)||values.length!==({head:model.headDim,ffn:model.ffn,hidden:model.hidden}[kind])||
      !values.every(v=>typeof v==='number'&&Number.isFinite(v)&&Number.isFinite(Math.fround(v))))throw Error('向量长度或数值与原生对象不符，需提供全部有限数值。');
    if(!Number.isSafeInteger(position)||position<0||!Number.isSafeInteger(tokenId)||tokenId<0||tokenId>=model.vocab||
      !['run','caseId','boundary'].every(k=>typeof record[k]==='string'&&record[k].trim()))throw Error('须提供运行、题号、token 位置与 ID、采集边界。');
    const id=JSON.stringify([record.run,record.caseId,position,measure,record.boundary]);
    return {id,label:record.caseId+' · token '+position+' · '+record.boundary,model:model.id,snapshot:model.snapshot,
      kind,block,head,tokenId,position,run:record.run,caseId:record.caseId,measure,boundary:record.boundary,
      values:Float32Array.from(values),source:'用户提供 · '+file+' · 当前会话；采集来源未经独立核验'};
  }
  $('experiment-field').onchange=async e=>{
    const file=e.target.files[0],ticket=++importVersion;e.target.value='';if(!file)return;
    try{
      if(file.size>16*1024*1024)throw Error('单个向量文件上限为 16 MiB。');
      const raw=await file.text();if(ticket!==importVersion)return;
      const f=normalized(JSON.parse(raw),file.name);
      const key=vectorKey(f),previous=loaded.get(key)||[];
      loaded.set(key,[f,...previous.filter(x=>x.id!==f.id)]);
      archivePreferred.delete(key);
      selectObject(f.kind,f.block,0,f.head);
      text('field-load-note','已导入 '+f.label+'。可在激活读数与数学工具中查看。');
    }catch(err){if(ticket===importVersion)text('field-load-note','导入失败。'+err.message);}
  };
  $('source-switch').onchange=e=>{
    const key=vectorKey(state);if(e.target.value==='archive')archivePreferred.add(key);else archivePreferred.delete(key);
    render();
  };
  $('clear-field').onclick=()=>{const key=vectorKey(state);loaded.delete(key);archivePreferred.delete(key);importVersion++;text('field-load-note','已移除当前对象的用户向量。');render();};
  $('field-example').onclick=()=>{
    if(state.kind==='parameter')return;
    const {kind,block,head}=state,size={head:model.headDim,ffn:model.ffn,hidden:model.hidden}[kind];
    const payload={schema,model:model.id,snapshot:model.snapshot,kind,block,...(kind==='head'?{head}:{}),
      run:'',caseId:'',position:null,tokenId:null,measure:measures[kind][kind==='head'?3:0],boundary:'',values:[],
      instructions:'填入实际运行及采集边界、token 位置与 ID，并填入 '+size+' 个原生数值。'};
    const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='activation-vector-template.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
  };
  $('zoom-in').onclick=()=>scene.zoom(1.12);$('zoom-out').onclick=()=>scene.zoom(1/1.12);
  $('camera-reset').onclick=()=>scene.reset();
  function display(){scene.update({grid:$('show-grid').checked,labels:$('show-labels').checked});}
  $('show-grid').onchange=display;$('show-labels').onchange=display;
  $('channel-view').onclick=()=>{headDetail=headDetail==='heads'?'channels':'heads';render();};
  $('head-detail').onchange=e=>{headDetail=e.target.value;render();};
  const palettes=Array.from(desk.querySelectorAll('.palette-grip'));
  function resetPalettes() {
    for(const g of palettes)for(const key of ['left','top','bottom','transform'])g.parentElement.style.removeProperty(key);
  }
  function placePalette(el,x,y){
    el.style.transform='none';el.style.bottom='auto';
    el.style.left=Math.max(0,Math.min(x,desk.clientWidth-el.offsetWidth))+'px';
    el.style.top=Math.max(0,Math.min(y,desk.clientHeight-el.offsetHeight))+'px';
  }
  for(const grip of palettes){
    const el=grip.parentElement;
    let drag;
    grip.onpointerdown=e=>{
      if(e.button!==0)return;
      const box=el.getBoundingClientRect();drag={dx:e.clientX-box.left,dy:e.clientY-box.top};
      grip.setPointerCapture(e.pointerId);e.preventDefault();
    };
    grip.onpointermove=e=>{if(drag)placePalette(el,e.clientX-drag.dx,e.clientY-drag.dy);};
    grip.onlostpointercapture=()=>{drag=null;};
    grip.onkeydown=e=>{
      const direction={ArrowRight:[12,0],ArrowLeft:[-12,0],ArrowDown:[0,12],ArrowUp:[0,-12]}[e.key];
      if(!direction)return;
      e.preventDefault();const box=el.getBoundingClientRect();placePalette(el,box.left+direction[0],box.top+direction[1]);
    };
  }
  window.addEventListener('resize',()=>{if(desk.classList.contains('fw-floating'))for(const g of palettes){
    const el=g.parentElement;if(el.style.top){const rect=el.getBoundingClientRect();placePalette(el,rect.left,rect.top);}
  }});
  $('layout-reset').onclick=()=>{windows.reset();resetPalettes();};
  let fallback=false;
  function full(on){
    if(!on)resetPalettes();
    windows.setFullscreen(on);$('view-screen').querySelector('span').textContent=on?'退出全屏':'全屏观察';
    $('view-screen').setAttribute('aria-pressed',String(on));
  }
  $('view-screen').onclick=async()=>{
    if(fallback){fallback=false;full(false);return;}
    if(document.fullscreenElement===desk){await document.exitFullscreen();return;}
    try{await desk.requestFullscreen();}catch{fallback=true;full(true);}
  };
  document.addEventListener('fullscreenchange',()=>full(document.fullscreenElement===desk));
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&fallback){fallback=false;full(false);}});
  chooseModel(data.model(initial.get('model'))?.id||model.id);
  if(initial.has('at')){try{jump(initial.get('at'));}catch(err){text('loc-note',err.message);}}
  if(initial.has('panel'))windows.open(initial.get('panel'));
  restoring=false;rememberAddress();
})();

