'use strict';

(() => {
  const memory = {
    origin:'history', draft:'', byModel:new Map(), tool:'native', measure:'', coordinate:0,
    position:0, source:0, hiddenLayer:null, lastRequest:undefined, lastRecord:null, queryToken:'', sourceToken:'',
    plugins:new Set(), managerOpen:false, geometryLayer:null, topGroups:2, dictionaryFeature:0
  };
  const positionLabels = ['输入末尾','最后一步前','Value 前','Code 前'];
  const headMeasures = [
    ['attention','注意力分配'],['q','Q 查询向量'],['k','K 键向量'],['v','V 值向量'],['write','O 投影写入']
  ];
  const neuronMeasures = [
    ['post_gate','门控后的激活'],['gate','gate 预激活'],['up','up 投影'],['down','down 投影写入']
  ];
  const toolOptions = [
    ['native','原生读数'],['hidden','HiddenState · 层间几何'],['projection','差向量投影 · 位置比较'],
    ['logit','Logit Lens · 词表读出'],['tuned','Tuned Lens · 校准读出']
  ];
  const extraTools = [['geometry','表征几何 · CKA / RSA'],['attn-stats','注意力分布'],['sae','SAE 特征字典']];
  const code = state => 'B'+state.block+'/'+(state.kind==='Head'?'H':'U')+state.index;
  let nextField;
  let hiddenValues;
  let neuronValues;
  let neuronSlices=[];

  function decodeBF16(value,shape) {
    const bytes = Uint8Array.from(atob(value),c=>c.charCodeAt(0));
    if(bytes.byteLength !== 2*shape.reduce((total,n)=>total*n,1))return null;
    const raw = new DataView(bytes.buffer);
    const values = new Float32Array(bytes.length/2);
    const bits = new Uint32Array(values.buffer);
    for(let i=0;i<bits.length;i++)bits[i]=raw.getUint16(i*2,true)<<16;
    return values;
  }

  function selection(api) {
    const state = api.state;
    const rows = api.records.filter(r=>r.model===state.model && !r.scope.includes('汇总'));
    if(memory.lastRequest!==state.liveRequest) {
      memory.lastRequest=state.liveRequest;
      if(rows.some(r=>r.id===state.recordId)) {
        memory.origin='history'; memory.byModel.set(state.model,state.recordId); memory.position=0; memory.source=0;
      }
    }
    const current = rows.find(r=>r.id===memory.byModel.get(state.model)) ||
      rows.find(r=>r.id===state.recordId) ||
      rows.find(r=>r.objects.some(o=>o.kind===state.kind && o.block===state.block && o.index===state.index)) ||
      rows[0];
    return {rows, record:memory.origin==='history'?current:null};
  }

  function fieldFor(record) {
    const field = window.explorerSystemField;
    if(!record || !field || record.run!==field.run || record.caseId!==field.caseId || record.snapshot!==field.snapshot)return null;
    if(nextField!==field) {nextField=field; hiddenValues=null; neuronValues=null;neuronSlices=[];}
    return field;
  }

  function ffnSlice(record,position) {
    const field=fieldFor(record);
    if(!field || !Number.isInteger(position)||position<0||position>=field.mlpShape[0])return null;
    neuronValues ||= decodeBF16(field.mlp,field.mlpShape);
    if(!neuronValues)return null;
    const width=field.mlpShape[1];
    return neuronSlices[position] ||= neuronValues.subarray(position*width,(position+1)*width);
  }

  function headCase(record) {
    const data = window.modelExplorerData;
    return record && data && record.run===data.run && record.snapshot===data.snapshot &&
      data.cases.find(item=>item.id===record.caseId);
  }

  function options(items,selected,esc) {
    return items.map(([value,label])=>'<option value="'+esc(value)+'"'+(String(value)===String(selected)?' selected':'')+'>'+esc(label)+'</option>').join('');
  }

  function fieldSelect(id,label,choices,value,esc) {
    return '<label class="lv-control">'+esc(label)+'<select id="lv-'+id+'">'+options(choices,value,esc)+'</select></label>';
  }

  function rowForSource(head,index) {
    if(index===head.rows.length) {
      return {label:'其他 token 合计',mass:Math.max(0,1-head.rows.reduce((n,row)=>n+row.mass,0))};
    }
    const row=head.rows[index];
    return {label:'第 '+(index+1)+' 行 · '+row.name,mass:row.mass};
  }

  function hiddenMetric(field,layer,position) {
    hiddenValues ||= decodeBF16(field.hidden,field.hiddenShape);
    if(!hiddenValues)return null;
    const [,sites,width] = field.hiddenShape;
    const start = (layer*sites+position)*width;
    const previous = ((layer-1)*sites+position)*width;
    let norm=0, priorNorm=0, dot=0;
    for(let i=0;i<width;i++) {
      const x=hiddenValues[start+i];
      norm+=x*x;
      if(layer>0) {
        const y=hiddenValues[previous+i];
        priorNorm+=y*y; dot+=x*y;
      }
    }
    return {rms:Math.sqrt(norm/width),cos:layer>0 && norm && priorNorm?dot/Math.sqrt(norm*priorNorm):null};
  }

  function positionProjection(field,layer,position) {
    hiddenValues ||= decodeBF16(field.hidden,field.hiddenShape);
    if(!hiddenValues)return null;
    const [,sites,width]=field.hiddenShape;
    const base=layer*sites*width;
    const end=base+(sites-1)*width;
    const selected=base+position*width;
    let numerator=0,denominator=0;
    for(let i=0;i<width;i++){
      const direction=hiddenValues[end+i]-hiddenValues[base+i];
      numerator+=(hiddenValues[selected+i]-hiddenValues[base+i])*direction;
      denominator+=direction*direction;
    }
    return denominator>0?numerator/denominator:null;
  }

  function pending(message) {
    return '<div class="lv-pending"><strong>待采集</strong><p>'+message+'</p></div>';
  }

  function nativeHead(state,record,head,esc) {
    const aligned = head && state.kind==='Head' && state.block===26 && state.index===36;
    if(!aligned)return pending('本输入在 '+esc(code(state))+' 的所选读数尚未随稿收录。');
    if(memory.measure==='attention') {
      const rows = head.rows.map((row,i)=>[i,'第 '+(i+1)+' 行 · '+row.name]);
      rows.push([head.rows.length,'其他 token 合计']);
      const selectedRow = rowForSource(head,memory.source);
      const bars = head.rows.map((r,i)=>{
        const row=rowForSource(head,i);
        return '<button class="lv-bar-row" data-lv-row="'+i+'" aria-pressed="'+(memory.source===i)+'"><span>'+esc(row.label)+'</span><span class="lv-track"><i style="width:'+
          (row.mass*100).toFixed(4)+'%"></i></span><b>'+ (row.mass*100).toFixed(1)+'%</b></button>';
      }).join('');
      return fieldSelect('source','来源范围',rows,memory.source,esc)+
        '<div class="lv-readout"><div><small>来源范围内的注意力总和</small><output id="lv-value">'+(100*selectedRow.mass).toFixed(3)+'%</output></div><small>查询位置 · 最后输入 token</small></div>'+
        '<div class="lv-bars" role="group" aria-label="五行记录的注意力">'+bars+'</div>';
    }
    return pending('这条归档保留了五行记录的注意力总和。所选向量或写入值需要单独采集。');
  }

  function nativeUnit(state,record,field,esc) {
    if(field && field.block===state.block && memory.measure==='post_gate' && state.index>=0 && state.index<field.mlpShape[1]) {
      const values=ffnSlice(record,memory.position);
      if(values) {
        const value = values[state.index];
        const site = field.positions[memory.position];
        return '<div class="lv-readout"><div><small>FFN 中间单元激活</small><output id="lv-value">'+esc(value)+'</output></div>'+
          '<small>token '+site.index+' · ID '+site.tokenId+'<br>SiLU(gate) × up · BF16</small></div><p class="lv-footnote">该位置已采集 B'+field.block+' 的全部 '+field.mlpShape[1].toLocaleString('en-US')+' 个单元，可在上方按编号切换。</p>';
      }
    }
    if(record && memory.measure==='gate' && record.model==='Qwen3-4B' && record.scope.includes('句对') &&
      record.objects.some(o=>o.kind===state.kind && o.block===state.block && o.index===state.index)) {
      const cells=record.metrics.map(m=>'<div><small>'+esc(m.label)+'</small><strong>'+esc(m.value)+'</strong></div>').join('');
      return '<div class="lv-archived-metrics">'+cells+'</div><p class="lv-footnote">历史句对保存了逐句 gate 峰值的语料标准分，单 token 的原始 gate 值待采集。</p>';
    }
    return pending('所选输入、层与 FFN 读出边界没有可回放的单元值。');
  }

  function render(host,api) {
    const {state,esc}=api;
    const {rows,record}=selection(api);
    const field=fieldFor(record);
    const head=headCase(record);
    const text=memory.origin==='custom'?memory.draft:record?(head?head.prompt:record.question):'';
    const object=code(state);
    const activeTools=[...toolOptions,...extraTools.filter(([id])=>memory.plugins.has(id))];
    const measures=state.kind==='Head'?headMeasures:neuronMeasures;
    if(record && memory.lastRecord!==record.id){
      memory.lastRecord=record.id;
      if(record.model==='Qwen3-4B' && record.scope.includes('句对'))memory.measure='gate';
    }
    if(!measures.some(([id])=>id===memory.measure))memory.measure=measures[0][0];
    if(!field || memory.position>=field.positions.length)memory.position=0;
    if(!head || memory.source>head.rows.length)memory.source=0;
    const hiddenLayer=memory.hiddenLayer??state.block+1;
    const compareLayer=memory.geometryLayer??Math.max(0,hiddenLayer-1);
    const coordKind=memory.measure==='write'||memory.measure==='down'?'残差流坐标':'头内通道';
    // Both archived Qwen3 configurations have 128 components per attention head.
    const componentCount=coordKind==='头内通道'?128:state.model==='Qwen3-14B'?5120:2560;
    if(memory.coordinate>=componentCount)memory.coordinate=0;
    const needsChannel=memory.tool==='native' && (state.kind==='Head' && memory.measure!=='attention' || memory.measure==='down');
    const controls=[];
    if(memory.tool==='native')controls.push(fieldSelect('measure','测量量',measures,memory.measure,esc));
    if(['hidden','projection','geometry'].includes(memory.tool) && field) {
      const layers=Array.from({length:field.hiddenShape[0]},(_,i)=>[i,i===0?'hs[0] · 嵌入层':'hs['+i+'] · B'+(i-1)+' 后']);
      controls.push(fieldSelect('hs-layer',memory.tool==='geometry'?'状态层 A':'状态层',layers,Math.min(hiddenLayer,layers.length-1),esc));
      if(memory.tool==='geometry')controls.push(fieldSelect('compare-layer','状态层 B',layers,Math.min(compareLayer,layers.length-1),esc));
    }
    const sites=field?field.positions.map((p,i)=>[i,positionLabels[i]+' · token '+p.index]):[];
    if(sites.length && !['geometry','attn-stats','sae'].includes(memory.tool))controls.push(fieldSelect('position','读出位置',sites,memory.position,esc));
    const requestToken=(id,label,value,hint)=>'<label class="lv-control">'+label+'<input type="number" min="0" step="1" id="lv-'+id+'" value="'+value+'" placeholder="'+hint+'"></label>';
    const requestSites=!field && !head;
    if(requestSites && !['geometry','sae'].includes(memory.tool))controls.push(requestToken('query-token','读出 token 编号',memory.queryToken,'默认最后输入 token'));
    const needsSource=memory.tool==='native' && state.kind==='Head' &&
      (['k','v'].includes(memory.measure) || memory.measure==='attention' && !head);
    if(needsSource)controls.push(requestToken('source-token','来源 token 编号',memory.sourceToken,'默认所有可见位置'));
    if(needsChannel)controls.push('<label class="lv-control">'+coordKind+'（0–'+(componentCount-1)+'）<input type="number" id="lv-channel" required min="0" max="'+(componentCount-1)+'" step="1" value="'+memory.coordinate+'"></label>');
    let result;
    if(memory.tool==='geometry'){
      if(field)hiddenValues ||= decodeBF16(field.hidden,field.hiddenShape);
      result=field && hiddenValues ? window.ExplorerPluginViews.geometry({field,values:hiddenValues,layerA:hiddenLayer,layerB:compareLayer,labels:positionLabels,esc}):
        pending('需同一批输入或位置在两个状态层上的完整 HiddenState，并核对位置顺序。已有原场可选择「数步后读代码 · U2429 的四次读数」。');
    }
    if(memory.tool==='attn-stats'){
      const aligned=head && state.kind==='Head' && state.block===26 && state.index===36;
      if(aligned)controls.push(fieldSelect('top-groups','累计来源数',Array.from({length:head.rows.length+1},(_,i)=>[i+1,'前 '+(i+1)+' 组']),memory.topGroups,esc));
      result=aligned?window.ExplorerPluginViews.attention({head,topK:memory.topGroups,esc}):
        pending('需所选 Head 在同一查询位置的完整来源分布。Orion 档案的 Phase 2983 单题可回放 B26/H36 的分组质量。');
    }
    const dictionary=window.AI2050SparseDictionary;
    const sameSnapshot=dictionary && state.model===dictionary.model && state.snapshot===dictionary.snapshot;
    const archiveLayer=sameSnapshot && dictionary.layers.find(layer=>layer.block===state.block);
    if(memory.tool==='sae'){
      result=archiveLayer?window.ExplorerPluginViews.dictionary({archive:dictionary,layer:archiveLayer,featureIndex:memory.dictionaryFeature,esc}):
        '<p class="lv-footnote">当前模型快照与块没有配套 SAE 字典。已有 Qwen3-4B 的归档可供切换。</p>';
      if(dictionary)result+='<div class="lv-actions" aria-label="转到已归档 SAE 字典">'+dictionary.blocks.map(b=>
        '<button class="lv-action" data-lv-dictionary-block="'+b+'" '+(sameSnapshot&&state.block===b?'disabled':'')+'>定位 '+esc(dictionary.model)+' · B'+b+'</button>').join('')+'</div>';
    }
    if(memory.tool==='native')result=state.kind==='Head'?nativeHead(state,record,head,esc):nativeUnit(state,record,field,esc);
    if(memory.tool==='hidden') {
      const metrics=field && hiddenMetric(field,Math.min(hiddenLayer,field.hiddenShape[0]-1),memory.position);
      result=metrics?'<div class="lv-archived-metrics"><div><small>状态向量 RMS</small><strong id="lv-value">'+metrics.rms.toFixed(5)+'</strong></div><div><small>与上一层的余弦</small><strong>'+
        (metrics.cos===null?'无前一层':metrics.cos.toFixed(5))+'</strong></div></div><p class="lv-footnote">同一 token 的 '+field.hiddenShape[2].toLocaleString('en-US')+' 个残差坐标。以上数值由归档原场在浏览器中计算。</p>':
        pending('需保存同一输入位置的跨层 HiddenState 原场，才能计算 RMS 和相邻层余弦。');
    }
    if(memory.tool==='logit')result=pending('需同快照的层状态、最终归一化和词表读出权重，才能比较各层的候选词与分差。');
    if(memory.tool==='tuned')result=pending('需同快照、对应层的已校准映射，以及最终归一化和词表读出权重；还需在独立材料上核对校准误差。');
    if(memory.tool==='projection') {
      const value=field?positionProjection(field,Math.min(hiddenLayer,field.hiddenShape[0]-1),memory.position):null;
      result=value===null?pending('需同一层四处 HiddenState 和非零的参考方向，才能计算位置投影。'):
        '<div class="lv-readout"><div><small>相对位置投影系数</small><output id="lv-value">'+value.toFixed(5)+'</output></div><small>输入末尾 = 0<br>Code 前 = 1</small></div><p class="lv-footnote">在所选层，以输入末尾到 Code 前的状态差定义方向。系数衡量其余位置沿此方向移动了多远。</p>';
    }
    const status=memory.tool==='sae'?'模型与块的字典档案':(field||head)?'历史实测回放':(record?'历史材料':'自定义输入');
    const choice=fieldSelect('record','已测试题目',rows.map(r=>[r.id,r.title]),record?.id,esc);
    const input=memory.origin==='custom'?'<label class="lv-control">输入文本<textarea id="lv-text" rows="4" placeholder="输入要送入模型的完整文本…">'+esc(memory.draft)+'</textarea></label>':
      choice+(record?'<blockquote class="lv-question">'+esc(record.question)+'</blockquote>'+(head?'<details class="lv-prompt"><summary>查看完整模型输入</summary><pre>'+esc(text)+'</pre></details>':'<p class="lv-footnote">'+esc(record.promptType)+'</p>'):'<p class="lv-footnote">此模型尚未收录历史输入。</p>');
    const tokenHint=field?'有 '+field.positions.length+' 个带编号的读出位置':head?'已存最后输入 token 的注意力':'按快照分词后可选择 token';
    host.innerHTML='<div class="lv-panel"><section class="lv-input-block"><div class="lv-top"><h3>输入文本</h3><div class="lv-origin" role="group" aria-label="输入来源">'+
      ['history','custom'].map((id,i)=>'<button id="lv-'+id+'" aria-pressed="'+(memory.origin===id)+'">'+['历史题目','自定义'][i]+'</button>').join('')+'</div></div>'+
      input+'<div class="lv-input-foot"><span>'+esc(tokenHint)+'</span><button class="lv-action" id="lv-config" '+(!text.trim()?'disabled':'')+'>配置采集</button></div></section>'+
      '<section class="lv-instrument"><div class="lv-top"><h3>读数与分析</h3><span class="lv-status">'+status+'</span></div>'+
      '<details id="lv-manager" class="lv-plugin-manager" '+(memory.managerOpen?'open':'')+'><summary>管理分析插件</summary>'+
      extraTools.map(([id,label])=>'<label><input type="checkbox" id="lv-enable-'+id+'" '+(memory.plugins.has(id)?'checked':'')+'>'+esc(label)+'</label>').join('')+'</details>'+
      '<div class="lv-controls">'+fieldSelect('tool','分析工具',activeTools,memory.tool,esc)+controls.join('')+'</div>'+
      '<div class="lv-result" aria-live="polite">'+result+'</div><div class="lv-actions"><button id="lv-stage" class="lv-action" '+(memory.tool==='sae'?'hidden':!text.trim()?'disabled':'')+'>准备补测</button>'+
      '<button id="lv-open-feature" class="lv-link">对象特征 ↗</button>'+(record?'<button id="lv-open-record" class="lv-link">历史证据 ↗</button>':'')+'</div></section></div>';
    const el = name=>host.querySelector('#lv-'+name);
    const rerender = focus=>{
      render(host,api);
      if(focus)el(focus)?.focus({preventScroll:true});
    };
    el('manager').ontoggle=()=>{memory.managerOpen=el('manager').open;};
    for(const [id] of extraTools)el('enable-'+id).onchange=e=>{
      memory.managerOpen=true;
      if(e.target.checked)memory.plugins.add(id);
      else {memory.plugins.delete(id);if(memory.tool===id)memory.tool='native';}
      rerender('enable-'+id);
    };
    if(el('dictionary-feature'))el('dictionary-feature').onchange=e=>{
      memory.dictionaryFeature=Number(e.target.value);rerender('dictionary-feature');
    };
    for(const btn of host.querySelectorAll('[data-lv-dictionary-block]'))btn.onclick=()=>{
      const block=Number(btn.dataset.lvDictionaryBlock);
      const target=api.records.find(r=>r.model===dictionary.model && r.snapshot===dictionary.snapshot);
      if(!target)return;
      const kind=state.kind, limit=kind==='Head'?target.dims.heads:target.dims.width;
      memory.dictionaryFeature=0;
      api.selectObject({model:target.model,block,kind,index:Math.min(state.index,limit-1)});
    };
    el('history').onclick=()=>{memory.origin='history'; rerender('history');};
    el('custom').onclick=()=>{memory.origin='custom'; rerender('text');};
    if(el('record'))el('record').onchange=e=>{
      const next=rows.find(r=>r.id===e.target.value);
      if(!next)return;
      memory.byModel.set(state.model,next.id); memory.position=0; memory.source=0; memory.hiddenLayer=null;
      const obj=next.objects.find(o=>o.kind===state.kind && o.block===state.block && o.index===state.index) || next.objects[0];
      if(next.model==='Qwen3-4B' && next.scope.includes('句对'))memory.measure='gate';
      api.selectObject({...obj,recordId:next.id});
      rerender('record');
    };
    if(el('text'))el('text').oninput=e=>{
      memory.draft=e.target.value;
      el('config').disabled=el('stage').disabled=!memory.draft.trim();
    };
    for(const key of ['tool','measure','position','source','hs-layer','compare-layer','top-groups']) {
      if(el(key))el(key).onchange=e=>{
        if(key==='hs-layer')memory.hiddenLayer=Number(e.target.value);
        else if(key==='compare-layer')memory.geometryLayer=Number(e.target.value);
        else if(key==='top-groups')memory.topGroups=Number(e.target.value);
        else memory[key]=key==='position'||key==='source'?Number(e.target.value):e.target.value;
        rerender(key);
      };
    }
    if(el('channel'))el('channel').onchange=e=>{
      if(e.target.validity.valid)memory.coordinate=e.target.valueAsNumber;
    };
    for(const [id,key] of [['query-token','queryToken'],['source-token','sourceToken']]){
      if(el(id))el(id).oninput=e=>{
        const v=e.target.value;
        memory[key]=/^\d+$/.test(v)&&Number.isSafeInteger(Number(v))?v:'';
        if(memory[key]!==v)e.target.value=memory[key];
      };
    }
    for(const btn of host.querySelectorAll('[data-lv-row]'))btn.onclick=()=>{
      memory.source=Number(btn.dataset.lvRow);
      rerender('source');
    };
    function stage(collect) {
      if(el('channel')&&!el('channel').reportValidity())return;
      const inputText=memory.origin==='custom'?memory.draft:text;
      if(!inputText.trim())return;
      const tool=activeTools.find(([id])=>id===memory.tool)[1];
      const measure=measures.find(([id])=>id===memory.measure)[1];
      const position=memory.tool==='geometry'?(field?field.positions.map((p,i)=>positionLabels[i]+' · '+p.index).join('、'):'待对齐的一组输入或位置'):field?positionLabels[memory.position]+' · token '+field.positions[memory.position].index:
        head || memory.queryToken===''?'最后输入 token':'token '+memory.queryToken+'（分词后校验）';
      let source='';
      if(head && memory.measure==='attention' && memory.tool==='native')source=rowForSource(head,memory.source).label;
      if(memory.tool==='attn-stats')source=head?'相同查询位置的五行正文与其余 token':'同一查询位置的完整来源分布';
      if(needsSource)source=memory.sourceToken===''?'所有可见位置':'token '+memory.sourceToken+'（分词后校验）';
      const requirements={native:measure,hidden:'同一 token 的相邻层 HiddenState',projection:'同层参考位置和所选位置的 HiddenState',logit:'层状态、最终归一化、词表读出权重',tuned:'与模型及层匹配的校准映射和词表读出权重',
        geometry:'hs['+hiddenLayer+'] 与 hs['+compareLayer+'] 在同组观察位置的完整残差向量',
        'attn-stats':'保存有序来源与分组规则，计算注意力熵、有效来源数和前 '+memory.topGroups+' 项质量',
        sae:'采集输入在指定层的原生状态并核对同快照的 SAE 编码器与读出边界'};
      const needs=['同快照分词和读出位置',requirements[memory.tool]];
      const materialScope=memory.origin==='custom'?'自定义完整文本':head?'完整历史模型输入':record.promptType;
      const isPair=record?.scope.includes('句对');
      if(isPair)needs.push('按 A、B 两条独立输入复现，沿用原语料标准化口径');
      else if(record && !head)needs.push('取回同一快照下的完整编码输入、模板与生成上下文');
      if(state.kind==='Head' && ['k','v'].includes(memory.measure) && memory.tool==='native')needs.push('该查询头到共享 KV 头的映射');
      const detail=memory.tool==='hidden'||memory.tool==='projection'?'hs['+hiddenLayer+']':memory.tool==='native'?measure:tool;
      api.stage({title:(collect?'采集输入':'补测 '+detail)+' · '+object,needs,
        model:state.model,object:{kind:state.kind,block:state.block,index:state.index},recordId:record?.id||null,
        inputText:memory.origin==='custom'||head?inputText:null,
        materials:record?.question||null,materialScope,pairedInputs:isPair?record.question.split('\n').map(s=>s.replace(/^[AB]\s+/,'')):null,
        inputSource:memory.origin,tool,measure:memory.tool==='native'?measure:null,
        hiddenLayer:memory.tool==='hidden'||memory.tool==='projection'?hiddenLayer:null,
        position,source,coordinate:needsChannel?memory.coordinate:null});
    }
    el('config').onclick=()=>stage(true);
    el('stage').onclick=()=>stage(false);
    el('open-feature').onclick=()=>api.openFeatures();
    if(record)el('open-record').onclick=()=>api.openRecord(record.id);
  }
  window.ExplorerLive={render,ffnSlice,fieldFor,
    nativeInput:api=>({record:selection(api).record,origin:memory.origin,position:memory.position}),
    useReadout:position=>{memory.position=position;}
  };
})();
