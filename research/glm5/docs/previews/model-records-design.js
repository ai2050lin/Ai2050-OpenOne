'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const records = window.modelRecords;
  const list = $('record-list');
  if(!Array.isArray(records)){
    $('result-count').textContent='归档摘录未加载。请从原预览目录打开此页。';
    $('filters').hidden=true;
    return;
  }
  const inputs = {q:$('question-filter'),model:$('model-filter'),kind:$('kind-filter'),index:$('object-filter'),receipt:$('receipt-filter')};
  const catalog = new Map(records.map(record=>[record.id,record]));
  const signatures = {'Qwen3-14B':{layers:40,heads:40,width:17408},'Qwen3-4B':{layers:36,heads:32,width:9728}};
  const symbols = {
    overview:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
    cube:'m12 2 9 5v10l-9 5-9-5V7z M3 7l9 5 9-5 M12 12v10',
    flask:'M9 3h6 M10 3v6L4 19q-1 2 2 2h12q3 0 2-2L14 9V3 M8 15h8',
    settings:'M4 6h16 M4 12h16 M4 18h16 M8 4v4 M16 10v4 M10 16v4',
    pulse:'M2 13h5l3-9 4 16 3-10 2 3h3',
    archive:'M3 4h18v4H3z M5 8v12h14V8 M9 12h6',
    feature:'M4 18V9 M9 21V3 M14 17V7 M19 15v-5',
    search:'M16 10A6 6 0 1 1 4 10a6 6 0 1 1 12 0 M15 15l6 6',
    expand:'M3 9V3h6 M15 3h6v6 M21 15v6h-6 M9 21H3v-6',
    database:'M3 5a9 3 0 1 0 18 0 9 3 0 1 0-18 0 M3 5v13a9 3 0 0 0 18 0V5 M3 12a9 3 0 0 0 18 0'
  };
  for(const el of document.querySelectorAll('[data-icon]')){
    el.innerHTML='<svg aria-hidden="true" viewBox="0 0 24 24"><path d="'+symbols[el.dataset.icon]+'"/></svg>';
  }
  const esc = value => String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const norm = value => String(value??'').normalize('NFKC').trim().toLowerCase();
  const coords = o => 'B'+o.block+'/'+(o.kind==='Head'?'H':'U')+o.index;
  const indexed = record => record.objects || [];
  const hasReceipt = record => record.status==='中心已读回';
  const modelNames=[...new Set(records.map(r=>r.model).filter(Boolean))].sort();
  for(const name of modelNames) inputs.model.add(new Option(name,name));
  function filters(){
    return {q:inputs.q.value,model:inputs.model.value,kind:inputs.kind.value,index:inputs.index.value,receipt:inputs.receipt.checked?'1':''};
  }
  function matchingObjects(r,f){
    const typed=norm(f.index).replace(/\s+/g,'').toUpperCase();
    return indexed(r).filter(o=>{
      if(f.kind && o.kind!==f.kind) return false;
      if(!typed) return true;
      const id=coords(o);
      return [id,id.split('/')[1],'B'+o.block,String(o.index)].includes(typed);
    });
  }
  function matches(r,f){
    if(f.model && r.model!==f.model || f.receipt && !hasReceipt(r)) return false;
    if((f.kind || f.index.trim()) && !matchingObjects(r,f).length) return false;
    const searchable=norm([r.title,r.question,r.caseId,r.run,r.prompt,'Phase '+r.phase].join(' '));
    return norm(f.q).split(/\s+/).every(word=>searchable.includes(word));
  }
  function encode(f,view='',object=''){
    const p=new URLSearchParams(Object.entries(f).filter(([,v])=>v));
    if(view)p.set('record',view);
    if(object)p.set('object',object);
    return '#'+p;
  }
  function readLocation(){
    const p=new URLSearchParams(location.hash.slice(1));
    for(const name of ['q','kind','index','model']) inputs[name].value=p.get(name)||'';
    inputs.receipt.checked=p.get('receipt')==='1';
    return {record:p.get('record'),object:p.get('object')};
  }
  const scene=window.createModelScene($('scene').querySelector('canvas'),{
    onSelect:(block,index)=>pickFromScene({kind:'Head',block,index}),
    onSelectUnit:(block,index)=>pickFromScene({kind:'MLP',block,index}),
  });
  let selected=null;
  let activeObject=null;
  let resultSet=[];
  function drawList(returnTo){
    const f=filters();
    resultSet=records.filter(r=>matches(r,f));
    $('result-count').innerHTML=resultSet.length+' 条记录 <small>'+new Set(resultSet.map(r=>r.model)).size+' 个模型</small>';
    $('clear-filters').disabled=!Object.values(f).some(Boolean);
    list.innerHTML=resultSet.map(r=>{
      const objects=matchingObjects(r,f);
      const obj=objects[0];
      const part=obj?'<span class="part '+(obj.kind==='MLP'?'mlp':'')+'">'+esc(coords(obj))+'</span>'+
        (objects.length>1?'<span class="more-objects">+'+(objects.length-1)+'</span>':''):'<span class="no-part">无对象索引</span>';
      return '<button type="button" class="ledger-grid'+(returnTo===r.id?' returned':'')+'" data-record="'+esc(r.id)+'">'+
        '<span class="title-cell"><strong class="record-name">'+esc(r.title)+'</strong><small class="mini-meta">Phase '+esc(r.phase)+' · '+esc(r.caseId||r.run||'研究记录')+(hasReceipt(r)?' · 中心已读回':'')+'</small></span>'+
        '<span class="model-cell">'+esc(r.model)+'</span><span class="object-cell">'+part+'</span>'+
        '<span class="level">'+esc(r.scope)+'</span><span class="chev" aria-hidden="true">›</span></button>';
    }).join('');
    $('zero-results').hidden=Boolean(resultSet.length);
  }
  function dims(r){return r.dims || signatures[r.model];}
  function canLocate(r,o){
    const d=dims(r);
    return d && /^[a-f\d]{64}$/i.test(r.snapshot||'') && o && ['Head','MLP'].includes(o.kind) &&
      Number.isInteger(o.block) && o.block>=0 && o.block<d.layers &&
      Number.isInteger(o.index) && o.index>=0 && o.index<(o.kind==='Head'?d.heads:d.width);
  }
  function applyObject(o){
    activeObject=o;
    $('object-picker').value=coords(o);
    scene.update({model:selected.model,...dims(selected),block:o.block,head:o.kind==='Head'?o.index:null,unit:o.kind==='MLP'?o.index:null,labels:true,grid:true});
    $('scene-position').textContent='结构示意 · '+(o.kind==='Head'?'注意力头':'MLP 中间单元');
  }
  function changeObject(o){
    applyObject(o);
    history.replaceState(history.state,'',encode(filters(),selected.id,coords(o)));
  }
  function pickFromScene(o){
    if(!selected || !activeObject)return;
    const found=indexed(selected).find(other=>coords(o)===coords(other));
    if(found && canLocate(selected,found))changeObject(found);
    else {
      applyObject(activeObject);
      $('scene-position').textContent=coords(o)+' 未收录于本记录，当前仍定位 '+coords(activeObject);
    }
  }
  function infoRow(key,value){
    return '<dt>'+esc(key)+'</dt><dd>'+esc(value||'未附')+'</dd>';
  }
  function fileRow(label,path){
    if(!path)return '';
    if(!/^(research|frontend|tests)\/[\w./-]+$/.test(path)||path.split('/').includes('..'))return infoRow(label,path);
    return '<dt>'+esc(label)+'</dt><dd><a href="../../../../'+esc(path)+'" target="_blank" rel="noopener">'+esc(path)+'</a></dd>';
  }
  function showDetail(r,wanted){
    selected=r;
    const f=filters();
    const available=indexed(r).filter(o=>canLocate(r,o));
    const relevant=matchingObjects(r,f).filter(o=>canLocate(r,o));
    const wantedObj=available.find(o=>coords(o)===wanted);
    const obj=wantedObj || relevant[0] || available[0];
    $('search-view').hidden=true;$('record-view').hidden=false;
    const within=resultSet.findIndex(a=>a.id===r.id);
    $('record-number').textContent=within<0?'历史测试':'当前结果 '+(within+1)+' / '+resultSet.length;
    $('record-meta').textContent=['Phase '+r.phase,r.model,r.scope,r.status].filter(Boolean).join(' · ');
    $('record-title').textContent=r.title;
    $('record-question').textContent=r.question;
    $('record-result').textContent=r.result;
    $('metrics').innerHTML=(r.metrics||[]).map(m=>'<div>'+infoRow(m.label,String(m.value))+'</div>').join('');
    const provenance=[['归档状态',r.status],['证据评审',r.evidenceStatus==='unreviewed'?'待审':r.evidenceStatus],['运行',r.run],['样本',r.caseId],['文件内位置',r.sourceKey],['模型快照',r.snapshot],['产物 SHA-256',r.artifactSha]];
    $('provenance').innerHTML=provenance.filter(([,value])=>value).map(([key,value])=>infoRow(key,value)).join('')+
      fileRow('资料来源',r.source)+fileRow('测试材料',r.materialSource)+fileRow('中心回执',r.receiptSource);
    $('provenance').parentElement.open=false;
    $('no-location').hidden=Boolean(obj);
    $('inspection').hidden=!obj;
    if(obj){
      $('model-name').textContent=r.model;
      $('scene-dims').textContent=dims(r).layers+' 块 · 快照 '+r.snapshot.slice(0,12);
      $('object-picker').replaceChildren(...available.map(o=>new Option(coords(o),coords(o))));
      applyObject(obj);scene.reset();
    } else {
      activeObject=null;
      $('no-location').textContent=indexed(r).length?'原生索引与模型快照尚未完整对齐，此记录只展示结果。':'这条记录没有归档到具体的 Head 或 MLP 单元，可查看测试结果与来源。';
    }
    window.scrollTo(0,0);
    $('record-title').focus({preventScroll:true});
  }
  function showList(saved={}){
    if(document.fullscreenElement===$('inspection'))document.exitFullscreen().catch(()=>{});
    markFull(false);
    selected=null;activeObject=null;
    $('record-view').hidden=true;$('search-view').hidden=false;
    drawList(saved.row);
    const row=[...list.children].find(el=>el.dataset.record===saved.row);
    list.scrollTop=saved.scroll||0;
    if(row){
      row.focus({preventScroll:true});
      const rect=row.getBoundingClientRect(),bounds=list.getBoundingClientRect();
      if(rect.bottom<bounds.top||rect.top>bounds.bottom) list.scrollTop+=rect.top-bounds.top;
    }
    window.scrollTo(0,saved.page||0);
  }
  function openRecord(id){
    const r=catalog.get(id);
    if(!r)return;
    const saved={row:id,scroll:list.scrollTop,page:window.scrollY};
    const f=filters();
    history.replaceState({saved},'',encode(f));
    history.pushState({saved,fromList:true},'',encode(f,id));
    showDetail(r);
  }
  function restore(){
    const {record,object}=readLocation();
    const saved=history.state?.saved||{};
    if(record && catalog.has(record)){
      resultSet=records.filter(r=>matches(r,filters()));
      showDetail(catalog.get(record),object);
    } else showList(saved);
  }
  $('back-to-list').onclick=()=>{
    if(history.state?.fromList){history.back();return;}
    const saved={row:selected?.id||''};
    history.replaceState({saved},'',encode(filters()));
    showList(saved);
  };
  function updateSearch(){
    history.replaceState({},'',encode(filters()));
    drawList();list.scrollTop=0;
  }
  $('filters').onsubmit=e=>e.preventDefault();
  for(const name of ['q','index']) inputs[name].oninput=updateSearch;
  for(const name of ['model','kind','receipt']) inputs[name].onchange=updateSearch;
  function clear(){
    $('filters').reset();inputs.receipt.checked=false;updateSearch();inputs.q.focus({preventScroll:true});
  }
  $('clear-filters').onclick=clear;$('zero-clear').onclick=clear;
  list.onclick=e=>{const row=e.target.closest('[data-record]');if(row)openRecord(row.dataset.record);};
  $('object-picker').onchange=e=>{
    const o=indexed(selected).find(obj=>coords(obj)===e.target.value);
    if(o && canLocate(selected,o))changeObject(o);
  };
  $('zoom-in').onclick=()=>scene.zoom(1.15);$('zoom-out').onclick=()=>scene.zoom(0.85);$('reset-view').onclick=()=>scene.reset();
  const inspect=$('inspection');
  function markFull(on){
    inspect.classList.toggle('fullspace',on);
    $('full').setAttribute('aria-label',on?'退出全屏':'全屏');
    $('full').querySelector('span').textContent=on?'退出全屏':'全屏';
  }
  $('full').onclick=async()=>{
    if(document.fullscreenElement)await document.exitFullscreen();
    else if(inspect.classList.contains('fullspace'))markFull(false);
    else {try{await inspect.requestFullscreen();}catch{markFull(true);}}
  };
  document.addEventListener('fullscreenchange',()=>markFull(document.fullscreenElement===inspect));
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!document.fullscreenElement)markFull(false);});
  if('scrollRestoration' in history)history.scrollRestoration='manual';
  window.addEventListener('popstate',restore);
  restore();
})();
