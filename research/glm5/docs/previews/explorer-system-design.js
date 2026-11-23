'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const records = window.modelRecords;
  const models = Object.fromEntries(records.map(r => [r.model, { ...r.dims, snapshot:r.snapshot }]));
  const initialRecord = records.find(r => r.id === 'p2933-counted0-u2429');
  const state = {mode:'live', model:initialRecord.model, ...initialRecord.objects[0], recordId:initialRecord.id, liveRequest:0, archiveId:null};
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const coord = obj => 'B'+obj.block+'/'+(obj.kind==='Head'?'H':'U')+obj.index;
  const symbols = {
    cube:'m12 2 9 5v10l-9 5-9-5V7z M3 7l9 5 9-5 M12 12v10', overview:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
    tests:'M9 3h6 M10 3v6L4 19q-1 2 2 2h12q3 0 2-2L14 9V3 M8 15h8', settings:'M3 5h18 M3 12h18 M3 19h18 M8 3v4 M16 10v4 M9 17v4',
    plan:'M7 4H4v17h16V4h-3 M8 2h8v4H8z M8 11h8 M8 15h6', archive:'M3 4h18v4H3z M5 8v12h14V8 M9 12h6',
    feature:'M4 17v-7 M9 21V3 M14 16V8 M19 18V5', pulse:'M2 13h5l3-9 4 16 3-10 2 3h3',
    expand:'M3 9V3h6 M15 3h6v6 M21 15v6h-6 M9 21H3v-6'
  };
  for (const el of document.querySelectorAll('i[data-icon]')) {
    el.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="'+symbols[el.dataset.icon]+'"/></svg>';
  }
  function valid(obj,model){
    const d=models[model];
    return d && ['MLP','Head'].includes(obj.kind) && Number.isInteger(obj.block) && obj.block>=0 && obj.block<d.layers &&
      Number.isInteger(obj.index) && obj.index>=0 && obj.index<(obj.kind==='Head'?d.heads:d.width);
  }
  const scene=window.createModelScene($('model-canvas'),{
    onSelect:(block,index)=>selectObject({kind:'Head',block,index}),
    onSelectUnit:(block,index)=>selectObject({kind:'MLP',block,index})
  });
  function record(id){return records.find(r=>r.id===id);}
  function linked(r){return (r?.objects || []).filter(o=>r.snapshot===models[r.model]?.snapshot && valid(o,r.model));}
  function syncScene(overrideIndex){
    const d=models[state.model];
    state.snapshot=d.snapshot;
    const noLink=state.mode==='archive' && state.archiveId && !linked(record(state.archiveId)).length;
    const selected=overrideIndex!=null?{...state,index:overrideIndex}:state;
    $('current-object').textContent=noLink?'无对象索引':coord(selected);
    $('model-select').value=state.model;
    $('coord-input').value=noLink?'':coord(selected);
    $('coord-info').classList.remove('error');
    $('coord-info').textContent=noLink?'此题仅有整模型结果。':'B0–'+(d.layers-1)+' · H0–'+(d.heads-1)+' · U0–'+(d.width-1);
    scene.update({model:state.model,...d,block:state.block,head:!noLink&&state.kind==='Head'?selected.index:null,
      unit:!noLink&&state.kind==='MLP'?selected.index:null,grid:$('grid').checked,labels:$('labels').checked});
  }
  function render(){
    syncScene();
    for(const name of ['live','archive','features']){
      $(name+'-panel').hidden=state.mode!==name;
      $('mode-'+name).setAttribute('aria-pressed',String(state.mode===name));
    }
    if(state.mode==='live')window.ExplorerLive.render($('live-panel'),api);
    if(state.mode==='features')window.ExplorerFeatures.render($('features-panel'),api);
    if(state.mode==='archive')window.ExplorerArchive.render($('archive-panel'),api);
    if(nativeVisible)native.render();
  }
  function selectObject(change){
    const next={...state,...change};
    if(!valid(next,next.model))return false;
    if(next.model!==state.model && !Object.hasOwn(change,'recordId'))next.recordId=null;
    if(state.mode==='archive' && state.archiveId && !linked(record(state.archiveId)).some(o=>next.model===record(state.archiveId).model&&coord(o)===coord(next))){
      next.archiveId=null;
      if(history.state?.archiveDetail)history.replaceState({},'',location.pathname+location.search);
    }
    Object.assign(state,next);
    native.resetSelection();
    render();
    return true;
  }
  function setMode(mode){
    state.mode=mode;
    render();
  }
  function bindRecord(r, preferred){
    const objects=linked(r);
    const previousMatch=r.model===state.model && objects.find(o=>coord(o)===coord(state));
    const target=objects.find(o=>preferred && coord(o)===coord(preferred)) || previousMatch || objects[0];
    if(r.model!==state.model && !target)Object.assign(state,{block:0,index:0,kind:'Head'});
    Object.assign(state,{model:r.model,recordId:r.id},target||{});
  }
  function openRecord(id,preferred,source){
    const r=record(id);
    if(!r)return;
    window.ExplorerArchive.remember(id);
    const origin=source || state.mode;
    const before={origin,model:state.model,kind:state.kind,block:state.block,index:state.index,recordId:state.recordId,mode:state.mode};
    history.replaceState({backTo:before},'',location.pathname+location.search);
    history.pushState({archiveDetail:id,origin},'','#record='+encodeURIComponent(id));
    bindRecord(r,preferred);
    state.archiveId=id;
    setMode('archive');
    const space=document.querySelector('.sys-space');
    if(space.getBoundingClientRect().bottom<90)space.scrollIntoView({block:'start'});
  }
  function returnToArchive(){
    state.archiveId=null;
    setMode('archive');
    window.ExplorerArchive.restore();
  }
  function showAllRecords(){
    history.replaceState({},'',location.pathname+location.search);
    returnToArchive();
  }
  function backArchive(){
    if(history.state?.archiveDetail && !history.state.deepLink)history.back();
    else {history.replaceState({},'',location.pathname+location.search);returnToArchive();}
  }
  function openLive(id){
    if(id && record(id)){bindRecord(record(id));state.liveRequest++;}
    setMode('live');
  }
  const api={state,records,esc,coord,linked,selectObject,openRecord,openLive,openFeatures:()=>setMode('features'),backArchive,showAllRecords,
    stage:spec=>window.ExplorerPlan.stage(spec),openLaws:()=>window.ExplorerPlan.openLaws()};
  let nativeVisible=false,inspectorChanged=false,viewAfterEntry='scene';
  const native=window.ExplorerNative.create(api,{
    onUnit(index){
      if(state.mode==='archive' && state.archiveId){syncScene(index);return;}
      state.index=index;inspectorChanged=true;syncScene();
    },
    onSample:openLive
  });
  window.ExplorerPlan.init(api);
  for(const mode of ['live','archive','features'])$('mode-'+mode).onclick=()=>{setNative(false);setMode(mode);};
  $('coord-form').onsubmit=e=>{
    e.preventDefault();
    const input=$('coord-input').value.normalize('NFKC').toUpperCase().replace(/\s/g,'');
    const m=/^B(\d+)\/(H|U)(\d+)$/.exec(input);
    if(!m || !selectObject({block:Number(m[1]),kind:m[2]==='H'?'Head':'MLP',index:Number(m[3])})){
      const d=models[state.model];$('coord-info').classList.add('error');
      $('coord-info').textContent='请输入 B0–'+(d.layers-1)+' / H0–'+(d.heads-1)+' 或 U0–'+(d.width-1);$('coord-input').focus();
    }
  };
  $('model-select').onchange=e=>{
    const model=e.target.value;
    const r=records.find(item=>item.model===model && linked(item).length);
    selectObject({model,recordId:r.id,...linked(r)[0]});
  };
  $('display-open').onclick=()=>{
    const box=$('display-options');box.hidden=!box.hidden;
    $('display-open').setAttribute('aria-expanded',String(!box.hidden));
  };
  for(const id of ['grid','labels'])$(id).onchange=()=>syncScene(nativeVisible?native.selectedRecordUnit():null);
  $('zoom-in').onclick=()=>scene.zoom(1.14);$('zoom-out').onclick=()=>scene.zoom(.86);$('zoom-reset').onclick=()=>scene.reset();
  const work=$('workbench');
  function setNative(show){
    const next=Boolean(show && work.classList.contains('is-full'));
    if(next===nativeVisible)return;
    nativeVisible=next;
    $('full-native').hidden=!next;$('console').hidden=next;
    work.classList.toggle('with-native-field',next);
    $('full-workspace').setAttribute('aria-pressed',String(!next));
    $('full-units').setAttribute('aria-pressed',String(next));
    if(next){native.render();return;}
    native.close();syncScene();
    if(inspectorChanged || state.mode==='live'){
      if(state.mode==='live')window.ExplorerLive.render($('live-panel'),api);
      if(state.mode==='features')window.ExplorerFeatures.render($('features-panel'),api);
    }
    inspectorChanged=false;
  }
  function full(on){
    work.classList.toggle('is-full',on);
    $('full-views').hidden=!on;$('native-open').hidden=on;
    $('fullscreen').setAttribute('aria-label',on?'退出全屏':'全屏');$('fullscreen').title=on?'退出全屏':'全屏';
    setNative(on && viewAfterEntry==='native');
    if(!on)viewAfterEntry='scene';
  }
  async function enter(view){
    viewAfterEntry=view;
    if(work.classList.contains('is-full')){setNative(view==='native');return;}
    try{await work.requestFullscreen();}catch{full(true);}
  }
  $('fullscreen').onclick=async()=>{
    if(document.fullscreenElement)await document.exitFullscreen();
    else if(work.classList.contains('is-full'))full(false);
    else await enter('scene');
  };
  $('native-open').onclick=()=>enter('native');
  $('full-workspace').onclick=()=>setNative(false);$('full-units').onclick=()=>setNative(true);
  document.addEventListener('fullscreenchange',()=>full(document.fullscreenElement===work));
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!document.fullscreenElement&&!$('plan-dialog').open)full(false);});
  if('scrollRestoration' in history)history.scrollRestoration='manual';
  addEventListener('popstate',e=>{
    const r=record(e.state?.archiveDetail);
    if(r){bindRecord(r);state.archiveId=r.id;setMode('archive');}
    else if(e.state?.backTo && e.state.backTo.origin!=='archive'){
      const {origin,mode,model,kind,block,index,recordId}=e.state.backTo;
      Object.assign(state,{model,kind,block,index,recordId,archiveId:null});
      setMode(mode);
      if(origin==='laws')window.ExplorerPlan.openLaws();
      if(origin==='questions')window.ExplorerPlan.openQuestions();
    }else returnToArchive();
  });
  const start=new URLSearchParams(location.hash.slice(1)), initial=record(start.get('record'));
  if(initial){history.replaceState({archiveDetail:initial.id,deepLink:true},'',location.href);bindRecord(initial);state.archiveId=initial.id;state.mode='archive';}
  else if(['archive','features'].includes(start.get('mode'))) state.mode=start.get('mode');
  render();
})();
