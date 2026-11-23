'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const data = window.modelExplorerData;
  const feature = data.feature;
  const models = {
    'Qwen3-14B': {layers:40,heads:40,width:17408},
    'Qwen3-4B': {layers:36,heads:32,width:9728},
  };
  const tabsByMode = {
    archive:['运行与对照','插件分析','成果与方法'],
    live:['运行与对照','插件分析'],
    features:['对象详情','插件分析'],
  };
  const tabMemory = {archive:'插件分析',live:'运行与对照',features:'对象详情'};
  const paths = {
    dashboard:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
    cube:'m12 2 9 5v10l-9 5-9-5V7z M3 7l9 5 9-5 M12 12v10',
    flask:'M9 3h6 M10 3v6L4 19q-1 2 2 2h12q3 0 2-2L14 9V3 M8 15h8',
    settings:'M4 6h16 M4 12h16 M4 18h16 M8 4v4 M16 10v4 M10 16v4',
    pulse:'M2 13h5l3-9 4 16 3-10 2 3h3',
    archive:'M3 4h18v4H3z M5 8v12h14V8 M9 12h6',
    feature:'M4 18V9 M9 21V3 M14 17V7 M19 15v-5',
    locate:'M12 2v4 M12 18v4 M2 12h4 M18 12h4 M19 12a7 7 0 1 1-14 0 7 7 0 1 1 14 0',
    search:'M16 10A6 6 0 1 1 4 10a6 6 0 1 1 12 0 M15 15l6 6',
    expand:'M3 9V3h6 M15 3h6v6 M21 15v6h-6 M9 21H3v-6',
    reset:'M3 9a9 9 0 1 1 0 7 M3 3v6h6',
    head:'M5 6h14v12H5z M8 3v3 M12 3v3 M16 3v3 M8 18v3 M12 18v3 M16 18v3 M8 10h8 M8 14h5',
    lens:'M13 3a8 8 0 1 1-4 15 8 8 0 0 1 4-15 M17 17l5 5',
    layers:'m12 3 10 5-10 5L2 8z M3 12l9 5 9-5 M3 16l9 5 9-5',
  };
  paths.sliders=paths.settings;
  function icon(name) {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="'+paths[name]+'"/></svg>';
  }
  document.querySelectorAll('[data-icon]').forEach(el => el.innerHTML=icon(el.dataset.icon));
  const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const state = {model:'Qwen3-14B',mode:'archive',block:26,kind:'H',index:36,plugin:'Head',sample:2,record:false,replay:false,position:0,draft:''};
  const nativeId = () => 'B'+state.block+'/'+state.kind+state.index;
  const knownHead = () => state.model==='Qwen3-14B' && nativeId()==='B26/H36';
  const knownUnit = () => state.model==='Qwen3-14B' && nativeId()==='B36/U2429';
  const current = () => data.cases[state.sample];
  const pct = value => (100*value).toFixed(1)+'%';
  const controls = {Head:'head',HiddenState:'layers',MLP:'feature','Logit Lens':'lens'};
  const scene = window.createModelScene($('scene'),{
    onSelect:(b,h)=>selectObject(b,'H',h),
    onSelectUnit:(b,u)=>selectObject(b,'U',u),
  });
  function selectObject(block,kind,index) {
    Object.assign(state,{block,kind,index,record:false,plugin:kind==='H'?'Head':'MLP'});
    $('locator-input').value='';
    $('loc-error').textContent='';
    render();
  }
  function samplePicker() {
    return '<label class="sample-picker">测试条件<select data-sample>'+data.cases.map((x,i)=>
      '<option value="'+i+'" '+(i===state.sample?'selected':'')+'>'+x.rule+'</option>').join('')+'</select></label>';
  }
  function sampleMeta() {
    return 'Phase 2983 <span>·</span> <code>'+esc(current().id)+'</code> <span>·</span> 最后一个输入 token';
  }
  function renderHead() {
    const sample=current();
    const peak=sample.rows.reduce((best,row,i,rows)=>row.mass>rows[best].mass?i:best,0);
    const outside=1-sample.rows.reduce((sum,row)=>sum+row.mass,0);
    const bars=sample.rows.map((row,i)=>{
      const cls='barrow'+(i===peak?' peak':'')+(i===sample.gold?' gold':'');
      return '<div class="'+cls+'"><span>第 '+(i+1)+' 行</span><span class="track"><i style="width:'+pct(row.mass)+'"></i></span><b>'+pct(row.mass)+'</b></div>';
    }).join('');
    return '<header class="analysis-head"><div><h2>B26/H36 关注了哪条记录？</h2><p class="source">'+sampleMeta()+'</p></div>'+samplePicker()+'</header>'+
      '<div class="analysis-grid"><div><div class="bars" aria-label="各记录占全部注意力的比例">'+bars+'</div><div class="chart-foot"><span><i class="chart-key gold"></i>正确行</span><span><i class="chart-key peak"></i>最高行</span><span>其余 token 合计 '+pct(outside)+'</span></div></div>'+
      '<section class="reading"><h3>五条记录中，第 '+(peak+1)+' 行权重最高</h3><p>题目要求读第 '+(sample.gold+1)+' 行。模型输出 <code>'+esc(sample.output)+'</code>，与正确答案一致。</p>'+
      '<button class="action" data-action="record" aria-label="打开历史测试">打开历史测试 <span aria-hidden="true">↗</span></button></section></div>';
  }
  function renderFeature() {
    return '<div class="analysis-head"><div><h2>B36/U2429 在四个位置的激活</h2><p class="source">Phase 2933 · Qwen3-14B · MLP 中间单元</p></div></div>'+
      '<div class="feature-values">'+feature.positions.map(p=>'<div><small>'+esc(p.name)+' · '+p.index+'</small><b class="mono">'+p.value+'</b></div>').join('')+'</div>'+
      '<p class="ordinary">同组 '+feature.groupSize+' 个单元在三段联合屏蔽后答对 '+feature.blockedCorrect+' / '+feature.total+' 个家族，原模型答对 '+feature.baseCorrect+' / '+feature.total+'。这组对照尚未给出该单元的独立功能。</p>';
  }
  function emptyPlugin() {
    const message=state.mode==='live'?'本静态稿只提供 B36 的四位置 MLP 归档回放。':state.mode==='archive'?'本稿收录的逐行注意力来自 Qwen3-14B 的 B26/H36。':'本稿收录的特征样例来自 Qwen3-14B 的 B36/U2429。';
    return '<h2>'+esc(state.plugin)+' · <code>'+esc(nativeId())+'</code></h2><p class="empty">当前对象与数据来源没有可展示的 '+esc(state.plugin)+' 分析。'+message+'</p>'+goToExample();
  }
  function goToExample() {
    const label=state.mode==='archive'?'查看 B26/H36 归档样本':state.mode==='features'?'查看 B36/U2429 特征样本':'打开 B36 归档回放';
    return '<button class="secondary" data-action="example">'+label+'</button>';
  }
  function renderRecord() {
    const sample=current();
    const rows=sample.rows.map((row,i)=>'<tr class="'+(i===sample.gold?'correct':'')+'"><td>'+(i+1)+'</td><td>'+row.label+'</td><td>'+esc(row.name)+'</td><td>'+esc(row.value)+'</td><td>'+pct(row.mass)+'</td></tr>').join('');
    return '<div class="record-head"><div><h2>Phase 2983 · 三线索选源</h2><p class="source">'+esc(sample.id)+' · '+esc(sample.rule)+' · Qwen3-14B / BF16 · B26/H36</p></div><button class="secondary" data-action="back">返回分析</button></div>'+
      '<div class="record-layout"><table><caption class="group-label">原题中的五条记录</caption><thead><tr><th>物理行</th><th>标签</th><th>名字</th><th>答案</th><th>注意力</th></tr></thead><tbody>'+rows+'</tbody></table><section><h3>本题输出 <code>'+esc(sample.output)+'</code></h3><p class="ordinary">正确记录为第 '+(sample.gold+1)+' 行。注意力取自最后一个输入 token，对各行原文范围求和。</p><details><summary>原始模型输入</summary><pre>'+esc(sample.prompt)+'</pre></details><details><summary>运行与快照</summary><pre>'+esc(data.run)+'\n'+esc(data.snapshot)+'</pre></details></section></div>';
  }
  function livePanel() {
    const p=feature.positions[state.position];
    return '<h2>运行与对照</h2><p class="ordinary">输入文本供本机前向使用。静态预览可以回放一份已采集的样本。</p>'+
      '<label>运行文本<textarea id="draft" rows="2" placeholder="输入要观察的文本">'+esc(state.draft)+'</textarea></label>'+
      '<div class="actions"><button class="secondary" disabled>开始本机运行</button><button class="action" data-action="replay">回放归档样例</button></div>'+
      (state.replay?'<div class="replay"><label class="field">Phase 2933 · 回放位置 <select id="position">'+feature.positions.map((x,i)=>'<option value="'+i+'" '+(i===state.position?'selected':'')+'>'+esc(x.name)+'</option>').join('')+'</select></label><p class="ordinary">B36/U2429 在 token '+p.index+' 的实测激活值为 <code>'+p.value+'</code>。</p></div>':'');
  }
  function taskContent(tab) {
    if(state.mode==='live') return livePanel();
    if(state.mode==='features') return knownUnit()?renderFeature():'<h2>'+esc(nativeId())+' 的对象详情</h2><p class="ordinary">当前静态资料没有该对象的特征解释，可从已归档的 MLP 单元查看实测值。</p>'+goToExample();
    if(knownUnit()) return renderFeature();
    if(!knownHead()) return '<h2>'+esc(nativeId())+' 的测试记录</h2><p class="ordinary">此预览未收录当前选择的逐题记录。</p>'+goToExample();
    const sample=current();
    if(tab==='成果与方法') return '<h2>Phase 2983 · 三线索选源</h2><p class="ordinary">同一组正文保留五条记录，分别按姓名、印刷标签和物理位置提问。比较每道题的正确行、该头的行注意力和最终输出。单道题的注意力排名只描述这次读数。</p><button class="action" data-action="record">打开历史测试</button>';
    return '<div class="analysis-head"><div><h2>同一组记录，三种提问条件</h2><p class="source">Phase 2983 · 冻结确认材料</p></div>'+samplePicker()+'</div><p class="ordinary">当前条件为'+sample.rule+'，正确答案是第 '+(sample.gold+1)+' 行的 <code>'+esc(sample.rows[sample.gold].value)+'</code>。</p><button class="action" data-action="open-head">查看 Head 分析</button>';
  }
  function render() {
    const focused=document.activeElement;
    const focusKey=['obj','plug','tab','sample'].find(k=>focused?.dataset && k in focused.dataset);
    const focusVal=focusKey && focused.dataset[focusKey];
    const dims=models[state.model];
    const tab=tabMemory[state.mode];
    $('model').value=state.model;
    $('archive-search-link').hidden=state.mode!=='archive';
    $('object-id').textContent=nativeId();
    $('dimension').textContent=dims.layers+' 个块 · 每块 '+(state.kind==='H'?dims.heads+' 个注意力头':dims.width.toLocaleString('zh-CN')+' 个 MLP 单元');
    $('objects').innerHTML=state.model==='Qwen3-14B'?[['B26/H36','注意力头','head'],['B36/U2429','MLP 单元','feature']].map(([id,label,image])=>
      '<button class="obj" data-obj="'+id+'" aria-label="'+id+'" aria-pressed="'+(id===nativeId())+'"><i>'+icon(image)+'</i><span><b>'+id+'</b><small>'+label+'</small></span></button>').join(''):'<p class="source">本预览只收录 14B 的对象归档。</p>';
    scene.update({model:state.model,block:state.block,head:state.kind==='H'?state.index:null,unit:state.kind==='U'?state.index:null,...dims,grid:$('grid').checked,labels:$('labels').checked});
    document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.mode===state.mode));
    $('tabs').innerHTML=tabsByMode[state.mode].map((t,i)=>'<button role="tab" id="tab-'+i+'" data-tab="'+t+'" aria-selected="'+(t===tab)+'" tabindex="'+(t===tab?'0':'-1')+'" aria-controls="'+(t==='插件分析'?'analysis-panel':'task-panel')+'">'+t+'</button>').join('');
    const analyzing=tab==='插件分析';
    $('tabs').hidden=state.record;
    $('analysis-panel').hidden=!analyzing||state.record;
    $('task-panel').hidden=analyzing||state.record;
    $('record-detail').hidden=!state.record;
    if(state.record) $('record-detail').innerHTML=renderRecord();
    else if(analyzing) {
      $('analysis-panel').setAttribute('aria-labelledby','tab-'+tabsByMode[state.mode].indexOf(tab));
      $('plugin-list').innerHTML=Object.entries(controls).map(([name,id])=>'<button class="plugin" data-plug="'+name+'" aria-pressed="'+(state.plugin===name)+'"><i>'+icon(id)+'</i>'+name+'</button>').join('');
      const headReady=state.mode==='archive' && state.plugin==='Head' && knownHead();
      const unitReady=(state.mode!=='live'||state.replay) && state.plugin==='MLP' && knownUnit();
      $('analysis-body').innerHTML=headReady?renderHead():unitReady?renderFeature():emptyPlugin();
    } else {
      $('task-panel').setAttribute('aria-labelledby','tab-'+tabsByMode[state.mode].indexOf(tab));
      $('task-panel').innerHTML=taskContent(tab);
    }
    if(focusKey && !focused.isConnected) document.querySelector('[data-'+focusKey+'="'+CSS.escape(focusVal)+'"]')?.focus({preventScroll:true});
  }
  function locate(text) {
    const match=text.trim().match(/^B(\d+)\s*\/\s*([HU])(\d+)$/i);
    const d=models[state.model];
    if(!match || +match[1]>=d.layers || +match[3]>=(match[2].toUpperCase()==='H'?d.heads:d.width)){
      $('loc-error').textContent='请输入 B0–B'+(d.layers-1)+'，搭配 H0–H'+(d.heads-1)+' 或 U0–U'+(d.width-1)+'。';return;
    }
    selectObject(+match[1],match[2].toUpperCase(),+match[3]);
  }
  $('search-form').addEventListener('submit',e=>{e.preventDefault();locate($('locator-input').value);});
  const savedModelSelections=new Map();
  $('model').onchange=e=>{
    savedModelSelections.set(state.model,[state.block,state.kind,state.index]);
    state.model=e.target.value;state.replay=false;
    selectObject(...(savedModelSelections.get(state.model)||[0,'H',0]));
  };
  document.addEventListener('change',e=>{
    if(e.target.matches('[data-sample]')) {state.sample=Number(e.target.value);render();}
    if(e.target.id==='position') {state.position=Number(e.target.value);render();$('position').focus();}
    if(['grid','labels'].includes(e.target.id)) render();
  });
  document.addEventListener('input',e=>{if(e.target.id==='draft') state.draft=e.target.value;});
  document.addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;
    if(b.dataset.mode){
      state.mode=b.dataset.mode;state.record=false;$('loc-error').textContent='';render();return;
    }
    if(b.dataset.obj){locate(b.dataset.obj);return;}
    if(b.dataset.tab){tabMemory[state.mode]=b.dataset.tab;state.record=false;render();return;}
    if(b.dataset.plug){state.plugin=b.dataset.plug;render();return;}
    switch(b.dataset.action){
      case 'record': state.record=true;render();$('record-detail').querySelector('button').focus();break;
      case 'back': state.record=false;tabMemory[state.mode]='插件分析';state.plugin='Head';render();$('analysis-body').querySelector('[data-action="record"]')?.focus();break;
      case 'open-head': tabMemory[state.mode]='插件分析';state.plugin='Head';render();break;
      case 'replay': case 'example':
        state.model='Qwen3-14B';
        if(state.mode==='archive') {tabMemory.archive='插件分析';selectObject(26,'H',36);}
        else {if(state.mode==='live'){state.replay=true;tabMemory.live='运行与对照';}selectObject(36,'U',2429);}
        break;
    }
  });
  $('display').onclick=()=>{
    $('display-tools').hidden=!$('display-tools').hidden;
    $('display').setAttribute('aria-expanded',String(!$('display-tools').hidden));
  };
  $('minus').onclick=()=>scene.zoom(0.85);$('plus').onclick=()=>scene.zoom(1.15);$('reset').onclick=()=>scene.reset();
  const work=$('workspace'), fullButton=$('fullscreen');
  function showFull(on) {
    work.classList.toggle('full',on);document.body.classList.toggle('fullscreen',on);
    fullButton.setAttribute('aria-label',on?'退出全屏':'全屏');
    fullButton.querySelector('span').textContent=on?'退出全屏':'全屏';
    (on?document.querySelector('.work-head'):document.querySelector('.right-actions')).append(fullButton);
    fullButton.focus({preventScroll:true});
  }
  fullButton.onclick=async()=>{
    if(work.classList.contains('full')){if(document.fullscreenElement)await document.exitFullscreen();else showFull(false);}
    else {try{await work.requestFullscreen();}catch{showFull(true);}}
  };
  document.addEventListener('fullscreenchange',()=>showFull(document.fullscreenElement===work));
  document.addEventListener('keydown',e=>{
    if(e.key==='Escape'){
      if(!$('display-tools').hidden){$('display').click();$('display').focus();}
      else if(work.classList.contains('full')&&!document.fullscreenElement)showFull(false);
    }
    if(e.target.matches('[data-tab]')&&['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){
      const ts=tabsByMode[state.mode],i=ts.indexOf(e.target.dataset.tab);
      const next=e.key==='Home'?0:e.key==='End'?ts.length-1:(i+(e.key==='ArrowRight'?1:-1)+ts.length)%ts.length;
      e.preventDefault();tabMemory[state.mode]=ts[next];state.record=false;render();$('tab-'+next).focus();
    }
  });
  const entryMode = new URLSearchParams(location.search).get('mode');
  if(Object.hasOwn(tabsByMode,entryMode))state.mode=entryMode;
  render();
})();
