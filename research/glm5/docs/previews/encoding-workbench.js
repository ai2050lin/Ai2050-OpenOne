(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const data = window.EncodingEvidence;
  const zh = window.EncodingChinese;
  const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const iconPaths = {
    overview:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
    layers:'m12 2 9 5-9 5-9-5z M3 12l9 5 9-5 M3 17l9 5 9-5',
    experiments:'M9 3h6 M10 3v7l-6 9q-1 2 2 2h12q3 0 2-2l-6-9V3 M8 15h8',
    settings:'M3 6h8m5 0h5 M3 12h3m5 0h10 M3 18h11m5 0h2 M12 3v6 M8 9v6 M16 15v6',
    file:'M5 2h9l5 5v15H5z M14 2v6h5 M8 12h8 M8 16h8',
    sliders:'M3 7h7m5 0h6 M3 17h11m5 0h2 M12 4v6 M16 14v6',
    screen:'M3 9V3h6 M15 3h6v6 M21 15v6h-6 M9 21H3v-6'
  };
  for (const icon of document.querySelectorAll('i[data-icon]')) {
    icon.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="'+iconPaths[icon.dataset.icon]+'"/></svg>';
  }
  const cases = data.headCases, ffn = data.ffn;
  const defaults = {
    head:{a:0,b:2,block:26,head:36,index:0,angle:'rows'},
    ffn:{a:0,b:3,block:36,head:null,index:2429,angle:'activation'},
    zh:{a:0,b:0,block:30,head:null,index:9517,angle:'pair'}
  };
  const questions = {
    head:{model:'Qwen3-14B',layers:40,heads:40,width:17408,phase:2983,angles:[['rows','来源行注意力'],['z','头输出 Z']]},
    ffn:{model:'Qwen3-14B',layers:40,heads:40,width:ffn.width,phase:2933,angles:[['activation','单元激活'],['hidden','跨层 HiddenState']]},
    zh:{model:'Qwen3-4B',layers:36,heads:32,width:9728,phase:261,angles:[['pair','逐句标准分'],['delta','配对差值']]}
  };
  const memory={}, drafts={};
  const state={q:'head',mode:'archive',tab:'contrast',dim:0,...defaults.head};
  const isHead=()=>state.q==='head';
  const isZh=()=>state.q==='zh';
  function context() {
    const s=state;
    const matched=isHead()?s.block===26 && s.head===36:isZh()?s.block===30 && Boolean(zh.record(s.a,s.index)):s.block===ffn.block;
    const materialKey=isHead()?cases[s.a].recordId+'|'+cases[s.b].recordId:isZh()?'zh:'+s.a+':'+s.b:ffn.record.id;
    return {...s,matched,materialKey,liveEmpty:s.mode==='live'&&!drafts[s.q]?.replay,action};
  }
  function selectQuestion(q) {
    memory[state.q]={a:state.a,b:state.b,index:state.index,head:state.head,block:state.block,angle:state.angle};
    if(q==='zh')window.EncodingTools.clear?.();
    Object.assign(state,{q},memory[q]||defaults[q]);
    $('locator-status').hidden=true;
  }
  function action(name,value) {
    if(name==='material') {openMaterial();return;}
    if(name==='index'){
      const size=isHead()?data.head.width:questions[state.q].width;
      if(!Number.isInteger(value)||value<0||value>=size)return;
      state.index=value;
    } else if(name==='tab')state.tab=value;
    else if(name==='restore'){
      if(!value||!Object.hasOwn(questions,value.q))return;
      if(state.q!==value.q)selectQuestion(value.q);
      for(const key of ['mode','tab','a','b','index','block','head'])if(Object.hasOwn(value,key))state[key]=value[key];
      state.angle=questions[state.q].angles.some(([id])=>id===value.angle)?value.angle:defaults[state.q].angle;
      if(Number.isInteger(value.dim)&&value.dim>=0&&value.dim<ffn.hiddenWidth)state.dim=value.dim;
    }
    render();
  }
  const scene=window.createModelScene($('scene-canvas'),{
    onSelect:(block,head)=>{state.block=block;state.head=head;render();},
    onSelectUnit:(block,index)=>{state.block=block;action('index',index);}
  });
  function options(select,entries,value) {
    const signature=JSON.stringify(entries);
    if(select.dataset.entries!==signature){
      select.replaceChildren(...entries.map(([v,text])=>new Option(text,String(v))));select.dataset.entries=signature;
    }
    select.value=String(value);
  }
  const numbered=entries=>entries.map((v,i)=>[i,v]);
  const pct=x=>(x*100).toFixed(1)+'%';
  function labels() {
    return isHead()?cases.map(c=>c.rule):isZh()?zh.options:ffn.positions.map(p=>p.label+' · '+p.index);
  }
  function badge(letter){return '<span class="letter '+letter.toLowerCase()+'">'+letter+'</span>';}
  function headRows(a,b) {
    const readings=c=>'目标 '+(c.gold+1)+' 行 · 峰值 '+(c.winner+1)+' 行 · 答 <b>'+esc(c.output)+'</b>';
    return '<p class="data-caption">末 token '+a.position+' 对五行正文的注意力 · A 蓝 / B 绿 · 满格 100%</p>'+
      a.rows.map((row,i)=>{
        const other=b.rows[i];
        return '<div class="mass-row"><span><strong>行 '+(i+1)+'</strong><small>'+esc(row.value)+'</small></span>'+
          '<div class="bar-pair" aria-hidden="true"><span class="bar-track"><i class="bar-a" style="width:'+100*row.mass+'%"></i></span>'+
          '<span class="bar-track"><i class="bar-b" style="width:'+100*other.mass+'%"></i></span></div>'+
          '<span class="percent"><span class="aval">'+pct(row.mass)+'</span> / <span class="bval">'+pct(other.mass)+'</span></span></div>';
      }).join('')+
      '<div class="results-line"><div>'+badge('A')+readings(a)+'</div><div>'+badge('B')+readings(b)+'</div></div>';
  }
  function traces(a,b,caption,unit,selected) {
    const lo=Math.min(0,...a,...b),hi=Math.max(0,...a,...b),span=hi-lo||1;
    const w=520,h=132,left=39,right=w-12,top=12,bottom=h-23;
    const px=i=>left+i*(right-left)/(a.length-1), py=v=>bottom-(v-lo)/span*(bottom-top);
    const points=row=>row.map((v,i)=>px(i).toFixed(2)+','+py(v).toFixed(2)).join(' ');
    const line=(row,color)=>'<polyline fill="none" stroke="'+color+'" stroke-width="1.9" points="'+points(row)+'"/>';
    const grid=[lo,(hi+lo)/2,hi].map(v=>'<line x1="'+left+'" y1="'+py(v)+'" x2="'+right+'" y2="'+py(v)+'" stroke="#e1e7ec"/><text x="1" y="'+(py(v)+3)+'">'+Number(v.toPrecision(3))+'</text>').join('');
    const cursor=Number.isInteger(selected)?'<line x1="'+px(selected)+'" y1="'+top+'" x2="'+px(selected)+'" y2="'+bottom+'" stroke="#596c80" stroke-dasharray="3 3"/>':'';
    return '<p class="data-caption">'+esc(caption)+' · A 蓝 / B 绿 · 共享纵轴</p>'+
      '<svg class="line-chart" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(caption)+'，两条原生读数曲线">'+grid+line(a,'#386ccb')+line(b,'#158378')+cursor+
      '<text x="'+left+'" y="'+(h-5)+'">'+unit+'0</text><text x="'+right+'" y="'+(h-5)+'" text-anchor="end">'+unit+(a.length-1)+'</text></svg>';
  }
  function pairReading(label,a,b,secondLabel=label) {
    return '<div class="results-line"><div>'+badge('A')+esc(label)+' <b>'+esc(a)+'</b></div><div>'+badge('B')+esc(secondLabel)+' <b>'+esc(b)+'</b></div></div>';
  }
  function signedBars(values) {
    const peak=Math.max(...values.map(Math.abs))||1;
    return '<p class="data-caption">B'+state.block+'/U'+state.index+' · 门控后激活 · 中线为零</p>'+values.map((v,i)=>{
      const flag=(i===state.a?' A':'')+(i===state.b?' B':'');
      return '<div class="ffn-row"><span>'+esc(ffn.positions[i].label)+' <b>'+flag+'</b></span><span class="signed-axis"><i style="'+(v<0?'right:50%;background:#b36a45':'left:50%;background:#3e73a8')+';width:'+50*Math.abs(v)/peak+'%"></i></span><output>'+v+'</output></div>';
    }).join('')+pairReading('token '+ffn.positions[state.a].index,values[state.a],values[state.b],'token '+ffn.positions[state.b].index);
  }
  function drawFigure(c) {
    const {angle}=state, el=$('measure-figure'), title=$('measure-title');
    title.textContent=questions[state.q].angles.find(([key])=>key===angle)[1];
    const metric=(isHead()&&angle==='z')||(!isZh()&&angle==='hidden');
    $('metric-controls').hidden=!metric||c.liveEmpty||!c.matched;
    if(c.liveEmpty||(!c.matched&&!isZh())){
      el.innerHTML='<div class="missing">'+(c.liveEmpty?'当前输入还没有模型读数。可回放归档材料查看已有测量。':'当前归档没有这个对象对应的原场。')+
        (!c.liveEmpty?'<button id="return-object">返回归档对象 '+(isHead()?'B26/H36':'B36/U2429')+'</button>':'')+'</div>';
      if($('return-object'))$('return-object').onclick=()=>{const d=defaults[state.q];Object.assign(state,{block:d.block,head:d.head,index:d.index});render();};
      return;
    }
    if(isZh()){zh.renderFigure(el,c);return;}
    if(metric){
      $('metric-label').textContent=isHead()?'选中通道 D':'残差坐标 d';
      $('metric-index').max=isHead()?data.head.width-1:ffn.hiddenWidth-1;
      $('metric-index').value=isHead()?state.index:state.dim;
    }
    if(isHead()){
      const a=cases[state.a],b=cases[state.b];
      el.innerHTML=angle==='rows'?headRows(a,b):traces(a.z,b.z,'B26/H36 · 输出投影前的 128 个分量','D',state.index)+pairReading('D'+state.index,a.z[state.index],b.z[state.index]);
    } else if(angle==='activation'){
      el.innerHTML=signedBars(ffn.positions.map((_,i)=>data.ffnValues(i)[state.index]));
    } else {
      const series=site=>Array.from({length:ffn.hiddenLayers},(_,l)=>data.hiddenValues(l,site)[state.dim]);
      const a=series(state.a),b=series(state.b);
      el.innerHTML=traces(a,b,'每层同一原生坐标 d'+state.dim+' · L0 为输入嵌入','L',null)+pairReading('L40 / d'+state.dim,a.at(-1),b.at(-1));
    }
    if(!isZh()&&state.a===state.b)el.insertAdjacentHTML('beforeend','<p class="data-caption">A 与 B 是同一份读数，可切换其中一项做对照。</p>');
  }
  function prompt() {return isHead()?cases[state.a].prompt:isZh()?zh.prompt(state.a):ffn.prompt;}
  function openMaterial() {
    const body=$('material-body'),c=context();
    if(c.liveEmpty){body.innerHTML='<h3>实时输入草稿</h3><pre>'+esc($('live-input').value||'尚未输入')+'</pre><p>尚无运行 ID 或读出记录。</p>';}
    else if(isZh()){zh.renderMaterial(body,c);}
    else {
      const a=isHead()?cases[state.a]:ffn,b=cases[state.b];
      let markup='<p>'+esc(questions[state.q].model)+' · Phase '+questions[state.q].phase+'</p>';
      if(isHead()){
        markup+=[['A',a],['B',b]].map(([k,r])=>'<h3>'+k+' · '+esc(r.rule)+'</h3><p>'+esc(r.questionZh)+'</p><details><summary>查看英文原始输入与题号</summary><code>'+esc(r.id)+'</code><pre lang="en">'+esc(r.prompt)+'</pre></details>').join('');
      }else markup+='<h3>顺序计算与查表</h3><p>'+esc(ffn.questionZh)+'</p><pre lang="en">'+esc(ffn.prompt)+'</pre><details><summary>查看记录中的生成前缀</summary><pre>'+esc(ffn.prefix)+'</pre></details>';
      const source=isHead()?data.head:ffn;
      markup+='<h3>归档来源</h3><p>模型快照 <code>'+esc(a.record.snapshot)+'</code></p><ul>'+source.sources.map(s=>'<li><a href="../../../../'+encodeURI(s.path)+'" target="_blank" rel="noopener">'+esc(s.label)+'</a></li>').join('')+'</ul>';
      body.innerHTML=markup;
    }
    $('material-dialog').showModal();
  }
  const allowed={
    archive:['object','contrast','tools','evidence'],live:['contrast','tools'],features:['object','tools','evidence']
  };
  function render() {
    const c=context(),q=questions[state.q],names=labels(),loc='B'+state.block+'/'+(isHead()?'H'+state.head:'U'+state.index);
    $('question').value=state.q;$('model-name').textContent=q.model;$('phase').textContent='Phase '+q.phase;
    $('scope-hint').textContent=isHead()?'同一家族 · 同一张五行表 · 只换提问规则':isZh()?'同一中文句对 · 同块不同单元':'同一道题 · 同一模型块 · 切换读出位置';
    $('a-label').textContent=isZh()?'中文句对':'观察 A';$('b-label').textContent=isZh()?'参照单元':'对照 B';
    $('a-tag').textContent=isZh()?'句对':'A';$('b-tag').textContent=isZh()?'参照':'B';
    options($('select-a'),numbered(names),state.a);
    options($('select-b'),numbered(isZh()?['U80','U9517']:names),state.b);
    options($('angle'),q.angles,state.angle);
    $('locate').value=loc;
    $('live-box').hidden=state.mode!=='live';
    if(state.mode==='live'){
      const draft=drafts[state.q]||{};
      if($('live-input').value!==(draft.text||''))$('live-input').value=draft.text||'';
      $('live-status').textContent=draft.replay?'回放当前选定材料的历史测量。':'离线预览中可编辑文本，实际前向需在客户端连接模型。';
    }
    for(const btn of document.querySelectorAll('[data-mode]'))btn.setAttribute('aria-pressed',String(btn.dataset.mode===state.mode));
    for(const btn of document.querySelectorAll('[data-tab]')){
      btn.hidden=!allowed[state.mode].includes(btn.dataset.tab);btn.setAttribute('aria-pressed',String(btn.dataset.tab===state.tab));
    }
    scene.update({model:q.model,layers:q.layers,heads:q.heads,width:q.width,block:state.block,head:isHead()?state.head:null,unit:isHead()?null:state.index,grid:$('grid').checked,labels:$('labels').checked});
    drawFigure(c);
    $('native').textContent=isZh()?'查看测量 ›':'查看全场 ›';
    if(isZh())zh.renderPanel($('work-panel'),c);else window.EncodingTools.render($('work-panel'),c);
  }
  $('question').onchange=e=>{selectQuestion(e.target.value);render();};
  for (const btn of document.querySelectorAll('[data-mode]'))btn.onclick=()=>{
    state.mode=btn.dataset.mode;if(!allowed[state.mode].includes(state.tab))state.tab=allowed[state.mode][0];render();
  };
  for (const btn of document.querySelectorAll('[data-tab]'))btn.onclick=()=>action('tab',btn.dataset.tab);
  for (const key of ['a','b'])$('select-'+key).onchange=e=>{
    state[key]=Number(e.target.value);
    if(state.mode==='live' && drafts[state.q]?.replay)drafts[state.q].text=prompt();
    render();
  };
  $('angle').onchange=e=>{state.angle=e.target.value;render();};
  $('metric-index').onchange=e=>{
    if(!e.target.value||!e.target.reportValidity())return;
    if(isHead())action('index',e.target.valueAsNumber);else{state.dim=e.target.valueAsNumber;render();}
  };
  $('coordinate').onsubmit=e=>{
    e.preventDefault();
    const query=$('locate').value.trim(),m=/^B(\d+)\s*\/\s*([HU])(\d+)$/i.exec(query),q=questions[state.q];
    const kind=isHead()?'H':'U',limit=isHead()?q.heads:q.width;
    if(!m||m[2].toUpperCase()!==kind||Number(m[1])>=q.layers||Number(m[3])>=limit){
      $('locator-status').textContent='本题请用 B0–B'+(q.layers-1)+' / '+kind+'0–'+kind+(limit-1)+' 定位。';$('locator-status').hidden=false;return;
    }
    $('locator-status').hidden=true;state.block=Number(m[1]);
    if(isHead())state.head=Number(m[3]);else state.index=Number(m[3]);
    render();
  };
  $('live-input').oninput=e=>{drafts[state.q]={text:e.target.value,replay:false};render();};
  $('replay').onclick=()=>{drafts[state.q]={text:prompt(),replay:true};render();};
  $('material').onclick=openMaterial;$('close-material').onclick=()=>$('material-dialog').close();
  $('native').onclick=()=>{action('tab','tools');if(!isZh())window.EncodingTools.openNative?.();$('work-panel').scrollIntoView({block:'start'});};
  $('display').onclick=()=>{const p=$('display-panel');p.hidden=!p.hidden;$('display').setAttribute('aria-expanded',String(!p.hidden));};
  $('grid').onchange=$('labels').onchange=render;$('reset-camera').onclick=()=>scene.reset();
  function fullscreen(active){$('analysis').classList.toggle('full',active);$('screen').setAttribute('aria-label',active?'退出全屏':'全屏观察');$('screen').title=active?'退出全屏':'全屏观察';}
  $('screen').onclick=async()=>{
    if(document.fullscreenElement){await document.exitFullscreen();return;}
    if($('analysis').classList.contains('full')){fullscreen(false);return;}
    try{await $('analysis').requestFullscreen();}catch{fullscreen(true);}
  };
  document.addEventListener('fullscreenchange',()=>fullscreen(document.fullscreenElement===$('analysis')));
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!document.fullscreenElement&&!$('material-dialog').open)fullscreen(false);});
  render();
})();
