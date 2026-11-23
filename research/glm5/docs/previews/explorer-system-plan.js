'use strict';
(() => {
  const $=id=>document.getElementById(id);
  const key='ai2050-explorer-design-drafts-v1';
  const topics=[
    {name:'知识网络',question:'同一种颜色，在不同物体上如何被复用？',test:'保留颜色，替换物体；再保留物体，更换颜色，比较原生响应和模型答案。',record:'p3008-color-base-45',link:'查看草莓颜色题',needs:['按物体家族切分发现集与留出集','在相同读出位置采集 Head 和 FFN','保存同色反例与异色对照']},
    {name:'组合推理',question:'名字、印刷行号和物理行序冲突时，模型采用哪条证据？',test:'只改变题目要求，保持五条正文记录相同。分别检查选择、写入和最终答案。',record:'p2983-confirm-000-02',link:'查看首行查找题',needs:['冻结互相冲突的线索和正确答案','保存 Q/K 选择、V/O 写入及下游读出','比较新家族预测与简单基线']},
    {name:'语法系统',question:'词序改变时，模型如何区分谁对谁做了什么？',test:'在相同词表上交换施事和受事，另设长度匹配对照与未见句式。',needs:['冻结词序最小对和未见句式','校验 tokenizer 位置与行为门','保留角色读出、反例和跨句式结果']}
  ];
  let api,tab='questions',pending=null,rows=[],route='local',lastRoute='local',notice='',opener=null;
  function init(shared){
    api=shared;
    try{const stored=JSON.parse(localStorage.getItem(key)||'[]');if(Array.isArray(stored))rows=stored.filter(r=>typeof r?.title==='string'&&Array.isArray(r.needs));}catch{}
    count();
    $('plan-open').onclick=()=>{pending=null;open(rows.length?'drafts':'questions');};
    $('plan-close').onclick=()=>$('plan-dialog').close();
    $('plan-dialog').addEventListener('close',()=>{if(opener?.isConnected)opener.focus({preventScroll:true});});
    $('plan-dialog').addEventListener('click',e=>{if(e.target===$('plan-dialog') && e.clientX<$('plan-dialog').getBoundingClientRect().left)$('plan-dialog').close();});
    $('plan-questions').onclick=()=>{tab='questions';paint();};
    $('plan-laws').onclick=()=>{tab='laws';paint();};
    $('plan-drafts').onclick=()=>{tab='drafts';paint();};
  }
  function count(){$('plan-count').textContent=String(rows.length);$('plan-count').hidden=!rows.length;}
  function open(which){
    tab=which;notice='';
    const dialog=$('plan-dialog');if(!dialog.open)opener=document.activeElement;
    paint();if(!dialog.open)dialog.showModal();
  }
  function stage(spec){
    const s=api.state, model=s.model;
    pending={...spec,needs:Array.isArray(spec.needs)?spec.needs:[spec.needs].filter(Boolean),
      model:spec.model||model,object:spec.object||api.coord(s),snapshot:api.records.find(r=>r.model===(spec.model||model))?.snapshot};
    route=lastRoute;
    open('drafts');
  }
  function questions(){
    const e=api.esc;
    $('plan-content').innerHTML='<p class="sys-plan-line">按研究问题积累单题观察、留出复现、原模型干预和反例。每项计划可交给本机或分布式节点。</p>'+
      topics.map((t,i)=>'<details class="sys-topic" '+(i===0?'open':'')+'><summary>'+t.name+' · '+t.question+'</summary><p>'+t.test+'</p>'+
      (t.record?'<button class="sys-button" data-topic-record="'+e(t.record)+'">'+e(t.link)+'</button>':'<p class="sys-muted">当前预览尚未收录此问题的历史题。</p>')+
      '<button class="sys-button" data-topic-plan="'+i+'">从当前对象设计测试</button></details>').join('');
    $('plan-content').onclick=ev=>{
      const button=ev.target.closest('button');if(!button)return;
      if(button.dataset.topicRecord){$('plan-dialog').close();api.openRecord(button.dataset.topicRecord,null,'questions');}
      if(button.dataset.topicPlan!=null){const t=topics[Number(button.dataset.topicPlan)];stage({title:t.question,needs:t.needs,topic:t.name});}
    };
  }
  function config(p){
    const e=api.esc;
    const fields=[['模型',p.model],['对象',typeof p.object==='string'?p.object:api.coord(p.object)],['模型快照',p.snapshot],['来源记录',p.recordId],['材料范围',p.materialScope],['工具',p.tool],['读出量',p.measure],['HiddenState 层',p.hiddenLayer],['读出位置',p.position],['来源位置',p.source],['分量',p.coordinate]];
    return '<details><summary>核对采集配置</summary>'+fields.filter(([,v])=>v!=null&&v!=='').map(([k,v])=>'<p>'+e(k)+' · <code>'+e(v)+'</code></p>').join('')+
      (p.inputText?'<p>输入材料</p><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+e(p.inputText)+'</pre>':'')+
      (p.pairedInputs?.length?p.pairedInputs.map((text,i)=>'<p>独立输入 '+(i+1)+' · '+e(text)+'</p>').join(''):
        p.materials&&!p.inputText?'<p>归档摘录 · '+e(p.materials)+'</p>':'')+'</details>';
  }
  function drafts(){
    const e=api.esc,p=pending;
    $('plan-content').onclick=null;
    $('plan-content').innerHTML=(p?'<section class="sys-plan-form"><small>新任务草稿</small><h3>'+e(p.title)+'</h3><p>'+e(p.model)+' · '+e(typeof p.object==='string'?p.object:api.coord(p.object))+'</p>'+
      '<ul>'+p.needs.map(n=>'<li>'+e(n)+'</li>').join('')+'</ul>'+config(p)+
      '<fieldset><legend>执行方式</legend><div class="sys-plan-routes"><label><input id="plan-local" type="radio" name="route" '+(route==='local'?'checked':'')+'>本机测试</label>'+
      '<label><input id="plan-distributed" type="radio" name="route" '+(route==='distributed'?'checked':'')+'>分布式测试</label></div><p id="route-description" class="sys-muted"></p></fieldset>'+
      '<button id="plan-save" class="sys-button sys-primary">保存任务草稿</button></section>':'')+
      '<div id="plan-list"><p class="sys-plan-line" role="status">'+e(notice||'草稿保存在当前设备。正式执行时先查重并冻结材料，完成后归档结果和反例，复核后更新对象档案。')+'</p>'+
      (!rows.length&&!p?'<p class="sys-note">还没有待测任务。从实时模式的采集配置、特征档案的缺项或研究问题建立第一项。</p>':'')+
      rows.map(r=>'<article class="sys-plan-card"><small>待执行 · '+(r.route==='distributed'?'分布式测试':'本机测试')+'</small><h3>'+e(r.title)+'</h3><p>'+e(r.model)+' · '+e(typeof r.object==='string'?r.object:api.coord(r.object))+'</p>'+config(r)+
      '<details><summary>验收时应保留的材料</summary><ul>'+r.needs.map(n=>'<li>'+e(n)+'</li>').join('')+'</ul></details></article>').join('')+'</div>';
    if(!p)return;
    function updateRoute(){
      route=$('plan-distributed').checked?'distributed':'local';
      $('route-description').textContent=route==='local'?'在本机使用同一模型快照，保存原生读数、配置与结果。':'按冻结家族分片，由节点领取任务，回传带哈希的产物，中心去重并单列独立复算。';
    }
    $('plan-local').onchange=updateRoute;$('plan-distributed').onchange=updateRoute;updateRoute();
    $('plan-save').onclick=()=>{
      rows.unshift({...pending,route});pending=null;lastRoute=route;count();
      try{localStorage.setItem(key,JSON.stringify(rows));notice='已在本机保存草稿。';}catch{notice='浏览器存储不可用，草稿暂存于当前页面。';}
      paint();
    };
  }
  function paint(){
    for(const name of ['questions','laws','drafts'])$('plan-'+name).setAttribute('aria-pressed',String(tab===name));
    if(tab==='questions')questions();
    else if(tab==='laws'){
      $('plan-content').onclick=null;
      const fromLaw={...api,openRecord:(id,preferred)=>api.openRecord(id,preferred,'laws')};
      window.ExplorerLaws.render($('plan-content'),fromLaw,()=>$('plan-dialog').close());
    }else drafts();
  }
  window.ExplorerPlan={init,stage,openLaws:()=>open('laws'),openQuestions:()=>open('questions')};
})();
