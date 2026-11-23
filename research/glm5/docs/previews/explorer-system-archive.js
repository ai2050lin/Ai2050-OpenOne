'use strict';
(() => {
  const $=id=>document.getElementById(id);
  const f={q:'',model:'',kind:'',index:'',receipt:false};
  let saved={row:'',scroll:0,page:0,workScroll:0};
  const norm=v=>String(v??'').normalize('NFKC').toLowerCase().trim();
  function objects(r,api){
    const typed=norm(f.index).toUpperCase().replace(/\s+/g,'');
    return api.linked(r).filter(o=>{
      const id=api.coord(o);
      return (!f.kind||o.kind===f.kind)&&(!typed||[id,id.split('/')[1],String(o.index),'B'+o.block].includes(typed));
    });
  }
  function remember(id){
    const list=$('archive-list');
    if(list?.closest('#archive-panel') && !$('archive-panel').hidden)saved={row:id,scroll:list.scrollTop,page:window.scrollY,workScroll:$('workbench').scrollTop};
  }
  function restore(){
    const list=$('archive-list');
    if(!list)return;
    list.scrollTop=saved.scroll;
    const row=[...list.children].find(el=>el.dataset.record===saved.row);
    if(row){row.classList.add('returned');row.focus({preventScroll:true});}
    window.scrollTo(0,saved.page);$('workbench').scrollTop=saved.workScroll;
  }
  function detail(host,api,r){
    const e=api.esc, links=api.linked(r), obj=api.coord(api.state);
    const options=links.map(o=>'<option '+(api.coord(o)===obj?'selected':'')+'>'+api.coord(o)+'</option>').join('');
    function file(label,path){
      if(!path || !/^(research|frontend|tests)\/[\w./-]+$/.test(path) || path.split('/').includes('..'))return '';
      return '<p>'+e(label)+' · <a href="../../../../'+e(path)+'" target="_blank" rel="noopener">'+e(path)+'</a></p>';
    }
    const origin=history.state?.origin;
    const backLabel=({laws:'返回规律索引',questions:'返回研究问题',features:'返回对象档案',live:'返回实时模式'})[origin]||'返回搜索结果';
    host.innerHTML='<article class="sys-section sys-archive-detail"><div class="sys-section-title"><h2>'+e(r.title)+'</h2><button id="archive-back" class="sys-button">← '+backLabel+'</button></div>'+
      '<p class="sys-muted">'+[r.model,'Phase '+r.phase,r.scope,r.status].map(e).join(' · ')+'</p><h3>测试题目</h3><p>'+e(r.question)+'</p><h3>测试结果</h3><p id="archive-result">'+e(r.result)+'</p>'+
      '<div class="sys-result-metrics">'+(r.metrics||[]).map(m=>'<span>'+e(m.label)+'<strong>'+e(m.value)+'</strong></span>').join('')+'</div>'+
      '<div class="sys-detail-actions">'+(links.length>1?'<label>关联对象 <select id="archive-objects">'+options+'</select></label>':'')+
      (links.length?'<button id="archive-features" class="sys-button">查看对象特征</button>':'<span class="sys-note">此记录缺少对象索引，可查看整模型结果。</span>')+
      (!r.scope.includes('汇总')?'<button id="archive-live" class="sys-button">用此题进入实时模式</button>':'')+
      (origin&&origin!=='archive'?'<button id="archive-all" class="sys-link">查找其他测试</button>':'')+'</div>'+
      '<details><summary>核对原始资料</summary><p>模型快照 · <code>'+e(r.snapshot)+'</code></p><p>'+[r.run,r.caseId,r.sourceKey].filter(Boolean).map(e).join(' · ')+'</p>'+
      file('归档来源',r.source)+file('冻结材料',r.materialSource)+file('中心回执',r.receiptSource)+(r.evidenceStatus==='unreviewed'?'<p>中心证据评审 · 待审</p>':'')+'</details></article>';
    $('archive-back').onclick=api.backArchive;
    if($('archive-all'))$('archive-all').onclick=api.showAllRecords;
    if($('archive-live'))$('archive-live').onclick=()=>api.openLive(r.id);
    if($('archive-features'))$('archive-features').onclick=api.openFeatures;
    if($('archive-objects'))$('archive-objects').onchange=ev=>api.selectObject(links.find(o=>api.coord(o)===ev.target.value));
  }
  function render(host,api){
    const r=api.records.find(row=>row.id===api.state.archiveId);
    if(r){detail(host,api,r);return;}
    const e=api.esc;
    host.innerHTML='<section class="sys-section"><div class="sys-section-title"><h2>查找历史测试</h2><small>本地归档摘录 · 可核对中心回执</small></div>'+
      '<div class="sys-filters"><label>测试题目<input id="archive-q" type="search" placeholder="题目、关键词或样本编号"></label>'+
      '<label>模型<select id="archive-model"><option value="">全部模型</option>'+[...new Set(api.records.map(r=>r.model))].map(m=>'<option>'+e(m)+'</option>').join('')+'</select></label>'+
      '<label>对象类型<select id="archive-kind"><option value="">全部类型</option><option value="Head">Head 注意力头</option><option value="MLP">FFN 神经元</option></select></label>'+
      '<label>原生编号<input id="archive-index" spellcheck="false" placeholder="如 B30/U9517"></label></div>'+
      '<div class="sys-archive-tools"><b id="archive-count" aria-live="polite"></b><div><label><input type="checkbox" id="archive-receipt">仅有中心回执</label><button class="sys-link" id="archive-clear">清除筛选</button></div></div>'+
      '<div class="sys-archive-table"><header class="sys-row"><span>测试题目</span><span>模型</span><span>关联对象</span><span>记录层级</span><span></span></header><div id="archive-list" aria-label="历史测试结果"></div></div></section>';
    function rows(){
      const result=api.records.filter(r=>{
        const hay=norm([r.question,r.title,r.caseId,r.run,'Phase '+r.phase].join(' '));
        return (!f.model||r.model===f.model)&&(!f.receipt||r.status==='中心已读回')&&(!(f.kind||f.index.trim())||objects(r,api).length)&&norm(f.q).split(/\s+/).every(q=>hay.includes(q));
      });
      $('archive-count').textContent=result.length+' 条记录';
      $('archive-list').innerHTML=result.map(r=>{
        const matches=objects(r,api);
        return '<button type="button" class="sys-row" data-record="'+e(r.id)+'"><span><strong>'+e(r.title)+'</strong><small>Phase '+e(r.phase)+' · '+e(r.status)+'</small></span><span>'+e(r.model)+'</span><span><code>'+e(matches.length?api.coord(matches[0]):'无对象索引')+'</code>'+(matches.length>1?' +'+(matches.length-1):'')+'</span><span>'+e(r.scope)+'</span><span>›</span></button>';
      }).join('')||'<p class="sys-note">没有匹配的历史记录。可清除筛选或更换对象编号。</p>';
    }
    for(const k of Object.keys(f)){
      const el=$('archive-'+k);
      if(k==='receipt')el.checked=f[k];else el.value=f[k];
      el.addEventListener(el.tagName==='SELECT'||k==='receipt'?'change':'input',()=>{
        f[k]=k==='receipt'?el.checked:el.value;
        saved={row:'',scroll:0,page:0,workScroll:0};rows();$('archive-list').scrollTop=0;
      });
    }
    $('archive-list').onclick=ev=>{
      const row=ev.target.closest('[data-record]');
      if(row){const r=api.records.find(r=>r.id===row.dataset.record);api.openRecord(r.id,objects(r,api)[0]);}
    };
    $('archive-clear').onclick=()=>{Object.assign(f,{q:'',model:'',kind:'',index:'',receipt:false});saved={row:'',scroll:0,page:0,workScroll:0};render(host,api);$('archive-q').focus({preventScroll:true});};
    rows();$('archive-list').scrollTop=saved.scroll;
  }
  window.ExplorerArchive={render,remember,restore};
})();
