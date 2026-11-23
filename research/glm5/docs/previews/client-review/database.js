/* Mounted in the native database lens. Study, question and observation identifiers stay distinct. */
function installDatabase() {
  const source=data.database;
  const armNames={base:'原始前向',target:'目标阻断',cut:'目标阻断',random:'匹配随机',sham:'假操作',restore:'恢复',restore_mlp:'仅恢复 FFN 组',ctrl:'匹配对照',control:'匹配对照'};
  const treeStudies={'RS.RECORD':[2983,2990,2997],'KN.ENTITY_REF':[2983,2990],'SEM.ATTRIBUTE':[3008,322]};
  const methods={2983:'自然观察',2990:'自然观察',2997:'阻断与恢复',3008:'生成与干预',322:'候选评分'};
  const runs={2983:data.cues[0].id,2990:data.cues[1].id,2997:data.path.runId,3008:data.color.run_id,322:'MA-H0-v5'};
  const goals={2983:'同一张表按名字、标签或首行查表时，注意力头选中哪条记录？',2990:'换一批姓名后，注意力头按三种规则查表的结果能否复现？',
    2997:'首行头写入的信息如何影响答案？整头恢复与 FFN 组恢复有什么差别？',3008:'能说出物体的颜色，能否用它判断两件物体颜色相同或不同？',322:'给定资料与常识冲突时，能否按要求切换答案？中文和英文表现如何？'};
  const receipt=p=>[2997,3008].includes(p);
  const components={2983:['B26/H36'],2990:['B26/H36'],2997:['B28/H21','U17216'],3008:['B28 · 32 单元'],322:[]};
  const ruleNames=[...rules,'按资料回答','按现实回答','颜色同异'];
  const resultNames={div:'头选错、模型答对',headOnly:'头选对、模型答错',both:'头和模型都对',neither:'头和模型都错',correct:'按本测试口径答对',wrong:'按本测试口径答错',rescued:'原始答对、阻断答错、恢复答对'};
  const labels={q:'关键词',branch:'研究方向',model:'模型',lang:'语言',part:'部件',method:'测试方法',rule:'取值规则',outcome:'结果',arm:'干预条件',archive:'归档'};
  const order=[2983,2990,2997,3008,322];
  const dirs=new Map(), searchable=new Map(), rows=new Map(order.map(p=>[p,[]]));
  for(const d of data.branches){
    let all=new Set();
    for(const g of d.groups){
      const phases=new Set(g.items.flatMap(i=>treeStudies[i.id]||[]));
      for(const i of g.items)dirs.set(i.id,{name:i.name,phases:treeStudies[i.id]||[],parent:d.name,group:g.name});
      dirs.set(d.name+'/'+g.name,{name:g.name,phases:[...phases],parent:d.name});
      phases.forEach(p=>all.add(p));
    }
    dirs.set(d.name,{name:d.name,phases:[...all]});
  }
  const normalize=s=>String(s??'').toLowerCase().replaceAll('/','').replace(/qwen(?:3[- ]?)?[- ]?(14|4)b\b/g,'qwen3-$1b');
  function add(p,row) {
    row.tokens=normalize([row.id,row.armId,row.item,row.gold,row.answer,row.rule,row.lang,row.prompt,row.extra,
      row.outcome&&resultNames[row.outcome],row.arm&&armNames[row.arm],row.correct?'答对':'答错'].filter(Boolean).join(' '));
    rows.get(p).push(row);
  }
  for(const [p,list] of rowsByPhase) for(const x of list) {
    const {c,f}=x;
    add(p,{id:c.id,rule:rules[c.rule],lang:'英文',part:['B26/H36'],correct:c.correct,outcome:caseOutcome(x),
      item:c.question,prompt:source.promptByPhase[p][c.id],gold:f.codes[c.gold],answer:c.output,cue:x});
  }
  const causalPrompts=new Map(source.causal.filter(r=>r[2]).map(r=>[r[0],r[2]]));
  for(const [id,arm,prompt,gold,answer,correct,margin,token] of source.causal) {
    add(2997,{id,arm,armId:id+'-'+arm,lang:'英文',part:arm==='restore_mlp'?['B28/H21','U17216']:['B28/H21'],
      rule:rules[2],item:'物理首行 · '+gold,prompt:prompt||causalPrompts.get(id),gold,answer,correct,margin,token});
  }
  for(const r of data.colorRows) {
    add(3008,{id:String(r.id),arm:r.arm,lang:'英文',part:['B28 · 32 单元'],rule:'颜色同异',item:r.item+' × '+r.ref,
      prompt:source.colors[r.id],gold:r.gold,answer:r.text,correct:r.correct,extra:(r.item.includes('strawberry')?'草莓 ':'')+(r.ref.includes('tomato')?'番茄 ':'')+r.color+' '+r.refcolor,color:r});
  }
  for(const [lang,i,entity,cell,gold,answer,scores] of source.bilingual) {
    const words=source.words[entity], zh=lang==='zh', rule=cell[0]==='doc'?'按资料回答':'按现实回答';
    add(322,{id:lang+'/'+i,lang:zh?'中文':'英文',part:[],rule,item:words.join(' / '),extra:entity+' '+cell.join(' '),
      prompt:source.h0Questions[lang][cell[0]].replace('{X}',words[zh?0:1]),gold,answer,correct:gold===answer,scores,cell,entity});
  }
  for(const p of order){
    const arms=new Map();
    for(const r of rows.get(p))if(r.arm){if(!arms.has(r.id))arms.set(r.id,{});arms.get(r.id)[r.arm]=r;}
    for(const group of arms.values()){
      const cut=group.cut||group.target;
      if(group.base?.correct && cut?.correct===false && group.restore?.correct){
        for(const row of [group.base,cut,group.restore])row.rescued=true;
      }
    }
    const r=rec(p);
    searchable.set(p,normalize([labelOf(r),r.title,goals[p],r.kind,modelOf(r),methods[p],runs[p],...components[p],
      ...[...dirs.values()].filter(d=>d.phases.includes(p)).map(d=>d.name)].join(' ')));
  }
  const filters={model:['Qwen3-14B','Qwen3-4B'],lang:['中文','英文'],part:['B26/H36','B28/H21','U17216','B28 · 32 单元'],
    method:[...new Set(Object.values(methods))],rule:ruleNames,outcome:Object.keys(resultNames),
    arm:[...new Set([...rows.values()].flat().map(r=>r.arm).filter(Boolean))],archive:['本稿附中心回执','本稿未附中心回执']};
  const optName=(key,v)=>key==='outcome'?resultNames[v]:key==='arm'?(armNames[v]||v)+' · '+v:key==='branch'?dirs.get(v)?.name||v:v;
  const clean=obj=>Object.fromEntries(Object.entries(obj||{}).filter(([k,v])=>typeof v==='string'&&v&&
    (k==='q'?v.length<=160:k==='branch'?dirs.has(v):filters[k]?.includes(v))));
  let query={}, picked={}, expanded=false, treeExpanded=false, treeRoot='', treeGroup='', currentList=null;
  function restore(){
    const q=new URL(globalThis.reviewRoute.href).searchParams;
    query=clean(Object.fromEntries(new URLSearchParams(q.get('db')||'')));
    try {picked=Object.fromEntries(Object.entries(JSON.parse(q.get('dbPick')||'{}')).filter(([p])=>rows.has(+p)).map(([p,f])=>[p,clean(f)]));}catch{picked={};}
    expanded=!!(query.arm||query.archive); treeRoot=dirs.get(query.branch)?.parent||query.branch||'';
    treeGroup=dirs.get(query.branch)?.group||'';treeExpanded=!!query.branch;
  }
  function persist(){
    const u=new URL(globalThis.reviewRoute.href),params=new URLSearchParams(query).toString();
    if(params)u.searchParams.set('db',params);else u.searchParams.delete('db');
    if(Object.keys(picked).length)u.searchParams.set('dbPick',JSON.stringify(picked));else u.searchParams.delete('dbPick');
    globalThis.reviewHistory.replaceState(null,'',u);
  }
  const tokens=q=>normalize(q||'').trim().split(/\s+/).filter(Boolean);
  function find(p,f){
    if(f.branch&&!dirs.get(f.branch)?.phases.includes(p)||f.model&&modelOf(rec(p))!==f.model||
      f.method&&methods[p]!==f.method||f.archive&&((f.archive===filters.archive[0])!==receipt(p)))return [];
    const partsByTerm=new Map(filters.part.map(v=>[normalize(v),v]));
    const keywords=tokens(f.q);
    const typedParts=keywords.filter(t=>partsByTerm.has(t)).map(t=>partsByTerm.get(t));
    const terms=keywords.filter(t=>!partsByTerm.has(t)&&(filters.lang.includes(t)||!searchable.get(p).includes(t)));
    return rows.get(p).filter(r=>(!f.lang||f.lang===r.lang)&&(!f.part||r.part.includes(f.part))&&(!f.arm||f.arm===r.arm)&&
      (!f.rule||f.rule===r.rule)&&(!f.outcome||(f.outcome==='correct'?r.correct:f.outcome==='wrong'?!r.correct:f.outcome==='rescued'?r.rescued:r.outcome===f.outcome))&&
      typedParts.every(t=>r.part.includes(t))&&terms.every(t=>r.tokens.includes(t)));
  }
  const ids=items=>[...new Set(items.map(r=>r.id))];
  const action=(type,title,value='',cls='')=>'<button type="button" class="'+cls+'" data-db="'+type+'" data-value="'+esc(value)+'">'+esc(title)+'</button>';
  function facet(k){
    return '<label class="db-facet"><span>'+labels[k]+'</span><select data-filter="'+k+'" aria-label="'+labels[k]+'"><option value="">全部</option>'+
      filters[k].map(v=>'<option value="'+esc(v)+'" '+(query[k]===v?'selected':'')+'>'+esc(optName(k,v))+'</option>').join('')+'</select></label>';
  }
  function tree(){
    const node=(id,title,n,level)=>'<button class="db-node l'+level+'" data-db="branch" data-value="'+esc(id)+'" aria-pressed="'+((query.branch||'')===id)+'"><span>'+esc(title)+'</span><small>'+n+'</small></button>';
    return '<aside class="db-tree '+(treeExpanded?'tree-open':'')+'"><div class="db-tree-title"><h2>研究方向</h2>'+action('tree-toggle',treeExpanded?'收起':'展开','','db-mobile')+'</div><div class="db-tree-body">'+node('','全部方向',order.length,0)+data.branches.map(d=>{
      const opened=treeRoot===d.name;
      return '<div class="db-tree-pair"><button class="twist" aria-label="'+(opened?'折叠':'展开')+esc(d.name)+'" aria-expanded="'+opened+'" data-db="expand-root" data-value="'+esc(d.name)+'">'+(opened?'▾':'▸')+'</button>'+node(d.name,d.name,dirs.get(d.name).phases.length,1)+'</div>'+
        (opened?d.groups.map(g=>{const id=d.name+'/'+g.name;return node(id,g.name,dirs.get(id).phases.length,2)+(treeGroup===g.name?g.items.map(i=>node(i.id,i.name,dirs.get(i.id).phases.length,3)).join(''):'');}).join(''):'');
    }).join('')+'<p class="db-tree-hint">同一测试可涉及多个方向。</p>'+
    (data.experimentSources?action('plans','研究计划','','textbtn'):'')+
    (treeStudies[query.branch]||data.branches.some(d=>d.groups.some(g=>g.items.some(i=>i.id===query.branch)))?action('design','查看该方向的设计题',query.branch,'textbtn'):'')+'</div></aside>';
  }
  function active(){
    const choices=Object.entries(query).filter(([,v])=>v);
    return choices.length?'<div class="db-chips">'+choices.map(([k,v])=>action('remove-filter',labels[k]+' · '+optName(k,v)+' ×',k)).join('')+action('reset','清空条件','','textbtn')+'</div>':'';
  }
  const roleNote=p=>p===2997?'B28/H21 整头干预 · U17216 仅作 128 单元组成员恢复':p===3008?'B28 的 32 个 FFN 单元 · 组级干预':p===322?'候选答案评分 · 行为层':'B26/H36 · 自然前向观察';
  function listHTML(){
    const hits=order.map(p=>[p,find(p,query)]).filter(([,rs])=>rs.length);
    const count=hits.reduce((sum,[,rs])=>sum+ids(rs).length,0), pickedIds=Object.keys(picked);
    return '<div class="db-resulthead"><div><h2>测试记录 <span>'+hits.length+'</span></h2><small>'+nf(count)+' 道命中题目 · 每项测试内按题去重</small></div><div class="db-compare-actions">'+(pickedIds.length?action('unpick','清空选择','','textbtn'):'')+
      '<button class="db-compare" data-db="compare" '+(pickedIds.length<2?'disabled title="勾选至少两项测试进行比较"':'')+'>'+ (pickedIds.length===1?'再选一项比较':'比较所选'+(pickedIds.length?' ('+pickedIds.length+')':''))+'</button></div></div>'+
    (!hits.length?'<section class="db-empty"><h3>没有符合全部条件的测试</h3><p>当前离线快照收录五项研究，可以移除单个条件继续查找。</p>'+action('reset','清空筛选','','primary')+'</section>':
    '<div class="db-table"><div class="db-thead"><span></span><span>测试 / 研究问题</span><span>测量与干预</span><span>命中题目</span></div>'+
    hits.map(([p,items])=>{const r=rec(p),n=ids(items).length;
      return '<article class="db-row"><input type="checkbox" data-db-pick="'+p+'" aria-label="选择 '+labelOf(r)+'" '+(picked[p]?'checked':'')+'><div class="db-id"><small>'+labelOf(r)+' · '+modelOf(r)+'</small><h3>'+action('study',r.title,p,'db-title')+'</h3><span>'+esc([...new Set(items.map(x=>x.lang))].join(' / '))+' · '+esc(r.area)+'</span></div>'+
        '<div class="db-method"><b>'+methods[p]+'</b><small>'+roleNote(p)+'</small></div><div class="db-number">'+action('cases',nf(n)+' 道 ↗',p,'db-hit')+(items.length>n?'<small>'+nf(items.length)+' 条读数</small>':'')+'</div></article>';
    }).join('')+'</div>');
  }
  function draw(){
    $('#main').innerHTML='<div class="db-wrap"><header class="db-head"><div><h1>测试数据库</h1><p>检索历史测试、逐题结果与模型部件的测量记录。</p></div><span class="db-scope">本地快照 · 5 项测试</span></header><div class="db-layout">'+tree()+
    '<div class="db-content"><section class="db-searchbox"><label class="db-search"><i aria-hidden="true"></i><input type="search" id="db-search" aria-label="检索测试与题目" placeholder="搜索研究问题、Phase、题号、物体或运行编号" maxlength="160" value="'+esc(query.q||'')+'"></label>'+
    '<div class="db-filters">'+['model','lang','part','method','rule','outcome'].map(facet).join('')+'</div><div class="db-extras">'+action('advanced',expanded?'收起更多条件':'更多条件','','textbtn')+'<div><span>示例查询</span> '+action('preset','首行分歧题','head','db-example')+' '+action('preset','U17216 组恢复','unit','db-example')+'</div></div>'+
    (expanded?'<div class="db-filters extra">'+['arm','archive'].map(facet).join('')+'</div>':'')+'<div id="db-active">'+active()+'</div></section><section id="db-results" aria-label="命中测试">'+listHTML()+'</section></div></div></div>';
    const route=new URL(globalThis.reviewRoute.href),phase=Number(route.searchParams.get('dbStudy'));
    if(route.searchParams.has('dbStudy')){
      route.searchParams.delete('dbStudy');
      globalThis.reviewHistory.replaceState(null,'',route);
      if(rows.has(phase))openStudy(phase);
    }
  }
  function summary(f){return Object.entries(f).map(([k,v])=>optName(k,v)).join(' · ')||'全部条件';}
  function openPlans(){
    const selected=query.branch;
    const groups=data.branches.flatMap(d=>d.groups.map(g=>{
      const items=g.items.filter(i=>!selected||[d.name,d.name+'/'+g.name,i.id].includes(selected));
      return items.length?'<details'+(selected?' open':'')+'><summary>'+esc(d.name+' / '+g.name)+' · '+items.length+' 项</summary><div class="links">'+
        items.map(i=>action('design',i.name,i.id,'textbtn')).join('')+'</div></details>':'';
    }));
    show('研究计划 · '+(dirs.get(selected)?.name||'全部方向'),'<p>选择设计题可查看准备改变的条件、示例问法和参考答案。</p>'+groups.join('')+
      '<div class="links">'+action('project-plan','查看项目研发计划','','textbtn')+'</div>');
  }
  function openStudy(p) {
    study(p);
    $('#dialog-body').insertAdjacentHTML('afterbegin','<div class="db-detail-meta">'+esc(modelOf(rec(p)))+' · '+esc(runs[p])+'</div><h3>研究问题</h3><p>'+esc(goals[p])+'</p><h3>测量与结果</h3>');
    $('#dialog-body').insertAdjacentHTML('beforeend','<section class="db-related"><h3>题目与部件</h3><div class="links">'+action('cases','查看当前条件下的命中题目',p,'textbtn')+
      components[p].map(v=>action('object',v,v,'textbtn')).join('')+'</div><p>'+roleNote(p)+'</p></section>'+
      (p===3008?archiveCard():p===322?link('../../../../scripts/mainline_h0_v5.py','实验脚本'):p===2997?'<div class="links">'+link(source.causalFile,'逐题干预结果 JSONL')+link(base+rec(p).dir+'/test.py','归档实验脚本')+'</div>':''));
    if(data.experimentSources&&receipt(p)){
      const color=p===3008,dir=base+(color?'color_reuse_v1/greedy/':'routine/run-t7dfn8_7/');
      $('#dialog-body').insertAdjacentHTML('beforeend','<details><summary>运行日志、审计与回执</summary><div class="links">'+
        link(dir+(color?'study-package/output.log':'review.json'),color?'原始运行日志':'审阅记录与日志尾段')+
        link(dir+(color?'study-receipt.json':'server-receipt.json'),'中心读回回执')+'</div></details>');
    }
  }
  function openCases(p,f=query,fromCompare=false){currentList={p,f:{...f},rs:find(p,f),page:0,fromCompare};drawCases();}
  function drawCases(){
    const {p,rs,page,f}=currentList,keys=ids(rs),start=page*10;
    show(labelOf(rec(p))+' · 命中题目',(currentList.fromCompare?action('compare','← 返回当前比较','','textbtn'):'')+'<div class="db-detail-meta">'+esc(summary(f))+'</div><div class="split"><b>'+nf(keys.length)+' 道题 · '+nf(rs.length)+' 条观测</b>'+action('study','打开测试记录',p,'textbtn')+'</div><div class="db-cases">'+keys.slice(start,start+10).map(id=>{
      const r=rs.find(x=>x.id===id);
      return '<div>'+action('case',id,id,'textbtn')+'<span>'+esc(r.item)+'</span></div>';
    }).join('')+'</div><div class="split"><button data-db="prev" '+(!start?'disabled':'')+'>上一页</button><small>'+ (keys.length?page+1:0)+' / '+Math.ceil(keys.length/10)+'</small><button data-db="next" '+(start+10>=keys.length?'disabled':'')+'>下一页</button></div>');
  }
  function openQuestion(id) {
    const {p,rs}=currentList, found=rs.filter(r=>r.id===id),r=found[0],back=action('back-cases','← 返回命中题目','','textbtn');
    if(r.cue){openCase(p,id);$('#dialog-body [data-act="back-list"]').outerHTML=back;$('#dialog-body').insertAdjacentHTML('beforeend','<details><summary>完整模型输入</summary><pre class="code">'+esc(r.prompt)+'</pre></details>'+action('object','查看本题的 B26/H36','B26/H36','textbtn'));return;}
    const table='<div class="scroller"><table class="gridtable"><tr><th>条件</th><th>预期</th><th>实际</th><th>结果</th></tr>'+found.map(x=>'<tr><td>'+esc(x.arm?(armNames[x.arm]||x.arm)+' · '+x.arm:x.rule)+'</td><td>'+esc(x.gold)+'</td><td>'+esc(x.answer)+'</td><td>'+(x.correct?'答对':'答错')+'</td></tr>').join('')+'</table></div>';
    show(labelOf(rec(p))+' / '+id,back+'<h3>'+esc(r.item)+'</h3><p>'+esc(r.lang)+' · '+esc(r.rule)+'</p><h3>'+(p===322?'冻结模板中的末问':'完整模型输入')+'</h3><pre class="code">'+esc(r.prompt)+'</pre>'+table+
    (r.scores?facts([['条件格',esc(r.cell.join(' / '))],['实体',esc(r.entity)],['候选对数得分',esc(JSON.stringify(r.scores))],['来源行',esc(id)+'，语种内从 1 编号']]):'')+
    (p===2997?facts(found.map(x=>['答案优势 · '+x.arm,esc(x.margin)]))+link(source.causalFile,'实际逐题输出 JSONL'):'')+studySources(rec(p)));
  }
  function compare(){
    const chosen=Object.entries(picked).map(([id,f])=>{const p=+id,rs=find(p,f);
      return {phase:p,run:runs[p],model:modelOf(rec(p)),scope:f,questions:ids(rs).length,observations:rs.length,correct_observations:rs.filter(r=>r.correct).length,
        outcomes:byPhase.has(p)?countOutcomes(rs.map(r=>r.cue)):null,metric:methods[p],source_sha256:byPhase.has(p)?byPhase.get(p).provenance['evaluation.json']:data.sourceHashes[p]};});
    const a=chosen.find(r=>r.phase===2983),b=chosen.find(r=>r.phase===2990);
    const same=a&&b&&Object.keys({...a.scope,...b.scope}).every(k=>a.scope[k]===b.scope[k])&&a.scope.rule;
    const combined=same?{questions:a.questions+b.questions,div:a.outcomes.div+b.outcomes.div}:null;
    const ref=a&&b?rules.filter(rule=>(!a.scope.rule||a.scope.rule===rule)||(!b.scope.rule||b.scope.rule===rule)).map(rule=>{
      const counts=classify([...rowsByPhase.get(2983),...rowsByPhase.get(2990)].filter(r=>rules[r.c.rule]===rule));
      return {rule,...counts};
    }):[];
    const txt=JSON.stringify({scope:'加入比较时各自保存的筛选条件',studies:chosen,pooled:combined,full_batch_reference:ref},null,2);
    show('比较所选测试','<p>每项测试使用勾选时的筛选条件。多干预实验按题目和观测分别计数。</p>'+
      (combined?'<div class="panelnote"><b>两批同口径共 '+nf(combined.questions)+' 道命中题</b>，其中头选错、模型答对 '+nf(combined.div)+' 道。</div>':'')+
      '<div class="scroller"><table class="gridtable"><tr><th>测试 / 量尺</th><th>筛选范围</th><th>命中题目</th><th>按各自口径答对 / 观测</th></tr>'+
      chosen.map(r=>'<tr><td>'+esc(labelOf(rec(r.phase)))+'<br><small>'+esc(r.metric)+'</small></td><td>'+esc(summary(r.scope))+'</td><td>'+action('compare-cases',nf(r.questions)+' 道',r.phase,'textbtn')+'</td><td>'+nf(r.correct_observations)+' / '+nf(r.observations)+'</td></tr>').join('')+'</table></div>'+
      (ref.length?'<h3>整批参照 · Phase 2983 与 2990</h3><p>以下按取值规则统计两批全部题目，可与上方筛出的子集对照。</p><div class="scroller"><table class="gridtable"><tr><th>规则</th><th>头选对记录</th><th>模型首 token 答对</th><th>头错、模型对</th></tr>'+
        ref.map(c=>'<tr><td>'+esc(c.rule)+'</td><td>'+nf(c.h)+' / '+nf(c.n)+'</td><td>'+nf(c.m)+' / '+nf(c.n)+'</td><td>'+nf(c.div)+'</td></tr>').join('')+'</table></div>':'')+
      '<div class="links"><a download="ai2050-test-comparison.json" href="data:application/json;charset=utf-8,'+encodeURIComponent(txt)+'">导出比较记录 JSON ↓</a></div>');
  }
  function openObject(v){state.part=v;show(v,partContent()+(locations[v]?action('locate','在模型分析中定位',v,'primary'):'')+'<div class="links">'+action('find-object','查此部件的全部记录',v,'textbtn')+'</div>');}
  function setQuery(next){query=clean(next);persist();draw();}
  root.addEventListener('input',e=>{if(e.target.id!=='db-search'||e.isComposing)return;query=clean({...query,q:e.target.value.trim()});persist();$('#db-results').innerHTML=listHTML();$('#db-active').innerHTML=active();});
  root.addEventListener('compositionend',e=>{if(e.target.id==='db-search')e.target.dispatchEvent(new Event('input',{bubbles:true}));});
  root.addEventListener('change',e=>{const k=e.target.dataset.filter,p=e.target.dataset.dbPick;
    if(k){setQuery({...query,[k]:e.target.value});root.querySelector('[data-filter="'+k+'"]')?.focus();}
    if(p){if(e.target.checked)picked[p]={...query};else delete picked[p];persist();$('#db-results').innerHTML=listHTML();root.querySelector('[data-db-pick="'+p+'"]')?.focus();}
  });
  root.addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;
    const k=b.dataset.db,v=b.dataset.value,a=b.dataset.act;
    if(!k){if(a==='openpart'){e.stopImmediatePropagation();openObject(v);}else if(a==='study'){e.stopImmediatePropagation();openStudy(+v);}else if(a==='jump-h36'){e.stopImmediatePropagation();$('#detail').close();setQuery({part:'B26/H36',rule:rules[2]});}return;}
    e.stopImmediatePropagation();
    switch(k){
      case 'reset':setQuery({});break;
      case 'remove-filter':setQuery({...query,[v]:''});break;
      case 'advanced':expanded=!expanded;draw();break;
      case 'tree-toggle':treeExpanded=!treeExpanded;draw();break;
      case 'expand-root':treeRoot=treeRoot===v?'':v;treeGroup='';draw();break;
      case 'branch':{const d=dirs.get(v);treeRoot=d?.parent||v;treeGroup=d?.group||(v.includes('/')?d.name:'');setQuery({...query,branch:v});break;}
      case 'preset':setQuery(v==='head'?{model:'Qwen3-14B',part:'B26/H36',rule:rules[2],outcome:'div'}:{part:'U17216',arm:'restore_mlp'});if(v==='unit'){expanded=true;draw();}break;
      case 'study':openStudy(+v);break;
      case 'plans':openPlans();break;
      case 'project-plan':window.openExperimentPlans();break;
      case 'design':designLeaf(v);break;
      case 'cases':openCases(+v);break;
      case 'prev':currentList.page--;drawCases();break;
      case 'next':currentList.page++;drawCases();break;
      case 'case':openQuestion(v);break;
      case 'back-cases':drawCases();break;
      case 'compare-cases':openCases(+v,picked[v],true);break;
      case 'compare':compare();break;
      case 'unpick':picked={};persist();draw();break;
      case 'object':openObject(v);break;
      case 'locate':actions.current.onLocate({snapshot:data.cues[0].snapshot,location:locations[v]});break;
      case 'find-object':$('#detail').close();setQuery({part:v});break;
    }
  },true);
  restore();return {draw,restore};
}
