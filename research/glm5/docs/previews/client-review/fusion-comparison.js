/* Inside the review's shadow root. The reader, cases and exports share selectedRows(). */
const outcomeLabels = {both:'头对、模型对',headOnly:'头对、模型错',div:'头错、模型对',neither:'头错、模型错'};
function caseOutcome(row) {
  if (row.head === row.c.gold) return row.c.correct ? 'both' : 'headOnly';
  return row.c.correct ? 'div' : 'neither';
}
function outcomeOptions(kind) {
  return Object.entries({all:'全部题', ...outcomeLabels}).map(([key,label]) =>
    '<option value="'+key+'" '+(key===kind?'selected':'')+'>'+label+'</option>').join('');
}
function countOutcomes(rows) {
  const counts = {both:0, headOnly:0, div:0, neither:0};
  for (const row of rows) counts[caseOutcome(row)]++;
  return counts;
}
function outcomeMatrix(rows) {
  const counts = countOutcomes(rows);
  function cell(kind) {
    return '<td><button aria-label="'+outcomeLabels[kind]+'，'+counts[kind]+' 道题" data-act="browse" data-value="'+kind+'" '+
      (counts[kind]?'':'disabled')+'>'+nf(counts[kind])+'</button></td>';
  }
  return '<table class="case-matrix" aria-label="按头选源与模型答案交叉查看题目"><thead><tr><td></td><th scope="col">模型答对</th><th scope="col">模型答错</th></tr></thead><tbody>'+
    '<tr><th scope="row">头选对</th>'+cell('both')+cell('headOnly')+'</tr><tr><th scope="row">头选错</th>'+cell('div')+cell('neither')+'</tr></tbody></table>';
}
function fusionComparison() {
  const groups = state.selected.filter(p => byPhase.has(p));
  const alongside = state.selected.filter(p => !byPhase.has(p));
  const rows = selectedRows();
  const total = classify(rows);
  const families = new Set(rows.map(x => x.p + '/' + x.f.id)).size;
  const ruleButtons = rules.map((name, i) =>
    '<button data-act="rule" data-value="'+i+'" aria-pressed="'+(i===state.rule)+'">'+name+'</button>').join('');
  const measures = [[total.h,'头选对记录','teal'], [total.m,'模型答对首个 token','blue']];
  const metrics = measures.map(([n,title,cls]) =>
    '<div class="stat"><b><i class="swatch '+cls+'"></i>'+title+'</b><div class="amount '+cls+'">'+pct(n,total.n)+'</div><small>'+nf(n)+' / '+nf(total.n)+' 题</small></div>').join('');
  const batches = groups.map(p => {
    const batch = classify(rows.filter(x=>x.p===p));
    const bars = [[batch.h,'a','头选对记录'],[batch.m,'b','模型答对首个 token']].map(([n,cls,title]) =>
      '<div class="track" aria-hidden="true"><div class="fill '+cls+'" style="width:'+(100*n/batch.n)+'%"></div></div><div class="num"><span class="sr">'+title+'</span>'+nf(n)+' / '+nf(batch.n)+'</div>').join('');
    return '<div class="plot"><div class="study">'+button('study','Phase '+p,p,'textbtn')+'</div>'+bars+'</div>';
  }).join('');
  const pickControls = records.map(r =>
    '<label><input type="checkbox" data-pick="'+r.p+'" '+(state.selected.includes(r.p)?'checked':'')+'>'+labelOf(r)+'</label>');
  const choices = (compatible) => pickControls.filter((_,i)=>byPhase.has(records[i].p)===compatible).join('');
  const drillButton = (mode,title,cls) => total.n ? button('browse',title,mode,cls) : '<button class="'+cls+'" disabled>'+esc(title)+'</button>';
  return '<header class="intro fusion-intro"><div class="scope-line">Qwen3-14B <span>／</span> 注意力头 B26/H36</div><h1>模型答对时，注意力头也选对了吗？</h1><p>在同一题中，对照这个头最关注的记录与模型最终给出的答案。</p></header>'+
    '<div class="fusion-grid"><section class="card chartcard"><header class="charthead split"><h2>选源与答题正确率</h2><div class="seg" role="group" aria-label="取值规则">'+ruleButtons+'</div></header>'+
    '<div class="stats">'+metrics+'</div><div class="mobile-cases">'+drillButton('div','查看 '+nf(total.div)+' 道分歧题 →','textbtn')+'</div><div class="bars"><div class="split batch-title"><h3>按测试批次</h3><small>正确题数 / 题数</small></div>'+
    (batches||'<p class="notice">勾选 Phase 2983 或 2990 可查看对应批次。</p>')+
    '<div class="chart-foot split"><span class="caption">'+rules[state.rule]+' · '+groups.length+' 批 · '+nf(families)+' 个确认家族</span>'+drillButton('all','查看全部题目 →','textbtn')+'</div></div></section>'+
    '<aside class="card case-card"><h2>分歧题</h2><div class="case-total">'+nf(total.div)+' <small>道</small></div><p>模型给出了正确答案，这个头最关注的却是另一条记录。</p>'+
    drillButton('div','查看分歧题 →','primary')+outcomeMatrix(rows)+'<div class="case-more">'+button('openpart','本头 B26/H36 的证据 →','B26/H36','textbtn')+button('study','追查首行路径 B28/H21 →',2997,'textbtn')+'</div></aside>'+
    '<section class="card set-card"><header class="split"><h2>比较集</h2>'+button('export','导出比较集 ↓','','textbtn')+'</header>'+
    '<div class="pick-row"><label>可合计的批次</label><div class="chips">'+choices(true)+'</div></div><div class="pick-row"><label>并排阅读</label><div class="chips">'+choices(false)+'</div></div></section>'+
    '<section class="card method-card"><h2>怎样读这组结果</h2><p>头的读数取五条记录中注意力质量最高的一条。模型的读数取首个输出 token。每个家族在每条规则下各有四道题。</p>'+
    '<details><summary>查看汇总口径与证据边界</summary><p>两批属于同一模型快照，采用不同姓名家族。按题数合计，组内题目共享材料。表外 token 另有注意力份额，分歧题只能说明这一个头的首选与最终输出不同。</p></details></section></div>'+
    (alongside.length?'<section class="card"><h2>相关测试并读</h2><p class="caption">以下研究的终点、部件或模型不同，各自保留原量尺。</p>'+alongside.map(p=>researchRow(rec(p))).join('')+'</section>':'');
}
