/* Analysis console for an offline review. Every read is scoped to the chosen archive. */
(() => {
  'use strict';
  const data = () => window.EncodingEvidence;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const val = v => Number.isFinite(v) ? String(v) : '缺读数';
  const rounded = v => Number.isFinite(v) ? Number(v.toPrecision(5)).toString() : '未定义';
  const id = (v,max) => Number.isInteger(v) && v>=0 && v<max;
  const ready = c => Boolean(c.matched && !c.liveEmpty);
  const isHead = c => c.q === 'head';
  const sides = c => isHead(c) ? data().headCases : data().ffn.positions;
  const sideName = (c,i) => isHead(c) ? sides(c)[i].rule : sides(c)[i].label;
  const position = (c,i) => isHead(c) ? sides(c)[i].position : sides(c)[i].index;
  const record = (c,i) => isHead(c) ? sides(c)[i].record : data().ffn.record;
  const scope = c => isHead(c) ? 'B'+c.block+'/H'+c.head : 'B'+c.block+'/U'+c.index;
  const sourceLink = (item,label) => '<a target="_blank" rel="noopener" href="explorer-system-design.html#record='+encodeURIComponent(item.id)+'">'+esc(label)+' ↗</a>';
  const field = (c,i) => isHead(c) ? sides(c)[i].z : data().ffnValues(i);
  const titles = {Head:'Head',MLP:'MLP',HiddenState:'HiddenState',lens:'Logit Lens',attention:'注意力分布',geometry:'表征几何'};
  const extras = ['attention','geometry'];
  const observationKey = 'ai2050.encoding-workbench.observations.v1';
  const enabledKey = 'ai2050.encoding-workbench.plugins.v1';
  function read(key) {
    try {const value=JSON.parse(localStorage.getItem(key)||'[]');return Array.isArray(value)?value:[];}
    catch {return [];}
  }
  const enabled = new Set(read(enabledKey).filter(k=>extras.includes(k)));
  let host, context, widget, widgetHost, domKey='', lastQuestion='', currentTool='Head';
  let currentSide='a', headBoundary='Z', hiddenLayer=37, hiddenCoord=0, geometryLayer=36;
  let status='';
  const heading = (title,text) => '<header class="et-heading"><h3>'+esc(title)+'</h3>'+(text?'<p>'+esc(text)+'</p>':'')+'</header>';
  const notice = (title,text) => '<div class="et-notice"><strong>'+esc(title)+'</strong><p>'+esc(text)+'</p></div>';
  function missing(c) {
    return c.liveEmpty ? notice('等待当前输入的采集结果','自定义文本需要由本机模型采集。可切回已采集的样例查看实测值。') :
      notice('所选对象缺少匹配记录','请定位到本题附带的 '+(isHead(c)?'B26/H36':'B36 的 FFN 单元')+'，或从测试记录选择另一份有原场的题目。');
  }
  function provenance(c) {
    const r=record(c,c.a);
    const sources=(isHead(c)?data().head:data().ffn).sources;
    return '<details class="et-source"><summary>来源与复核</summary><dl class="et-meta"><dt>运行</dt><dd>'+esc(r.run)+'</dd><dt>模型快照</dt><dd>'+esc(r.snapshot)+'</dd>'+
      sources.map(s=>'<dt>'+esc(s.label)+'</dt><dd><a href="../../../../'+esc(s.path)+'" target="_blank" rel="noopener">'+esc(s.path)+'</a>'+(s.sha256?'<br>SHA-256 '+esc(s.sha256):'')+'</dd>').join('')+'</dl></details>';
  }
  function objectPanel(c) {
    if(!ready(c))return missing(c);
    const d=data(), a=sides(c)[c.a], b=sides(c)[c.b], r=record(c,c.a);
    const names=isHead(c)?'两个问题使用同一组五条记录，读出点各在最后一个输入 token。':'四个位置来自同一道代码查找题，按实际 token 边界定位。';
    const text=isHead(c)?'注意力描述查询位置对各条记录分配的权重。Z 是同一个头汇聚值向量后的 128 维输出。':
      'B36 的 FFN 以 SiLU(gate) × up 写出中间激活，保存了每个位置的全部 '+d.ffn.width.toLocaleString('en-US')+' 个原生单元。';
    const originals=isHead(c)?'<div class="et-pair">'+[a,b].map((x,i)=>'<article><b>'+['A','B'][i]+' · '+esc(x.rule)+'</b><p lang="en">'+esc(x.question)+'</p>'+sourceLink(x.record,'查看这道原题')+'</article>').join('')+'</div>':
      '<div class="et-pair"><article><b>A · '+esc(a.label)+'</b><p>token '+a.index+' · 词表 ID '+a.tokenId+'</p></article><article><b>B · '+esc(b.label)+'</b><p>token '+b.index+' · 词表 ID '+b.tokenId+'</p></article></div>'+sourceLink(r,'查看同题材料');
    return heading(scope(c)+' 的读出边界',text)+originals+'<p class="et-caption">'+esc(names)+'</p>'+provenance(c);
  }
  function results(c) {
    if(isHead(c)){
      const pair=[sides(c)[c.a],sides(c)[c.b]];
      const lines=pair.map((row,i)=>{
        const winner=row.rows[row.winner],target=row.rows[row.gold];
        const rate=data().head.cohort.fractions[i===0?c.a:c.b];
        return '<tr><th scope="row">'+['A','B'][i]+' · '+esc(row.rule)+'</th><td>第 '+(row.gold+1)+' 行 · '+esc(target.value)+'</td><td>第 '+(row.winner+1)+' 行 · '+esc(winner.value)+'</td><td><code>'+esc(row.output)+'</code></td><td>'+(100*rate).toFixed(1)+'%</td></tr>';
      });
      const different=pair.find(r=>r.rows[r.winner].value!==r.output);
      const finding=different?'「'+different.rule+'」下，最高行对应 '+different.rows[different.winner].value+'，模型却输出 '+different.output+'。这一题推翻了“每次都由最高行决定答案”的预测。':
        c.a===c.b?'A 与 B 选中了同一题，只能复核本条读数。':'这两条记录均与预测相符。「按物理首行」条件提供了可区分该解释的反例。';
      return heading('规则、预测与模型输出','按“总是采用注意力最高行的值”逐题预测，结果与模型实际输出对照。')+
        '<div class="et-scroll"><table><thead><tr><th>条件</th><th>规则指定</th><th>最高行预测</th><th>模型输出</th><th>确认集选行命中率</th></tr></thead><tbody>'+lines.join('')+'</tbody></table></div>'+
        '<div class="et-finding"><strong>对照结果</strong><p>'+esc(finding)+'</p></div>'+
        '<p class="et-caption">这个头仍可能参与计算。确认哪条通路影响输出，需要对同一模型做匹配干预。</p>';
    }
    const a=field(c,c.a)[c.index],b=field(c,c.b)[c.index];
    const reading=c.a===c.b?'A 与 B 指向同一个位置。选择另一个 token 可检验位置变化。':a===b?'两个已选位置的原值相同。':
      '同一单元在「'+sideName(c,c.a)+'」为 '+val(a)+'，在「'+sideName(c,c.b)+'」为 '+val(b)+'。';
    return heading('两个位置的原始激活','固定模型、题目、块与单元，读取两个指定 token 的原始激活。')+
      '<div class="et-pair">'+[c.a,c.b].map((i,n)=>'<article><b>'+['A','B'][n]+' · '+esc(sideName(c,i))+'</b><p class="et-reading">'+val(field(c,i)[c.index])+'</p><small>token '+position(c,i)+' · U'+c.index+'</small></article>').join('')+'</div>'+
      '<div class="et-finding"><strong>对照结果</strong><p>'+esc(reading)+'</p></div><p class="et-caption">同题各位置的前缀和 token 也在变化。固定位置效应、计数语义和单元的因果作用各自需要独立对照。</p>';
  }
  function crossFamily(c){
    if(isHead(c)){
      const group=data().head.cohort;
      const conditions=data().headCases.map((s,i)=>s.rule+' '+(100*group.fractions[i]).toFixed(1)+'%');
      return '<details class="et-source"><summary>对照 '+group.familyCount+' 个确认家族</summary><p class="et-caption">每个家族在四种交换条件下分别问三种规则，每种规则共 '+(group.caseCount/group.fractions.length)+' 题。目标行得到最高注意力的比例为 '+esc(conditions.join('、'))+'。模型答案的正确率另行计分。</p></details>';
    }
    const d=data().ffn;
    return '<details class="et-source"><summary>对照已有的组阻断测试</summary><p class="et-caption">固定 '+d.groupSize+' 个单元组成一组，在 '+d.total+' 个配对家族上，基线答对 '+d.baseCorrect+' 个，阻断后答对 '+d.blockedCorrect+' 个。对单个 U'+c.index+' 的必要性需要另行取证。</p></details>';
  }
  function selector(label,name,options,selected) {
    return '<label>'+esc(label)+' <select data-et="'+name+'" aria-label="'+esc(label)+'">'+options.map(([v,t])=>
      '<option value="'+esc(v)+'"'+(String(v)===String(selected)?' selected':'')+'>'+esc(t)+'</option>').join('')+'</select></label>';
  }
  function pluginToolbar(){
    const list=['Head','MLP','HiddenState','lens',...enabled];
    return '<div class="et-toolnav" role="group" aria-label="分析插件">'+list.map(k=>
      '<button type="button" data-et-tool="'+k+'" aria-pressed="'+(currentTool===k)+'">'+titles[k]+'</button>').join('')+
      '<details class="et-manage"><summary>管理插件</summary><div>'+extras.map(k=>'<label><input type="checkbox" data-et-enable="'+k+'" '+(enabled.has(k)?'checked':'')+'> '+titles[k]+'</label>').join('')+'</div></details></div>';
  }
  function changeTool(host,c) {
    const panel=host.querySelector('[data-et-panel]');
    panel.innerHTML='';
    if(!ready(c)){panel.innerHTML=missing(c);return;}
    const d=data();
    if(currentTool==='lens'){
      panel.innerHTML=notice('Logit Lens 需要输出读出权重','逐层词表预测需要同一模型快照的最终归一化参数、词表输出矩阵和相应 HiddenState。当前样例没有附带完整读出权重。');
      return;
    }
    if((['Head','attention'].includes(currentTool)&&!isHead(c))||(['MLP','HiddenState','geometry'].includes(currentTool)&&isHead(c))){
      panel.innerHTML=notice('当前材料缺少这项测量',isHead(c)?'本题保存了最后输入位置的行注意力与头输出 Z。查看 HiddenState 或 FFN 可切换到 B36 的位置对照题。':'本题保存了多个位置的 HiddenState 和 FFN 激活。查看逐行注意力可切换到 B26/H36 的条件选行题。');
      return;
    }
    const sideOptions=[['a','A · '+sideName(c,c.a)],['b','B · '+sideName(c,c.b)]];
    if(currentTool==='Head'||currentTool==='MLP'){
      let controls=selector('查看条件','side',sideOptions,currentSide);
      if(currentTool==='Head')controls+=selector('头内测量','boundary',[['Z','Z · 输出投影前'],['Q','Q · 查询'],['K','K · 键'],['V','V · 值']],headBoundary);
      panel.innerHTML='<div class="et-controls">'+controls+'</div><div data-et-field></div>';
      widgetHost=panel.querySelector('[data-et-field]');
      widget=window.AI2050NativeField.create(widgetHost,{onPick:i=>context.action('index',i)});
      return;
    }
    if(currentTool==='HiddenState'){
      panel.innerHTML='<div class="et-controls">'+selector('读取层','layer',Array.from({length:d.ffn.hiddenLayers},(_,i)=>[i,i===0?'L0 · 嵌入':'L'+i]),hiddenLayer)+
        '<label>残差流坐标 <input aria-label="残差流坐标" data-et="coord" type="number" step="1" min="0" max="'+(d.ffn.hiddenWidth-1)+'" value="'+hiddenCoord+'"></label></div><div data-et-hidden></div>';
      drawHidden(panel,c);
      return;
    }
    if(currentTool==='attention'){
      const rowStats=i=>{
        const masses=d.headCases[i].rows.map(r=>r.mass);
        const other=1-masses.reduce((s,v)=>s+v,0);
        if(other<0 || !window.AI2050AttentionStats)return null;
        return window.AI2050AttentionStats.summarize([...masses,other]);
      };
      panel.innerHTML=heading('注意力分布','按五条正文记录及其余 token 合为六组，用熵比较来源分配的集中程度。')+
        '<div class="et-pair">'+[c.a,c.b].map((i,n)=>{const stats=rowStats(i);return '<article><b>'+['A','B'][n]+' · '+esc(sideName(c,i))+'</b><p class="et-reading">'+(stats?.ok?rounded(stats.entropyBits)+' bit':'缺读数')+'</p></article>';}).join('')+'</div>';
      return;
    }
    const opts=Array.from({length:d.ffn.hiddenLayers},(_,i)=>[i,'L'+i]);
    panel.innerHTML=heading('不同层保留了多少位置关系？','使用该题全部四个已采集位置逐一配对，每处保留 '+d.ffn.hiddenWidth+' 个坐标。')+
      '<div class="et-controls">'+selector('第一层','layer',opts,hiddenLayer)+selector('第二层','geometry-layer',opts,geometryLayer)+'</div><div data-et-geometry></div>';
    drawGeometry(panel);
  }
  function paintField(c) {
    if(!widget)return;
    const i=currentSide==='b'?c.b:c.a, isH=isHead(c), missingZ=isH&&headBoundary!=='Z';
    widget.update({identity:[c.q,c.block,c.head,i,headBoundary,c.materialKey].join('/'),kind:isH?'head':'ffn',
      size:isH?data().head.width:data().ffn.width,values:missingZ?null:field(c,i),selected:c.index,
      source:sideName(c,i)+' · token '+position(c,i)+' · '+record(c,i).caseId,measure:isH?headBoundary:'SiLU(gate) × up',
      vectorLabel:isH?'B'+c.block+'/H'+c.head+' · '+({Q:'查询通道',K:'键通道',V:'值通道',Z:'头内输出'})[headBoundary]:'B'+c.block+' · FFN 中间单元',missing:missingZ?'该历史题只保存了 Z 的逐通道向量。所选边界未采集。':'',
      comparisons:!isH?sides(c).map((s,j)=>({label:s.label+' · token '+s.index,value:field(c,j)[c.index],active:i===j})):null});
  }
  function drawHidden(panel,c) {
    const d=data(), n=d.ffn.hiddenLayers,i=hiddenCoord;
    const series=[c.a,c.b].map(site=>Array.from({length:n},(_,l)=>d.hiddenValues(l,site)?.[i]));
    const host=panel.querySelector('[data-et-hidden]');
    if(series.some(s=>s.some(x=>!Number.isFinite(x)))){host.innerHTML=notice('坐标缺少完整原场','请核对位置和坐标。');return;}
    const low=Math.min(0,...series.flat()),high=Math.max(0,...series.flat()),span=high-low||1;
    const x=j=>12+((j/(n-1))*596),y=v=>70-(v-low)/span*58;
    host.innerHTML='<p class="et-caption">同一残差流坐标在 '+n+' 个层边界的原值。L0 为词嵌入，L1 起为块输出。</p>'+
      '<svg class="et-trace" viewBox="0 0 622 90" role="img" aria-label="两个位置逐层的残差流原值">'+
      '<path d="M12 '+y(0)+'H608" stroke="#cbd7df"/><path d="M'+x(hiddenLayer)+' 7V72" stroke="#8d9ba5" stroke-dasharray="3 3"/>'+
      series.map((s,k)=>'<polyline fill="none" stroke="'+['#3266a7','#bd643f'][k]+'" stroke-width="2" points="'+s.map((v,l)=>x(l)+','+y(v)).join(' ')+'"/>').join('')+
      '<text x="12" y="86">L0</text><text x="585" y="86">L'+(n-1)+'</text></svg>'+
      '<div class="et-pair">'+series.map((s,k)=>'<article><b>'+['A','B'][k]+' · '+esc(sideName(c,k?c.b:c.a))+'</b><p class="et-reading">'+val(s[hiddenLayer])+'</p><small>L'+hiddenLayer+' · 坐标 '+i+'</small></article>').join('')+'</div>'+
      '<p class="et-caption">同一编号跨层可追踪，功能是否相同仍需验证。</p>';
  }
  function drawGeometry(panel){
    const d=data(),rows=l=>d.ffn.positions.map((_,i)=>d.hiddenValues(l,i));
    const r=window.AI2050RepresentationGeometry?.compare(rows(hiddenLayer),rows(geometryLayer));
    const target=panel.querySelector('[data-et-geometry]');
    if(!r||r.error){target.innerHTML=notice('缺少层间比较输入',r?.error||'几何插件未载入。');return;}
    target.innerHTML='<div class="et-pair"><article><b>线性 CKA</b><p class="et-reading">'+rounded(r.cka)+'</p></article><article><b>RSA · Spearman ρ</b><p class="et-reading">'+rounded(r.rsa)+'</p></article></div><p class="et-caption">四个位置均来自同一题。零方差或距离排序退化时显示未定义。</p>';
  }
  function cleanWidget(){widget?.destroy();widget=null;widgetHost=null;}
  function measurement(c) {
    const r=record(c,c.a), states=[c.a,c.b].map(site=>sideName(c,site)+' · token '+position(c,site));
    const angle=c.angle ?? (isHead(c)?'z':'activation'),dim=c.dim??0,sites=[c.a,c.b];
    let native,name,details=['',''];
    if(angle==='rows'){
      const pair=sites.map(i=>sides(c)[i]);
      native=pair.map(row=>row.rows[row.winner].mass);
      name=scope(c)+' · 最高行注意力';
      details=pair.map(row=>'最高行 '+(row.winner+1)+' · '+row.rows[row.winner].value+'，目标行 '+(row.gold+1)+'，模型输出 '+row.output);
    }else if(angle==='hidden'){
      const layer=data().ffn.hiddenLayers-1;
      native=sites.map(site=>data().hiddenValues(layer,site)[dim]);
      name='HiddenState · L'+layer+' / d'+dim;
      details=sites.map(()=>'跨层轨迹末点 · 最后模型块输出，最终归一化前');
    }else{
      native=sites.map(site=>field(c,site)[c.index]);
      name=isHead(c)?scope(c)+' / D'+c.index+' · Z':scope(c)+' · 门控后激活';
    }
    return {q:c.q,mode:c.mode,tab:'evidence',angle,dim,a:c.a,b:c.b,index:c.index,block:c.block,head:c.head,materialKey:c.materialKey,
      model:r.model,snapshot:r.snapshot,run:r.run,records:[record(c,c.a).id,record(c,c.b).id],states,native,name,details,
      sources:(isHead(c)?data().head:data().ffn).sources.filter(s=>s.sha256).map(s=>({path:s.path,sha256:s.sha256}))};
  }
  const noteKey = m => JSON.stringify([m.snapshot,m.run,m.records,m.q,m.a,m.b,m.block,m.head,m.index,m.angle,m.angle==='hidden'?m.dim:null]);
  function savedNotes(){
    return read(observationKey).filter(n=>n && ['head','ffn'].includes(n.q) && Array.isArray(n.states) && n.states.length===2 &&
      Array.isArray(n.native)&&n.native.length===2 && Array.isArray(n.records)&&typeof n.name==='string'&&typeof n.key==='string').map(n=>{
        if(n.angle)return n;
        // Earlier previews always saved Z or the gated FFN value.
        const migrated={...n,angle:isHead(n)?'z':'activation',dim:0,details:['','']};
        return {...migrated,key:noteKey(migrated)};
      });
  }
  function validNote(n){
    if(!n)return false;
    const src=n.q==='head'?data().headCases:data().ffn.positions;
    if(!id(n.a,src.length)||!id(n.b,src.length)||!id(n.index,n.q==='head'?data().head.width:data().ffn.width))return false;
    if(!(isHead(n)?['rows','z']:['hidden','activation']).includes(n.angle)||
      n.angle==='hidden'&&!id(n.dim,data().ffn.hiddenWidth))return false;
    const r=record(n,n.a);
    if(n.snapshot!==r.snapshot||n.run!==r.run||n.records[0]!==r.id||n.records[1]!==record(n,n.b).id||
      n.block!==(n.q==='head'?data().head.block:data().ffn.block)||(isHead(n)&&n.head!==data().head.index))return false;
    const now=measurement(n);
    return Array.isArray(n.sources) && n.sources.length===now.sources.length &&
      n.sources.every(source=>source && now.sources.some(current=>current.path===source.path&&current.sha256===source.sha256)) &&
      now.native.every((v,i)=>v===n.native[i]) && Array.isArray(n.details) && now.details.every((s,i)=>s===n.details[i]);
  }
  function evidencePanel(c){
    const saved=savedNotes();
    const r=ready(c)?record(c,c.a):null;
    let current=missing(c);
    if(r){
      const m=measurement(c), existing=saved.find(n=>n.key===noteKey(m));
      current=heading('保留这次观察',m.name+' · '+m.states[0]+' / '+m.states[1])+
        '<div class="et-pair">'+m.native.map((v,i)=>'<article><b>'+['A','B'][i]+' · '+(m.angle==='rows'?'最高行质量':'原值')+'</b><p class="et-reading">'+val(v)+'</p>'+(m.details[i]?'<small>'+esc(m.details[i])+'</small>':'')+'</article>').join('')+'</div>'+
        '<form data-et-note><label>观察笔记 <input name="memo" maxlength="320" value="'+esc(existing?.note)+'" placeholder="记录想复核的现象（选填）"></label><button type="submit">'+(existing?'更新笔记':'保存到本机')+'</button></form>'+
        '<p class="et-caption">按模型快照、两份读出和对象编号保存。笔记用于回访当前观察，正式证据须通过研究归档流程。</p>'+provenance(c);
    }
    const history=saved.length?'<h4>已存观察</h4><ul class="et-notes">'+saved.map((n,i)=>{
      const valid=validNote(n);
      return '<li><div><b>'+esc(n.name)+'</b><p>'+esc(n.states.join(' / '))+'</p>'+(n.note?'<p>'+esc(n.note)+'</p>':'')+'</div><button type="button" data-et-restore="'+i+'" '+(valid?'':'disabled')+'>'+(valid?'恢复观察':'来源待核对')+'</button></li>';
    }).join('')+'</ul>':'';
    return current+'<p class="et-status" role="status">'+esc(status)+'</p>'+history;
  }
  function render(target,c) {
    if(host!==target){cleanWidget();domKey='';}
    host=target;context=c;
    if(lastQuestion!==c.q){currentTool=isHead(c)?'Head':'MLP';lastQuestion=c.q;}
    const key=[c.tab,c.q,c.a,c.b,c.block,c.head,c.materialKey,c.matched,c.liveEmpty,currentTool,...enabled].join('/');
    const stable=c.tab==='tools'&&key===domKey&&widgetHost?.isConnected;
    if(stable){paintField(c);return;}
    domKey=key;
    if(c.tab!=='evidence')status='';
    cleanWidget();
    host.onclick=onClick;host.onchange=onChange;host.onsubmit=onSubmit;
    const markup=c.tab==='tools'?pluginToolbar()+'<div data-et-panel></div>':c.tab==='object'?objectPanel(c):
      c.tab==='evidence'?evidencePanel(c):ready(c)?results(c)+crossFamily(c):missing(c);
    host.innerHTML='<div class="et-root">'+markup+'</div>';
    if(c.tab==='tools'){changeTool(host,c);paintField(c);}
  }
  function onClick(event){
    const button=event.target.closest('button');
    if(!button)return;
    if(button.hasAttribute('data-et-tool')){
      currentTool=button.dataset.etTool;render(host,context);host.querySelector('[data-et-tool="'+currentTool+'"]')?.focus();
    }
    if(button.hasAttribute('data-et-restore')){
      const n=savedNotes()[Number(button.dataset.etRestore)];
      if(validNote(n)){
        status='已恢复两份读出、对象编号和原值。';
        context.action('restore',{q:n.q,mode:n.mode==='features'?'features':'archive',a:n.a,b:n.b,index:n.index,block:n.block,head:n.head,angle:n.angle,dim:n.dim,tab:'evidence',materialKey:n.materialKey});
      }
    }
  }
  function onChange(event){
    const target=event.target, key=target.dataset.et;
    if(target.hasAttribute('data-et-enable')){
      const k=target.dataset.etEnable;
      if(target.checked)enabled.add(k);else enabled.delete(k);
      if(!target.checked&&currentTool===k)currentTool=isHead(context)?'Head':'MLP';
      try{localStorage.setItem(enabledKey,JSON.stringify([...enabled]));}catch{/* session availability only */}
      render(host,context);
      const details=host.querySelector('.et-manage');details.open=true;
      details.querySelector('[data-et-enable="'+k+'"]').focus();
    }
    if(key==='side'){currentSide=target.value;paintField(context);}
    if(key==='boundary'){headBoundary=target.value;paintField(context);}
    if(key==='layer'||key==='geometry-layer'||key==='coord'){
      const d=data(),value=Number(target.value),max=key==='coord'?d.ffn.hiddenWidth:d.ffn.hiddenLayers;
      if(!target.value||!id(value,max)){target.reportValidity();return;}
      if(key==='layer')hiddenLayer=value;
      else if(key==='geometry-layer')geometryLayer=value;
      else hiddenCoord=value;
      if(currentTool==='HiddenState')drawHidden(host,context);else drawGeometry(host);
    }
  }
  function onSubmit(event){
    if(!event.target.matches('[data-et-note]'))return;
    event.preventDefault();
    if(!ready(context))return;
    const m=measurement(context);
    const key=noteKey(m);
    const newNote={...m,key,savedAt:new Date().toISOString(),note:String(new FormData(event.target).get('memo')||'').trim()};
    try{
      localStorage.setItem(observationKey,JSON.stringify([newNote,...savedNotes().filter(n=>n.key!==key)]));
      status='已保存，可从下方恢复观察。';
    }catch{status='当前浏览器未允许本地存储，请核对存储设置。';}
    render(host,context);
  }
  function clear(){
    cleanWidget();
    if(host){host.onclick=null;host.onchange=null;host.onsubmit=null;host.replaceChildren();}
    host=null;context=null;domKey='';lastQuestion='';status='';
  }
  function openNative(){
    if(!host || !context || context.tab!=='tools')return;
    currentTool=isHead(context)?'Head':'MLP';
    headBoundary='Z';currentSide='a';domKey='';
    render(host,context);
  }
  window.EncodingTools={render,clear,openNative};
})();
