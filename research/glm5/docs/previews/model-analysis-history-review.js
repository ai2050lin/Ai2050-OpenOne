(() => {
  const data = window.historyReviewData;
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const percent = value => (100*value).toFixed(1)+'%';
  const records = [
    {phase: 2983, title: '按姓名、印刷标签还是物理首行选源', status: '反例', level: 'negative',
     relation: '同一头 · 已测输入', head: true,
     summary: '首行题答对 silver，H36 的最大记录注意力落在第 4 行。',
     goal: '分别交换姓名、印刷 ROW 和物理位置，检查三种线索怎样影响选行。',
     design: '同一个家族内控制表格和记录身份，改变查询线索或交换位置。当前摘录中的三道题只换问法。',
     limit: '注意力最高行有时不同于答案来源。Q/K 候选未进正式原模型干预。',
     source: 'clt_b26_three_cues_v3 · confirm-000-00 / 01 / 02', dialog: 'head'},
    {phase: 2971, title: '四记录选源与 B26/H36 写入', status: '局部证据', level: 'positive',
     relation: '同一头 · 不同材料', head: true,
     summary: '留出选行一致率 94.48%，同预算随机为 22.17%。',
     goal: '冻结紧凑 QKV 候选，预测该头的选行、写入和同层 MLP 响应。',
     design: '独立家族比较原生坐标与同预算随机，原生阻断配同输入恢复。',
     limit: '注意力优势受到干预影响，最终答案首选保持。下游预测使用真实其他头和残差背景。',
     source: 'clt_b26_compact_v1 · Phase 2971'},
    {phase: 3004, title: '旧行号与当前姓名冲突', status: '路线已结案', level: 'closed',
     relation: '候选交接路径 · 行为停门', head: true,
     summary: '旧规则在冲突表中 27/256，现名规则 252/256。',
     goal: '检验旧行与当前姓名冲突时，模型能否按两条规则分别取值。',
     design: '128 个新家族。分开测旧行计算、干扰、显式行位取值和完整问答。',
     limit: '旧规则自然行为不可靠。B25H18 → B26H36 的候选交接干预未执行。仅换人名或题量沿用结案。',
     source: data.old[0].run_id, dialog: 'old'},
    {phase: 2981, title: '档案样式会改变答案取自哪里吗', status: '行为门未过', level: 'negative',
     relation: '同一头仅列为候选 · 不同任务', head: true,
     summary: '4,096 条确认答案。对象与数字可分别取自档案和修订。',
     goal: '独立改变档案样式、内容与指定来源，检验完整自由生成。',
     design: '两个新句式、256 个独立确认家族，分开评分对象和数字。',
     limit: '预注册效应方向失败，B26/H36 的阻断与原生恢复均未启动。本条没有该头的干预读数。',
     source: 'content_origin_20261118_v1 · Phase 2981'},
    {phase: 3053, title: '整层语料热图与 U7708 两端原文', status: '解释留出未复现', level: 'negative',
     relation: '同一单元 · 语料观察', unit: true,
     summary: '档案原值可回查，候选词在留出高响应片段命中 0/4。',
     goal: '检查选中神经元、对应语料原文和数值能否回到同一来源。',
     design: '复用已发布的 B23 原生 BF16 档案。正负两端采用原留出语料固定读出。',
     limit: '两端同时改变语种、话题与上下文。功能解释及跨层复用尚未取得这里所需的独立证据。',
     source: data.unit.run, dialog: 'unit'},
  ];
  const modeTab = {live:'run',history:'history',features:'object'};
  const addresses = {head:{kind:'head',block:26,index:36}, unit:{kind:'unit',block:23,index:7708}};
  const state = {mode:'history',tab:'history',object:{...addresses.head},caseIndex:2,pole:0,phase:2983,full:false,
                 plugin:'heads',draft:'names',why:'',liveInput:''};
  const idOf = o => 'B'+o.block+'/'+(o.kind==='head'?'H':'U')+o.index;
  const isHead = () => idOf(state.object)===idOf(addresses.head);
  const isUnit = () => idOf(state.object)===idOf(addresses.unit);
  const headCase = () => data.head.cases[state.caseIndex];
  const row = () => data.unit.rows[state.pole];
  const caseRules = ['按姓名','按印刷 ROW','按物理首行'];
  const selectedRecord = () => records.find(r=>r.phase===state.phase);
  const mark = r => '<span class="badge '+r.level+'">'+r.status+'</span>';
  const scene = window.createModelScene($('structure'), {
    onSelect: (block,index) => chooseObject({kind:'head',block,index}),
    onSelectUnit: (block,index) => chooseObject({kind:'unit',block,index}),
    onSelectBlock: block => chooseObject({...state.object,block}),
  });
  function recordRelation(record) {
    return record.head && isHead() || record.unit && isUnit();
  }
  function chooseObject(object, focusHistory=false) {
    state.object = object;
    if (isHead()) state.phase = 2983;
    else if (isUnit()) state.phase = 3053;
    if (focusHistory) {
      state.tab = 'history';
      $('linked-only').checked = true;
      $('search').value = '';
      $('filter').value = 'all';
    }
    if (isHead() || isUnit()) state.plugin = isHead()?'heads':'mlp';
    render();
  }
  function renderScene() {
    const o = state.object;
    $('scene-address').textContent = idOf(o);
    $('scene-context').textContent = isHead() ? headCase().id+' · 末位读出' : isUnit() ? '留出语料 · token '+row().position : '结构定位 · 本稿未载入响应';
    const choice = $('object');
    choice.querySelector('option[value="picked"]')?.remove();
    if (isHead() || isUnit()) choice.value = isHead()?'head':'unit';
    else {
      choice.add(new Option(idOf(o)+' · 当前选择','picked'));
      choice.value = 'picked';
    }
    scene.update({model:'Qwen3-14B',layers:40,heads:40,width:17408,block:o.block,
      mode:o.kind==='head'?'head':'ffn',head:o.kind==='head'?o.index:null,unit:o.kind==='unit'?o.index:null});
  }
  function metrics(firstLabel,first,secondLabel,second) {
    return '<dl class="metric-pair"><div><dt>'+firstLabel+'</dt><dd>'+first+
      '</dd></div><div><dt>'+secondLabel+'</dt><dd class="secondary">'+second+'</dd></div></dl>';
  }
  function segments(labels, active, action) {
    return '<div class="segments">'+labels.map((label,i)=>'<button data-act="'+action+'" data-value="'+i+
      '" aria-pressed="'+(active===i)+'">'+label+'</button>').join('')+'</div>';
  }
  function bars(c) {
    const peak = Math.max(...c.mass);
    return '<div class="bars" aria-label="各物理记录行的注意力质量">'+c.order.map((record,i)=>{
      const value=c.mass[record], color=record===c.gold?'gold':value===peak?'peak':'';
      return '<div class="bar-row"><span>第 '+(i+1)+' 行</span><span class="bar-track"><i class="'+color+
        '" style="width:'+percent(value)+'"></i></span><output>'+percent(value)+'</output></div>';
    }).join('')+'</div>';
  }
  function inspector() {
    const el = $('inspector');
    if (isHead()) {
      const c=headCase(), gold=c.order.indexOf(c.gold)+1, peak=c.order.indexOf(c.mass.indexOf(Math.max(...c.mass)))+1;
      el.innerHTML = '<p class="eyebrow">Phase 2983 · 同表条件对照</p><h2>题目要求与头部注意力</h2>'+
        segments(caseRules,state.caseIndex,'case')+metrics('模型实答',c.output,'注意力峰值','第 '+peak+' 行')+
        bars(c)+'<p class="caption">题目目标是第 '+gold+' 行 · B26/H36 最后输入 token</p>'+
        '<div><button class="button link-button" data-act="open-case" data-value="head">查看原题和行表 →</button></div>';
    } else if (isUnit()) {
      el.innerHTML = '<p class="eyebrow">Phase 3053 · 既有语料档案</p><h2>同一单元的两个留出端点</h2>'+
        segments(['正端语境','负端语境'],state.pole,'pole')+
        metrics('当前 BF16 激活',String(row().activation),'候选词留出复查','0 / 4')+
        '<blockquote>'+esc(row().text)+'</blockquote><p class="caption">'+esc(row().document)+' · token '+row().position+
        '</p><div><button class="button link-button" data-act="open-case" data-value="unit">查看读数、原文和来源 →</button></div>';
    } else {
      el.innerHTML = '<p class="eyebrow">结构定位</p><h2>'+idOf(state.object)+'</h2><p class="caption">本评审包没有这一位置的原生响应。可切回已有摘录定位。</p>'+
        '<div><button class="button compact" data-act="locate" data-value="head">打开 B26/H36 档案</button></div>';
    }
  }
  function listRecords() {
    const query = $('search').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const filter = $('filter').value;
    const matches = records.filter(r => (!$('linked-only').checked || recordRelation(r)) &&
      (filter==='all' || r.level===filter) &&
      query.every(q=>[r.title,r.summary,r.phase,r.status,r.source,r.relation,r.head?'B26/H36':'B23/U7708'].join(' ').toLowerCase().includes(q)));
    if (!matches.some(r=>r.phase===state.phase)) state.phase=matches[0]?.phase ?? null;
    $('history-scope').textContent = $('linked-only').checked
      ? idOf(state.object)+' · 同一对象的实测、相关候选与停门记录'
      : '当前评审包 · 按机制与测试证据检索';
    $('match-count').textContent = matches.length+' 条摘录';
    $('history-list').innerHTML = matches.length
      ? matches.map(r=>'<button class="record-item" data-act="choose-record" data-value="'+r.phase+'" aria-pressed="'+(r.phase===state.phase)+
        '"><span class="item-title"><span><small>'+r.phase+'</small><b>'+r.title+'</b></span>'+mark(r)+
        '</span><span class="caption">'+r.summary+'</span></button>').join('')
      : '<p class="empty">当前筛选没有匹配的摘录。</p><div class="empty"><button class="button compact" data-act="reset-search">清除筛选</button></div>';
    const selected = selectedRecord();
    $('record-detail').innerHTML = selected ?
      '<p class="eyebrow">Phase '+selected.phase+' · '+selected.relation+'</p><h3>'+selected.title+'</h3><dl>'+
      '<div><dt>目标</dt><dd>'+selected.goal+'</dd></div>'+
      '<div><dt>设计</dt><dd>'+selected.design+'</dd></div>'+
      '<div><dt>实际结果</dt><dd>'+selected.summary+'</dd></div>'+
      '<div><dt>反例或未过门原因</dt><dd>'+selected.limit+'</dd></div></dl>'+
      '<footer>'+(selected.dialog?'<button class="button compact" data-act="open-case" data-value="'+selected.dialog+'">查原题 / 原文</button> ':'')+
      '<button class="button compact" data-act="review" data-value="'+selected.phase+'">与旧设计核对</button></footer>'+
      '<p class="source">'+selected.source+'</p>'
      : '<p class="empty">选择一条已有记录，查看目标、设计、结果和适用边界。</p>';
  }
  const plugins = [['hidden','HiddenState'],['mlp','MLP'],['heads','Head'],['lens','Logit Lens']];
  function renderTool() {
    const panel = $('tool-panel'), o = idOf(state.object);
    if (state.tab==='object') {
      panel.innerHTML='<h2>对象详情 · '+o+'</h2><p>静态权重与本次响应使用同一原生索引。</p>'+
        (isUnit()?'<p>本档案记录门控乘积 SiLU(gate) × up。两段语料各有自己的 token 位置。</p>':
          isHead()?'<p>本档案记录最后输入 token 对五条记录的注意力质量。Q/K 选源与 V/O 写回有各自的读数。</p>':
          '<p>已定位该原生索引。当前评审包未载入该位置的逐题数据。</p>')+
        '<p class="source">模型快照 '+data.snapshot+'</p><button class="button" data-act="history">回查此对象的测试记录</button>';
    } else if (state.tab==='run') {
      panel.innerHTML='<h2>运行与对照</h2>'+
        (state.mode==='live'?'<p>本机输入区 · 本稿仅显示交互位置。可选用已保存样本查看联动。</p>'+
        '<label for="live-input">输入文本</label><textarea id="live-input" placeholder="输入一段文本">'+esc(state.liveInput)+'</textarea><p class="caption">静态评审不加载模型。</p>':
          '<p>同一档案改问法，比较输出和选行变化。</p>'+segments(caseRules,state.caseIndex,'case'))+
        '<button class="button" data-act="open-case" data-value="'+(isUnit()?'unit':'head')+'">打开已有样本</button> '+
        '<button class="button" data-act="history">先查相似历史</button>';
    } else {
      const switcher='<div class="segments">'+plugins.map(([key,label])=>'<button data-act="plugin" data-value="'+key+
        '" aria-pressed="'+(key===state.plugin)+'">'+label+'</button>').join('')+'</div>';
      const heat = isUnit() && state.plugin==='mlp' ? '<p>同一语料位置 · U'+data.unit.first+'–U'+(data.unit.first+row().values.length-1)+'</p>'+
        '<div class="heat-line" aria-label="原生激活局部热图">'+row().values.map((v,i)=>{
          const shade = Math.min(Math.abs(v)*0.5,0.86)+0.04;
          return '<span class="'+(data.unit.first+i===data.unit.index?'chosen':'')+'" style="background:'+
            (v>=0?'rgba(37,99,192,':'rgba(193,103,32,')+shade+')" title="U'+(data.unit.first+i)+' · '+v+'"></span>';
        }).join('')+'</div><p class="caption">蓝为正，橙为负。框选 U7708，实际值 '+row().activation+'。</p>'
        : isHead() && state.plugin==='heads'?'<p>B26/H36 · 五行注意力质量</p>'+bars(headCase())
        : '<p class="caption">该样本的这个插件字段没有被载入。切到 Head 或 MLP 的对应存档样本可查看实测值。</p>';
      panel.innerHTML='<h2>插件分析 · '+o+'</h2>'+switcher+heat+
        '<button class="button compact" data-act="history">查这个读数关联的历史</button>';
    }
  }
  function render() {
    renderScene();
    inspector();
    for (const b of $('modes').querySelectorAll('button')) b.setAttribute('aria-pressed',b.dataset.mode===state.mode);
    for (const b of $('console-tabs').querySelectorAll('button')) b.setAttribute('aria-pressed',b.dataset.tab===state.tab);
    $('history-panel').hidden = state.tab!=='history';
    $('tool-panel').hidden = state.tab==='history';
    if (state.tab==='history') listRecords();
    else renderTool();
  }
  function openDialog(dialog) {
    if (!dialog.open) dialog.showModal();
  }
  function source(name) {
    const s=data.sources[name];
    return '<p class="source">'+esc(s.path)+'<br>SHA-256 '+s.sha256+'</p>';
  }
  function openCase(kind) {
    const dialog=$('case-dialog'), c=headCase();
    let title, readout;
    if (kind==='head') {
      title='原题与模型实答';
      const max=Math.max(...c.mass);
      readout='<p class="eyebrow">Phase 2983 · '+c.id+' · 最后输入 token</p>'+
        '<blockquote>'+esc(c.question)+'</blockquote>'+
        metrics('模型实答',c.output,'题目答案',data.head.codes[c.gold])+
        '<table><thead><tr><th>物理行</th><th>ROW</th><th>姓名</th><th>CODE</th><th>注意力质量</th></tr></thead><tbody>'+
        c.order.map((i,k)=>'<tr class="'+(i===c.gold?'gold-record':c.mass[i]===max?'peak-record':'')+'"><td>'+(k+1)+'</td><td>'+
          esc(c.labels[i])+'</td><td>'+esc(data.head.names[i])+'</td><td>'+esc(data.head.codes[i])+'</td><td>'+percent(c.mass[i])+
          (i===c.gold?' · 目标':'')+(c.mass[i]===max?' · 峰值':'')+'</td></tr>').join('')+
        '</tbody></table><details><summary>实际模型输入与来源</summary><blockquote><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+esc(c.prompt)+
        '</pre></blockquote>'+source('prompt')+source('head')+'</details>';
    } else if (kind==='old') {
      title='旧行号路线的真实反例';
      const tasks=['仅旧表报行','增加现行表后报旧行','显式给行，取代号','完整旧规则','完整现名规则'];
      readout='<p class="eyebrow">Phase 3004 · 128 个新家族 · 路线结案</p>'+
        '<p>完整旧规则在冲突表中答对 27/256，完整现名规则为 252/256。旧规则行为未达 85% 门，候选交接干预未执行。</p>'+
        '<table><thead><tr><th>家族 000 的对照</th><th>目标</th><th>模型实答</th></tr></thead><tbody>'+
        data.old.map((r,i)=>'<tr><td>'+tasks[i]+'<br><small>'+r.id+'</small></td><td>'+esc(r.gold_token)+'</td><td>'+esc(r.top_token)+'</td></tr>').join('')+
        '</tbody></table>'+source('old');
    } else {
      title='U7708 的留出语料原文';
      readout='<p class="eyebrow">Phase 3053 · '+data.unit.run+'</p>'+data.unit.rows.map(r=>
        '<blockquote>'+esc(r.text)+'<br><span class="source">'+esc(r.document)+' · token '+r.position+' · BF16 '+r.activation+'</span></blockquote>').join('')+
        '<p>候选词在确认高响应片段命中 1/4，在留出高响应片段命中 0/4。</p>'+source('unit');
    }
    dialog.innerHTML='<div class="dialog-body"><header class="dialog-head"><h2 id="case-title">'+title+
      '</h2><button class="close" data-act="close" aria-label="关闭原始证据">×</button></header>'+readout+
      '</div><footer class="dialog-foot"><button class="button primary" data-act="close">返回当前视角</button></footer>';
    openDialog(dialog);
  }
  function reviewForm(phase) {
    const dialog = $('review-dialog'), record = records.find(r=>r.phase===phase);
    const fixture = phase===3004 || !record;
    const labels = fixture ? ['复用原运行','只换人名','带新证据复核'] : ['复用原运行','改动材料或操作'];
    const choices = fixture ? ['reuse','names','delta'] : ['reuse','delta'];
    if (!choices.includes(state.draft)) state.draft=choices[0];
    dialog.dataset.phase=fixture?'3004':String(phase);
    dialog.innerHTML='<div class="dialog-body"><header class="dialog-head"><div><p class="eyebrow">方案核对 · 评审示例</p>'+
      '<h2 id="review-title">开始之前，先读旧结果</h2></div><button class="close" data-act="close" aria-label="关闭方案核对">×</button></header>'+
      '<p><b>匹配记录　Phase '+(fixture?'3004 · 旧行号与当前姓名冲突':record.phase+' · '+record.title)+'</b></p>'+
      '<div class="review-choices">'+choices.map((key,i)=>'<label><input type="radio" name="draft-kind" value="'+key+'" '+
      (state.draft===key?'checked':'')+'> '+labels[i]+'</label>').join('')+'</div>'+
      '<div id="review-compare"></div></div><footer class="dialog-foot"><button class="button primary" data-act="open-matching-record">打开相匹配的旧记录</button>'+
      '<button class="button" data-act="close">返回观察</button></footer>';
    renderReviewComparison();
    openDialog(dialog);
  }
  function renderReviewComparison() {
    const phase=Number($('review-dialog').dataset.phase), record=records.find(r=>r.phase===phase);
    const old=phase===3004, key=state.draft;
    const change = key==='reuse'?'引用原运行与原评分口径':key==='names'?'更换人物名，其他条件沿用':'待填写真实变化及来源';
    const verdict = key==='reuse'?'可先复用归档':key==='names'?'同一命题 · 沿用结案':'待核对新增证据';
    const note = key==='reuse'?'查看原题、模型实答和失败条件。在同一原场上先完成可复用的分析。':
      key==='names'?'旧规则在冲突表中只有 27/256。更换人名未改变结案所依据的判据。':
      old?'说明哪项证据改变了原行为门的判定。需按原结案条件核对来源。':'说明改变了哪个因素、为何现有运行无法回答，并附证据来源。';
    $('review-compare').innerHTML='<table><thead><tr><th>核对项</th><th>已有实验</th><th>拟议复测</th></tr></thead><tbody>'+
      (old?'<tr><th>机制问题</th><td>旧行与现行冲突时，能否按旧行取值？</td><td>同一主张</td></tr>'+
        '<tr><th>模型与操作</th><td>Qwen3-14B · 旧行／现名对照</td><td>沿用</td></tr>':
        '<tr><th>机制问题</th><td>'+record.goal+'</td><td>按当前选项核对</td></tr>')+
      '<tr><th>材料与判据</th><td>'+ (old?'128 个新家族 · 逐格 85% 行为门':record.design)+'</td><td>'+change+'</td></tr>'+
      '<tr><th>已有结果</th><td>'+record.summary+'</td><td>待审查对应证据</td></tr></tbody></table>'+
      '<div class="review-result"><strong>'+verdict+'</strong>'+note+'</div>'+
      (key==='delta'?'<p><label>改变原判断的证据 / 本次新增验证<textarea id="novelty" rows="3" placeholder="已有运行、原题或结果链接，以及它改变的判据">'+esc(state.why)+'</textarea></label></p>':'')+
      '<details class="ai-context"><summary>查看供 AI 参考的历史证据</summary><dl>'+
      '<div><dt>机制问题</dt><dd>'+record.goal+'</dd></div>'+
      '<div><dt>已有结论及范围</dt><dd>'+record.summary+' '+record.limit+'</dd></div>'+
      '<div><dt>拟议改动</dt><dd>'+change+'</dd></div>'+
      '<div><dt>出处</dt><dd class="source">Phase '+record.phase+' · '+record.source+'</dd></div></dl></details>';
  }
  function toggleFull(force=!state.full) {
    state.full=force;
    $('workspace').classList.toggle('full-screen',force);
    $('fullscreen').setAttribute('aria-pressed',force);
    $('fullscreen').textContent=force?'退出全屏（Esc）':'⛶ 全屏';
    document.body.style.overflow=force?'hidden':'';
    requestAnimationFrame(renderScene);
  }
  $('modes').addEventListener('click',e=>{
    const btn=e.target.closest('[data-mode]');
    if (!btn) return;
    state.mode=btn.dataset.mode;
    state.tab=modeTab[state.mode];
    render();
  });
  $('console-tabs').addEventListener('click', e=>{
    const btn=e.target.closest('[data-tab]');
    if(!btn) return;
    state.tab=btn.dataset.tab;
    render();
  });
  $('object').addEventListener('change',e=>chooseObject({...addresses[e.target.value]},state.tab==='history'));
  $('fullscreen').addEventListener('click',()=>toggleFull());
  $('open-review').addEventListener('click',()=>{state.draft='names';reviewForm(3004);});
  $('search').addEventListener('input',listRecords);
  $('filter').addEventListener('change',listRecords);
  $('linked-only').addEventListener('change',listRecords);
  $('tool-panel').addEventListener('input',e=>{if(e.target.id==='live-input') state.liveInput=e.target.value;});
  $('review-dialog').addEventListener('change',e=>{
    if(e.target.name==='draft-kind'){state.draft=e.target.value;renderReviewComparison();}
  });
  $('review-dialog').addEventListener('input',e=>{if(e.target.id==='novelty')state.why=e.target.value;});
  document.addEventListener('keydown',e=>{
    if(e.key==='Escape' && state.full && !$('case-dialog').open && !$('review-dialog').open) toggleFull(false);
  });
  document.addEventListener('click',e=>{
    const btn=e.target.closest('button');
    if(!btn)return;
    const {act,value,camera}=btn.dataset;
    if(camera==='reset')scene.reset();
    if(camera==='in'||camera==='out')scene.zoom(camera==='in'?1.12:1/1.12);
    if(act==='case'){state.caseIndex=Number(value);render();$('inspector').querySelector('[data-act="case"][data-value="'+value+'"]')?.focus({preventScroll:true});}
    if(act==='pole'){state.pole=Number(value);render();$('inspector').querySelector('[data-act="pole"][data-value="'+value+'"]')?.focus({preventScroll:true});}
    if(act==='plugin'){state.plugin=value;renderTool();$('tool-panel').querySelector('[data-value="'+value+'"]')?.focus({preventScroll:true});}
    if(act==='locate')chooseObject({...addresses[value]});
    if(act==='history'){state.tab='history';render();$('search').focus();}
    if(act==='choose-record'){state.phase=Number(value);listRecords();$('history-list').querySelector('[aria-pressed="true"]')?.focus({preventScroll:true});}
    if(act==='reset-search'){$('search').value='';$('filter').value='all';$('linked-only').checked=false;listRecords();$('search').focus();}
    if(act==='review'){state.draft=Number(value)===3004?'names':'reuse';reviewForm(Number(value));}
    if(act==='open-case')openCase(value);
    if(act==='close')btn.closest('dialog').close();
    if(act==='open-matching-record'){
      const dialog=$('review-dialog');
      state.phase=Number(dialog.dataset.phase);
      state.mode='history';state.tab='history';
      $('linked-only').checked=recordRelation(selectedRecord());
      $('filter').value='all';$('search').value=String(state.phase);
      dialog.close();render();$('search').focus();
    }
  });
  render();
})();

