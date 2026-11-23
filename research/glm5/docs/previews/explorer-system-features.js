/* Offline dossiers. Each is scoped to an archived model snapshot and native object. */
(() => {
  'use strict';
  const q14 = '1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce';
  const q4 = '6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad';
  const objectName = o => 'B' + o.block + '/' + (o.kind === 'Head' ? 'H' : 'U') + o.index;
  const examples = {
    h36: {model:'Qwen3-14B',kind:'Head',block:26,index:36},
    u9517: {model:'Qwen3-4B',kind:'MLP',block:30,index:9517},
    h21: {model:'Qwen3-14B',kind:'Head',block:28,index:21},
    u80: {model:'Qwen3-4B',kind:'MLP',block:30,index:80},
    u2429: {model:'Qwen3-14B',kind:'MLP',block:36,index:2429},
    u17216: {model:'Qwen3-14B',kind:'MLP',block:28,index:17216}
  };
  const headRun = 'frontend/src/components/v8/threeCueEvidence.json';
  const dossiers = {
    'Qwen3-14B B26/H36': {
      snapshot:q14, label:'注意力头 · 条件选行', level:'注意力观察',
      claim:'按姓名或印刷标签提问时，这个头通常看向目标记录。',
      limit:'改问物理首行时，头看向目标的比例降至 33.4%。在 Orion 这道题上，模型答对了 silver，头看得最多的却是第 4 行。',
      chartTitle:'注意力最大的一行与目标行相符的比例', chartUnit:'Phase 2983 · 五条候选记录',
      bars:[['按姓名','99.0%',99.0],['按印刷标签','99.8%',99.8],['按物理首行','33.4%',33.4]],
      method:'在最后输入 token，分别累计落在各条记录上的注意力，然后取五条中的最大者。图中是各类提示的命中比例。',
      reading:'Q 和 K 对选行的预测均未达到原实验的准入标准。该结果限定了仅凭这个头的注意力落点解释最终答案的范围。',
      source:headRun, key:'runs[0].natural / runs[0].predictor',
      coverage:['保留五行注意力分布','Q/K 的选行预测未过门','本运行未进入因果检验'],
      ids:['p2983-confirm-000-00','p2983-confirm-000-02'],
      task:'复核五行候选的注意力命中率', needs:['同一模型快照','原冻结题集和五行跨度','最后输入位置的注意力']
    },
    'Qwen3-4B B30/U9517': {
      snapshot:q4, label:'原生 FFN 神经元 · 门控投影', level:'语料与句对观察',
      claim:'含香港或两岸相关词的三个句子，都比各自的对照句触发了更高的门控响应。',
      limit:'高响应文本也包括维基百科的地区词转换标记。读方向与写方向的对齐分差为 0.333，超过原验收上限 0.20，这个主题解释仍是候选。',
      chartTitle:'同一神经元在成对句子中的响应差', chartUnit:'A − B · 语料标准差',
      pairs:['claude261-u9517-s9-1','claude261-u9517-s9-2','claude261-u9517-s9-3'],
      pairLabels:['香港往来 / 巴黎往来','两岸关系 / 健康饮食','深圳过关 / 车站出发'],
      method:'每句取各 token 的最大门控投影（gate_proj，SiLU 之前），按同份语料的均值和标准差换成 z 分数，再算成对差值。语料覆盖 3,904 个窗口、983,030 个有效位置。',
      reading:'本项目的 Goodfire 风格单元分析得到的正端词包括「香港」「港澳」「台湾」，劈半余弦为 0.915。S8 读对齐为 0.406，写对齐为 0.073。这里测量模型中的具体 FFN 单元。SAE 特征另需记录字典版本和特征编号。',
      source:'research/neurons/goodfire/GF-L30-U9517-local.json',key:'S1 / checks / S6 / S8 / S9',
      coverage:['门控投影与三组句对','独立话题预测待补','已有层输入方向干预，单元独立效应待测'],
      ids:['claude261-u9517-s9-1','claude261-u9517-s9-2'],
      task:'区分地名、话题与地区词转换标记',needs:['独立来源与留出语料','匹配地名频率和文本格式','同一门控位置与读写检验']
    },
    'Qwen3-4B B30/U80': {
      snapshot:q4,label:'原生 FFN 神经元 · 门控投影',level:'语料与句对观察',
      claim:'这个单元对香港和两岸三组句对的响应差接近零，其中一组方向相反。',
      limit:'三组中仅两组符号符合预期。原报告 S9 未通过；这组材料不足以给该单元命名。',
      chartTitle:'同一神经元在成对句子中的响应差', chartUnit:'A − B · 语料标准差',
      pairs:['claude261-u80-s9-1','claude261-u80-s9-2'],
      pairLabels:['香港往来 / 巴黎往来','两岸关系 / 健康饮食'],
      extraBars:[['深圳过关 / 车站出发','+0.220']],
      method:'每句取最大门控投影的语料标准分，再计算两个句子的差。原报告 S9.pairs 收录全部三组；本页历史题库节选前两组。',
      reading:'本项目的 Goodfire 风格单元分析中，S8 读写对齐分差为 0.138，通过该项阈值。S9 三组平均差为 +0.047，未通过符号一致性门槛。',
      source:'research/neurons/goodfire/GF-L30-U80-local.json',key:'checks / S6 / S8 / S9',
      coverage:['门控投影与三组句对','句对一致性未过门','已有层输入方向干预，单元独立效应待测'],
      ids:['claude261-u80-s9-1','claude261-u80-s9-2'],task:'从高响应上下文形成新的可检验标签',
      needs:['原生门控读数','正负端自然文本','按来源独立的验证句对']
    },
    'Qwen3-14B B36/U2429': {
      snapshot:q14,label:'原生 FFN 神经元 · 位置轨迹',level:'单题激活观察',
      claim:'在这道数步读代码题中，同一单元的四处读数既有正值，也有负值。',
      limit:'四个位置都来自一道题。跨题的功能、稳定性和对答案的影响仍缺少独立证据。',
      chartTitle:'四个归档位置的原生激活',chartUnit:'原生读数',metricId:'p2933-counted0-u2429',
      method:'按归档读出边界查看同一单元的原始值，四个位置分别为输入末尾、最后一步前、Value 前和 Code 前。',
      reading:'跨词汇的分布统计、匹配对照以及单元对模型输出的效应尚待补充。',
      source:'frontend/src/components/v8/data/graphObservationExample.json',key:'native/confirm-000/native.npz',
      coverage:['保留四个读出位置','跨独立题目的预测待补','单元干预待补'],
      ids:['p2933-counted0-u2429'],task:'比较不同题目中四个读出位置的单元响应',needs:['冻结读出边界','独立材料与非计数对照','同一单元的原生激活']
    },
    'Qwen3-14B B28/H21': {
      snapshot:q14,label:'注意力头 · 原模型阻断',level:'匹配对照干预 · 待审',
      claim:'在 192 个家族的物理首行题中，阻断该头相对随机头使正确答案的平均优势再降 0.330。',
      limit:'在一条 Orion 单题中，阻断后模型仍答 fire，正确项优势由 10.25 升至 10.50。总体均值不能替代单题预测。',
      chartTitle:'Orion 单题：正确项相对竞争项的优势',chartUnit:'单题匹配对照',metricId:'p2997-orion-head-block',
      method:'比较基线、定向阻断和同题随机头。将单题优势与运行汇总分开读取。',
      reading:'模型干预测得范围内的平均效应。中心回执可核验，证据评审状态为待审。',
      source:'tests/glm5/result/routine/run-t7dfn8_7/review.json',key:'conclusion.causal',
      coverage:['此处收录模型答案优势','跨题型预测待补','定向阻断与同题随机头，待审'],
      ids:['p2997-orion-head-block','p2997-first-line-cohort'],task:'独立复核定向头与随机头的优势差',
      needs:['冻结模型及材料','同题随机头与原样对照','按家族汇总的置信区间']
    },
    'Qwen3-14B B28/U17216': {
      snapshot:q14,label:'原生 FFN 神经元 · 单元组成员',level:'已关联组干预',
      claim:'该单元属于 128 个冻结单元组成的恢复组，组恢复的平均优势增益为 −0.151。',
      limit:'现有干预同时作用于整组。单独改变 U17216 会怎样，记录中没有答案。',
      chartTitle:'同一组的归档指标',chartUnit:'192 个家族 · 组干预',metricId:'p2997-first-line-cohort',
      method:'以共同快照、运行和冻结单元列表关联单元组成员。组干预结果保留原有测量层级。',
      reading:'需要获得单元级选择性、单独干预和匹配单元对照，才能评估这个成员的独立作用。',
      source:'tests/glm5/result/routine/run-t7dfn8_7/review.json',key:'conclusion.causal',
      coverage:['保存冻结组成员身份','单元级预测待补','整组恢复已测，单元独立效应待补'],
      ids:['p2997-first-line-cohort'],task:'收集 U17216 的单元级观察与匹配对照',
      needs:['单元级测量','匹配同层单元','对组成员与组整体分别记账']
    }
  };
  const tabs = {overview:'证据与反例',sources:'测量与来源',gaps:'待验证问题'};
  let activeTab = 'overview';
  let previousObject = '';

  function render(host, api) {
    const esc = api.esc;
    const object = objectName(api.state);
    const name = api.state.model + ' ' + object;
    if (name !== previousObject) activeTab = 'overview';
    previousObject = name;
    const known = dossiers[name];
    const profile = (!api.state.snapshot || api.state.snapshot === known?.snapshot) ? known : undefined;
    const records = api.records.filter(r => r.model === api.state.model &&
      (!api.state.snapshot || r.snapshot === api.state.snapshot) &&
      (!profile || r.snapshot === profile.snapshot) && r.objects.some(o => o.kind === api.state.kind &&
        o.index === api.state.index && o.block === api.state.block));
    const byId = id => records.find(r => r.id === id);
    const exampleLinks = Object.entries(examples).map(([key,o]) =>
      '<button type="button" data-feature-example="'+key+'" '+(o.model+' '+objectName(o)===name?'aria-current="true"':'')+'>'+
      esc(o.model.replace('Qwen3-',''))+' · '+objectName(o)+'</button>').join('');
    const related = rows => rows.map(r =>
      '<button type="button" class="fd-record" data-feature-record="'+esc(r.id)+'"><span><strong>'+esc(r.title)+'</strong><small>Phase '+esc(r.phase)+' · '+esc(r.scope)+'</small></span><span aria-hidden="true">↗</span></button>').join('');
    let chart = '';
    if (profile) {
      const bars = profile.pairs ? profile.pairs.flatMap((id,i) => {
        const val = byId(id)?.metrics.find(m=>m.label === 'A − B')?.value;
        return val === undefined ? [] : [[profile.pairLabels[i],val]];
      }).concat(profile.extraBars || []) : profile.bars;
      if (bars) {
        chart = bars.map(([label,val,percent]) => {
          const width = profile.pairs ? Math.min(50,50*Math.abs(Number(val))/6) : percent;
          return '<li><span>'+esc(label)+'</span><i aria-hidden="true"><b style="width:'+width+'%" class="'+(Number(val)<0?'negative':'')+'"></b></i><strong>'+esc(val)+'</strong></li>';
        }).join('');
        chart = '<ul class="fd-bars'+(profile.pairs?' diverging':'')+'">'+chart+'</ul>';
      } else {
        chart = '<dl class="fd-nums">'+(byId(profile.metricId)?.metrics || []).map(m =>
          '<div><dt>'+esc(m.label)+'</dt><dd>'+esc(m.value)+'</dd></div>').join('')+'</dl>';
      }
    }
    const nextTask = profile?.task || '建立 '+name+' 的首份特征档案';
    let view;
    if (!profile) {
      view = '<div class="fd-empty"><h3>该对象尚无特征档案</h3><p>从自然文本中的响应和匹配对照开始，逐步补齐上下文、特征解释与干预证据。</p><button type="button" class="fd-action" data-feature-stage>准备特征采集</button></div>';
    } else if (activeTab === 'sources') {
      const levels = ['观察','预测','干预'];
      const coverage = '<h4>证据覆盖</h4><dl class="fd-coverage">'+profile.coverage.map((v,i)=>
        '<div><dt>'+levels[i]+'</dt><dd>'+esc(v)+'</dd></div>').join('')+'</dl>';
      view = '<div class="fd-grid"><article><h4>测量口径</h4><p>'+esc(profile.method)+'</p><h4>分析边界</h4><p>'+esc(profile.reading)+'</p>'+
        coverage+'<a class="fd-source" href="../../../../'+esc(profile.source)+'" target="_blank" rel="noopener">查看原始分析文件 ↗</a><small class="fd-code">'+esc(profile.key)+'</small>'+
        '<small class="fd-code">快照 '+esc(profile.snapshot)+'</small></article><article><h4>对应的历史测试 <small>'+records.length+' 条</small></h4>'+
        related(records)+(records.some(r=>r.caseId||r.promptType==='原始最小句对')?'<button type="button" class="fd-action secondary" data-feature-live>在实时模式查看原题 →</button>':'')+'</article></div>';
    } else if (activeTab === 'gaps') {
      view = '<div class="fd-plan"><h4>'+esc(nextTask)+'</h4><p>'+esc(profile.limit)+'</p><ul>'+profile.needs.map(s=>'<li>'+esc(s)+'</li>').join('')+
        '</ul><button type="button" class="fd-action" data-feature-stage>准备补测任务</button></div>';
    } else {
      const cases = profile.ids.map(byId).filter(Boolean);
      const pairExample = profile.pairs && byId(profile.pairs[0]);
      const quote = pairExample ? '<details class="fd-pair"><summary>查看第一组原句及读数</summary><p>'+esc(pairExample.question)+'</p><small>'+
        pairExample.metrics.slice(0,2).map(m=>esc(m.label)+' '+esc(m.value)).join('　')+'</small></details>' : '';
      view = '<div class="fd-grid"><article><h4>'+esc(profile.chartTitle)+'</h4><small class="fd-caption">'+esc(profile.chartUnit)+'</small>'+chart+quote+
        '</article><article><h4>解释的边界</h4><p>'+esc(profile.limit)+'</p><h4>对应测试</h4>'+related(cases)+'</article></div>';
    }
    const tabList = profile ? '<nav class="fd-tabs" aria-label="特征档案栏目">'+Object.entries(tabs).map(([id,title])=>
      '<button type="button" aria-pressed="'+(activeTab===id)+'" data-feature-tab="'+id+'">'+title+'</button>').join('')+'</nav>' : '';
    host.innerHTML = '<section id="feature-dossier" class="feature-dossier"><header class="fd-head"><div><small>'+esc(profile?.label || (api.state.kind==='Head'?'原生注意力头':'原生 FFN 神经元'))+'</small><h3>'+esc(object)+' 的特征档案</h3></div>'+
      (profile?'<span class="fd-level">'+esc(profile.level)+'</span>':'')+'</header>'+
      (profile?'<p class="fd-claim">'+esc(profile.claim)+'</p>':'')+tabList+view+
      '<footer class="fd-examples"><span>已建档对象</span>'+exampleLinks+'<button type="button" data-feature-laws>跨测试规律 →</button></footer></section>';
    host.onclick = event => {
      const button = event.target.closest('button');
      if (!button || !host.contains(button)) return;
      if (button.dataset.featureExample) api.selectObject(examples[button.dataset.featureExample]);
      if (button.dataset.featureRecord) api.openRecord(button.dataset.featureRecord);
      if (button.hasAttribute('data-feature-laws')) api.openLaws();
      if (button.hasAttribute('data-feature-live')) {
        const input = records.find(r => r.caseId || r.promptType === '原始最小句对');
        if (input) api.openLive(input.id);
      }
      if (button.hasAttribute('data-feature-stage')) api.stage({title:nextTask,needs:profile?.needs ||
        ['模型快照与原生编号','自然前向记录与上下文','冻结材料和匹配对照']});
      if (button.dataset.featureTab) {
        activeTab = button.dataset.featureTab;
        render(host,api);
        host.querySelector('[data-feature-tab="'+activeTab+'"]').focus({preventScroll:true});
      }
    };
  }
  window.ExplorerFeatures = {render};
})();
