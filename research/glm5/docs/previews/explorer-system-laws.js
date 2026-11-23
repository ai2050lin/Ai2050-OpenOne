/* Local index of provisional claims. Record IDs refer to the original archive excerpts. */
(() => {
  'use strict';
  const objects = {
    h36:{model:'Qwen3-14B',kind:'Head',block:26,index:36},
    h21:{model:'Qwen3-14B',kind:'Head',block:28,index:21},
    u9517:{model:'Qwen3-4B',kind:'MLP',block:30,index:9517},
    u80:{model:'Qwen3-4B',kind:'MLP',block:30,index:80}
  };
  const entries = {
    lookup:{
      domain:'关系检索', state:'有证据的候选规律', title:'查询线索会改变注意力头的选行规律',
      claim:'在五行英文档案中，B26/H36 更容易按姓名或印刷标签看中目标记录。改问物理首行时，即使这个头看错行，模型也可能答对。',
      scope:'Qwen3-14B，在最后一个输入 token 累计每行的注意力。姓名、印刷标签、物理首行分别指向不同记录。',
      evidence:'两次运行均有注意力观察。B28/H21 有首行任务的匹配阻断对照。两个头之间的信息传递仍待验证。',
      cases:[
        ['相符','p2983-confirm-000-00','按名字问 Tebaka，第 2 行的注意力最高，模型答出 wood。'],
        ['反例','p2983-confirm-000-02','按物理首行问，B26/H36 看向第 4 行，模型仍答对 silver。'],
        ['关联干预','p2997-orion-head-block','另一注意力头 B28/H21 被阻断后，这道同类首行题仍答对 fire。']
      ],
      across:'Phase 2983 与 2990 中，按姓名的命中率分别为 99.0%、99.2%，按标签均为 99.8%，按物理首行为 33.4%、40.2%。两次运行使用同系列条件，跨语料推广仍待检验。',
      crossId:'p2990-confirm-000-02', objects:['h36','h21'],
      sources:[['原始两运行的五选一汇总','frontend/src/components/v8/threeCueEvidence.json']],
      next:'用独立改写的五行材料复核提示条件与落点的关系',
      prediction:'预测在未见材料中，按姓名和按标签的命中率各自高于按首行。若任一命中率差的区间整体小于等于零，该项推广失败。区间跨零时留作未定。',
      needs:['独立的家族与改写，冻结目标行','模型快照和五行跨度','逐题分别保存头落点与答案','事先冻结按家族配对的区间算法和置信水平'],
      model:'Qwen3-14B', object:'B26/H36', recordId:'p2983-confirm-000-02'
    },
    knowledge:{
      domain:'知识与语义', state:'有证据的候选规律', title:'同层两个神经元对地区词上下文的响应不同',
      claim:'在三组中文句对里，含香港或两岸相关词的句子都使 U9517 的门控响应比对照句高至少 4.56 个标准差。U80 对三组的差值均落在 ±0.32 左右。',
      scope:'Qwen3-4B 第 30 块。逐句取单元最大门控投影，在各自的语料分布上标准化，再求 A 减 B。',
      evidence:'同语料、同句对的跨单元比较。跨材料的语义选择性和每个单元对答案的独立因果效应仍待检验。',
      cases:[
        ['相符','claude261-u9517-s9-1','比较香港与巴黎的城市往来，U9517 的响应差为 +5.219。'],
        ['同题对照','claude261-u80-s9-1','同一句对在 U80 上的差为 +0.240。'],
        ['反向观测','claude261-u80-s9-2','两岸关系与健康饮食这组对照，在 U80 上的差为 −0.320。']
      ],
      across:'两个单元使用同一份语料。U9517 的高响应片段也有地区词转换标记，S8 读写对齐差为 0.333，超过 0.20 的门槛。词面和文本格式仍可解释部分现象。',
      objects:['u9517','u80'],
      sources:[['U9517 原始分析','research/neurons/goodfire/GF-L30-U9517-local.json'],['U80 原始分析','research/neurons/goodfire/GF-L30-U80-local.json']],
      next:'检验 U9517 在词形与格式配平后能否区分地区话题',
      prediction:'预测在配平后的新材料中，U9517 的成对差仍大于 U80。若两单元差异的区间整体小于等于零，话题选择性的这一预测失败。区间跨零时留作未定。',
      needs:['独立中文语料','配平地名词形、长度和格式标记','保留 U80 对照与各自的语料标准分','事先冻结按家族配对的区间算法和置信水平'],
      model:'Qwen3-4B',object:'B30/U9517、B30/U80',recordId:'claude261-u9517-s9-1'
    },
    grammar:{
      domain:'语法',state:'待设计',title:'主谓数一致能跨名词与句式复用吗？',
      claim:'只改变主语的单复数，在动词出现前读出状态，看模型是否仍能为未见名词与句式选对动词形式。',
      scope:'拟从英文主谓一致起步。原生对象和读出位置须在发现材料中确定，冻结后再检验。',
      evidence:'此索引尚未关联符合该命题的正式记录。',
      next:'检验数一致规则能否跨名词与句式推广',needs:['先查重并冻结主语数与干扰名词数的最小对','分离发现家族、未见名词和未见句式','先验收行为，再冻结候选对象与匹配对照'],
      model:'Qwen3-14B',object:'全模型行为，原生对象待定位'
    }
  };
  let domain = 'all';
  let expanded = 'lookup';

  function render(host,api,close) {
    const {esc} = api;
    const cards = Object.entries(entries).filter(([id])=>domain==='all'||domain===id);
    function entryHtml(id,entry) {
      const cases = (entry.cases || []).map(([label,record,context]) => '<li><div><small>'+esc(label)+'</small><span>'+esc(context)+
        '</span></div><button type="button" class="law-link" data-law-record="'+esc(record)+'">查看测试 ↗</button></li>').join('');
      const casesSection = cases ? '<h4>实例与反例</h4><ul class="law-cases">'+cases+'</ul>' : '';
      const related = (entry.objects || []).map(key=>{
        const o = objects[key];
        return '<button type="button" class="sys-button" data-law-object="'+key+'">'+esc(o.model)+' · '+api.coord(o)+'</button>';
      }).join('');
      const sources = (entry.sources || []).map(([name,path])=>'<a href="../../../../'+esc(path)+'" target="_blank" rel="noopener">'+esc(name)+' ↗</a>').join(' · ');
      return '<details class="sys-topic law-card" data-law-id="'+id+'" '+(expanded===id?'open':'')+'><summary><span>'+esc(entry.title)+'</span><small>'+esc(entry.domain)+' · '+esc(entry.state)+'</small></summary>'+
        '<p class="law-claim">'+esc(entry.claim)+'</p><h4>适用范围</h4><p>'+esc(entry.scope)+'</p>'+casesSection+
        (entry.across?'<h4>比较与限制</h4><p>'+esc(entry.across)+'</p>':'')+
        (entry.crossId?'<button type="button" class="law-link" data-law-record="'+esc(entry.crossId)+'">查看另一运行的首行题 ↗</button>':'')+
        '<h4>现有证据</h4><p>'+esc(entry.evidence)+'</p>'+related+
        (sources?'<div class="law-sources">'+sources+'</div>':'')+
        '<div class="law-next"><h4>下一项检验</h4><p>'+esc(entry.next)+'</p>'+(entry.prediction?'<p>'+esc(entry.prediction)+'</p>':'')+
        '<button type="button" class="sys-button" data-law-stage="'+id+'">准备测试草稿</button></div></details>';
    }
    host.innerHTML = '<section id="law-index"><label class="law-filter" for="law-domain">研究方向<select id="law-domain"><option value="all">全部方向</option>'+
      Object.entries(entries).map(([key,entry])=>'<option value="'+key+'">'+esc(entry.domain)+'</option>').join('')+'</select></label>'+
      cards.map(([key,entry])=>entryHtml(key,entry)).join('')+'</section>';
    host.querySelector('#law-domain').value=domain;
    host.querySelector('#law-domain').onchange=event=>{
      domain=event.target.value;
      expanded=domain==='all'?'lookup':domain;
      render(host,api,close);
      host.querySelector('#law-domain').focus({preventScroll:true});
    };
    host.onclick = event => {
      const summary = event.target.closest('summary');
      if (summary) {
        const card=summary.closest('[data-law-id]');
        if (card) expanded=card.open?'':card.dataset.lawId;
      }
      const button=event.target.closest('button');
      if(!button||!host.contains(button))return;
      if (button.dataset.lawRecord) {
        close();
        api.openRecord(button.dataset.lawRecord);
      } else if (button.dataset.lawObject) {
        close();
        api.selectObject(objects[button.dataset.lawObject]);
        api.openFeatures();
      } else if (button.dataset.lawStage) {
        const draft=entries[button.dataset.lawStage];
        api.stage({title:draft.next,needs:draft.prediction?[...draft.needs,draft.prediction]:draft.needs,topic:draft.domain,model:draft.model,object:draft.object,recordId:draft.recordId});
      }
    };
  }
  window.ExplorerLaws = {render};
})();
