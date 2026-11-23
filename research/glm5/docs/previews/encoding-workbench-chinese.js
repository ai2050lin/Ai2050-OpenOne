// S9 archives on Qwen3-4B. Readouts are per-sentence peak gate z-scores.
(() => {
  'use strict';
  const options = ['香港与巴黎的往来','两岸关系与健康饮食','过关与出行'];
  const units = [80,9517];
  const snapshot = '6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad';
  const esc = x => String(x??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const sourceHref = r => 'explorer-system-design.html#record='+encodeURIComponent(r.id);
  const fileHref = r => '../../../../'+r.source;
  const placeholder = (title,body) => '<div class="et-notice"><strong>'+esc(title)+'</strong><p>'+esc(body)+'</p></div>';
  const title = (h,p) => '<header class="et-heading"><h3>'+esc(h)+'</h3><p>'+esc(p)+'</p></header>';
  const method = '先取该单元在一句话各 token 上的门控预激活峰值，再用该单元在参照语料中的均值和标准差换算。';

  function record(i,unit) {
    if (!Number.isInteger(i) || !options[i] || !units.includes(unit)) return null;
    return window.EncodingEvidence.chinesePairs.find(r=>r.id==='claude261-u'+unit+'-s9-'+(i+1) &&
      r.snapshot===snapshot && r.model==='Qwen3-4B' && r.objects.some(o=>o.kind==='MLP'&&o.block===30&&o.index===unit)) || null;
  }
  function prompt(i) {return record(i,9517)?.question || '';}
  function getSites(c) {
    return c.b===0?80:9517;
  }
  function metrics(r) {
    return r && ['A · 标准分','B · 标准分','A − B'].map(label=>r.metrics.find(m=>m.label===label)?.value);
  }
  function pair(c) {
    const primary = c.matched && c.block===30 && !c.liveEmpty ? record(c.a,c.index) : null;
    const ref = getSites(c);
    const secondary = primary && ref!==c.index ? record(c.a,ref):null;
    return {primary,secondary,ref};
  }
  function unavailable(c) {
    return c.liveEmpty ? placeholder('当前输入等待采集','这段新文本没有已匹配的 S9 句对读数。切回归档句对可继续复核。') :
      placeholder('当前对象缺少句对摘录','可在 Qwen3-4B 第 30 块定位 U9517，查看此处收录的逐句门控峰值。');
  }
  function original(i) {
    const text = prompt(i);
    if (!text) return '';
    return '<div class="et-pair">'+text.split('\n').map((line,j)=>'<article><b>句 '+(j?'B':'A')+'</b><p lang="zh-CN">'+esc(line.replace(/^[AB]\s*/u,''))+'</p></article>').join('')+'</div>';
  }
  function archiveLink(r) {
    return '<a target="_blank" rel="noopener" href="'+esc(sourceHref(r))+'">历史句对 · U'+r.objects[0].index+' ↗</a>';
  }
  function rawLink(r) {
    return '<a target="_blank" rel="noopener" href="'+esc(fileHref(r))+'">单元原档案 · U'+r.objects[0].index+' ↗</a>';
  }
  function source(r) {
    return '<details class="et-source"><summary>来源与复核</summary><dl class="et-meta">'+
      '<dt>模型</dt><dd>'+esc(r.model)+'</dd><dt>快照</dt><dd>'+esc(r.snapshot)+'</dd>'+
      '<dt>归档索引</dt><dd>'+esc(r.sourceKey)+'</dd><dt>文件</dt><dd>'+rawLink(r)+'</dd>'+
      '<dt>文件 SHA-256</dt><dd>'+esc(r.artifactSha)+'</dd></dl></details>';
  }
  function refNote(c,p) {
    if(p.ref===c.index)return '参照与当前单元相同。可切换参照，比较另一个单元的响应。';
    if(!p.secondary)return '随稿未索引此句对中 U'+p.ref+' 的摘录，可在其完整 S9 档案中复核。';
    return '两个单元分别用各自的语料均值和标准差定标。';
  }
  function rows(c,p) {
    return [{unit:c.index,label:'当前',r:p.primary},...(c.index!==p.ref?[{unit:p.ref,label:'参照',r:p.secondary}]:[])];
  }
  function readingTable(c,p) {
    return '<table><thead><tr><th>单元</th><th>句 A</th><th>句 B</th><th>A − B</th></tr></thead><tbody>'+
      rows(c,p).map(({unit,label,r})=>{
        const v=metrics(r);
        return '<tr><th scope="row">'+label+' U'+unit+'</th>'+(v?v.map(x=>'<td>'+esc(x)+'</td>').join(''):'<td colspan="3">此摘录未收录</td>')+'</tr>';
      }).join('')+'</tbody></table>';
  }
  function locateButtons(handler,c) {
    handler.querySelectorAll('[data-ch-unit]').forEach(b=>b.onclick=()=>c.action('index',Number(b.dataset.chUnit)));
    handler.querySelectorAll('[data-ch-material]').forEach(b=>b.onclick=()=>c.action('material'));
  }
  function renderFigure(host,c) {
    const p=pair(c);
    if(!p.primary){host.innerHTML='<div class="et-root">'+unavailable(c)+'</div>';return;}
    const delta = c.angle==='delta';
    const series=rows(c,p).map(x=>({...x,v:metrics(x.r)}));
    const finite=series.filter(x=>x.v).flatMap(x=>(delta?[x.v[2]]:x.v.slice(0,2)).map(Number));
    const span=Math.max(1,...finite.map(Math.abs));
    const bar = (s,k) => {
      const x=Number(s.v[k]), color=k===1?'var(--b)':'var(--a)';
      const track=delta ? 'background:linear-gradient(90deg,#eef2f6 49.8%,#8b9fab 49.9%,#8b9fab 50.2%,#eef2f6 50.3%)' : '';
      const offset=delta?(x<0?50-50*Math.abs(x)/span:50):0;
      return '<div style="display:grid;grid-template-columns:48px minmax(20px,1fr) 55px;gap:9px;align-items:center;font-size:12px;padding:3px 0"><span>'+
        ['句 A','句 B','A − B'][k]+'</span><span class="bar-track" aria-hidden="true" style="position:relative;height:8px;'+track+'"><i style="position:absolute;left:'+offset+
        '%;width:'+((delta?50:100)*Math.abs(x)/span)+'%;height:100%;background:'+(delta&&x<0?'#a35c36':color)+'"></i></span><strong style="text-align:right;font-variant-numeric:tabular-nums">'+esc(s.v[k])+'</strong></div>';
    };
    host.innerHTML='<div class="et-root"><p class="data-caption">'+esc(options[c.a])+' · '+(delta?'A − B，居中零线，左右同尺度':'句子各自的峰值标准分，两图共用尺度')+'</p>'+
      series.map(s=>'<section style="margin:10px 0 8px"><button type="button" class="text-btn" data-ch-unit="'+s.unit+'">'+s.label+'单元 · B30/U'+s.unit+'</button>'+
        (s.v?(delta?bar(s,2):bar(s,0)+bar(s,1)):'<p class="et-caption">此摘录未收录</p>')+'</section>').join('')+
      '<p class="et-caption">'+esc(refNote(c,p))+'</p></div>';
    locateButtons(host,c);
  }
  function renderPanel(host,c) {
    const p=pair(c);
    if(!p.primary){host.innerHTML='<div class="et-root">'+unavailable(c)+'</div>';return;}
    const r=p.primary, v=metrics(r);
    let body;
    if(c.tab==='object'){
      body=title('第 30 块 · FFN 单元 U'+c.index,method)+
        '<p>'+esc(r.model)+' 第 30 块有 '+r.dims.width.toLocaleString('en-US')+' 个 FFN 中间单元。此题的归档读数定位到 U'+c.index+' 的 gate 预激活。</p>'+source(r);
    }else if(c.tab==='contrast'){
      body=title('同一对原句，比较两个单元','A 与 B 是分别输入模型的两句话。归档在四舍五入前计算 A − B。')+
        original(c.a)+readingTable(c,p)+'<p class="et-caption">'+esc(refNote(c,p))+'</p>';
    }else if(c.tab==='tools'){
      body=title('MLP · 逐句峰值标准分',method)+readingTable(c,p)+
        '<p class="et-caption">z = (句内最大 gate − 语料均值) / 语料标准差。归档摘录保留逐句摘要；查看峰值发生在哪个 token 或绘制整层热力图，还需对应的原生向量。</p>'+source(r);
    }else {
      const refDoc = record(0,p.ref);
      body=title('这组读数支持什么','问题改变了地名、语义或措辞，可回到原句检查究竟比较了哪些变化。')+
        '<p>在「'+esc(options[c.a])+'」中，U'+c.index+' 的 A − B 为 <b>'+esc(v[2])+'</b> 个标准差。'+(p.secondary?'同一句对在 U'+p.ref+' 上为 <b>'+esc(metrics(p.secondary)[2])+'</b>。':'')+'</p>'+
        '<p class="et-caption">句对很少，峰值的位置也可能不同。要判断该单元是否编码特定语义、是否影响输出，还需控制词长与实体、使用未见句式，并检验对输出的干预效应。</p>'+
        '<div class="et-controls"><button type="button" data-ch-material>查看原句与来源</button> '+archiveLink(r)+(p.secondary?' · '+archiveLink(p.secondary):refDoc&&p.ref!==c.index?' · '+rawLink(refDoc):'')+'</div>'+source(r);
    }
    host.innerHTML='<div class="et-root">'+body+'</div>';
    locateButtons(host,c);
  }
  function renderMaterial(host,c) {
    const anchor=record(c.a,9517);
    if(c.liveEmpty||!anchor){host.innerHTML='<div class="et-root">'+unavailable(c)+'</div>';return;}
    const matched=units.map(u=>record(c.a,u)).filter(Boolean);
    host.innerHTML='<div class="et-root">'+title(options[c.a]+' · 中文原句','Qwen3-4B · S9 句对归档')+original(c.a)+
      '<p class="et-caption">'+esc(method)+'</p>'+matched.map(r=>'<p>'+archiveLink(r)+'</p>'+source(r)).join('')+'</div>';
  }
  window.EncodingChinese=Object.freeze({options,record,prompt,renderFigure,renderPanel,renderMaterial});
})();
