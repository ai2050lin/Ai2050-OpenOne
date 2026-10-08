/* 脉络透镜 v4：顶层三 tab —— 行业进展 / 平台进度 / 当前机器进度
   v4（M1 接线）：平台进度 tab 接中心节点 GET /api/distributed/summary（deploy/distributed_service.py，deploy/ 不入 git），
   行业进展接 GET /api/news（arXiv 抓取+内置兜底）；失败自动回退 demo 数据（distributedData.js）
   并显示 DEMO 徽标，成功显示 LIVE 徽标。当前机器进度 = 本机 Agent 状态（demo 接入点不变）。
   三带节点点击出详情；同行差异以「对比卡」登记入账本（四级证据体系）。 */
import { useEffect, useState } from 'react';
import { LOCAL_NODE, PLATFORM_STATS, NODES, TEMPLATES, AGG_STEPS, AGG_DIMS, NEWS, MI_ERAS, RSA_ERAS, RSA_NOTE } from './distributedData.js';

/* API 前缀：与研发透镜（LensProcess）同一惯例；禁止在组件里散落硬编码地址 */
const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

/* 当前机器进度 · RDC 主线（demo 接入点 ← atlas_ledger.json / phase_queue.json） */
const MAINLINE=[
  {lb:'P4–P7',tt:'主轴三段\n权重绑定定 is-a',st:'done'},
  {lb:'P8–P11',tt:'写入端分布式\n栈=软门',st:'done'},
  {lb:'P17–21',tt:'w+com_V≈26\n跨精度复现',st:'done'},
  {lb:'P35',tt:'A 闸门 seal\nQ08=甲',st:'done'},
  {lb:'Q03',tt:'E_read 基线\n0.332 · 5% 门 0/3',st:'done'},
  {lb:'Q04',tt:'E_ar 装置\n4/4 PASS',st:'done'},
  {lb:'Q05',tt:'E_ar 测量\n形状 flat/sat',st:'done'},
  {lb:'Q06',tt:'C_steer 基座\n0.0000 · 无定向杠杆',st:'done'},
  {lb:'Q07',tt:'KPI 曲线 v0\nQ03–Q06 汇总',st:'now'},
  {lb:'远期',tt:'权重级证明\nN2h1-α-1',st:'todo'},
];

/* 平台进度 · 平台发布时间线（demo 接入点 ← 发布仓 / 部署记录；客户端与服务的版本时间线） */
const RELEASES=[
  {lb:'v1–v3',tt:'三部分平台改造\n3D 空间 / 研发 / 数据',st:'done'},
  {lb:':8501',tt:'InterPLM 部署\nESM-2 SAE 2548 特征',st:'done'},
  {lb:'v4–v5',tt:'融合驾驶舱 v5\n三透镜对象路由 /rdc-fusion',st:'done'},
  {lb:'3589dbb',tt:'发布仓推送\n46.97 MiB',st:'done'},
  {lb:'v5-home',tt:'v5 设为默认首页\n官网 6 页同步换肤',st:'now'},
  {lb:'next',tt:'真实数据接入\n激活←collect.npz · 队列←phase_queue',st:'todo'},
];

/* 行业进展 · 同行对比卡（demo 接入点 ← industry.json；每卡必带「对比」注脚登记视角差） */
const PEERS_ROW1=[
  {src:'Anthropic',date:'2025.03',h:'Circuit Tracing: Attribution Graphs',p:'层内→跨层电路自动化追踪；与 Q05 消融口径可桥接',
   note:'对比：Attribution Graphs 依赖人工假设遍历；本项目 Q05 用精确可加预算自动化消融 — 视角差登记入账本'},
  {src:'OpenAI',date:'2020 · archived',h:'Microscope：模型级可视化集合',p:'对象导航 + 多视图挂载的原型范式',
   note:'对比：Microscope 已归档，其「对象→多视图」范式即 v5 透镜路由来源'},
  {src:'Anthropic',date:'2022–2024',h:'Toy Models of Superposition / SAE',p:'特征叠加与稀疏字典分解基线',
   note:'对比：Superposition 对应本项目跨族近正交观察（P4–P7）；叠加上限与 G−1=5 维读位槽相关'},
];
const PEERS_ROW2=[
  {src:'开源生态',date:'已部署 :8501',h:'InterPLM — 蛋白质 SAE 对齐',p:'SAE 结论跨域同构的旁证',note:'对比：InterPLM 结论：ESM-2 SAE 叠加结论与 LLM 同构 → 支持本项目跨模型比较策略'},
  {src:'开源生态',date:'2024',h:'SAEBench — SAE 系统评估',p:'装置级评估标准的对标项',note:'SAEBench 提供 SAE 评估标准；Q04/Q05 的装置门与 metric_dict v4 口径可对标'},
];

function PeerCard({c}){
  return (
    <div className="fw-peer-card" onClick={()=>window.alert(c.note)} title={c.note}>
      <div className="src"><span>{c.src}</span><span>{c.date}</span></div>
      <h6>{c.h}</h6>
      <p>{c.p}</p>
    </div>
  );
}

/* 行业进展 · 领域阶段时间轴（可复用）：机制可解释性（MI_ERAS）与表征相似性分析（RSA_ERAS）各一条
   数据 ← eras（阶段数组）；live 可选——传入时最右追加「最新」实时节点（← /api/news）。
   点击节点 → 下方详情面板：阶段目标 / 成果 / 代表论文（可点击跳原文）。 */
function EraTimeline({eras,live}){
  const [sel,setSel]=useState(eras[0].id);
  const isLive = !!live && sel==='live';
  const era = eras.find(e=>e.id===sel) || eras[0];
  const liveNode = live ? {
    years:'NOW', title:'最新节点 · 实时动态',
    goal: live.newsLive
      ? '持续追踪领域最前沿：中心节点每 60 分钟抓取一次 arXiv「mechanistic interpretability」最新论文，前端每 30s 轮询刷新。'
      : '中心节点离线或抓取失败——当前显示内置兜底列表（最后一次成功抓取前登记的静态条目），重启中心节点后自动恢复实时。',
    outcome: live.newsLive
      ? (live.newsSrc==='arxiv' ? '来源：arXiv API · 按提交日期倒序取最新 10 篇' : '来源：中心节点缓存')
      : '来源：内置兜底列表（builtin-fallback）',
    papers: live.newsItems||[],
  } : null;
  const cur = isLive ? liveNode : era;
  return (
    <div className="fw-era">
      <div className="fw-era-tl">
        {eras.map(e=>(
          <button key={e.id} type="button" className={'fw-era-node'+(sel===e.id?' sel':'')} onClick={()=>setSel(e.id)}>
            <span className="fw-era-dot"/>
            <span className="fw-era-years">{e.years}</span>
            <span className="fw-era-name">{e.title}</span>
          </button>
        ))}
        {live && (
          <button type="button" className={'fw-era-node live'+(isLive?' sel':'')} onClick={()=>setSel('live')}
                  title={live.newsLive?'/api/news · '+live.newsSrc:'中心节点离线'}>
            <span className="fw-era-dot"/>
            <span className="fw-era-years">NOW</span>
            <span className="fw-era-name">最新{' · '}{live.newsLive?'LIVE':'兜底'}</span>
          </button>
        )}
      </div>
      <div className="fw-era-detail">
        <div className="fw-era-hd">
          <b>{cur.years} · {cur.title}</b>
          {isLive && <span className={'fw-src-chip '+(live.newsLive?'live':'demo')}>{live.newsLive?'● LIVE '+live.newsSrc:'○ FALLBACK'}</span>}
        </div>
        <div className="fw-era-body">
          <div className="fw-era-block"><h6>阶段目标</h6><p>{cur.goal}</p></div>
          <div className="fw-era-block"><h6>关键成果</h6><p>{cur.outcome}</p></div>
        </div>
        <div className="fw-era-papers">
          <h6>{isLive?'最新论文（实时流）':'代表论文'} <span>{(cur.papers||[]).length} 篇 · 点击跳原文</span></h6>
          {(cur.papers||[]).map(pw=>(
            <div key={pw.title} className="fw-era-paper" style={pw.url?{cursor:'pointer'}:undefined}
                 onClick={()=>{ if(pw.url) window.open(pw.url,'_blank','noopener'); }}>
              <span className="fw-era-pdate">{pw.date}</span>
              <span className="fw-era-ptitle">{pw.title}</span>
              <span className="fw-era-ptag">{pw.tag}</span>
              <span className="fw-era-psrc">{pw.src}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Lane({color,label,small,children}){
  return (
    <div className="fw-lane">
      <div className="fw-lane-head"><span className="ln-dot" style={{background:color}}/>{label} <small>{small}</small></div>
      {children}
    </div>
  );
}

function Timeline({items,tag}){
  return (
    <div className="fw-tl">
      {items.map(n=>(
        <button key={n.lb} className={'fw-tl-node '+n.st} onClick={()=>window.alert(n.lb+'：'+n.tt.replace('\n',' · ')+'（详情接入 '+tag+' 后开放）')}>
          <div className="fw-tl-dot"/>
          <div className="lb">{n.lb}</div>
          <div className="tt">{n.tt.split('\n').map((t,i)=><span key={i}>{t}<br/></span>)}</div>
        </button>
      ))}
    </div>
  );
}

function fmtAgo(ts){
  if(!ts) return '—';
  const d = Math.max(0, Math.round(Date.now()/1000 - ts));
  return d<60 ? d+'s 前' : d<3600 ? Math.round(d/60)+'m 前' : Math.round(d/3600)+'h 前';
}

/* 分布式测试协议（TM 通用契约 v2，2026-10-07）：目标 / 流程 / 模板格式 / 产出格式
   与 deploy/dist_templates_seed.py + dist_runner_tm.py 严格同步；归总纪律对齐 research_os 13.x 红线 */
const PROTO_STEPS = [
  ['预注册 S2', '模板包 = 受控语料 + 固定 runner，design_sha 内容冻结；改动必须升版本'],
  ['领任务 S3', '节点 register → claim（6h 租约），下载 bundle（corpus.json + runner.py）'],
  ['采集上传 S4', '残差流目标 token（first/mid/last 层）→ 逐维 η² → summary.json + means.npz 分块上传，sha256 内容寻址，design_sha 不符 422 拒收'],
  ['分桶归总 S6', '按 模型×kind×seed 分桶；桶内 median + 符号一致性；禁止跨桶简单平均（13.1 红线）'],
  ['证据升格', '观测 → observed；过迁移干预门（换句式 / held-out 语言复现）才可登记 mechanism_evidence'],
];
const PROTO_FIELDS = [
  ['analysis', "oneway 单因素（TM-01..03）｜factorial 双因子网格（TM-04/05）"],
  ['factors', "因子名对：[topic, frame] / [entity, position] —— 句式=因子水平，变化=操纵"],
  ['layers', "采集层位 [first, mid, last]：first=第1层输出，mid=中层，last=末层"],
  ['items', "{id, cond{因子水平}, text, target}：锚点=target 末位 token，与绝对位置无关"],
  ['fingerprint', "反词嵌入指纹参照集 {within 类内词, out 类外词}——特征 ID 依据纹匹配而非激活峰值"],
  ['eta2_by_factor', "逐维 two-way 分解均值：η²(A)+η²(B)+η²(interaction)=1；交互项=推理组合特征领地"],
  ['per_layer', "每层位一组 η² 分解 —— 位置/内容效应的深度剖面"],
  ['means.npz', "keys / means / eta2_factor_dim(3×d) / cos（cell 方向矩阵）"],
];
/* 首轮结果（qwen3-4b 单节点 · 2026-10-07，登记用，非结论） */
const PROTO_FIRST = [
  ['TM-04 句式×知识', '末层 η²：句式 0.53 > 交互 0.30 > 知识 0.17；指纹类内等角 spread 0.010–0.029、类外 sep 0.046–0.057'],
  ['TM-05 实体×位置', '位置 η² 随深度 0.15→0.52 增、实体 0.77→0.29 减；「末层位置≈0」预注册预测被证伪——登记为新发现（候选解释：末层在编码下一 token 的句法规划）'],
  ['TM-06 前缀扰动（苹果用例）', 'top-32 head 集合 Jaccard：同内容跨前缀 0.73–0.94 ≫ 跨内容 0.28–0.39 → head 招募=内容主导；filler 下目标词 73% 注意力被无关前缀吸走但 head 集合保持；读出方向内容轴互聚（+0.63~0.87）/ 跨内容对立（−0.9）'],
];

function ProtocolCard(){
  return (
    <div className="fw-pl-card">
      <h6>分布式测试协议 <span>TM 通用契约 v2 · 与 deploy/ 模板包严格同步</span></h6>
      <div style={{fontSize:12, lineHeight:1.6, marginBottom:8}}>
        <b>目标</b>：把「句式万千」从噪声翻转为<b>免费的因子操纵</b>——同一知识嵌入 N 种句式、同一实体滑过 N 个位置，
        逐维方差分解把效应按因子归位；特征 ID 依据反词嵌入指纹（类内全 token 行投影等角）而非激活峰值。
        元法则：<b>变化=因子，不变=指纹；η² 定编码，干预定因果</b>。
      </div>
      <div className="fw-agg-steps">
        {PROTO_STEPS.map(([a,b],i)=>(<div key={a} className={'fw-agg-step'+(i<=2?' past':'')}><b>{i+1}. {a}</b><span>{b}</span></div>))}
      </div>
    </div>
  );
}

function ProtocolFormatCard(){
  return (
    <div className="fw-pl-card">
      <h6>模板与产出格式 <span>corpus.json（测试模板） · summary.json v2（产出模板）</span></h6>
      <div className="fw-pl-row head"><span>字段</span><span className="r">说明</span></div>
      {PROTO_FIELDS.map(([k,v])=>(
        <div key={k} className="fw-pl-row" style={{gridTemplateColumns:'110px 1fr', alignItems:'baseline'}}>
          <span className="fw-mono" style={{fontWeight:600}}>{k}</span>
          <span className="dim2" style={{fontSize:11.5, lineHeight:1.5}}>{v}</span>
        </div>
      ))}
      <div style={{marginTop:8, paddingTop:8, borderTop:'1px solid var(--fw-line, rgba(128,128,128,.2))'}}>
        <div style={{fontSize:9.5, color:'var(--fw-text-3, #94a3b8)', marginBottom:4}}>首轮登记注记（静态 · 实时分桶见下方 AGG 卡）：</div>
        {PROTO_FIRST.map(([k,v])=>(
          <div key={k} style={{fontSize:11.5, lineHeight:1.6, marginBottom:4}}>
            <span className="fw-mono" style={{fontWeight:600}}>{k}</span> · <span className="dim2">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* AGG 实时桶（ui_decoupled_plan_v1 §3「首轮结果卡改为读 /api/agg 真值」）：
   对每个有结果的模板拉 GET /api/agg/{tm_id}（AGG-v0 分桶回显，模型×kind×seed），
   桶内摘要只渲染协议键（eta2_by_factor / eta2_by_cond / topk_jaccard）；
   跨桶简单平均被禁止（research_os 13.1），此处只回显不做汇总计算 */
function AggLiveCard({templates}){
  const [agg,setAgg]=useState(null);
  useEffect(()=>{
    let dead=false;
    const tms=(templates||[]).filter(t=>t.results>0).map(t=>t.tm_id);
    if(!tms.length){ setAgg({}); return undefined; }
    Promise.all(tms.map(id=>
      fetch(`${API_BASE}/api/agg/${encodeURIComponent(id)}`).then(r=>r.ok?r.json():null).catch(()=>null)
    )).then(rs=>{ if(!dead) setAgg(Object.fromEntries(tms.map((id,i)=>[id,rs[i]]))); });
    return ()=>{dead=true;};
  },[templates]);
  const entries = agg ? Object.entries(agg) : null;
  return (
    <div className="fw-pl-card">
      <h6>聚合结果 · AGG 实时分桶 <span>{agg ? 'GET /api/agg/{tm_id} · 桶=模型×kind×seed（只回显，禁跨桶平均）' : '拉取中…'}</span></h6>
      {!agg && <div className="fw-pc-empty">拉取聚合桶…</div>}
      {agg && !entries.length && <div className="fw-pc-empty">尚无结果——节点上传后此处自动出现。</div>}
      {entries && entries.map(([tm,a])=>(
        <div key={tm} className="fw-agg-tpl">
          <div className="fw-agg-tpl-h"><b className="fw-mono">{tm}</b><span className="dim2 fw-mono" style={{fontSize:9}}>{a&&a.agg_version}</span></div>
          {a && Object.entries(a.buckets||{}).map(([bk,items])=>{
            const withEta2 = items.map(it=>it.summary&&it.summary.eta2_by_factor).filter(Boolean);
            const eta2 = withEta2.length ? withEta2[0] : null;
            const cond = items.map(it=>it.summary&&it.summary.eta2_by_cond).filter(Boolean)[0]||null;
            const jacc = items.map(it=>it.summary&&it.summary.topk_jaccard).filter(Boolean)[0]||null;
            return (
              <div key={bk} className="fw-agg-bucket">
                <span className="fw-mono dim2" style={{fontSize:9.5}}>{bk}</span>
                <b className="fw-mono" style={{fontSize:10}}>{items.length} 条</b>
                {eta2 && <span className="fw-mono" style={{fontSize:9.5}}>
                  η² {Object.entries(eta2).filter(([,v])=>typeof v==='number').map(([k,v])=>k+' '+v.toFixed(2)).join('/')}
                </span>}
                {cond && <span className="fw-mono dim2" style={{fontSize:9.5}}>{Object.entries(cond).map(([k,v])=>k+' '+(v.toFixed?v.toFixed(2):v)).join(' · ')}</span>}
                {jacc && <span className="fw-mono dim2" style={{fontSize:9.5}}>{Object.entries(jacc).slice(0,2).map(([k,v])=>k+' '+(v.toFixed?v.toFixed(2):v)).join(' · ')}</span>}
              </div>
            );
          })}
          {a && !Object.keys(a.buckets||{}).length && <div className="dim2" style={{fontSize:9.5,padding:'2px 12px 6px'}}>无桶数据</div>}
        </div>
      ))}
    </div>
  );
}

export default function LensProgress({on,onGo}){
  const [view,setView]=useState('machine');
  /* 中心节点实时态（M1 接线）：summary=平台进度、newsLive=行业进展；拉不到则回退 demo */
  const [dist,setDist]=useState(null);
  const [distNews,setDistNews]=useState(null);
  const [newsSrc,setNewsSrc]=useState('');
  useEffect(()=>{
    let dead=false;
    const load=()=>{
      const ctl=new AbortController();
      const t=setTimeout(()=>ctl.abort(),4000);
      Promise.all([
        fetch(`${API_BASE}/api/distributed/summary`,{signal:ctl.signal}).then(r=>r.ok?r.json():null).catch(()=>null),
        fetch(`${API_BASE}/api/news`,{signal:ctl.signal}).then(r=>r.ok?r.json():null).catch(()=>null),
      ]).then(([s,n])=>{
        if(dead) return;
        if(s&&s.version) setDist(s);
        if(n&&n.items&&n.items.length){ setDistNews(n.items); setNewsSrc(n.source||''); }
      }).catch(()=>{}).finally(()=>clearTimeout(t));
    };
    load();
    const iv=setInterval(load,30000);
    return ()=>{dead=true;clearInterval(iv);};
  },[]);

  /* 平台进度数据源：LIVE（中心节点）或 DEMO（distributedData.js） */
  const liveStats = dist ? [
    {n:dist.nodes_online, l:'在线节点', s:'共 '+dist.nodes_total+' 注册'},
    {n:dist.templates.filter(t=>t.status==='open').length, l:'开放模板', s:'模板注册表 S2'},
    {n:dist.results_total, l:'已上传结果', s:'sha256 内容寻址'},
    {n:dist.downloads_total, l:'结果下载', s:'他人引用计数'},
    {n:dist.agg_version, l:'聚合版本', s:dist.version},
  ] : PLATFORM_STATS;
  const liveNodes = dist ? dist.nodes.map(n=>({
    id:n.node_id, gpu:n.gpu||'—', model:n.model||n.name||'—', tm:'—', prog:0,
    cls: (Date.now()/1000-n.last_seen<120)?'run':'idle', up:fmtAgo(n.last_seen),
  })) : NODES;
  const liveTpls = dist ? dist.templates.map(t=>({
    id:t.tm_id, dim:t.dim, name:t.name, me:false,
    agg: t.results>0 ? 'collecting' : 'todo',
    nodes:t.claims_active, cells:t.results+' 结果',
  })) : TEMPLATES;
  const tplTip = t => t.name+' · '+(t.nodes||0)+' 节点执行中 · '+t.cells;
  const tplAggTxt = t => t.agg==='done'?'已聚合':(t.agg==='collecting'?'收集中':'待分发');
  const newsItems = distNews || NEWS;
  return (
    <section className={'fw-view fw-progress'+(on?' on':'')}>
      <div className="fw-rg-inner">
        {/* ===== 顶层三 tab：行业进展 / 平台进度 / 当前机器进度 ===== */}
        <div className="fw-pt-tabs" role="tablist">
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='news'?' on':'')} onClick={()=>setView('news')}>行业进展<small>机制可解释性动态</small></button>
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='platform'?' on':'')} onClick={()=>setView('platform')}>平台进度<small>{dist ? dist.nodes_online+' 节点在线 · '+dist.templates.length+' 模板' : '47 节点 · 12 模板'}</small></button>
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='machine'?' on':'')} onClick={()=>setView('machine')}>当前机器进度<small>NODE-A3F7 · 本机</small></button>
        </div>

        {/* ===== 当前机器进度：本机节点 + RDC 主线 ===== */}
        {view==='machine'&&(<>
          {/* 本机节点横幅（demo：接 GET /api/distributed/summary 后取真实本机状态） */}
          <div className="fw-node-banner">
            <span className="fw-nid">{LOCAL_NODE.id}</span>
            <span className="fw-nchip">{LOCAL_NODE.gpu}</span>
            <span className="fw-nchip">{LOCAL_NODE.model}</span>
            <span className={'fw-nstat '+LOCAL_NODE.status}>● 执行中</span>
            <span className="fw-ntask">{LOCAL_NODE.task}</span>
            <span className="fw-nprog"><i style={{width:(LOCAL_NODE.progress*100).toFixed(0)+'%'}}/></span>
            <span className="fw-nmono">{LOCAL_NODE.cells}/{LOCAL_NODE.cells_total} cells</span>
            <span className="fw-nmono">上传 {LOCAL_NODE.uploaded}</span>
            <span className="fw-nmono dim">{LOCAL_NODE.server}</span>
          </div>

          <Lane color="#0284c7" label="自己的 · RDC 主线" small="F#3734 所在证据链 · Phase 4 → Q07 · 点击节点看详情">
            <Timeline items={MAINLINE} tag="atlas_ledger.json"/>
          </Lane>

          <div className="fw-rg-note">
            <b>路线图逻辑</b>：三个 tab 共享同一套证据口径——本机进度挂真实实验（节点必带 sealed 产物），平台进度对应可运行的版本与聚合产物（AGG-v0 分桶 → v1 跨节点 η²），业界以新闻流与对比卡登记视角差（has_data / observed / generalization_checked / mechanism_evidence 四级体系）。节点接入：见 deploy/README_CENTOS.md（node_agent.py register → run）。
          </div>
          <div style={{textAlign:'center',padding:'4px 0 10px'}}>
            <button className="fw-tbtn" onClick={()=>onGo('process')}>回到当前任务 →</button>
          </div>
        </>)}

        {/* ===== 平台进度：分布式统计 + 节点 + 模板矩阵 + 聚合分析 + 发布时间线 ===== */}
        {view==='platform'&&(<>
          <div className="fw-pl-stats">
            <span className={'fw-src-chip '+(dist?'live':'demo')} title={dist?dist.version:'接 GET /api/distributed/summary 后显示实时数据'}>
              {dist?'● LIVE 中心节点':'○ DEMO（中心节点离线）'}
            </span>
            {liveStats.map(s=>(
              <div key={s.l} className="fw-pl-stat"><div className="n">{s.n}</div><div className="l">{s.l}</div><div className="s">{s.s}</div></div>
            ))}
          </div>
          <div className="fw-pl-grid">
            {/* 节点列表（LIVE=summary.nodes 最近心跳；DEMO=distributedData.js） */}
            <div className="fw-pl-card">
              <h6>节点目录 <span>{dist?'按最近心跳排序 · '+liveNodes.length+' 台展示':'8/47 展示 · 接 /api/distributed/summary'}</span></h6>
              <div className="fw-pl-row head"><span>节点</span><span>硬件</span><span>当前模板</span><span>进度</span><span className="r">上传</span></div>
              {liveNodes.map(n=>(
                <div key={n.id} className="fw-pl-row">
                  <span className={'fw-mono'+(n.me?' me':'')}>{n.id}{n.me?'（本机）':''}</span>
                  <span className="dim2">{n.gpu} · {n.model}</span>
                  <span>{n.tm}</span>
                  <span><span className="fw-pl-prog"><i style={{width:(n.prog*100).toFixed(0)+'%'}}/></span></span>
                  <span className={'r fw-mono fw-st-'+n.cls}>{n.cls==='agg'?'聚合中':(n.cls==='run'?'在线':'空闲')} · {n.up}</span>
                </div>
              ))}
            </div>
            {/* 模板矩阵 + 聚合管线 */}
            <div style={{display:'flex',flexDirection:'column',gap:10,minWidth:0}}>
              <div className="fw-pl-card">
                <h6>模板测试矩阵 <span>每模板=受控语料+固定口径，可分发任意机器</span></h6>
                <div className="fw-tmx">
                  {liveTpls.map(t=>(
                    <div key={t.id} className={'fw-tmx-cell'+(t.me?' me':'')} title={tplTip(t)}>
                      <b>{t.id}</b>
                      <span className="fw-tmx-dim">{t.dim}</span>
                      <span className={'fw-tmx-agg '+t.agg}>{tplAggTxt(t)}</span>
                      <span className="fw-tmx-n">{t.nodes||0}节点 · {t.cells}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="fw-pl-card">
                <h6>聚合分析管线 <span>综合大量模板结果 → 语言编码机制</span></h6>
                <div className="fw-agg-steps">
                  {AGG_STEPS.map(([a,b],i)=>(
                    <div key={a} className={'fw-agg-step'+(i<=3?' past':'')}>
                      <b>{i+1}. {a}</b><span>{b}</span>
                    </div>
                  ))}
                </div>
                <div className="fw-agg-dims">
                  {AGG_DIMS.map(d=>(
                    <div key={d.dim} className="fw-agg-dim">
                      <b>{d.dim}</b>
                      <span className={'fw-tmx-agg '+d.state}>{d.state==='done'?'已聚合':(d.state==='partial'?'部分':'待开始')}</span>
                      <p>{d.note} <em>{d.src}</em></p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ===== 分布式测试协议（v2 通用契约）：目标/流程/格式 —— 任意句型可扩展 ===== */}
          <div style={{display:'grid', gridTemplateColumns:'minmax(280px,5fr) minmax(320px,7fr)', gap:10, alignItems:'stretch'}}>
            <ProtocolCard/>
            <ProtocolFormatCard/>
          </div>

          {/* ===== AGG 实时分桶（/api/agg 真值，协议键渲染）——首轮静态登记见上卡注记 ===== */}
          <AggLiveCard templates={dist ? dist.templates : []}/>

          <Lane color="#059669" label="平台发布" small="可视化客户端与服务端的版本时间线">
            <Timeline items={RELEASES} tag="release registry"/>
          </Lane>
        </>)}

        {/* ===== 行业进展：领域阶段时间轴（点击节点看阶段/论文） + 实时动态流 + 同行对比卡 ===== */}
        {view==='news'&&(<>
          <div className="fw-news-head">
            行业动态 · 机制可解释性
            <span className={'fw-src-chip '+(distNews?'live':'demo')} title={distNews?'GET /api/news · '+newsSrc:'中心节点离线，显示内置兜底列表'}>
              {distNews?'● LIVE '+(newsSrc==='arxiv'?'arXiv 抓取':'中心节点'):'○ DEMO（内置兜底）'}
            </span>
            <span>领域五阶段时间轴 · 最新论文见时间轴 NOW 节点{distNews?' · 每 30s 自动刷新':' · 接入 RSS / arXiv / 官方博客后自动更新'}</span>
          </div>
          <EraTimeline eras={MI_ERAS} live={{newsItems,newsLive:!!distNews,newsSrc}}/>

          {/* ===== 表征相似性分析（RSA）：第二条阶段时间轴（格式与机制可解释性时间轴一致） ===== */}
          <div className="fw-news-head" style={{marginTop:16}}>
            表征相似性分析（RSA）
            <span>方法演进时间轴 · 比较的是「关系结构」而非坐标 · 跨模型/跨物种可比 · 点击节点看阶段与论文</span>
          </div>
          <EraTimeline eras={RSA_ERAS}/>
          <div className="fw-rg-note" style={{marginTop:10}}>
            <b>与本项目证据链的对比</b>：{RSA_NOTE}
          </div>

          <Lane color="#6366f1" label="行业主流 · 对比卡" small="点击对比卡 → 与本项目证据链的视角差">
            <div className="fw-peer-row">{PEERS_ROW1.map((c,i)=><PeerCard key={i} c={c}/>)}</div>
            <div className="fw-peer-row" style={{marginBottom:0}}>
              {PEERS_ROW2.map((c,i)=><PeerCard key={i} c={c}/>)}
            </div>
          </Lane>
        </>)}
      </div>
    </section>
  );
}
