/* 脉络透镜 v8（本地进度归一版）：两 tab —— 行业进展 / 图谱
   - 行业进展：领域阶段时间轴（MI / RSA）+ 实时论文流（/api/news）+ 同行对比卡
   - 图谱：AtlasLens 整体嵌入（图谱全景/跨线账本/结果库/模板与技术），?lens=atlas 兼容路由 + ?atlasSel=FTR-xx 直开特征详情
   P4 归一（design/client_mission_ia_review_v1.md）：「本地进度」tab 撤销——
   本机节点横幅（LOCAL_NODE 静态 demo 死件）删除；调度队列快照（/api/tasks）并入研发·平台运行·节点与租约
   （PlatformOps v3.4，本机状态唯一家）；RDC 主线摘要（≡账本/总览最近闭环，第三家）删除，
   唯一 Phase 时间线 = 图谱·跨线账本。
   LIVE/DEMO 范式：服务不可达显示空态提示，不伪造数据。 */
import { API_BASE } from '../../../config.js';
import { useEffect, useState } from 'react';
import { NEWS, MI_ERAS, RSA_ERAS, RSA_NOTE, TECH_CATEGORIES } from './distributedData.js';
import { Lane } from './TimelineBits.jsx';
import AtlasLens from './AtlasLens.jsx';

/* API 前缀：与研发透镜（LensProcess）同一惯例；禁止在组件里散落硬编码地址 */

/* 本地进度 · RDC 主线摘要注记：唯一事实源 = 图谱透镜·跨线账本（P1 起不再硬编码），
   本透镜「本地进度」只渲染 GET /api/ai-rnd/atlas/ledger 的最近闭环摘要。 */

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

/* M7-P1 技术类别查找（PROJECTIONS/era cat → 名称/颜色/短名，数据 ← TECH_CATEGORIES） */
const catOf=id=>TECH_CATEGORIES.find(c=>c.id===id)||{};
const catColor=id=>(catOf(id).color)||'#888780';
const catShort=id=>(catOf(id).short)||id;

/* 行业进展 · 领域阶段时间轴（可复用）：机制可解释性（MI_ERAS）与表征相似性分析（RSA_ERAS）各一条
   数据 ← eras（阶段数组）；live 可选——传入时最右追加「最新」实时节点（← /api/news）。
   点击节点 → 下方详情面板：阶段目标 / 成果 / 代表论文（可点击跳原文）。
   M7-P1：era 带 cat（四类技术归属）→ 顶部类别筛选 chips，非命中类别置灰（不隐藏，保时间轴完整）。 */
function EraTimeline({eras,live}){
  const [sel,setSel]=useState(eras[0].id);
  const [catSel,setCatSel]=useState('all');   // M7-P1 类别筛选
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
      <div className="fw-cat-chips">
        <button type="button" className={'fw-proj-chip'+(catSel==='all'?' on':'')} onClick={()=>setCatSel('all')}><span>全部类别</span></button>
        {TECH_CATEGORIES.map(c=>(
          <button key={c.id} type="button" className={'fw-proj-chip'+(catSel===c.id?' on':'')} style={{'--cat':c.color}}
                  title={c.name} onClick={()=>setCatSel(c.id)}>
            <i style={{background:c.color}}/><span>{c.short||c.name}</span>
          </button>
        ))}
      </div>
      <div className="fw-era-tl">
        {eras.map(e=>{
          const dim=catSel!=='all'&&e.cat!==catSel;
          return (
            <button key={e.id} type="button" className={'fw-era-node'+(sel===e.id?' sel':'')+(dim?' dim':'')} onClick={()=>setSel(e.id)}>
              <span className="fw-era-dot"/>
              <span className="fw-era-years">{e.years}</span>
              <span className="fw-era-name">{e.title}</span>
              <span className="fw-era-cat" style={{color:catColor(e.cat)}}>{catShort(e.cat)}</span>
            </button>
          );
        })}
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

export default function LensProgress({on,onGo,initialView,initialAtlasTab,initialAtlasSel}){
  const [view,setView]=useState(initialView||'news');
  /* v6：对比卡默认隐藏，点击展开 */
  const [peersOpen,setPeersOpen]=useState(false);
  /* initialView 变化响应（lens=atlas 别名路由 → 图谱 tab） */
  useEffect(()=>{ if(initialView) setView(initialView); },[initialView]);
  /* 中心节点实时态（M1 接线）：newsLive=行业进展；拉不到则回退 demo
     （P4：/api/tasks 与 ledger 拉取随「本地进度」tab 撤销——调度快照在 PlatformOps，时间线在图谱·账本） */
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

  const newsItems = distNews || NEWS;
  return (
    <section className={'fw-view fw-progress'+(on?' on':'')}>
      <div className="fw-rg-inner">
        {/* ===== 顶层两 tab：行业进展 / 图谱（本地进度已归一：调度→研发·平台运行，时间线→图谱·账本） ===== */}
        <div className="fw-pt-tabs" role="tablist">
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='news'?' on':'')} onClick={()=>setView('news')}>行业进展<small>机制可解释性 · 表征相似性分析</small></button>
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='atlas'?' on':'')} onClick={()=>setView('atlas')}>图谱<small>三域全景 · 账本 · 结果库 · 模板与技术</small></button>
        </div>

        {/* ===== 图谱：AtlasLens 整体嵌入（initialAtlasSel ← 顶栏特征源选择器 ?atlasSel=） ===== */}
        {view==='atlas'&&<AtlasLens on onGo={onGo} initialTab={initialAtlasTab} initialSel={initialAtlasSel}/>}

        {/* ===== 行业进展：两大部分（机制可解释性 / 表征相似性分析），每部分收进一个模块 ===== */}
        {view==='news'&&(<>
          {/* ===== PART 01 机制可解释性：领域阶段时间轴（含 NOW 实时流）+ 同行对比卡 ===== */}
          <section className="fw-domain">
            <header className="fw-domain-hd">
              <span className="fw-domain-idx">PART 01</span>
              <div className="fw-domain-tt">
                <h3>机制可解释性</h3>
                <p className="fw-domain-sub">模型内部怎么算 · 领域阶段时间轴 + 实时论文流 + 同行对比卡 · 点击时间轴节点看阶段与论文</p>
              </div>
              <span className={'fw-src-chip '+(distNews?'live':'demo')} title={distNews?'GET /api/news · '+newsSrc:'中心节点离线，显示内置兜底列表'}>
                {distNews?'● LIVE '+(newsSrc==='arxiv'?'arXiv 抓取':'中心节点'):'○ DEMO（内置兜底）'}
              </span>
            </header>
            <EraTimeline eras={MI_ERAS} live={{newsItems,newsLive:!!distNews,newsSrc}}/>
            <div className="fw-peers-sec">
              <button type="button" className={'fw-peers-btn'+(peersOpen?' on':'')} onClick={()=>setPeersOpen(v=>!v)}
                      title={peersOpen?'收起对比卡':'展开对比卡'}>
                行业主流 · 对比卡<small>{peersOpen?'点击收起':'点击展开 · 5 条视角差'}</small>
              </button>
              {peersOpen&&(
                <Lane color="#6366f1" label="行业主流 · 对比卡" small="点击对比卡 → 与本项目证据链的视角差">
                  <div className="fw-peer-row">{PEERS_ROW1.map((c,i)=><PeerCard key={i} c={c}/>)}</div>
                  <div className="fw-peer-row" style={{marginBottom:0}}>
                    {PEERS_ROW2.map((c,i)=><PeerCard key={i} c={c}/>)}
                  </div>
                </Lane>
              )}
            </div>
          </section>

          {/* ===== PART 02 表征相似性分析：方法演进时间轴 + 与本项目证据链对比 ===== */}
          <section className="fw-domain alt">
            <header className="fw-domain-hd">
              <span className="fw-domain-idx">PART 02</span>
              <div className="fw-domain-tt">
                <h3>表征相似性分析</h3>
                <p className="fw-domain-sub">模型内部怎么表示 · 方法演进时间轴 · 比较的是「关系结构」而非坐标，跨模型/跨物种可比 · 点击节点看阶段与论文</p>
              </div>
            </header>
            <EraTimeline eras={RSA_ERAS}/>
            <div className="fw-rg-note" style={{marginTop:10}}>
              <b>与本项目证据链的对比</b>：{RSA_NOTE}
            </div>
          </section>
        </>)}
      </div>
    </section>
  );
}
