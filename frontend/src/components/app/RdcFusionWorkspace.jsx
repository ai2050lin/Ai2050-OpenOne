/* ============================================================
   RdcFusionWorkspace —— v5.1 四透镜融合驾驶舱（路由 /rdc-fusion）
   融合主张：3D 可视化 / AI 研发 / 行业进展 / 数据结果 = 同一研究对象的
   四个透镜（空间·过程·脉络·数据），共享对象路由 + 对象卡 + 事件总线。
   M3-P2/P3（design/ui_decoupled_plan_v2.md §3–4）：
   - 对象卡 = 纯渲染器：只认 object_card.v1 schema（/api/object/{fid}，
     注册表 server/object_registry.json；未注册 404 → DEMO_OBJECT 回退）
   - URL 即状态：?lens=&obj=（对象路由落地，⌘K / 面包屑 / 跨透镜链接联动）
   - ⌘K / HomeView 统计 / ticker：live 优先（queue/objects/distributed summary），
     接不通回退 demoData.js 并保持 DEMO 徽标或注记。
   视觉 token：slate 体系 + emerald 激活梯度。
   ============================================================ */
import { useEffect, useRef, useState } from 'react';
import LensSpatial from './rdc_fusion/LensSpatial.jsx';
import LensProcess from './rdc_fusion/LensProcess.jsx';
import LensProgress from './rdc_fusion/LensProgress.jsx';
import NewbieGuide from './rdc_fusion/NewbieGuide.jsx';
import AtlasPanorama from './rdc_fusion/AtlasPanorama.jsx';
import { TICKER, CMDK_GROUPS, DEMO_OBJECT, DEMO_QUEUE } from './rdc_fusion/demoData.js';
import { MISSION } from './rdc_fusion/distributedData.js';
import './rdc_fusion/rdc_fusion.css';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');
const DEFAULT_OBJ = 'F#3734';

/*    五轴模型（design/client_ia_refactor_plan_v1.md §3）：导航 6→5。
   技术透镜已拆分：「按结果」面 → 图谱透镜·结果库；「按技术」面 → 研发透镜·技术运行。
   P1+P2（design/client_mission_ia_review_v1.md）：总览改「图谱驾驶舱」——
   主体 = AtlasPanorama 三域全景只读态（打开即图谱）；事件流（DEMO）→ 最近闭环（接 atlas ledger）；
   通知 badge 去写死（= queue pending 数）；入口卡 ×4 撤销（使命链压缩为一行 + 快速入口三按钮）。 */
const LENS_NAME={spatial:'空间 · 几何',process:'研发 · 生产',progress:'路线 · 时间线',atlas:'路线 · 图谱',home:'总览 · 图谱驾驶舱'};

function Tok({t,dfa}){
  const [txt,v]=t;
  const style=v>0?{background:'rgba(52,211,153,'+v+')',color:v>0.6?'#022c22':'#064e3b'}:undefined;
  return <span className={'fw-tok'+(dfa?' dfa':'')} style={style}>{txt}</span>;
}

/* ── 对象卡渲染器（object_card.v1 schema 驱动；LIVE/DEMO 同构） ──
   2026-10-08 瘦身：metrics/links 静态叙事已删（防第二事实源腐化），
   保留 ID/层位/激活示例/技术足迹/派实验动作；指标类内容以分析技术视图为准 */
function ObjectCard({obj,live,onDispatch}){
  const acts=obj.activations||[];
  const evClass='fw-ev-badge '+(obj.evidence==='mechanism_evidence'?'fw-ev-mech':'fw-ev-obs');
  return (
    <>
      <div className="fw-f-name">{obj.label}</div>
      <div className="fw-f-meta">
        <div className="kv">特征 ID<b className="fw-mono">{obj.id}</b></div>
        {obj.layer&&<div className="kv">层 / 位置<b>{obj.layer}</b></div>}
        {Array.isArray(obj.tech_footprint)&&obj.tech_footprint.length>0&&(
          <div className="kv" title="分析技术足迹（analysis registry）：已对该对象运行过的技术 · 到分析技术可再次运行">
            技术足迹<b className="fw-mono">{obj.tech_footprint.map(t=>t.tech).join(' · ')}</b>
          </div>
        )}
      </div>

      {acts.length>0&&(
        <div className="fw-act-box">
          <div className="fw-act-head"><span>ACTIVATIONS · 最大激活示例</span><span>max {(() => {
            let mx=0; acts.forEach(l=>(l.toks||[]).forEach(t=>{ if(t[1]>mx) mx=t[1]; })); return mx.toFixed(2);
          })()}</span></div>
          <div className="fw-act-text">
            {acts.map((line,i)=>(
              <div key={i}>
                {(line.toks||[]).map((t,j)=><Tok key={j} t={t} dfa={line.dfa}/>)}
                {line.dfa&&<span className="fw-act-note">{line.note||' ← 一词多义对照（弱激活）'}</span>}
                <br/>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="fw-side-actions">
        <button className="fw-tbtn primary" onClick={onDispatch}>＋ 派 AI 实验</button>
      </div>
      <div className={'fw-src-chip mini '+(live?'live':'demo')}>{live?'对象 LIVE（注册表+结果富化）':'对象 DEMO'}</div>
    </>
  );
}

/* HomeView · 图谱驾驶舱（P2 改造，design/client_mission_ia_review_v1.md）：
   使命主产物 = 语言机制图谱 → 总览主体 = 图谱三域全景（只读，复用 AtlasPanorama），
   侧栏 = 生产状态（queue）/ 最近闭环（接 atlas ledger，替代原 DEMO 事件流）/ 快速入口。
   hero 压缩为使命一行；入口卡 ×4 撤销（与导航重复，S3 命名漂移一并消除）。 */
function HomeView({onGo,home,ledger}){
  const st=home||{sealed:DEMO_QUEUE.sealed,count:DEMO_QUEUE.count,tpls:'—',results:'—',objects:1,queue:DEMO_QUEUE.queue};
  const q=st.queue||DEMO_QUEUE.queue;
  const nx=q.find(x=>x.status==='pending');
  const pend=q.filter(x=>x.status==='pending')[1];
  const last=[...q].reverse().find(x=>x.status==='sealed');
  const rows=ledger?ledger.measurements:[];
  return (
    <div className="fw-view fw-home on">
      <div className="fw-hero">
        <div className="fw-mission">
          <h2><b className="fw-brand">{MISSION.brand}</b>{MISSION.title}</h2>
          <div className="fw-mission-line">使命：破解 LLM 语言编码机制 —— 主产物 = 语言机制图谱（知识网络 / 分析推理 / 语法系统 → 语言族 → 特征 × E 级证据）</div>
        </div>
        <div className="fw-hero-num">
          <div className="stat"><div className="n">{st.count?`${st.sealed}/${st.count}`:'—'}</div><div className="l">队列 sealed</div></div>
          <div className="stat"><div className="n">{st.tpls}</div><div className="l">测试模板</div></div>
          <div className="stat"><div className="n">{st.results}</div><div className="l">结果条目</div></div>
        </div>
      </div>

      <div className="fw-home-atlas">
        <div className="fw-home-atlas-hd">
          <b>语言机制图谱 · 三域全景</b>
          <button className="fw-tbtn" onClick={()=>onGo('atlas')} title="路线 → 图谱：全景交互 / 特征详情 / 账本 / 结果库">打开图谱工作区 →</button>
        </div>
        <AtlasPanorama onPick={()=>onGo('atlas')}/>
      </div>

      <div className="fw-side-col">
        <div className="fw-mini">
          <h4>进行中 <span style={{fontSize:10,color:'var(--fw-blue)',cursor:'pointer'}} onClick={()=>onGo('process')}>全部 →</span></h4>
          {nx&&<div className="fw-task"><span className="st fw-st-run"/><div className="t"><b>{nx.id} · {nx.title}</b><span>next · {nx.deliverable||''}</span></div></div>}
          {pend&&<div className="fw-task"><span className="st fw-st-que"/><div className="t"><b>{pend.id} · {pend.title}</b><span>queued · {pend.block||''}</span></div></div>}
          {last&&<div className="fw-task"><span className="st fw-st-done"/><div className="t"><b>{last.id} · {last.title}</b><span>sealed {last.res_sha8||''}</span></div></div>}
        </div>
        <div className="fw-mini">
          <h4>最近闭环 {ledger&&<span style={{fontSize:10,color:'var(--fw-blue)',cursor:'pointer'}} onClick={()=>onGo('atlas','ledger')}>全部 {ledger.total} 条 →</span>}</h4>
          {rows.map(m=>(
            <div key={m.phase} className="fw-ledger-row" onClick={()=>onGo('atlas','ledger')} title={(m.verdict||'')+' · 点击打开跨线账本'}>
              <span className="mono">P{m.phase}</span><span className="t">{m.name}</span><span className="ln">{m.line}</span>
            </div>
          ))}
          {!ledger&&<div className="fw-evt"><span>接 GET /api/ai-rnd/atlas/ledger 后显示实时闭环；服务未启动时空态，不伪造数据。</span></div>}
        </div>
        <div className="fw-mini">
          <h4>快速入口</h4>
          <div className="fw-home-go">
            <button type="button" onClick={()=>onGo('spatial')}>空间 · 3D 几何与实时权重</button>
            <button type="button" onClick={()=>onGo('process')}>研发 · 图谱生产闭环</button>
            <button type="button" onClick={()=>onGo('progress')}>路线 · 进展与时间线</button>
          </div>
        </div>
      </div>
      <NewbieGuide/>
    </div>
  );
}

export default function RdcFusionWorkspace(){
  const params=new URLSearchParams(typeof window!=='undefined'?window.location.search:'');
  const [lens,setLens]=useState(()=>params.get('lens')||'spatial');
  const [objId,setObjId]=useState(()=>params.get('obj')||DEFAULT_OBJ);
  const [atlasTab,setAtlasTab]=useState(()=>params.get('atlasTab')||null);  // 图谱透镜子 tab（URL 即状态）
  const [atlasSel,setAtlasSel]=useState(()=>params.get('atlasSel')||null);  // 图谱特征选中（?atlasSel=FTR-xx；P3 特征源选择器）
  const [atlasReg,setAtlasReg]=useState(null);   // /api/ai-rnd/atlas/registry features（特征源选择器数据源，LIVE）
  const [featOpen,setFeatOpen]=useState(false);  // 特征源下拉开合
  const [objLive,setObjLive]=useState(null);     // /api/object/{fid} → {object,related_results}
  const [objects,setObjects]=useState(null);     // /api/objects
  const [rndQueue,setRndQueue]=useState(null);   // /api/ai-rnd/queue
  const [distStats,setDistStats]=useState(null); // /api/distributed/summary
  const [atlasLedger,setAtlasLedger]=useState(null); // /api/ai-rnd/atlas/ledger（总览·最近闭环，替代 DEMO 事件流）
  const [cmdk,setCmdk]=useState(false);
  const [about,setAbout]=useState(false);
  // const [sideOpen,setSideOpen]=useState(false);  // 对象卡已隐藏，恢复时连同右侧 aside 一起取消注释
  const queryRef=useRef(null);

  const syncURL=(l,o,t,s)=>{
    try{
      let url=`?lens=${l}&obj=${encodeURIComponent(o)}`;
      if(l==='atlas'&&t) url+=`&atlasTab=${encodeURIComponent(t)}`;
      if(s) url+=`&atlasSel=${encodeURIComponent(s)}`;   // 跨透镜贯通：图谱/空间 3D 共用同一选中态
      window.history.replaceState(null,'',url);
    }catch{ /* 隐私模式等 */ }
  };
  const go=(l,t,s)=>{ setLens(l); if(s!==undefined) setAtlasSel(s||null); if(l==='atlas') setAtlasTab(t||null); setAbout(false); setCmdk(false); setFeatOpen(false); syncURL(l,objId,t,s); };
  const selectObj=id=>{ setObjId(id); setCmdk(false); setAbout(false); syncURL(lens,id); };

  /* 对象数据：LIVE 优先，未注册/离线 → DEMO_OBJECT */
  useEffect(()=>{
    let dead=false;
    (async()=>{
      try{
        const r=await fetch(`${API_BASE}/api/object/${encodeURIComponent(objId)}`);
        const p=r.ok?await r.json():null;
        if(!dead) setObjLive(p&&p.registered&&p.object?p:null);
      }catch{ if(!dead) setObjLive(null); }
    })();
    return ()=>{ dead=true; };
  },[objId]);

  useEffect(()=>{
    let dead=false;
    (async()=>{
      try{ const r=await fetch(`${API_BASE}/api/objects`); const p=r.ok?await r.json():null;
        if(!dead&&p&&Array.isArray(p.objects)&&p.objects.length) setObjects(p.objects); }catch{ /* demo */ }
      try{ const r=await fetch(`${API_BASE}/api/ai-rnd/queue`); const p=r.ok?await r.json():null;
        if(!dead&&p&&Array.isArray(p.queue)&&p.queue.length) setRndQueue(p); }catch{ /* demo */ }
      try{ const r=await fetch(`${API_BASE}/api/distributed/summary`); const p=r.ok?await r.json():null;
        if(!dead&&p&&p.templates) setDistStats(p); }catch{ /* demo */ }
      try{ const r=await fetch(`${API_BASE}/api/ai-rnd/atlas/ledger?limit=8`); const p=r.ok?await r.json():null;
        if(!dead&&p&&Array.isArray(p.measurements)) setAtlasLedger(p); }catch{ /* 空态 */ }
      try{ const r=await fetch(`${API_BASE}/api/ai-rnd/atlas/registry`); const p=r.ok?await r.json():null;
        if(!dead&&p&&Array.isArray(p.features)) setAtlasReg(p.features); }catch{ /* 空态 */ }
    })();
    return ()=>{ dead=true; };
  },[]);

  useEffect(()=>{
    const onKey=e=>{
      if((e.metaKey||e.ctrlKey)&&String(e.key).toLowerCase()==='k'){e.preventDefault();setCmdk(v=>!v);}
      if(e.key==='Escape'){setCmdk(false);setAbout(false);setFeatOpen(false);}
    };
    window.addEventListener('keydown',onKey);
    return ()=>window.removeEventListener('keydown',onKey);
  },[]);
  useEffect(()=>{ if(cmdk&&queryRef.current) queryRef.current.focus(); },[cmdk]);

  const obj=(objLive&&objLive.object)||DEMO_OBJECT;
  const objIsLive=Boolean(objLive);
  const tickerText=(()=>{
    if(rndQueue){
      const nx=rndQueue.queue.find(x=>x.status==='pending');
      const last=[...rndQueue.queue].reverse().find(x=>x.status==='sealed');
      const rows=[`[queue] ${rndQueue.sealed}/${rndQueue.count} sealed · next ${nx?nx.id:'—'}`];
      if(last) rows.push(`[queue] ${last.id} sealed ${last.res_sha8||''}`);
      if(distStats) rows.push(`[templates] ${distStats.templates.length} · 结果 ${distStats.results_total}`);
      rows.push(`[objects] 注册对象 ${objects?objects.length:1}`);
      return rows;
    }
    return TICKER;
  })();

  const cmdkGroups=[
    {gl:'对象 · FEATURES',items:(objects&&objects.length?objects:[{id:obj.id,label:obj.label,layer:obj.layer}]).map(o=>({k:o.id,t:o.label,d:o.layer||'',go:'spatial',obj:o.id}))},
    {gl:'任务 · TASKS',items:(()=>{
      if(!rndQueue) return CMDK_GROUPS[1].items;
      const nx=rndQueue.queue.find(x=>x.status==='pending');
      const sealed=[...rndQueue.queue].reverse().filter(x=>x.status==='sealed').slice(0,2);
      const items=[];
      if(nx) items.push({k:nx.id,t:nx.title+' · NEXT',d:nx.block||'',go:'process'});
      sealed.forEach(s=>items.push({k:s.id,t:s.title+' · SEALED',d:s.res_sha8||'',go:'process',obj:objId}));
      return items;
    })()},
    CMDK_GROUPS[2],
  ];

  const ticker=tickerText.concat(tickerText);

  return (
    <div className="fw-root">
      {/* ===== 顶栏：全局上下文 ===== */}
      <header className="fw-topbar">
        <div className="fw-logo"><div className="fw-logo-mark">A2</div>Ai2050 Atlas <small>v5 · RDC</small></div>
        <button className="fw-select" title="演示选择器（模型接入后由此驱动）"><span className="dot" style={{background:'#6366f1'}}/><b>qwen3-4b</b><span className="car">▾</span></button>
        {/* 特征源选择器（P3-lite）：LIVE = atlas registry 在册特征（FTR-xx），选中 → 图谱全景打开该特征详情；
            不可达时显示注记空态（不伪造特征列表）。对象宇宙归一挂账：FTR→点云真实坐标映射见
            design/atlas_pointcloud_mapping_protocol_v1.md */}
        <div className="fw-select-wrap">
          <button className="fw-select" onClick={()=>setFeatOpen(v=>!v)}
                  title="特征源 = 语言机制图谱在册特征（GET /api/ai-rnd/atlas/registry）；选中后打开图谱全景详情">
            <b className="fw-mono">{atlasSel||'特征源'}</b><span style={{color:'var(--fw-text-3)',fontSize:10}}>图谱 FTR</span><span className="car">▾</span>
          </button>
          {featOpen&&(
            <div className="fw-feat-menu">
              {atlasReg ? atlasReg.map(f=>(
                <div key={f.id} className="fw-feat-item" title={f.statement}
                     onClick={()=>{ setFeatOpen(false); go('atlas',null,f.id); }}>
                  <span className="k fw-mono">{f.id}</span>
                  <span className="t">{f.statement}</span>
                </div>
              )) : (
                <div className="fw-feat-item dim">接 GET /api/ai-rnd/atlas/registry 后列出在册特征（服务未启动时空态，不伪造）。</div>
              )}
              {atlasReg&&atlasSel&&(
                <div className="fw-feat-item clear" onClick={()=>{ setFeatOpen(false); go('atlas',null,null); }}>清除选中（回到全景默认）</div>
              )}
            </div>
          )}
        </div>
        <div className="fw-crumb">
          <button className="obj" onClick={()=>selectObj(objId)} title={obj.label}>{objId}</button>
          <span className="lens-chip">{LENS_NAME[lens]}</span>
        </div>
        <button className="fw-cmdk" onClick={()=>setCmdk(true)}>搜索对象、任务、论文… <kbd>⌘K</kbd></button>
        <button className="fw-bell" title="任务通知（待处理 Phase 数，实时）">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.7 21a2 2 0 0 1-3.4 0"/></svg>
          {(()=>{ const n=rndQueue?rndQueue.queue.filter(x=>x.status==='pending').length:0; return n>0?<span className="badge">{n}</span>:null; })()}
        </button>
        <div className="fw-avatar">R</div>
      </header>

      <div className="fw-frame">
        {/* ===== 左轨 ===== */}
        <nav className="fw-rail">
          <button className={'fw-rbtn'+(lens==='home'?' on':'')} onClick={()=>go('home')} title="总览">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 10.5 12 3l9 7.5V21H3z"/></svg>总览
          </button>
          <div className="fw-rdiv"/>
          <button className={'fw-rbtn'+(lens==='spatial'?' on':'')} onClick={()=>go('spatial')} title="空间透镜 3D">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><ellipse cx="12" cy="12" rx="10" ry="4.2"/><ellipse cx="12" cy="12" rx="10" ry="4.2" transform="rotate(60 12 12)"/></svg>空间
          </button>
          <button className={'fw-rbtn'+(lens==='process'?' on':'')} onClick={()=>go('process')} title="AI 研发">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m8 6-6 6 6 6M16 6l6 6-6 6M13 4l-3 16"/></svg>研发
          </button>
          <button className={'fw-rbtn'+(lens==='progress'||lens==='atlas'?' on':'')} onClick={()=>go('progress')} title="路线 · 行业进展 / 图谱 / 本地进度">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M2 12h5M9 12h5M17 8l5 4-5 4V8z"/><circle cx="7" cy="12" r="2.5"/><circle cx="15" cy="12" r="2.5"/></svg>路线
          </button>
          <div className="fw-rdiv"/>
          {/* 账本入口已隐藏，需要时取消注释即可恢复 */}
          {/* <button className="fw-rbtn" onClick={()=>window.alert('账本视图（registry / 队列 / 证据等级）：沿用旧版工作台入口，本页不重复实现')} title="账本">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 4h13a3 3 0 0 1 3 3v13H7a3 3 0 0 1-3-3zM8 8h8M8 12h8M8 16h5"/></svg>账本
          </button> */}
          {/* 图谱不再设独立导航：已并入路线透镜中间 tab；?lens=atlas 作为 URL 兼容键仍可达 */}
        </nav>

        {/* ===== 主视口 ===== */}
        <main className="fw-main">
          <div className="fw-lensbar">
            <div className="fw-obj-title">
              {objId} <span className="chip">{obj.layer||''} · {obj.label}</span>
              <span className={'fw-ev-badge '+(obj.evidence==='mechanism_evidence'?'fw-ev-mech':'fw-ev-obs')}>● {obj.evidence}</span>
            </div>
            <div className="fw-lens-tools">
              {lens!=='home'&&(
                <button className={'fw-tbtn'+(about?' on':'')} onClick={()=>setAbout(v=>!v)} title="四透镜融合设计说明">ⓘ 融合主张</button>
              )}
              <button className="fw-tbtn" onClick={()=>window.alert('导出当前视图为 PNG / 分享链接（URL 即状态：?lens=&obj=）')}>分享</button>
            </div>
          </div>

          <div className="fw-viewport">
            {lens==='home'&&<HomeView onGo={go} ledger={atlasLedger} home={{sealed:rndQueue?rndQueue.sealed:DEMO_QUEUE.sealed,count:rndQueue?rndQueue.count:DEMO_QUEUE.count,queue:rndQueue?rndQueue.queue:DEMO_QUEUE.queue,tpls:distStats?String(distStats.templates.length):'—',results:distStats?String(distStats.results_total):'—'}}/>}
            {lens==='spatial'&&<LensSpatial on onGo={go}/>}
            {lens==='process'&&<LensProcess on onGo={go}/>}
            {lens==='progress'&&<LensProgress on onGo={go} initialAtlasSel={atlasSel}/>}
            {/* lens=atlas：URL 兼容别名 → 路线透镜·图谱 tab（导航已并入路线） */}
            {lens==='atlas'&&<LensProgress on onGo={go} initialView="atlas" initialAtlasTab={atlasTab} initialAtlasSel={atlasSel}/>}
            {about&&lens!=='home'&&(
              <div className="fw-about-pop">
                <span className="t">v5 融合主张 —— 四界面如何成为「一个界面」</span>
                参照：OpenAI Microscope（对象→多视图）、OpenScience（AI 工作台内嵌科学渲染）、Multi-Agent Visualizer（3D+实时研究）、AlphaFold DB（左 viewer 右 metadata）。
                <div className="pt"><b>① 对象路由即状态</b>——顶栏中央 <em>qwen3-4b / 6-l6-teal / {objId}</em> 是全局唯一地址；URL 即状态（<em>?lens=&amp;obj=</em>），⌘K 直达对象。</div>
                <div className="pt"><b>② 四透镜不换对象</b>——3D / AI 研发 / 路线图 / 数据浏览器是同一对象的<em>空间·过程·脉络·数据</em>四个视角；切透镜只换主视口，对象上下文（激活、证据等级、关联）恒在。</div>
                <div className="pt"><b>③ 协议是唯一稳定契约</b>——界面是协议的通用渲染器：测试模板（factors/eta2/fingerprint）、对象卡（object_card.v1）、队列（phase_queue schema）皆只认协议键；新模板 / 新对象注册后零界面改动自动出现。</div>
                <div className="pt"><b>④ 跨透镜动作闭环</b>——结果卡可「在 3D 查看 / 到分析技术运行」；底部事件流是四个透镜共用的<em>事件总线</em>，实验→结构→进展自动流转。</div>
                <div style={{textAlign:'right',marginTop:10}}>
                  <button className="fw-tbtn" onClick={()=>setAbout(false)}>知道了</button>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* ===== 右侧对象卡已隐藏（对象上下文仍经 URL ?obj= 与 ⌘K 全局生效；需要时取消注释即可恢复） =====
        <aside className={'fw-side'+(sideOpen?' open':'')}>
          <div className="fw-side-head">
            <button type="button" className="fw-side-toggle" onClick={()=>setSideOpen(v=>!v)}
                    title={sideOpen?'收起对象卡——对象 ID 切换仍走顶栏/⌘K/URL':'展开对象卡（激活示例 / 技术足迹 / 派实验）'}>
              <span className="fw-tr">{sideOpen?'▾':'▸'}</span>
              <span className="t">对象卡</span>
              <span className="fw-mono">{objId}</span>
              <span className={'fw-ev-badge '+(obj.evidence==='mechanism_evidence'?'fw-ev-mech':'fw-ev-obs')}>● {obj.evidence}</span>
            </button>
            {sideOpen&&<span className="sub">随透镜联动</span>}
          </div>
          {sideOpen&&(
            <div className="fw-side-body">
              <ObjectCard obj={obj} live={objIsLive} onDispatch={()=>go('process')}/>
            </div>
          )}
        </aside>
        */}
      </div>

      {/* ===== 底部状态条：全局事件总线 ===== */}
      <footer className="fw-statusbar">
        <span className="fw-st-chip"><span className="ok">●</span> qwen3-4b bf16 · GPU 21.3/24G</span>
        <span className="fw-st-chip">{rndQueue?`${rndQueue.sealed}/${rndQueue.count} SEALED`:'DEMO 状态'}</span>
        <div className="fw-ticker"><div className="fw-ticker-in">
          {ticker.map((t,i)=><span key={i}>{t}</span>)}
        </div></div>
      </footer>

      {/* ===== ⌘K ===== */}
      {cmdk&&(
        <div className="fw-mask" onMouseDown={e=>{if(e.target===e.currentTarget)setCmdk(false)}}>
          <div className="fw-cmdk-panel">
            <input ref={queryRef} placeholder="搜索：特征 / 任务 / 论文 / 假设 / Phase…"/>
            {cmdkGroups.map(g=>(
              <div key={g.gl} className="fw-cmdk-group">
                <div className="gl">{g.gl}</div>
                {g.items.map(it=>(
                  <div key={it.k} className="fw-cmdk-item" onClick={()=>{ if(it.obj) selectObj(it.obj); go(it.go); }}>
                    <span className="k">{it.k}</span>{it.t}<span className="d">{it.d}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
