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
import LensData from './rdc_fusion/LensData.jsx';
import { EVENTS, TICKER, CMDK_GROUPS, DEMO_OBJECT, DEMO_QUEUE } from './rdc_fusion/demoData.js';
import { TECH_CATEGORIES, DEMO_COVERAGE, LANG_TEMPLATES, ANALYSES, MISSION } from './rdc_fusion/distributedData.js';
import './rdc_fusion/rdc_fusion.css';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');
const DEFAULT_OBJ = 'F#3734';

const LENS_NAME={spatial:'空间透镜',process:'过程透镜',progress:'脉络透镜',data:'分析技术',home:'总览'};

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

/* M7-P1 技术四分类状态标签（渲染词表，非实验内容） */
const CAT_STATUS={measured:'已测',device_built:'装置已建',planned:'装置待建'};

function HomeView({onGo,home}){
  const st=home||{sealed:DEMO_QUEUE.sealed,count:DEMO_QUEUE.count,tpls:'—',results:'—',objects:1,queue:DEMO_QUEUE.queue};
  const q=st.queue||DEMO_QUEUE.queue;
  const nx=q.find(x=>x.status==='pending');
  const pend=q.filter(x=>x.status==='pending')[1];
  const last=[...q].reverse().find(x=>x.status==='sealed');
  /* M7-P1 技术版图：覆盖度矩阵（P2 真源 = /api/coverage；当前 DEMO_COVERAGE） */
  const cov=home&&home.coverage?home.coverage:DEMO_COVERAGE;
  const covOf=(tpl,cat)=>{const c=cov.find(x=>x.tpl===tpl&&x.cat===cat);return c?c.status:'empty';};
  const coveredCats=TECH_CATEGORIES.filter(c=>cov.some(x=>x.cat===c.id&&x.status==='done')).length;
  const nTech=cat=>ANALYSES.filter(a=>a.category===cat.id).length;
  const cellsOf=(cat,onlyDone)=>cov.filter(x=>x.cat===cat.id&&(!onlyDone||x.status==='done')).length;
  return (
    <div className="fw-view fw-home on">
      <div className="fw-hero">
        <div className="fw-mission">
          <h2><b className="fw-brand">{MISSION.brand}</b>{MISSION.title}</h2>
          <div className="fw-mission-chain">
            {MISSION.chain.map((s,i)=>(
              <span key={s.k} className="fw-mission-step">
                <i>{s.k}</i>
                <span>{s.v}</span>
              </span>
            ))}
          </div>
        </div>
        <div className="fw-hero-num">
          <div className="stat"><div className="n">{st.count?`${st.sealed}/${st.count}`:'—'}</div><div className="l">队列 sealed</div></div>
          <div className="stat"><div className="n">{st.tpls}</div><div className="l">测试模板</div></div>
          <div className="stat"><div className="n">{st.results}</div><div className="l">结果条目</div></div>
          <div className="stat"><div className="n">{coveredCats}/{TECH_CATEGORIES.length}</div><div className="l">技术覆盖</div></div>
        </div>
      </div>
      <div className="fw-techmap">
        <div className="fw-techmap-head">
          <b>技术版图 · 四类分析技术</b>
          <span>结果空间 = 对象 × 语言模板 × 分析技术 · 覆盖度矩阵：实心=已有结果 · 半透明=部分/装置已建 · 虚线=空格（点空格去研发透镜领缺口任务）</span>
        </div>
        <div className="fw-techmap-cards">
          {TECH_CATEGORIES.map(c=>(
            <button key={c.id} type="button" className="fw-tc-card" style={{'--cat':c.color}} title={c.note} onClick={()=>onGo('data')}>
              <div className="hd"><i style={{background:c.color}}/><b>{c.name}</b><span className="fw-tc-st">{CAT_STATUS[c.status]||c.status}</span></div>
              <div className="ms">{c.methods.join(' · ')}</div>
              <div className="ft"><span>{nTech(c)} 项技术</span><span>覆盖 {cellsOf(c,true)}/{cellsOf(c)} 格</span><span className="dim2">{c.cost}</span></div>
            </button>
          ))}
        </div>
        <div className="fw-cov-grid">
          <div className="fw-cov-row head"><span>模板 \ 技术</span>{TECH_CATEGORIES.map(c=><span key={c.id} style={{color:c.color}}>{c.short||c.name}</span>)}</div>
          {LANG_TEMPLATES.map(t=>(
            <div key={t.id} className="fw-cov-row">
              <span className="tpl">{t.name}</span>
              {TECH_CATEGORIES.map(c=>{
                const s=covOf(t.id,c.id);
                return <button key={c.id} type="button" className={'fw-cov-cell '+s}
                               title={t.name+' × '+c.name+' · 状态 '+s} onClick={()=>onGo('process')}/>;
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="fw-entry-col">
      <button className="fw-entry" onClick={()=>onGo('spatial')}>
        <div className="ic" style={{background:'#eff6ff',color:'#0284c7'}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><ellipse cx="12" cy="12" rx="10" ry="4.2"/></svg>
        </div>
        <h3>空间透镜 · 3D</h3>
        <p>1,024 特征点云（L6），簇=模式族；高亮特征与 top-k 邻居；E_ar 方向叠加；14B/9B 跨模型对比。</p>
        <span className="tag">接 collect.npz · three.js→WebGPU</span>
      </button>
      <button className="fw-entry" onClick={()=>onGo('process')}>
        <div className="ic" style={{background:'#f5f3ff',color:'#6d28d9'}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m8 6-6 6 6 6M16 6l6 6-6 6"/></svg>
        </div>
        <h3>过程透镜 · AI 研发</h3>
        <p>预注册 → 冻结 → 执行 → 独立复核闭环；队列 / 工作区 / 工件查看器同屏（协议驱动，LIVE/DEMO 双模）。</p>
        <span className="tag">接 :5001 /api/ai-rnd · 队列←phase_queue</span>
      </button>
      <button className="fw-entry" onClick={()=>onGo('progress')}>
        <div className="ic" style={{background:'#fffbeb',color:'#b45309'}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 17l6-6 4 4 7-8"/><path d="M14 7h6v6"/></svg>
        </div>
        <h3>脉络透镜 · 行业进展</h3>
        <p>本项目证据链时间线 + 同行泳道（Anthropic / OpenAI / 开源生态）；节点挂论文与 sealed 产物，可对比登记。</p>
        <span className="tag">接 industry.json · MEMO 索引</span>
      </button>
      <button className="fw-entry" onClick={()=>onGo('data')}>
        <div className="ic" style={{background:'#ecfdf5',color:'#059669'}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>
        </div>
        <h3>分析技术 · 结果浏览器</h3>
        <p>分析技术注册表驱动：选技术 → 契约匹配结果 → 展开自动运行（RSA/RDM、η² 分解、CKA）；亦可按结果浏览。新模板注册即自动出现。</p>
        <span className="tag">接 /api/templates · /api/results（summary_digest）· 技术注册表</span>
      </button>
      </div>
      <div className="fw-side-col">
        <div className="fw-mini">
          <h4>进行中 <span style={{fontSize:10,color:'var(--fw-blue)',cursor:'pointer'}} onClick={()=>onGo('process')}>全部 →</span></h4>
          {nx&&<div className="fw-task"><span className="st fw-st-run"/><div className="t"><b>{nx.id} · {nx.title}</b><span>next · {nx.deliverable||''}</span></div></div>}
          {pend&&<div className="fw-task"><span className="st fw-st-que"/><div className="t"><b>{pend.id} · {pend.title}</b><span>queued · {pend.block||''}</span></div></div>}
          {last&&<div className="fw-task"><span className="st fw-st-done"/><div className="t"><b>{last.id} · {last.title}</b><span>sealed {last.res_sha8||''}</span></div></div>}
        </div>
        <div className="fw-mini">
          <h4>事件流</h4>
          {EVENTS.map((e,i)=><div key={i} className="fw-evt"><span className="tm">{e.tm}</span><span>{e.txt}</span></div>)}
        </div>
      </div>
    </div>
  );
}

export default function RdcFusionWorkspace(){
  const params=new URLSearchParams(typeof window!=='undefined'?window.location.search:'');
  const [lens,setLens]=useState(()=>params.get('lens')||'spatial');
  const [objId,setObjId]=useState(()=>params.get('obj')||DEFAULT_OBJ);
  const [objLive,setObjLive]=useState(null);     // /api/object/{fid} → {object,related_results}
  const [objects,setObjects]=useState(null);     // /api/objects
  const [rndQueue,setRndQueue]=useState(null);   // /api/ai-rnd/queue
  const [distStats,setDistStats]=useState(null); // /api/distributed/summary
  const [cmdk,setCmdk]=useState(false);
  const [about,setAbout]=useState(false);
  // const [sideOpen,setSideOpen]=useState(false);  // 对象卡已隐藏，恢复时连同右侧 aside 一起取消注释
  const queryRef=useRef(null);

  const syncURL=(l,o)=>{
    try{ window.history.replaceState(null,'',`?lens=${l}&obj=${encodeURIComponent(o)}`); }catch{ /* 隐私模式等 */ }
  };
  const go=l=>{ setLens(l); setAbout(false); setCmdk(false); syncURL(l,objId); };
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
    })();
    return ()=>{ dead=true; };
  },[]);

  useEffect(()=>{
    const onKey=e=>{
      if((e.metaKey||e.ctrlKey)&&String(e.key).toLowerCase()==='k'){e.preventDefault();setCmdk(v=>!v);}
      if(e.key==='Escape'){setCmdk(false);setAbout(false);}
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
        <button className="fw-select" title="演示选择器（特征源注册后由此驱动）"><b>6-l6-teal</b><span style={{color:'var(--fw-text-3)',fontSize:10}}>RDC 特征源</span><span className="car">▾</span></button>
        <div className="fw-crumb">
          <button className="obj" onClick={()=>selectObj(objId)} title={obj.label}>{objId}</button>
          <span className="lens-chip">{LENS_NAME[lens]}</span>
        </div>
        <button className="fw-cmdk" onClick={()=>setCmdk(true)}>搜索对象、任务、论文… <kbd>⌘K</kbd></button>
        <button className="fw-bell" title="任务通知">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.7 21a2 2 0 0 1-3.4 0"/></svg>
          <span className="badge">2</span>
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
          <button className={'fw-rbtn'+(lens==='progress'?' on':'')} onClick={()=>go('progress')} title="行业路线图">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M2 12h5M9 12h5M17 8l5 4-5 4V8z"/><circle cx="7" cy="12" r="2.5"/><circle cx="15" cy="12" r="2.5"/></svg>路线
          </button>
          <div className="fw-rdiv"/>
          {/* 账本入口已隐藏，需要时取消注释即可恢复 */}
          {/* <button className="fw-rbtn" onClick={()=>window.alert('账本视图（registry / 队列 / 证据等级）：沿用旧版工作台入口，本页不重复实现')} title="账本">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 4h13a3 3 0 0 1 3 3v13H7a3 3 0 0 1-3-3zM8 8h8M8 12h8M8 16h5"/></svg>账本
          </button> */}
          <button className={'fw-rbtn'+(lens==='data'?' on':'')} onClick={()=>go('data')} title="分析技术 · 技术注册表 + 结果浏览器（协议驱动）">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>技术
          </button>
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
            {lens==='home'&&<HomeView onGo={go} home={{sealed:rndQueue?rndQueue.sealed:DEMO_QUEUE.sealed,count:rndQueue?rndQueue.count:DEMO_QUEUE.count,queue:rndQueue?rndQueue.queue:DEMO_QUEUE.queue,tpls:distStats?String(distStats.templates.length):'—',results:distStats?String(distStats.results_total):'—'}}/>}
            {lens==='spatial'&&<LensSpatial on onGo={go}/>}
            {lens==='process'&&<LensProcess on onGo={go}/>}
            {lens==='progress'&&<LensProgress on onGo={go}/>}
            {lens==='data'&&<LensData on/>}
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
