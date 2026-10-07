/* ============================================================
   RdcFusionWorkspace —— v5 三透镜融合驾驶舱（路由 /rdc-fusion）
   融合主张：3D 可视化 / AI 研发 / 行业进展 = 同一研究对象的
   三个透镜（空间·过程·脉络），共享对象路由 + 对象卡 + 事件总线。
   视觉 token：slate 体系 + emerald 激活梯度；demo 数据为真实研究数据，
   接入点已注释（collect.npz / research OS registry / industry.json）。
   ============================================================ */
import { useEffect, useRef, useState } from 'react';
import LensSpatial from './rdc_fusion/LensSpatial.jsx';
import LensProcess from './rdc_fusion/LensProcess.jsx';
import LensProgress from './rdc_fusion/LensProgress.jsx';
import { EVENTS, TICKER, ACT_LINES, CMDK_GROUPS } from './rdc_fusion/demoData.js';
import './rdc_fusion/rdc_fusion.css';

const LENS_NAME={spatial:'空间透镜',process:'过程透镜',progress:'脉络透镜',home:'总览'};

function Tok({t,dfa}){
  const [txt,v]=t;
  const style=v>0?{background:'rgba(52,211,153,'+v+')',color:v>0.6?'#022c22':'#064e3b'}:undefined;
  return <span className={'fw-tok'+(dfa?' dfa':'')} style={style}>{txt}</span>;
}

function HomeView({onGo}){
  return (
    <div className="fw-view fw-home on">
      <div className="fw-hero">
        <div>
          <h2>同一对象，三种透镜</h2>
          <p>空间（3D 点云：特征在哪里、邻居是谁）· 过程（AI 研发工作台：对它做过什么实验）· 脉络（行业路线图：它支撑哪个大问题）。三界面共享顶栏的<b>对象路由</b>与右侧<b>对象卡</b>，切透镜不换对象。</p>
        </div>
        <div className="fw-hero-num">
          <div className="stat"><div className="n">304</div><div className="l">Ledger 条目</div></div>
          <div className="stat"><div className="n">39</div><div className="l">MEMO Phase</div></div>
          <div className="stat"><div className="n">6/30</div><div className="l">队列 sealed</div></div>
        </div>
      </div>
      <div className="fw-entry-col">
      <button className="fw-entry" onClick={()=>onGo('spatial')}>
        <div className="ic" style={{background:'#eff6ff',color:'#0284c7'}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><ellipse cx="12" cy="12" rx="10" ry="4.2"/></svg>
        </div>
        <h3>空间透镜 · 3D</h3>
        <p>1,024 特征点云（L6），簇=模式族；高亮 F#3734 与 top-k 邻居；E_ar 方向叠加；14B/9B 跨模型对比。</p>
        <span className="tag">接 collect.npz · three.js→WebGPU</span>
      </button>
      <button className="fw-entry" onClick={()=>onGo('process')}>
        <div className="ic" style={{background:'#f5f3ff',color:'#6d28d9'}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m8 6-6 6 6 6M16 6l6 6-6 6"/></svg>
        </div>
        <h3>过程透镜 · AI 研发</h3>
        <p>预注册 → 冻结 → 执行 → 独立复核闭环；队列、脚本、终端、结果卡同屏；结果内嵌可视化（OpenScience / prism_ai 范式）。</p>
        <span className="tag">接 research OS registry · q06 RUNNING</span>
      </button>
      <button className="fw-entry" onClick={()=>onGo('progress')}>
        <div className="ic" style={{background:'#fffbeb',color:'#b45309'}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 17l6-6 4 4 7-8"/><path d="M14 7h6v6"/></svg>
        </div>
        <h3>脉络透镜 · 行业进展</h3>
        <p>本项目证据链时间线 + 同行泳道（Anthropic / OpenAI / 开源生态）；节点挂论文与 sealed 产物，可对比登记。</p>
        <span className="tag">接 industry.json · MEMO 索引</span>
      </button>
      </div>
      <div className="fw-side-col">
        <div className="fw-mini">
          <h4>进行中 <span style={{fontSize:10,color:'var(--fw-blue)',cursor:'pointer'}}>全部 →</span></h4>
          <div className="fw-task"><span className="st fw-st-run"/><div className="t"><b>Q06 · C_steer 基座测量</b><span>step 2/5 · 承重轴枚举</span></div></div>
          <div className="fw-task"><span className="st fw-st-que"/><div className="t"><b>N 线 P3–P7 补 Ledger</b><span>queued · 跨线账本确认</span></div></div>
          <div className="fw-task"><span className="st fw-st-done"/><div className="t"><b>Q05 · E_ar(k) 正式测量</b><span>sealed 1acb1e78 · 复核 47/0</span></div></div>
        </div>
        <div className="fw-mini">
          <h4>事件流 <span style={{fontSize:10,color:'var(--fw-blue)',cursor:'pointer'}}>账本 →</span></h4>
          {EVENTS.map((e,i)=><div key={i} className="fw-evt"><span className="tm">{e.tm}</span><span>{e.txt}</span></div>)}
        </div>
      </div>
    </div>
  );
}

export default function RdcFusionWorkspace(){
  const [lens,setLens]=useState('spatial');
  const [cmdk,setCmdk]=useState(false);
  const [about,setAbout]=useState(false);
  const queryRef=useRef(null);

  useEffect(()=>{
    const onKey=e=>{
      if((e.metaKey||e.ctrlKey)&&String(e.key).toLowerCase()==='k'){e.preventDefault();setCmdk(v=>!v);}
      if(e.key==='Escape'){setCmdk(false);setAbout(false);}
    };
    window.addEventListener('keydown',onKey);
    return ()=>window.removeEventListener('keydown',onKey);
  },[]);
  useEffect(()=>{ if(cmdk&&queryRef.current) queryRef.current.focus(); },[cmdk]);

  const go=l=>{setLens(l);setAbout(false);setCmdk(false);};
  const tickerText=TICKER.concat(TICKER);

  return (
    <div className="fw-root">
      {/* ===== 顶栏：全局上下文 ===== */}
      <header className="fw-topbar">
        <div className="fw-logo"><div className="fw-logo-mark">A2</div>Ai2050 Atlas <small>v5 · RDC</small></div>
        <button className="fw-select"><span className="dot" style={{background:'#6366f1'}}/><b>qwen3-4b</b><span className="car">▾</span></button>
        <button className="fw-select"><b>6-l6-teal</b><span style={{color:'var(--fw-text-3)',fontSize:10}}>RDC 特征源</span><span className="car">▾</span></button>
        <div className="fw-crumb">
          <span className="obj">F#3734</span>
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
          <button className="fw-rbtn" onClick={()=>window.alert('账本视图（registry / 队列 / 证据等级）：沿用旧版工作台入口，本页不重复实现')} title="账本">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 4h13a3 3 0 0 1 3 3v13H7a3 3 0 0 1-3-3zM8 8h8M8 12h8M8 16h5"/></svg>账本
          </button>
          <button className="fw-rbtn" onClick={()=>window.alert('数据源：collect.npz / InterPLM :8501 / atlas_ledger.json')} title="数据源">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>数据
          </button>
        </nav>

        {/* ===== 主视口 ===== */}
        <main className="fw-main">
          <div className="fw-lensbar">
            <div className="fw-obj-title">
              F#3734 <span className="chip">L6 · is-a 上位：水果族</span>
              <span className="fw-ev-badge fw-ev-mech">● mechanism_evidence</span>
            </div>
            <div className="fw-lens-tools">
              {lens!=='home'&&(
                <button className={'fw-tbtn'+(about?' on':'')} onClick={()=>setAbout(v=>!v)} title="三界面融合设计说明">ⓘ 融合主张</button>
              )}
              <button className="fw-tbtn" onClick={()=>window.alert('导出当前视图为 PNG / 分享链接（URL 即状态）')}>分享</button>
            </div>
          </div>

          <div className="fw-viewport">
            {lens==='home'&&<HomeView onGo={go}/>}
            {lens==='spatial'&&<LensSpatial on/>}
            {lens==='process'&&<LensProcess on onGo={go}/>}
            {lens==='progress'&&<LensProgress on onGo={go}/>}
            {about&&lens!=='home'&&(
              <div className="fw-about-pop">
                <span className="t">v5 融合主张 —— 三界面如何成为「一个界面」</span>
                参照：OpenAI Microscope（对象→多视图）、OpenScience（AI 工作台内嵌科学渲染）、Multi-Agent Visualizer（3D+实时研究）、AlphaFold DB（左 viewer 右 metadata）。
                <div className="pt"><b>① 对象路由即状态</b>——顶栏中央 <em>qwen3-4b / 6-l6-teal / F#3734</em> 是全局唯一地址；URL 即状态（对象路由范式），⌘K 直达对象。</div>
                <div className="pt"><b>② 三透镜不换对象</b>——3D / AI 研发 / 路线图是同一对象的<em>空间·过程·脉络</em>三个视角；切透镜只换主视口，对象上下文（激活、证据等级、关联）恒在。</div>
                <div className="pt"><b>③ 跨透镜动作闭环</b>——结果卡可「在 3D 查看 / 登记路线图」；对象卡可「派 AI 实验」；底部事件流是三个透镜共用的<em>事件总线</em>，实验→结构→进展自动流转。</div>
                <div style={{textAlign:'right',marginTop:10}}>
                  <button className="fw-tbtn" onClick={()=>setAbout(false)}>知道了</button>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* ===== 右侧对象卡（特征卡范式） ===== */}
        <aside className="fw-side">
          <div className="fw-side-head">
            <span className="t">对象卡 · OBJECT CARD</span>
            <span className="sub">随透镜联动</span>
          </div>
          <div className="fw-side-body">
            <div className="fw-f-name">is-a 上位关系：水果族</div>
            <div className="fw-f-meta">
              <div className="kv">特征 ID<b>F#3734</b></div>
              <div className="kv">层 / 位置<b>L6 · write 端</b></div>
              <div className="kv">E_read<b>0.331615</b></div>
              <div className="kv">share_max<b>3.0% 内</b></div>
            </div>

            <div className="fw-act-box">
              <div className="fw-act-head"><span>ACTIVATIONS · 最大激活示例</span><span>max 4.21</span></div>
              <div className="fw-act-text">
                {ACT_LINES.map((line,i)=>(
                  <div key={i}>
                    {line.toks.map((t,j)=><Tok key={j} t={t} dfa={line.dfa}/>)}
                    {line.dfa&&<span className="fw-act-note"> ← 一词多义对照（弱激活）</span>}
                    <br/>
                  </div>
                ))}
              </div>
              <button className="fw-act-more" onClick={()=>window.alert('展开全部 81 条激活示例（接入 collect.npz 后开放）')}>展开全部示例 ▾</button>
            </div>

            <div className="fw-lens-links">
              <button className="fw-lens-link" onClick={()=>go('spatial')}>
                <svg viewBox="0 0 24 24" fill="none" stroke="#0284c7" strokeWidth="2"><circle cx="12" cy="12" r="3"/><ellipse cx="12" cy="12" rx="10" ry="4.2"/></svg>
                <b>空间</b>水果族簇 · top-5 邻居在 0.31–0.44<span className="arr">→</span>
              </button>
              <button className="fw-lens-link" onClick={()=>go('process')}>
                <svg viewBox="0 0 24 24" fill="none" stroke="#6d28d9" strokeWidth="2"><path d="m8 6-6 6 6 6M16 6l6 6-6 6"/></svg>
                <b>过程</b>Q05 四臂消融已覆盖 · Q06 steering 计划中<span className="arr">→</span>
              </button>
              <button className="fw-lens-link" onClick={()=>go('progress')}>
                <svg viewBox="0 0 24 24" fill="none" stroke="#b45309" strokeWidth="2"><path d="M3 17l6-6 4 4 7-8"/><path d="M14 7h6v6"/></svg>
                <b>脉络</b>支撑 Q05/Q06 节点 · 对标 Attribution Graphs<span className="arr">→</span>
              </button>
            </div>
          </div>
          <div className="fw-side-actions">
            <button className="fw-tbtn primary" onClick={()=>window.alert('派发 AI 实验：预注册 → 冻结 → 执行（写入队列）')}>＋ 派 AI 实验</button>
          </div>
        </aside>
      </div>

      {/* ===== 底部状态条：全局事件总线 ===== */}
      <footer className="fw-statusbar">
        <span className="fw-st-chip"><span className="ok">●</span> qwen3-4b bf16 · GPU 21.3/24G</span>
        <span className="fw-st-chip">Q06 RUNNING · step 2/5</span>
        <div className="fw-ticker"><div className="fw-ticker-in">
          {tickerText.map((t,i)=><span key={i}>{t}</span>)}
        </div></div>
      </footer>

      {/* ===== ⌘K ===== */}
      {cmdk&&(
        <div className="fw-mask" onMouseDown={e=>{if(e.target===e.currentTarget)setCmdk(false)}}>
          <div className="fw-cmdk-panel">
            <input ref={queryRef} placeholder="搜索：特征 / 任务 / 论文 / 假设 / Phase…"/>
            {CMDK_GROUPS.map(g=>(
              <div key={g.gl} className="fw-cmdk-group">
                <div className="gl">{g.gl}</div>
                {g.items.map(it=>(
                  <div key={it.k} className="fw-cmdk-item" onClick={()=>go(it.go)}>
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
