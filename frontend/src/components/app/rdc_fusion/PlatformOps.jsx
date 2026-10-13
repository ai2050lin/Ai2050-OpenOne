/* PlatformOps · 平台运行面板（v3.4）：研发透镜右栏常驻的基础设施监控。
   模块（3）：平台概览（键值行）/ 节点与租约（3 列行式+活跃租约+调度队列快照）/ 模板与聚合（模板行，有结果的行下挂 AGG 桶摘要）。
   v3.4：原「路线·本地进度」tab 撤销后的归一承接——调度队列快照（GET /api/tasks 统计）与本机 agent
   终端指路（node_agent.py status，凭据不出终端）并入「节点与租约」卡；本机状态唯一家 = 本卡本机行。
   v3.3 去重：概览删「开放模板」「聚合版本」两行；节点表删 LIVE 恒空的死列（5→3 列）；
   AggLiveCard 桶摘要按 tm_id 嵌入模板行。
   数据源：GET /api/distributed/summary（30s 轮询）+ GET /api/tasks（调度快照）+ GET /api/agg/{tm_id}；
   离线回退 DEMO 并挂徽标。注册表 / 研究包 / 技术成熟度在图谱「模板与技术」；结果浏览在图谱「结果库」。 */
import { useEffect, useState } from 'react';
import { PLATFORM_STATS, NODES, TEMPLATES } from './distributedData.js';
import { fmtAgo } from './TimelineBits.jsx';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

/* AGG 桶摘要行（嵌入模板行下方）：只渲染协议键（eta2_by_factor / eta2_by_cond / topk_jaccard）；
   跨桶简单平均被禁止（research_os 13.1），此处只回显不做汇总计算（ui_decoupled_plan_v1 §3） */
function AggBuckets({a}){
  const buckets = Object.entries(a.buckets||{});
  if(!buckets.length) return <div className="dim2" style={{fontSize:9.5,padding:'1px 0 3px'}}>无桶数据</div>;
  return buckets.map(([bk,items])=>{
    const eta2 = items.map(it=>it.summary&&it.summary.eta2_by_factor).filter(Boolean)[0]||null;
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
  });
}

export default function PlatformOps({schedSlot,onOpenDist}){
  const [dist,setDist]=useState(null);   // 中心节点实时态：summary（拉不到则回退 demo）
  const [agg,setAgg]=useState(null);     // tm_id → GET /api/agg/{tm_id}（仅对有结果的模板拉取）
  const [schedQ,setSchedQ]=useState(null); // v3.4：调度队列快照 ← GET /api/tasks（原路线·本地进度归一）
  useEffect(()=>{
    let dead=false;
    const load=()=>{
      const ctl=new AbortController();
      const t=setTimeout(()=>ctl.abort(),4000);
      Promise.all([
        fetch(`${API_BASE}/api/distributed/summary`,{signal:ctl.signal}).then(r=>r.ok?r.json():null).catch(()=>null),
        fetch(`${API_BASE}/api/tasks`,{signal:ctl.signal}).then(r=>r.ok?r.json():null).catch(()=>null),
      ]).then(([s,q])=>{
        if(dead) return;
        if(s&&s.version) setDist(s);
        setSchedQ(q&&q.stats?q:null);
      }).finally(()=>clearTimeout(t));
    };
    load();
    const iv=setInterval(load,30000);
    return ()=>{dead=true;clearInterval(iv);};
  },[]);

  // AGG 实时桶：对每个有结果的模板拉 GET /api/agg/{tm_id}；离线（dist=null）不拉
  const aggTms=(dist?dist.templates:[]).filter(t=>t.results>0).map(t=>t.tm_id);
  useEffect(()=>{
    let dead=false;
    if(!aggTms.length){ setAgg({}); return undefined; }
    Promise.all(aggTms.map(id=>
      fetch(`${API_BASE}/api/agg/${encodeURIComponent(id)}`).then(r=>r.ok?r.json():null).catch(()=>null)
    )).then(rs=>{ if(!dead) setAgg(Object.fromEntries(aggTms.map((id,i)=>[id,rs[i]]))); });
    return ()=>{dead=true;};
  },[dist]); // eslint-disable-line react-hooks/exhaustive-deps

  // 平台概览（v3.3 去重：删「开放模板」「聚合版本」，明细卡已承担）
  const statKeep = dist ? ['在线节点','已上传结果','结果下载'] : ['在线节点','已上传 cells','结果下载'];
  const stats = (dist ? [
    {n:dist.nodes_online, l:'在线节点', s:'共 '+dist.nodes_total+' 注册'},
    {n:dist.templates.filter(t=>t.status==='open').length, l:'开放模板', s:'模板注册表 S2'},
    {n:dist.results_total, l:'已上传结果', s:'sha256 内容寻址'},
    {n:dist.downloads_total, l:'结果下载', s:'他人引用计数'},
    {n:dist.agg_version, l:'聚合版本', s:dist.version},
  ] : PLATFORM_STATS).filter(s=>statKeep.includes(s.l));

  // 节点行（v3.3：3 列=节点/硬件/状态·心跳；LIVE 无 per-node 模板与进度，死列撤销）
  const liveNodes = dist ? dist.nodes.map(n=>({
    id:n.node_id, gpu:n.gpu||'—', model:n.model||n.name||'—',
    cls: (Date.now()/1000-n.last_seen<120)?'run':'idle', up:fmtAgo(n.last_seen),
  })) : NODES;
  // 模板行（LIVE=summary.templates；DEMO=distributedData.js）
  const liveTpls = dist ? dist.templates.map(t=>({
    id:t.tm_id, dim:t.dim, name:t.name, me:false,
    agg: t.results>0 ? 'collecting' : 'todo',
    nodes:t.claims_active, cells:t.results+' 结果', results:t.results,
  })) : TEMPLATES;
  const tplTip = t => t.name+' · '+(t.nodes||0)+' 节点执行中 · '+t.cells;
  const tplAggTxt = t => t.agg==='done'?'已聚合':(t.agg==='collecting'?'收集中':'待分发');

  return (
    <section className="fw-data-embed">
      <div className="fw-rg-inner">
        <div className="fw-pl-stack">
          {/* ===== ① 平台概览（LIVE=中心节点 / DEMO=distributedData；键值列表行） ===== */}
          <div className="fw-pl-card">
            <h6>平台概览
              <span className={'fw-src-chip '+(dist?'live':'demo')} title={dist?dist.version:'接 GET /api/distributed/summary 后显示实时数据'}>
                {dist?'● LIVE 中心节点':'○ DEMO（中心节点离线）'}
              </span>
            </h6>
            {stats.map(s=>(
              <div key={s.l} className="fw-pl-kv" title={s.s}>
                <span className="k">{s.l}</span>
                <span className="s fw-mono">{s.s}</span>
                <span className="v fw-mono">{s.n}</span>
              </div>
            ))}
          </div>

          {/* ===== ② 节点与租约（LIVE=summary.nodes 最近心跳；DEMO=distributedData.js） ===== */}
          <div className="fw-pl-card">
            <h6>节点与租约 <span>{dist?'心跳 '+liveNodes.length+' 台 · 租约 /api/tasks':'8/63 展示 · 接 /api/distributed/summary'}</span></h6>
            <div className="fw-pl-row head"><span>节点</span><span>硬件</span><span className="r">状态 · 心跳</span></div>
            {liveNodes.map(n=>(
              <div key={n.id} className="fw-pl-row">
                <span className={'fw-mono'+(n.me?' me':'')}>{n.id}{n.me?'（本机）':''}</span>
                <span className="dim2">{n.gpu} · {n.model}</span>
                <span className={'r fw-mono fw-st-'+n.cls}>{n.cls==='agg'?'聚合中':(n.cls==='run'?'在线':'空闲')} · {n.up}</span>
              </div>
            ))}
            {schedQ && (
              <div className="fw-sched-mini fw-mono" title="调度队列只读快照（GET /api/tasks）：原「路线·本地进度」归一至此">
                调度队列：活跃 {schedQ.stats.claimed||0} · 完成 {schedQ.stats.done||0} · 失败 {schedQ.stats.failed||0} · 过期 {schedQ.stats.expired||0} · 单租约 {schedQ.lease_hours}h
              </div>
            )}
            {!schedQ && <div className="fw-sched-mini dim2">调度队列快照不可达（GET /api/tasks）。</div>}
            <div className="fw-sched-mini dim2" title="凭据不出终端：node_credentials.json 仅存本机，浏览器不读取">
              本机 agent 详细状态 → 终端 <code>node_agent.py status</code> · 全队列 → <code>node_agent.py queue</code>
            </div>
            {schedSlot&&(
              <div style={{marginTop:8}}>
                <div className="fw-m3-more" style={{fontWeight:600,color:'var(--fw-text-2,#475569)'}}>活跃租约</div>
                {schedSlot}
              </div>
            )}
          </div>

          {/* ===== ③ 模板与聚合（tm_id 单一出处；有结果的行下挂 AGG 桶摘要；行可点击 → 「运行」展开结果） ===== */}
          <div className="fw-pl-card">
            <h6>模板与聚合 <span>每模板=受控语料+固定口径 · AGG 桶只回显禁平均</span></h6>
            <div className="fw-tmx-list">
              {liveTpls.map(t=>{
                const clickable=onOpenDist&&(t.results||0)>0;
                const a=agg?agg[t.id]:null;
                return (
                  <div key={t.id} className="fw-tmx-block">
                    <button type="button" className={'fw-tmx-item'+(t.me?' me':'')+(clickable?' link':'')}
                            title={clickable?tplTip(t)+' · 点击在「运行」展开结果':tplTip(t)}
                            onClick={()=>{ if(clickable) onOpenDist({tm_id:t.id,name:t.name,results:t.results}); }}>
                      <b className="fw-mono">{t.id}</b>
                      <span className="fw-tmx-dim">{t.dim}</span>
                      <span className={'fw-tmx-agg '+t.agg}>{tplAggTxt(t)}</span>
                      <span className="fw-tmx-n">{t.nodes||0}节点 · {t.cells}</span>
                    </button>
                    {a&&<div className="fw-agg-sub"><AggBuckets a={a}/></div>}
                  </div>
                );
              })}
            </div>
            {dist&&agg&&!Object.keys(agg).length&&<div className="fw-pc-empty">尚无结果——节点上传后此处自动出现。</div>}
            {dist&&agg===null&&<div className="fw-pc-empty">拉取聚合桶…</div>}
          </div>

          {/* ===== P2 迁入插槽：调度队列 / 分布式模板结果（状态在 LensProcess，渲染归此） ===== */}
        </div>
      </div>
    </section>
  );
}
