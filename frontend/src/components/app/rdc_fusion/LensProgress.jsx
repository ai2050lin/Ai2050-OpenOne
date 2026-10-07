/* 脉络透镜 v3：顶层三 tab —— 业界新闻 / 平台进度 / 当前机器进度
   分布式研发平台叙事：大量机器各领一个模板测试 → 上传服务器 → 他人下载
   → 综合全部模板结果分析 LLM 语言编码机制（demo 数据见 distributedData.js）。
   三个 tab 与内容对应：业界新闻=业界动态+同行对比卡；平台进度=分布式统计+
   节点+模板矩阵+聚合管线+平台发布时间线；当前机器进度=本机节点+RDC 主线。
   每带节点点击出详情；同行差异以「对比卡」登记入账本（四级证据体系）。 */
import { useState } from 'react';
import { LOCAL_NODE, PLATFORM_STATS, NODES, TEMPLATES, AGG_STEPS, AGG_DIMS, NEWS } from './distributedData.js';

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

/* 业界新闻 · 同行对比卡（demo 接入点 ← industry.json；每卡必带「对比」注脚登记视角差） */
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

export default function LensProgress({on,onGo}){
  const [view,setView]=useState('machine');
  return (
    <section className={'fw-view fw-progress'+(on?' on':'')}>
      <div className="fw-rg-inner">
        {/* ===== 顶层三 tab：业界新闻 / 平台进度 / 当前机器进度 ===== */}
        <div className="fw-pt-tabs" role="tablist">
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='news'?' on':'')} onClick={()=>setView('news')}>业界新闻<small>机制可解释性动态</small></button>
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='platform'?' on':'')} onClick={()=>setView('platform')}>平台进度<small>47 节点 · 12 模板</small></button>
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
            <b>路线图逻辑</b>：三个 tab 共享同一套证据口径——本机进度挂真实实验（节点必带 sealed 产物），平台进度对应可运行的版本与聚合产物（AGG-v3），业界以新闻流与对比卡登记视角差（has_data / observed / generalization_checked / mechanism_evidence 四级体系）。
          </div>
          <div style={{textAlign:'center',padding:'4px 0 10px'}}>
            <button className="fw-tbtn" onClick={()=>onGo('process')}>回到当前任务 →</button>
          </div>
        </>)}

        {/* ===== 平台进度：分布式统计 + 节点 + 模板矩阵 + 聚合分析 + 发布时间线 ===== */}
        {view==='platform'&&(<>
          <div className="fw-pl-stats">
            {PLATFORM_STATS.map(s=>(
              <div key={s.l} className="fw-pl-stat"><div className="n">{s.n}</div><div className="l">{s.l}</div><div className="s">{s.s}</div></div>
            ))}
          </div>
          <div className="fw-pl-grid">
            {/* 节点列表 */}
            <div className="fw-pl-card">
              <h6>在线节点 <span>8/47 展示 · 完整列表接 /api/distributed/summary</span></h6>
              <div className="fw-pl-row head"><span>节点</span><span>硬件</span><span>当前模板</span><span>进度</span><span className="r">上传</span></div>
              {NODES.map(n=>(
                <div key={n.id} className="fw-pl-row">
                  <span className={'fw-mono'+(n.me?' me':'')}>{n.id}{n.me?'（本机）':''}</span>
                  <span className="dim2">{n.gpu} · {n.model}</span>
                  <span>{n.tm}</span>
                  <span><span className="fw-pl-prog"><i style={{width:(n.prog*100).toFixed(0)+'%'}}/></span></span>
                  <span className={'r fw-mono fw-st-'+n.cls}>{n.cls==='agg'?'聚合中':(n.cls==='run'?'运行中':'空闲')} · {n.up}</span>
                </div>
              ))}
            </div>
            {/* 模板矩阵 + 聚合管线 */}
            <div style={{display:'flex',flexDirection:'column',gap:10,minWidth:0}}>
              <div className="fw-pl-card">
                <h6>模板测试矩阵 <span>每模板=受控语料+固定口径，可分发任意机器</span></h6>
                <div className="fw-tmx">
                  {TEMPLATES.map(t=>(
                    <div key={t.id} className={'fw-tmx-cell'+(t.me?' me':'')} title={t.name+' · '+t.nodes+' 节点 · '+t.cells+' cells'}>
                      <b>{t.id}</b>
                      <span className="fw-tmx-dim">{t.dim}</span>
                      <span className={'fw-tmx-agg '+t.agg}>{t.agg==='done'?'已聚合':(t.agg==='collecting'?'收集中':'待分发')}</span>
                      <span className="fw-tmx-n">{t.nodes}节点 · {t.cells}</span>
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

          <Lane color="#059669" label="平台发布" small="可视化客户端与服务端的版本时间线">
            <Timeline items={RELEASES} tag="release registry"/>
          </Lane>
        </>)}

        {/* ===== 业界新闻：动态流 + 同行对比卡 ===== */}
        {view==='news'&&(<>
          <div className="fw-news-head">业界动态 · 机制可解释性 <span>demo 数据，接入 RSS / arXiv / 官方博客后自动更新</span></div>
          <div className="fw-news-list" style={{marginBottom:14}}>
            {NEWS.map(nw=>(
              <div key={nw.title} className="fw-news-card">
                <span className="fw-news-date">{nw.date}</span>
                <div className="fw-news-main">
                  <h5>{nw.title}</h5>
                  <p>{nw.p}</p>
                </div>
                <div className="fw-news-meta">
                  <span className="fw-news-tag">{nw.tag}</span>
                  <span className="fw-news-src">{nw.src}</span>
                </div>
              </div>
            ))}
          </div>

          <Lane color="#6366f1" label="业界主流 · 对比卡" small="点击对比卡 → 与本项目证据链的视角差">
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
