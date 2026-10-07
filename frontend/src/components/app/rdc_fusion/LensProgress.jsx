/* 脉络透镜：行业路线图（本项目证据链时间线 + 同行泳道） */
const MAINLINE=[
  {lb:'P4–P7',tt:'主轴三段\n权重绑定定 is-a',st:'done'},
  {lb:'P8–P11',tt:'写入端分布式\n栈=软门',st:'done'},
  {lb:'P17–21',tt:'w+com_V≈26\n跨精度复现',st:'done'},
  {lb:'P35',tt:'A 闸门 seal\nQ08=甲',st:'done'},
  {lb:'Q03',tt:'E_read 基线\n0.332 · 5% 门 0/3',st:'done'},
  {lb:'Q04',tt:'E_ar 装置\n4/4 PASS',st:'done'},
  {lb:'Q05',tt:'E_ar 测量\n形状 flat/sat',st:'done'},
  {lb:'Q06',tt:'C_steer 基座\n承重轴+端口替换',st:'now'},
  {lb:'远期',tt:'权重级证明\nN2h1-α-1',st:'todo'},
];
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

export default function LensProgress({on,onGo}){
  return (
    <section className={'fw-view fw-progress'+(on?' on':'')}>
      <div className="fw-rg-inner">
        <div className="fw-lane">
          <div className="fw-lane-head"><span className="ln-dot" style={{background:'#0284c7'}}/>本项目 · RDC 主线 <small>F#3734 所在证据链 · Phase 4 → 39</small></div>
          <div className="fw-tl">
            {MAINLINE.map(n=>(
              <button key={n.lb} className={'fw-tl-node '+n.st} onClick={()=>window.alert(n.lb+'：'+n.tt.replace('\n',' · ')+'（详情卡接入 atlas_ledger.json 后开放）')}>
                <div className="fw-tl-dot"/>
                <div className="lb">{n.lb}</div>
                <div className="tt">{n.tt.split('\n').map((t,i)=><span key={i}>{t}<br/></span>)}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="fw-lane">
          <div className="fw-lane-head"><span className="ln-dot" style={{background:'#6366f1'}}/>同行进展 <small>点击节点 → 与本项目证据链对比</small></div>
          <div className="fw-peer-row">{PEERS_ROW1.map((c,i)=><PeerCard key={i} c={c}/>)}</div>
          <div className="fw-peer-row" style={{marginBottom:0}}>
            {PEERS_ROW2.map((c,i)=><PeerCard key={i} c={c}/>)}
          </div>
        </div>

        <div className="fw-rg-note">
          <b>路线图逻辑</b>：每个节点携带证据（Phase / sealed 产物 / 论文 DOI），点击节点切换对象路由；同行差异以「对比卡」登记入账本（has_data / observed / generalization_checked / mechanism_evidence 四级体系）。
        </div>
        <div style={{textAlign:'center',padding:'4px 0 10px'}}>
          <button className="fw-tbtn" onClick={()=>onGo('process')}>回到当前任务 Q06 →</button>
        </div>
      </div>
    </section>
  );
}
