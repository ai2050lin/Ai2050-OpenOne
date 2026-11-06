import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import MechanismScene,{NativeArray} from './MechanismScene.jsx';

const dims={n_layers:36,d_ff:9728,d_model:2560,n_heads:32,n_kv_heads:8,head_dim:128};
const roles={approved:'签署记录的目标属性',other:'另一记录的属性',draft:'草稿中的错误属性',selector:'选择键'};
const ops={clean:'干净',joint_1:'联合移除', 'joint_0.5':'联合衰减 0.5',head_only:'仅 Head',mlp_only:'仅 MLP',matched:'逐向量等范数对照',matched_contextual_v1:'初版对照',
  kv_replay:'全部变动 KV 回放',kv_rescue:'恢复原 KV',source_only:'只回放源 token',other_slots_only:'只回放其余槽'};
const sets={discovery:'发现集读出',screen:'完整干净筛查',causal:'联合干预与反例',by_role:'按源角色清零 V',capability:'能力监测'};
const pct=x=>Number.isFinite(x)?`${(100*x).toFixed(1)}%`:'—';
const num=x=>Number.isFinite(x)?(x!==0&&Math.abs(x)<.001?x.toExponential(2):x.toFixed(3)):'—';
const bounded=(x,max)=>Math.max(0,Math.min(Math.max(0,max),Math.floor(Number(x)||0)));

function useCase(index,list){
  const size=list?.chunk_size||64,sha=list?.chunks?.[Math.floor(index/size)];
  const response=useApi(sha?`/api/research-library/artifacts/${sha}`:null);
  return {...response,row:response.data?.cases?.[index%size]};
}

function decode(values){
  const mem=new ArrayBuffer(4),u=new Uint32Array(mem),f=new Float32Array(mem);
  return values.map(v=>{u[0]=v<<16;return f[0]});
}

function Summary({doc}){
  const c=doc.causal,m=c.summary.all;
  return <>
    <p>在 {doc.route.discovery_n} 个发现背景上冻结源角色 {doc.route.roles.map(k=>roles[k]).join('、')}，写入包括 {doc.route.heads.map(([b,h])=>`B${b}/H${h}`).concat(doc.route.mlps.map(b=>`B${b} MLP`)).join('、')}。</p>
    <p>两个留出筛查 {Object.values(doc.screen.summary).reduce((s,r)=>s+r.screened,0)} 个新背景，共纳入 {m.n} 个低余量样本{m.n?`，干净间隔中位 ${num(m.median_clean_gap)}`:''}。</p>
    {m.n>0&&<><p><strong>回放 {c.checks.replay_logits_bitwise_equal}/{m.n}，恢复 {c.checks.rescue_logits_bitwise_equal}/{m.n} 逐位一致。</strong> 联合移除的生成首词变化为 {m.operations.joint_1.first_word_changes} 条。预注册效应门{m.passed?'通过':m.eligible?'未通过':'确认资格未达'}。</p>
      <table className="v6-table"><thead><tr><th>回答查询</th><th>固定 Q/K 后有真实 V 读出</th><th>预测与真实变化的中位余弦</th></tr></thead><tbody>
        {Object.entries(c.read_summary).map(([b,r])=><tr key={b}><td>B{b}</td><td>{r.nonzero_read}/{m.n}</td><td>{num(r.median_cosine)}</td></tr>)}
      </tbody></table>
      {doc.by_role&&<details><summary>两个源角色各自的实际读取核验</summary>
        <table className="v6-table"><thead><tr><th>源角色</th><th>层</th><th>非零变化</th><th>变化方向余弦中位</th></tr></thead><tbody>
          {Object.entries(doc.by_role.summary).flatMap(([k,read])=>Object.entries(read).map(([b,r])=><tr key={`${k}-${b}`}><td>{roles[k]}</td><td>B{b}</td><td>{r.nonzero_read}/{r.n}</td><td>{num(r.median_cosine)}</td></tr>))}
        </tbody></table></details>}</>}
    <p>两项能力监测沿用 2 个百分点非劣效门，使用已暴露的正式题池。</p>
    <table className="v6-table"><thead><tr><th>操作</th><th>正式题数</th><th>净损失区间（百分点）</th><th>能力门</th></tr></thead><tbody>
      {Object.entries(doc.capability.summary).map(([k,v])=><tr key={k}><td>{k==='joint'?'联合移除':'等范数写入'}</td><td>{v.n}</td><td>{v.loss_interval.map(x=>(x*100).toFixed(2)).join(' ～ ')}</td><td>{v.status==='noninferior'?'通过':v.status==='inferior'?'超过允许损失':'尚未证实'}</td></tr>)}
    </tbody></table>
  </>;
}

function Native({doc}){
  const [index,setIndex]=useState(0),[role,setRole]=useState(doc.route.roles[0]),[block,setBlock]=useState(35),[head,setHead]=useState(0);
  const [field,setField]=useState('values'),[state,setState]=useState('delta');
  const report=useCase(index,doc.cases.causal),raw=useCase(index,doc.native),single=useCase(index,doc.cases.by_role);
  if(!doc.native.n)return <Empty>本轮没有入选的原生留出样本。</Empty>;
  const r=report.row,v=raw.row,read=r?.read?.roles?.[role]?.[String(block)];
  const anchor=r?.anchors?.[role]?.[0],group=Math.floor(head/4),p=anchor?v?.positions.indexOf(anchor.position):-1;
  const clean=p>=0&&v?.clean?.[block]?.[field]?.[p]?.[group];const edited=p>=0&&v?.edited?.[block]?.[field]?.[p]?.[group];
  const before=clean?decode(clean):[],after=edited?decode(edited):[];
  const values=state==='clean'?before:state==='edited'?after:after.map((x,i)=>x-before[i]);
  return <section aria-label="回答查询与更早源 token 的真实三维读出">
    <div className="v6-toolbar">
      <label>样本 <input aria-label="早期路径样本" type="number" min="0" max={doc.native.n-1} value={index} onChange={e=>setIndex(bounded(e.target.value,doc.native.n-1))}/></label>
      <label>源角色 <select aria-label="早期源角色" value={role} onChange={e=>setRole(e.target.value)}>{Object.entries(roles).map(([key,label])=><option key={key} value={key}>{label}</option>)}</select></label>
      <label>回答读取层 <select aria-label="早期读取层" value={block} onChange={e=>setBlock(Number(e.target.value))}><option value="34">B34</option><option value="35">B35</option></select></label>
      <label>查询 Head <input aria-label="早期读取 Head" type="number" min="0" max="31" value={head} onChange={e=>setHead(bounded(e.target.value,31))}/></label>
    </div>
    {(report.error||raw.error)?<Empty>{report.error||raw.error}</Empty>:!r||!v?<p>读取归档的源坐标和原生缓存…</p>:<>
      <p>{r.id} · 源 {r.anchors[role].map(a=>`#${a.position} ${JSON.stringify(a.text)}`).join('、')} · 回答查询 #1033 · GQA 组 {group}</p>
      <MechanismScene manifest={dims} block={block} head={head} unit={0} headOverlay={read.mass_by_head} onBlock={b=>{if(b===34||b===35)setBlock(b)}}
        onHead={setHead} onUnit={()=>{}} showUnitMarker={false} caption="回答位置重建的原生注意力 · 蓝点为各 Head 对当前源角色的权重总和"/>
      <p>H{head} 的注意力权重总和 {num(read.mass_by_head[head])}，投影后读取量 {num(read.projected_norm_by_head[head])}。该角色包含多个子词时，原生缓存阵列显示首个子词。</p>
      <div className="v6-toolbar"><label>缓存字段 <select aria-label="早期原生 KV 字段" value={field} onChange={e=>setField(e.target.value)}><option value="keys">旋转后 K</option><option value="values">V</option></select></label>
        <label>显示 <select aria-label="早期原生 KV 比较" value={state} onChange={e=>setState(e.target.value)}><option value="delta">编辑减干净</option><option value="clean">干净</option><option value="edited">编辑后</option></select></label></div>
      {values.length?<NativeArray values={values} selected={-1} label={`早期源 token ${field} 原生阵列`}/>:<p>该角色有读出记录，原生缓存只归档本轮冻结的干预角色。</p>}
      <details><summary>实际清零 V 的 Head 变化、K/V 传播范围与逐位核验</summary><pre className="v8-artifact-json">{JSON.stringify({jointRead:r.read_verify,roleRead:single.row?.roles?.[role],changed:r.changed_slots,replay:r.replay_logits_bitwise_equal,rescue:r.rescue_logits_bitwise_equal},null,2)}</pre></details>
    </>}
  </section>;
}

function Dose({doc}){
  return <>{Object.entries(doc.causal.summary).filter(([,r])=>r.operations).map(([key,r])=><details key={key} open={key==='all'}>
    <summary>{key} · {r.n} 个 · {r.passed?'达到效应门':r.eligible?'未达到效应门':'分组描述'}</summary>
    <table className="v6-table"><thead><tr><th>操作</th><th>首词正确率</th><th>首词变化</th><th>翻错 / 转对</th><th>全词表 KL</th><th>平均间隔差</th></tr></thead><tbody>
      {Object.entries(r.operations).map(([k,v])=><tr key={k}><td>{ops[k]}</td><td>{pct(v.accuracy)}</td><td>{v.first_word_changes}</td><td>{v.losses} / {v.gains}</td><td>{num(v.mean_KL)}</td><td>{num(v.gap_delta.mean)}</td></tr>)}
    </tbody></table><details><summary>配对置信区间与等范数比较</summary><pre className="v8-artifact-json">{JSON.stringify(r,null,2)}</pre></details>
  </details>)}</>;
}

function Evidence({doc}){
  const [kind,setKind]=useState(doc.cases.causal.n?'causal':'screen'),[index,setIndex]=useState(0);
  const list=doc.cases[kind],r=useCase(index,list);
  return <>
    <div className="v6-toolbar"><label>证据集合 <select aria-label="早期路径证据集合" value={kind} onChange={e=>{setKind(e.target.value);setIndex(0)}}>
      {Object.entries(doc.cases).filter(([,v])=>v.n).map(([k,v])=><option key={k} value={k}>{sets[k]}（{v.n}）</option>)}</select></label>
      <label>证据索引 <input aria-label="早期路径证据索引" type="number" min="0" max={list.n-1} value={index} onChange={e=>setIndex(bounded(e.target.value,list.n-1))}/></label></div>
    <details><summary>首词变化与高分布变化反例</summary>{doc.counterexamples.map(r=><button className="fw-tbtn" key={r.index} onClick={()=>{setKind('causal');setIndex(r.index)}}>{r.index} · {r.expected} → {r.after||'空'} · KL {num(r.KL)}</button>)}</details>
    {r.error&&<Empty>{r.error}</Empty>}{r.row&&<pre className="v8-artifact-json">{JSON.stringify(r.row,null,2)}</pre>}
    <details><summary>预注册、发现集冻结与产物哈希</summary><pre className="v8-artifact-json">{JSON.stringify({prereg:doc.prereg,route:doc.route,refs:doc.refs},null,2)}</pre></details>
  </>;
}

const tabs=[['summary','结果与能力'],['native','源 token 与原生 KV 3D'],['dose','联合干预与恢复'],['cases','逐题与反例']];
export default function MlpEarlyReadReport({doc}){
  const [tab,setTab]=useState(()=>{const q=new URLSearchParams(window.location.search).get('readview');return tabs.some(([k])=>k===q)?q:'summary'});
  return <section aria-label="早期源 token 读取与联合写入">
    <p><strong>回答读取的早期源 token 与联合写入路径</strong></p>
    <p className="v8-note">{doc.run_id} · 本地 Qwen3-4B BF16 · 已归档</p>
    <div className="v6-toolbar">{tabs.map(([k,label])=><button key={k} className="fw-tbtn" aria-pressed={tab===k} onClick={()=>setTab(k)}>{label}</button>)}</div>
    {tab==='summary'&&<Summary doc={doc}/>}{tab==='native'&&<Native doc={doc}/>}{tab==='dose'&&<Dose doc={doc}/>}{tab==='cases'&&<Evidence doc={doc}/>}
  </section>;
}
