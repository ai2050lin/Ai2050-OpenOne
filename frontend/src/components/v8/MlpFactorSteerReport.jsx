import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import MechanismScene,{NativeArray} from './MechanismScene.jsx';

const dimensions={n_layers:36,d_model:2560,d_ff:9728,n_heads:32,n_kv_heads:8,head_dim:128};
const tasks={lookup:'属性查询',dispatch:'派遣规则（新任务）'};
const factors={evidence_when_A_selected:'改属性，选中该记录',evidence_when_B_selected:'改属性，选中另一记录',
  key_switch_different_attributes:'改键，两条属性不同',key_switch_same_attribute:'改键，两条属性相同',draft_at_fixed_truth_and_key:'事实与键固定，仅改草稿'};
const roles={attribute:'签署属性词',selector:'选择键',tail_a:'前缀尾 −1',tail_b:'前缀尾'};
const operations={clean:'干净',targeted:'定向写入',half:'半剂量',matched:'严格等范数',head_only:'仅 Head',mlp_only:'仅 MLP',
  replay:'全部变动 KV 回放',rescue:'恢复干净 KV',source_only:'只回放源槽'};
const sets={factorial:'最小对照',selection:'发现集写入筛选',screen_lookup:'属性查询筛查',screen_dispatch:'新任务筛查',
  lookup:'属性查询逐题',dispatch:'派遣规则逐题',capability:'能力监测'};
const pct=v=>Number.isFinite(v)?`${(100*v).toFixed(1)}%`:'—';
const num=v=>!Number.isFinite(v)?'—':v!==0&&Math.abs(v)<.001?v.toExponential(2):v.toFixed(3);
const bound=(v,n)=>Math.max(0,Math.min(Math.max(0,n-1),Math.floor(Number(v)||0)));

function useCase(list,index){
  const size=list?.chunk_size||64,sha=list?.chunks?.[Math.floor(index/size)];
  const r=useApi(sha?`/api/research-library/artifacts/${sha}`:null);
  return {row:r.data?.cases?.[index%size],error:r.error};
}

function bf16(values){
  const buf=new ArrayBuffer(4),u=new Uint32Array(buf),f=new Float32Array(buf);
  return (values||[]).map(v=>{u[0]=v<<16;return f[0]});
}

function Summary({doc}){
  const recipe=doc.selection.recipe;
  return <>
    <p><strong>{doc.factorial.one_token_pairs} 个单 token 最小对照。</strong> 正向间隔表示更偏向 fishing，选择键效应按目标记录属性定向。</p>
    <table className="v6-table"><thead><tr><th>对照</th><th>平均间隔变化</th><th>95% 区间</th></tr></thead><tbody>
      {Object.entries(doc.factorial.factorial_effects).map(([k,v])=><tr key={k}><td>{factors[k]}</td><td>{num(v.mean)}</td><td>{v.ci95.map(num).join(' ～ ')}</td></tr>)}
    </tbody></table>
    <p>发现集冻结 {recipe.heads.length} 个 Head、{recipe.mlps.length} 个 MLP 写入，剂量 α={recipe.alpha}。新任务沿用该组。</p>
    <details><summary>发现集候选组合与控制结果</summary><pre className="v8-artifact-json">{JSON.stringify({recipe,sweep:doc.selection_summaries},null,2)}</pre></details>
    {Object.entries(doc.tasks).map(([key,v])=><p key={key}>{tasks[key]} · {v.summary.n} 条低余量 · 定向首词 {v.summary.operations?.targeted?.targeted_alt??0} 条 · 回放/恢复 {v.checks?.replay_logits_bitwise_equal??0}/{v.checks?.rescue_logits_bitwise_equal??0} · {v.summary.passed?'效应门通过':v.summary.eligible?'效应门未通过':'确认资格未达'}</p>)}
    <table className="v6-table"><thead><tr><th>能力操作</th><th>题数</th><th>净损失区间（百分点）</th><th>能力门</th></tr></thead><tbody>
      {Object.entries(doc.capability.summary).map(([key,v])=><tr key={key}><td>{operations[key]}</td><td>{v.n}</td><td>{v.loss_interval.map(x=>(x*100).toFixed(2)).join(' ～ ')}</td><td>{v.status==='noninferior'?'通过':v.status==='inferior'?'超过容许损失':'尚未证实'}</td></tr>)}
    </tbody></table>
  </>;
}

function Native({doc}){
  const choices=Object.keys(doc.native).filter(k=>doc.native[k].n);
  const [task,setTask]=useState(choices[0]||'lookup'),[index,setIndex]=useState(0),[role,setRole]=useState('attribute');
  const [group,setGroup]=useState(0),[field,setField]=useState('values'),[view,setView]=useState('delta');
  const raw=useCase(doc.native[task],index),proof=useCase(doc.cases[task],index);
  if(!choices.length)return <Empty>暂无入选原生样本。</Empty>;
  const native=raw.row,r=proof.row;
  const before=bf16(native?.clean?.[role]?.[field]?.[group]),after=bf16(native?.edited?.[role]?.[field]?.[group]);
  const values=view==='clean'?before:view==='edited'?after:after.map((v,i)=>v-before[i]);
  const [block,head]=doc.selection.recipe.heads[0];
  return <section aria-label="定向写入的原生 KV 三维证据">
    <div className="v6-toolbar"><label>任务 <select aria-label="定向路径任务" value={task} onChange={e=>{setTask(e.target.value);setIndex(0)}}>{choices.map(k=><option key={k} value={k}>{tasks[k]}</option>)}</select></label>
      <label>样本 <input type="number" aria-label="定向原生样本" min="0" max={doc.native[task].n-1} value={index} onChange={e=>setIndex(bound(e.target.value,doc.native[task].n))}/></label>
      <label>源 <select aria-label="定向源角色" value={role} onChange={e=>setRole(e.target.value)}>{Object.entries(roles).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
      <label>GQA <input type="number" aria-label="定向 KV 组" min="0" max="7" value={group} onChange={e=>setGroup(bound(e.target.value,8))}/></label></div>
    {raw.error||proof.error?<Empty>{raw.error||proof.error}</Empty>:!native||!r?<p>读取服务器归档原生槽…</p>:<>
      <p>{r.id} · 源 #{native.positions[role]} · 目标 {r.contrast} · 原首词 {r.conditions.clean.word} · 干预后 {r.conditions.targeted.word||'（空）'}</p>
      <MechanismScene manifest={dimensions} block={block} head={head} unit={0} showUnitMarker={false} onBlock={()=>{}} onHead={()=>{}} onUnit={()=>{}}
        caption="冻结写入组的首个 Head 位置 · B35 真实缓存值显示在下方"/>
      <div className="v6-toolbar"><label>字段 <select aria-label="定向缓存字段" value={field} onChange={e=>setField(e.target.value)}><option value="keys">RoPE 后 K</option><option value="values">V</option></select></label>
        <label>显示 <select aria-label="定向缓存比较" value={view} onChange={e=>setView(e.target.value)}><option value="delta">编辑减干净</option><option value="clean">干净</option><option value="edited">编辑</option></select></label></div>
      <NativeArray values={values} selected={-1} label="B35 早期源原生 KV"/>
      <details><summary>源位置、传播范围、回放和恢复</summary><pre className="v8-artifact-json">{JSON.stringify({anchors:r.anchors,changed:r.changed_slots,replay:r.replay_logits_bitwise_equal,rescue:r.rescue_logits_bitwise_equal},null,2)}</pre></details>
    </>}
  </section>;
}

function Effect({doc}){
  return <>{Object.entries(doc.tasks).map(([k,v])=>v.summary.operations&&<details open key={k}><summary>{tasks[k]} · {v.summary.n} 条</summary>
    <table className="v6-table"><thead><tr><th>操作</th><th>原答案正确率</th><th>定向首词</th><th>全词表 KL</th><th>平均定向间隔差</th></tr></thead><tbody>
      {Object.entries(v.summary.operations).map(([op,x])=><tr key={op}><td>{operations[op]}</td><td>{pct(x.accuracy)}</td><td>{x.targeted_alt}</td><td>{num(x.mean_KL)}</td><td>{num(x.target_gap_gain.mean)}</td></tr>)}
    </tbody></table><details><summary>配对检验与范数核验</summary><pre className="v8-artifact-json">{JSON.stringify({summary:v.summary,checks:v.checks},null,2)}</pre></details>
  </details>)}</>;
}

function Cases({doc}){
  const [kind,setKind]=useState(doc.cases.lookup?.n?'lookup':'factorial'),[index,setIndex]=useState(0);
  const list=doc.cases[kind],result=useCase(list,index);
  return <><div className="v6-toolbar"><label>证据 <select aria-label="定向路径证据集合" value={kind} onChange={e=>{setKind(e.target.value);setIndex(0)}}>
    {Object.entries(doc.cases).filter(([,v])=>v.n).map(([key,v])=><option key={key} value={key}>{sets[key]}（{v.n}）</option>)}</select></label>
    <label>索引 <input type="number" aria-label="定向路径证据索引" min="0" max={list.n-1} value={index} onChange={e=>setIndex(bound(e.target.value,list.n))}/></label></div>
    <details><summary>定向首词与对照反例</summary>{doc.counterexamples.map(r=><button className="fw-tbtn" key={`${r.task}-${r.index}`} onClick={()=>{setKind(r.task);setIndex(r.index)}}>{tasks[r.task]} #{r.index} · {r.expected} → {r.targeted||'空'} / {r.matched||'空'}</button>)}</details>
    {result.error&&<Empty>{result.error}</Empty>}{result.row&&<pre className="v8-artifact-json">{JSON.stringify(result.row,null,2)}</pre>}
    <details><summary>预注册、冻结方法与内容哈希</summary><pre className="v8-artifact-json">{JSON.stringify({prereg:doc.prereg,refs:doc.refs},null,2)}</pre></details></>;
}

const tabs=[['summary','最小对照与能力'],['native','原生 KV 3D'],['effect','定向效应与恢复'],['cases','逐题与反例']];
export default function MlpFactorSteerReport({doc}){
  const [tab,setTab]=useState(()=>{const v=new URLSearchParams(window.location.search).get('factorview');return tabs.some(([k])=>k===v)?v:'summary'});
  return <section aria-label="最小因子对照与写入定向复核"><p><strong>属性、选择键与草稿的最小对照</strong></p><p className="v8-note">{doc.run_id} · 本地 Qwen3-4B BF16</p>
    <div className="v6-toolbar">{tabs.map(([k,v])=><button className="fw-tbtn" key={k} aria-pressed={tab===k} onClick={()=>setTab(k)}>{v}</button>)}</div>
    {tab==='summary'&&<Summary doc={doc}/>}{tab==='native'&&<Native doc={doc}/>}{tab==='effect'&&<Effect doc={doc}/>}{tab==='cases'&&<Cases doc={doc}/>}</section>;
}
