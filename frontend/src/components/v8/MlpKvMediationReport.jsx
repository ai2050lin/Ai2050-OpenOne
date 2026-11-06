import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import MechanismScene,{NativeArray} from './MechanismScene.jsx';

const dims={n_layers:36,d_ff:9728,d_model:2560,n_heads:32,n_kv_heads:8,head_dim:128};
const tabs=[['summary','结果与能力'],['native','原生 KV 3D'],['dose','剂量与恢复'],['cases','逐题与反例']];
const names={clean:'干净前缀',head_half:'B34/H2 衰减 0.5',head_full:'B34/H2 移除',kv_replay:'回放编辑 KV',kv_rescue:'恢复干净 KV',
  k_only:'仅回放 K',v_only:'仅回放 V',matched:'等范数扰动',B33_MLP:'移除 B33 MLP',B34_H2:'移除 B34/H2',
  replay_both:'回放 B34＋B35',rescue_both:'恢复 B34＋B35',only_B34:'仅回放 B34',only_B35:'仅回放 B35'};
const cohorts={pilot:'校准试跑',formal:'正式行为门',mediation:'正式路径干预',earlier:'B33 MLP 路径',screen:'低余量筛查',targeted:'低余量干预',capability:'能力监测'};
const pct=v=>Number.isFinite(v)?`${(v*100).toFixed(1)}%`:'—';
const num=(v,d=3)=>!Number.isFinite(v)?'—':v!==0&&Math.abs(v)<.001?v.toExponential(2):v.toFixed(d);
const pp=v=>Number.isFinite(v)?(100*v).toFixed(2):'—';
const bounded=(v,max)=>Math.max(0,Math.min(Math.max(0,max),Math.floor(Number(v)||0)));
const artifact=sha=>sha?`/api/research-library/artifacts/${sha}`:null;

function useCase(index,list){
  const size=list?.chunk_size||64;
  const response=useApi(artifact(list?.chunks?.[Math.floor(index/size)]));
  return {...response,row:response.data?.cases?.[index%size]};
}

function decodeBits(values){
  const scratch=new ArrayBuffer(4),bits=new Uint32Array(scratch),floats=new Float32Array(scratch);
  return values.map(value=>{bits[0]=value<<16;return floats[0]});
}

function Summary({doc}){
  const m=doc.mediation,s=m.summary.all,t=doc.targeted?.summary?.all;
  const b=doc.earlier;
  return <>
    <p><strong>{m.checks.replay_logits_bitwise_equal}/{s.n} 个样本重现干预 logits，{m.checks.rescue_logits_bitwise_equal}/{s.n} 个恢复干净 logits。</strong> 源位置是问题前 #991，B34/H2 的写入经下一块 B35 的原生 K/V 槽传到后续回答分布。</p>
    <p>新材料使用带错误草稿的长文本。正式行为门{doc.formal.passed?'通过':'未通过'}，{doc.formal.n} 条的正确率为 {pct(doc.formal.accuracy)}，中位首词间隔为 {num(doc.formal.median_gap,1)}。</p>
    <p>正式样本的生成效应门{s.passed?'通过':'未通过'}。其中 {m.summary.low_margin.n} 个干净答案正确且间隔不超过 8。</p>
    {doc.screen&&<p>独立筛选 {doc.screen.n_screened} 个新背景，{doc.screen.eligible} 个符合低余量条件。实际干预 {t?.n||0} 个，生成效应门{t?.passed?'通过':t?.eligible?'未通过':'确认资格未达'}。</p>}
    <details><summary>B33 MLP 的双层缓存路径及末层负对照</summary>
      <p>B33 MLP 在 {b.summary.n} 个样本中经 B34 与 B35 的源槽传递。两个槽的联合回放在 {b.checks.replay_logits_bitwise_equal} 个样本逐位重现 logits，恢复在 {b.checks.rescue_logits_bitwise_equal} 个样本逐位一致。</p>
      <pre className="v8-artifact-json">{JSON.stringify({operations:b.summary.operations,controls:b.negative_controls},null,2)}</pre>
    </details>
    <p>能力保持使用已暴露的正式监测池和 2 个百分点的非劣效门。</p>
    <table className="v6-table"><thead><tr><th>操作</th><th>正式题数</th><th>净损失区间（百分点）</th><th>能力门</th></tr></thead><tbody>
      {Object.entries(doc.capability.summary).map(([key,v])=><tr key={key}><td>{names[key]}</td><td>{v.n}</td><td>{v.loss_interval.map(pp).join(' ～ ')}</td><td>{v.status==='noninferior'?'通过':v.status==='inferior'?'超过允许损失':'尚未证实'}</td></tr>)}
    </tbody></table>
  </>;
}

function Native({doc}){
  const [cohort,setCohort]=useState(doc.native.targeted?'targeted':'mediation');
  const list=doc.native[cohort];
  const [index,setIndex]=useState(0),[group,setGroup]=useState(0),[field,setField]=useState('K'),[show,setShow]=useState('delta');
  const raw=useCase(index,list),record=useCase(index,doc.cases[cohort]);
  const values=raw.row?.fields,actual=record.row;
  const selected=values&&field!=='head34_2'?field==='incoming_b35'?decodeBits(values[`clean_${field}`]):decodeBits(values[`clean_${field}`][group]):values?decodeBits(values.clean_head34_2):[];
  const alternate=values&&field!=='head34_2'?(field==='incoming_b35'?decodeBits(values[`edited_${field}`]):decodeBits(values[`edited_${field}`][group])):selected;
  const shown=show==='clean'?selected:show==='edited'?alternate:selected.map((v,i)=>alternate[i]-v);
  return <section aria-label="问题前原生 KV 三维轨迹">
    <div className="v6-toolbar"><label>采集 <select aria-label="KV 研究批次" value={cohort} onChange={e=>{setCohort(e.target.value);setIndex(0)}}>{Object.keys(doc.native).map(k=><option key={k} value={k}>{cohorts[k]}</option>)}</select></label>
      <label>样本 <input aria-label="KV 原生样本索引" type="number" min="0" max={list.n-1} value={index} onChange={e=>setIndex(bounded(e.target.value,list.n-1))}/></label>
      <label>GQA 组 <input aria-label="KV 原生组" type="number" min="0" max="7" value={group} onChange={e=>setGroup(bounded(e.target.value,7))}/></label>
    </div>
    {raw.error?<Empty>{raw.error}</Empty>:!values?<p>读取归档的原生 BF16 缓存槽…</p>:<>
      <p>{actual?.id||raw.row.id} · 源 token #991 · B34/H2 → B35 输入残差 → B35 K/V → 回答位置。每个 KV 组服务四个查询 Head。</p>
      <MechanismScene manifest={dims} block={34} head={2} unit={0} onBlock={()=>{}} onHead={()=>{}} onUnit={()=>{}} showUnitMarker={false}
        caption="原生结构索引 · B34/H2 为干预点 · B35 的真实 KV 阵列在下方"/>
      <div className="v6-toolbar"><label>字段 <select aria-label="KV 原生字段" value={field} onChange={e=>{setField(e.target.value);if(e.target.value==='head34_2')setShow('clean')}}>
        <option value="K">B35 旋转后 K</option><option value="V">B35 V</option><option value="incoming_b35">B35 注意力前残差</option><option value="head34_2">原始 B34/H2 输出</option>
      </select></label><label>显示 <select aria-label="KV 比较方式" value={show} disabled={field==='head34_2'} onChange={e=>setShow(e.target.value)}>
        <option value="delta">编辑减干净</option><option value="clean">干净</option><option value="edited">编辑后</option>
      </select></label></div>
      <NativeArray values={shown} selected={-1} label={`B35 上游完整原生 ${field} 阵列`}/>
      {actual&&<p>回放 logits 逐位一致 {actual.replay_logits_bitwise_equal?'是':'否'}，恢复 logits 逐位一致 {actual.rescue_logits_bitwise_equal?'是':'否'}。</p>}
      <details><summary>源槽坐标、全部八组的扰动范数与产物哈希</summary><pre className="v8-artifact-json">{JSON.stringify({delta_norm:actual?.delta_KV_norm_by_GQA,native:list.provenance},null,2)}</pre></details>
    </>}
  </section>;
}

function Dose({doc}){
  return <>{[['mediation',doc.mediation],['targeted',doc.targeted]].filter(([,r])=>r?.summary?.all?.operations).map(([key,r])=>{
    const group=r.summary.all;
    return <details key={key} open><summary>{cohorts[key]} · {group.n} 个 · 生成效应门{group.passed?'通过':group.eligible?'未通过':'确认资格未达'}</summary>
      <table className="v6-table"><thead><tr><th>操作</th><th>首词正确率</th><th>翻错 / 转对</th><th>相对干净 KL</th><th>平均首词间隔</th></tr></thead><tbody>
        {Object.entries(group.operations).map(([name,v])=><tr key={name}><td>{names[name]}</td><td>{pct(v.accuracy)}</td><td>{v.losses} / {v.gains}</td><td>{num(v.mean_KL)}</td><td>{num(v.mean_gap)}</td></tr>)}
      </tbody></table><details><summary>配对区间、K/V 交互与数值核验</summary><pre className="v8-artifact-json">{JSON.stringify({summary:r.summary,checks:r.checks},null,2)}</pre></details>
    </details>;
  })}</>;
}

function Cases({doc}){
  const [cohort,setCohort]=useState(doc.cases.targeted?.n?'targeted':'mediation'),[index,setIndex]=useState(0);
  const list=doc.cases[cohort],entry=useCase(index,list);
  return <><div className="v6-toolbar"><label>证据集合 <select aria-label="KV 证据集合" value={cohort} onChange={e=>{setCohort(e.target.value);setIndex(0)}}>
    {Object.entries(doc.cases).filter(([,v])=>v.n).map(([key,v])=><option key={key} value={key}>{cohorts[key]}（{v.n}）</option>)}
  </select></label><label>索引 <input aria-label="KV 证据索引" type="number" min="0" max={list.n-1} value={index} onChange={e=>setIndex(bounded(e.target.value,list.n-1))}/></label></div>
    {entry.error&&<Empty>{entry.error}</Empty>}{entry.row&&<pre className="v8-artifact-json">{JSON.stringify(entry.row,null,2)}</pre>}
    <details><summary>预注册、材料来源、原始研究与内容哈希</summary><pre className="v8-artifact-json">{JSON.stringify({prereg:doc.prereg,targeted:doc.targeted_prereg,refs:doc.refs},null,2)}</pre></details>
  </>;
}

export default function MlpKvMediationReport({doc}){
  const [tab,setTab]=useState(()=>{
    const selected=new URLSearchParams(window.location.search).get('kvview');return tabs.some(([key])=>key===selected)?selected:'summary';
  });
  return <section aria-label="Qwen3 原生缓存路径与低余量复核">
    <p><strong>问题前写入怎样经 KV 到达后续回答</strong></p>
    <p className="v8-note">{doc.run_id} · 本地 Qwen3-4B BF16 · {doc.state==='complete'?'已归档':'独立低余量复核进行中'}</p>
    <div className="v6-toolbar">{tabs.map(([key,label])=><button className="fw-tbtn" key={key} aria-pressed={tab===key} onClick={()=>setTab(key)}>{label}</button>)}</div>
    {tab==='summary'&&<Summary doc={doc}/>}{tab==='native'&&<Native doc={doc}/>}{tab==='dose'&&<Dose doc={doc}/>}{tab==='cases'&&<Cases doc={doc}/>}
  </section>;
}
