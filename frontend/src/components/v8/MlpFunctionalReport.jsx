import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import MechanismScene,{NativeArray} from './MechanismScene.jsx';

const dims={n_layers:36,d_ff:9728,d_model:2560,n_heads:32,n_kv_heads:8,head_dim:128};
const tabs=[['dictionary','字典行为保真'],['paths','问题前路径'],['native','真实原场 3D'],['causal','干预与能力'],['evidence','材料与反例']];
const operation={clean:'原模型',zero:'B35 写入置零',archived_sae:'原 SAE',mse_control:'同预算 MSE 对照',functional_sae:'行为校准 SAE',native256:'原生 256 单元',
  'joint_0.5':'联合移除 α=0.5',joint_1:'联合移除 α=1',attention_1:'仅移除 Head',mlp_1:'仅移除 MLP',head34_2:'仅移除 B34/H2',random_1:'等范数扰动对照',upstream_joint:'上游联合移除'};
const labels={location:'地理属性',industry:'生计属性'};
const ni={noninferior:'通过',inferior:'超过允许损失',inconclusive:'尚未证实',not_tested:'未测试'};
const number=(x,d=4)=>Number.isFinite(x)?x.toFixed(d):'—';
const smallNumber=x=>Number.isFinite(x)&&x!==0&&Math.abs(x)<0.0001?x.toExponential(2):number(x);
const pct=x=>Number.isFinite(x)?`${(100*x).toFixed(1)}%`:'—';
const bounded=(v,max)=>Math.min(Math.max(0,Math.floor(Number(v)||0)),Math.max(0,max));
const fromQuery=(key,fallback,max)=>bounded(new URLSearchParams(window.location.search).get(key)??fallback,max);
const artifact=sha=>sha?`/api/research-library/artifacts/${sha}`:null;

function Fidelity({doc}){
  return <>
    <p>{doc.conclusion}</p>
    <p>两个新留出各有 256 篇 1024 token 文档，每篇用 8 个下一词位置。原生单元与字典使用相同的每位置活跃预算 k=256。</p>
    {Object.entries(doc.dictionary?.summary||{}).map(([split,entry])=><details key={split} open={split==='lockbox'}>
      <summary>{split} · {entry.documents} 篇 · 同预算比较门{entry.passed?'通过':'未通过'}</summary>
      <table className="v6-table"><thead><tr><th>操作</th><th>KL</th><th>下一词 CE</th><th>FVU</th><th>损失恢复</th></tr></thead><tbody>
        {Object.entries(entry.operations).map(([name,v])=><tr key={name}><td>{operation[name]||name}</td><td>{number(v.KL_clean)}</td><td>{number(v.CE)}</td><td>{number(v.FVU)}</td><td>{pct(v.loss_recovered)}</td></tr>)}
      </tbody></table>
      <details><summary>配对区间与对照</summary><pre className="v8-artifact-json">{JSON.stringify(entry.comparisons,null,2)}</pre></details>
    </details>)}
    <details><summary>校准目标、冻结训练与开发曲线</summary><pre className="v8-artifact-json">{JSON.stringify(doc.training,null,2)}</pre></details>
  </>;
}

function Paths({doc,choose}){
  const [attr,setAttr]=useState('location'),[block,setBlock]=useState(35),[showAll,setShowAll]=useState(false);
  const full=useApi(showAll?artifact(doc.refs?.paths):null);
  const path=doc.paths;
  if(!path)return <Empty>正在采集并核对问题前原场。</Empty>;
  const rev=doc.revision?.replication?.[attr];
  const observation=rev||path.observations[attr];
  const writers=rev?.writers||path.selected[attr];
  const remainder=attr==='industry'?doc.revision?.remainder?.summary?.lockbox:null;
  return <>
    <p>原始材料行为门{path.behavior_gate?'通过':'未通过'}。{doc.revision?`修订材料行为门${doc.revision.passed?'通过':'未通过'}。`:''}路径由原发现集提名，修订使用冻结的方向和写入，在两组未见材料上检查。</p>
    <div className="v6-toolbar" aria-label="选择属性">{Object.entries(labels).map(([key,label])=><button key={key} className="fw-tbtn" aria-pressed={attr===key} onClick={()=>setAttr(key)}>{label}</button>)}</div>
    <p>进入 B35 的固定方向在锁箱中单条预测准确率 {pct(observation.incoming.lockbox.classification_accuracy)}。按两种属性配对方式的排序正确率分别为 {observation.incoming.lockbox.paired_accuracy_by_assignment.map(pct).join(' / ')}。</p>
    <table className="v6-table"><thead><tr><th>原生写入</th><th>发现集平均投影差</th><th>确认／锁箱</th><th>样本原场</th></tr></thead><tbody>
      {writers.map(p=><tr key={`${p.block}-${p.head}`}><td>B{p.block} {p.head===null?'MLP':`H${p.head}`}</td><td>{number(p.discovery_mean)}</td><td>{p.replicated?'复现':'未达复现门'}</td><td><button className="fw-tbtn" onClick={()=>choose(p.block,p.head??0)}>定位 3D</button></td></tr>)}
    </tbody></table>
    <details><summary>候选的两版留出区间、置换与投影误差</summary><pre className="v8-artifact-json">{JSON.stringify({writers,residual:observation,original:path.observations[attr]},null,2)}</pre></details>
    {remainder&&<details><summary>两 Head 写入与全残差的反例对照</summary>
      <p>锁箱中，两 Head 的联合投影在两种配对方式上的排序正确率为 {remainder.selected_heads.paired_accuracy_by_R.map(pct).join(' / ')}，完整残差为 {remainder.full_residual.paired_accuracy_by_R.map(pct).join(' / ')}。这项事后对照保留了方向一致却被全残差覆盖的样本。</p>
      <pre className="v8-artifact-json">{JSON.stringify(remainder,null,2)}</pre></details>}
    <details onToggle={e=>setShowAll(e.currentTarget.open)}><summary>查看所有原生写入的效应</summary>
      <label>层块 <input type="number" min="0" max="35" value={block} onChange={e=>setBlock(bounded(e.target.value,35))}/></label>
      {full.error&&<Empty>{full.error}</Empty>}
      {full.data&&<table className="v6-table"><thead><tr><th>写入</th><th>发现 R0 / R1</th><th>确认 R0 / R1</th><th>锁箱 R0 / R1</th></tr></thead><tbody>
        {full.data.all_writer_effects.filter(row=>row.attribute===attr&&row.block===block).map(row=><tr key={row.head??'mlp'}><td>{row.head===null?'MLP':`H${row.head}`}</td>{['discovery','confirm','lockbox'].map(split=><td key={split}>{row[split].map(v=>number(v)).join(' / ')}</td>)}</tr>)}
      </tbody></table>}
    </details>
  </>;
}

function Native({count,revision,block,setBlock,head,setHead}){
  const [cohort,setCohort]=useState('original');
  const [sample,setSample]=useState(()=>fromQuery('usample',count-1,count-1)),[field,setField]=useState('head_z');
  const revCount=revision?.native?.fields?.incoming?.[0]||0;
  const revBlocks=Object.keys(revision?.native?.fields||{}).filter(k=>k.startsWith('head')).map(k=>Number(k.slice(4)));
  const max=cohort==='revision'?revCount-1:count-1;
  const chosenBlock=cohort==='revision'&&!revBlocks.includes(block)?revBlocks.at(-1):block;
  const q=useApi(count?`/api/traces/mlp-functional/v1/samples/${sample}?block=${chosenBlock}&cohort=${cohort}`:null);
  const d=q.data;
  if(!count)return <Empty>原生采集完成后可按样本回查。</Empty>;
  return <section aria-label="问题前原生路径3D">
    <div className="v6-toolbar">
      {revCount>0&&<label>采集批次 <select aria-label="上游采集批次" value={cohort} onChange={e=>{setCohort(e.target.value);setSample(0);setField('head_z')}}><option value="original">原始 · 全层</option><option value="revision">修订 · 候选层</option></select></label>}
      <label>样本 <input aria-label="上游原生样本" type="number" min="0" max={max} value={sample} onChange={e=>setSample(bounded(e.target.value,max))}/></label>
      <label>块 {cohort==='revision'?<select aria-label="上游原生层块" value={chosenBlock} onChange={e=>setBlock(Number(e.target.value))}>{revBlocks.map(b=><option key={b} value={b}>B{b}</option>)}</select>:<input aria-label="上游原生层块" type="number" min="0" max="35" value={block} onChange={e=>setBlock(bounded(e.target.value,35))}/>}</label>
      <label>Head <input aria-label="上游原生 Head" type="number" min="0" max="31" value={head} onChange={e=>setHead(bounded(e.target.value,31))}/></label>
    </div>
    {q.error?<Empty>{q.error}。离线可查汇总、病例和来源。</Empty>:!d?<p>读取登记的 BF16 原场…</p>:<>
      <p>样本 {d.index} · {d.queries[0].split} · 问题前 token #{d.position}。同一前缀在两种问题下的生成分别为 {d.queries.map(r=>r.word||'（空）').join(' / ')}。</p>
      <MechanismScene manifest={dims} block={chosenBlock} head={head} unit={0} embedding={d.embedding} headOverlay={d.head_z.map(z=>Math.hypot(...z))}
        onBlock={setBlock} onHead={setHead} onUnit={()=>{}} showUnitMarker={false} caption="问题前 BF16 原场 · Head 色深为输出范数 · MLP 输出在下方"/>
      <label>原生字段 <select aria-label="上游原生字段" value={field} onChange={e=>setField(e.target.value)}>
        {[['head_z',`H${head} 的完整 128 维输出`],['all_heads','全部 32 Head 输出'],['mlp_output','MLP 的残差写入'],['embedding','完整词嵌入'],['residual_before_block','本块输入残差'],['residual_next_boundary','下一原生残差边界'],['incoming_b35','进入 B35 MLP 的残差']].filter(([key])=>cohort==='original'||!key.startsWith('residual_')).map(([key,label])=><option key={key} value={key}>{label}</option>)}
      </select></label>
      {field==='mlp_output'&&!d.mlp_output?<p>B35 的 MLP 在本次上游边界之后。</p>:<NativeArray values={field==='head_z'?d.head_z[head]:field==='all_heads'?d.head_z.flat():d[field]} selected={-1} label="问题前完整原生阵列"/>}
      <p className="v8-note">橙色层板为 MLP 定位。此处显示的是 2560 维残差写入。原生中间单元与逐源注意力行属于其他采集边界。</p>
      <details><summary>双问题行为、位置与原场来源</summary><pre className="v8-artifact-json">{JSON.stringify({queries:d.queries,provenance:d.provenance},null,2)}</pre></details>
    </>}
  </section>;
}

function Causal({doc}){
  return <>
    <p>干预在问题前同一 token 衰减选中的原生写入。每次从实际输入重算下游层和 KV。生成首词、分布变化与残差投影分别评分。</p>
    {!doc.generated?<Empty>范围干预尚未完成。</Empty>:doc.generated.status!=='complete'?<Empty>行为门未达到，保留不适用记录。</Empty>:
      Object.entries(doc.generated.summary).filter(([,v])=>v?.operations).map(([key,v])=><details key={key} open={key===doc.generated.summary.primary_attribute}><summary>{labels[key]} · {v.n} 条 · {key===doc.generated.summary.primary_attribute?(v.passed?'主因果门通过':'主因果门未通过'):'跨属性对照'}</summary>
        <table className="v6-table"><thead><tr><th>操作</th><th>生成正确率</th><th>相对原模型 KL</th><th>{labels[key]}残差投影</th><th>首词间隔</th></tr></thead><tbody>
          {Object.entries(v.operations).map(([name,o])=><tr key={name}><td>{operation[name]||name}</td><td>{pct(o.accuracy)}</td><td>{smallNumber(o.mean_KL)}</td><td>{number(o[`mean_signed_${key}_residual`])}</td><td>{number(o.mean_answer_gap)}</td></tr>)}
        </tbody></table><details><summary>逐题配对、效应区间与随机对照</summary><pre className="v8-artifact-json">{JSON.stringify(v,null,2)}</pre></details>
      </details>)}
    {doc.capability&&<><p>能力保持按 2 个百分点的非劣效门判断。使用已暴露的正式监测池，按本轮各实际操作复算。</p>
      <table className="v6-table"><thead><tr><th>操作</th><th>正式题数</th><th>净损失区间（百分点）</th><th>能力门</th></tr></thead><tbody>
        {Object.entries(doc.capability.summary).filter(([,v])=>v?.loss_interval).map(([name,v])=><tr key={name}><td>{operation[name]||name}</td><td>{v.n}</td><td>{v.loss_interval.map(x=>number(x*100,2)).join(' ～ ')}</td><td>{ni[v.status]||v.status}</td></tr>)}
      </tbody></table></>}
    {doc.flow_boundary&&<details><summary>B35/H25 写入的时间边界</summary>
      <p>单独移除问题前 B35/H25 后，{doc.flow_boundary.local_residual_changed} 条的局部残差改变。后续回答的 logits 逐位一致检验{doc.flow_boundary.all_later_query_logits_bitwise_equal?'通过':'未通过'}，全层前缀 KV 核验{doc.flow_boundary.all_prefix_KV_bitwise_equal?'通过':'未通过'}。</p>
      <pre className="v8-artifact-json">{JSON.stringify(doc.flow_boundary,null,2)}</pre></details>}
  </>;
}

function Evidence({doc}){
  const [part,setPart]=useState(doc.cases?.generated?.n?'generated':'natural'),[index,setIndex]=useState(0);
  const list=doc.cases?.[part],start=list?.chunks[Math.floor(index/(list?.chunk_size||64))];
  const q=useApi(artifact(start));
  const row=q.data?.cases[index%(list?.chunk_size||64)];
  return <>
    <div className="v6-toolbar"><label>证据集合 <select aria-label="上游证据集合" value={part} onChange={e=>{setPart(e.target.value);setIndex(0)}}>
      {[['natural','自然文档与保真'],['controlled','原版长文本与行为门'],['revision','修订材料与留出'],['generated','干预剂量与逐题反例'],['capability','能力逐题结果']].filter(([key])=>doc.cases?.[key]?.n).map(([key,label])=><option key={key} value={key}>{label}</option>)}
    </select></label><label>索引 <input aria-label="上游证据索引" type="number" min="0" max={(list?.n||1)-1} value={index} onChange={e=>setIndex(bounded(e.target.value,(list?.n||1)-1))}/></label></div>
    {q.error&&<Empty>{q.error}</Empty>}
    {row&&<pre className="v8-artifact-json">{JSON.stringify(row,null,2)}</pre>}
    {doc.counterexamples?.length>0&&<details><summary>生成反例索引（{doc.counterexamples.length} 条）</summary>
      {doc.counterexamples.slice(0,24).map(r=><button className="fw-tbtn" key={r.index} onClick={()=>{setPart('generated');setIndex(r.index)}}>样本 {r.index} · {r.expected} → {r.edited||'空'}</button>)}
    </details>}
    {doc.behavior_errors?.length>0&&<details><summary>原版行为失败（{doc.behavior_errors.length} 条）</summary>
      {doc.behavior_errors.map(r=><button className="fw-tbtn" key={r.index} onClick={()=>{setPart('controlled');setIndex(r.index)}}>样本 {r.index} · {r.expected} → {r.generated||r.word||'空'}</button>)}
    </details>}
    <details><summary>预注册、内容哈希与原始文件</summary><pre className="v8-artifact-json">{JSON.stringify({prereg:doc.prereg,refs:doc.refs},null,2)}</pre></details>
  </>;
}

export default function MlpFunctionalReport({doc}){
  const [tab,setTab]=useState(()=>{
    const value=new URLSearchParams(window.location.search).get('uview');
    return tabs.some(([key])=>key===value)?value:'dictionary';
  });
  const [block,setBlock]=useState(()=>fromQuery('ublock',35,35));
  const [head,setHead]=useState(()=>fromQuery('uhead',0,31));
  return <section aria-label="行为保真字典与问题前上游路径">
    <p><strong>行为保真字典 · 问题前属性路径</strong></p>
    <p className="v8-note">{doc.run_id} · 本地 Qwen3-4B BF16 · {doc.state==='complete'?'本轮已归档':'研究进行中'}</p>
    <div className="v6-toolbar">{tabs.map(([key,label])=><button className="fw-tbtn" key={key} aria-pressed={tab===key} onClick={()=>setTab(key)}>{label}</button>)}</div>
    {tab==='dictionary'&&<Fidelity doc={doc}/>}
    {tab==='paths'&&<Paths doc={doc} choose={(b,h)=>{setBlock(b);setHead(h);setTab('native')}}/>}
    {tab==='native'&&<Native count={doc.native_count} revision={doc.revision} block={block} setBlock={setBlock} head={head} setHead={setHead}/>}
    {tab==='causal'&&<Causal doc={doc}/>}
    {tab==='evidence'&&<Evidence doc={doc}/>}
  </section>;
}
