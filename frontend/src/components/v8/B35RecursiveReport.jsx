import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';

const pct = value => Number.isFinite(value) ? (100*value).toFixed(1)+'%' : '—';
const num = value => Number.isFinite(value) ? value.toPrecision(4) : '—';
const artifact = hash => /^[0-9a-f]{64}$/.test(hash||'') ? '/api/research-library/artifacts/'+hash : null;
const parts = {discovery:'三步发现集',confirm:'三步确认集',lockbox:'六步留出集',pilot:'第二版试跑',pilot_v1:'第一版失败试跑'};
const stepLabels = ['箭头表','叙述式表'];

export function useB35Features(archive, snapshot, active) {
  const study = active && archive?.studies?.find(s => s.model_snapshot===snapshot && s.stages?.b35_features?.block===35);
  const stage = study?.stages?.b35_features;
  const index = useApi(artifact(stage?.index_sha256));
  return {study, stage, index, data:index.data?.schema==='b35_recursive_feature_arrays.v1' &&
      index.data.model_snapshot===snapshot && index.data.block===35 ? index.data : null};
}

export function useB35Unit(index, unit) {
  const size = index?.chunk_size;
  const valid = Number.isInteger(unit) && unit>=0 && unit<index?.unit_count && size>0;
  const row = useApi(valid ? artifact(index.chunks[Math.floor(unit/size)]) : null);
  const data = valid && row.data?.schema==='b35_recursive_units.v1' && row.data.model_snapshot===index.model_snapshot &&
      row.data.block===35 && row.data.start===Math.floor(unit/size)*size ? row.data.units[unit%size] : null;
  return {...row, data:data?.unit===unit ? data : null};
}

export function B35UnitFeatures({row,index,recursive=false,onUpstream}) {
  if(row.error)return <Empty>{row.error}</Empty>;
  if(!row.data)return <p className="v8-note">读取 B35 单元归档…</p>;
  const u = row.data;
  const cs = u.confirm, ls = u.lockbox;
  if(recursive)return <section aria-label="选中单元的递归泛化数据">
    <h4>B35/U{u.unit} · 短链到长链</h4>
    <p>{u.behavior_eligible ? '两套留出均复现发现集方向' : u.replicated ? '方向重复，整体行为门未通过' : '未满足双留出复现门'}</p>
    <table className="v6-table"><thead><tr><th>测量</th><th>短链确认</th><th>长链第 4–6 步</th></tr></thead><tbody>
      <tr><th>同方向文档</th><td>{cs.same_sign}/{cs.n}</td><td>{ls.same_sign}/{ls.n}</td></tr>
      <tr><th>定向一阶间隔贡献</th><td>{num(cs.local_mean)}</td><td>{num(ls.local_mean)}</td></tr>
      <tr><th>相对均值基线的预测增益</th><td>{pct(cs.prediction.gain_vs_mean)}</td><td>{pct(ls.prediction.gain_vs_mean)}</td></tr>
      <tr><th>相对前一步激活的预测增益</th><td>{pct(cs.prediction.gain_vs_previous)}</td><td>{pct(ls.prediction.gain_vs_previous)}</td></tr>
      <tr><th>自身历史错误时的激活 RMS 偏差</th><td>{num(cs.history_error_rms_delta)}</td><td>{num(ls.history_error_rms_delta)}</td></tr>
    </tbody></table>
    <p className="v8-note">贡献在正确历史下估计。预测使用同一步真实 B34 激活。权重余弦仅用于选择上游候选。</p>
    <div className="v6-toolbar">
      {['gate','value'].map(key=><div key={key}><b>{key==='gate'?'Gate 来源':'Value 来源'}</b>{u.upstream[key].map(p=>
        <button className="fw-tbtn" key={p.unit} onClick={()=>onUpstream?.(p.unit)} disabled={!onUpstream}>B34/U{p.unit} · {num(p.cosine)}</button>)}</div>)}
    </div>
  </section>;
  return <section aria-label="B35 原生神经元特征">
    <h4>B35/U{u.unit} · MLP 神经元特征</h4>
    <p className="v8-note">{index.natural.tokens.toLocaleString()} 个自然语料 token · SwiGLU 中间激活</p>
    <p>均值 {num(u.mean)} · 标准差 {num(u.std)} · 范围 {num(u.min)} ～ {num(u.max)}</p>
    <p className="v8-note">Gate/Value 符号比例 {index.natural.regime_order.map((k,i)=>k+' '+pct(u.regime_fraction[i])).join(' · ')}</p>
    {['max','min'].map(key=>{const p=u.poles[key];return <p key={key}>{key==='max'?'最大':'最小'} {num(p.a)} · {p.source} · <span>{p.left}<mark>{p.hit}</mark>{p.right}</span></p>})}
    <ResearchDisclosure title="输出列读出与完整单元数据">
      <p className="v8-note">下列词元由输出权重近似读出。语义标签需要独立复现。</p>
      {['promotes','suppresses'].map(key=><p key={key}>{key==='promotes'?'正向':'反向'} {u.readout[key].map(t=>JSON.stringify(t.text)+' #'+t.id).join(' · ')}</p>)}
      <pre className="v8-artifact-json">{JSON.stringify(u,null,2)}</pre>
    </ResearchDisclosure>
  </section>;
}

function CaseExplorer({doc}) {
  const [part,setPart]=useState('lockbox'),[item,setItem]=useState(doc.error_cases.lockbox[0]??0);
  const list=doc.case_indexes[part];
  const api=useApi(artifact(list.chunks[Math.floor(item/list.chunk_size)]));
  const row=api.data?.cases?.[item%list.chunk_size];
  return <><div className="v6-toolbar"><label>样本组 <select aria-label="B35 样本组" value={part} onChange={e=>{setPart(e.target.value);setItem(doc.error_cases[e.target.value]?.[0]??0)}}>
    {Object.keys(parts).map(k=><option key={k} value={k}>{parts[k]}</option>)}</select></label>
    <label>索引 <input aria-label="B35 样本索引" type="number" min="0" max={list.n-1} value={item}
      onChange={e=>setItem(Math.max(0,Math.min(list.n-1,Math.trunc(Number(e.target.value)||0))))}/></label>
    {doc.error_cases[part]?.length>0&&<label>反例 <select aria-label="B35 生成反例" value={doc.error_cases[part].includes(item)?item:''} onChange={e=>setItem(Number(e.target.value))}>
      <option value="" disabled>选择未完成样本</option>{doc.error_cases[part].map(n=><option key={n} value={n}>#{n}</option>)}</select></label>}</div>
    {api.error?<Empty>{api.error}</Empty>:row?<><p>{row.id} · {row.source.title} · 预期 {row.gold.join(' ')} · 实际 {row.outcome.free.map(t=>t.text).join('')}</p>
      <details><summary>逐步输出、原始 token 和语料来源</summary><pre className="v8-artifact-json">{JSON.stringify(row,null,2)}</pre></details></>:<p>读取中心逐题数据…</p>}</>;
}

export function B35DepthResults({doc}) {
  return <section aria-label="B35 递归泛化">
    <p><strong>{doc.eligible_units.toLocaleString()} / {doc.d_ff.toLocaleString()}</strong> 个单元达到跨深度贡献方向复现门 · 整体行为门{doc.behavior_gate.passed?'通过':'未通过'}</p>
    <p className="v8-note">只在三步发现集选方向。确认集使用新文档和新循环。六步留出采用第 4–6 步，逐步记录金标准历史与自身生成历史。</p>
    <table className="v6-table"><thead><tr><th>划分</th><th>句式</th><th>步</th><th>教师强制当步</th><th>自由生成整段前缀</th></tr></thead><tbody>
      {['confirm','lockbox'].flatMap(part=>doc.steps[part].map(r=><tr key={part+':'+r.frame+':'+r.step}>
        <td>{parts[part]}</td><td>{stepLabels[r.frame]}</td><td>{r.step}</td>
        <td>{r.teacher_token_correct}/{r.n}</td><td>{r.free_prefix_correct}/{r.n}</td></tr>))}
    </tbody></table>
    <p className="v8-note">独立确认与六步留出的置乱通过数 {doc.shuffle_passed_units.confirm} / {doc.shuffle_passed_units.lockbox}。B34 单候选对两种基线均改善的单元 {doc.upstream_improves_both.toLocaleString()}。</p>
    <ResearchDisclosure title="逐题记录与反例"><CaseExplorer doc={doc}/></ResearchDisclosure>
  </section>;
}

export function B35LayerResults({sha}) {
  const report=useApi(artifact(sha));
  if(report.error)return <Empty>{report.error}</Empty>;
  if(report.data?.schema!=='b35_recursive_report.v1')return <p className="v8-note">读取递归泛化归档…</p>;
  return <B35DepthResults doc={report.data}/>;
}

export default function B35RecursiveReport({doc}) {
  const [unit,setUnit]=useState(0);
  const index=useApi(artifact(doc.feature_index_sha256));
  const row=useB35Unit(index.data,unit);
  const deepLink = '?lens=spatial&view=parameters&ptab=neurons&sblock=36&smodule=mlp&sindex='+unit+'&snapshot='+doc.model_snapshot;
  return <section aria-label="Qwen3-4B 最后一层研究">
    <h3>Qwen3-4B B35 · {doc.d_ff.toLocaleString()} 个原生神经元</h3>
    <a href={deepLink}>在模型结构 3D 中打开该单元</a>
    <B35DepthResults doc={doc}/>
    <label>单元索引 <input aria-label="B35 研究单元索引" type="number" min={0} max={doc.d_ff-1} value={unit}
      onChange={e=>setUnit(Math.max(0,Math.min(doc.d_ff-1,Math.trunc(Number(e.target.value)||0))))}/></label>
    {index.data&&<><B35UnitFeatures row={row} index={index.data}/><B35UnitFeatures row={row} index={index.data} recursive/></>}
    <ResearchDisclosure title="适用范围">{doc.limitations.map(text=><p key={text}>{text}</p>)}</ResearchDisclosure>
  </section>;
}
