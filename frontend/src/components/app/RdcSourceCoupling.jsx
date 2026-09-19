import {useEffect,useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
const API=(import.meta.env.VITE_RDC_CONSTRUCTION_API||'http://127.0.0.1:5004/api/rdc-construction')+'/questions';
const fmt=x=>x==null?'未定义':Number(x).toPrecision(6);
const columns=[['passage_mass_fraction','篇章注意力比例'],['question_mass_fraction','问题比例'],['other_mass_fraction','其余提示比例'],
  ['passage_conditional_effective_source_fraction','篇章有效来源 / token数'],['passage_coupling_L2','篇章配对项 L2'],['passage_permutation_delta_L2','置换读取变化 L2'],
  ['Cauchy_Schwarz_permutation_bound','Cauchy–Schwarz 上界'],['FP64_read_minus_native_BF16_preO_L2','FP64 − 原生 pre-O L2']];
export default function SourceCoupling({model,refresh}){
  const [cohort,setCohort]=useState('drop'),[split,setSplit]=useState('diagnostic'),[coordinate,setCoordinate]=useState('permuted_minus_native_read_mean_square');
  const [response,setResponse]=useState(null),[error,setError]=useState(null);const id=JSON.stringify([model,refresh,cohort,split,coordinate]);
  useEffect(()=>{let active=true;fetch(API+'/source-coupling?'+new URLSearchParams({model,cohort,split,coordinate})).then(async r=>{const v=await r.json();if(!r.ok)throw new Error(v.detail||r.statusText);if(active){setResponse({id,value:v});setError(null);}}).catch(e=>active&&setError({id,text:e.message}));return()=>{active=false;};},[id]);
  const data=response?.id===id?response.value:null,summary=data?.summary;
  return <div id="natural-source-coupling"><h3>来源配对对照有多强：全部注意力头与坐标</h3>
    <p>事后新增观察：既定置换只打乱篇章位置的 value，保留 query、key、注意力权重以及问题／其余提示的 value。这里检查真实扰动大小，不把“预测没变”直接解释为来源不重要。其余位置也可能承载早层已处理的篇章信息。</p>
    <div className="prefix-controls"><label>来源配对语料<select aria-label="来源配对语料" value={cohort} onChange={e=>setCohort(e.target.value)}>{['drop','quoref'].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>来源配对划分<select aria-label="来源配对划分" value={split} onChange={e=>setSplit(e.target.value)}>{['diagnostic','validation','train'].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>来源配对坐标量<select aria-label="来源配对坐标量" value={coordinate} onChange={e=>setCoordinate(e.target.value)}><option value="permuted_minus_native_read_mean_square">置换 − 原生：逐坐标均方变化</option><option value="native_read_mean">原生写回逐坐标均值</option><option value="permuted_read_mean">置换写回逐坐标均值</option></select></label></div>
    {error?.id===id&&<p role="alert">{error.text}</p>}
    {!data?.complete&&<p role="status">{data?.status||'正在读取当前来源配对分析…'}</p>}
    {data?.complete&&<><p>{summary.contexts} 个完整篇章，{summary.questions} 题；下表保留全部 {data.head_values.length} 个原生头，顺序不变。不是按显著性或幅度选头。</p>
      <div className="prefix-table"><table data-testid="source-coupling-summary"><thead><tr><th>全部头平均篇章比例</th><th>问题比例</th><th>其余提示比例</th><th>原生写回均方</th><th>置换写回均方变化</th><th>逐题相对平方 L2 变化均值</th></tr></thead><tbody><tr>
        {['passage_mass_fraction','question_mass_fraction','other_mass_fraction'].map(k=><td key={k}>{fmt(summary.head_equal_means[k])}</td>)}
        {['native_read_mean_square','native_permuted_read_MSE','native_permuted_read_relative_squared_L2'].map(k=><td key={k}>{fmt(summary.postO_question_means[k])}</td>)}
      </tr></tbody></table></div>
      <h4>相对问题差异的扰动，而不只相对整体能量</h4>
      <p>先减去每篇四题共同均值，再检查置换改变了多少问题差异。使用原来已冻结的训练均值／尺度，不重新拟合。查询核仅为完整 [H12; 读取] 对全部 768 个训练问题的内积，不是完整预测器。</p>
      {data.relative_strength?.complete?<div className="prefix-table"><table data-testid="source-coupling-relative"><thead><tr><th>实际坐标／核空间</th><th>总置换均方</th><th>共同篇章变化</th><th>原问题变化能量</th><th>置换问题变化</th><th>相对问题变化能量</th><th>汇总问题响应余弦</th></tr></thead><tbody>{data.relative_strength.summaries.map(r=><tr key={r.space}><td>{r.space}</td>{['permutation_MSE','common_context_change_MSE','within_native_mean_square','within_permutation_MSE','relative_question_permutation_energy','pooled_question_response_cosine'].map(k=><td key={k}>{fmt(r.values[k])}</td>)}</tr>)}</tbody></table></div>:<p role="status">相对问题差异与冻结核检查尚未完成。</p>}
      <p>篇章读取 = 注意力总量 × 平均 value + 权重与 value 的配对项。以上分解用真实 BF16 字解码后的 FP64 加和；不与原生 BF16 舍入混同。有效来源数衡量权重集中度，不是语义概念数；范数不是因果贡献。</p>
      <div className="prefix-table" style={{maxHeight:520,overflow:'auto'}}><table data-testid="source-coupling-heads"><thead><tr><th>原生头</th>{columns.map(([k,t])=><th key={k}>{t}</th>)}</tr></thead><tbody>{data.head_values.map((row,i)=><tr key={i}><td>{i}</td>{columns.map(([k])=><td key={k}>{fmt(row[data.head_columns.indexOf(k)])}</td>)}</tr>)}</tbody></table></div>
      <p>{data.field.axes}</p><Field key={id} data={data.field}/>
      <details><summary>全部数值、有效样本数、分解定义与来源身份</summary><pre>{JSON.stringify(data,null,2)}</pre></details>
    </>}
  </div>;
}
