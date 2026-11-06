import {useState} from 'react';
import {useApi,API_BASE} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import MechanismScene,{NativeArray} from './MechanismScene.jsx';
const axes={parameter_locator:'参数可定位',parameter_projection_checked:'真实参数投影已核验',natural_statistics:'有自然统计',controlled_response_recorded:'有受控响应',natural_context_prediction_attempted:'做过前缀预测',natural_context_prediction_passed:'前缀预测通过',heldout_answer_direction:'限定答案方向复现',code_response_question:'中性答码响应：问题后',code_response_assistant:'中性答码响应：assistant边界',code_function_eligible:'中性答码功能解释过门',natural_dictionary_association_attempted:'自然字典关联已评估',natural_dictionary_association_repeated:'字典关联跨语料复现（描述）',individual_causal_mechanism:'个体机制闭合',graph_predictive_closed:'图谱预测闭合',semantic_monosemanticity_confirmed:'语义单义性证实'};
const num=v=>Number.isFinite(v)?v.toPrecision(5):'—';
function NaturalExamples({block,unit}){
  const q=useApi(`/api/neurons/unit?model_key=6be1034f0b55a6df-bf16&block=${block}&unit=${unit}&n_poles=3`);
  if(q.error)return <Empty>{q.error}；统计账本仍可查，未加载的自然原片不生成替代文本。</Empty>;
  return <details open><summary>当前单元的自然原片、历史检验及反例</summary><pre className="v8-artifact-json">{JSON.stringify(q.data,null,2)}</pre></details>;
}
export default function MlpInventoryReport({doc:d}){
  const [block,setBlock]=useState(0),[unit,setUnit]=useState(0),[metric,setMetric]=useState('A_RMS'),[natural,setNatural]=useState(false);
  const r=useApi(d.layer_refs?.[String(block)]?`/api/research-library/artifacts/${d.layer_refs[String(block)]}`:null),layer=r.data;
  const values=layer?(layer.natural_statistics[metric]||layer.flags[metric]?.map(Number)):undefined;
  const evidence=layer?.case_evidence_cards?.[`${block}:${unit}`],attempt=layer?.natural_attempt_cards?.[`${block}:${unit}`];
  return <section aria-label="全部MLP单元逆向证据账本"><p><strong>{d.total_units.toLocaleString()} 个原生MLP单元 · 36层完整索引</strong></p><p className="v8-note">每个单元有稳定身份、自然统计和证据状态。覆盖完整不代表功能已解释；0表示本账本没有该类证据，不表示该单元没有作用。</p>
    <details><summary>全量覆盖与研究缺口</summary><table className="v6-table"><thead><tr><th>证据维度</th><th>单元数</th></tr></thead><tbody>{Object.entries(d.evidence_counts).map(([k,n])=><tr key={k}><td>{axes[k]||k}</td><td>{n.toLocaleString()}</td></tr>)}</tbody></table><p className="v8-note">自然统计实际计入 {d.raw_census_tokens.toLocaleString()} token；不是把20M预算写成实测量。限定答案方向不等于完整语义或控制开关。</p></details>
    <div className="v6-toolbar"><label>层块 <input aria-label="全MLP账本层块" type="number" min="0" max="35" value={block} onChange={e=>setBlock(Math.max(0,Math.min(35,Number(e.target.value)||0)))}/></label><label>单元 <input aria-label="全MLP账本单元" type="number" min="0" max="9727" value={unit} onChange={e=>setUnit(Math.max(0,Math.min(9727,Number(e.target.value)||0)))}/></label><select aria-label="全MLP账本着色" value={metric} onChange={e=>setMetric(e.target.value)}>{[['A_RMS','自然激活RMS'],['A_mean','自然激活均值'],['A_max','自然激活最大值'],['A_min','自然激活最小值'],...Object.entries(axes)].map(([k,t])=><option key={k} value={k}>{t}</option>)}</select></div>
    {r.error?<Empty>{r.error}</Empty>:!layer?<p>读取当前层全部9728单元…</p>:<><MechanismScene manifest={d.dimensions} block={block} head={0} unit={unit} activation={values} caption={`全量账本 · ${metric} · 语料统计/证据标记，不是当前样本激活`} onBlock={setBlock} onUnit={setUnit} onHead={()=>{}}/><p className="v8-note">Embedding和Head仅为经核对的结构位置；本图的MLP着色来自原始语料统计或明确证据标记，没有用标签生成激活。所有36层可选，当前层全部9728单元不截断。</p><NativeArray values={values} selected={unit} onPick={setUnit} label="当前层全部原生MLP账本数值"/>
      <p><strong>B{block}/U{unit}</strong> · {metric}={num(values?.[unit])} · RMS={num(layer.natural_statistics.A_RMS[unit])} · 范围[{num(layer.natural_statistics.A_min[unit])}, {num(layer.natural_statistics.A_max[unit])}]</p>
      <p className="v8-note">{layer.unit_ids[unit]}</p><table className="v6-table"><tbody>{Object.entries(axes).map(([k,t])=><tr key={k}><td>{t}</td><td>{layer.flags[k]?.[unit]?'✓ 有相应记录':'— 无该类证据'}</td></tr>)}</tbody></table><p className="v8-note">中性答码响应的行为门未通过，不能升级为功能解释；字典关联使用预先固定的描述阈值，不是语义或因果证据。</p>{d.research_refs&&<p><a href="?lens=spatial&view=research&run=MLP-ATLAS-CODE-20261104-v2">全量答码研究与反例</a>{' · '}<a href="?lens=spatial&view=research&run=MLP-ATLAS-DICTIONARY-20261104-v1">全36层字典、重建与自然原片</a></p>}
      <p><a href={`?lens=spatial&view=parameters&pmodel=${d.local_model_id}&pblock=${block}&pkind=mlp&punit=${unit}`}>读取这个单元的真实参数</a>{' · '}<button className="fw-tbtn" onClick={()=>setNatural(v=>!v)}>{natural?'收起自然原片':'读取自然原片与历史检验'}</button></p>
      {d.research_refs?.fidelity_v2_report&&<p><a href={`?lens=spatial&view=research&run=MLP-FIDELITY-20261104-v2&fview=dictionary&fblock=${block}&funit=${unit}`}>查看本层新版字典与已登记关联</a> · <a href="?lens=spatial&view=research&run=MLP-FIDELITY-20261104-v2&fview=causal">查看生成干预、反例和能力检验</a></p>}
      {natural&&<NaturalExamples key={`${block}:${unit}`} block={block} unit={unit}/>}
      {(evidence||attempt)&&<details><summary>已登记研究卡与失败范围</summary>{[evidence,attempt].filter(Boolean).map(sha=><p key={sha}><a href={`${API_BASE}/api/research-library/artifacts/${sha}`} target="_blank" rel="noreferrer">完整原始研究卡 · {sha.slice(0,16)}</a></p>)}</details>}
      <details><summary>全部读出标签、符号分区和来源</summary><pre className="v8-artifact-json">{JSON.stringify({readout_token_ids:layer.natural_statistics.direct_readout_top_token_ids[unit],bottom_token_ids:layer.natural_statistics.direct_readout_bottom_token_ids[unit],sign_counts:layer.natural_statistics.GV_sign_counts[unit],source:layer.raw_census_layer_sha256,scope:layer.scope},null,2)}</pre></details>
    </>}
    <details><summary>完成标准与下一大阶段</summary>{d.next_stages&&<table className="v6-table"><thead><tr><th>下一阶段</th><th>验收要求</th><th>状态</th></tr></thead><tbody>{d.next_stages.map(row=><tr key={row.id}><td>{row.title}</td><td>{row.acceptance}</td><td>研究计划，尚未完成</td></tr>)}</tbody></table>}<pre className="v8-artifact-json">{JSON.stringify({completion:d.definition_of_completion,next_run:d.next_run_id,limitations:d.limitations,plan_sha256:d.research_plan_sha256},null,2)}</pre></details>
  </section>;
}
