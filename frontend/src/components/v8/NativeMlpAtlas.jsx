import { useEffect, useMemo, useState } from 'react';
import { API_BASE, useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { NativeArray } from './MechanismScene.jsx';
import { TailScene } from './MlpTailAtlas.jsx';

const BASE='/api/research/native-mlp';
const BLOCKS=[18,19,20,21,22,23,24,25,26,27,28];
const fmt=v=>Number.isFinite(v)?v.toPrecision(5):'未检验';
const initial=(key,fallback,min,max)=>{if(typeof window==='undefined')return fallback;const n=Number(new URLSearchParams(window.location.search).get(key));return Number.isInteger(n)&&n>=min&&n<=max?n:fallback};
const apiUrl=(path,data)=>`${BASE}${path}${path.includes('?')?'&':'?'}snapshot=${data.model_snapshot}&run=${encodeURIComponent(data.run_id)}`;

function Dose({rows}){
  const points=rows.filter(r=>Number.isFinite(r.mean_nll_change_matched));
  if(!points.length)return <p className="v8-note">本单元未完成有效匹配的写入干预；观察档案仍可完整查询。</p>;
  const min=Math.min(0,...points.map(r=>r.mean_nll_change_matched)),max=Math.max(.00001,...points.map(r=>r.mean_nll_change_matched));
  const y=v=>170-(v-min)/(max-min)*140;
  return <><svg viewBox="0 0 680 215" role="img" aria-label="原生真实单元剂量曲线" style={{width:'100%',maxWidth:850}}>
    <line x1="70" x2="640" y1={y(0)} y2={y(0)} stroke="#cbd5e1"/>
    {[0,.5,1,2].map(v=><text key={v} x={70+v*275} y="195">{v}</text>)}
    {[min,max].map((v,i)=><text key={i} x="0" y={y(v)}>{fmt(v)}</text>)}
    {['native','norm_random'].map((condition,i)=><polyline key={condition} fill="none" stroke={i?'#de793c':'#3279bd'} strokeWidth="2" points={[...points.filter(r=>r.condition===condition),{dose:1,mean_nll_change_matched:0}].sort((a,b)=>a.dose-b.dose).map(r=>`${70+r.dose*275},${y(r.mean_nll_change_matched)}`).join(' ')}/>)}
  </svg><p className="v8-note">横轴：原生列写入倍数（1为实测原始回放基线）；纵轴：下一词预测损失变化。蓝为原生列，橙为等范数方向。每个剂量分别取两条件的共同合格样本，各剂量分母可不同；连线用于读图，不是拟合模型或语义控制成功率。</p>
  <table className="v6-table"><thead><tr><th>剂量 / 条件</th><th>匹配/总数</th><th>全部样本KL</th><th>全部样本首词变化</th></tr></thead><tbody>{rows.map(r=><tr key={r.id}><td>{r.dose} / {r.condition==='native'?'原生列':'等范数'}</td><td>{r.n_matched}/{r.n_total}</td><td>{fmt(r.mean_kl_all)}</td><td>{r.argmax_changes_all}</td></tr>)}</tbody></table></>;
}

export default function NativeMlpAtlas({doc,hideScene=false,selection,onSelect}){
  if(!doc)return <Empty>尚无已审计的原生神经元档案。</Empty>;
  if(doc.schema!=='native_mlp_atlas_report.v1'||!doc.run_id||!doc.model_snapshot||!Number.isInteger(doc.width)||!doc.config?.hidden_size)return <Empty>研究报告身份或模型维度缺失，无法读取原生档案。</Empty>;
  return <Atlas key={doc.run_id} data={doc} hideScene={hideScene} selection={selection} onSelect={onSelect}/>;
}

function Atlas({data,hideScene,selection,onSelect}){
  const blocks=data.blocks||BLOCKS;
  const first=blocks[0],last=blocks[blocks.length-1],width=data.width,dim=data.config.hidden_size;
  const url=path=>apiUrl(path,data);
  const [localBlock,setBlock]=useState(()=>initial('native_block',first,first,last));
  const [localUnit,setUnit]=useState(()=>initial('native_unit',0,0,width-1));
  const requestedBlock=selection?.block??localBlock;
  const requestedUnit=selection?.unit??localUnit;
  const block=blocks.includes(requestedBlock)?requestedBlock:blocks[0],unit=Number.isInteger(requestedUnit)?Math.max(0,Math.min(width-1,requestedUnit)):0;
  const [tab,setTab]=useState('response'),[metric,setMetric]=useState('A_std');
  const [field,setField]=useState('A'),[split,setSplit]=useState('holdout'),[pole,setPole]=useState('positive');
  const [kind,setKind]=useState('rho-holdout'),[direction,setDirection]=useState('outgoing'),[allEdges,setAllEdges]=useState(false);
  const [weight,setWeight]=useState('gate'),[coordinate,setCoordinate]=useState(0);
  const [rawOpen,setRawOpen]=useState(false),[rawStage,setRawStage]=useState('natural'),[rawSample,setRawSample]=useState(0);
  const a=useApi(hideScene?null:url('/layers/18'));
  const b=useApi(hideScene?null:url('/layers/19'));
  const c=useApi(hideScene?null:url('/layers/20'));
  const d=useApi(hideScene?null:url('/layers/21'));
  const e=useApi(hideScene?null:url('/layers/22'));
  const f=useApi(hideScene?null:url('/layers/23'));
  const g=useApi(hideScene?null:url('/layers/24'));
  const h=useApi(hideScene?null:url('/layers/25'));
  const i=useApi(hideScene?null:url('/layers/26'));
  const j=useApi(hideScene?null:url('/layers/27'));
  const k=useApi(hideScene?null:url('/layers/28'));
  const layers=useMemo(()=>({18:a.data,19:b.data,20:c.data,21:d.data,22:e.data,23:f.data,24:g.data,25:h.data,26:i.data,27:j.data,28:k.data}),[a.data,b.data,c.data,d.data,e.data,f.data,g.data,h.data,i.data,j.data,k.data]);
  const actualDirection=block===first?'outgoing':block===last?'incoming':direction;
  const detail=useApi(url(`/units/${block}/${unit}?field=${field}&split=${split}&pole=${pole}`));
  const relation=useApi(tab==='relations'?url(`/relations/${block}/${unit}?direction=${actualDirection}&kind=${kind}`):null);
  const raw=useApi(tab==='evidence'&&rawOpen?url(`/interventions/${rawStage}/${block}/${unit}?sample=${rawSample}`):null);
  const selected=detail.data,layerError=[a,b,c,d,e,f,g,h,i,j,k].find(x=>x.error)?.error;
  useEffect(()=>{const next=new URL(window.location.href);next.searchParams.set('native_block',block);next.searchParams.set('native_unit',unit);window.history.replaceState(null,'',next)},[block,unit]);
  const pick=(layer,index)=>{setBlock(layer);setUnit(index);onSelect?.(layer,index)};
  return <section aria-label="原生全量MLP研究">
    <h3>{data.model_name} · B{first}–B{last} 神经元档案</h3>
    <p className="v8-note">{(blocks.length*width).toLocaleString()} 个原生单元 · BF16 · {data.documents?.toLocaleString()} 篇 1,024-token 长文本。模型共{data.config.num_hidden_layers}层，这里是零基 B{first}–B{last}。全量观察不等于全部语义已经破解。</p>
    {data.regimes&&<details><summary>11层覆盖与研究汇总</summary><p className="v8-note">下表双负份额先对每个单元计算，再在层内取均值。G/V/A候选是三个字段的筛查计数，不能相加称为已破解神经元。</p><table className="v6-table"><thead><tr><th>层</th><th>双负 |A| 份额</th><th>SiLU负导数比例</th><th>响应候选 G / V / A</th></tr></thead><tbody>{data.regimes.map(r=>{const c=data.prediction?.layers?.find(x=>x.block===r.block)?.counts;return <tr key={r.block}><td>B{r.block}</td><td>{fmt(r.holdout_DN_mass_mean)}</td><td>{fmt(r.holdout_negative_silu_derivative_mean)}</td><td>{c?`${c.G} / ${c.V} / ${c.A}`:'未检验'}</td></tr>})}</tbody></table><p>已做有限写入干预：{data.total_intervention_units}个单元；正式语义确认：0。响应预测和单父递推的结果与反例保留在验证视图。</p></details>}
    <div className="v6-toolbar">
      <label>层 <select aria-label="原生研究层" value={block} onChange={e=>pick(Number(e.target.value),unit)}>{blocks.map(x=><option key={x} value={x}>B{x}（第{x+1}层）</option>)}</select></label>
      <label>单元 <input aria-label="原生单元" type="number" min="0" max={width-1} value={unit} onChange={e=>pick(block,Math.max(0,Math.min(width-1,Number(e.target.value)||0)))}/></label>
      <label>着色 <select aria-label="原生着色特性" value={metric} onChange={e=>setMetric(e.target.value)}>{[['A_std','实际乘积变化'],['A_mean','实际乘积均值'],['G_std','gate变化'],['V_std','value变化'],['sign_0','正gate × 正value'],['sign_3','负gate × 负value'],...(data.regimes?[['double_negative_mass','双负绝对激活份额'],['negative_silu_derivative','SiLU负导数区间']]:[])].map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
    </div>
    {!hideScene&&(layerError?<Empty>层数据缺失：{layerError}</Empty>:blocks.every(x=>layers[x])?<TailScene blocks={blocks} width={width} modelName={data.model_name} layers={layers} metric={metric} block={block} unit={unit} relation={tab==='relations'?relation.data:null} allEdges={allEdges} onPick={pick}/>:<p>读取 {blocks.filter(x=>layers[x]).length}/{blocks.length} 层完整阵列…</p>)}
    <p className="v8-note">蓝为正、橙为负；按原生编号排列。每层色标按当前数组最大绝对值缩放，不能据颜色深浅直接比较不同层。</p>
    <div className="v6-toolbar" role="tablist" aria-label="原生单元研究视图">{[['response','响应与语境'],['parameters','完整参数'],['relations','上下游关联'],['evidence','验证与反例']].map(([key,label])=><button key={key} className="fw-tbtn" role="tab" aria-selected={tab===key} onClick={()=>setTab(key)}>{label}</button>)}</div>
    {detail.error?<Empty>{detail.error}</Empty>:!selected?<p>读取 B{block}/U{unit} 的原始证据…</p>:<>
      {tab==='response'&&<>
        <p><strong>B{block}/U{unit}</strong> · {selected.candidate.description}</p>
        <p className="v8-note">语境词由发现集提取，仅供检索；不是神经元的正式语义名称。已验证语义：未确认。</p>
        <details><summary>原生写出列的词表读出线索</summary><p className="v8-note">直接连接词表的近似，忽略后续层与上下文归一化；不等于这个单元的意义或实际生成。</p>{['positive','negative'].map(side=><p key={side}>{side==='positive'?'正端':'负端'}：{selected.parameter_readout?.[side]?.map(x=>`${JSON.stringify(x.token)} (${fmt(x.value)})`).join(' / ')}</p>)}</details>
        <div className="v6-toolbar"><label>分量 <select value={field} onChange={e=>setField(e.target.value)}><option value="G">gate 预激活 G</option><option value="V">value V</option><option value="A">真实乘积 A</option></select></label>
          <label>材料 <select aria-label="原生材料划分" value={split} onChange={e=>setSplit(e.target.value)}><option value="discovery">发现集</option><option value="confirm">确认分片</option><option value="holdout">留出分片</option></select></label>
          <label>响应端 <select value={pole} onChange={e=>setPole(e.target.value)}><option value="positive">高值端</option><option value="negative">低值端</option></select></label></div>
        <table className="v6-table"><thead><tr><th>分量</th><th>均值</th><th>标准差</th><th>最小</th><th>最大</th></tr></thead><tbody>{Object.entries(selected.statistics).map(([k,v])=><tr key={k}><td>{k}</td>{['mean','std','min','max'].map(f=><td key={f}>{fmt(v[f])}</td>)}</tr>)}</tbody></table>
        {selected.regimes&&<><p>当前{split==='discovery'?'发现集':split==='confirm'?'确认分片':'留出分片'}：双负绝对激活份额 {fmt(selected.regimes[`${split}_quadrant_abs_mass_fraction`]?.[3])}；SiLU负导数区间比例 {fmt(selected.regimes[`${split}_negative_silu_derivative_fraction`])}。</p><p className="v8-note">双负份额按 |A| 统计，不是输出能量或语义贡献比例。负gate不等于关闭。</p><details><summary>四象限频率、绝对激活与平方幅值份额</summary><table className="v6-table"><thead><tr><th>G / V</th><th>频率</th><th>|A|份额</th><th>A²份额</th></tr></thead><tbody>{['+ / +','+ / −','− / +','− / −'].map((label,q)=><tr key={label}><td>{label}</td>{['quadrant_fraction','quadrant_abs_mass_fraction','quadrant_energy_fraction'].map(name=><td key={name}>{fmt(selected.regimes[`${split}_${name}`]?.[q])}</td>)}</tr>)}</tbody></table></details></>}
        <p className="v8-note">以上统计覆盖每篇8个固定位置；下面另保留512–1023全部位置的极值，避免只看固定位置漏检。</p>
        <details><summary>全部长前缀扫描的{pole==='positive'?'高':'低'}值：{fmt(selected.full_prefix_pole.value)} · token {selected.full_prefix_pole.token_position}</summary><pre style={{whiteSpace:'pre-wrap'}}>{selected.full_prefix_pole.prefix}</pre></details>
        {selected.examples.map((x,i)=><details key={`${x.document_index}:${x.token_position}`} open={i===0}><summary>样本 {x.document_index} · token {x.token_position} · G {fmt(x.G)} / V {fmt(x.V)} / A {fmt(x.A)}</summary><p className="v8-note">{x.source.source} · {x.source.split} · {x.source.license}</p><pre style={{whiteSpace:'pre-wrap'}}>{x.prefix}</pre><details><summary>后续文本（不参与该位置计算）</summary><pre style={{whiteSpace:'pre-wrap'}}>{x.following_not_causal}</pre></details></details>)}
      </>}
      {tab==='parameters'&&<>
        <p>该单元的 gate 行、up 行和 down 列均保留全部 {dim.toLocaleString()} 个参数。</p>
        <div className="v6-toolbar"><label>参数 <select aria-label="原生参数分量" value={weight} onChange={e=>setWeight(e.target.value)}><option value="gate">gate_proj 行</option><option value="up">up_proj 行</option><option value="down">down_proj 列</option></select></label><label>坐标 <input aria-label="原生参数坐标" type="number" min="0" max={dim-1} value={coordinate} onChange={e=>setCoordinate(Math.max(0,Math.min(dim-1,Number(e.target.value)||0)))}/></label><output>{selected.weights[weight][coordinate]}</output></div>
        <NativeArray values={selected.weights[weight]} selected={coordinate} onPick={setCoordinate} label={`B${block}/U${unit} ${weight} 全${dim}参数`}/>
        <details><summary>全部参数原值</summary><pre className="v8-artifact-json">{JSON.stringify(selected.weights[weight])}</pre></details>
        <details><summary>原生计算核对：输入 → G/V → A → 写出列</summary><p>样本 {selected.arithmetic.document_index}，位置 {selected.arithmetic.position}。实际 G={fmt(selected.arithmetic.native_G)}，V={fmt(selected.arithmetic.native_V)}，A={fmt(selected.arithmetic.native_A)}。</p><p className="v8-note">G=Wgate·z，V=Wup·z，A=SiLU(G)×V，单元写出=A×Wdown列。FP64点积与原始BF16矩阵运算可有舍入差异；单列贡献不等于整个MLP输出。</p><p>FP64 gate点积 {fmt(selected.arithmetic.fp64_gate_dot)} / up点积 {fmt(selected.arithmetic.fp64_up_dot)}</p><NativeArray values={selected.arithmetic.normalized_input} selected={coordinate} onPick={setCoordinate} label={`归一化输入全部${dim}坐标`}/><NativeArray values={selected.arithmetic.native_column_contribution} selected={coordinate} onPick={setCoordinate} label={`单元写出贡献全部${dim}坐标`}/><details><summary>完整计算输入与贡献原值</summary><pre className="v8-artifact-json">{JSON.stringify(selected.arithmetic)}</pre></details></details>
      </>}
      {tab==='relations'&&<>
        <div className="v6-toolbar"><label>方向 <select aria-label="原生关联方向" value={actualDirection} onChange={e=>setDirection(e.target.value)}><option value="outgoing" disabled={block===last}>下一层</option><option value="incoming" disabled={block===first}>上一层</option></select></label>
          <label>证据 <select aria-label="原生关联证据" value={kind} onChange={e=>setKind(e.target.value)}>{[['rho-discovery','发现集共同响应'],['rho-confirm','确认分片共同响应'],['rho-holdout','留出分片共同响应'],['weight-gate','写出与gate权重对齐'],['weight-up','写出与up权重对齐']].map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label><label><input type="checkbox" checked={allEdges} onChange={e=>setAllEdges(e.target.checked)}/> 绘制全部关联</label></div>
        <p className="v8-note">每次查询返回相邻层全部{width.toLocaleString()}个单元。共同响应和静态权重对齐均不代表因果传递；权重余弦没有包含中间注意力和归一化。</p>
        {selected.frozen_parent&&<p className="v8-note">发现侧冻结的单父预测器：B{selected.frozen_parent.block}/U{selected.frozen_parent.unit}，留出MSE/均值基线 {fmt(selected.frozen_parent.holdout_mse_ratio)}；这是响应预测关系，未确认为因果边。</p>}
        {relation.error?<Empty>{relation.error}</Empty>:relation.data?<><NativeArray values={relation.data.values} selected={0} onPick={u=>pick(actualDirection==='incoming'?block-1:block+1,u)} label={`相邻层全部${width}关联`}/><details><summary>全部关联原值</summary><pre className="v8-artifact-json">{JSON.stringify(relation.data.values)}</pre></details></>:<p>从原始数据计算完整关联…</p>}
      </>}
      {tab==='evidence'&&<>
        <p>候选解释先在发现集建立，再检验其他文档。材料来自既有语料，不能称为全项目从未见过的盲测。</p>
        <table className="v6-table"><thead><tr><th>分量 / 划分</th><th>上下文/词身份预测误差比</th><th>上下文误差</th><th>置乱对照误差</th></tr></thead><tbody>{Object.entries(selected.prediction).flatMap(([f,v])=>Object.entries(v.splits).map(([s,r])=><tr key={f+s}><td>{f} / {s}</td><td>{fmt(r.conditional_over_lexical)}</td><td>{fmt(r.conditional_mse)}</td><td>{fmt(r.shuffled_mse)}</td></tr>))}</tbody></table>
        {selected.standardized_prediction&&<details open><summary>标准化预测与完整词身份对照</summary><p className="v8-note">本轮分析前冻结；仅用发现侧标准化与拟合。加入完整token身份加背景对照，仍不等于语义确认。</p><table className="v6-table"><thead><tr><th>分量 / 划分</th><th>上下文MSE</th><th>匹配token基线MSE</th><th>完整token与背景MSE</th><th>移位对照MSE</th><th>响应候选</th></tr></thead><tbody>{['G','V','A'].flatMap(f=>['confirm','holdout'].map(s=><tr key={f+s}><td>{f} / {s}</td>{['conditional_mse','matched_current_mse','exact_token_background_mse','shuffled_mse'].map(k=><td key={k}>{fmt(selected.standardized_prediction[`${f}_${s}_${k}`])}</td>)}<td>{selected.standardized_prediction[`${f}_candidate`]?'通过筛查':'未通过'}</td></tr>))}</tbody></table></details>}
        {selected.standardized_prediction&&<details><summary>G/V组合能否预测实际乘积</summary><p className="v8-note">组合预测为 SiLU(预测G)×预测V。因为非线性与条件相关，它通常不等于A的条件均值；这里直接与原生A和直接预测A比较。</p><table className="v6-table"><thead><tr><th>划分</th><th>组合预测MSE</th><th>直接预测A的MSE</th><th>均值基线MSE</th></tr></thead><tbody>{['confirm','holdout'].map(s=><tr key={s}><td>{s}</td>{['composed_mse','conditional_mse','mean_mse'].map(k=><td key={k}>{fmt(selected.standardized_prediction[`A_${s}_${k}`])}</td>)}</tr>)}</tbody></table><p>组合预测筛查：{selected.standardized_prediction.A_composed_candidate?'通过（两分片均比直接预测至少降低10%误差且胜过均值）':'未通过'}。此项筛查与上下文候选口径分开记账。</p></details>}
        <p className="v8-note">比值小于1表示优于表中对应词身份基线。通过响应预测筛选仍不等于语义或因果机制确认。</p>
        <p>本单元语境词复现：确认分片 {selected.candidate.repeat.confirm.hits}/4；留出分片 {selected.candidate.repeat.holdout.hits}/4。零命中和反例完整保留。</p>
        <div className="v6-toolbar"><label>已测单元 <select aria-label="原生已测干预单元" value={`${block}:${unit}`} onChange={e=>{const [b,u]=e.target.value.split(':').map(Number);pick(b,u)}}><option value={`${block}:${unit}`}>当前 B{block}/U{unit}</option>{data.intervention?.selected_units?.filter(r=>r.block!==block||r.unit!==unit).map(r=><option key={`${r.block}:${r.unit}`} value={`${r.block}:${r.unit}`}>B{r.block}/U{r.unit}</option>)}</select></label></div>
        <Dose rows={data.intervention?.natural?.rows?.filter(r=>r.block===block&&r.unit===unit)||[]}/>
        <details><summary>能力保持检查（已暴露的历史选择题）</summary><table className="v6-table"><thead><tr><th>倍数 / 条件</th><th>原对改错</th><th>原错改对</th><th>损失比例上界</th><th>2%门</th></tr></thead><tbody>{data.intervention?.capability?.rows?.filter(r=>r.block===block&&r.unit===unit).map(r=><tr key={r.id}><td>{r.dose} / {r.condition}</td><td>{r.lost}</td><td>{r.gained}</td><td>{fmt(r.lost_fraction_upper95)}</td><td>{r.capability_gate?'通过本项':'未通过'}</td></tr>)}</tbody></table><p className="v8-note">这里检查有限位置、有限剂量的选择题副作用，不能代表所有任务或自由生成都保持不变。</p></details>
        {selected.intervention_cases?.map(r=><details key={r.sample}><summary>实测样本 {r.id} · {r.argmax===r.baseline_argmax?'首词未变':'首词变化'} · 损失变化 {fmt(r.nll-r.baseline_nll)}</summary><p>原预测 {JSON.stringify(r.original_prediction)} → 编辑后 {JSON.stringify(r.edited_prediction)}；语料下一词 {JSON.stringify(r.target_token)}</p><pre style={{whiteSpace:'pre-wrap'}}>{r.prefix}</pre></details>)}
        <button className="fw-tbtn" onClick={()=>setRawOpen(v=>!v)}>{rawOpen?'收起逐样本记录':'查看逐样本干预与失败对照'}</button>
        {rawOpen&&<div><div className="v6-toolbar"><label>实验 <select aria-label="原生逐样本实验" value={rawStage} onChange={e=>{setRawStage(e.target.value);setRawSample(0)}}><option value="natural">自然文本（256篇）</option><option value="capability">能力检查（512题）</option></select></label><label>样本 <input aria-label="原生干预样本" type="number" min="0" max={rawStage==='natural'?255:511} value={rawSample} onChange={e=>setRawSample(Math.max(0,Math.min(rawStage==='natural'?255:511,Number(e.target.value)||0)))}/></label></div>{raw.error?<Empty>{raw.error}</Empty>:raw.data?<><pre style={{whiteSpace:'pre-wrap'}}>{raw.data.input_text}</pre><details><summary>本样本全部剂量、对照、恢复与身份</summary><pre className="v8-artifact-json">{JSON.stringify(raw.data,null,2)}</pre></details></>:<p>读取原始记录…</p>}<a href={`${API_BASE}${url(`/interventions/${rawStage}/${block}/${unit}`)}`} target="_blank" rel="noreferrer">打开该单元全部逐样本原始行</a></div>}
        <details><summary>跨层递推、干预状态与限制</summary><pre className="v8-artifact-json">{JSON.stringify({prediction:data.neighbor_summary,path:data.intervention,limitations:data.limitations},null,2)}</pre></details>
        <details><summary>模型与运行身份</summary><pre className="v8-artifact-json">{JSON.stringify({run:data.run_id,source_run:selected.source_run_id,snapshot:data.model_snapshot,precision:data.precision,scope:data.scope},null,2)}</pre></details>
      </>}
    </>}
  </section>;
}
