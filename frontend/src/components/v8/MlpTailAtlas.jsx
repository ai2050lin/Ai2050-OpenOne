import { useEffect, useMemo, useRef, useState } from 'react';
import { panFixedCanvas, pitchedScreenY, useScenePointer, SCENE_MOUSE_HINT, useSceneWheel } from '../scenePointer.js';
import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { NativeArray } from './MechanismScene.jsx';
import MlpConditionCard from './MlpConditionCard.jsx';

const BASE='/api/research/mlp-tail';
const BLOCKS=[32,33,34,35];
const EMPTY=[];
const initialNumber=(key,fallback,min,max)=>{if(typeof window==='undefined')return fallback;const raw=new URLSearchParams(window.location.search).get(key),n=Number(raw);return raw!==null&&Number.isFinite(n)&&n>=min&&n<=max?Math.floor(n):fallback};
const num=v=>Number.isFinite(v)?v.toPrecision(5):'未测';
const color=(v,max,alpha=1)=>!Number.isFinite(v)?'#c7d1df':v>=0?`rgba(42,120,214,${alpha*(.13+.8*Math.abs(v)/(max||1))})`:`rgba(235,104,52,${alpha*(.13+.8*Math.abs(v)/(max||1))})`;

function TailVector({values,label,onPick}){
  const [index,setIndex]=useState(0);
  const pick=i=>{setIndex(i);onPick?.(i)};
  return <div><NativeArray values={values} selected={index} onPick={pick} label={label}/><label>{label} · 坐标 <input type="number" min="0" max={values.length-1} value={index} onChange={e=>pick(Math.max(0,Math.min(values.length-1,Number(e.target.value)||0)))}/></label> = <output>{values[index]}</output><details><summary>全部原始数值（{values.length}项）</summary><pre className="v8-artifact-json">{JSON.stringify(values)}</pre></details></div>;
}

function TailDose({rows}){
  if(!rows?.length)return <p className="v8-note">本单元未进行原模型干预；保留关联记录，等待实验。</p>;
  const max=Math.max(.000001,...rows.map(r=>r.mean_target_response_norm||0));
  return <><svg viewBox="0 0 680 220" role="img" aria-label="有效等范数匹配子集的真实剂量响应曲线" style={{maxWidth:850,width:'100%'}}>
    {[0,.5,1.5,2].map(f=><text key={f} x={60+f*285} y="198">{f}</text>)}
    {[0,.5,1].map(v=><g key={v}><line x1="60" x2="640" y1={170-v*140} y2={170-v*140} stroke="#dbe3ec"/><text x="3" y={175-v*140}>{num(v*max)}</text></g>)}
    {['native_unit','matched_noise'].map((c,i)=><polyline key={c} fill="none" stroke={i?'#de793c':'#3279bd'} strokeWidth="2" points={rows.filter(r=>r.condition===c).sort((a,b)=>a.factor-b.factor).map(r=>`${60+r.factor*285},${170-r.mean_target_response_norm/max*140}`).join(' ')}/>)}
    <text x="200" y="216">单元原生写出倍数（1为未编辑；只连接实测剂量）</text></svg>
    <p className="v8-note">蓝：原生单元写入；橙：等范数随机方向。纵轴为下一层全部单元响应变化范数。各剂量仅用双方均通过1%范数匹配的相同文档，样本数可不同。这是描述性复核，不是语义或能力改善结论。</p>
    <table className="v6-table"><thead><tr><th>倍数 / 条件</th><th>有效文档</th><th>全词表KL</th><th>预测损失变化</th><th>首词变化</th></tr></thead><tbody>{rows.map(r=><tr key={`${r.factor}:${r.condition}`}><td>{r.factor} / {r.condition==='native_unit'?'原生':'对照'}</td><td>{r.n}/96</td><td>{num(r.mean_KL)}</td><td>{num(r.mean_nll_change)}</td><td>{r.top1_changes}</td></tr>)}</tbody></table></>;
}

export function TailScene({blocks=BLOCKS,layers,metric,block,unit,relation,allEdges,onPick}){
  const ref=useRef(null),hits=useRef([]);
  const [view, setView] = useState({ yaw: -.48, pitch: 0, zoom: 1 });
  const { yaw, pitch, panX = 0, panY = 0, zoom } = view;
  const count=blocks.length*9728;
  const height=blocks.length>4?840:590;
  const vectors=useMemo(()=>Object.fromEntries(blocks.map(b=>[b,layers[b]?.arrays?.[metric]||EMPTY])),[blocks,layers,metric]);
  useEffect(()=>{
    const canvas=ref.current,ctx=canvas?.getContext('2d');if(!ctx)return;
    ctx.clearRect(0,0,900,height);hits.current=[];
    const mid=(blocks[0]+blocks.at(-1))/2;
    const spacing=(height-185)/(blocks.length-1);
    const point=(b,u)=>{const x=(u%128-63.5)/64,z=(Math.floor(u/128)-37.5)/60;
      return [450+panX+zoom*185*(x*Math.cos(yaw)-z*Math.sin(yaw)), (height-55)/2+panY+pitchedScreenY(spacing*(b-mid),zoom*185*(x*Math.sin(yaw)+z*Math.cos(yaw)),pitch,1,35/185)];};
    const selected=point(block,unit),values=relation?.values;
    const destination=relation?.direction==='incoming'?relation.source:relation?.target;
    let edges=0;
    if(values?.length===9728&&blocks.includes(destination)){
      const max=Math.max(1e-12,...values.map(Math.abs));
      values.forEach((v,i)=>{if(!allEdges&&Math.abs(v)<max*.2)return;const end=point(destination,i);ctx.beginPath();ctx.moveTo(...selected);ctx.lineTo(...end);ctx.strokeStyle=color(v,max,.13);ctx.stroke();edges++;});
    }
    for(const b of blocks){
      const array=values?.length===9728&&b===destination?values:vectors[b];
      const max=Math.max(1e-12,...array.map(Math.abs));
      for(let i=0;i<9728;i++){
        const [x,y]=point(b,i);ctx.fillStyle=color(array[i],max);ctx.fillRect(x-1.1,y-1.1,2.2,2.2);hits.current.push({x,y,b,u:i});
      }
      ctx.fillStyle='#334155';ctx.font='13px sans-serif';ctx.fillText(`B${b} · 第${b+1}层 · 9,728单元`,50,point(b,0)[1]-16);
    }
    ctx.strokeStyle='#111827';ctx.lineWidth=2;ctx.strokeRect(selected[0]-4,selected[1]-4,8,8);ctx.lineWidth=1;
    ctx.fillStyle='#475569';ctx.font='12px sans-serif';ctx.fillText(`全部${count.toLocaleString('en-US')}单元 · 原生编号 · 当前B${block}/U${unit} · 绘制${edges}/9,728条关系`,35,height-38);
    ctx.fillText('蓝正橙负。板内位置按原生单元编号排布。连线含义见证据类型。',35,height-15);
  },[blocks,count,height,vectors,block,unit,relation,allEdges,yaw,pitch,panX,panY,zoom]);
  const pointer = useScenePointer({
    onPan: (dx, dy, canvas) => setView(v => panFixedCanvas(v, dx, dy, canvas)),
    onRotate: (dx, dy) => setView(v => ({ ...v, yaw: v.yaw + dx / 180, pitch: Math.max(-1, Math.min(1, v.pitch + dy / 180)) })),
    onPick: e => {
      const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)*900/r.width,y=(e.clientY-r.top)*height/r.height;
            const candidates=hits.current.filter(h=>Math.hypot(h.x-x,h.y-y)<5).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y));if(candidates.length)onPick(candidates[0].b,candidates[0].u)
    },
  });
  const zoomBy = delta => setView(v => ({ ...v, zoom: Math.max(0.65, Math.min(1.5, v.zoom + delta)) }));
  useSceneWheel(ref, delta => zoomBy(delta < 0 ? 0.1 : -0.1));
  return <><canvas ref={ref} width="900" height={height} style={{width:'100%',maxWidth:1050,touchAction:'none',cursor:'grab'}} aria-label={`Qwen3-4B B${blocks[0]}–B${blocks.at(-1)}全部${count}个MLP单元及分类型关联3D`}
    title={SCENE_MOUSE_HINT} {...pointer} />
    <div className="v6-toolbar"><button className="fw-tbtn" onClick={()=>zoomBy(0.1)}>放大</button><button className="fw-tbtn" onClick={()=>zoomBy(-0.1)}>缩小</button><button className="fw-tbtn" onClick={()=>setView({ yaw: -.48, pitch: 0, zoom: 1 })}>重置视角</button></div></>;
}

export default function MlpTailAtlas(){
  const report=useApi(BASE);
  if(report.error)return <Empty>{report.error}；尾层研究未接通，不使用演示数据。部署后可读取全部单元。</Empty>;
  if(!report.data)return <p className="v8-note">读取真实 MLP 图谱…</p>;
  return <TailAtlasContent key={report.data.run_id} data={report.data}/>;
}

function TailAtlasContent({data}){
  const blocks=data.blocks||BLOCKS,first=blocks[0],last=blocks.at(-1);
  const [block,setBlock]=useState(()=>initialNumber('tail_block',first,first,last)),[unit,setUnit]=useState(()=>initialNumber('tail_unit',2663,0,9727)),[metric,setMetric]=useState('G_mean'),[kind,setKind]=useState('rho-holdout'),[direction,setDirection]=useState('outgoing'),[allEdges,setAllEdges]=useState(false);
  const [split,setSplit]=useState('holdout'),[field,setField]=useState('G'),[pole,setPole]=useState('positive');
  const [factor,setFactor]=useState(2);
  const b28=useApi(first<=28?`${BASE}/layers/28`:null),b29=useApi(first<=29?`${BASE}/layers/29`:null),
        b30=useApi(first<=30?`${BASE}/layers/30`:null),b31=useApi(first<=31?`${BASE}/layers/31`:null),
        b32=useApi(`${BASE}/layers/32`),b33=useApi(`${BASE}/layers/33`),
        b34=useApi(`${BASE}/layers/34`),b35=useApi(`${BASE}/layers/35`);
  const layers=useMemo(()=>({28:b28.data,29:b29.data,30:b30.data,31:b31.data,32:b32.data,33:b33.data,34:b34.data,35:b35.data}),
    [b28.data,b29.data,b30.data,b31.data,b32.data,b33.data,b34.data,b35.data]);
  const detail=useApi(`${BASE}/units/${block}/${unit}?split=${split}&field=${field}&pole=${pole}`);
  const actualDirection=block===last?'incoming':block===first?'outgoing':direction;
  const relation=useApi(`${BASE}/relations/${block}/${unit}?direction=${actualDirection}&kind=${kind}&factor=${factor}`);
  const selected=detail.data,layer=layers[block];
  const layerError=[b28,b29,b30,b31,b32,b33,b34,b35].find(response=>response.error)?.error;
  useEffect(()=>{if(typeof window!=='undefined'){const url=new URL(window.location.href);url.searchParams.set('tail_block',block);url.searchParams.set('tail_unit',unit);window.history.replaceState(null,'',url)}},[block,unit]);
  const pick=(b,u)=>{setBlock(b);setUnit(u)};
  return <section aria-label="尾层MLP原生特性与关联研究">
    <h3>Qwen3-4B · B{first}–B{last} MLP</h3><p className="v8-note">{data.n_units.toLocaleString('en-US')} 个原生单元 · 2,048 篇 1,024-token 长文本 · gate、value、乘积和上下游分别看。全量覆盖不等于全部语义已经破解。</p>
    {data.extension&&<p className="v8-note">B28–B31 新增真实原生观察。B31→B32 可双向回查共同响应与权重对齐。递推和干预沿用 B32–B35 已有实验范围。</p>}
    {data.admissibility&&<p className="v8-note">研究状态：探索性观察。干预尚缺事前功效预注册，运行溯源在事后补做审计；不计为正式机制确认。原始冻结与负结果保留。</p>}
    <div className="v6-toolbar"><label>层 <select aria-label="尾层研究层" value={block} onChange={e=>setBlock(Number(e.target.value))}>{blocks.map(b=><option key={b} value={b}>B{b}（第{b+1}层）</option>)}</select></label>
      <label>单元 <input aria-label="尾层原生单元" type="number" min="0" max="9727" value={unit} onChange={e=>setUnit(Math.max(0,Math.min(9727,Number(e.target.value)||0)))}/></label>
      <label>特性 <select aria-label="尾层着色特性" value={metric} onChange={e=>setMetric(e.target.value)}>{[['G_mean','门控均值'],['G_std','门控变化'],['V_mean','value均值'],['V_std','value变化'],['A_mean','乘积均值'],['A_std','乘积变化'],['sign_0','正门控×正value'],['sign_1','正门控×负value'],['sign_2','负门控×正value'],['sign_3','负门控×负value']].map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
      <label>方向 <select aria-label="尾层关系方向" value={actualDirection} onChange={e=>setDirection(e.target.value)}><option value="outgoing" disabled={block===last}>下一层（出向）</option><option value="incoming" disabled={block===first}>上一层（入向）</option></select></label>
      <label>关系 <select aria-label="尾层关系证据类型" value={kind} onChange={e=>setKind(e.target.value)}>{[['rho-discovery','发现集共同响应'],['rho-confirm','确认集共同响应'],['rho-holdout','留出集共同响应'],['weight-gate','写出→gate权重对齐'],['weight-up','写出→value权重对齐'],['causal','已测原模型有限干预']].map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
      <label><input type="checkbox" checked={allEdges} onChange={e=>setAllEdges(e.target.checked)}/> 绘制全部关系</label></div>
    {kind==='causal'&&<div className="v6-toolbar"><label>已测源单元 <select aria-label="已测干预源单元" value={`${block}:${unit}`} onChange={e=>{const [b,u]=e.target.value.split(':').map(Number);pick(b,u);setDirection('outgoing')}}><option value={`${block}:${unit}`}>当前 B{block}/U{unit}</option>{[...new Map(data.path.summaries.map(r=>[`${r.source}:${r.unit}`,r])).values()].filter(r=>r.source!==block||r.unit!==unit).map(r=><option key={`${r.source}:${r.unit}`} value={`${r.source}:${r.unit}`}>B{r.source}/U{r.unit}</option>)}</select></label><label>写出倍数 <select aria-label="干预写出倍数" value={factor} onChange={e=>setFactor(Number(e.target.value))}>{[0,.5,1.5,2].map(f=><option key={f}>{f}</option>)}</select></label><p className="v8-note">有限干预从源单元向下一层查询；B35没有下一层MLP。</p></div>}
    {layerError?<Empty>部分层数据读取失败。{layerError}</Empty>:blocks.every(b=>layers[b])?<TailScene blocks={blocks} layers={layers} metric={metric} block={block} unit={unit} relation={relation.data} allEdges={allEdges} onPick={pick}/>:<p className="v8-note">正在读取 {blocks.filter(b=>layers[b]).length} / {blocks.length} 层原生数据…</p>}
    <MlpConditionCard block={block} unit={unit} snapshot={data.model_snapshot}/>
    <details><summary>原关键词线索（保留历史，不作为单元名称）</summary><p>{layer?.candidate_labels?.[unit]||'读取候选语境线索…'}</p></details>
    <p className="v8-note">{relation.error||relation.data?.scope||'读取关系…'} 候选关键词与当前 token 特征需要进一步验证，不作为单义性或因果解释。</p>
    {layer&&<details className="mp-fold"><summary>当前层全部9728单元的二维查询与数值</summary><TailVector key={`${block}:${metric}`} values={layer.arrays[metric]} onPick={setUnit} label={`B${block}全部9728单元的${metric}二维查询`}/></details>}
    <details className="mp-fold"><summary>当前单元的两极语境、正负门控与完整参数</summary><div className="mp-fold-body">
      <div className="v6-toolbar"><select aria-label="尾层证据划分" value={split} onChange={e=>setSplit(e.target.value)}>{['discovery','confirm','holdout'].map(s=><option key={s}>{s}</option>)}</select>
        <select aria-label="尾层两极字段" value={field} onChange={e=>setField(e.target.value)}>{['G','V','A'].map(f=><option key={f}>{f}</option>)}</select>
        <select aria-label="尾层两极方向" value={pole} onChange={e=>setPole(e.target.value)}><option value="positive">正极</option><option value="negative">负极</option></select></div>
      {detail.error?<Empty>{detail.error}</Empty>:selected&&<><p className="v8-note">G=gate预激活，V=up/value，A=SiLU(G)×V。负门控×负value也能产生正A，需分开解释。正负两极不是两类神经元。</p>
        {selected.candidate?.repeat&&<p className="v8-note">发现集候选关键词在确认集极值位置命中 {selected.candidate.repeat.confirm.cue_hit_count}/{selected.candidate.repeat.confirm.n_pole_positions}，在留出集命中 {selected.candidate.repeat.holdout.cue_hit_count}/{selected.candidate.repeat.holdout.n_pole_positions}。该统计取各划分门控正极，保留零命中。</p>}
        <table className="v6-table"><thead><tr><th>字段</th><th>均值</th><th>标准差</th><th>最小</th><th>最大</th></tr></thead><tbody>{Object.entries(selected.statistics).map(([f,s])=><tr key={f}><td>{f}</td>{['mean','std','min','max'].map(k=><td key={k}>{num(s[k])}</td>)}</tr>)}</tbody></table>
        <p className="v8-note">四种G/V符号组合比例：{selected.sign_regime_fractions.map(num).join(' / ')}。当前token线索相关：{Object.entries(selected.cue_correlations||{}).map(([k,v])=>`${k}=${num(v)}`).join('；')}。这些线索包含词身份混杂。</p>
        <details><summary>写出方向偏向哪些词（全词表参数读出）</summary><p className="v8-note">仅原生down列与unembedding行的余弦；未包含上下文归一化、后续计算与因果验证。</p>{['positive','negative'].map(sign=><p key={sign}><b>{sign==='positive'?'正极':'负极'}</b> {selected.parameter_readout?.[sign]?.map(t=>`${JSON.stringify(t.text)} [${t.token_id}] ${num(t.cosine)}`).join('；')}</p>)}</details>
        {selected.examples.map((e,i)=><details key={e.document_index+':'+e.token_position}><summary>原片{i+1} · 文档{e.document_index} / token{e.token_position} · {e.regime} · G {num(e.G)} / V {num(e.V)} / A {num(e.A)}</summary><p className="v8-note">{e.source.source} · {e.source.license}</p><pre className="v8-sample-text">{e.prefix}</pre><details><summary>后续文本（不作为该位置的因果输入）</summary>{e.following_not_causal}</details></details>)}
        <details><summary>gate行、value行、down列：各2560个真实BF16参数</summary>{Object.entries(selected.weights).map(([k,v])=><TailVector key={k} values={v} label={`B${block}/U${unit}完整${k}参数`}/>)}</details>
      </>}
    </div></details>
    <details className="mp-fold"><summary>全部9728个关联值与匹配控制</summary><div className="mp-fold-body">{relation.data?.values&&<TailVector key={`${block}:${unit}:${kind}:${actualDirection}`} values={relation.data.values} label="全量关联数组（坐标为相邻层单元）"/>}{relation.data?.control_values&&<TailVector values={relation.data.control_values} label="严格等范数控制的完整目标响应"/>}<p className="v8-note">仅绘制强边时仍保留所有数值；没有把Top-K定义成主干。</p></div></details>
    <details className="mp-fold"><summary>跨层预测、连续递推、干预与未通过项</summary><div className="mp-fold-body"><p>{data.conclusion}</p>
      <TailDose rows={data.path.matched_summaries?.filter(r=>r.source===block&&r.unit===unit)}/>
      <table className="v6-table"><thead><tr><th>边界</th><th>划分</th><th>8父单元FVU</th><th>token基线FVU</th><th>G/V组合FVU</th></tr></thead><tbody>{data.prediction.one_step.map(r=><tr key={r.source+r.split}><td>B{r.source}→B{r.target}</td><td>{r.split}</td><td>{num(r.fvu.sparse_direct)}</td><td>{num(r.fvu.token)}</td><td>{num(r.fvu.sparse_composed)}</td></tr>)}</tbody></table>
      <p className="v8-note">FVU越低越好，1是发现集均值基线。异常大值与递推失败保留，不裁剪补过门。这里使用上一层实测A初始化，不是图谱独立预测下一词。</p>
      <p className="v8-note">原模型：{data.path.documents}个语境、{data.path.rows}个干预/对照观测；完整恢复已核验。{data.path.capability}</p>
      <details><summary>完整结果、反例索引和限制</summary><pre className="v8-artifact-json">{JSON.stringify({rollout:data.prediction.rollout,path:data.path.summaries,limitations:data.limitations},null,2)}</pre></details>
    </div></details>
  </section>;
}
