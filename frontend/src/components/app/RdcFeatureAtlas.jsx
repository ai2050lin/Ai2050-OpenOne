import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { API_CONFIG } from '../../config/api';
import './RdcFeatureAtlas.css';

const API = `${API_CONFIG.main.replace(/\/$/, '')}/api/rdc`;
const RUN_CHOICES = [
  ['i_factorial','I/J · 4096因子组合 / 全单元条件预测'],['k_long','K · 八类长生成 / 全词表预测'],
  ['aligned_qwen4','L · Qwen4 首内容对齐'],['aligned_qwen14','L · Qwen14 首内容对齐'],['aligned_glm4','L · GLM4 首内容对齐'],
  ['m_order','M/N · 输出顺序 / 历史KV条件预测'],['o_generalization','O · 新材料 / 冻结注意力预测推广'],['e_confirmation','E · 独立材料 / 同形核查 / 固定读尺'],
  ['g_generation','G · 当前 next-token 与生成阶段'],['scale_qwen14','H · Qwen3-14B 非量化全场'],['scale_glm4','H · GLM4 非量化全场'],
  ['a_native','A · 真实MLP单元与标量参数'],['b_relations','B · 关系 / 角色 / 语法四格'],['c_generation','C · 自回归与原生输出'],
  ['s1','S1 · Qwen3-4B 真实语言'],['s2pilot','S2试运行 · 冻结读取器独立确认'],['s0','S0 · 已知结构校准']
];
async function json(path, signal) {
  const r = await fetch(`${API}${path}`, { signal, cache: 'no-store' });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}
const number = x => Number(x).toPrecision(7);

function FieldPoints({ data, colorScale, gain, onSelect }) {
  const { positions, colors, lookup } = useMemo(() => {
    const p = [], c = [], lookup = [];
    data.values.forEach((tokens, l) => tokens.forEach((coords, t) => coords.forEach((value, k) => {
      p.push((k - coords.length / 2) * (12 / coords.length), l * 2, (data.token_ids[t]-(data.token_ids[0]+data.token_ids.at(-1))/2) * .48);
      const intensity = Math.min(1, Math.abs(value) / Math.max(colorScale, 1e-12) * gain);
      const color = new THREE.Color(value >= 0 ? '#ff8c65' : '#4fc8fa').lerp(new THREE.Color('#263347'), 1-intensity);
      c.push(color.r, color.g, color.b);
      lookup.push({ value, layer: data.layer_ids[l], token: data.token_ids[t], coordinate: data.coordinate_start+k });
    })));
    return { positions: new Float32Array(p), colors: new Float32Array(c), lookup };
  }, [data, colorScale, gain]);
  return <group>
    <points onClick={e => { e.stopPropagation(); if (lookup[e.index]) onSelect(lookup[e.index]); }}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={.2} vertexColors sizeAttenuation />
    </points>
    <Text position={[0,-1,-5]} fontSize={.35} color="#bacce2">{`原生坐标 ${data.coordinate_start}–${data.coordinate_start+data.coordinate_count-1}`}</Text>
    <Text position={[-8,2,0]} fontSize={.3} color="#bacce2">{data.field==='pooled_H36_prediction'?'真实 / 预测 / 误差':data.field.startsWith('continuity_')||data.field.startsWith('conditional_')?'对照行（含义见下方）':['linear_reader_contribution','mlp_units','parameter_path','output_ledger','output_units','source_ledger'].includes(data.field)?'数值 / 系数 / 贡献（量纲不同）':'层 / 检查点'}</Text>
    <Text position={[0,-1,5]} fontSize={.3} color="#bacce2">真实 token 位置</Text>
  </group>;
}

function ExternalGraph({ row, samples, onSelectSample }) {
  const families = [...new Set(samples.map(r => r.family || r.structure))];
  const support=typeof row?.fact_truth==='boolean'?row.fact_truth:typeof row?.same_group==='boolean'?row.same_group:null;
  const relation=row?.relation_edges?.[0]?.type||row?.relation;
  const relationText=relation?`${relation} · ${support===null?'未定义支持标签':support?'材料支持':'材料不支持'}`:typeof row?.same_group==='boolean'?`任务标注：${row.same_group?'同组':'不同组'}`:`${row?.family||'词项参照'} · 不推断同组关系`;
  return <group>
    {families.map((f, i) => {
      const angle=i/families.length*Math.PI*2; const p=[Math.cos(angle)*7,0,Math.sin(angle)*7];
      return <group key={f} position={p}>
        <mesh onClick={() => { const next=samples.find(r=>(r.family||r.structure)===f&&r.committed); if(next) onSelectSample(next.sample_id); }}>
          <sphereGeometry args={[.35,16,16]} /><meshStandardMaterial color={f===row?.family?'#ffa26e':'#466584'} />
        </mesh>
        <Text position={[0,.8,0]} fontSize={.3} color="#d2e0ed">{f}</Text>
      </group>;
    })}
    {row?.u && <>
      <Text position={[-2,2,0]} fontSize={.6} color="#ffb59b">{row.u}</Text>
      <Text position={[2,2,0]} fontSize={.6} color="#7cdaf2">{row.v}</Text>
      <Line points={[[-1.5,2,0],[1.5,2,0]]} color={support===null?'#7791ad':support?'#70d9a1':'#d6a46a'} dashed lineWidth={2} />
      <Text position={[0,3.2,0]} fontSize={.32} color="#bcd0df">{relationText}</Text>
      <Text position={[0,-1.5,0]} fontSize={.3} color="#91a2b6">节点距离是界面布局，不是模型语义距离</Text>
    </>}
  </group>;
}

function AlgorithmBars({ rows }) {
  const maximum = Math.max(.001,...rows.map(r=>r.mse));
  return <group>
    {rows.map((r,i) => <group key={`${r.algorithm}-${i}`} position={[(i-(rows.length-1)/2)*1.5,0,0]}>
      <mesh position={[0,r.mse/maximum*3,0]}><boxGeometry args={[.8,Math.max(.01,r.mse/maximum*6),.8]} /><meshStandardMaterial color={r.algorithm.startsWith('A0')?'#677b92':'#63d2cb'} /></mesh>
      <Text position={[0,-.8,0]} rotation={[-Math.PI/4,0,0]} fontSize={.22} color="#bacce2">{r.algorithm}</Text>
      <Text position={[0,r.mse/maximum*6+.35,0]} fontSize={.22} color="#e2edf8">{r.mse.toPrecision(3)}</Text>
    </group>)}
    <Text position={[0,-2,0]} fontSize={.3} color="#a4bacd">测试 MSE · 越低越好 · 此视图柱高按当前组最大值缩放</Text>
  </group>;
}

export default function RdcFeatureAtlas() {
  const [run,setRun]=useState('i_factorial');
  return <RdcRunView key={run} run={run} onRunChange={setRun}/>;
}

function RdcRunView({run,onRunChange}) {
  const [state,setState]=useState({}), [samples,setSamples]=useState([]);
  const [sample,setSample]=useState(''), [view,setView]=useState('field'), [field,setField]=useState(run==='s0'?'u':'h');
  const [layer,setLayer]=useState(0), [tokenSelection,setTokenSelection]=useState({sample:'',value:0}), [coordinate,setCoordinate]=useState(0);
  const [rawData,setData]=useState(null), [error,setError]=useState(''), [selected,setSelected]=useState(null);
  const [colorScale,setColorScale]=useState(1), [gain,setGain]=useState(1), [follow,setFollow]=useState(true);
  const [result,setResult]=useState(null), [eventCount,setEventCount]=useState(0), [group,setGroup]=useState('');
  const [figures,setFigures]=useState([]);
  const [parameter,setParameter]=useState(null), [component,setComponent]=useState('q'), [pRow,setPRow]=useState(0), [pStart,setPStart]=useState(0);
  const [modelId,setModelId]=useState(''), [coefficient,setCoefficient]=useState(null);
  const [pollError,setPollError]=useState(''), [predictionAlgorithm,setPredictionAlgorithm]=useState('A1_linear');
  const [classIndex,setClassIndex]=useState(0);
  const [unit,setUnit]=useState(0), [block,setBlock]=useState(0);
  const [inputCoordinate,setInputCoordinate]=useState(0), [outputCoordinate,setOutputCoordinate]=useState(0);
  const isGeneration=['c_generation','g_generation'].includes(run);
  const isScale=run.startsWith('scale_');
  const isAligned=run.startsWith('aligned_');
  const isConditional=['i_factorial','k_long','m_order','o_generalization'].includes(run)||isAligned;
  const hasSteps=isGeneration||['k_long','m_order','o_generalization'].includes(run)||isAligned;
  const modelName=run.endsWith('qwen14')?'Qwen3-14B':run.endsWith('glm4')?'glm4-9b-chat-hf':'qwen3-4b';
  const nativeWidth=modelName==='Qwen3-14B'?5120:modelName==='glm4-9b-chat-hf'?4096:2560;
  const nativeUnits=modelName==='Qwen3-14B'?17408:modelName==='glm4-9b-chat-hf'?13696:9728;
  const maxLayer=nativeWidth===2560?36:40;
  const isContinuity=view.startsWith('continuity_');
  const isConditionalView=view.startsWith('conditional_');
  const row=samples.find(r=>r.sample_id===sample);
  const anchorToken=view==='source_ledger'?0:row?.query_position??(row?.spans?Math.min(...row.spans.u.positions,...row.spans.v.positions):0);
  const token=tokenSelection.sample===sample?tokenSelection.value:anchorToken;
  const setToken=value=>setTokenSelection({sample,value});
  const visibleState=state.run_id===run?state:{};
  const requestKey=[run,sample,field,layer,token,coordinate,view,predictionAlgorithm,classIndex,unit,block,inputCoordinate,outputCoordinate].join('|');
  const data=rawData?.requestKey===requestKey?rawData:null;
  const cursor=useRef(0), follows=useRef(true);
  useEffect(()=>{follows.current=follow;},[follow]);
  useEffect(() => {
    let stopped=false; let timer; let version=''; let cachedSamples=[]; const controller=new AbortController();
    cursor.current=0;
    const poll=async()=> {
      try {
        const [s,e]=await Promise.all([json(`/runs/${run}/status`,controller.signal),json(`/runs/${run}/events?after=${cursor.current}`,controller.signal)]);
        if(stopped)return;
        const nextVersion=JSON.stringify([s.state,s.completed,s.states,s.updated_at]);
        if(nextVersion!==version){
          const [m,r]=await Promise.all([json(`/runs/${run}/material`,controller.signal),json(`/runs/${run}/results`,controller.signal)]);
          if(stopped)return;cachedSamples=m.samples;version=nextVersion;
          setSamples(m.samples);setFigures(r.figures||[]);setResult(r.result?{...r.result,extension:r.extension,measurement_warning:r.extension?.measurement_warning,results:[...(r.result.results||[]),...(r.extension?.results||[])]}:null);
        }
        setState(s);cursor.current=e.cursor;setEventCount(e.cursor);setPollError('');
        const available=cachedSamples.filter(x=>x.committed);
        setSample(old => {
          if(follows.current&&s.current_sample&&available.some(x=>x.sample_id===s.current_sample))return s.current_sample;
          return available.some(x=>x.sample_id===old)?old:available[0]?.sample_id||'';
        });
      } catch(err) { if(!stopped&&err.name!=='AbortError')setPollError(err.message); }
      if(!stopped)timer=setTimeout(poll,1500);
    };
    poll();return()=>{stopped=true;controller.abort();clearTimeout(timer);};
  },[run]);
  useEffect(()=> {
    if(!sample)return undefined;
    const c=new AbortController();
    const query=new URLSearchParams({sample,field,layer,layers:field==='h'?4:1,token,tokens:16,coordinate,width:128});
    const path=view==='conditional_sources'?`/conditional/source_groups?${new URLSearchParams({sample,layer,coordinate,width:128})}`:view.startsWith('conditional_')?`/conditional/inspect?${new URLSearchParams({run,sample,view:view.replace('conditional_',''),layer,coordinate,width:128})}`:view==='parameter_path'&&isConditional?`/conditional/parameter_path?${new URLSearchParams({run,sample,layer,unit,input_coordinate:inputCoordinate,output_coordinate:outputCoordinate,coordinate,width:128})}`:view.startsWith('continuity_')?`/continuity/inspect?${new URLSearchParams({sample,view:view.replace('continuity_',''),layer,target:classIndex%2,coordinate,width:128})}`:['mlp_units','parameter_path'].includes(view)?`/mechanism/${view}?${new URLSearchParams({run,sample,layer,token,unit,block,class_index:classIndex,coordinate,input_coordinate:inputCoordinate,output_coordinate:outputCoordinate,width:128})}`:view==='source_ledger'?`/mechanism/source_ledger?${new URLSearchParams({run,sample,token,coordinate,width:32})}`:['output_ledger','output_units'].includes(view)?`/mechanism/${view}?${new URLSearchParams({run,sample,coordinate,width:128})}`:view==='prediction'?`/prediction?${new URLSearchParams({sample,split:'word',algorithm:predictionAlgorithm,coordinate,width:128})}`:view==='contribution'?`/linear_contribution?${new URLSearchParams({sample,layer,class_index:classIndex,coordinate,width:128})}`:`/runs/${run}/field?${query}`;
    json(path,c.signal).then(d=>{setData({...d,requestKey});setSelected(null);setError('');}).catch(e=>{if(e.name!=='AbortError')setError(e.message);});
    return()=>c.abort();
  },[run,sample,field,layer,token,coordinate,view,predictionAlgorithm,classIndex,unit,block,inputCoordinate,outputCoordinate,requestKey,isConditional]);
  const available=samples.filter(r=>r.committed);
  const results=result?.results||[];
  const groups=[...new Set(results.map(r=>r.structure||`${r.split}/${r.target}/${r.representation}`))];
  const currentGroup=groups.includes(group)?group:groups[0];
  const chartRows=results.filter(r=>(r.structure||`${r.split}/${r.target}/${r.representation}`)===currentGroup&&!r.oracle);
  const models=(result?.models||[]).filter(m=>m.algorithm==='A2_quadratic');
  async function inspectParameter() {
    try {setParameter(await json(`/parameter?${new URLSearchParams({component,model_name:modelName,layer:Math.min(layer,maxLayer-1),row:pRow,start:pStart,count:16})}`));setError('');}
    catch(e){setError(e.message);}
  }
  async function inspectCoefficient() {
    try {setCoefficient(await json(`/coefficient?${new URLSearchParams({run,model_id:modelId||models[0]?.model_id,j:pRow,start:pStart,count:16})}`));setError('');}
    catch(e){setError(e.message);}
  }
  return <div className="rdc-app">
    <header className="rdc-header"><div><small>RDC / LANGUAGE STRUCTURE LAB</small><h1>语言结构 · 原生坐标 · 提取算法</h1></div><a href="/rdc-prefix">共同自然前缀图谱 ↗</a><a href="/">返回原研究客户端 ↗</a></header>
    <div className="rdc-status"><span className={visibleState.state==='running'?'rdc-dot active':'rdc-dot'} />
      <strong data-testid="run-state">{run.toUpperCase()} · {visibleState.state||'连接中'}</strong>
      <span data-testid="run-progress">{visibleState.completed||0} / {visibleState.total||0} 已提交</span>
      <span>事件游标 {eventCount}</span><span>更新 {visibleState.updated_at?.slice(11,19)||'—'}</span>
      <span>{run==='s0'?'合成校准：不是LLM证据':'真实模型落盘数据；运行完成后为回放'}</span>
    </div>
    <main className="rdc-layout">
      <aside className="rdc-controls">
        <label>实验范围<select aria-label="实验范围" value={run} onChange={e=>onRunChange(e.target.value)}>{RUN_CHOICES.map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label>
        <label>真实样本<select aria-label="真实样本" value={sample} onChange={e=>{setFollow(false);setSample(e.target.value);}}>{available.map(r=><option key={r.sample_id} value={r.sample_id}>{r.sample_id}{r.u?` · ${r.u} / ${r.v}`:''}</option>)}</select></label>
        <label className="rdc-check"><input type="checkbox" checked={follow} onChange={e=>setFollow(e.target.checked)} />跟随最新已提交样本</label>
        <label>坐标域<select aria-label="坐标域" value={field} onChange={e=>setField(e.target.value)}>
          {(run==='s0'?['u','v','c']:isConditional?['h',...(['i_factorial','k_long'].includes(run)?['h_full']:[]),...(run==='i_factorial'?['u_mean','v_mean']:[]),...(['k_long','m_order','o_generalization'].includes(run)?['logits']:[]),...(['m_order','o_generalization'].includes(run)?['q','k','v','p','attention_out','head_output']:[]),'gate','up','a','down','mlp_x','attention_x']:isScale?['h','postnorm','gate','up','a','down']:isGeneration?['h','postnorm','q','k','v','p','gate','up','a','down','attention_out','head_output','attention_x','mlp_x']:['h','postnorm','q','k','v','qnorm','knorm','gate','up','a','down','attention_x','mlp_x']).map(x=><option key={x}>{x}</option>)}
        </select></label>
        <div className="rdc-grid">
          <label>起始层<input aria-label="起始层" type="number" min="0" max={maxLayer} value={layer} onChange={e=>setLayer(Math.max(0,Number(e.target.value)))} /></label>
          <label>起始token<input aria-label="起始token" type="number" min="0" value={token} onChange={e=>setToken(Math.max(0,Number(e.target.value)))} /></label>
        </div>
        <label>原生坐标起点<input aria-label="原生坐标起点" type="number" min="0" value={coordinate} onChange={e=>setCoordinate(Math.max(0,Number(e.target.value)))} /></label>
        <div className="rdc-grid"><button disabled={view==='source_ledger'} onClick={()=>setCoordinate(Math.max(0,coordinate-128))}>← 前128列</button><button disabled={view==='source_ledger'} onClick={()=>setCoordinate(coordinate+128)}>后128列 →</button></div>
        {view==='source_ledger'&&<div className="rdc-grid"><button onClick={()=>setToken(Math.max(0,token-16))}>← 前16来源</button><button onClick={()=>setToken(Math.min((row?.tokens.length||1)-1,token+16))}>后16来源 →</button></div>}
        <div className="rdc-grid">
          <label>固定色标 ±<input aria-label="固定色标" type="number" min="0.000001" step="0.1" value={colorScale} onChange={e=>setColorScale(Number(e.target.value))} /></label>
          <label>低值显示增益<input aria-label="低值增益" type="number" min="0.1" value={gain} onChange={e=>setGain(Number(e.target.value))} /></label>
        </div>
        <p className="rdc-note">红正蓝负；增益不改变数值。固定索引切片不筛Top-K。H0为输入嵌入，H1–{maxLayer}为块后残差。{isConditional?`全部${nativeWidth}坐标可查询。h为当前query；h_full仅预声明面板。I原生层11/23/35；K层23逐步、11/35仅分析步；M原生来源层11/23/35；O仅第1/4/8步L23原生量；L为模型指定检查点。不把当前query叫完整历史场；跨模型坐标编号不对应。`:isScale?'规模模型原生MLP仅在末位置采集，层0/12/26/39；坐标按各自模型编号，不做跨模型编号对齐。':isGeneration?'逐步记录当前query的全坐标，以及所有历史source的K/V/P。原生层11/23/35；P列是head索引，K/V列是KV head×128坐标。':'其他场覆盖完整提示词前向；原生组件仅在U/V/末位置采集，层0/11/23/35。'} H、MLP单元与权重不同。</p>
        {hasSteps&&row&&<label>同一前缀的生成步<select aria-label="生成步" value={sample} onChange={e=>{setFollow(false);setSample(e.target.value);setCoordinate(0);}}>{available.filter(r=>r.prefix_id===row.prefix_id).map(r=><option key={r.sample_id} value={r.sample_id}>{`步${r.generation_step} · 预测 ${r.next_token} · 位置${r.query_position}`}</option>)}</select></label>}
        {row&&<div className="rdc-identity"><strong>{row.u||row.structure||row.family}{row.v?` ↔ ${row.v}`:''}</strong><p>{row.language} · {row.word_split||row.split} · {row.input_order||row.order}</p><p>外部目标：{row.target||(isConditional?(row.expected_yes!==undefined?(row.expected_yes?'Yes / 是':'No / 否'):'详见冻结材料及实际行为评分'):'已知连续函数')}</p></div>}
        {row?.prompt&&<details><summary>真实模型输入与token</summary><pre>{row.prompt}</pre><p>{row.tokens.length} tokens</p></details>}
        {data?.original_input&&<details className="rdc-note" data-testid="original-input"><summary>初始真实输入与外部约束</summary><pre>{JSON.stringify(data.original_input,null,2)}</pre><p>当前前缀 {row?.tokens.length} tokens；生成历史以所选步骤为准，外部目标仅供核对。</p><pre>{JSON.stringify(row?.tokens,null,2)}</pre></details>}
      </aside>
      <section className="rdc-workspace">
        {isConditional&&<nav className="rdc-tabs">{(run==='i_factorial'?[['gate','同层条件重建'],['forecast_a','提前预测全单元'],['forecast_down','提前预测写回']]:run==='k_long'?[['gate','冻结门跨任务重建'],['forecast_h','长生成状态预测']]:run==='m_order'?[['gate','冻结门顺序对照'],['sources','全来源写回向量'],['forecast_attention','历史KV条件注意力预测']]:run==='o_generalization'?[['forecast_attention','冻结注意力预测推广']]:[]).map(([id,label])=><button key={id} className={view===`conditional_${id}`?'selected':''} onClick={()=>{setView(`conditional_${id}`);setLayer(23);setCoordinate(0);if(id.startsWith('forecast')){setFollow(false);const test=available.find(r=>r.word_split==='test'&&(!['k_long','o_generalization'].includes(run)||r.analysis_selected)&&(run!=='m_order'||r.result_field_onset));if(test)setSample(test.sample_id);}if(id==='sources'){setFollow(false);const boundary=available.find(r=>r.result_field_onset);if(boundary)setSample(boundary.sample_id);}}}>{label}</button>)}<button onClick={()=>{setView('parameter_path');setLayer(isAligned?maxLayer-1:23);setCoordinate(0);if(run==='o_generalization'){setFollow(false);const chosen=available.find(r=>r.analysis_selected);if(chosen)setSample(chosen.sample_id);}}}>真实标量参数路径</button></nav>}
        {run==='m_order'&&<p className="rdc-note">M只改变Result/Trace/Neutral段落顺序；不是完全同长度执行。当前query全层全坐标保存；完整来源K/V/P只在Result边界，先点“全来源写回向量”定位，再切回原生场查询。H与logits的列分别为隐坐标和词表ID；P列为32个head。</p>}
        {run==='o_generalization'&&<><p className="rdc-note">O为512条全留出新材料；最多观察12个新token，不作完整答案评分。各步H0–36全坐标保留，L23原生单元与全部来源K/V/P、logits仅在实际第1/4/8步。“冻结注意力预测推广”中的预测器在N冻结，O不重拟合。</p><button onClick={()=>{setView('conditional_forecast_token_conditioned');setLayer(23);setCoordinate(0);setFollow(false);const chosen=available.find(r=>r.analysis_selected&&r.unit>=12);if(chosen)setSample(chosen.sample_id);}}>探索性token条件重拟合</button><p className="rdc-note">P是看过O结果后的探索性实体重划分，不是独立确认。比较全历史条件预测、当前embedding条件及预测H23后通过真实参数计算的路线；与上面的冻结O结果分开保留。</p></>}
        {isConditional&&<p className="rdc-note" data-testid="conditional-scope">坐标编号逐一可查，未以Top-K/PCA定义主干。同层 g/up 重建、H12 提前预测、真实权重恒等式是三种不同证据。首内容对齐也不等于跨模型计算同构。</p>}
        {run==='e_confirmation'&&<nav className="rdc-tabs">{[['shape','自然 / 同形全坐标'],['ruler','固定跨层读尺'],['gate','全单元条件门'],['forecast','H12 → H24 预测']].map(([id,label])=><button key={id} className={view===`continuity_${id}`?'selected':''} onClick={()=>{setView(`continuity_${id}`);setCoordinate(0);setLayer(id==='gate'?23:24);if(id==='forecast'){setFollow(false);const test=available.find(r=>r.word_split==='test');if(test)setSample(test.sample_id);}}}>{label}</button>)}<button onClick={()=>{setView('parameter_path');setLayer(23);setCoordinate(0);setToken(row?.prompt_ids.length-1||0);}}>新材料单参数路径</button></nav>}
        {isContinuity&&<p className="rdc-note" data-testid="continuity-scope">固定原生坐标；自然/同形是数值协议对照，不是纯语义分离。读尺是外部固定读取器；条件门近似和提前预测分开评价。<select aria-label="固定读尺目标" value={classIndex%2} onChange={e=>setClassIndex(Number(e.target.value))}><option value="0">材料支持 t</option><option value="1">请求答案 y</option></select></p>}
        <nav className="rdc-tabs">{[['field','原生坐标场'],['external','外部关系图'],['algorithms','算法对照'],...(run==='a_native'?[['mlp_units','全MLP单元'],['parameter_path','单参数输入路径']]:[]),...(isGeneration?[['output_ledger','原生输出账本'],['output_units','输出MLP单元'],['source_ledger','全部历史来源']]:[]),...(run==='s1'?[['prediction','预测与真实场']]:[]),...(run==='s2pilot'?[['contribution','坐标贡献账本']]:[])].map(([id,label])=><button key={id} className={view===id?'selected':''} onClick={()=>{setView(id);if(['mlp_units','parameter_path'].includes(id)){setLayer(11);setCoordinate(0);setToken(anchorToken);}if(['output_ledger','output_units','source_ledger'].includes(id))setCoordinate(0);if(id==='source_ledger')setToken(0);if(id==='contribution'){setCoordinate(0);setLayer(36);}if(id==='prediction'){setFollow(false);setCoordinate(0);const test=available.find(r=>r.word_split==='test');if(test)setSample(test.sample_id);}}}>{label}</button>)}</nav>
        {['mlp_units','parameter_path'].includes(view)&&<div className="rdc-grid"><label>单元 j (0–{nativeUnits-1})<input aria-label="MLP单元" type="number" min="0" max={nativeUnits-1} value={unit} onChange={e=>setUnit(Number(e.target.value))}/></label>{!isConditional&&<><label>读取块<select aria-label="读取块" value={block} onChange={e=>setBlock(Number(e.target.value))}>{['U','V','C'].map((b,i)=><option key={b} value={i}>{b}</option>)}</select></label><label>读取类别<input aria-label="读取类别" type="number" min="0" max="7" value={classIndex} onChange={e=>setClassIndex(Number(e.target.value))}/></label></>}</div>}
        {(error||pollError)&&<div className="rdc-error" role="alert">{error||pollError}</div>}
        {result?.measurement_warning&&<p className="rdc-note" data-testid="measurement-warning">{result.measurement_warning}</p>}
        {view==='prediction'&&<label>H12全坐标 → H36词跨度均值 · 词条留出测试集<select aria-label="预测算法" value={predictionAlgorithm} onChange={e=>setPredictionAlgorithm(e.target.value)}>{['A0_mean','A0_distance','A1_linear','A2_quadratic','A2_cubic','A3_ordered_pair','A4_conditional'].map(a=><option key={a}>{a}</option>)}</select></label>}
        {view==='contribution'&&<label>冻结线性读取器的目标族（分数不是概率）<select aria-label="贡献目标族" value={classIndex} onChange={e=>setClassIndex(Number(e.target.value))}>{['fruit','whole_plant','animal','tool','action','property','function_word','punctuation'].map((f,i)=><option key={f} value={i}>{f}</option>)}</select></label>}
        {view==='algorithms'&&<select aria-label="算法结果组" value={currentGroup||''} onChange={e=>setGroup(e.target.value)}>{groups.map(g=><option key={g}>{g}</option>)}</select>}
        <div className="rdc-canvas" data-testid="rdc-canvas">
          <Canvas camera={{position:[11,12,18],fov:48}}><color attach="background" args={['#09121f']} /><ambientLight intensity={1.3}/><directionalLight position={[10,20,5]} intensity={2}/>
            {(isContinuity||isConditionalView||['field','prediction','contribution','mlp_units','parameter_path','output_ledger','output_units','source_ledger'].includes(view))&&data&&<FieldPoints data={data} colorScale={colorScale} gain={gain} onSelect={setSelected}/>}
            {view==='external'&&<ExternalGraph row={row} samples={samples} onSelectSample={id=>{setFollow(false);setSample(id);}}/>}
            {view==='algorithms'&&<AlgorithmBars rows={chartRows}/>}
            <gridHelper args={[26,26,'#263d56','#15263a']} position={[0,-2.5,0]}/><OrbitControls makeDefault />
          </Canvas>
          {!data&&(isContinuity||isConditionalView||['field','prediction','contribution','mlp_units','parameter_path','output_ledger','output_units','source_ledger'].includes(view))&&<div className="rdc-empty">等待有效坐标切片；未完成的样本不会显示为完成。</div>}
        </div>
        {view==='prediction'&&data&&<p className="rdc-note">三排依次为真实H36、提取器预测、预测减真实；不是三个模型层。全2560坐标MSE：{number(data.full_coordinate_mse)}。预测输入只有H12三个完整块，未读取目标H36。</p>}
        {view==='contribution'&&data&&<p className="rdc-note" data-testid="contribution-sum">三排为输入x、提取器系数w、乘积xw，量纲不同；全7680项相加＋偏置{number(data.bias)}＝{number(data.full_sum)}。列0–2559为U，2560–5119为V，5120–7679为C。词跨度使用全坐标均值，C为当前问题/末token。不是真实权重或因果归因；层仅支持0/12/24/36。</p>}
        {data&&['mlp_units','parameter_path','output_ledger','output_units','source_ledger'].includes(view)&&<div className="rdc-note" data-testid="mechanism-ledger"><p>{data.axes}</p><p>真实权重组合的账本，不等于独立因果齿轮或机制闭合。</p>{data.full_sum!==undefined&&<p>全量总和 {number(data.full_sum)}</p>}{data.full_dots&&<p>全{nativeWidth}项 gate/up 点积：{data.full_dots.map(number).join(' / ')}；观测 gate/up/a：{data.observed_gate_up_a.map(number).join(' / ')}；舍入残差：{data.rounding_remainder.map(number).join(' / ')}</p>}{data.account&&<pre>{JSON.stringify(data.account,null,2)}</pre>}</div>}
        {data&&<p className="rdc-note">{data.axes}</p>}
        {isContinuity&&data&&<p className="rdc-note" data-testid="continuity-value">{data.full_sum!==undefined?`固定读尺分数 ${number(data.full_sum)}`:data.full_coordinate_mse!==undefined?`全2560坐标预测MSE ${number(data.full_coordinate_mse)}`:data.full_max_abs!==undefined?`同位置全2560坐标最大数值差 ${number(data.full_max_abs)}`:'完整9728单元可按原生编号查询，非Top-K筛选。'}</p>}
        {isConditionalView&&data&&<pre className="rdc-note" data-testid="conditional-value">{JSON.stringify({rows:data.layer_ids,full_coordinate_mse:data.full_coordinate_mse,evidence:data.source_mode},null,2)}</pre>}
        {view==='conditional_sources'&&data?.account&&<details className="rdc-note"><summary>全部来源分组与舍入核对</summary><p>投影份额可为负或超过1，不是独立概率或因果重要性。Result字段起点可能只是中文词的首字节片段，数量任务也尚未到首个数字。</p><pre>{JSON.stringify(data.account,null,2)}</pre></details>}
        {view==='parameter_path'&&<div className="rdc-grid"><label>输入坐标 i<input aria-label="链输入坐标" type="number" min="0" max={nativeWidth-1} value={inputCoordinate} onChange={e=>setInputCoordinate(Number(e.target.value))}/></label><label>写回坐标 k<input aria-label="链写回坐标" type="number" min="0" max={nativeWidth-1} value={outputCoordinate} onChange={e=>setOutputCoordinate(Number(e.target.value))}/></label></div>}
        {view==='parameter_path'&&data?.scalar_chain&&<details open data-testid="scalar-chain"><summary>输入坐标 → 真实单元 → 真实写回参数</summary><pre>{JSON.stringify(data.scalar_chain,null,2)}</pre></details>}
        <div className="rdc-readout">
          <span data-testid="field-coverage">{data?`${data.shown_values.toLocaleString()} / ${data.total_stored_values.toLocaleString()} 当前切片数值 · ${data.dtype}`:'尚无坐标'}</span>
          <span data-testid="coordinate-value">{selected&&data?`H/L ${selected.layer} · token ${selected.token} · 坐标 ${selected.coordinate} = ${number(selected.value)}`:data?`首显示值 ${number(data.values[0][0][0])} · 点击空间点查询精确坐标`:'—'}</span>
        </div>
        {data?.behavior&&<div className="rdc-behavior" data-testid="actual-behavior">{isConditional?<><p>实际行为与评分范围（内容、顺序、格式、停止独立记录）</p><pre>{JSON.stringify(data.behavior,null,2)}</pre></>:isScale?`实际首token：${JSON.stringify(data.behavior.actual_token)} (ID ${data.behavior.first_argmax}) · 首token正确 ${String(data.behavior.argmax_answer_correct)} · 候选对偏好正确 ${String(data.behavior.pair_correct)} · 尚非完整自由生成；空白前缀不能按语义答错解释`:run==='e_confirmation'?`尚非自由生成；自然首token ID ${data.behavior.natural_argmax} / 同形首token ID ${data.behavior.matched_argmax}；目标 ${data.behavior.expected_target}；自然正确 ${String(data.behavior.natural_correct)} / 同形正确 ${String(data.behavior.matched_correct)}`:<>实际输出：{data.behavior.generated} · 外部目标：{data.behavior.expected_target} · {data.behavior.correct?'按规则匹配':'未匹配'} · EOS {String(data.behavior.eos)}</>}</div>}
        {view==='algorithms'&&<table><thead><tr><th>算法</th><th>测试MSE</th><th>准确率</th><th>n</th></tr></thead><tbody>{chartRows.map((r,i)=><tr key={i}><td>{r.algorithm}</td><td>{number(r.mse)}</td><td>{r.accuracy===undefined?'—':`${(r.accuracy*100).toFixed(1)}%`}</td><td>{r.n}</td></tr>)}</tbody></table>}
        {figures.length>0&&<details className="rdc-parameters" data-testid="scientific-figures"><summary>全坐标科学图与原始数值</summary><p>按原生编号保留全部坐标/单元。页面缩览不用于读取细点；打开原图检查，聚合、色标和裁剪边界以显示说明为准。</p>{figures.map(f=><figure key={f.id}><figcaption>{f.title} · <a href={`${API}/figures/${f.id}`} target="_blank" rel="noreferrer">原图</a> · <a href={`${API}/figures/${f.id}?asset=values`}>全部绘图数值</a> · <a href={`${API}/figures/${f.id}?asset=contract`}>显示说明</a></figcaption><img loading="lazy" style={{width:'100%',height:'auto'}} src={`${API}/figures/${f.id}`} alt={f.title}/></figure>)}</details>}
        {result&&<details className="rdc-parameters"><summary>阶段结果与证据范围</summary><p>来自真实运行的汇总与限制；条件近似、提前预测、原生恒等式分别记录。逐算法指标另见算法对照。</p><pre data-testid="stage-evidence">{JSON.stringify({...result,results:undefined,models:undefined},null,2)}</pre></details>}
        <details className="rdc-parameters"><summary>真实参数 / 提取器坐标交互账本</summary><p>真实权重只读查询；提取器系数独立标识，不视为LLM原参数或唯一因果来源。</p>
          <div className="rdc-grid"><label>组件<select aria-label="参数组件" value={component} onChange={e=>setComponent(e.target.value)}>{['q','k','v','gate','up','down'].map(c=><option key={c}>{c}</option>)}</select></label>
            <label>权重行 / 交互j<input aria-label="参数行" type="number" min="0" value={pRow} onChange={e=>setPRow(Number(e.target.value))}/></label>
            <label>列起点<input aria-label="参数列" type="number" min="0" value={pStart} onChange={e=>setPStart(Number(e.target.value))}/></label>
            <button onClick={inspectParameter}>读取真实权重</button></div>
          {parameter&&<pre data-testid="native-parameter">{parameter.key}{'\n'}{parameter.physical_row!==parameter.row?`逻辑单元行 ${parameter.row} → 实际存储行 ${parameter.physical_row}\n`:''}{parameter.values.map((v,i)=>`[${parameter.physical_row??parameter.row},${parameter.start+i}] ${number(v)}`).join('\n')}</pre>}
          {models.length>0&&<><select aria-label="提取器模型" value={modelId||models[0]?.model_id} onChange={e=>setModelId(e.target.value)}>{models.map(m=><option key={m.model_id}>{m.model_id}</option>)}</select><button onClick={inspectCoefficient}>读取二阶交互系数</button></>}
          {coefficient&&<pre data-testid="extractor-coefficient">{coefficient.source}{'\n'}M[{coefficient.j}, {coefficient.start}…] = {coefficient.values.map(number).join(', ')}</pre>}
        </details>
      </section>
    </main>
  </div>;
}
