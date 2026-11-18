import {useMemo,useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {apiUrl} from '../../config.js';
import {NativeArray} from './MechanismScene.jsx';
import './hiddenstate_components.css';

const number=x=>Number.isFinite(x)?x.toPrecision(4):'未定义';
const percent=x=>Number.isFinite(x)?`${(100*x).toFixed(1)}%`:'未定义';
const names={color:'查询颜色',language:'指定输出语言',color_language:'颜色与语言交互'};
function decodeArray(doc){
  if(!doc||doc.encoding!=='base64'||doc.dtype!=='<f4'||doc.shape?.join(',')!=='2,2,3,2560')return null;
  try {
  const text=atob(doc.data),bytes=Uint8Array.from(text,c=>c.charCodeAt(0));
  if(bytes.length!==2*2*3*2560*4)return null;
  const view=new DataView(bytes.buffer);return Array.from({length:bytes.length/4},(_,i)=>view.getFloat32(i*4,true));
  } catch {return null;}
}
function NativeComponents({doc,layer,readout,factor}){
  const resource=useApi(doc.coordinate_artifacts?.[layer]?`/api/research-library/artifacts/${doc.coordinate_artifacts[layer]}`:null);
  const [coordinate,setCoordinate]=useState(0),[split,setSplit]=useState(1);
  const values=useMemo(()=>decodeArray(resource.data?.array),[resource.data]);
  const aligned=resource.data?.model_snapshot===doc.model_snapshot&&resource.data?.hidden_index===layer;
  if(!doc.coordinate_artifacts?.[layer])return <p role="status">此报告未包含该层坐标产物，请更新研究数据包或查看材料索引。</p>;
  if(resource.error)return <p role="status">坐标数据读取失败：{resource.error}</p>;
  if(!resource.data)return <p>读取该层完整坐标…</p>;
  if(!aligned||!values)return <p role="status">数据形状或模型身份不匹配，未展示响应。</p>;
  const r=readout==='answer_prefix'?1:0,k=['language','color','color_language'].indexOf(factor);
  const offset=((split*2+r)*3+k)*2560,vector=values.slice(offset,offset+2560);
  return <><label>数据 <select aria-label="成分坐标来源" value={split} onChange={e=>setSplit(Number(e.target.value))}><option value="0">发现集拟合</option><option value="1">旧评估集实测</option></select></label>
    <p>hs[{layer}] · {names[factor]} · 原生坐标 {coordinate} = {number(vector[coordinate])}</p>
    <label>原生坐标 <input aria-label="成分原生坐标" type="number" min="0" max="2559" value={coordinate} onChange={e=>setCoordinate(Math.min(2559,Math.max(0,Number(e.target.value)||0)))}/></label>
    <NativeArray values={vector} selected={coordinate} onPick={setCoordinate} label="成分完整原生系数"/>
    <p className="v8-note">向量是八条件正交对照的平均系数，原生顺序、全部2560维；不是单神经元，也不表示“该坐标只存这一种信息”。</p>
  </>;
}
function Curve({rows,factor,selected,onSelect}){
  const choices=factor==='language'?[['language','直接语言读出','#246caf'],['language_norm_only','仅向量长度','#b08045']]:[['color','直接颜色读出','#246caf'],['color_from_query_record','查询与记录读出后相乘','#b08045'],['color_norm_only','仅向量长度','#848b94']];
  const x=l=>45+l/36*700,y=v=>220-v*180;
  return <><div className="hs-curve-scroll" tabIndex="0" aria-label="全层曲线，可横向滚动"><svg viewBox="0 0 780 265" className="v8-dose-chart" role="img" aria-label="全层信息可读出轨迹">
    {[.5,.9,1].map(v=><g key={v}><line x1="45" x2="745" y1={y(v)} y2={y(v)} stroke="#d8e0e8" strokeDasharray="4 3"/><text x="5" y={y(v)+4}>{percent(v)}</text></g>)}
    {[0,6,12,18,24,30,36].map(l=><text key={l} x={x(l)} y="247" textAnchor="middle">{l===36?'36＊':l}</text>)}
    {choices.map(([key,label,color])=><g key={key}><polyline fill="none" stroke={color} strokeWidth="2" points={rows.slice(0,36).map(r=>`${x(r.hidden_index)},${y(r.probes[key])}`).join(' ')}/><line x1={x(35)} x2={x(36)} y1={y(rows[35].probes[key])} y2={y(rows[36].probes[key])} stroke={color} strokeDasharray="3 3"/>{rows.map(r=><circle key={r.hidden_index} cx={x(r.hidden_index)} cy={y(r.probes[key])} r={r.hidden_index===selected?5:2.5} fill={color} onClick={()=>onSelect(r.hidden_index)}><title>{`${label} · hs[${r.hidden_index}] ${percent(r.probes[key])}`}</title></circle>)}</g>)}
    <text x="390" y="263" textAnchor="middle">HiddenState 索引；＊最终归一化后</text>
  </svg></div><p>{choices.map(([,label,color])=><span key={label} style={{color,marginRight:18}}>● {label}</span>)}</p></>;
}

export default function HiddenStateComponentsReport({doc}){
  const [tab,setTab]=useState('flow'),[readout,setReadout]=useState('answer_prefix'),[factor,setFactor]=useState('color'),[layer,setLayer]=useState(doc.onsets.color??20);
  const rows=doc.rows.filter(r=>r.readout===readout),current=rows.find(r=>r.hidden_index===layer),component=current.components[factor];
  return <section className="hs-components-report" aria-label="全层HiddenState成分形成报告">
    <p><strong>颜色与输出语言状态怎样形成</strong> · Qwen3-4B · 37个状态边界 × 2个位置 × 全部2560坐标</p>
    <p>复用128个发现家族、512个已暴露评估家族。当前为探索性定位，没有新增盲测或原模型干预。先区分信息可读、条件响应与真实计算来源。</p>
    {doc.interpretation&&<p>{doc.interpretation}</p>}
    <div className="v6-toolbar" role="group" aria-label="HiddenState分析内容">{[['flow','全层形成'],['native','完整原生坐标'],['transition','状态更新'],['audit','边界与反例']].map(([k,label])=><button key={k} className="fw-tbtn" aria-pressed={tab===k} onClick={()=>setTab(k)}>{label}</button>)}</div>
    <div className="hs-controls"><label>读出位置 <select aria-label="成分读出位置" value={readout} onChange={e=>setReadout(e.target.value)}><option value="answer_prefix">答案前：全部条件可见</option><option value="record_end">记录结束：查询与语言指令尚不可见</option></select></label>
      <label>成分 <select aria-label="状态成分" value={factor} onChange={e=>setFactor(e.target.value)}>{Object.entries(names).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
      <label>状态边界 <select aria-label="HiddenState边界" value={layer} onChange={e=>setLayer(Number(e.target.value))}>{Array.from({length:37},(_,l)=><option key={l} value={l}>{l===0?'hs[0] 嵌入':l===36?'hs[36] B35与最终归一化后':`hs[${l}] B${l-1}后`}</option>)}</select></label></div>
    {readout==='record_end'&&<p>此位置看不到查询对象和输出语言要求。这里的颜色/语言读出应为机会水平；记录中的红蓝映射本身已经可见，两者不能混称。</p>}
    {layer===36&&<p>此状态经过最终 RMSNorm。35→36 不纳入原始残差增量和状态更新比较，避免把归一化当作新信息写入。</p>}
    {tab==='flow'&&<>
      {factor!=='color_language'?<Curve rows={rows} factor={factor} selected={layer} onSelect={setLayer}/>:<p>交互系数表示颜色条件效应随语言变化。它不是独立分类标签，不为它伪造准确率曲线。</p>}
      <p>hs[{layer}] · 系数范数 {number(component.contrast_norm)}；发现与旧评估方向余弦 {number(component.discovery_transfer_cosine)}；当前八条件响应方差中的系数能量 {percent(component.controlled_variance_fraction)}。</p>
      <p className="v8-note">能量比例仅针对当前配平实验的变化，不是模型信息的“颜色占比”。零方差时显示未定义。可读出起点按连续3个原始状态准确率≥90%记录，不代表信息首次产生。</p>
      <details><summary>比较读出方法与跨条件迁移</summary>
      <table className="v6-table"><thead><tr><th>同一位置的读出</th><th>准确率</th></tr></thead><tbody>{[['color','直接颜色'],['color_from_query_record','查询和记录读出后相乘'],['language','指定输出语言'],['color_norm_only','颜色：仅看向量长度'],['language_norm_only','语言：仅看向量长度'],['color_train_L1','颜色：英文条件拟合→中文评估'],['color_train_L-1','颜色：中文条件拟合→英文评估'],['language_train_c1','语言：红色条件拟合→蓝色评估'],['language_train_c-1','语言：蓝色条件拟合→红色评估']].map(([key,label])=><tr key={key}><td>{label}</td><td>{percent(current.probes[key])}</td></tr>)}</tbody></table>
      <p>条件读出器拟合对应条件的中心和方向，尺度共用整个发现集的方差；不能视为训练时完全没见过另一条件。</p>
      {readout==='answer_prefix'&&doc.audit?.language_transfer_diagnostics?.filter(d=>d.hidden_index===layer&&d.trained_color===1).map(d=><p key={d.hidden_index}>红色→蓝色语言读出的均值诊断：英语得分 {number(d.mean_english_score)}，中文得分 {number(d.mean_chinese_score)}，阈值为0。两者同号时，中心偏移就可能使固定阈值失效；这不是语言信息消失的证据。红蓝条件的语言方向余弦 {number(d.direction_cosine_red_blue)}。此项为事后几何诊断，没有重新校准准确率。</p>)}
      </details>
      <details><summary>完整状态预测与输出读出</summary>
      <p>全状态预测误差：八条件 / 四状态 {number(current.eight_over_four_mse)}；四状态 / 常数 {number(current.four_over_constant_mse)}。四状态基线由人工提供颜色关系，不能当成上游计算机制。</p>
      <p>最终权重读出诊断：{current.lens.calibrated?`四答案选择准确率 ${percent(current.lens.four_choice_accuracy)}，终点与原输出通过数值校准。`:'终点未过校准，暂不展示该轨迹数值。'}中间层仍经过原模型后续计算，此曲线不等于它实际会立即输出的答案。</p>
      </details>
    </>}
    {tab==='native'&&<NativeComponents doc={doc} layer={layer} readout={readout} factor={factor}/>}
    {tab==='transition'&&<>
      <p>用真实源层状态预测下一层；比较原样复制、平均增量、已知颜色/语言的四状态增量，以及从源状态读出颜色/语言再预测增量。它们只做一步预测，不是完整模型替代。</p>
      {current.step?<table className="v6-table"><thead><tr><th>hs[{layer}] → hs[{layer+1}]</th><th>原生均方误差</th></tr></thead><tbody>{[['identity','原样复制'],['mean_increment','发现集平均增量'],['oracle_four_state','提供颜色和语言状态'],['decoded_state','仅从源状态推断颜色和语言']].map(([k,v])=><tr key={k}><td>{v}</td><td>{number(current.step[k])}</td></tr>)}</tbody></table>:<p>当前边界不提供可比的下一层原始残差转移。</p>}
      <p>当前成分增量范数 {number(component.contrast_increment_norm)}；与前一层方向余弦 {number(component.previous_direction_cosine)}。这里只能定位整块净变化，缺少 attention后/MLP后数据，不能归因于某个模块。</p>
      <p>所有样本最后一个token的嵌入相同。只靠这个向量自循环的确定性系统无法产生不同答案；闭合预测必须另外表示前缀信息或KV历史。</p>
    </>}
    {tab==='audit'&&<>
      <p>640个原生文件哈希已核验。记录结束处，未来查询和语言条件的状态逐位一致；最终token的embedding也逐位一致。既有材料只允许探索，颜色之外的逻辑、风格和语法尚未验证。</p>
      {doc.counterexamples.map((c,i)=><details key={i}><summary>{names[c.factor]} · hs[{c.hidden_index}] · 家族{c.family} · 正确{percent(c.accuracy)}</summary><pre className="v8-sample-text">{c.prompt}</pre><p>显示该家族第一种条件；准确率汇总其全部八种条件。最差样本不必然代表失败。</p></details>)}
      {Object.entries(doc.artifacts||{}).map(([k,v])=><p key={k}><a href={apiUrl(`/api/research-library/artifacts/${v}`)} target="_blank" rel="noreferrer">{k}</a></p>)}
      <details><summary>完整限制、起点与负对照</summary><pre className="v8-artifact-json">{JSON.stringify({limits:doc.limits,onsets:doc.onsets,negative_controls:doc.negative_controls},null,2)}</pre></details>
    </>}
  </section>;
}
