import {useEffect, useMemo, useRef, useState} from 'react';
import {useNative} from './MechanismTraceExplorer.jsx';
import {EXAMPLE_SITES} from './GraphWorkspaceEvidence.jsx';
import {readObservationArchive} from './nativeObservationArchive.js';
import {unpackField} from './nativeTraceField.js';
import './nativeFieldInspector.js';
import './nativeFieldInspector.css';

const headMeasures = [
  ['head_z','Z · 输出投影前'], ['head_q','Q · 旋转位置编码后'],
  ['head_k','K · 旋转位置编码后'], ['head_v','V · 值向量'],
];
const ffnMeasures = [
  ['mlp_activation','门控后的激活'], ['mlp_gate','Gate · 门控预激活'], ['mlp_value','Up · 值投影'],
];

function FieldCanvas({identity,kind,size,values,selected,source,missing,measure,comparisons,headColumns,onPick}){
  const host=useRef(null),view=useRef(null),pick=useRef(onPick);
  useEffect(()=>{pick.current=onPick;},[onPick]);
  useEffect(()=>{
    view.current=globalThis.AI2050NativeField.create(host.current,{onPick:index=>pick.current?.(index)});
    return ()=>{view.current?.destroy();view.current=null;};
  },[]);
  useEffect(()=>{
    view.current.update({identity,kind,size,values,selected,source,missing,measure,comparisons,headColumns,
      vectorLabel:headColumns?'整层 Head · 行为 H，列为 D':undefined});
  },[identity,kind,size,values,selected,source,missing,measure,comparisons,headColumns]);
  return <div ref={host}/>;
}

export default function FullNativeField({kind,model,snapshot,block,head,unit,ffnSize,headWidth,headCount,
    kvHeadCount,example,site=0,sample,trace,position,mlpOk,sourceError,bundled,atlas,
    channel=0,onPick,onChannel,onHead}) {
  const [hMeasure,setHeadMeasure]=useState('head_z');
  const [fMeasure,setFfnMeasure]=useState('mlp_activation');
  const [headScope,setHeadScope]=useState('head');
  const isHead=kind==='head', wholeHeads=isHead&&headScope==='layer';
  const choices=atlas?[['atlas',atlas.title]]:wholeHeads?headMeasures.slice(0,1):isHead?headMeasures:ffnMeasures;
  const measure=atlas?'atlas':wholeHeads?'head_z':isHead?hMeasure:fMeasure;
  const size=wholeHeads?headWidth*headCount:isHead?headWidth:ffnSize, traceId=trace?.run?.id;
  const haveBlock=Number.isInteger(block)&&block>=0;
  const haveObject=haveBlock&&Number.isInteger(size)&&size>0&&(!isHead||
    (Number.isInteger(head)&&head>=0&&head<headCount));
  const hasToken=Number.isInteger(position)&&position>=0&&position<trace?.run?.tokens?.length;
  const missingReadout=hasToken&&(!isHead||measure==='head_z')&&Array.isArray(trace.run.readout)&&!trace.run.readout.includes(position);
  const fieldPath=!atlas&&haveObject&&snapshot&&traceId&&hasToken&&!missingReadout
    ? '/api/traces/'+encodeURIComponent(traceId)+'/field?block='+block+'&position='+position+
      '&boundary='+measure+(isHead&&!wholeHeads?'&head='+head:'') : null;
  const lookup=useNative(trace?.base??null,fieldPath);
  const bundleData=bundled?.data, selectedRow=bundled?.row;
  const fieldHead=isHead&&!wholeHeads?head:0;
  const archivedField=useMemo(()=>!atlas&&bundled&&readObservationArchive({...bundled,measure,head:fieldHead,
    headCount,kvHeadCount,headWidth,size,wholeHeads}),
  [atlas,bundled,measure,fieldHead,headCount,kvHeadCount,headWidth,size,wholeHeads]);
  const archiveText=bundleData?.text||selectedRow?.text;
  const headChannel=channel>=0&&channel<headWidth?channel:0;
  const selected=wholeHeads?head*headWidth+headChannel:isHead?headChannel:unit;
  let source=model+' · B'+block+(isHead&&!wholeHeads?'/H'+head:''),missing='',values=null,contextKey='',comparisons=null;
  let archiveSha=null,partialHead=null;
  if(!haveObject){
    missing=haveBlock?'当前快照缺少可验证的对象编号或原生维度。':'嵌入层没有注意力头或 FFN 中间单元。';
  }else if(atlas){
    source+=' · '+atlas.source;
    contextKey=atlas.identity+':'+atlas.source+':'+atlas.metric;
    values=atlas.values;
    missing=atlas.missing;
  }else if(sourceError){
    missing=sourceError;
  }else if(archivedField){
    const field=archivedField;
    contextKey=bundleData.run+':'+selectedRow.position;
    source+=' · '+field.source;
    missing=field.missing||'';
    values=field.values||null;
    if(wholeHeads&&values&&field.partialHead!=null){
      partialHead=new Float32Array(size).fill(NaN);
      partialHead.set(values,field.partialHead*headWidth);
      values=null;
    }
    archiveSha=field.sha;
    if(!missing&&!values)missing='原生向量长度与当前模型维度不一致。';
  }else if(trace){
    contextKey=String(traceId)+':'+position;
    source+=' · 轨迹 '+traceId+' · token '+position;
    if(missingReadout)missing='本轨迹未在此 token 采集模块边界，请选择标记为模块读出的位置。';
    else if(!fieldPath || trace.base==null)missing='缺少采集节点、当前快照或有效 token 位置。';
    else if(lookup.error)missing='采集节点未提供此位置的 '+choices.find(x=>x[0]===measure)[1]+'。'+lookup.error;
    else if(!lookup.data)missing='正在读取当前轨迹的原生向量…';
    else {
      values=unpackField(lookup.data,{traceId,snapshot,block,position,measure,head,headCount,kvHeadCount,width:isHead?headWidth:size,wholeHeads});
      if(!values)missing='返回的运行、快照、读出边界或向量维度与当前对象不一致。';
      else if(['head_k','head_v'].includes(measure))source+=' · 共享 KV 头 '+lookup.data.kv_group+' · 来源位置';
      else if(isHead)source+=' · 查询位置';
    }
  }else if(!isHead&&example?.snapshot===snapshot&&example.block===block&&Number.isInteger(site)&&example.positions[site]){
    contextKey=example.run+':'+example.caseId+':'+site;
    source+=' · Phase '+example.phase+' · '+example.caseId+' · token '+example.positions[site].index;
    if(measure==='mlp_activation'){
      values=example.field('mlp',block+1,site);
      if(values?.length===size)comparisons=example.positions.map((p,i)=>({
        label:EXAMPLE_SITES[i]+' · token '+p.index,active:i===site,
        value:example.field('mlp',block+1,i)?.[selected]
      }));
    }
    else missing='此样例只归档了门控后的 FFN 激活，缺少单独的 Gate 和 Up 原场。';
  }else if(!isHead&&sample && sample.d?.model_snapshot===snapshot && !sample.live){
    contextKey='MA-H2:'+sample.d.i+':'+position;
    source+=' · MA-H2 样本 #'+sample.d.i+' · token '+sample.d.tokpos?.[position];
    if(mlpOk && measure==='mlp_activation' && block<sample.NB)values=sample.mlp(block);
    else missing='冻结样本只在答案位置收录了门控后的 FFN 原场。';
  }else if(example)missing=`该记录只发布 B${example.block} 的 FFN 激活，未采集当前对象。`;
  else missing=isHead?'当前快照与位置缺少所选 Head 的原场。请选择随包归档或包含模块读出的轨迹。':
    '当前快照与位置缺少所选块的 FFN 原场。请选择随包归档或包含模块读出的轨迹。';
  if(values && (values.length!==size || !values.every(Number.isFinite))){
    values=null;comparisons=null;missing='归档向量含非有限值，或长度与原生维度不一致。';
  }
  const name=choices.find(x=>x[0]===measure)[1];
  return <section className="ow-native-inspector" aria-label="原生全场读数">
    {isHead&&<label>Head 范围 <select aria-label="Head 范围" value={headScope} onChange={event=>setHeadScope(event.target.value)}>
      <option value="head">单个 Head</option><option value="layer">整层 Head</option>
    </select></label>}
    {!atlas&&<label>测量量 <select aria-label="全场测量量" value={measure}
      onChange={event=>(isHead?setHeadMeasure:setFfnMeasure)(event.target.value)}>
      {choices.map(([id,title])=><option key={id} value={id}>{title}</option>)}
    </select></label>}
    {haveObject?<FieldCanvas identity={[snapshot,contextKey,kind,block,wholeHeads?'layer':isHead?head:'ffn',measure].join(':')}
      kind={kind} size={size} values={partialHead||values} selected={selected} source={source} missing={missing} measure={name}
      comparisons={comparisons} headColumns={wholeHeads?headWidth:null}
      onPick={index=>{if(wholeHeads){onHead?.(Math.floor(index/headWidth));onChannel?.(index%headWidth);}
        else if(isHead)onChannel?.(index);else onPick?.(index);}}/>:
      <p role="status">{missing}</p>}
    {archiveSha&&(values||partialHead)&&<details><summary>核对原始归档</summary>
      {bundleData.manifestSha256&&<p>BF16 原始位型 · 清单 SHA-256 <code>{bundleData.manifestSha256}</code></p>}
      {bundleData.receiptSha256&&<p>中心技术归档回执 SHA-256 <code>{bundleData.receiptSha256}</code></p>}
      <p>原始数组或块 SHA-256 <code>{archiveSha}</code></p>
      {bundled.data.trace&&<p>轨迹 <code>{bundled.data.trace}</code></p>}
      {archiveText&&<p>{archiveText}</p>}
    </details>}
  </section>;
}
