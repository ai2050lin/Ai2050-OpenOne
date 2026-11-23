import AttentionPatternExplorer from './AttentionPatternExplorer.jsx';
import RepresentationGeometry, {TraceRepresentationGeometry} from './RepresentationGeometry.jsx';
import {archiveHeadCase, EXAMPLE_SITES} from './GraphWorkspaceEvidence.jsx';
import {useNative} from './MechanismTraceExplorer.jsx';

function TraceAttention({trace,snapshot,head,block,query,onPin}) {
  const id=trace.run.id;
  const path='/api/traces/'+encodeURIComponent(id)+'/attention-row?block='+block+'&head='+head+'&query='+query;
  const {data,error}=useNative(trace.base,path);
  const source='轨迹 '+id+' · 快照 '+snapshot.slice(0,12)+' · B'+block+'/H'+head+' · 查询 token '+query;
  if(!data)return <AttentionPatternExplorer source={source} reason={error||'正在读取该查询位置的注意力行…'}/>;
  // A trace only contains tokens up to this generation step. Later keys in a
  // full-prefix row must have zero mass under the decoder causal mask.
  const weights=data.weights,tokens=trace.run.tokens;
  const valid=data.schema==='mechanism_attention_row.v1' && data.trace_id===id &&
    data.block===block && data.head===head && data.query_position===query && (data.model_snapshot==null||data.model_snapshot===snapshot) &&
    Number.isFinite(data.validation) && data.validation>=0 && data.validation<=.05 &&
    Array.isArray(weights) && weights.length===tokens.length && weights.slice(query+1).every(x=>x===0);
  return valid?<AttentionPatternExplorer source={source} weights={weights.slice(0,query+1)} labels={tokens.slice(0,query+1)} onPin={onPin}/>:
    <AttentionPatternExplorer source={source} reason="返回的轨迹、位置、来源长度或重建校验与当前选择不符。"/>;
}

export function WorkspaceAttention({snapshot,block,head,headCase,archiveHead,selection,sample,trace,query,reason,onPin}) {
  if(block<0 || head==null)return <AttentionPatternExplorer reason="请在共享 3D 空间选中一个 Head，或按 B层号/H头号 定位。"/>;
  if(archiveHead){
    const r=archiveHeadCase(selection,headCase);
    if(!r)return <AttentionPatternExplorer reason="所选条件没有随包归档的注意力分布。"/>;
    const rows=r.order.map(index=>r.mass[index]);
    const remainder=1-rows.reduce((sum,p)=>sum+p,0);
    return <AttentionPatternExplorer source={'Phase 2983 · '+r.id+' · B26/H36 · 最后输入 token · 快照 '+snapshot.slice(0,12)}
      weights={[...rows,remainder]} labels={[...rows.map((_,i)=>'第 '+(i+1)+' 行'),'其他 token 合计']} grouped onPin={onPin}/>;
  }
  if(trace)return <TraceAttention trace={trace} snapshot={snapshot} block={block} head={head} query={query} onPin={onPin}/>;
  if(sample && head<sample.NH && block<sample.NB){
    return <AttentionPatternExplorer weights={sample.att(block,head)} onPin={onPin}
      source={'MA-H2 冻结样本 #'+sample.d.i+' · B'+block+'/H'+head+' · 答案位置查询 · 快照 '+snapshot.slice(0,12)}/>;
  }
  return <AttentionPatternExplorer reason={reason||'当前输入和 Head 缺少匹配的注意力归档。'}/>;
}

export function WorkspaceGeometry({snapshot,layer,layerCount,width,example,sample,trace,reason,onPin}) {
  if(trace){
    const positions=[...new Set(trace.run.readout||[])].filter(i=>Number.isInteger(i)&&i>=0&&i<trace.run.tokens.length)
      .map(i=>({key:i,label:'Token '+i}));
    return <TraceRepresentationGeometry base={trace.base} traceId={trace.run.id} snapshot={snapshot}
      dimensions={width} positions={positions} layer={layer} layerCount={layerCount} onPin={onPin}/>;
  }
  if(example){
    const positions=example.positions.map((p,i)=>({key:i,label:EXAMPLE_SITES[i]+' · '+p.index}));
    return <RepresentationGeometry key={example.caseId} positions={positions} read={(i,l)=>example.field('hs',l,i)}
      layer={layer} layerCount={layerCount} onPin={onPin} source={'Phase '+example.phase+' · '+example.caseId+' · 快照 '+snapshot.slice(0,12)}/>;
  }
  if(sample){
    const seen=new Set(), positions=[];
    sample.d.tokpos.forEach((position,i)=>{
      if(Number.isInteger(position)&&position>=0&&position<sample.T&&!seen.has(position)){
        seen.add(position);positions.push({key:i,label:(sample.d.positions?.[i]||'Token')+' · '+position});
      }
    });
    return <RepresentationGeometry key={sample.d.i} positions={positions} read={sample.hs} layer={layer}
      layerCount={layerCount} onPin={onPin} source={'MA-H2 冻结样本 #'+sample.d.i+' · 快照 '+snapshot.slice(0,12)}/>;
  }
  return <RepresentationGeometry layer={layer} layerCount={layerCount}
    reason={reason||'当前记录没有同一输入的多个 HiddenState 位置。可在实时模式回放 B36 实测样例。'}/>;
}
