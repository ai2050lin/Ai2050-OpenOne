import { useEffect,useRef,useState } from 'react';
import { getJson,useApi } from '../app/rdc_fusion/api.js';
import { nodeBridge,nodeCall } from '../app/rdc_fusion/contributeUtil.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import { EVIDENCE_LABEL,fragmentLocation,snapshotMatches,responseCompatible } from './researchFragmentUtil.js';

export default function ResearchFocus({model,onLocate,onResponse}){
  const q=new URLSearchParams(window.location.search),sha=q.get('artifact'),id=q.get('feature');
  const doc=useApi(sha?`/api/research-library/artifacts/${sha}`:null);
  const card=doc.data?.feature_cards?.find(c=>c.id===id);
  const [state,setState]=useState({}),[evidenceOpen,setEvidenceOpen]=useState(false);
  const generation=useRef(0);
  useEffect(()=>{generation.current+=1;return()=>{generation.current+=1}},[model?.id,sha,id]);
  const evidence=useApi(evidenceOpen&&card?`/api/research-library/artifacts/${card.evidence.artifact_sha256}`:null);
  const [responseRequest,setResponseRequest]=useState({sha:'',key:0});
  const [traceSteps,setTraceSteps]=useState({});
  const responseSha=responseRequest.sha;
  const response=useApi(responseSha?`/api/research-library/artifacts/${responseSha}`:null);
  const compatible=responseCompatible(response.data,card,doc.data?.model_snapshot,model?.config);
  const verified=state.verified&&state.model===model?.id;
  useEffect(()=>{
    if(compatible&&verified&&onResponse)onResponse({...response.data,tokens:response.data.tokens??responseRequest.tokens,local_model_id:model.id});
  },[compatible,verified,response.data,responseRequest.key,responseRequest.tokens,model?.id,onResponse]);
  if(!sha)return null;
  async function locate(){
    const request=generation.current;
    setState({busy:true,model:model?.id});
    try{
      const position=fragmentLocation(card,model?.config);
      if(!position)throw Error('此特征的坐标空间或边界不支持当前参数定位');
      const path=`/parameters/identity?model=${encodeURIComponent(model.id)}`;
      const actual=nodeBridge()?await nodeCall('GET',path):await getJson(`/api${path}`,{online:true,timeoutMs:120000});
      if(request!==generation.current)return;
      if(!snapshotMatches(doc.data.model_snapshot,actual))throw Error('完整权重快照不一致，未绑定本机参数。请选择对应权重目录。');
      onLocate(position);setState({verified:true,model:model.id});
    }catch(e){if(request===generation.current)setState({error:e.message})}
  }
  return <section className="v8-card v8-research-focus" aria-label="研究碎片定位">
    <h3>{card?.label||'研究碎片'}</h3>{doc.error?<Empty>{doc.error}<ExtraPackPrompt pack={doc.missingPack}/></Empty>:!doc.data?<Empty>读取研究版本…</Empty>:!card?<Empty>该版本没有对应特征，请从研究库重新选择。</Empty>:<>
      <p className="v8-note">{EVIDENCE_LABEL[card.evidence.level]} · {card.boundary} · B{card.layer} / U{card.index?.unit} · {card.run_id}</p>
      <p className="v8-note">研究权重快照 {doc.data.model_snapshot}。首次核验读取完整权重文件，可能需要数十秒。</p>
      <div className="v6-toolbar"><button className="fw-tbtn" disabled={(state.busy&&state.model===model?.id)||!model} onClick={locate}>{state.busy&&state.model===model?.id?'正在核验全部权重…':'核验本机权重并定位'}</button><button className="fw-tbtn" onClick={()=>setEvidenceOpen(!evidenceOpen)}>{evidenceOpen?'收起证据':'查看证据与输入'}</button></div>
      {state.error&&<Empty>{state.error}</Empty>}{state.verified&&state.model===model?.id&&<p className="v8-note" role="status">✓ 完整快照一致，已定位到对应层与原生单元；当前色彩是索引选中态。</p>}
      {response.error&&<Empty>{response.error}<ExtraPackPrompt pack={response.missingPack}/></Empty>}
      {response.data&&!compatible&&<Empty>响应的快照、边界、读出或完整阵列维数不匹配，未叠加到参数空间。</Empty>}
      {evidenceOpen&&(evidence.error?<Empty>{evidence.error}<ExtraPackPrompt pack={evidence.missingPack}/></Empty>:!evidence.data?<Empty>读取证据…</Empty>:<details open><summary>研究证据 · {card.evidence.artifact_sha256.slice(0,12)}</summary>
        <p className="v8-note">反例索引 {card.counterexamples?.length||0} 条{Number.isInteger(card.counterexamples_total)?`，完整报告共 ${card.counterexamples_total} 条`:''}；因果检验未进行时不画因果路径。</p>
        {evidence.data.sample_preview?.length>0&&<div className="v8-sample-list">{evidence.data.sample_preview.map(s=>{const activation=s.activation_by_layer?.[card.layer]??s.activation;const traces=s.trace_responses?.[card.layer]||[];const selected=traces.find(t=>t.step===traceSteps[s.id])||traces[0];const responseRef=selected?.sha256||s.responses?.[card.layer]||s.response_sha256;return <details key={s.id}><summary>{s.entity} · {s.split}{Number.isFinite(activation)?` · A=${activation.toPrecision(5)}`:''}</summary><p className="v8-note">实际输入 · {s.tokens} tokens · {s.id}{s.entity_split?` · 实体划分 ${s.entity_split}`:''}</p><pre className="v8-sample-text">{s.text}</pre>{traces.length>0&&<label>生成步 <select aria-label={`生成步 ${s.id}`} value={selected.step} onChange={e=>setTraceSteps({...traceSteps,[s.id]:Number(e.target.value)})}>{traces.map(t=><option key={t.step} value={t.step}>第{t.step}步 · 预测token {t.token_id}</option>)}</select></label>}{responseRef&&<button className="fw-tbtn" disabled={!verified} onClick={()=>{setResponseRequest(prev=>({sha:responseRef,key:prev.key+1,tokens:s.tokens+(selected?.step||0)}));setEvidenceOpen(false)}}>显示该输入的完整MLP响应</button>}</details>})}</div>}
        <details><summary>完整结构化报告与原场哈希</summary><pre className="v8-sample-text">{JSON.stringify(evidence.data,null,2)}</pre></details>
      </details>)}
    </>}
  </section>;
}
