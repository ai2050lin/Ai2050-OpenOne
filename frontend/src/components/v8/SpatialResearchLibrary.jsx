import {useCallback,useEffect,useMemo,useState} from 'react';
import {API_BASE,useApi} from '../app/rdc_fusion/api.js';
import {useResearchContext} from '../app/researchContext.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import ResearchInterventionReport from './ResearchInterventionReport.jsx';
import {EVIDENCE_COLORS,EVIDENCE_NAMES,researchLocation} from './researchLocation.js';
import {useMainline} from './data/mainlineState.js';

const datasetOf=o=>o.body?.scope?.dataset||o.body?.dataset||o.dataset||'';
const titleOf=o=>o.title||o.body?.label||o.body?.relation||o.source||o.id||o.run_id;
export default function SpatialResearchLibrary({onLocate,onOverlay}) {
  const context=useResearchContext(),{revision}=useMainline();
  const [snapshot,setSnapshot]=useState(context.snapshot),[run,setRun]=useState(context.run),[level,setLevel]=useState(''),[dataset,setDataset]=useState('');
  const [entry]=useState(()=>{const q=new URLSearchParams(typeof window==='undefined'?'':window.location.search);return {feature:q.get('feature'),artifact:q.get('artifact')};});
  const [query,setQuery]=useState(entry.feature||''),[kind,setKind]=useState(entry.feature?'objects':entry.artifact?'artifacts':'studies'),[page,setPage]=useState(0),[overlay,setOverlay]=useState(false),[selection,setChosen]=useState(undefined),[message,setMessage]=useState('');
  const [lookupQuery,setLookupQuery]=useState(entry.feature||'');
  useEffect(()=>{const timer=window.setTimeout(()=>setLookupQuery(query),300);return()=>window.clearTimeout(timer);},[query]);
  const objectQuery=new URLSearchParams({objects_only:'true',object_limit:'12',object_offset:String(kind==='objects'?page*12:0),model:snapshot||'',run_id:run||'',query:lookupQuery,level,dataset});
  const api=useApi('/api/traces/research-space?object_limit=1');
  const objectApi=useApi(`/api/traces/research-space?${objectQuery}`);
  const objectCurrent=Boolean(objectApi.data&&objectApi.data.mainline_revision===api.data?.mainline_revision);
  const data=useMemo(()=>api.data?{...api.data,objects:objectCurrent?objectApi.data.objects:[],objects_page:objectCurrent?objectApi.data.objects_page:null}:null,[api.data,objectApi.data,objectCurrent]);
  const linkedObject=data?.objects.find(o=>o.id===entry.feature&&(!snapshot||o.model_snapshot===snapshot));
  const linkedArtifact=data?.artifacts.find(a=>a.sha===entry.artifact);
  const chosen=selection===undefined?(linkedObject||linkedArtifact||null):selection;
  const {reload:reloadIndex}=api,{reload:reloadObjects}=objectApi;
  const reload=useCallback(()=>{reloadIndex();reloadObjects();},[reloadIndex,reloadObjects]);useEffect(()=>{reload();},[revision,reload]);
  const models=data?.models;
  const model=models?.find(m=>m.snapshot===snapshot);
  const ready=Boolean(data&&(!revision||data.mainline_revision===revision));
  const select=useCallback(row=>{setChosen(row);setMessage('');},[]);
  const objects=useMemo(()=>!ready?[]:data.objects.filter(o=>(!snapshot||o.model_snapshot===snapshot)&&(!run||o.body.run_id===run)&&(!level||o.body.evidence?.level===level)&&(!dataset||(datasetOf(o)||'unlabelled')===dataset)&&(!query||[titleOf(o),o.id,o.sha].join(' ').toLowerCase().includes(query.toLowerCase()))),[data,ready,snapshot,run,level,dataset,query]);
  const points=useMemo(()=>objects.flatMap(o=>{const location=researchLocation(o,model);return location?[{...location,id:o.id,sha:o.sha,level:o.body.evidence?.level,label:titleOf(o)}]:[];}),[objects,model]);
  const pickPoint=useCallback(point=>{const row=objects.find(o=>o.id===point.id&&o.sha===point.sha);if(row)select(row);},[objects,select]);
  useEffect(()=>{onOverlay?.(overlay&&ready?{snapshot,points,onPick:pickPoint}:null);},[onOverlay,overlay,ready,snapshot,points,pickPoint]);
  useEffect(()=>()=>onOverlay?.(null),[onOverlay]);
  const allStudies=data?.studies||[];
  const studies=allStudies.filter(s=>(!snapshot||!s.model_snapshot||s.model_snapshot===snapshot)&&(!run||s.run_id===run)&&(!query||[s.title,s.run_id].join(' ').toLowerCase().includes(query.toLowerCase())));
  const artifacts=(data?.artifacts||[]).filter(a=>(!snapshot||!a.model||a.model===snapshot)&&(!run||a.source?.startsWith(run+'/'))&&(!dataset||(datasetOf(a)||'unlabelled')===dataset)&&(!query||[a.source,a.sha,a.kind].join(' ').toLowerCase().includes(query.toLowerCase())));
  const rows=kind==='studies'?studies:kind==='objects'?objects:artifacts;
  const pagedObjects=kind==='objects';
  const rowCount=pagedObjects?(data?.objects_page?.total||0):rows.length;
  const change=fn=>value=>{fn(value);setPage(0);setChosen(null);setMessage('');};
  const target=chosen?.type==='edge'?data?.objects.find(o=>o.id===chosen.body.target&&o.model_snapshot===chosen.model_snapshot):chosen;
  const targetModel=models?.find(m=>m.snapshot===target?.model_snapshot);
  const location=researchLocation(target,targetModel);
  const study=chosen?.study_key?chosen:null;
  const sha=study?.report_sha256||chosen?.body?.evidence?.artifact_sha256||chosen?.sha;
  const datasets=[...new Set([...(data?.objects||[]),...(data?.artifacts||[])].map(datasetOf).filter(Boolean))];
  if(api.error)return <Empty>{api.error}；成果库未连接，不显示模拟研究。<button className="fw-tbtn" onClick={reload}>重新读取成果库</button></Empty>;
  if(!ready)return <Empty>{data?'成果库与研究记录版本尚未同步。':'读取成果库…'}<button className="fw-tbtn" onClick={reload}>重新读取成果库</button></Empty>;
  return <section className="v8-card aw-library" aria-label="成果库">
    <h3>成果库</h3><p className="v8-note">{data.counts.studies} 个研究工作包 · {data.counts.objects} 个图谱对象 · {data.counts.artifacts} 个产物版本。对象数量不代表机制破解率。</p>
    <div className="v6-toolbar"><label>模型 <select aria-label="成果模型" value={snapshot} onChange={e=>{change(setSnapshot)(e.target.value);setRun('');}}><option value="">全部模型</option>{snapshot&&!models.some(m=>m.snapshot===snapshot)&&<option value={snapshot}>{snapshot.slice(0,12)} · 未登记结构</option>}{models.map(m=><option key={m.snapshot} value={m.snapshot}>{m.snapshot.slice(0,12)} · {m.n_layers||'未知'} 层</option>)}</select></label>
      <label>研究 <select aria-label="成果研究" value={run} onChange={e=>change(setRun)(e.target.value)}><option value="">全部研究</option>{run&&!allStudies.some(s=>s.run_id===run)&&<option value={run}>{run} · 未登记工作包</option>}{allStudies.filter(s=>!snapshot||!s.model_snapshot||s.model_snapshot===snapshot).map(s=><option key={s.study_key} value={s.run_id}>{s.title}</option>)}</select></label>
      <button className="fw-tbtn" onClick={()=>{setSnapshot('');setRun('');setDataset('');setLevel('');setQuery('');setPage(0);setChosen(null);}}>查看全部成果</button><button className="fw-tbtn" onClick={reload}>刷新成果库</button>
    </div>
    <div className="v6-toolbar"><input aria-label="成果检索" value={query} onChange={e=>change(setQuery)(e.target.value)} placeholder="研究、特征、来源或 SHA"/><label>数据集 <select aria-label="成果数据集" value={dataset} onChange={e=>{change(setDataset)(e.target.value);if(kind==='studies')setKind('objects');}}><option value="">全部数据标注</option><option value="unlabelled">未标注数据集</option>{datasets.map(d=><option key={d}>{d}</option>)}</select></label><label>证据 <select aria-label="成果证据等级" value={level} onChange={e=>{change(setLevel)(e.target.value);setKind('objects');}}><option value="">全部证据等级</option>{Object.entries(EVIDENCE_NAMES).map(([v,n])=><option value={v} key={v}>{n}</option>)}</select></label></div>
    <div className="v6-toolbar" role="group" aria-label="成果类型">{[['studies','研究报告'],['objects','定位卡与关联'],['artifacts','全部产物版本']].map(([id,title])=><button type="button" className="fw-tbtn" key={id} aria-pressed={kind===id} onClick={()=>{setKind(id);setPage(0);setChosen(null);}}>{title}</button>)}</div>
    <label className="v8-note"><input type="checkbox" checked={overlay} disabled={!model?.n_layers} onChange={e=>setOverlay(e.target.checked)}/> 在上方空间显示成果分布</label><p className="v8-note">{model?`${points.length} 个卡片有可核验的原生定位；${objects.filter(o=>o.type==='feature').length-points.length} 个特征卡尚无受支持的单元定位。`:'选择模型后可叠加原生定位。'} 未定位报告、边和历史版本仍保留在列表中。图层只在上方模型快照一致时显示。</p>
    {overlay&&<div className="v6-toolbar">{Object.entries(EVIDENCE_NAMES).map(([k,n])=><span key={k} style={{color:EVIDENCE_COLORS[k]}}>● {n}</span>)}</div>}
    <p className="v8-note">当前筛选 {rowCount} 项{data.objects_page?.filtered?' · 定位卡按页读取，空间叠加显示当前页；完整单元阵列在研究报告中查看。':''}</p><div className="aw-library-list">{(pagedObjects?rows:rows.slice(page*12,(page+1)*12)).map(row=><button type="button" className="fw-tbtn" key={row.study_key||`${row.sha}:${row.id||''}`} onClick={()=>select(row)}>{titleOf(row)}<small>{row.body?.evidence?.level?EVIDENCE_NAMES[row.body.evidence.level]||row.body.evidence.level:row.sha?.slice(0,12)||row.run_id}</small></button>)}</div>
    {(objectApi.loading||lookupQuery!==query)&&<p role="status">正在读取定位卡，筛选条件可继续编辑…</p>}
    {objectApi.data&&!objectCurrent&&<p role="status">定位卡与研究记录版本尚未同步。<button className="fw-tbtn" onClick={reload}>重新同步</button></p>}
    {objectApi.error&&<Empty>定位卡未读取：{objectApi.error}<button className="fw-tbtn" onClick={reloadObjects}>重试定位卡</button></Empty>}
    {!rows.length&&!(pagedObjects&&(objectApi.loading||objectApi.error))&&<Empty>当前筛选没有匹配成果；可清除筛选查看未定位和历史记录。</Empty>}
    <div className="v6-toolbar"><button className="fw-tbtn" disabled={!page} onClick={()=>setPage(p=>p-1)}>上一页</button><span>{page+1} / {Math.max(1,Math.ceil(rowCount/12))}</span><button className="fw-tbtn" disabled={(page+1)*12>=rowCount} onClick={()=>setPage(p=>p+1)}>下一页</button></div>
    {chosen&&<section className="v8-card" aria-label="选中成果"><h4>{titleOf(chosen)}</h4>{location?<button className="fw-tbtn primary" onClick={()=>setMessage(onLocate?.({snapshot:target.model_snapshot,run:target.body.run_id,trace:target.body.scope?.trace_id,location})||'已定位到上方空间，可在对象详情查看参数和研究记录。')}>定位到当前空间</button>:<p className="v8-note">此项没有可直接定位的原生单元；报告和原始记录完整保留。</p>}{message&&<p role="status">{message}</p>}
      {study?.traces?.map(t=><p key={t.trace_id}><a href={`?lens=spatial&view=live&snapshot=${study.model_snapshot||''}&trace=${t.trace_id}&tblock=${t.layer||0}&tunit=${t.unit||0}&thead=${t.head||0}`}>打开轨迹 {t.sample_id||t.trace_id.slice(0,12)}</a></p>)}
      {sha&&<ResearchDisclosure key={sha} title="完整报告与原始证据"><ResearchInterventionReport sha={sha} rawFallback hideScene planSha={study?.stages?.native_mlp_plan?.plan_sha256} capabilityInputsSha={study?.stages?.client_acceptance?.capability_inputs_sha256}/><a href={`${API_BASE}/api/research-library/artifacts/${sha}`} target="_blank" rel="noreferrer">打开原始产物</a></ResearchDisclosure>}
      <ResearchDisclosure key={`metadata:${chosen.sha||chosen.study_key}`} title="版本、来源与定位记录"><pre className="v8-artifact-json">{JSON.stringify(chosen,null,2)}</pre></ResearchDisclosure>
    </section>}
  </section>;
}
