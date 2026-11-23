import { Suspense, lazy, useEffect, useState } from 'react';
import { ADMIN_LIVE, API_BASE, authJson, getSession, useApi } from '../app/rdc_fusion/api.js';
import { useAuth } from '../app/rdc_fusion/authContext.js';
import './test_records.css';
import RoleRereadArchive from './RoleRereadArchive.jsx';
import ResearchDirections from './ResearchDirections.jsx';
import { readDatabaseLocation, writeDatabaseLocation } from './testDatabaseLocation.js';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import ArchivedCaseResults from './ArchivedCaseResults.jsx';
import {nativeAddress} from './workspaceRecordReplay.js';
import {useWorkspaceCatalog,useWorkspaceResource} from './nativeObservationArchive.js';
import WorkspaceEvidenceSearch from './WorkspaceEvidenceSearch.jsx';

const base = '/api/test-archive';
const ThreeCueComparison = lazy(()=>import('./ThreeCueComparison.jsx'));
const ArchivedHeadCase = lazy(()=>import('./GraphWorkspaceEvidence.jsx').then(module=>({default:module.ArchivedHeadCase})));
const FirstRecordArchive = lazy(()=>import('./FirstRecordArchive.jsx'));
const ResearchInterventionReport = lazy(()=>import('./ResearchInterventionReport.jsx'));
const pageSize = 12;
const decisions = {new_acquisition:'新增采集', independent_replication:'独立复现', analysis_only:'复用采集做新分析', recapture:'重新采集'};
const reasons = {same_analysis_key:'分析键相同',same_acquisition_key:'采集键相同',shared_language_family:'语言家族重合',shared_domain:'研究领域重合'};
const suggestions = {reuse_results:'可复用结果',reuse_acquisition:'可复用原场',already_running:'已有采集正在运行',
  recapture_required:'所需原场缺失或退役',review_incomplete:'需检查未完成记录',review_similar:'需比较方案'};
const advancedKeys=['snapshot','target','method','archive_state','artifact_role','decision'];
const phases = id => id.replace(/^gf-/, 'Phase ');
const privateApi = (path, options) => authJson(base+path, { ...options, redirect:'error' });
function fromCenter(run) {
  const p = run.plan;
  return {id:run.id, title:p.title, question:p.family_ids.join('、'), objective:p.objective, principle:p.principle,
    type:decisions[run.decision], status:run.state, model:p.acquisition.model_id, snapshot:p.acquisition.model_snapshot,
    mechanisms:[...p.domains,...p.family_ids],result:run.final?.summary,
    gate:p.criteria, counter:run.final?.counterexamples, novelty:run.reason, refs:p.history_refs,
    sourceRun:run.source_run_id, plan:p, preflightId:run.preflight_id, runs:[{id:run.id,activity:run.state, result:run.final?.summary}],
    archive:run.archive_state,review:run.receipt?.evidence_status || 'unreviewed',
    reusableField:run.archive_state==='server_verified'&&run.receipt?.artifacts?.some(artifact=>artifact.role==='native_field'&&artifact.availability==='uploaded'),
    caseFile:run.receipt?.artifacts?.find(artifact=>artifact.role==='case_results' && artifact.availability==='uploaded' &&
      artifact.name.endsWith('.jsonl') && /^[a-f0-9]{64}$/.test(artifact.sha256))};
}
function fields(row) {
  return [row.id,row.phase,row.title,row.question,row.type,row.model,row.status,row.objective,row.principle,row.result,row.counter,row.gate,...(row.mechanisms || []),...(row.objects || []),...(row.runs || []).map(r=>r.id)].join(' ').toLowerCase();
}
function pilotMatches(records,q) {
  const parts = q.toLowerCase().trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return records;
  const tokens = new Set(parts.flatMap(s => s.match(/[a-z0-9/_-]+|[\u3400-\u9fff]{2}/g)||[]));
  return records.map(row => {
    const text = fields(row);
    const exact = parts.filter(s=>text.includes(s)).length;
    const count = [...tokens].filter(s=>text.includes(s)).length;
    return {row, score:exact*100+count};
  }).filter(r=>r.score>0).sort((a,b)=>b.score-a.score).map(r=>r.row);
}
function Detail({row,onSelect,onRetest,onReplay,onOpenNativeArchive,archives}) {
  const [receipt,setReceipt]=useState(null),[receiptError,setReceiptError]=useState('');
  const sources = [row.protocol,row.evidence,...(row.source_versions || []),...(row.runs || []).map(r=>r.source)].filter(Boolean);
  const links = new Set([...(row.related || []),...(row.refs || [])]);
  return <article className="tr-detail" aria-label="测试设计与实际运行">
    {row.plan?.family_ids?.some(id => ['CLT-ROLE-REREAD','COLOR-NAMING-COMPARISON'].includes(id)) && row.preflightId && <RoleRereadArchive row={row}/>}
    <small>{[row.question || row.group,row.phase && 'Phase '+row.phase,row.type,row.status].filter(Boolean).join(' · ')}</small>
    <h4>{row.title}</h4>
    {row.mechanisms?.length>0 && <p>目标语言机制 · {row.mechanisms.join('、')}</p>}
    {row.objects && <p>内部计算对象 · {row.objects.join('、')}</p>}
    <dl>{[['目标',row.objective],['测试原理',row.principle],['实际结果',row.result],['反例与限制',row.counter],['行为门与未过门原因',row.gate],
      ['本次复测新增验证',row.novelty],['来源运行',row.sourceRun],['技术归档',row.archive],['科学复核',row.review]].map(([label,value],i)=>(i<5 || value) && <div key={label}><dt>{label}</dt><dd>{typeof value==='string' && value || '来源未单列此项'}</dd></div>)}</dl>
    {row.stages && <ul>{row.stages.map(([name,state])=><li key={name}>{name} · {state}</li>)}</ul>}
    <h5>实际运行与数据来源</h5>{row.publication && <p>{row.publication}</p>}{row.runs?.length ? <ul>{row.runs.map(r=><li key={r.id}>{r.activity} · {r.identity || '运行 ID'} <code>{r.id}</code> {r.model} {r.result} {r.note}</li>)}</ul> : <p>{row.type==='历史综述' || row.status==='阶段综述'?'本轮为历史复核，没有新增模型运行。':'没有独立运行 ID 绑定到这条索引。'}</p>}
    {links.size>0 && <div className="tr-related">关联测试 {[...links].map(id=><button type="button" key={id} onClick={()=>onSelect(id)}>{id.startsWith('gf-')?phases(id):id}</button>)}</div>}
    {row.earlier?.length>0 && <details><summary>逐轮历史记录（{row.earlier.length}）</summary><div className="tr-list">{row.earlier.map(item=><button type="button" key={item.id} onClick={()=>onSelect(item.id)}>{item.label} · {item.title}</button>)}</div></details>}
    <details><summary>复核出处与版本</summary><p>记录 <code>{row.id}</code> {row.model}</p>{row.memo && <p>{row.memo} · Phase {row.phase}<br/><code>{row.phase_sha256}</code>（段落正文）</p>}
      {row.snapshot && <p>模型快照 <code>{row.snapshot}</code></p>}
      {row.material?.map(m=><p key={m.split}>材料 {m.split} · {m.families} 家族 <code>{m.sha256}</code></p>)}
      {sources.map(s=><p key={s.path+s.sha256}>{s.path}<br/><code>{s.sha256}</code></p>)}{row.other_sources?.map(s=><p key={s}>同内容出处 · {s}</p>)}{row.plan && <pre>{JSON.stringify(row.plan,null,2)}</pre>}</details>
    {row.public_report && <p><a href={API_BASE+'/api/research-library/artifacts/'+row.public_report} target="_blank" rel="noreferrer">打开成果库同一运行的报告</a></p>}
    {row.caseFile && <ArchivedCaseResults key={row.caseFile.sha256} artifact={row.caseFile} run={row.id} snapshot={row.snapshot} onReplay={onReplay} onOpenArchive={onOpenNativeArchive}
      archiveEntry={archives.find(item=>item.run===row.id&&item.snapshot===row.snapshot&&item.caseId)}/>}
    {onReplay&&row.plan&&!row.caseFile&&<p className="v8-note">该运行没有可读取的逐题产物。可查看计划、结果摘要和归档状态。</p>}
    {row.preflightId && <details><summary>原始查重回执</summary><code>{row.preflightId}</code>
      <button type="button" onClick={async()=>{try{setReceiptError('');setReceipt(await privateApi('/preflights/'+encodeURIComponent(row.preflightId)));}catch(e){setReceiptError(e.message);}}}>读回当时的候选与匹配理由</button>
      {receiptError && <p role="alert">{receiptError}</p>}{receipt && <pre>{JSON.stringify(receipt,null,2)}</pre>}</details>}
    <button type="button" onClick={()=>onRetest(row)}>以此记录新建复测</button>
    {row.reusableField&&<button type="button" onClick={()=>onRetest({...row,initialDecision:'analysis_only'})}>基于此原场新建分析</button>}
  </article>;
}
function comparePlan(oldPlan, plan) {
  function canonical(value) {
    if(Array.isArray(value))return value.map(canonical);
    if(value && typeof value==='object')return Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])]));
    return value;
  }
  const entries = [['目标',oldPlan.objective,plan.objective],['材料版本',oldPlan.dataset,plan.dataset],
    ['模型快照',oldPlan.acquisition.model_snapshot,plan.acquisition.model_snapshot],
    ['输入与采集',oldPlan.acquisition,plan.acquisition],['分析方法',oldPlan.analysis,plan.analysis],['判定标准',oldPlan.criteria,plan.criteria]];
  return entries.map(([k,a,b])=><div key={k}><dt>{k}</dt><dd>{JSON.stringify(canonical(a))===JSON.stringify(canonical(b))?'相同':<details><summary>有变化</summary><pre>此前 {JSON.stringify(a,null,2)}{'\n'}本次 {JSON.stringify(b,null,2)}</pre></details>}</dd></div>);
}

function NewTest({prior,admin,pilot,byId,onCreated,onCancel}) {
  const [term,setTerm]=useState(prior?.phase || ''), [reference,setReference]=useState(prior);
  const [raw,setRaw]=useState(()=>prior?.initialDecision?JSON.stringify(prior.plan,null,2):''), [novelty,setNovelty]=useState(''),
    [decision,setDecision]=useState(prior?.initialDecision||'new_acquisition');
  const [receipt,setReceipt]=useState(null),[match,setMatch]=useState(null),[source,setSource]=useState(prior?.plan ? prior.id : '');
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[runId,setRunId]=useState('');
  let spec;
  try {spec=JSON.parse(raw);} catch { /* The editable JSON can be incomplete. */ }
  const matching=pilotMatches(pilot.records,term || spec?.title || '').slice(0,7);
  function invalidate(){setReceipt(null);setMatch(null);setError('');setRunId('');}
  function choose(row){invalidate();setReference(row);setSource(row.plan ? row.id : '');}
  async function action(fn){setBusy(true);setError('');try{await fn();}catch(e){setError(e.message);}finally{setBusy(false);}}
  async function preflight(){
    invalidate();
    if (!spec || !spec.dataset || !spec.acquisition || !spec.analysis || !spec.objective) throw Error('请导入完整的 research_test_plan.v1 冻结方案。');
    const plan=reference && !reference.plan ? {...spec,history_refs:[...new Set([...(spec.history_refs || []),reference.id])]} : spec;
    const r=await privateApi('/preflight',{method:'POST',body:plan});
    if (!r.preflight_id || !Array.isArray(r.matches)) throw Error('中心没有返回有效查重回执。');
    setReceipt(r);setRunId('record-'+crypto.randomUUID());
  }
  async function register(){
    const body={preflight_id:receipt.preflight_id,run_id:runId,decision,reason:novelty.trim(),source_run_id:source};
    const run=await privateApi('/runs',{method:'POST',body});
    if(run.id!==runId || run.preflight_id!==receipt.preflight_id)throw Error('登记回执身份不匹配。');
    onCreated(fromCenter(run));
  }
  return <section aria-label="新建测试" className="tr-create"><h4>新建测试</h4>
    <fieldset disabled={busy}><label>先查相似记录<input value={term} onChange={e=>setTerm(e.target.value)} placeholder="机制、测试类型或 Phase"/></label>
    <div className="tr-related">{matching.map(r=><button key={r.id} onClick={()=>choose(r)}>{phases(r.id)} · {r.type}</button>)}</div>
    {reference && <div className="tr-compare"><b>此前验证</b><p>{reference.title}</p><p>{reference.principle}</p><p>{reference.result}</p><p>{reference.gate}</p></div>}
    <label>导入冻结方案（JSON）<input type="file" accept=".json,application/json" onChange={e=>{const f=e.target.files[0];if(f)action(async()=>{invalidate();if(f.size>512*1024)throw Error('方案超过 512 KiB。');setRaw(await f.text());});}}/></label>
    <label>冻结方案内容<textarea rows="4" value={raw} onChange={e=>{invalidate();setRaw(e.target.value);}} placeholder="research_test_plan.v1"/></label>
    {spec?.objective && <p><b>本次目标</b> {spec.objective}</p>}
    <button disabled={!admin || !ADMIN_LIVE || !raw.trim()} onClick={()=>action(preflight)}>查询中心并冻结方案</button>
    {(!admin || !ADMIN_LIVE) && <p>浏览 Goodfire 记录可离线使用。中心查重和运行登记需要管理员登录可写研究工作台。</p>}
    {receipt && <><h5>中心查重回执</h5><code>{receipt.preflight_id}</code><p>中心运行匹配 {receipt.matches.length} 条，历史索引建议 {receipt.historical_matches?.length ?? '未提供'} 条。</p>
      <div className="tr-related">{receipt.matches.map(m=><button key={m.run_id} onClick={()=>action(async()=>{const run=await privateApi('/runs/'+encodeURIComponent(m.run_id));setMatch(fromCenter(run));setSource(run.id);})}>{m.title} · {m.reasons.map(x=>reasons[x] || x).join('、')}{suggestions[m.suggestion]&&' · '+suggestions[m.suggestion]}{m.new_validation&&<small>{m.new_validation}</small>}</button>)}
      {(receipt.historical_matches || []).map(m=><button key={m.id} onClick={()=>action(async()=>{const old=byId.get(m.id) || await privateApi('/history/'+encodeURIComponent(m.id));choose(old);})}>{m.title} · {m.type}{m.new_validation&&<small>{m.new_validation}</small>}</button>)}</div>
      {match && <><p>对照运行 {match.id}</p><dl>{comparePlan(match.plan,spec)}</dl></>}
      <label>复测方式<select value={decision} onChange={e=>setDecision(e.target.value)}>{Object.entries(decisions).map(([k,v])=><option value={k} key={k}>{v}</option>)}</select></label>
      <label>来源运行 ID<input value={source} onChange={e=>setSource(e.target.value)} placeholder="复用采集时必填"/></label></>}
    <label>这次多验证了什么<textarea value={novelty} onChange={e=>setNovelty(e.target.value)} placeholder="例如新增两套未见句式，仍按原行为门检验反转。"/></label>
    <button disabled={!receipt || !novelty.trim() || (decision==='analysis_only'&&!source)} onClick={()=>action(register)}>登记测试</button>
    <button onClick={onCancel}>返回记录</button></fieldset>{error && <p role="alert">{error}</p>}{busy && <p role="status">正在处理…</p>}
  </section>;
}

export default function TestRecords({model,sharedLocation=false,onOpenNativeArchive,workspaceSearch,onReplay}) {
  const {user}=useAuth();
  return <Ledger key={user?.user_id || user?.role || 'guest'} admin={user?.role==='admin' && Boolean(getSession())}
    model={model} sharedLocation={sharedLocation} onOpenNativeArchive={onOpenNativeArchive} workspaceSearch={workspaceSearch} onReplay={onReplay}/>;
}
function Ledger({admin,model,sharedLocation,onOpenNativeArchive,workspaceSearch,onReplay}) {
  const [initial]=useState(()=>sharedLocation ? readDatabaseLocation() : {});
  const {archives}=useWorkspaceCatalog();
  const goodfire = useWorkspaceResource('goodfire-records');
  const selection = useWorkspaceResource(initial.record||initial.scope==='published'?'selection-cases':null);
  const first = useWorkspaceResource(initial.record||initial.scope==='published'?'first-record':null).data;
  const headExample=archives.find(item=>item.presentation==='selection');
  const focus=initial.record;
  const publication=selection.data?.evidence?.runs?.find(item=>item.id===focus||String(item.phase)===focus);
  const publishedFocus=publication?.id||(focus===first?.runId?focus:'');
  const pilot=goodfire.data||{records:[],groups:{}};
  const byId=new Map(pilot.records.map(row=>[row.id,row]));
  const sceneSnapshot=admin&&/^[a-f0-9]{64}$/.test(workspaceSearch?.snapshot||'') ? workspaceSearch.snapshot : '';
  const sceneModel=admin&&!sceneSnapshot ? workspaceSearch?.modelId || '' : '';
  const [scope,setScope]=useState(workspaceSearch?'evidence':initial.fromHead?'published':focus?.startsWith('record-')?'center':initial.scope || 'evidence');
  const [query,setQuery]=useState(workspaceSearch?.target || (focus?.startsWith('record-')?focus:'') || initial.query || ''),[group,setGroup]=useState(initial.group || ''),[page,setPage]=useState(0);
  const [onlyModel,setOnlyModel]=useState(false),[kind,setKind]=useState(initial.kind || '');
  const [modelFilter,setModelFilter]=useState(sceneModel || initial.model || ''),[part,setPart]=useState(initial.part || '');
  const [domain,setDomain]=useState(initial.domain || ''),[state,setState]=useState(initial.state || '');
  const [row,setRow]=useState(null);
  const [focusId,setFocusId]=useState(focus || '');
  const [selected,setSelected]=useState(()=>initial.selected?.slice(0,4) || []);
  const [draft,setDraft]=useState(undefined),[remote,setRemote]=useState(null),[error,setError]=useState(''),[refresh,setRefresh]=useState(0);
  const [centerFamily,setCenterFamily]=useState(initial.centerFamily || '');
  const [advanced,setAdvanced]=useState(sceneSnapshot ? {...initial.advanced,snapshot:sceneSnapshot} : initial.advanced || {});
  const [publishedRun,setPublishedRun]=useState(publishedFocus);
  useEffect(()=>{
    if(!publishedFocus)return;
    setScope('published');setPublishedRun(publishedFocus);setFocusId(publishedFocus);
  },[publishedFocus]);
  const local=pilotMatches(pilot.records,query).filter(r=>(!group || r.group===group) && (!kind || r.type===kind)
    && (!part || fields(r).includes(part.trim().toLowerCase()))
    && (!modelFilter || r.model.toLowerCase()===modelFilter.toLowerCase())
    && (!onlyModel || (model && r.model.toLowerCase()===model.toLowerCase())));
  const requestKey=JSON.stringify([scope,query,page,refresh,centerFamily,modelFilter,domain,state,advanced]);
  useEffect(()=>{
    if(sharedLocation)writeDatabaseLocation({source:scope,q:query,group,method:kind,part,model:modelFilter,domain,state,family:centerFamily,
      ...Object.fromEntries(advancedKeys.map(key=>['run_'+key,advanced[key]]))},
      row?.id || focusId,selected);
  },[sharedLocation,scope,query,group,kind,part,modelFilter,domain,state,centerFamily,advanced,row,focusId,selected]);
  useEffect(()=>{
    if((scope!=='center' && scope!=='history') || !admin) return;
    let current=true;
    const args=new URLSearchParams({q:query,offset:String(page*pageSize),limit:String(pageSize)});
    if(scope==='center') {
      if(centerFamily) args.set('family',centerFamily);
      if(modelFilter) args.set('model',modelFilter);
      if(domain) args.set('domain',domain);
      if(state) args.set('state',state);
      for(const key of advancedKeys)if(advanced[key])args.set(key,advanced[key]);
    }
    const timer=setTimeout(()=>privateApi((scope==='center'?'/runs?':'/history?')+args).then(r=>{
      if(!Array.isArray(r.items))throw Error('中心未提供此索引，请核对中心版本。');
      if(scope==='center'&&advancedKeys.some(key=>args.has(key)&&r.applied_filters?.[key]!==args.get(key)))
        throw Error('当前中心未应用高级筛选。清除高级筛选可继续查询。');
      if(current)setRemote({key:requestKey,total:r.total,items:scope==='center'?r.items.map(fromCenter):r.items});
    }).catch(e=>{if(current)setError(e.message);}),180);
    return ()=>{current=false;clearTimeout(timer);};
  },[scope,query,page,refresh,admin,requestKey,centerFamily,modelFilter,domain,state,advanced]);
  const isLocal=scope==='goodfire';
  const data=remote?.key===requestKey ? remote : null;
  const items=isLocal?local.slice(page*pageSize,(page+1)*pageSize):data?.items || [];
  const total=isLocal?(goodfire.data?local.length:undefined):data?.total;
  const currentRow=row || (focusId && items.find(item=>item.id===focusId || item.phase===focusId));
  const chooseScope=s=>{setScope(s);setRow(null);setFocusId('');setPage(0);setQuery(workspaceSearch?.target||'');
    setError('');setCenterFamily('');setModelFilter(s==='center'?sceneModel:'');};
  function filter(fn,value){fn(value);setRow(null);setFocusId('');setPage(0);setError('');}
  const advance=(key,value)=>filter(setAdvanced,{...advanced,[key]:value.trim()});
  async function follow(id){
    if(byId.has(id)){setScope('goodfire');setGroup('');setKind('');setOnlyModel(false);setModelFilter('');setPart('');setQuery(id);setPage(0);setRow(byId.get(id));return;}
    if(!admin){setError('逐轮历史归档需要中心管理员会话。');return;}
    try{setRow(await privateApi('/history/'+encodeURIComponent(id)));}catch(e){setError(e.message);}
  }
  async function openEvidenceRecord(id,kind){
    if(kind==='study'){await follow(id);return;}
    if(kind==='public'){
      chooseScope('published');setPublishedRun(id);setFocusId(id);return;
    }
    setError('');
    try{
      const row=kind==='private' ? fromCenter(await privateApi('/runs/'+encodeURIComponent(id)))
        : await privateApi('/history/'+encodeURIComponent(id));
      chooseScope(kind==='private'?'center':'history');setQuery(id);setRow(row);
    }catch(error){setError(error.message);}
  }
  async function reuseEvidence(id){
    setError('');
    try{
      const run=fromCenter(await privateApi('/runs/'+encodeURIComponent(id)));
      if(!run.reusableField)throw Error('当前运行没有登记可复用的原场。');
      setDraft({...run,initialDecision:'analysis_only'});
    }catch(error){setError(error.message);}
  }
  if(sharedLocation && initial.fromHead && initial.caseId && scope==='published' && Boolean(publishedFocus)){
    return <section className="tr-panel" aria-label="测试记录"><Suspense fallback={<p>正在读取历史原题…</p>}>
      <ArchivedHeadCase runId={focusId} caseId={initial.caseId} model={initial.headModel} fullscreen={initial.headFullscreen}/>
    </Suspense></section>;
  }
  if(draft!==undefined)return <NewTest prior={draft} admin={admin} pilot={pilot} byId={byId}
    onCancel={()=>setDraft(undefined)} onCreated={r=>{chooseScope('center');setQuery(r.id);setRow(r);setRefresh(v=>v+1);setDraft(undefined);}}/>;
  return <section className="tr-panel" aria-label="测试记录"><div className="tr-heading"><h3>测试记录</h3><button className="fw-tbtn primary" onClick={()=>setDraft(null)}>新建测试</button></div>
    {workspaceSearch&&<p className="v8-note">来自 3D · {workspaceSearch.modelLabel}{workspaceSearch.target&&' · '+workspaceSearch.target}</p>}
    {headExample&&workspaceSearch?.target===nativeAddress(headExample.location)&&onOpenNativeArchive&&scope!=='published'&&<button type="button" className="fw-tbtn"
      onClick={()=>{setPublishedRun(headExample.run);chooseScope('published');}}>查看此部件的已发布原题</button>}
    {onOpenNativeArchive&&<details className="tr-native-replays"><summary>回放已归档的原生读数</summary>
      <ul>{archives.map(archive=><li key={archive.id}>
        <button type="button" className="fw-tbtn" onClick={()=>onOpenNativeArchive(archive.id)}>{archive.title}</button>
        <span>{archive.coverage}</span>
      </li>)}</ul>
    </details>}
    <div className="tr-controls"><label>浏览目录<select aria-label="记录来源" value={scope} onChange={e=>chooseScope(e.target.value)}><option value="evidence">统一证据检索</option><option value="directions">研究方向树</option><option value="goodfire">Goodfire 系列</option><option value="published">已发布选源与干预</option><option value="history">历史回填</option><option value="center">中心运行</option></select></label>
    {scope!=='evidence' && scope!=='directions' && scope!=='published' && <label className="tr-search-field">搜索测试<input value={query} maxLength={500} onChange={e=>filter(setQuery,e.target.value)} placeholder="例如 1111、B26/H36、授权"/></label>}
    {scope==='center' && centerFamily && <div>模式族 {centerFamily} <button type="button" onClick={()=>filter(setCenterFamily,'')}>清除模式族筛选</button></div>}
    {isLocal && <><label>机制问题<select value={group} onChange={e=>filter(setGroup,e.target.value)}><option value="">所有机制问题</option>{Object.entries(pilot.groups).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
    <label>测试类型<select value={kind} onChange={e=>filter(setKind,e.target.value)}><option value="">全部类型</option>{[...new Set(pilot.records.map(r=>r.type))].map(t=><option key={t}>{t}</option>)}</select></label>
    {sharedLocation && <><label>模型<select value={modelFilter} onChange={e=>filter(setModelFilter,e.target.value)}><option value="">全部模型</option>{[...new Set(pilot.records.map(r=>r.model))].map(name=><option key={name}>{name}</option>)}</select></label>
      <label>模型部件<input value={part} placeholder="B26/H36" onChange={e=>filter(setPart,e.target.value)}/></label></>}
    {model && <label><input type="checkbox" checked={onlyModel} onChange={e=>filter(setOnlyModel,e.target.checked)}/>仅当前模型 {model}</label>}
    <button type="button" className="fw-tbtn" onClick={goodfire.reload}>同步 Goodfire 记录</button>
    {goodfire.local&&<p role="status">正在查看上次同步的公开记录。</p>}{goodfire.error&&<p role="alert">{goodfire.error}</p>}</>}</div>
    {scope==='center' && admin && <><div className="tr-controls"><label>模型标识<input value={modelFilter} onChange={e=>filter(setModelFilter,e.target.value)}/></label>
      <label>研究领域<select value={domain} onChange={e=>filter(setDomain,e.target.value)}><option value="">全部</option><option value="KN">知识网络</option><option value="RS">功能推理</option><option value="GR">语法规则</option></select></label>
      <label>模式族 ID<input value={centerFamily} onChange={e=>filter(setCenterFamily,e.target.value)}/></label>
      <label>运行状态<select value={state} onChange={e=>filter(setState,e.target.value)}><option value="">全部</option><option value="running">运行中</option><option value="completed">已完成</option><option value="failed">失败</option><option value="interrupted">中断</option></select></label></div>
      <details className="tr-native-replays"><summary>高级筛选</summary><div className="tr-controls">
        <label>模型快照<input maxLength={64} value={advanced.snapshot||''} onChange={e=>advance('snapshot',e.target.value)}/></label>
        <label>原生地址<input placeholder="B26/H36" maxLength={20} value={advanced.target||''} onChange={e=>advance('target',e.target.value)}/></label>
        <label>方法版本<input maxLength={200} value={advanced.method||''} onChange={e=>advance('method',e.target.value)}/></label>
        <label>复测方式<select value={advanced.decision||''} onChange={e=>advance('decision',e.target.value)}><option value="">全部</option>{Object.entries(decisions).map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label>
        <label>归档状态<select value={advanced.archive_state||''} onChange={e=>advance('archive_state',e.target.value)}><option value="">全部</option><option value="server_verified">服务端已校验</option><option value="missing_artifacts">缺少产物</option><option value="pending">待提交</option></select></label>
        <label>有已上传产物<select value={advanced.artifact_role||''} onChange={e=>advance('artifact_role',e.target.value)}><option value="">不限</option><option value="case_results">逐题结果</option><option value="native_field">原生场文件</option><option value="evidence">证据</option><option value="dictionary">字典</option></select></label>
        <button type="button" className="fw-tbtn" onClick={()=>filter(setAdvanced,{})}>清除高级筛选</button>
      </div><p className="v8-note">原生地址匹配冻结方案中明确登记的地址。</p></details></>}
    {scope==='directions'&&<div><ResearchDirections onRecord={follow} admin={admin}
      onFamily={family=>{chooseScope('center');setCenterFamily(family);}}/></div>}
    {scope==='center' && focusId && !currentRow && <PublicRun run={focusId}/>}
    {scope==='evidence' ? <>
      {error&&<p role="alert">{error}</p>}
      <WorkspaceEvidenceSearch query={query} onQuery={setQuery} admin={admin} workspaceSearch={workspaceSearch}
        onRecord={openEvidenceRecord} onOpenArchive={onOpenNativeArchive} onReuse={reuseEvidence}/>
    </> : scope==='directions' ? null : scope==='published' ? <Suspense fallback={<p>正在读取已发布证据…</p>}>
      {publishedRun&&publishedRun===first?.runId ? <FirstRecordArchive/> : <ThreeCueComparison key={publishedRun} initialRunId={publishedRun} initialCaseId={initial.caseId}
        onOpenCase={onOpenNativeArchive&&headExample?(caseId)=>onOpenNativeArchive(headExample.id,{caseId,scene:true}):undefined}/>}
    </Suspense>
    : !isLocal&&!admin ? <p>历史回填和中心运行需要中心管理员会话。访客可浏览 Goodfire 系列及已发布选源与干预。</p> :
    <><p role="status">{total===undefined ? error?'读取失败':'正在读取…' : total+' 条记录'}{scope==='history'?' · 备忘录条目和产物按来源标注，独立运行另列。':''}</p>
    {error && <p role="alert">{error} <button onClick={()=>{setError('');setRefresh(v=>v+1);}}>重试</button></p>}
    {isLocal && sharedLocation && selected.length>0 && <ResearchComparison byId={byId} selected={selected} onClear={()=>setSelected([])}/>}
    <div className="tr-layout"><div className="tr-list">{items.map(r=><button key={r.id} aria-pressed={currentRow?.id===r.id} onClick={()=>{setRow(r);setFocusId('');}}><small>{r.phase?'Phase '+r.phase+' · ':''}{r.type} · {r.status}</small><strong>{r.title}{r.id.match(/-v\d+$/)?.[0].replace('-',' · ')}</strong>
      {isLocal && <span className="tr-excerpt">{[r.objective,r.principle,r.result].filter(Boolean).join(' ')}</span>}</button>)}{total===0 && <p>当前筛选没有匹配记录。</p>}
    {total>pageSize && <div className="tr-page"><button disabled={!page} onClick={()=>{setPage(p=>p-1);setRow(null);}}>上一页</button> {page+1} / {Math.ceil(total/pageSize)} <button disabled={(page+1)*pageSize>=total} onClick={()=>{setPage(p=>p+1);setRow(null);}}>下一页</button></div>}</div>
    {currentRow && <div>{isLocal&&sharedLocation&&byId.has(currentRow.id)&&<label className="tr-compare-pick"><input type="checkbox" checked={selected.includes(currentRow.id)}
      disabled={selected.length>=4&&!selected.includes(currentRow.id)}
      onChange={e=>setSelected(e.target.checked?[...selected,currentRow.id]:selected.filter(id=>id!==currentRow.id))}/>加入跨测试比较</label>}
      <Detail key={currentRow.id} row={currentRow} onSelect={follow} onRetest={setDraft} onReplay={onReplay} onOpenNativeArchive={onOpenNativeArchive} archives={archives}/></div>}</div></>}
  </section>;
}

function PublicRun({run}) {
  const index=useApi('/api/research-library/workpackages');
  const entry=index.data?.workpackages?.find(item=>item.body.run_id===run)?.body;
  const sha=entry?.stages?.registration_validated?.report_sha256;
  if(index.loading)return <p role="status">正在查找此运行的公开报告…</p>;
  if(!sha)return <p className="v8-note">当前公开索引未提供该运行的报告。运行 <code>{run}</code></p>;
  return <ResearchDisclosure title={entry.title || run} initiallyOpen>
    <Suspense fallback={<p>正在读取公开报告…</p>}><ResearchInterventionReport sha={sha} rawFallback hideScene/></Suspense>
  </ResearchDisclosure>;
}

function ResearchComparison({byId,selected,onClear}) {
  const entries=selected.map(id=>byId.get(id));
  return <details className="tr-compare"><summary>跨测试比较（{entries.length} 项）</summary>
    <p>逐项核对目标、模型、结果与限制。各项结果沿用原有口径。</p>
    <button type="button" className="fw-tbtn" onClick={onClear}>清空比较</button>
    <div className="tr-comparison-grid">{entries.map(row=><article key={row.id}><h4>{row.title}</h4>
      <dl>{[['模型',row.model],['目标',row.objective],['结果',row.result],['反例与限制',row.counter]]
        .map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value || '来源未单列此项'}</dd></div>)}</dl></article>)}</div>
  </details>;
}

