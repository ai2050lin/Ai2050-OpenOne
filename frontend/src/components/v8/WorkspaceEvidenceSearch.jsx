import { useEffect, useState } from 'react';
import { authJson, useApi } from '../app/rdc_fusion/api.js';
import { useWorkspaceCatalog } from './nativeObservationArchive.js';
import { workbenchHref } from './v8Util.js';

const endpoint = '/api/research/workspace';
const sources = {study:'机制研究',run:'实际运行',case:'逐题记录',history:'历史出处'};
const levels = {source:'已索引',run:'已关联运行',case:'逐题结果',native:'可回放原场',reviewed:'已科学复核'};
const keys = ['snapshot','target','family','position','token_id'];
const labels = {snapshot:'模型快照',target:'原生部件',family:'材料家族',position:'Token 位置',token_id:'Token ID'};
const pageSize = 12;

function useEvidence(path, admin) {
  const publicResult = useApi(admin || !path ? null : endpoint+path);
  const [privateResult,setPrivate] = useState(null);
  useEffect(()=>{
    if(!admin||!path) return undefined;
    const request = new AbortController();
    authJson(endpoint+'/private'+path,{signal:request.signal,redirect:'error',timeoutMs:20000})
      .then(data=>{if(!request.signal.aborted)setPrivate({path,data});})
      .catch(error=>{if(!request.signal.aborted)setPrivate({path,error:error.message});});
    return ()=>request.abort();
  },[admin,path]);
  if(!admin) return publicResult;
  return privateResult?.path===path ? privateResult : {loading:Boolean(path)};
}

function requestedLocation(point, target) {
  const match = /^B(\d+)\/([HU])(\d+)$/.exec(target||'');
  if(!match) return point.location;
  return {layer:Number(match[1])+1,module:match[2]==='H'?'attn':'mlp',index:Number(match[3])};
}

function Playback({item,point,target,onOpenArchive}) {
  const location = requestedLocation(point,target);
  const text = `3D 原值 · token ${point.position}（ID ${point.token_id}）`;
  const caseId = point.presentation==='selection'?item.case_id:'';
  const site = point.presentation==='path-example'?Number(point.row):null;
  if(onOpenArchive) return <button type="button" className="fw-tbtn" onClick={()=>onOpenArchive(point.archive,
    {caseId,scene:true,location,row:point.row,site})}>{text}</button>;
  const params = new URLSearchParams({lens:'spatial',view:'graph',gview:'structure',snapshot:item.snapshot,
    observeMode:'archive',observeArchive:point.archive,observeArchiveRow:point.row,
    observeTab:'plugin',observePlugin:location.module==='attn'?'heads':'mlp',
    sblock:location.layer,smodule:location.module,sindex:location.index});
  if(caseId)params.set('headCase',caseId);
  if(site!=null)params.set('observeExampleSite',String(site));
  return <a className="fw-tbtn" href={workbenchHref(params.toString())}>{text}</a>;
}

function Coverage({data}) {
  const kinds = Object.entries(sources).filter(([key])=>data.scope_totals[key]);
  return <details><summary>本次可查范围</summary>
    <table className="v6-table" aria-label="证据覆盖统计"><thead><tr><th>索引类型</th>{Object.values(levels).map(label=><th key={label}>{label}</th>)}</tr></thead>
      <tbody>{kinds.map(([kind,label])=><tr key={kind}><th scope="row">{label}</th>
        {Object.keys(levels).map(level=><td key={level}>{data.coverage_counts[level]?.[kind]||0}</td>)}</tr>)}</tbody></table>
    <p className="v8-note">研究、运行、题目和出处可互相引用，按类型分别计数。逐题结果与可回放原场按中心实际读取的范围统计。</p>
    {data.historical_links>0&&<p className="v8-note">{data.historical_links} 条历史出处已按原文哈希关联到对应研究记录。</p>}
    {data.omissions?.budget>0&&<p className="v8-note">部分大文件超出本次索引预算，可从运行详情读取。未纳入文件 {data.omissions.budget} 项。</p>}
    {data.omissions?.unreadable>0&&<p className="v8-note">{data.omissions.unreadable} 个逐题文件缺失、损坏或格式暂未适配。</p>}
    {data.omissions?.identity>0&&<p className="v8-note">{data.omissions.identity} 条逐题编号或快照未能绑定。</p>}
  </details>;
}

function EvidenceDetail({item,target,onOpenArchive,onRecord,onReuse}) {
  const [answer,setAnswer] = useState(item.source!=='case');
  return <article className="tr-detail" aria-label="证据链详情">
    <h4>{item.title}</h4>
    <p className="v8-note">{item.topic}{item.model&&` · ${item.model}`}{item.phase&&` · Phase ${item.phase}`}</p>
    {item.case_id&&<p><code>{item.case_id}</code>{item.family&&` · 家族 ${item.family}`}{item.arm&&` · ${item.arm}`}</p>}
    {item.input&&<p className="tr-evidence-text">{item.input}</p>}
    {item.source==='case'&&item.output&&<><button className="fw-tbtn" type="button" aria-pressed={answer} onClick={()=>setAnswer(!answer)}>
      {answer?'收起输出与反例':'查看模型输出与反例'}</button>
      {answer&&<p>实际输出 <code>{item.output}</code>{typeof item.correct==='boolean'&&(item.correct?' · 判对':' · 判错')}</p>}</>}
    {item.prompt&&<details><summary>原始模型输入</summary><pre className="tr-evidence-text">{item.prompt}</pre></details>}
    <dl>{[['目标',item.objective],['原理',item.principle],['结果',item.result],['反例与限制',answer?item.counter:null],
      ['行为门',item.gate],['相对前轮新增验证',item.novelty],['回放覆盖',item.note]].filter(([,value])=>value)
      .map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
    <div className="tr-related">
      {item.readouts?.map(point=><Playback key={`${point.archive}:${point.row}`} item={item} point={point}
        target={target} onOpenArchive={onOpenArchive}/>)}
      {item.detail_source==='goodfire'&&<button type="button" onClick={()=>onRecord(item.id,'study')}>完整研究记录</button>}
      {item.source==='history'&&<button type="button" onClick={()=>onRecord(item.id,'history')}>查看历史出处</button>}
      {item.study_id&&<button type="button" onClick={()=>onRecord(item.study_id,'study')}>打开同源 Goodfire 记录</button>}
      {item.run_id&&item.visibility==='private'&&<button type="button" onClick={()=>onRecord(item.run_id,'private')}>查看中心运行与产物</button>}
      {item.run_id&&item.visibility==='public'&&item.targets?.length>0&&<button type="button" onClick={()=>onRecord(item.run_id,'public')}>查看已发布课题</button>}
      {item.source_case&&<button type="button" onClick={()=>onRecord(`${item.source_case.run} ${item.source_case.case}`,'lookup')}>查看同题原档</button>}
      {item.earlier&&<button type="button" onClick={()=>onRecord(item.earlier,'lookup')}>查看前序记录</button>}
      {item.linked_runs?.map(id=><button key={id} type="button" onClick={()=>onRecord(id,'lookup')}>运行 {id}</button>)}
      {item.visibility==='private'&&item.run_id&&item.coverage.includes('native')&&<button type="button" onClick={()=>onReuse(item.run_id)}>基于此原场新建分析</button>}
    </div>
    {item.proofs?.length>0&&<details><summary>已核对的原始出处</summary>
      <ul>{item.proofs.map(proof=><li key={`${proof.path}:${proof.sha256}`}>{proof.path}<br/><code>{proof.sha256}</code></li>)}</ul></details>}
    {item.field_notes?.length>0&&<details><summary>原生文件留存状态</summary><ul>{item.field_notes.map((note,index)=><li key={index}>
      {{uploaded:'文件已上传',retired_not_archived:'已退役',historically_missing:'历史缺件'}[note]||note}</li>)}</ul></details>}
    {item.run_id&&<p className="v8-note">运行 <code>{item.run_id}</code>{item.review==='unreviewed'&&' · 科学复核待登记'}</p>}
    {item.snapshot&&<p className="v8-note">快照 <code>{item.snapshot}</code></p>}
  </article>;
}

export default function WorkspaceEvidenceSearch({query,onQuery,admin,workspaceSearch,onRecord,onOpenArchive,onReuse}) {
  const [filters,setFilters] = useState(()=>({snapshot:workspaceSearch?.snapshot||'',target:workspaceSearch?.target||''}));
  const [page,setPage] = useState(0);
  const [selected,setSelected] = useState('');
  const [path,setPath] = useState('');
  const [revision,setRevision] = useState(0);
  const {catalog} = useWorkspaceCatalog();
  const params = new URLSearchParams({limit:pageSize,offset:page*pageSize});
  if(revision)params.set('refresh',revision);
  if(query.trim())params.set('q',query.trim());
  for(const [key,value] of Object.entries(filters))if(value)params.set(key,value);
  const nextPath = '/evidence?'+params;
  useEffect(()=>{
    const timer = setTimeout(()=>setPath(nextPath),180);
    return ()=>clearTimeout(timer);
  },[nextPath]);
  const result = useEvidence(path,admin);
  const data = path===nextPath&&result.data?.schema==='workspace_evidence_search.v1'?result.data:null;
  const item = data?.items.find(row=>row.id===selected);
  const change = (key,value) => {setFilters(old=>({...old,[key]:value}));setPage(0);setSelected('');};
  const update = value => {onQuery(value);setPage(0);setSelected('');};
  return <div className="tr-unified" aria-label="统一证据检索">
    <div className="tr-controls">
      <label className="tr-search-field">搜索证据<input value={query} maxLength={500} onChange={event=>update(event.target.value)}
        placeholder="机制、题干、实际输出、反例或运行编号"/></label>
      <label>记录类型<select value={filters.source||''} onChange={event=>change('source',event.target.value)}>
        <option value="">当前可见的全部类型</option>{Object.entries(sources).filter(([key])=>admin||key!=='history')
          .map(([key,label])=><option key={key} value={key}>{label}</option>)}</select></label>
      <label>证据覆盖<select value={filters.coverage||''} onChange={event=>change('coverage',event.target.value)}>
        <option value="">当前可见的全部</option>{Object.entries(levels).map(([key,label])=><option key={key} value={key}>{label}</option>)}</select></label>
      <label><input type="checkbox" checked={filters.counter==='true'} onChange={event=>change('counter',event.target.checked?'true':'')}/>有反例或已登记限制</label>
    </div>
    <details open={keys.some(key=>filters[key])}><summary>按快照、家族、部件和 Token 精确筛选</summary>
      <div className="tr-controls">{keys.map(key=><label key={key}>{labels[key]}
        {key==='snapshot'&&catalog?<select value={filters[key]||''} onChange={event=>change(key,event.target.value)}>
          <option value="">所有快照</option>
          {filters[key]&&!catalog.models.some(model=>model.model_snapshot===filters[key])&&<option value={filters[key]}>{filters[key].slice(0,12)}</option>}
          {catalog.models.map(model=><option key={model.model_snapshot} value={model.model_snapshot}>{model.model} · {model.model_snapshot.slice(0,12)}</option>)}</select>:
          <input value={filters[key]||''} placeholder={key==='target'?'例如 B26/H36':undefined} onChange={event=>change(key,event.target.value)}/>}
      </label>)}</div>
    </details>
    {!workspaceSearch&&!query&&data?.learning_case&&<button type="button" className="fw-tbtn" onClick={()=>{
      setFilters({source:'case',coverage:'native',target:data.learning_case.target});
      onQuery(data.learning_case.case_id);setPage(0);setSelected(data.learning_case.id);
    }}>学习案例 · 读一题选源实验</button>}
    {!admin&&<p className="v8-note">当前检索公开资料。历史索引与私有运行供中心管理员查看。</p>}
    {data?<><p role="status">{data.total} 条匹配{result.local?' · 上次同步的公开检索结果':''}</p><Coverage data={data}/></>:
      result.error?<p role="alert">统一检索暂不可用 · {result.error} <button onClick={()=>setRevision(value=>value+1)}>重试</button></p>:<p role="status">正在查找匹配的证据…</p>}
    {data&&<div className="tr-layout"><div className="tr-list">{data.items.map(row=>
      <button type="button" key={row.id} aria-pressed={item?.id===row.id} onClick={()=>setSelected(row.id)}>
        <small>{sources[row.source]}{row.phase&&` · Phase ${row.phase}`}{row.visibility==='private'&&' · 私有归档'}</small>
        <strong>{row.title}{row.case_id&&` · ${row.case_id}`}</strong>
        <span>{row.coverage.map(level=>levels[level]).join(' · ')}</span></button>)}
      {data.total===0&&<p>没有找到匹配记录。可放宽部件或状态筛选。</p>}
      {data.total>pageSize&&<div className="tr-page"><button disabled={!page} onClick={()=>{setSelected('');setPage(page-1);}}>上一页</button>
        {page+1} / {Math.ceil(data.total/pageSize)}<button disabled={(page+1)*pageSize>=data.total} onClick={()=>{setSelected('');setPage(page+1);}}>下一页</button></div>}
    </div>{item&&<EvidenceDetail key={item.id} item={item} target={filters.target}
      onOpenArchive={onOpenArchive} onRecord={(id,kind)=>{if(kind==='lookup'){setFilters({});update(id);}else onRecord(id,kind);}} onReuse={onReuse}/>}</div>}
  </div>;
}
