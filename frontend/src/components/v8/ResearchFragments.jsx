import { useEffect, useState } from 'react';
import { atMainlineRevision, useMainline } from './data/mainlineState.js';
import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import { EVIDENCE_LABEL,fragmentHref } from './researchFragmentUtil.js';

export default function ResearchFragments({onArtifact}){
  const [offset,setOffset]=useState(0);
  const {revision}=useMainline();
  const data=atMainlineRevision(useApi(`/api/research-library/fragments?offset=${offset}&limit=50`),revision);
  const {reload:reloadFragments}=data;
  useEffect(() => { reloadFragments(); },[revision,reloadFragments]);
  return <section className="v8-fragments"><div className="v6-toolbar"><h3>图谱拼图 · 当前修订</h3><button className="fw-tbtn" onClick={data.reload}>刷新拼图</button></div>
    <p className="v8-note">同一模型与特征保留完整历史，当前显示最近登记版本。负结果会如实替换当前声明；关联不等于因果。</p>
    {data.error?<Empty>{data.error}<ExtraPackPrompt pack={data.missingPack}/></Empty>:!data.data?<Empty>读取拼图…</Empty>:!data.data.features.length?<Empty>尚无显式特征碎片。完成研究后登记特征卡、条件、证据与反例。</Empty>:<div className="v8-fragment-list">{data.data.features.map(f=><article className="v8-card" key={f.id}><div className="v6-toolbar"><h4>{f.body.label||f.id}</h4><span className="v8-fragment-grade">{EVIDENCE_LABEL[f.body.evidence.level]||f.body.evidence.level}</span></div>
      <p className="v8-note">{f.body.run_id} · {f.model_snapshot.slice(0,16)} · {f.body.boundary}</p>
      {f.body.summary?.metrics&&<div className="v8-library-stats">{Object.entries(f.body.summary.metrics).map(([k,v])=><span key={k}><b>{v.auc?.toFixed(3)??'—'}</b>{({confirm:'确认集 AUC',lockbox:'未见实体 AUC',unseen_frame_and_entities:'未见实体与句式 AUC'})[k]||k}<small>{v.n_entities} 个实体家族</small></span>)}</div>}
      <div className="v6-toolbar"><button className="fw-tbtn" onClick={()=>onArtifact(f.body.evidence.artifact_sha256)}>查看原始证据</button><a className="fw-tbtn" href={fragmentHref(f)}>在参数空间定位</a></div>
      <details><summary>反例、限制与 {f.history.length} 次修订</summary><p className="v8-note">因果检验：{f.body.causal_tests==='not_measured'?'未进行':f.body.causal_tests||'未记录'}；当前登记反例 {f.body.counterexamples?.length||0} 条。</p>{f.body.summary?.limitations?.map(t=><p className="v8-note" key={t}>{t}</p>)}{f.history.map(h=><div className="v6-toolbar" key={h.sha}><span>{EVIDENCE_LABEL[h.level]||h.level}</span><button className="fw-tbtn" onClick={()=>onArtifact(h.sha)}>{h.sha.slice(0,12)}</button></div>)}</details>
    </article>)}</div>}
    <div className="v6-toolbar"><button className="fw-tbtn" disabled={!offset} onClick={()=>setOffset(Math.max(0,offset-50))}>上一页</button><span className="v8-note">当前特征 {data.data?.total??'—'}；数量不是破解比例</span><button className="fw-tbtn" disabled={offset+50>=(data.data?.total||0)} onClick={()=>setOffset(offset+50)}>下一页</button></div>
  </section>;
}
