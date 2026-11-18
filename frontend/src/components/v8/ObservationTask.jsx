import { useState } from 'react';
import { TASKS, taskPair, compareTaskRuns } from './observationTasks.js';
import ResearchDisclosure from './ResearchDisclosure.jsx';

export default function ObservationTask({ execute, observe, preview, disabled, busy }) {
  const [kind,setKind]=useState('binding'),[variant,setVariant]=useState(0),[rows,setRows]=useState(()=>taskPair('binding'));
  const [results,setResults]=useState({}),[error,setError]=useState(''),[pending,setPending]=useState('');
  const build=(k,v)=>{setKind(k);setVariant(v);const next=taskPair(k,v);setRows(next);setResults({});setError('');preview(next[0].text);};
  const run=async row=>{setPending(row.id);setError('');try{const result=await execute(row.text,row.condition);setResults(s=>({...s,[row.id]:result}));}catch(e){setError(e.message);}finally{setPending('');}};
  const a=results[rows[0].id],b=results[rows[1].id],comparison=compareTaskRuns(a,b);
  return <section className="aw-task" aria-label="自动文本任务">
    <div className="v6-toolbar"><label>任务 <select aria-label="观察任务类型" value={kind} disabled={busy||!!pending} onChange={e=>build(e.target.value,variant)}>{TASKS.map(t=><option key={t.id} value={t.id}>{t.name}</option>)}</select></label><button type="button" className="fw-tbtn" disabled={busy||!!pending} onClick={()=>build(kind,variant+1)}>生成另一组文本</button><b>{Object.keys(results).length} / 2 已运行</b></div>
    <p>{TASKS.find(t=>t.id===kind).question}</p>
    <p className="v8-note">探索任务 · 2 条长文本，逐条手动运行，每条最多 1024 输入 token。使用本机 Qwen3-4B、占用 GPU，并保存原生轨迹；超长输入由服务拒绝，不截断。这里只检验下一 token，不是完整回答或正式研究结论。</p>
    {disabled&&<p className="v8-note">{disabled}</p>}
    {rows.map(row=>{const result=results[row.id],top=result?.top?.[0];return <div className="aw-task-row" key={row.id}>
      <b>条件 {row.condition}</b><span>{row.text.length} 字符 · 目标颜色：{row.expected}</span>
      <button type="button" className="fw-tbtn" disabled={busy||!!pending} onClick={()=>preview(row.text)}>查看文本</button>
      <button type="button" className="fw-tbtn primary" disabled={!!disabled||busy||!!pending} onClick={()=>run(row)}>{pending===row.id?'采集与分析中…':result?'重新运行':'运行并分析'}</button>
      {result&&<><button type="button" className="fw-tbtn" disabled={busy||!!pending} onClick={()=>observe(result,row.text,row.condition)}>在3D中观察</button><span>下一 token：{top?JSON.stringify(top.text):'缺失'} · {result.tokens.length} token · {result.elapsed} ms</span><small>下一 token 颜色匹配：{!top?'未检验':top.text.trim().startsWith(row.expected)?'是（非完整答案判定）':'否/单个 token 不足以判定'}</small></>}
    </div>;})}
    {error&&<p role="alert" className="ac-err">{error}；已完成结果保留，可重试本条。</p>}
    <p className="v8-note">{comparison.reason}</p>
    {comparison.layers.length>0&&<><h4>原生坐标逐层对照</h4><p className="v8-note">相对变化 = ‖B−A‖₂ / ‖A‖₂。点击某层，在3D查看条件 B；颜色深浅不表示因果重要性。L0 是词嵌入基线，末层已过最终归一化。</p>
      <div className="aw-task-layers">{comparison.layers.map(l=><button type="button" className="fw-tbtn" key={l.layer} disabled={busy||!!pending} title={`cos=${l.cosine??'未定义'}；相对变化=${l.relativeL2??'未定义'}`} onClick={()=>observe(b,rows[1].text,rows[1].condition,l.layer)}>L{l.layer}<br/>{l.relativeL2?.toFixed(3)??'—'}</button>)}</div>
      <ResearchDisclosure title="全部层数值与分析口径"><table className="v6-table"><thead><tr><th>层</th><th>余弦相似度</th><th>相对 L2</th></tr></thead><tbody>{comparison.layers.map(l=><tr key={l.layer}><td>{l.layer}</td><td>{l.cosine?.toFixed(6)??'未定义'}</td><td>{l.relativeL2?.toFixed(6)??'未定义'}</td></tr>)}</tbody></table><p>全部原生坐标参与计算，没有用 Top-K 定义结构。仅两条探索输入，无独立留出、统计检验或因果干预；不能据此定位语义神经元。任务汇总仅保留在当前页面会话。</p></ResearchDisclosure></>}
    <ResearchDisclosure title="输入、轨迹与原始分析记录"><pre className="v8-artifact-json">{JSON.stringify({schema:'observation_task.v1',kind,variant,scope:'unregistered_exploration',rows:rows.map(r=>({...r,trace:results[r.id]?.id,snapshot:results[r.id]?.snapshot,readout:results[r.id]?.readout,top:results[r.id]?.top})),analysis:comparison},null,2)}</pre></ResearchDisclosure>
  </section>;
}
