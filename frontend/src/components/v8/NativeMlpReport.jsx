import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import {NativeArray} from './MechanismScene.jsx';

const pct=x=>Number.isFinite(x)?`${(100*x).toFixed(1)}%`:'—';
const num=x=>Number.isFinite(x)?x.toPrecision(4):'—';
const points=x=>Number.isFinite(x)?(100*x).toFixed(2):'—';
const NI={noninferior:'通过非劣效门',inferior:'超过允许损失',inconclusive:'尚未证实能力保持'};
const FOLD=['发现','确认','锁箱'];
const POLE={Gpos:'Gate 正极',Gneg:'Gate 负极',Apos:'A 正极',Aneg:'A 负极'};

function InterventionErrors({doseRef,capabilityRef,inputsRef}){
  const d=useApi(`/api/research-library/artifacts/${doseRef}`),c=useApi(`/api/research-library/artifacts/${capabilityRef}`);
  const inputs=useApi(inputsRef?`/api/research-library/artifacts/${inputsRef}`:null);
  if(d.error||c.error)return <Empty>{d.error||c.error}</Empty>;
  if(!d.data||!c.data)return <p>读取逐条干预与能力记录…</p>;
  const b=new Map(d.data.result.rows.filter(r=>r.condition==='base').map(r=>[r.pair_index,r]));
  const losses=d.data.result.rows.filter(r=>r.condition!=='base'&&b.get(r.pair_index)?.correct&&!r.correct);
  const eb=new Map(c.data.base.map(r=>[r.id,r]));const changes=c.data.edited.filter(r=>eb.get(r.id)?.correct!==r.correct);const questions=new Map((inputs.data?.items||[]).map(q=>[q.id,q]));
  return <><p className="v8-note">自然续写退化记录 {losses.length} 条；能力答案改变 {changes.length} 条（含参考题，逐条标注）。同一输入跨剂量可重复出现。</p>{losses.map((r,i)=><details key={i}><summary>配对 {r.pair_index} · {r.condition} · 原对改错</summary><pre className="v8-artifact-json">{JSON.stringify({base:b.get(r.pair_index),edited:r},null,2)}</pre></details>)}{changes.map(r=><details key={r.id}><summary>{r.id} · {r.gate_eligible?'正式能力题':'参考题不入门'} · {r.correct?'原错改对':'原对改错'}</summary><pre className="v8-artifact-json">{JSON.stringify({question:questions.get(r.id),base:eb.get(r.id),edited:r},null,2)}</pre></details>)}</>;
}

function DosePlot({effects}){
  const names=['base','target_half','target_full'];if(!effects.base)return null;
  const xy=names.map((n,i)=>[60+i*220,130-effects[n].accuracy*100]);
  return <svg viewBox="0 0 570 175" role="img" aria-label="自然续写真实剂量曲线" style={{width:'100%',maxWidth:650,maxHeight:200}}>{[0,.5,1].map(v=><g key={v}><line x1="60" x2="500" y1={130-v*100} y2={130-v*100} stroke="#e2e8f0"/><text x="48" y={134-v*100} textAnchor="end" fontSize="10">{100*v}%</text></g>)}<polyline points={xy.map(p=>p.join(',')).join(' ')} stroke="#2979bf" strokeWidth="2" fill="none"/>{xy.map(([x,y],i)=><g key={i}><circle cx={x} cy={y} r="4" fill="#2979bf"/><text x={x} y={y-12} textAnchor="middle" fontSize="12">{pct(effects[names[i]].accuracy)}</text><text x={x} y="150" textAnchor="middle" fontSize="12">α={i/2}</text></g>)}{effects.random_full&&<><circle cx="500" cy={130-effects.random_full.accuracy*100} r="4" fill="#dd7843"/><text x="490" y="172" textAnchor="end" fontSize="12">同能量随机 α=1：{pct(effects.random_full.accuracy)}</text></>}</svg>;
}

function DocumentContext({index,refs,position}){
  const ref=refs?.[Math.floor(index/32)];const result=useApi(ref?`/api/research-library/artifacts/${ref}`:null);
  if(result.error)return <Empty>{result.error}</Empty>;
  const doc=result.data?.documents.find(r=>r.doc_index===index);
  if(!doc)return <p className="v8-note">读取原始长文…</p>;
  return <><p className="v8-note">{doc.title||doc.doc_id} · {doc.source} · {doc.split} · {doc.license} · 读出 #{position}，实际可见 {position+1} token。{position<512?'短可见前缀：仅作极值探索，不计入长前缀主检验。':'长前缀。'}以下只显示读出时可见的前缀。</p>
    {doc.url&&<a href={doc.url} target="_blank" rel="noreferrer">来源原文与署名</a>}{doc.prefix_texts?.[position]!==undefined?<pre className="v8-sample-text">{doc.prefix_texts[position]}</pre>:<Empty>该位置的可见前缀未归档；请回查分词，不用包含后文的全文替代。</Empty>}
    <details><summary>分词位置、数据版本及原文（含读出之后的内容）</summary><pre className="v8-artifact-json">{JSON.stringify(doc,null,2)}</pre></details></>;
}

export function NativeNeuronCard({sha,documentRefs,referenceTrace}){
  const result=useApi(sha?`/api/research-library/artifacts/${sha}`:null);const [pole,setPole]=useState('2:Gpos'),[error,setError]=useState(0),[open,setOpen]=useState(null);
  if(result.error)return <Empty>{result.error}</Empty>;
  const c=result.data?.card;if(!c)return <Empty>读取原生单元说明书…</Empty>;
  const examples=c.poles[pole]?.examples||[],bad=c.counterexamples[error];
  return <section aria-label="原生MLP单元说明书"><p><strong>B{c.layer} / U{c.unit} · {c.replicated?'两次留出响应预测通过':'未通过两次留出响应预测'}</strong></p>
    {referenceTrace&&<a className="fw-tbtn" href={`?lens=spatial&view=live&trace=${referenceTrace.trace_id}&tblock=${c.layer}&thead=0&tunit=${c.unit}`}>在统一参考长文3D中定位此单元</a>}
    <p className="v8-note">{c.selection==='stratified_random'?'分层随机样本':'历史定向样本（单独统计）'}；A=SiLU(G)×V。解释来源是前缀字符类别统计，尚未得到独立语义标签。</p>
    <table className="v6-table"><thead><tr><th>划分</th><th>排序正确率</th><th>相对随机基线提升（百分点）</th><th>配对 p</th><th>打乱负对照</th></tr></thead><tbody>{[['确认',c.confirm,c.shuffle_confirm],['锁箱',c.lockbox,c.shuffle_lockbox]].map(([s,m,n])=><tr key={s}><td>{s}</td><td>{pct(m.accuracy)}</td><td>{points(m.delta)}</td><td>{num(m.p)}</td><td>{pct(n.accuracy)}</td></tr>)}</tbody></table>
    <details><summary>预测器条目、全层统计与读入/写出</summary><pre className="v8-artifact-json">{JSON.stringify({terms:c.prefix_predictor_terms,population:c.population,alignment:c.read_write_alignment,reader_formula_audit:c.reader_formula_audit,notes:c.alignment_note},null,2)}</pre><p className="v8-note">系数描述固定线性预测器，不自动等于语义。读入与写出方向不同不构成解释失败。</p></details>
    <label>自然语料两极 <select aria-label="神经元两极" value={pole} onChange={e=>{setPole(e.target.value);setOpen(null)}}>{FOLD.flatMap((f,i)=>Object.entries(POLE).map(([k,v])=><option key={`${i}:${k}`} value={`${i}:${k}`}>{f} · {v}</option>))}</select></label>
    <p className="v8-note">每个例子来自不同文档；展示极值用于检索，完整采集包含弱响应。中心化输入方向一致度 {num(c.poles[pole]?.coherence)}，仅作描述。</p>
    {examples.map((r,i)=><div key={`${pole}:${i}`}><button className="fw-tbtn" onClick={()=>setOpen(open===i?null:i)}>文档 {r.doc_index} / #{r.position} · {num(r.value)}</button>{open===i&&<><p className="v8-note">原生 G={num(r.GVA?.G)} · V={num(r.GVA?.V)} · A={num(r.GVA?.A)}</p><DocumentContext index={r.doc_index} position={r.position} refs={documentRefs}/></>}</div>)}
    <details><summary>排序预测反例（共 {c.all_error_count}，保留前 {c.counterexamples.length} 条及全部误差数组）</summary>{bad?<><select aria-label="神经元反例" value={error} onChange={e=>setError(Number(e.target.value))}>{c.counterexamples.map((r,i)=><option value={i} key={i}>配对 {r.pair.a}/{r.pair.b} · #{r.pair.position}</option>)}</select><p>实际 A：{bad.actual.map(num).join(' / ')}；预测 A：{bad.prediction.map(num).join(' / ')}</p><DocumentContext index={bad.pair.a} position={bad.pair.position} refs={documentRefs}/><DocumentContext index={bad.pair.b} position={bad.pair.position} refs={documentRefs}/></>:<p>冻结口径下未保存排序错误；不表示任意语境都成立。</p>}</details>
  </section>;
}

export default function NativeMlpReport({doc:d,planSha,capabilityInputsSha}){
  const [tab,setTab]=useState('result'),[index,setIndex]=useState(0),[layer,setLayer]=useState('all'),[showErrors,setShowErrors]=useState(false);const a=d.analysis,c=d.cards[index];
  const [populationLayer,setPopulationLayer]=useState(0),[fold,setFold]=useState(2),[channel,setChannel]=useState(2),[moment,setMoment]=useState('RMS');
  const popref=d.artifact_refs.population?.[populationLayer];const population=useApi(tab==='population'&&popref?`/api/research-library/artifacts/${popref}`:null);
  const plan=useApi(tab==='source'&&planSha?`/api/research-library/artifacts/${planSha}`:null);
  const rows=d.cards.map((r,i)=>({...r,i})).filter(r=>layer==='all'||r.layer===Number(layer));
  return <section className="v8-intervention-report" aria-label="原生MLP长文本研究报告"><p className="v8-note">{d.run_id} · 本地 Qwen3-4B / BF16 · {d.model_snapshot.slice(0,16)}</p>
    <p><strong>{a.random_replicated}/432 个随机单元通过两次留出响应预测</strong>；{a.random_replicated?'这表示一种有限的上下文规律，尚非单元完整语义或机制解释。':'当前前缀字符类别预测器未通过双留出门；保留负结果，不据此判定单元不可解释。'}</p>
    <div className="v6-toolbar" role="group" aria-label="MLP研究报告内容">{[['result','结果与判据'],['population','全层统计阵列'],['units','单元说明书'],['dose','剂量与能力'],['trace','3D计算路径'],['source','材料与边界']].map(([k,t])=><button key={k} className="fw-tbtn" aria-pressed={tab===k} onClick={()=>setTab(k)}>{t}</button>)}</div>
    {tab==='result'&&<><div className="v8-library-stats"><span><b>2,048</b>自然长文 / 每篇1,024 token</span><span><b>350,208</b>全层单元统计（未全体解释）</span><span><b>{a.random_replicated}/432</b>分层随机单元双留出通过</span></div>
      <p className="v8-note">512 篇确认、512 篇锁箱来自旧普查未使用的维基文档。每次检验 256 对独立文档，读出当前 token 和绝对位置严格相同；模型预训练是否见过原文未知。</p>
      <p>打乱训练响应的负对照双留出通过 {a.random_shuffle_double_pass}/432。历史定向的 34 个单元单独保留。</p>
      <p className="v8-note">固定门：确认与锁箱均提升≥15个百分点、单侧配对 p&lt;0.05/432，且非平局比例≥80%。预测器只读当前 token 与此前文本；词汇/位置基线在配对内相同。</p>
      <p className="v8-note">全部单元保留 G/V/A 均值、平方和、极值与符号分区；466 单元另存逐 token 无损原场。并非对所有 token 保存了全部模块原场。</p></>}
    {tab==='population'&&<><div className="v6-toolbar"><label>块 <input aria-label="普查块" type="number" min="0" max="35" value={populationLayer} onChange={e=>setPopulationLayer(Math.max(0,Math.min(35,Number(e.target.value)||0)))}/></label><select aria-label="普查划分" value={fold} onChange={e=>setFold(Number(e.target.value))}>{FOLD.map((f,i)=><option key={i} value={i}>{f}</option>)}</select><select aria-label="普查GVA" value={channel} onChange={e=>setChannel(Number(e.target.value))}>{['G','V','A'].map((f,i)=><option key={i} value={i}>{f}</option>)}</select><select aria-label="普查统计量" value={moment} onChange={e=>setMoment(e.target.value)}><option value="RMS">RMS</option><option value="mean">原始均值</option></select></div><p className="v8-note">按原生索引完整显示 B{populationLayer} 的9,728单元；无TopK筛选。当前划分有效位置 {population.data?.count[fold]?.toLocaleString()||'—'}。统计不能当作语义解释。</p>{population.error?<Empty>{population.error}</Empty>:<NativeArray values={population.data?.[`GVA_${moment}`]?.[fold]?.[channel]} label="全9728单元自然语料统计"/>}<p className="v8-note">保留原始均值与RMS，数组色标为当前数组max|值|；跨层比较需同时核对原始数值。G/V符号分区、平方和与极值可回查归档。</p></>}
    {tab==='units'&&<><div className="v6-toolbar"><label>层 <select aria-label="单元说明书层" value={layer} onChange={e=>{const v=e.target.value;setLayer(v);setIndex(d.cards.findIndex(r=>v==='all'||r.layer===Number(v)))}}><option value="all">全部36层</option>{Array.from({length:36},(_,i)=><option value={i} key={i}>B{i}</option>)}</select></label><select aria-label="单元说明书索引" value={index} onChange={e=>setIndex(Number(e.target.value))}>{rows.map(r=><option value={r.i} key={r.i}>B{r.layer}/U{r.unit} · {r.selection==='stratified_random'?'随机':'定向'} · {r.replicated?'通过':'未通过'}</option>)}</select></div><NativeNeuronCard key={c.sha256} sha={c.sha256} documentRefs={d.artifact_refs.documents} referenceTrace={d.examples[0]}/></>}
    {tab==='dose'&&<><p><strong>16单元联合操作：{d.dose.supported?'通过冻结效应门':'未通过冻结效应门'}</strong> · 能力保持：{NI[d.capability.status]||d.capability.status}</p>
      <p className="v8-note">只修改被测 query 的单元值，过去位置不动。随机组同时匹配层、激活 RMS、down 列范数及二者乘积。联合效应不能升级为单个单元必要性。</p>
      <DosePlot effects={d.dose.effects}/>
      <table className="v6-table"><thead><tr><th>操作</th><th>自然下一token正确率</th><th>相对base损失（百分点）</th><th>完整词表 KL</th><th>NLL变化</th></tr></thead><tbody>{Object.entries(d.dose.effects).map(([s,r])=><tr key={s}><td>{s}</td><td>{pct(r.accuracy)}</td><td>{points(r.loss.mean)}</td><td>{num(r.KL.mean)}</td><td>{num(r.NLL_shift.mean)}</td></tr>)}</tbody></table>
      <p className="v8-note">正式能力题 {d.capability.n}，净损失 {points(d.capability.net_loss)} 个百分点；95%区间 [{d.capability.loss_interval.map(points).join(', ')}] 个百分点；非劣效界 2 个百分点。旧题池新操作重测，140 道参考题不入门。</p>
      <details onToggle={e=>setShowErrors(e.currentTarget.open)}><summary>干预与能力反例 · 逐条展开</summary>{showErrors&&<InterventionErrors doseRef={d.artifact_refs.doses} capabilityRef={d.artifact_refs.capability} inputsRef={capabilityInputsSha}/>}</details>
      <details><summary>完整剂量曲线、匹配范围、逐条反例与数值审计</summary><pre className="v8-artifact-json">{JSON.stringify({dose:d.dose,selection:d.selection,capability:d.capability,audit:d.numeric_audit},null,2)}</pre></details></>}
    {tab==='trace'&&<><p className="v8-note">8条真实长前缀保留全部37层隐态、36块Q/K/V、全部Head和9728单元读出阵列。3D按模型层和原生索引布局；参数贡献是局部直接计算，并非完整因果机制。</p>{d.examples.map(r=><p key={r.trace_id}><a href={r.url}>配对{r.pair_index} / {r.side} · {r.n_tokens} tokens → 真实3D与单元证据</a></p>)}</>}
    {tab==='source'&&<>{plan.data&&<details><summary>下一阶段大任务 · {plan.data.title}（待执行）</summary>{plan.data.tasks.map((r,i)=><p className="v8-note" key={i}><strong>{r.title}</strong>：{r.detail}</p>)}</details>}{d.limitations.map((r,i)=><p className="v8-note" key={i}>{r}</p>)}<a href="https://already-clean.goodfire.com/" target="_blank" rel="noreferrer">Goodfire研究来源</a><details><summary>冻结预注册、完整来源与原场哈希</summary><pre className="v8-artifact-json">{JSON.stringify({prereg:d.preregistration,freeze:d.freeze,native:d.native,artifacts:d.artifact_refs},null,2)}</pre></details></>}
  </section>;
}
