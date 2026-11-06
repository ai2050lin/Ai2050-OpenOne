import {useEffect,useState} from 'react';
import {API_BASE} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import MechanismScene,{NativeArray} from './MechanismScene.jsx';
import {NativeNeuronCard} from './NativeMlpReport.jsx';
import {FunctionNeuronEvidence} from './NativeFunctionReport.jsx';
import {useApi} from '../app/rdc_fusion/api.js';
import {publishResearchContext} from '../app/researchContext.js';

export async function request(base,path,opts={}){
  const response=await fetch(base+path,{cache:'no-store',...opts,headers:{'Content-Type':'application/json',...opts.headers}});
  const d=await response.json();if(!response.ok)throw Error(typeof d.detail==='string'?d.detail:`HTTP ${response.status}`);return d;
}
export function useNative(base,path){
  const [state,setState]=useState({});useEffect(()=>{if(!path||base===null)return;const ac=new AbortController();
    request(base,path,{signal:ac.signal}).then(data=>{if(!ac.signal.aborted)setState({base,path,data})}).catch(e=>{if(!ac.signal.aborted)setState({base,path,error:e.message})});return()=>ac.abort();
  },[base,path]);return state.path===path&&state.base===base?state:{};
}
const num=x=>Number.isFinite(x)?x.toPrecision(6):'—';
const initial=n=>Math.max(0,Number(new URLSearchParams(window.location.search).get(n))||0);
const pct=x=>Number.isFinite(x)?`${(100*x).toFixed(2)}%`:'—';

export default function MechanismTraceExplorer(){
  const [base,setBase]=useState(null),[version,setVersion]=useState(0),[rootId,setRootId]=useState(()=>new URLSearchParams(window.location.search).get('trace')||''),[traceId,setTraceId]=useState('');
  const [block,setBlock]=useState(()=>Math.min(35,initial('tblock'))),[position,setPosition]=useState(null),[head,setHead]=useState(()=>Math.min(31,initial('thead'))),[unit,setUnit]=useState(()=>Math.min(9727,initial('tunit'))),[coordinate,setCoordinate]=useState(0),[connect,setConnect]=useState(false),[tab,setTab]=useState(()=>new URLSearchParams(window.location.search).get('ttab')||'embedding');
  const [status,setStatus]=useState({}),[text,setText]=useState(''),[limit,setLimit]=useState(2048),[message,setMessage]=useState(''),[running,setRunning]=useState(false),[catalog,setCatalog]=useState([]);
  const [entity,setEntity]=useState('');
  const [overlay,setOverlay]=useState(false),[responseFamily,setResponseFamily]=useState('conditional'),[responseSplit,setResponseSplit]=useState('lockbox'),[responseChannel,setResponseChannel]=useState(2);
  useEffect(()=>{let dead=false;(async()=>{for(const b of [...new Set([API_BASE,'http://127.0.0.1:5001','http://127.0.0.1:5002'])]){
    try{const d=await request(b,'/api/traces/catalog');if(!dead){setBase(b);setCatalog(d.traces);if(!rootId){const first=d.traces.find(x=>x.deep&&['heldout_entity_frame','unseen_combination','unseen_entity'].includes(x.split))||d.traces.find(x=>x.deep)||d.traces[0];if(first)setRootId(first.trace_id)}}return}catch{}}
    if(!dead)setMessage('未连接到支持原生轨迹的本机服务；不会显示模拟数据。')})();return()=>{dead=true}},[version]);
  useEffect(()=>{setTraceId(rootId);setConnect(false)},[rootId]);
  const m=useNative(base,traceId?`/api/traces/${traceId}/manifest`:null),d=m.data;
  useEffect(()=>{if(d)publishResearchContext({trace:d.trace_id,run:d.run_id,snapshot:d.model_snapshot})},[d?.trace_id]);
  useEffect(()=>{if(!d)return;const q=new URLSearchParams(window.location.search);q.set('trace',rootId);q.set('tblock',block);q.set('thead',head);q.set('tunit',unit);q.set('ttab',tab);if(d.readout_positions.includes(position))q.set('tpos',position);window.history.replaceState({},'',window.location.pathname+'?'+q.toString())},[d?.trace_id,rootId,block,head,unit,tab,position]);
  const children=useNative(base,rootId?`/api/traces/catalog?parent=${rootId}`:null);
  useEffect(()=>{if(d){const requested=Number(new URLSearchParams(window.location.search).get('tpos'));setPosition(d.readout_positions.includes(requested)?requested:d.readout_positions.at(-1));setConnect(false)}},[d?.trace_id]);
  const context=useNative(base,d?`/api/traces/context?run_id=${encodeURIComponent(d.run_id)}&model_snapshot=${d.model_snapshot}`:null);
  const profile=useNative(base,d&&tab==='evidence'&&context.data&&!context.data.study_type?`/api/traces/profile?run_id=${encodeURIComponent(d.run_id)}&block=${block}&space=mlp`:null);
  const p=position??d?.readout_positions.at(-1);const valid=!!d&&p!==undefined;
  const responseRef=context.data?.family_profile_refs?.[responseFamily]?.[String(block)];
  const response=useApi(overlay&&responseRef?`/api/research-library/artifacts/${responseRef}`:null);
  const responseValid=response.data?.model_snapshot===d?.model_snapshot&&response.data?.block===block;
  const rd=d?.readout_positions.indexOf(p)??-1;
  const stage=d?.response_profile_stage_indices?.[rd]??rd;
  const responseValues=responseValid?response.data.splits[responseSplit]?.GVA_delta?.[stage]?.[responseChannel]:undefined;
  const responseHeads=responseValid?response.data.splits[responseSplit]?.head_delta_norm?.[stage]:undefined;
  const responseEmbedding=responseValid?response.data.splits[responseSplit]?.embedding_delta?.[stage]:undefined;

  const field=(boundary,b=block)=>valid?`/api/traces/${traceId}/field?block=${b}&position=${p}&boundary=${boundary}`:null;
  const embedding=useNative(base,field('embedding',0)),hidden=useNative(base,field('hidden_state',block+1)),activation=useNative(base,field('mlp_activation')),gate=useNative(base,field('mlp_gate')),
    value=useNative(base,field('mlp_value')),hz=useNative(base,field('head_z'));
  const attention=useNative(base,valid?`/api/traces/${traceId}/attention-row?block=${block}&head=${head}&query=${p}`:null);
  const link=useNative(base,valid&&connect?`/api/traces/${traceId}/connections?block=${block}&position=${p}&head=${head}&unit=${unit}&coordinate=${coordinate}`:null);
  useEffect(()=>{if(base!==null)request(base,'/api/live/status').then(setStatus).catch(()=>{})},[base,version]);
  useEffect(()=>{if(base===null||status.status!=='loading')return;const t=setTimeout(()=>request(base,'/api/live/status').then(setStatus).catch(()=>{}),1500);return()=>clearTimeout(t)},[base,status]);
  async function action(kind){try{setMessage('');await request(base,`/api/live/${kind}`,{method:'POST'});setVersion(v=>v+1)}catch(e){setMessage(e.message)}}
  async function run(){setRunning(true);setMessage('');try{const trace=await request(base,'/api/live/run',{method:'POST',body:JSON.stringify({text,max_tokens:limit})});setRootId(trace.trace_id);setTraceId(trace.trace_id);setVersion(v=>v+1)}catch(e){setMessage(e.message)}finally{setRunning(false)}}
  const clamp=(x,n)=>Math.max(0,Math.min(n-1,Number(x)||0));
  const entities=[...new Set(catalog.map(r=>r.entity||'实时观察'))];
  const selectedEntity=entity||catalog.find(r=>r.trace_id===rootId)?.entity||entities[0]||'';
  function chooseEntity(v){setEntity(v);const r=catalog.find(r=>(r.entity||'实时观察')===v);if(r)setRootId(r.trace_id)}
  return <div className="v8-route"><section className="v8-card"><h2>真实运行与计算路径</h2><p className="v8-note">存档研究与实时采集使用同一个原生轨迹格式。灰Embedding、蓝Attention、橙MLP；连线表示当前样本的局部计算贡献，不自动代表因果机制。</p>
    <div className="v6-toolbar"><select aria-label="研究实体" value={selectedEntity} onChange={e=>chooseEntity(e.target.value)}>{entities.map(v=><option key={v}>{v}</option>)}</select><select aria-label="真实研究轨迹" value={rootId} onChange={e=>setRootId(e.target.value)} style={{maxWidth:'100%'}}><option value="">选择已采集轨迹</option>{catalog.filter(r=>(r.entity||'实时观察')===selectedEntity).map(r=><option key={r.trace_id} value={r.trace_id}>{r.scope} · {r.n_tokens} tokens · {r.sample_id}{r.deep?' · 全token深采':''}</option>)}</select><button className="fw-tbtn" onClick={()=>setVersion(v=>v+1)}>刷新真实轨迹</button></div>
    <details><summary>输入长文本进行真实采集</summary><p className="v8-note">上限含实际聊天模板；超过上限时明确停止，不截断。保存全部37层隐态与Q/K/V，各层完整MLP数组保留在声明的模块读出位置。存档浏览无需占用GPU。</p>
      <textarea aria-label="真实采集文本" value={text} onChange={e=>setText(e.target.value)} rows={5} style={{width:'100%'}}/>
      <div className="v6-toolbar"><select aria-label="实际token上限" value={limit} onChange={e=>setLimit(Number(e.target.value))}>{[512,1024,2048].map(n=><option key={n}>{n}</option>)}</select><span>本机模型：{status.status||'读取状态'}</span><button className="fw-tbtn" onClick={()=>action('load')}>加载本机模型</button><button className="fw-tbtn" disabled={running} onClick={()=>action('unload')}>卸载本机模型</button><button className="fw-tbtn" disabled={running||!text||base===null} onClick={run}>{running?'真实采集中…':'采集并保存原生轨迹'}</button></div></details>
    {message&&<Empty>{message}</Empty>}{m.error&&<Empty>{m.error}</Empty>}
  </section>
  {d&&<><section className="v8-card"><b>{d.entity} · {d.scope} · {d.n_tokens} tokens</b><p className="v8-note">{d.run_id} · {d.split} · BF16 · 快照 {d.model_snapshot}。模块读出之后的线索不被解释为该位置已有的信息。</p>
    {d.intervention&&<p className="v8-note"><strong>真实操作复采</strong> · {JSON.stringify(d.intervention)}。编辑后A与未编辑G/V的乘积可以不同；本次操作不等于移除历史位置。</p>}
    {d.input_attention_mask&&<details><summary>实际批次padding与注意力mask</summary><pre className="v8-artifact-json">{JSON.stringify(d.input_attention_mask)}</pre></details>}
    {context.data?.reference_mode&&<p className="v8-note"><strong>{context.data.reference_mode}</strong>；系统范围指令在实体之前，早位置也可响应此指令。</p>}
    <div className="v6-toolbar"><label>生成步 <select aria-label="实际生成步" value={traceId} onChange={e=>setTraceId(e.target.value)}><option value={rootId}>前缀 → 首个新token</option>{children.data?.traces.map(r=><option key={r.trace_id} value={r.trace_id}>第{r.generation_step}步 · 前缀{r.n_tokens} tokens</option>)}</select></label>
      <label>块 <input aria-label="轨迹块" type="number" min="0" max={d.n_layers-1} value={block} onChange={e=>setBlock(clamp(e.target.value,d.n_layers))}/></label>
      <label>模块读出 <select aria-label="真实query位置" value={p} onChange={e=>setPosition(Number(e.target.value))}>{d.readout_positions.map((i,j)=><option key={i} value={i}>#{i} {d.tokens[i]} · {d.readout_annotations?.[j]?.role||''}{i===d.entity_position?' · 实体':i===d.document_colour_position?' · 文档颜色':''}</option>)}</select></label>
      <label>Head <input aria-label="轨迹Head" type="number" min="0" max={d.n_heads-1} value={head} onChange={e=>setHead(clamp(e.target.value,d.n_heads))}/></label>
      <label>MLP单元 <input aria-label="轨迹MLP单元" type="number" min="0" max={d.d_ff-1} value={unit} onChange={e=>setUnit(clamp(e.target.value,d.d_ff))}/></label></div>
    <details><summary>完整实际输入与证据来源</summary><pre className="v8-sample-text">{d.prompt}</pre><p className="v8-note">原生位置、层和边界来自 {d.boundaries}。注意力行是从实际post-RoPE Q/K重算，误差与实际SDPA输出核对；不是注意力权重的因果性声明。</p><pre className="v8-artifact-json">{JSON.stringify({trace_id:d.trace_id,files:d.files,reconstruction:d.reconstruction,provenance:d.provenance},null,2)}</pre></details>
  </section>
  <section className="v8-card"><a className="fw-tbtn" href={`?lens=spatial&view=research&run=${d.run_id}`}>全部研究成果与历史版本 · 3D总图</a>
    {context.data?.family_profile_refs&&<><div className="v6-toolbar"><label><input aria-label="叠加功能角色响应" type="checkbox" checked={overlay} onChange={e=>setOverlay(e.target.checked)}/>叠加题族配对响应</label><select aria-label="3D响应题族" value={responseFamily} onChange={e=>setResponseFamily(e.target.value)}>{Object.keys(context.data.family_profile_refs).map(f=><option key={f}>{f}</option>)}</select><select aria-label="3D响应划分" value={responseSplit} onChange={e=>setResponseSplit(e.target.value)}><option value="discovery">发现</option><option value="confirm">确认</option><option value="lockbox">锁箱</option></select><select aria-label="3D响应GVA" value={responseChannel} onChange={e=>setResponseChannel(Number(e.target.value))}>{['G','V','A'].map((v,i)=><option key={v} value={i}>{v}</option>)}</select></div><p className="v8-note">叠加为全部配对的正角色−负角色均值差，这是主实验组统计，按读出角色对齐，并非当前极性对照自身均值；原生索引不排序。Head颜色为平均输出差向量的范数。3D统计着色与右侧单样本原值分别读取；统计模式隐藏单样本参数连线。</p>{overlay&&(response.error?<Empty>{response.error}</Empty>:!responseValues?<p>该块/位置的响应叠加尚未加载；不使用单样本值替代。</p>:<p className="v8-note">{response.data.readout_labels[stage]} · {responseSplit} · U{unit}均值差 {num(responseValues[unit])}；全部9728单元，显示值7位有效数字，原场无损保留。</p>)}</>}
  </section>
  <div className="v8-trace-layout"><section className="v8-card"><MechanismScene manifest={d} block={block} head={head} unit={unit} embedding={overlay?responseEmbedding:embedding.data?.values} activation={overlay?responseValues:activation.data?.values} headZ={hz.data?.values} headOverlay={overlay?responseHeads:undefined} caption={overlay?`${responseFamily} / ${responseSplit} · 正角色−负角色均值差`:'当前单样本原生值'} connections={overlay?null:link.data} onBlock={setBlock} onHead={setHead} onUnit={setUnit}/><p className="v8-note">当前块的全部 {d.d_ff} 个MLP单元参与索引显示；其他层灰色为未展开。叠加值蓝正橙负、统一max|值|，黑框为选中；幅值不等于语义或因果重要性。</p></section>
  <section className="v8-card"><div className="v6-toolbar">{[['embedding','词嵌入'],['attention','全部Head'],['mlp','完整MLP阵列'],['connections','参数与关联'],['evidence','特性与证据'],['output','实际输出']].map(([k,t])=><button className="fw-tbtn" key={k} aria-pressed={tab===k} onClick={()=>setTab(k)}>{t}</button>)}</div>
    {tab==='evidence'&&context.data?.study_type==='native_mlp_long'&&<><p className="v8-note">自然长文原生MLP研究：相同token与位置配对，前文特征预测留出响应；没有文档/现实范围行为门。</p>{context.data.unit_cards[`${block}:${unit}`]?<NativeNeuronCard key={context.data.unit_cards[`${block}:${unit}`]} sha={context.data.unit_cards[`${block}:${unit}`]} documentRefs={context.data.document_refs}/>:<Empty>该单元未进入466单元逐token样本，保留实际当前阵列与全层统计；暂无留出说明书。</Empty>}<a className="fw-tbtn" href={`?lens=process&station=plan&run=${d.run_id}`}>查看完整研究、剂量和能力保持</a></>}
    {tab==='evidence'&&context.data?.study_type==='native_mlp_function'&&<><FunctionNeuronEvidence context={context.data} block={block} unit={unit} readoutRole={d.readout_annotations?.[rd]?.role}/>{overlay&&<NativeArray values={responseValues} selected={unit} onPick={setUnit} label="完整题族配对响应均值差"/>}</>}
    {tab==='evidence'&&context.data?.study_type&& !['native_mlp_long','native_mlp_function'].includes(context.data.study_type)&&<><p className="v8-note">本运行是校准或其他归档类型，不自动生成语义结论；完整报告、负结果和轨迹均在成果总图保留。</p><a href={`?lens=spatial&view=research&run=${d.run_id}`}>查看该运行的完整研究成果</a></>}
    {tab==='evidence'&&!context.data?.study_type&&<>{context.error?<Empty>{context.error}；实时未注册观察不自动产生研究结论。</Empty>:context.data&&<><p><strong>本次仲裁行为门：{context.data.report_summary.behavior_gate?'通过':'未通过'}</strong></p><p className="v8-note">留出文档范围 {pct(context.data.report_summary.scope_accuracy['文档'])}；现实范围 {pct(context.data.report_summary.scope_accuracy['现实'])}。下面展示范围提示的响应特性，包括错误仲裁；不是已确认的语义单元。</p><p>B{block} / U{unit}：文档−现实响应 · 发现集 {num(profile.data?.discovery_delta[unit])}；{context.data.heldout_profile_label||'留出实体/新句式'} {num(profile.data?.heldout_delta[unit])}</p><>{profile.error&&<Empty>{profile.error}</Empty>}</><NativeArray values={profile.data?.heldout_delta} selected={unit} onPick={setUnit} label="全部单元留出范围响应"/><p className="v8-note">保留全部9728单元，按原生索引排序；描述性均值差，没有全单位族错误控制或因果晋级。</p><a className="fw-tbtn" href={`?lens=process&station=plan&run=${d.run_id}`}>查看预注册、能力保持与全部反例</a><details><summary>统一研究数据 · 历史来源（{context.data.historical_evidence.length}）</summary>{context.data.historical_evidence.map(e=><p className="v8-note" key={e.artifact_sha256}>{e.title} · {e.precision} · <a href={`${base}/api/research-library/artifacts/${e.artifact_sha256}`} target="_blank" rel="noreferrer">原始报告及哈希</a></p>)}<p className="v8-note">历史来源只统一索引和证据，不将不同模型、精度和任务混合平均或自动升级。</p></details></>}</>}
    {tab==='embedding'&&<><p>token #{p} / ID {d.token_ids[p]} · 全部 {d.d_model} 原生坐标</p>{embedding.error?<Empty>{embedding.error}</Empty>:<NativeArray values={embedding.data?.values} selected={coordinate} onPick={setCoordinate} label="完整词嵌入"/>}<p className="v8-note">E[{coordinate}]={num(embedding.data?.values[coordinate])}；这是当前token的输入查表值，语言特征需要跨样本的证据，不能由单个数值命名。</p><details><summary>沿相同坐标查看 HiddenState[{block+1}]</summary>{hidden.error?<Empty>{hidden.error}</Empty>:<NativeArray values={hidden.data?.values} selected={coordinate} onPick={setCoordinate} label="完整HiddenState"/>}<p className="v8-note">h[{block+1},{p},{coordinate}]={num(hidden.data?.values[coordinate])}。共37层，随块索引查询；末层已经最终归一化。同坐标不自动代表跨层相同语义。</p></details></>}
    {tab==='attention'&&<><div className="v8-head-grid">{Array.from({length:d.n_heads},(_,h)=><button key={h} className={h===head?'on':''} onClick={()=>setHead(h)}>H{h}<small>KV{Math.floor(h/(d.n_heads/d.n_kv_heads))}</small></button>)}</div><p>H{head} · query #{p}「{d.tokens[p]}」· KV组 {attention.data?.kv_group}</p>{attention.error?<Empty>{attention.error}</Empty>:<NativeArray values={attention.data?.weights} selected={d.document_colour_position??-1} label="完整注意力来源行"/>}<p className="v8-note">按源token顺序完整保留，未画TopK。重算输出相对L2误差 {num(attention.data?.validation)}。点击输入位置只改变该query，不冒用末位注意力。</p><details><summary>全部源token与权重</summary><pre className="v8-artifact-json">{JSON.stringify(attention.data?.weights.map((v,i)=>({position:i,token:d.tokens[i],weight:v})),null,2)}</pre></details></>}
    {tab==='mlp'&&<><p>B{block} · 全部 {d.d_ff}/{d.d_ff} 单元 · {d.intervention?.activation_is_intervened?'A为操作后值；未操作值才满足SiLU(G)×V':'A=SiLU(G)×V'}</p>{activation.error?<Empty>{activation.error}</Empty>:<NativeArray values={activation.data?.values} selected={unit} onPick={setUnit} label="完整MLP中间单元阵列"/>}<p>U{unit}: G={num(gate.data?.values[unit])} · V={num(value.data?.values[unit])} · A={num(activation.data?.values[unit])}</p><p className="v8-note">G/V/A均为实际BF16前向值，不是由特征标签生成的颜色。</p></>}
    {tab==='connections'&&<><p className="v8-note">核验完整权重快照后，用实际W_gate/W_up/W_down/W_O和样本响应计算。首次核验需要读取权重文件；不会加载GPU模型。</p><label>输出坐标 <input aria-label="关联输出坐标" type="number" min="0" max={d.d_model-1} value={coordinate} onChange={e=>setCoordinate(clamp(e.target.value,d.d_model))}/></label><button className="fw-tbtn" onClick={()=>setConnect(true)}>核验参数并显示计算关联</button>
      {link.error&&<Empty>{link.error}</Empty>}{connect&&!link.data&&!link.error&&<p>正在核验原始权重与计算贡献…</p>}
      {link.data&&<><p>完整快照一致 · H{head}→U{unit} gate局部贡献 {num(link.data.head_gate_terms[head])}</p><p className="v8-note">这是固定实际RMS的局部直接归因，尚未证明因果必要性。残差贡献另计 {num(link.data.residual_gate_term)}。</p>
        <table className="v6-table"><thead><tr><th>真实计算</th><th>原生前向</th><th>全项求和</th></tr></thead><tbody>{[['gate','Gate输入'],['value','Value输入'],['attention_output_coordinate','Attention输出坐标'],['mlp_output_coordinate','MLP输出坐标']].map(([k,t])=><tr key={k}><td>{t}</td><td>{num(link.data.observed[k])}</td><td>{num(link.data.reconstructed[k])}</td></tr>)}</tbody></table>
        <a className="fw-tbtn" href={`?lens=spatial&view=parameters&pmodel=${link.data.local_model_id}&pblock=${block}&pkind=mlp&punit=${unit}&trace=${traceId}&run=${encodeURIComponent(d.run_id)}&snapshot=${d.model_snapshot}`}>打开同一模型的完整参数切片</a><details><summary>全部原生坐标/单元贡献与来源</summary><pre className="v8-artifact-json">{JSON.stringify(link.data,null,2)}</pre></details></>}
    </>}
    {tab==='output'&&<><p>完整前缀的实际下一token分布（前12项展示）</p><table className="v6-table"><thead><tr><th>token</th><th>概率</th></tr></thead><tbody>{d.output_top.map(x=><tr key={x.token_id}><td>{x.text} / {x.token_id}</td><td>{num(x.probability)}</td></tr>)}</tbody></table><p className="v8-note">此表是模型实测输出。上方局部计算关联不是一个已经能独立替代原模型的图谱预测器。</p></>}
  </section></div></>}
  </div>;
}
