import { useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE } from '../app/rdc_fusion/api.js';
import { publishResearchContext, useResearchContext } from '../app/researchContext.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { ArrayCanvas } from './ParameterExplorer.jsx';
import { request, useNative } from './MechanismTraceExplorer.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import { workbenchHref } from './v8Util.js';

const idOk = value => /^[0-9a-f]{64}$/.test(value || '');
const num = x => Number.isFinite(x) ? x.toPrecision(6) : '—';
const query = () => new URLSearchParams(window.location.search);

function storeQuery(key, value) {
  const url = new URL(window.location.href);
  if (value == null || value === '') url.searchParams.delete(key);
  else url.searchParams.set(key, String(value));
  if (url.href !== window.location.href) window.history.replaceState(window.history.state, '', url);
}

function recordedPosition(trace, saved) {
  const positions = trace?.readout_positions || [];
  return saved != null && saved !== '' && positions.includes(Number(saved)) ? Number(saved) : positions.at(-1);
}

function roleAt(trace, p) {
  const ri = trace.readout_positions.indexOf(p);
  return trace.readout_annotations?.[ri]?.role ||
    (p === trace.entity_position ? '实体' : p === trace.n_tokens - 1 ? '末位' : '');
}

function aligned(a, b, pa, pb) {
  if (!a || !b || pa == null || pb == null || a.run_id !== b.run_id ||
      (a.generation_step ?? 0) !== (b.generation_step ?? 0)) return false;
  const ra = roleAt(a, pa), rb = roleAt(b, pb);
  return ra && rb ? ra === rb : pa === pb && a.token_ids[pa] === b.token_ids[pb];
}

function fieldPath(trace, position, block, boundary) {
  return trace && position != null && block >= 0 ?
    `/api/traces/${trace.trace_id}/field?${new URLSearchParams({ block, position, boundary })}` : null;
}

function arrayFor(result, trace, p, block, boundary, width) {
  const d = result.data;
  if (!d || !trace || d.trace_id !== trace.trace_id || d.position !== p || d.boundary !== boundary ||
      (boundary === 'embedding' ? d.layer !== 0 : d.block !== block || d.model_snapshot !== trace.model_snapshot) ||
      d.values?.length !== width || !d.values.every(Number.isFinite)) return null;
  if (boundary !== 'head_z') return d.values;
  const { n_heads: heads, head_dim: size } = trace;
  return Array.from({ length: heads }, (_, head) => {
    let squares = 0;
    for (let i = head * size; i < (head + 1) * size; i++) squares += d.values[i] ** 2;
    return Math.sqrt(squares);
  });
}

const traceLabel = row => [row.sample_id || row.entity || row.trace_id.slice(0, 12), row.scope,
  row.generation_step ? `生成步 ${row.generation_step}` : '', row.split, `${row.n_tokens} tokens`].filter(Boolean).join(' · ');

function TraceSelect({ rows, value, onChange, label, empty }) {
  return <label>{label} <select className="v8-trace-select" aria-label={label} value={value} onChange={e => onChange(e.target.value)}>
    <option value="">{empty}</option>
    {value && !rows.some(row => row.trace_id === value) && <option value={value}>{value.slice(0, 12)} · 链接指定</option>}
    {rows.map(row => <option key={row.trace_id} value={row.trace_id}>{traceLabel(row)}</option>)}
  </select></label>;
}

function PositionSelect({ trace, position, onChange, label }) {
  if (!trace) return null;
  return <label>{label} <select aria-label={label} value={position} onChange={e => onChange(e.target.value)}>
    {trace.readout_positions.map(p => <option key={p} value={p}>#{p} · {trace.tokens?.[p]} {roleAt(trace,p)}</option>)}
  </select></label>;
}

function Capture({ base, onTrace, onRefresh }) {
  const [text, setText] = useState(''), [limit, setLimit] = useState(2048);
  const [status, setStatus] = useState(''), [busy, setBusy] = useState(false);
  const [modelStatus,setModelStatus]=useState(null);
  const [statusVersion,refreshStatus]=useState(0);
  useEffect(()=>{
    const controller=new AbortController();
    let timer;
    const read=()=>request(base,'/api/live/status',{signal:controller.signal}).then(data=>{
      if(controller.signal.aborted)return;
      setModelStatus(data);
      if(data.status==='loading')timer=setTimeout(read,1200);
    }).catch(error=>{if(!controller.signal.aborted)setStatus(error.message)});
    read();
    return ()=>{clearTimeout(timer);controller.abort()};
  },[base,statusVersion]);
  async function action(kind) {
    setBusy(true);
    try {
      const data = await request(base, `/api/live/${kind}`, {
        method: 'POST', ...(kind === 'run' ? { body: JSON.stringify({ text, max_tokens: limit }) } : {}),
      });
      setStatus(data.message || '');
      if(kind!=='run')refreshStatus(n=>n+1);
      if (kind === 'run' && idOk(data.trace_id)) { onTrace(data.trace_id); onRefresh(); }
    } catch (error) { setStatus(error.message); }
    finally { setBusy(false); }
  }
  return <div className="v8-route"><textarea aria-label="长文本采集输入" value={text} onChange={e => setText(e.target.value)} rows={4} />
    <div className="v6-toolbar"><label>最大 tokens <input type="number" min="1" max="2048" value={limit} onChange={e => setLimit(Number(e.target.value))} /></label>
      <button type="button" className="fw-tbtn" disabled={busy || modelStatus?.available===false} onClick={() => action('load')}>加载本机模型</button>
      <button type="button" className="fw-tbtn" disabled={busy || modelStatus?.status==='loading'} onClick={() => action('unload')}>卸载本机模型</button>
      <button type="button" className="fw-tbtn" disabled={busy || !text.trim() || !Number.isInteger(limit) || limit<1 || limit>2048 || modelStatus?.available===false}
        onClick={() => action('run')}>采集并保存原生轨迹</button>
    </div><p className="v8-note" role="status">{[modelStatus?.model_name,modelStatus?.error||modelStatus?.reason||modelStatus?.progress||modelStatus?.status,status].filter(Boolean).join(' · ')||'读取本机模型状态…'}</p></div>;
}

function Connections({ base, trace, block, position, selection, initialHead }) {
  const [other, setOther] = useState(selection.module==='mlp'?initialHead??null:null), [coordinate, setCoordinate] = useState(0), [path, setPath] = useState(null);
  const attention = selection.module === 'attn';
  const result = useNative(base, path);
  const head = attention ? selection.index : other;
  const unit = attention ? other : selection.index;
  const d=result.data?.verified_full_snapshot===true&&result.data.model_snapshot===trace.model_snapshot&&result.data.trace_id===trace.trace_id&&
    result.data.block===block&&result.data.position===position&&result.data.head===head&&result.data.unit===unit&&result.data.coordinate===coordinate?result.data:null;
  return <div className="v8-route">
    <div className="v6-toolbar"><label>{attention ? '关联 MLP 单元' : '关联 Head'} <input type="number" min="0" max={(attention ? trace.d_ff : trace.n_heads) - 1}
      value={other ?? ''} onChange={e => { setOther(e.target.value === '' ? null : Number(e.target.value)); setPath(null); }} /></label>
      <label>输出坐标 <input type="number" min="0" max={trace.d_model - 1} value={coordinate} onChange={e => { setCoordinate(Number(e.target.value)); setPath(null); }} /></label>
      <button type="button" className="fw-tbtn" disabled={other == null || !Number.isInteger(other) || other < 0 || other >= (attention ? trace.d_ff : trace.n_heads) ||
        !Number.isInteger(coordinate) || coordinate < 0 || coordinate >= trace.d_model} onClick={() =>
        setPath(`/api/traces/${trace.trace_id}/connections?${new URLSearchParams({block,position,head,unit,coordinate})}`)}>核验权重并读取计算路径</button>
    </div>
    {result.error ? <Empty>{result.error}</Empty> : d ? <>
      <p>H{head} → U{unit} · 局部 Gate 贡献 {num(d.head_gate_terms?.[head])} · 残差贡献 {num(d.residual_gate_term)}</p>
      <p className="v8-note">实测 Gate {num(d.observed?.gate)} · 复算 {num(d.reconstructed?.gate_via_heads_plus_residual)}。此项固定样本的 RMS，仅说明该次计算的局部贡献。</p>
      <ResearchDisclosure title="完整路径及来源"><pre className="v8-artifact-json">{JSON.stringify(d,null,2)}</pre></ResearchDisclosure>
    </> : path && <p role="status">正在核验完整快照和读取路径…</p>}
  </div>;
}

export function useSpatialTrace({ view, snapshot, selection, onToken, onSourceSnapshot, legacy }) {
  const active = view === 'live' || view === 'response', comparing = view === 'response';
  const context = useResearchContext();
  const [catalog, setCatalog] = useState(null), [base, setBase] = useState(null), [error, setError] = useState('');
  const [version, setVersion] = useState(0), [pick, setPick] = useState(() => context.trace || context.context_trace);
  const [pickB, setPickB] = useState(() => query().get('straceb') || '');
  const [positions, setPositions] = useState(() => ({a: query().get('sposrun') || query().get('tpos'), b: query().get('sposb')}));
  const [colour, setColour] = useState('delta');
  const previousSnapshot = useRef(snapshot);
  useEffect(()=>{
    if (previousSnapshot.current && snapshot && previousSnapshot.current !== snapshot && !context.trace) {
      setPick(''); setPickB(''); setPositions({}); storeQuery('straceb',null); storeQuery('sposrun',null); storeQuery('sposb',null); storeQuery('tpos',null);
    }
    if(snapshot)previousSnapshot.current=snapshot;
  },[snapshot,context.trace]);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    (async () => {
      for (const origin of new Set([API_BASE, 'http://127.0.0.1:5001', 'http://127.0.0.1:5002'])) {
        try {
          const data = await request(origin, '/api/traces/catalog', {signal:controller.signal});
          if (Array.isArray(data.traces) && !controller.signal.aborted) {
            setCatalog(data.traces); setBase(origin); setError(''); return;
          }
        } catch { if (controller.signal.aborted) return; }
      }
      if (!controller.signal.aborted) setError('当前无法读取原生轨迹服务。');
    })();
    return () => controller.abort();
  }, [active, version]);
  const rows = useMemo(() => (catalog || []).filter(row => snapshot && row.model_snapshot === snapshot), [catalog, snapshot]);
  const aId = pick || rows.find(row => row.run_id === context.run)?.trace_id || rows[0]?.trace_id || '';
  const manifest = id => active && idOk(id) ? `/api/traces/${id}/manifest` : null;
  const aResult = useNative(base, manifest(aId));
  const bResult = useNative(base, comparing ? manifest(pickB) : null);
  useEffect(()=>{
    if(active&&pick&&aResult.data?.trace_id===pick&&aResult.data.model_snapshot!==snapshot)
      onSourceSnapshot?.(aResult.data.model_snapshot);
  },[active,pick,aResult.data,snapshot,onSourceSnapshot]);
  const a = snapshot && aResult.data?.model_snapshot === snapshot ? aResult.data : null;
  const b = snapshot && bResult.data?.model_snapshot === snapshot ? bResult.data : null;
  const root = a?.parent_trace_id || a?.trace_id;
  const rootB = b?.parent_trace_id || b?.trace_id;
  const child = useNative(base, active && idOk(root) ? `/api/traces/catalog?parent=${root}` : null);
  const childB = useNative(base, comparing && idOk(rootB) ? `/api/traces/catalog?parent=${rootB}` : null);
  const pa = recordedPosition(a, positions.a), pb = recordedPosition(b, positions.b);
  const block = selection.layer == null ? -1 : Math.max(0, selection.layer - 1);
  const kind = selection.module;
  const index = selection.index;
  const boundary = kind === 'embedding' ? 'embedding' : kind === 'attn' ? 'head_z' : kind === 'mlp' ? 'mlp_activation' : '';
  const compat = aligned(a,b,pa,pb);
  const aField = useNative(base, boundary ? fieldPath(a,pa,block,boundary) : null);
  const bField = useNative(base, boundary && compat ? fieldPath(b,pb,block,boundary) : null);
  const width = kind === 'attn' ? a?.n_heads * a?.head_dim : kind === 'embedding' ? a?.d_model : a?.d_ff;
  const aValues = useMemo(()=>arrayFor(aField,a,pa,block,boundary,width),[aField.data,a,pa,block,boundary,width]);
  const bValues = useMemo(()=>arrayFor(bField,b,pb,block,boundary,width),[bField.data,b,pb,block,boundary,width]);
  const delta = useMemo(()=>aValues && bValues && aValues.length === bValues.length ? bValues.map((v,i)=>v-aValues[i]) : null,[aValues,bValues]);
  const maxAbs = delta ? Math.max(1e-12,...aValues.map(Math.abs),...bValues.map(Math.abs),...delta.map(Math.abs)) : undefined;
  const embedMatches = kind !== 'embedding' || index == null || index === a?.token_ids?.[pa];
  const embedPairMatches = kind !== 'embedding' || (embedMatches && a?.token_ids?.[pa] === b?.token_ids?.[pb]);
  const pointValues = comparing ? (embedPairMatches && delta ? (colour==='a'?aValues:colour==='b'?bValues:delta) : null) : embedMatches ? aValues : null;
  const gate = useNative(base, view==='live' && kind==='mlp' && index!=null ? fieldPath(a,pa,block,'mlp_gate') : null);
  const value = useNative(base, view==='live' && kind==='mlp' && index!=null ? fieldPath(a,pa,block,'mlp_value') : null);
  const attention = useNative(base, view==='live' && kind==='attn' && index!=null && a && pa!=null ?
    `/api/traces/${a.trace_id}/attention-row?${new URLSearchParams({block,head:index,query:pa})}` : null);
  const ar = attention.data?.trace_id === a?.trace_id && attention.data?.block === block &&
    attention.data?.query_position === pa && attention.data?.head === index ? attention.data : null;
  const gv = arrayFor(gate,a,pa,block,'mlp_gate',width);
  const vv = arrayFor(value,a,pa,block,'mlp_value',width);
  function selectA(id) {
    setPick(id); setPositions(old=>({...old,a:null})); storeQuery('sposrun',null); storeQuery('tpos',null);
    publishResearchContext({trace:id,context_trace:''});
  }
  useEffect(()=>{
    if (active && a?.trace_id) publishResearchContext({trace:a.trace_id,run:a.run_id});
  },[active,a?.trace_id,a?.run_id]);
  function selectPosition(key, value) {
    setPositions(old=>({...old,[key]:value})); storeQuery(key==='a'?'sposrun':'sposb',value);
  }
  function selectB(id){
    setPickB(id);storeQuery('straceb',id);selectPosition('b',null);
  }
  const sources = a && [rows.find(row=>row.trace_id===root) || { ...a, trace_id:root, generation_step:0 },
    ...(child.data?.traces || []).filter(row=>row.model_snapshot===snapshot)];
  const selectedA = sources?.find(row=>row.trace_id===aId);
  const catalogRows = selectedA ? rows : rows.filter(row=>!row.parent_trace_id);
  const optionsB = rows.filter(row=>row.trace_id!==root && (!a || row.run_id===a.run_id));
  const sourcesB = b && [rows.find(row=>row.trace_id===rootB) || { ...b, trace_id:rootB, generation_step:0 },
    ...(childB.data?.traces || []).filter(row=>row.model_snapshot===snapshot)];
  const controls = active && <section className="v8-card v8-spatial-source" aria-label="轨迹与对照条件">
    <div className="v6-toolbar">
      <TraceSelect rows={catalogRows} value={selectedA ? root : aId} onChange={selectA} label={comparing?'样本 A':'研究样本'} empty="选择同快照轨迹" />
      {sources?.length>1&&<TraceSelect rows={sources} value={aId} onChange={selectA} label="实际生成步" empty="选择前缀或生成步" />}
      <PositionSelect trace={a} position={pa} onChange={v=>selectPosition('a',v)} label={comparing?'A 读出位置':'真实query位置'} />
      {comparing&&<><TraceSelect rows={optionsB} value={rootB||pickB} onChange={selectB} label="样本 B" empty="选择同实验轨迹" />
        {sourcesB?.length>1&&<TraceSelect rows={sourcesB} value={pickB} onChange={selectB} label="B 生成步" empty="选择前缀或生成步" />}
        <PositionSelect trace={b} position={pb} onChange={v=>selectPosition('b',v)} label="B 读出位置" /></>}
      <button type="button" className="fw-tbtn" onClick={()=>setVersion(n=>n+1)}>刷新轨迹</button>
    </div>
    {error || aResult.error || bResult.error ? <Empty>{error || aResult.error || bResult.error}</Empty>
      : (aResult.data&&!a || bResult.data&&!b) ? <Empty>轨迹快照与当前模型不一致。</Empty>
        : !catalog ? <p role="status">正在读取归档轨迹…</p> : !aId && <Empty>当前快照暂无可查询的归档轨迹。</Empty>}
    {comparing && a && b && !compat && <Empty>请选择同一次研究、同生成步和同读出角色的样本。</Empty>}
    {a && kind==='embedding' && a.token_ids?.[pa]!=null && !comparing && <button type="button" className="fw-tbtn"
      onClick={()=>onToken(a.token_ids[pa])}>定位当前 token #{a.token_ids[pa]}</button>}
    {view==='live' && base!==null && <ResearchDisclosure title="本机采集与模型控制"><Capture base={base} onTrace={selectA} onRefresh={()=>setVersion(n=>n+1)} /></ResearchDisclosure>}
  </section>;
  const colourControls=comparing&&delta&&embedPairMatches&&<div className="v6-toolbar" role="group" aria-label="响应着色">
    {[['a','样本 A'],['b','样本 B'],['delta','差值 B − A']].map(([key,label])=><button type="button" key={key} className="fw-tbtn"
      aria-pressed={colour===key} onClick={()=>setColour(key)}>{label}</button>)}
    <span className="v8-note">同一色标</span>
  </div>;
  const detail = active && <>
    {a && <p className="v8-note">{a.run_id} · {traceLabel(a)} · 读出 #{pa}{b && comparing ? `；B · ${traceLabel(b)} · 读出 #${pb}` : ''}</p>}
    {aField.error || bField.error ? <Empty>{aField.error || bField.error}</Empty> : null}
    {comparing ? <>
      {delta && embedPairMatches ? <>
      <p className="v8-note">{kind==='attn'?'Head 值为实际输出向量范数。':'数值来自两个样本的原生读出。'}</p>
      {index!=null && kind!=='embedding' && <p>A {num(aValues[index])} · B {num(bValues[index])} · B − A {num(delta[index])}</p>}</>
        : <p className="v8-note">{kind==='embedding'&&!embedPairMatches?'词嵌入对照需两个读出命中同一 token ID。':'选择兼容的 A、B 轨迹与同一层的模块，查看原生对照。'}</p>}
    </> : a && <>
      {kind==='mlp'&&index!=null&&<><p>G {num(gv?.[index])} · V {num(vv?.[index])} · A {num(aValues?.[index])}</p>
        {a.intervention?.activation_is_intervened&&<p className="v8-note">A 是干预后的值，G、V 是原始读出。</p>}</>}
      {kind==='attn'&&index!=null&&<><p>Head 输出范数 {num(aValues?.[index])}{ar? ` · KV ${ar.kv_group} · 重算误差 ${num(ar.validation)}` : ''}</p>
        {attention.error&&<Empty>{attention.error}</Empty>}
        {ar&&<ResearchDisclosure title="完整注意力来源"><ArrayCanvas count={ar.weights.length} values={ar.weights} label="原生来源权重" />
          <details><summary>源 token 与权重 JSON</summary><pre className="v8-artifact-json">{JSON.stringify(ar.weights.map((weight,i)=>[i,a.tokens[i],weight]),null,2)}</pre></details>
        </ResearchDisclosure>}
      </>}
      {index!=null && ['attn','mlp'].includes(kind) && pa!=null && <ResearchDisclosure title="真实计算路径" initiallyOpen={legacy?.tab==='connections'} key={`${aId}:${block}:${pa}:${kind}:${index}`}>
        <Connections base={base} trace={a} block={block} position={pa} selection={selection} initialHead={legacy?.head} />
      </ResearchDisclosure>}
      <ResearchDisclosure title="输入、输出与轨迹来源" initiallyOpen={legacy?.tab==='output'}><pre className="v8-sample-text">{a.prompt}</pre>
        <table className="v6-table"><thead><tr><th>下一 token</th><th>概率</th></tr></thead><tbody>{(a.output_top || []).map(row=><tr key={row.token_id}><td>{row.text} / {row.token_id}</td><td>{num(row.probability)}</td></tr>)}</tbody></table>
        <details><summary>采集来源 JSON</summary><pre className="v8-artifact-json">{JSON.stringify({trace_id:a.trace_id,boundaries:a.boundaries,provenance:a.provenance},null,2)}</pre></details>
        <a className="v6-linkbtn" href={workbenchHref(new URLSearchParams({lens:'spatial',view:'graph',gview:'internal',
          run:a.run_id,snapshot:a.model_snapshot,trace:a.trace_id,tpos:pa}))}>查看该运行的研究记录</a>
      </ResearchDisclosure>
    </>}
  </>;
  return {controls,colourControls,detail,pointValues,maxAbs,pointNote:pointValues ? (comparing?'样本对照':'当前样本读出') : ''};
}
