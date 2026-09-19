import {useEffect, useMemo, useRef, useState} from 'react';
import {Canvas} from '@react-three/fiber';
import {OrbitControls} from '@react-three/drei';
import './RdcPrefixAtlas.css';
import './RdcJointAtlas.css';

const API = 'http://127.0.0.1:5001/api/rdc-joint';
const fmt = value => value == null ? '—' : Number(value).toPrecision(6);
async function get(path, params) {
  const response = await fetch(API + path + (params ? '?' + new URLSearchParams(params) : ''));
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || response.statusText);
  return response.json();
}

function Coordinates({data}) {
  const geometry = useMemo(() => {
    const rows = data.values.length, cols = data.values[0].length;
    const positions = new Float32Array(rows * cols * 3), colors = new Float32Array(rows * cols * 3);
    const limit = Math.asinh(data.whole_field_absmax || 1);
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      const index = (row * cols + col) * 3, value = Math.asinh(data.values[row][col]) / limit;
      positions.set([(col / Math.max(1, cols - 1) - .5) * 30, (row / Math.max(1, rows - 1) - .5) * 18, value * 8], index);
      colors.set(value >= 0 ? [.78, .34, .16] : [.08, .4, .67], index);
    }
    return {positions, colors};
  }, [data]);
  return <Canvas camera={{position: [19, 18, 33], fov: 50}} aria-label="原生坐标与观察序号的三维数值视图">
    <color attach="background" args={['#f1f5f8']} />
    <axesHelper args={[12]} />
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[geometry.positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[geometry.colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={.045} vertexColors transparent opacity={.65} sizeAttenuation />
    </points>
    <OrbitControls makeDefault />
  </Canvas>;
}

export function Field({data}) {
  const ref = useRef(null);
  const [mapping, setMapping] = useState('asinh'), [picked, setPicked] = useState(null), [three, setThree] = useState(false);
  useEffect(() => {
    if (!data?.values?.length || !ref.current) return;
    const canvas = ref.current, rows = data.values.length, cols = data.values[0].length;
    canvas.width = cols; canvas.height = rows;
    const ctx = canvas.getContext('2d'), bitmap = ctx.createImageData(cols, rows), max = data.whole_field_absmax || 1;
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const raw = data.values[y][x], value = mapping === 'linear' ? raw / max : Math.asinh(raw) / Math.asinh(max);
      const weight = Math.min(1, Math.abs(value)), i = (y * cols + x) * 4;
      bitmap.data[i] = Math.round(value >= 0 ? 247 : 247 - weight * 206);
      bitmap.data[i + 1] = Math.round(249 - weight * 161);
      bitmap.data[i + 2] = Math.round(value < 0 ? 252 : 252 - weight * 218);
      bitmap.data[i + 3] = 255;
    }
    ctx.putImageData(bitmap, 0, 0);
  }, [data, mapping]);
  if (!data) return <p className="prefix-muted">选择材料和视图，再点击读取。切换条件后旧结果不会继续冒充新查询。</p>;
  return <div className="prefix-field joint-field">
    <p>{data.normalization}</p>
    <div className="prefix-controls"><label>颜色映射<select value={mapping} onChange={e => setMapping(e.target.value)}><option value="asinh">asinh（单位尺度1，无裁剪）</option><option value="linear">原值线性</option></select></label>
      <span>原生宽度 {data.native_width} · 当前列 [{data.start}, {data.end}) · 色标 ±{fmt(data.whole_field_absmax)}</span>
      <button onClick={() => setThree(v => !v)}>{three ? '收起' : '打开'}全部点三维数值视图</button></div>
    <canvas ref={ref} data-testid="joint-field" aria-label="全原生坐标热力图" style={{height: Math.max(170, Math.min(500, data.values.length * 12))}}
      onClick={event => {
        const box = event.currentTarget.getBoundingClientRect();
        const x = Math.min(data.values[0].length - 1, Math.floor((event.clientX - box.left) / box.width * data.values[0].length));
        const y = Math.min(data.values.length - 1, Math.floor((event.clientY - box.top) / box.height * data.values.length));
        setPicked({data, text: `${data.labels[y]}；原生列 ${data.start + x} = ${data.values[y][x]}`});
      }} />
    <p className="prefix-muted">纵轴：{data.labels[0]} → {data.labels.at(-1)}。缩略图像素会合并显示；数字及索引不压缩，点击或缩小列范围核对。</p>
    {picked?.data === data && <output>{picked.text}</output>}
    {three && <><div className="joint-scene"><Coordinates data={data} /></div><p className="prefix-muted">三维轴：横向=所选原生列序号，纵向=行/层/步骤序号，高度=asinh数值。保留当前数组全部点；显示比例不是神经元真实空间、流形或语义距离。拖动旋转，滚轮缩放。</p></>}
    {data.MSE != null && <p>MSE {fmt(data.MSE)} · {data.available_inputs}</p>}
    {data.download && <a href={'http://127.0.0.1:5001' + data.download}>下载该来源全部已采集原生数组</a>}
  </div>;
}

function Scalar({sample, blocks, extensionId = '', positions = null}) {
  const [block, setBlock] = useState(6), [position, setPosition] = useState(0), [unit, setUnit] = useState(0), [input, setInput] = useState(0), [output, setOutput] = useState(0);
  const [record, setRecord] = useState(null), [error, setError] = useState('');
  const identity = JSON.stringify([sample, extensionId, block, position, unit, input, output]);
  async function query() {
    try { const value = await get(extensionId ? '/extension-scalar' : '/scalar', {sample, id: extensionId, block, position_index: position, unit, input_coordinate: input, output_coordinate: output}); setRecord({identity, value}); setError(''); }
    catch (e) { setError(e.message); setRecord(null); }
  }
  const value = record?.identity === identity ? record.value : null;
  return <div><p>任意坐标/MLP单元/标量参数可查询。下方会返回全部2560输入项与全部9728输出项，不以单个高亮项代替整体。</p>
    <div className="prefix-controls"><label>原生block<select value={block} onChange={e => setBlock(Number(e.target.value))}>{blocks.map(b => <option key={b}>{b}</option>)}</select></label>
      <label>位置槽<select value={position} onChange={e => setPosition(Number(e.target.value))}>{(positions ? positions.map(p => `原生token ${p}`) : ['首token', '第二token', '锚点0', '锚点1']).map((v, i) => <option key={v} value={i}>{v}</option>)}</select></label>
      {[['输入坐标', input, setInput, 2559], ['MLP单元', unit, setUnit, 9727], ['输出坐标', output, setOutput, 2559]].map(([label, v, setter, max]) => <label key={label}>{label}<input type="number" min={0} max={max} value={v} onChange={e => setter(Number(e.target.value))} /></label>)}
      <button disabled={!sample} onClick={query}>读取全部参数贡献</button></div>
    {error && <p role="alert">{error}</p>}{value && <>{value.native_position != null && <p>{value.sample_id} · block {value.block} · 原生token {value.native_position}</p>}<pre>{JSON.stringify(value.chain, null, 2)}</pre><Field data={value.input_terms} /><Field data={value.unit_terms} /></>}
  </div>;
}

function Scale() {
  const [model, setModel] = useState('qwen4'), [samples, setSamples] = useState([]), [sample, setSample] = useState(''), [part, setPart] = useState('late'), [data, setData] = useState(null), [error, setError] = useState('');
  const identity = JSON.stringify([model, sample, part]);
  useEffect(() => { let active = true; get('/scale-samples', {model}).then(rows => { if (active) { setSamples(rows); setSample(rows[0]?.sample_id || ''); setData(null); } }).catch(e => active && setError(e.message)); return () => { active = false; }; }, [model]);
  async function query() { try { setData({identity, value: await get('/scale-field', {model, sample, part})}); setError(''); } catch (e) { setData(null); setError(e.message); } }
  return <section><h2>顺序加载的非量化模型</h2><p>同一自然来源，各自训练与分词；完整原生宽度不同，坐标索引不做跨模型语义对齐。未完成的模型会保持空列表。</p>
    <div className="prefix-controls"><label>模型<select value={model} onChange={e => setModel(e.target.value)}>{['qwen4', 'qwen14', 'glm4'].map(x => <option key={x}>{x}</option>)}</select></label>
      <label>来源<select value={sample} onChange={e => setSample(e.target.value)}>{samples.map(r => <option key={r.sample_id}>{r.sample_id}</option>)}</select></label>
      <label>状态<select value={part} onChange={e => setPart(e.target.value)}>{['early', 'mid', 'late', 'postnorm', 'incoming_embedding', 'all_layer_energy'].map(x => <option key={x}>{x}</option>)}</select></label><button disabled={!sample} onClick={query}>读取该模型全坐标</button></div>
    {error && <p role="alert">{error}</p>}<Field data={data?.identity === identity ? data.value : null} /></section>;
}

function SummaryFields({extension = false}) {
  const [items, setItems] = useState([]), [id, setId] = useState(''), [kind, setKind] = useState('event_layers'), [sourceId, setSourceId] = useState(''), [row, setRow] = useState(0), [value, setValue] = useState(null), [error, setError] = useState('');
  const identity = JSON.stringify([id, row]);
  useEffect(() => { let active = true; get(extension ? '/extension-index' : '/analysis-index').then(x => { if (active) { setItems(x); setId(x[0]?.id || ''); setSourceId(x[0]?.sample_id || ''); } }).catch(e => active && setError(e.message)); return () => { active = false; }; }, [extension]);
  const item = items.find(x => x.id === id), total = item ? item.shape.slice(0, -1).reduce((a, b) => a * b, 1) : 0;
  const visibleItems = extension ? items.filter(x => x.kind === kind && x.sample_id === sourceId) : items;
  async function query() { try { const result = await get(extension ? '/extension-field' : '/analysis-field', {id, row_start: row, row_count: 37, count: item.shape.at(-1)}); setValue({identity, result}); setError(''); } catch (e) { setValue(null); setError(e.message); } }
  const data = value?.identity === identity ? value.result : null;
  return <section><h2>{extension ? '事件、接续与新材料完整数组' : '全坐标汇总数组与逐行回查'}</h2><p>保留原始张量的轴顺序；每页最多37行，列覆盖完整原生宽度。行号可遍历全部张量，不做隐式截断。{extension && '包括事件全层/原生因子、新材料全部事件行与预定样例、训练因子样例及全部接续原场。原生值与拟合值按数组名称区分；事件不是新语义类别。'}</p>
    <div className="prefix-controls">{extension && <><label>数组类别<select value={kind} onChange={e => {const next=e.target.value,first=items.find(x=>x.kind===next);setKind(next);setSourceId(first?.sample_id || '');setId(first?.id || '');setRow(0);}}>{[...new Set(items.map(x=>x.kind))].map(x=><option key={x}>{x}</option>)}</select></label>
      <label>事件来源<select value={sourceId} onChange={e=>{setSourceId(e.target.value);setId(items.find(x=>x.kind===kind&&x.sample_id===e.target.value)?.id || '');setRow(0);}}>{[...new Set(items.filter(x=>x.kind===kind).map(x=>x.sample_id))].map(x=><option key={x}>{x}</option>)}</select></label></>}
      <label>{extension ? '事件原生数组' : '已登记数组'}<select value={id} onChange={e => { setId(e.target.value); setRow(0); }}>{visibleItems.map(x => <option key={x.id} value={x.id}>{x.id} [{x.shape.join('×')}]</option>)}</select></label>
      <label>{extension ? '事件起始行' : '起始展平行'}<input type="number" min={0} max={Math.max(0, total - 1)} value={row} onChange={e => setRow(Number(e.target.value))} /></label><span>总行数 {total}</span><button disabled={!id} onClick={query}>{extension ? '读取事件页全部坐标' : '读取这一页全部坐标'}</button></div>
    {error && <p role="alert">{error}</p>}{data?.source && <><blockquote>{data.source.text}</blockquote><p>来源 {data.source.sample_id} · {data.source.source_group} · 原张量 [{data.tensor_shape.join('×')}] · 原生位置 [{data.positions.join(', ')}]</p></>}<Field data={data} />
    {extension && item?.kind === 'regime_factors' && <Scalar key={item.sample_id} sample={item.sample_id} extensionId={item.id} blocks={[6,16,34]} positions={item.positions} />}</section>;
}

export default function RdcJointAtlas() {
  const [summary, setSummary] = useState(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const [scope, setScope] = useState('fresh'), [samples, setSamples] = useState([]), [sample, setSample] = useState(''), [language, setLanguage] = useState('all');
  const [mode, setMode] = useState('field'), [layer, setLayer] = useState('h12'), [view, setView] = useState('raw'), [position, setPosition] = useState(0), [start, setStart] = useState(0), [count, setCount] = useState(2560);
  const [choice, setChoice] = useState('current_KL'), [anchor, setAnchor] = useState(0), [branch, setBranch] = useState('KL_history'), [block, setBlock] = useState(6), [part, setPart] = useState('activation');
  const [relation, setRelation] = useState('ud:nmod'), [control, setControl] = useState('exact_distance_POS_noninitial'), [split, setSplit] = useState('test'), [matrixView, setMatrixView] = useState('train_z'), [matrixRow, setMatrixRow] = useState(0), [matrixCol, setMatrixCol] = useState(0);
  const [response, setResponse] = useState(null), [detail, setDetail] = useState(null);
  const identity = JSON.stringify([scope, sample, mode, layer, view, position, start, count, choice, anchor, branch, block, part, relation, control, split, matrixView, matrixRow, matrixCol]);
  const current = response?.identity === identity ? response.value : null;
  const source = samples.find(r => r.sample_id === sample);
  useEffect(() => { let active = true; get('/overview').then(x => active && setSummary(x)).catch(e => active && setError(e.message)); return () => { active = false; }; }, []);
  useEffect(() => { let active = true; get('/samples', {scope}).then(rows => { if (active) { setSamples(rows); setSample(rows[0]?.sample_id || ''); setLanguage('all'); setResponse(null); } }).catch(e => active && setError(e.message)); return () => { active = false; }; }, [scope]);
  useEffect(() => { if (!sample) return; let active = true; get('/sample', {scope, sample}).then(value => active && setDetail({scope, sample, value})).catch(e => active && setError(e.message)); return () => { active = false; }; }, [scope, sample]);
  const metadata = detail?.scope === scope && detail.sample === sample ? detail.value : null;
  const filtered = samples.filter(r => (language === 'all' || r.language === language) && (mode !== 'generation' || r.generated) && (!['factors', 'scalar'].includes(mode) || r.native_factors));
  function changeMode(next) {
    setMode(next); setStart(0); setCount(next === 'factors' ? 9728 : 2560); setResponse(null); setError('');
    if (['generation', 'factors', 'scalar'].includes(next)) { setScope('fresh'); const row = samples.find(r => next === 'generation' ? r.generated : r.native_factors); if (scope === 'fresh' && row) setSample(row.sample_id); }
  }
  async function query() {
    setBusy(true); setError('');
    try {
      let value;
      if (mode === 'field') value = await get('/field', {scope, sample, layer, view, position_index: position, start, count});
      else if (mode === 'prediction') value = await get('/prediction', {scope, sample, choice, anchor, start, count});
      else if (mode === 'factors') value = await get('/factors', {sample, block, part, start, count});
      else if (mode === 'generation') value = await get('/generation', {sample, branch, start, count});
      else value = await get('/matrix', {relation, control, split, view: matrixView, row_start: matrixRow, column_start: matrixCol, count: 48});
      setResponse({identity, value});
    } catch (e) { setResponse(null); setError(e.message); } finally { setBusy(false); }
  }
  const probabilities = [...(summary?.probability || []).filter(r => r.split === 'test'), ...(summary?.confirmation?.probability || [])];
  const relationNames = [...new Set((summary?.relations?.entries || []).map(r => r.relation))];
  const blocks = summary?.native?.blocks || [6, 12, 23];
  const tail = summary?.extension?.tail_confirmation, regimes = summary?.extension?.native_regimes;
  return <main className="prefix-app joint-app">
    <header><div><small>RDC · INDEPENDENT JOINT ATLAS · PHASE 2719+</small><h1>自然语言 · 条件坐标联合图谱</h1><p>外部关系、全坐标响应、真实参数与逐步输出，在同一来源上核对。</p></div>
      <nav><a href="/rdc-relation">2715–2718记录</a><a href="/rdc-prefix">早期前缀图谱</a><a href="/rdc">既有3D机制客户端</a><a href="/">主页</a></nav></header>
    <section className="prefix-warning"><h2>已找到可复核拼图，尚未形成语言闭合机制</h2>
      <div className="joint-stats"><span><b>{summary?.material?.main_units ?? '—'}</b> 主来源</span><span><b>{summary?.material?.fresh_units ?? '—'}</b> 冻结后确认来源</span><span><b>2560</b> 4B完整坐标</span><span><b>151936</b> 全词表目标</span></div>
      <p>H12/H23/H36覆盖每个token；embedding及所有36层输出覆盖六个已声明位置。实体、篇章和依存标注是分析条件，不等于已解释知识或推理。</p>
      <button onClick={() => get('/overview').then(setSummary).catch(e => setError(e.message))}>刷新已完成结果</button></section>
    <section><nav className="prefix-tabs">{[['field', '全场与全层'], ['matrix', '类型关系全矩阵'], ['prediction', '状态 / 概率规则'], ['factors', '原生全部单元'], ['scalar', '单参数全项核对'], ['generation', '32步无刷新接续']].map(([id, label]) => <button key={id} className={mode === id ? 'active' : ''} onClick={() => changeMode(id)}>{label}</button>)}</nav>
      {mode !== 'matrix' && <div className="prefix-controls"><label>材料<select value={scope} disabled={['generation', 'factors', 'scalar'].includes(mode)} onChange={e => setScope(e.target.value)}><option value="fresh">独立确认256来源</option><option value="main">主来源512</option></select></label>
        <label>语言<select value={language} onChange={e => { const value = e.target.value; setLanguage(value); const row = samples.find(r => (value === 'all' || r.language === value) && (mode !== 'generation' || r.generated) && (!['factors', 'scalar'].includes(mode) || r.native_factors)); if (row) setSample(row.sample_id); }}>{['all', 'en', 'zh'].map(l => <option key={l}>{l}</option>)}</select></label>
        <label className="prefix-wide">真实来源<select value={sample} onChange={e => setSample(e.target.value)}>{filtered.map(r => <option key={r.sample_id} value={r.sample_id}>{r.sample_id} · {r.genre} · {r.text.slice(0, 55)}</option>)}</select></label></div>}
      {mode === 'field' && <div className="prefix-controls"><label>原场<select value={layer} onChange={e => { setLayer(e.target.value); setView('raw'); }}>{['h12', 'h23', 'h36', 'postnorm', 'all_layers'].map(l => <option key={l}>{l}</option>)}</select></label>
        {layer === 'all_layers' && <label>全层位置<select value={position} onChange={e => setPosition(Number(e.target.value))}>{(source?.positions || []).map((p, i) => <option key={i} value={i}>token {p}（位置槽{i}）</option>)}</select></label>}
        <label>规范化<select value={view} onChange={e => setView(e.target.value)}><option value="raw">原始值</option><option value="source_RMS">每行自身RMS</option>{['h12', 'h23'].includes(layer) && <option value="train_z">冻结训练z-score</option>}</select></label></div>}
      {mode === 'prediction' && <div className="prefix-controls"><label>冻结选择<select value={choice} onChange={e => setChoice(e.target.value)}>{Object.keys(summary?.choices || {current_KL: 1}).map(c => <option key={c}>{c}</option>)}</select></label><label>锚点<select value={anchor} onChange={e => setAnchor(Number(e.target.value))}><option value={0}>第一个</option><option value={1}>第二个</option></select></label></div>}
      {mode === 'factors' && <div className="prefix-controls"><label>原生block<select value={block} onChange={e => setBlock(Number(e.target.value))}>{blocks.map(b => <option key={b}>{b}</option>)}</select></label><label>全部因子<select value={part} onChange={e => setPart(e.target.value)}>{['input', 'attention', 'mlp_input', 'gate', 'up', 'activation', 'mlp', 'output', 'attention_probability'].map(p => <option key={p}>{p}</option>)}</select></label></div>}
      {mode === 'generation' && <label>自主分支<select value={branch} onChange={e => setBranch(e.target.value)}>{['MSE_embedding', 'KL_embedding', 'KL_history'].map(b => <option key={b}>{b}</option>)}</select></label>}
      {mode === 'matrix' && <><p>全部6,553,600坐标对参与研究；此处从完整原場与匹配索引精确重算48×48切片。文档→窗口→边等权，无匹配不回退。图中连线/距离不是物理计算连接。</p>
        <div className="prefix-controls"><label>类型<select value={relation} onChange={e => setRelation(e.target.value)}>{relationNames.map(r => <option key={r}>{r}</option>)}</select></label><label>对照<select value={control} onChange={e => setControl(e.target.value)}>{['exact_distance_POS_noninitial', 'distance_band_POS_noninitial', 'same_dependent_ID'].map(c => <option key={c}>{c}</option>)}</select></label><label>划分<select value={split} onChange={e => setSplit(e.target.value)}><option>train</option><option>test</option></select></label><label>数值<select value={matrixView} onChange={e => setMatrixView(e.target.value)}>{['raw', 'train_z', 'source_RMS'].map(v => <option key={v}>{v}</option>)}</select></label>
          <label>起始行<input type="number" min={0} max={2559} value={matrixRow} onChange={e => setMatrixRow(Number(e.target.value))} /></label><label>起始列<input type="number" min={0} max={2559} value={matrixCol} onChange={e => setMatrixCol(Number(e.target.value))} /></label></div></>}
      {!['matrix', 'scalar'].includes(mode) && <div className="prefix-controls"><label>原生起始列<input type="number" min={0} value={start} onChange={e => setStart(Number(e.target.value))} /></label><label>列数<input type="number" min={1} max={mode === 'factors' ? 9728 : 2560} value={count} onChange={e => setCount(Number(e.target.value))} /></label><button onClick={() => { setStart(0); setCount(mode === 'factors' ? 9728 : 2560); }}>恢复全部列</button></div>}
      {mode === 'scalar' ? <Scalar sample={sample} blocks={blocks} /> : <><button disabled={busy || (mode !== 'matrix' && !sample)} onClick={query}>{busy ? '读取 / 精确重算中…' : mode === 'matrix' ? '重算完整来源矩阵切片' : '读取全坐标结果'}</button>
        {error && <p className="prefix-error" role="alert">{error}</p>}<Field data={mode === 'generation' ? current?.field : current} /></>}
      {mode === 'matrix' && current && <details><summary>匹配组数、全部统计与边界</summary><pre>{JSON.stringify({groups: current.groups, windows: current.windows, statistics: current.statistics}, null, 2)}</pre></details>}
      {mode === 'generation' && current?.record && <div className="joint-transcripts"><h2>同一起点，独立续写分支</h2><blockquote>{current.record.initial_prefix_text}</blockquote><div><article><h3>原生贪心续写</h3><p>{current.record.native_text}</p><small>{current.record.native_stop}</small></article>{Object.entries(current.record.branches).map(([name, b]) => <article key={name}><h3>{name}</h3><p>{b.text}</p><small>{b.stop} · 首次分叉（零基）{b.first_divergence_zero_based ?? '无'} · 重复三元组{fmt(b.repeated_trigram_fraction)}</small></article>)}</div><p>更低同前缀KL可能伴随退化重复。初始状态后不刷新；原生参照只作诊断，不反馈给自主更新器。</p><details><summary>每一步指标与实际token ID</summary><pre>{JSON.stringify(current.record, null, 2)}</pre></details></div>}
      {metadata && mode !== 'matrix' && <details><summary>外部语言图谱、词面/跨度与真实输入（回溯标注，不是在线输入）</summary><blockquote>{metadata.text}</blockquote><p>材料族：{metadata.language_mode_families.join(' · ')}；来源组：{metadata.source_group}</p><p>{metadata.warning}</p>
        <div className="prefix-table"><table><thead><tr><th>原始关系类型</th><th>dependent token</th><th>head token</th><th>最晚端点</th></tr></thead><tbody>{metadata.retrospective_graph.map((g, i) => <tr key={i}><td>{g.type}</td><td>{g.dependent_token}</td><td>{g.head_token}</td><td>{g.available_after_token}</td></tr>)}</tbody></table></div><details><summary>完整实体、原词级标注与采集校验</summary><pre>{JSON.stringify(metadata, null, 2)}</pre></details></details>}
    </section>
    <section><h2>全词表概率与独立确认</h2><p>KL训练和MSE训练是不同目标；自然后继评分不等于自由生成正确性，也不等于事实或推理能力。原生FP32底线单独保留。</p><div className="prefix-table"><table><thead><tr>{['材料', '输入范围', '路线', 'KL', 'argmax一致率', '后继token NLL'].map(t => <th key={t}>{t}</th>)}</tr></thead><tbody>{probabilities.map(r => <tr key={r.split + r.scope + r.route}><td>{r.split}</td><td>{r.scope}</td><td>{r.route}</td><td>{fmt(r.KL)}</td><td>{fmt(r.argmax_agreement)}</td><td>{fmt(r.predicted_observed_token_NLL)}</td></tr>)}</tbody></table></div></section>
    <Scale />
    {!!tail?.sources && <section><h2>自动续研：稀少放大、跨层抵消与概率解释</h2>
      <div className="joint-stats"><span><b>{tail.sources}</b> 新自然窗口</span><span><b>{tail.tokens}</b> 新原生token</span><span><b>{tail.event_count}</b> 定义内新事件</span><span><b>{fmt(tail.evaluation?.pooled?.reports?.full_H12_linear_logistic?.average_precision)}</b> 冻结预测器AP</span></div>
      <p>新事件仅占非首token约0.042%，却贡献约63.2%的H23能量。默认0.5阈值只有2/19召回：排序信号不是可靠检测器，也不是语义类别。</p>
      {!!regimes?.sources && <><p>全部19个新事件：H16→H17放大，H34→H35抵消。完整原生路径采集{regimes.sources}个来源、{regimes.probes}个位置；对照按相同token ID、同语料、不同来源组匹配。它是条件化数值组织，不等于知识/推理机制。</p>
        <div className="prefix-table"><table><thead><tr><th>新条件</th><th>block16 MLP能量</th><th>block34 MLP能量</th><th>block34输出能量</th><th>原生输出熵</th></tr></thead><tbody>{['event','same_ID_non_event','first'].map(role=>{const r=regimes.condition_summaries?.['new_'+role];return r && <tr key={role}><td>{role}</td><td>{fmt(r.blocks['16'].MLP_energy)}</td><td>{fmt(r.blocks['34'].MLP_energy)}</td><td>{fmt(r.blocks['34'].output_energy)}</td><td>{fmt(r.native_output_entropy)}</td></tr>;})}</tbody></table></div>
        <p>首位置block6参照幅度有过重复归一化，已保留原件并纠正；第16/34块原始数据和层选择未改变。仅相似方向不证明相同功能，原生低熵也不证明因果作用。</p></>}
      <p>温度对照结果：当前状态KL训练保留超出温度的收益；两条时序路线被验证集单温度超过。详情与完整数组均可在下方回查。</p></section>}
    <SummaryFields />
    <SummaryFields extension />
    <section><h2>原生计算、失败范围与研究状态</h2>{[['全层形成位置', summary?.layers], ['真实gate/up/down与全部单元', summary?.native], ['32步自主与条件化分支', summary?.generation], ['顺序模型检查', summary?.scale], ['自动续研阶段', summary?.extension], ['独立确认及成组区间', summary?.confirmation], ['附件审查修正', summary?.review], ['最终审计', summary?.integrity]].map(([label, data]) => <details key={label}><summary>{label}</summary><pre>{JSON.stringify(data || {}, null, 2)}</pre></details>)}<p>接口只读，不在浏览器查询时自动加载模型或触发GPU作业。RDC名称保留；未宣称普遍语义齿轮、新闭合定理或AGI证明。</p></section>
    {!!summary?.figures?.length && <section><h2>全坐标科学图</h2>{summary.figures.map(f => <figure key={f.path}><figcaption>{f.path} · {f.view || f.description}</figcaption><a href={API + '/figure/' + f.path}><img src={API + '/figure/' + f.path} loading="lazy" alt={f.path} /></a></figure>)}</section>}
  </main>;
}
