import { lazy, Suspense, useEffect, useMemo, useState, useSyncExternalStore } from 'react';

import { DESKTOP, OFFLINE_CLIENT, useApi } from '../app/rdc_fusion/api.js';
import { getExtraEntry, storeTick, subscribe } from '../app/rdc_fusion/dataStore.js';
import { NeuronUnitCardContent } from '../app/rdc_fusion/NeuronUnitCard.jsx';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { publishResearchContext, useResearchContext } from '../app/researchContext.js';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import ParameterExplorer, { ArrayCanvas, useParameters } from './ParameterExplorer.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import Tower, { towerIndexPosition } from './Tower.jsx';
import { EVIDENCE_LABEL } from './researchFragmentUtil.js';
import { shortModel, workbenchHref } from './v8Util.js';
import { useSpatialTrace } from './SpatialTrace.jsx';
import { B35LayerResults, B35UnitFeatures, useB35Features, useB35Unit } from './B35RecursiveReport.jsx';
import { AttributionGraph, BasisCompare, HeadFeature, HeadInternals, InfoFlowMap, LayerList, MinimalPairs, RelationBoards } from './ClientGuide.jsx';
import './core_kit.css';

const ResponseExplorer = lazy(() => import('../app/rdc_fusion/ResponseExplorer.jsx'));
const MODULES = { attn: 1, mlp: 1 };
const TYPE_COLORS = ['#d6541e', '#7a3fbf', '#0f766e', '#abc2de', '#245ca8', '#e2e7ee'];
const TYPE_LABELS = ['只认一个词', '认词也看上下文', '偶尔强激活', '经常激活', '负向为主', '其他'];
const VIEW_TEXT = {
  graph: ['LLM 编码图谱 · 结构与参数', '在 3D 空间里看完整的模型：每一层、每个 Head、每个 MLP 神经元。点任何一个，查看它的特征和相关研究。'],
  parameters: ['LLM 编码图谱 · 结构与参数', '在 3D 空间里看完整的模型：每一层、每个 Head、每个 MLP 神经元。点任何一个，查看它的特征和相关研究。'],
  live: ['关联机制 · 运行与路径', '模型读一句话时，信息一层层流动。这里看流中有什么信息，以及一句话的计算路径。'],
  response: ['关联机制 · 响应对照', '只改句子里的一处，看模型内部哪里跟着变。这是连接语言图谱和内部图谱的主要方法。'],
};
const EMPTY = { layer: null, module: '', index: null, mode: '', coordinate: null };
const KEYS = { layer: 'sblock', module: 'smodule', index: 'sindex', mode: 'smode', coordinate:'scoord' };
const count = value => Number.isInteger(value) && value > 0 ? value : 0;
const numbered = value => Number.isInteger(value) && value >= 0;
const format = n => n.toLocaleString('zh-CN');
const label = (kind, index) => `${kind === 'attn' ? 'H' : 'U'}${index}`;

function readSelection(view, paramTab) {
  const params = new URLSearchParams(window.location.search);
  const integer = key => /^\d+$/.test(params.get(key) || '') ? Number(params.get(key)) : null;
  if (params.has(KEYS.layer)) return { layer: integer(KEYS.layer), module: params.get(KEYS.module) || '',
    index: integer(KEYS.index), mode: params.get(KEYS.mode) || '', coordinate: integer(KEYS.coordinate) };
  if (view === 'parameters') {
    if (paramTab === 'neurons' && integer('block') != null)
      return { ...EMPTY, layer: integer('block') + 1, module: 'mlp', index: integer('unit') };
    if (params.get('pkind') === 'embedding')
      return { ...EMPTY, layer:0, module:'embedding', index:integer('prow') ?? 0, coordinate:integer('pcol') };
    if (integer('pblock') != null || params.has('pkind') || params.has('punit') || params.has('phead')) {
      const mlp = params.get('pkind') === 'mlp' || (!params.has('pkind') && params.has('punit'));
      return { ...EMPTY, layer:(integer('pblock') ?? 0)+1, module:mlp?'mlp':'attn', index:integer(mlp?'punit':'phead') ?? 0 };
    }
  }
  if (view === 'live' && (params.has('tblock') || params.has('ttab'))) {
    const tab=params.get('ttab');
    if(tab==='embedding')return {...EMPTY,layer:0,module:'embedding'};
    const head=tab==='attention'||(!params.has('tunit')&&params.has('thead'));
    return {...EMPTY,layer:(integer('tblock')??0)+1,module:head?'attn':'mlp',index:integer(head?'thead':'tunit')??0};
  }
  if (view === 'response' && integer('rlayer') != null) return {...EMPTY,layer:integer('rlayer')};
  /* layer: undefined = 用户还没选过层；结构页据此默认打开最后一层（Phase 334）。点“返回全层”会显式设为 null */
  return { ...EMPTY, layer: undefined };
}

function writeSelection(selection) {
  const url = new URL(window.location.href);
  for (const [field, key] of Object.entries(KEYS)) {
    const value = selection[field];
    if (value === '' || value == null) url.searchParams.delete(key);
    else url.searchParams.set(key, String(value));
  }
  for (const key of ['pblock','pkind','phead','punit','tblock','thead','tunit','ttab','rlayer']) url.searchParams.delete(key);
  if (url.href !== window.location.href) window.history.replaceState(window.history.state, '', url);
}

function selectModel(models, modelKey, snapshot) {
  if (snapshot) {
    const scoped = models.filter(model => model.model_snapshot === snapshot);
    return scoped.find(model => model.model === modelKey) || (scoped.length === 1 ? scoped[0] : null);
  }
  return modelKey ? models.find(model => model.model === modelKey) :
    models.find(model => model.census_run && count(model.n_layers)) || models.find(model => count(model.n_layers));
}

function modelDimensions(model, architecture, localConfig) {
  const arch = architecture?.available && architecture.model_key === model.model && architecture.n_layers === model.n_layers &&
    (!model.d_model || architecture.d_model === model.d_model) && (!model.d_ff || architecture.d_ff === model.d_ff) ? architecture : null;
  const config = model.config || localConfig;
  return { arch, heads: count(model.n_heads ?? config?.num_attention_heads ?? arch?.n_heads),
    units: count(model.d_ff ?? config?.intermediate_size ?? arch?.d_ff),
    vocab: count(arch?.vocab_size ?? config?.vocab_size) };
}

function focusLocation(body, layers, dims) {
  const index = body.index || {};
  const block = body.layer;
  if (!numbered(block) || block >= layers) return null;
  if (index.space === 'mlp' && numbered(index.unit) && index.unit < dims.units) return { layer: block + 1, module: 'mlp', index: index.unit };
  if (index.space === 'attention' && numbered(index.head) && index.head < dims.heads) return { layer: block + 1, module: 'attn', index: index.head };
  return null;
}

function toPoint(location, dims) {
  const n = location.module === 'attn' ? dims.heads : dims.units;
  const [x, z] = towerIndexPosition(location.module, n, location.index);
  return { hs: location.layer, x, z };
}

function line(from, to, color, dash, width) {
  return { hs1: from.hs, hs2: to.hs, p1: [from.x, from.z], p2: [to.x, to.z], color, dash, width };
}

function researchGraph(archive, snapshot, selected, nLayers, dims) {
  if (!archive || !snapshot || selected.index == null) return { cards: [], links: [], lines: [] };
  const objects = (archive.objects || []).filter(item => item.model_snapshot === snapshot);
  const features = new Map(objects.filter(item => item.type === 'feature').map(item => [item.id, item]));
  const cards = [...features.values()].filter(item => {
    const loc = focusLocation(item.body, nLayers, dims);
    return loc?.layer === selected.layer && loc.module === selected.module && loc.index === selected.index;
  });
  const ids = new Set(cards.map(item => item.id));
  const links = [], lines = [];
  for (const entry of objects) {
    if (entry.type !== 'edge') continue;
    const { source, target } = entry.body;
    if (!ids.has(source) && !ids.has(target)) continue;
    const incoming = ids.has(target);
    const otherId = incoming ? source : target;
    const other = features.get(otherId)?.body;
    const location = other && focusLocation(other, nLayers, dims);
    links.push({ entry, incoming, otherId, other, location });
    if (location && (location.layer !== selected.layer || location.index !== selected.index || location.module !== selected.module)) {
      const anchor = toPoint(selected, dims), neighbor = toPoint(location, dims);
      lines.push(incoming ? line(neighbor, anchor, '#54789a') : line(anchor, neighbor, '#54789a'));
    }
  }
  return { cards, links, lines };
}

function neighbors(candidate, model, block, index, units) {
  if (!candidate?.available || candidate.block !== block || candidate.unit !== index ||
      candidate.n_layers !== model.n_layers || candidate.d_ff !== units ||
      (candidate.model_key && candidate.model_key !== model.census_key) ||
      (candidate.run && candidate.run !== model.census_run) ||
      (candidate.model_snapshot && candidate.model_snapshot !== model.model_snapshot)) return null;
  const valid = (items, b) => (items || []).filter(row => numbered(b) && b < model.n_layers &&
    row.block === b && numbered(row.unit) && row.unit < units && Number.isFinite(row.score));
  return { upstream: valid(candidate.upstream, block - 1), downstream: valid(candidate.downstream, block + 1),
    verified: Boolean(candidate.model_snapshot && candidate.model_snapshot === model.model_snapshot) };
}

function EvidenceSource({ sha }) {
  const source = useApi(`/api/research-library/artifacts/${encodeURIComponent(sha)}`);
  if (source.error) return <Empty>{source.error}<ExtraPackPrompt pack={source.missingPack} /></Empty>;
  return source.data ? <pre className="v8-artifact-json">{JSON.stringify(source.data, null, 2)}</pre> : <p className="v8-note">读取原始证据…</p>;
}

function useCandidateCircuit(query, modelKey) {
  const packedPath = query ? `/api/neurons/circuit?${query}&k=5` : null;
  const source = useApi(query ? OFFLINE_CLIENT ? packedPath : `/api/neurons/circuit-archive?${query}` : null);
  const tick = useSyncExternalStore(subscribe, storeTick);
  const [saved, setSaved] = useState(null);
  const fallback = !OFFLINE_CLIENT && source.error ? packedPath : null;
  useEffect(() => {
    if (!fallback) return undefined;
    let disposed = false;
    getExtraEntry(fallback).then(data => {
      if (!disposed) setSaved({ path: fallback, data });
    }).catch(() => {});
    return () => { disposed = true; };
  }, [fallback, tick]);
  if (fallback && saved?.path === fallback && saved.data) return { data: saved.data };
  return fallback ? { ...source, error: '可下载预计算的跨层权重候选包。', missingPack: `circuit:${modelKey}` } : source;
}

function RecursiveExamples({ censusKey, layers, units, onPick }) {
  const api = useApi(`/api/neurons/explained?model_key=${encodeURIComponent(censusKey)}`);
  const rows = api.data?.model_key === censusKey ? (api.data.results || []).filter(row =>
    numbered(row.block) && row.block < layers && numbered(row.unit) && row.unit < units && String(row.cause || '').includes('递归泛化')) : [];
  if (api.error) return <Empty>{api.error}</Empty>;
  if (!api.data) return <p className="v8-note">读取解释检验…</p>;
  return rows.length ? <ul className="v8-structure-links">{rows.map(row => <li key={`${row.block}:${row.unit}`}>
    <button type="button" className="v6-linkbtn" onClick={() => onPick({ layer: row.block + 1, module: 'mlp', index: row.unit, mode: 'relations' })}>
      B{row.block} · U{row.unit}
    </button><span> · {row.grade}</span>
  </li>)}</ul> : <p className="v8-note">该模型暂无标为递归泛化的解释检验。</p>;
}

function GraphCards({ graph, archive, snapshot }) {
  if (archive.error) return <Empty>{archive.error}<ExtraPackPrompt pack={archive.missingPack} /></Empty>;
  if (!archive.data) return <p className="v8-note">读取已登记的研究卡…</p>;
  if (!graph.cards.length) return <p className="v8-note">该索引暂无已定位的研究卡。</p>;
  function renderCard(item) {
      const { body } = item, evidence = body.evidence || {};
      const params = new URLSearchParams({ lens: 'process', station: 'execution', run: body.run_id || '', snapshot });
      return <div key={item.id}>
        <b>{body.label || item.id}</b><p className="v8-note">{EVIDENCE_LABEL[evidence.level] || '已有登记'}{body.boundary && ` · ${body.boundary}`}</p>
        {body.run_id && <a href={workbenchHref(params.toString())}>查看研究报告</a>}
        <ResearchDisclosure title="定位记录 JSON"><pre className="v8-artifact-json">{JSON.stringify(body, null, 2)}</pre></ResearchDisclosure>
        {/^[0-9a-f]{64}$/.test(evidence.artifact_sha256 || '') && <ResearchDisclosure title="原始证据 JSON">
          <EvidenceSource sha={evidence.artifact_sha256} />
        </ResearchDisclosure>}
      </div>;
  }
  return <div className="v8-structure-records">
    {graph.cards.slice(0, 3).map(renderCard)}
    {graph.cards.length > 3 && <ResearchDisclosure title={`更多研究卡（${graph.cards.length - 3}）`}>
      <div className="v8-structure-records">{graph.cards.slice(3).map(renderCard)}</div>
    </ResearchDisclosure>}
  </div>;
}

function Relations({ graph, archive, candidate, paths, onPick, query, recursive }) {
  return <>
    {recursive && <p className="v8-note"><b>递归泛化线索</b> · 该单元的解释检验记录了这一原因分类。前后层候选仍需在相应语境上独立验证。</p>}
    <h4>已登记关系</h4>
    {archive.error && <Empty>{archive.error}<ExtraPackPrompt pack={archive.missingPack} /></Empty>}
    {!archive.data && !archive.error && <p className="v8-note">读取图谱关系…</p>}
    {archive.data && (graph.links.length ? <ul className="v8-structure-links">{graph.links.map(({ entry, incoming, otherId, other, location }) => {
      const name = location ? `B${location.layer - 1} · ${label(location.module, location.index)}` : other?.label || (otherId?.startsWith('language:') ? otherId.slice(9) : '已登记的非单元节点');
      return <li key={entry.id}>
        <span>{incoming ? '来源' : '去向'} · </span>
        {location ? <button type="button" className="v6-linkbtn" onClick={() => onPick(location)}>{name}</button> : <b>{name}</b>}
        <span> · {EVIDENCE_LABEL[entry.body.evidence?.level] || '已有登记'}</span>
        <ResearchDisclosure title="关系记录"><pre className="v8-artifact-json">{JSON.stringify(entry.body, null, 2)}</pre></ResearchDisclosure>
      </li>;
    })}</ul> : <p className="v8-note">该索引暂无已登记的图谱关系。</p>)}
    {query && <>
      <h4>前后层权重候选</h4>
      {candidate.error ? <Empty>{candidate.error}<ExtraPackPrompt pack={candidate.missingPack} /></Empty>
        : paths ? <>
          <p className="v8-note">虚线表示预计算的相邻层权重路径候选。{paths.verified ? '权重快照已匹配。' : '旧候选包按普查桶与层尺寸定位，完整权重身份待核验。'}</p>
          <div className="v8-structure-neighbors">
            {[[paths.upstream, '上游 · 前一层写入'], [paths.downstream, '下游 · 后一层读取']].map(([rows, title]) =>
              <div key={title}><b>{title}</b>
                {rows.length ? rows.map(row => <button type="button" className="v8-cir-row" key={row.unit}
                  onClick={() => onPick({ layer: row.block + 1, module: 'mlp', index: row.unit })}>
                  <strong>B{row.block} · U{row.unit}</strong><span>{row.via === 'head' ? `经 H${row.head}` : '残差直连'}</span>
                  <em>{row.score.toFixed(4)}</em>{row.explained && <small>{row.explained}</small>}
                </button>) : <p className="v8-note">暂无此方向的候选记录。</p>}
              </div>)}
          </div>
        </> : <p className="v8-note">{candidate.data?.reason || (candidate.data ? '候选记录的身份或维度与当前选择不同。' : '读取预计算的前后层候选…')}</p>}
    </>}
  </>;
}

function EmbeddingIndex({ total, index, coordinate, dimensions, values, maxAbs, onToken, onCoordinate }) {
  const [page, setPage] = useState(0);
  const size=256, start=(index==null?page:Math.floor(index/size))*size;
  return <div className="v8-route">
    <p className="v8-note">词表 {total ? format(total) : '—'} 行 · {format(dimensions)} 个原生坐标 / 行</p>
    {total>0 && <><div className="v6-toolbar"><label>词表 Token ID <input aria-label="词表 Token ID" type="number" min="0" max={total-1}
      value={index ?? ''} onChange={e=>{const v=Number(e.target.value);if(e.target.value==='')onToken(null);else if(Number.isInteger(v)&&v>=0&&v<total)onToken(v)}}/></label>
      <button type="button" className="fw-tbtn" disabled={start===0} onClick={()=>{setPage(start/size-1);onToken(null)}}>前一组</button>
      <button type="button" className="fw-tbtn" disabled={start+size>=total} onClick={()=>{setPage(start/size+1);onToken(null)}}>后一组</button>
      <span className="v8-note">#{start}–{Math.min(total-1,start+size-1)}</span>
    </div>
    <ArrayCanvas label="词表 Token ID" count={Math.min(size,total-start)} columns={32} selected={index==null?null:index-start}
      color="#adbacb" onPick={i=>onToken(start+i)}/>
    </>}
    {index!=null && <><p className="v8-note">Token #{index} · 完整向量</p>
      <ArrayCanvas label="词嵌入原生坐标" count={dimensions} columns={64} values={values} maxAbs={maxAbs}
        selected={coordinate} color="#adbacb" onPick={onCoordinate}/>
      {coordinate!=null && <p>E[{coordinate}] {Number.isFinite(values?.[coordinate])?values[coordinate].toPrecision(6):'· 待读取'}</p>}
    </>}
  </div>;
}

export default function ModelStructure({ modelKey, onModel, view='graph', paramTab, onGraphMode, tm, onTm, historical, compact = false }) {
  const source = useApi('/api/model/skeleton', { pollWhilePending: true });
  const { snapshot, pmodel } = useResearchContext();
  const [selection, setSelection] = useState(()=>readSelection(view,paramTab));
  const models = source.data?.models || [];
  const registered = selectModel(models, modelKey, snapshot);
  const localOnly = Boolean(pmodel && !snapshot && !modelKey);
  const needLocalDimensions = DESKTOP && selection.layer != null && Boolean(registered?.model_snapshot);
  const localModels = useParameters(localOnly || needLocalDimensions ? '/api/parameters/models' : null);
  const local = localOnly ? localModels.data?.models?.find(row=>row.id===pmodel) : null;
  const model = local ? {model:local.name,n_layers:local.config.num_hidden_layers,d_model:local.config.hidden_size,
    d_ff:local.config.intermediate_size,config:local.config} : registered;
  const boundSnapshot=snapshot||model?.model_snapshot||'';
  const [legacy,setLegacy]=useState(()=>{
    const params=new URLSearchParams(window.location.search);
    return {tab:view==='live'?params.get('ttab'):'',template:view==='response'&&params.has('rlayer')&&params.has('tm'),
      head:/^\d+$/.test(params.get('thead')||'')?Number(params.get('thead')):null};
  });
  const defaultPanel=['live','response'].includes(view)?'data':'feature';
  const [structureView,setStructureView]=useState('3d');
  const [colorMode,setColorMode]=useState('type');
  const [showRelations,setShowRelations]=useState(true);
  const [panel,setPanel]=useState(()=>paramTab==='neurons'||legacy.tab==='evidence'?'evidence':defaultPanel);
  const [priorView,setPriorView]=useState(view);
  if(priorView!==view){setPriorView(view);setPanel(defaultPanel);setLegacy({});}
  const [embedding,setEmbedding]=useState(null),[archiveOverlay,setArchiveOverlay]=useState(null);
  const [featurePlot,setFeaturePlot]=useState('');
  const [narrow, setNarrow] = useState(() => window.innerWidth <= 720);
  /* 精简版响应对照：首屏只有最小对照，3D 层塔与当前数据点按钮再展开（Phase 333） */
  const [sceneOpen, setSceneOpen] = useState(() => new URLSearchParams(window.location.search).has('sblock'));
  useEffect(() => {
    const resize = () => setNarrow(window.innerWidth <= 720);
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);

  const layers = count(model?.n_layers);
  const layer = selection.layer === undefined && ['graph','parameters'].includes(view) && layers > 0 ? layers
    : numbered(selection.layer) && selection.layer <= layers ? selection.layer : null;
  const arch = useApi(layer != null && model ? `/api/model/arch?model_key=${encodeURIComponent(model.model)}` : null, { online: true });
  const module = layer===0?'embedding':layer>0&&['attn','mlp'].includes(selection.module)?selection.module:'';
  const candidates = needLocalDimensions ? (localModels.data?.models || []).filter(({ config }) =>
    config.num_hidden_layers === layers && config.hidden_size === model?.d_model &&
    (!model.d_ff || config.intermediate_size === model.d_ff)) : [];
  const candidate = candidates.find(row => row.id === pmodel) ||
    candidates.find(row => row.name.toLowerCase() === model?.model.split('|')[0].toLowerCase()) ||
    (candidates.length === 1 ? candidates[0] : null);
  const needsVerification = candidate && !arch.data?.available;
  const localIdentity = useParameters(needsVerification ? `/api/parameters/identity?model=${encodeURIComponent(candidate.id)}` : null);
  const localConfig = needsVerification && localIdentity.data?.sha256 === boundSnapshot ? candidate.config : null;
  const dims = useMemo(() => model ? modelDimensions(model, arch.data, localConfig) :
    { arch: null, heads: 0, units: 0, vocab: 0 }, [model, arch.data, localConfig]);
  const n = module==='attn'?dims.heads:module==='mlp'?dims.units:0;
  const vocab=dims.vocab;
  const index = module && numbered(selection.index) && selection.index < (module==='embedding'?vocab:n) ? selection.index : null;
  const coordinate=module==='embedding'&&numbered(selection.coordinate)&&selection.coordinate<model?.d_model?selection.coordinate:null;
  const mode = index != null && selection.mode === 'relations' ? 'relations' : '';
  const selected = useMemo(() => ({ layer, module, index, mode, coordinate }), [layer, module, index, mode, coordinate]);
  const block = layer - 1;
  const change = next => {
    if(next.layer!==selection.layer||next.module!==selection.module||next.index!==selection.index){
      const url=new URL(window.location.href);
      url.searchParams.delete('prow');url.searchParams.delete('pcol');
      window.history.replaceState(null,'',url);
      if(legacy.tab)setLegacy(current=>({...current,tab:'',head:null}));
    }
    setSelection(next);writeSelection(next);
  };
  useEffect(()=>{
    if(selection.layer!=null)writeSelection(selection);
  },[]); // Copy the legacy selection before changing modes.
  const pickLayer = layer => change({ ...EMPTY, layer });
  const pickModule = (layer, module) => change({ ...EMPTY, layer, module });
  const pickIndex = index => change({ ...selected, index, coordinate:null });
  const trace=useSpatialTrace({view,snapshot:boundSnapshot,selection:selected,onToken:pickIndex,legacy,
    onSourceSnapshot:!snapshot&&!modelKey?id=>{if(models.some(row=>row.model_snapshot===id))publishResearchContext({snapshot:id})}:undefined});
  const artifact = new URLSearchParams(window.location.search).get('artifact');
  const fragment = new URLSearchParams(window.location.search).get('feature');
  const linked = useApi(view==='parameters' && index==null && artifact ? `/api/research-library/artifacts/${encodeURIComponent(artifact)}` : null);
  useEffect(()=>{
    if(!linked.data||index!=null||!model)return;
    if(!snapshot&&linked.data.model_snapshot&&models.some(row=>row.model_snapshot===linked.data.model_snapshot)){
      publishResearchContext({snapshot:linked.data.model_snapshot});return;
    }
    const card=linked.data.feature_cards?.find(item=>item.id===fragment);
    if(linked.data.model_snapshot===boundSnapshot&&card&&Number.isInteger(card.layer)&&card.layer<layers&&
       card.index?.space==='mlp'&&Number.isInteger(card.index.unit)&&card.index.unit<dims.units)
      change({...EMPTY,layer:card.layer+1,module:'mlp',index:card.index.unit});
  },[linked.data,fragment,index,model,boundSnapshot,snapshot,layers,dims.units,models]);

  const query = module === 'mlp' && index != null && model?.census_run
    ? new URLSearchParams({ model_key: model.census_key, run: model.census_run, block, unit: index }).toString() : '';
  const unit = useApi(query ? `/api/neurons/unit?${query}` : null);
  const card = unit.data?.schema === 'unit_card.v1' && unit.data.model_snapshot === model?.model_snapshot &&
    unit.data.layer === block && unit.data.unit === index ? unit.data : null;
  const isB35 = module==='mlp' && block===35 && n===9728 && Boolean(boundSnapshot);
  const archive = useApi((index != null && model?.model_snapshot) || isB35 ? '/api/traces/research-space' : null);
  const b35 = useB35Features(archive.data,boundSnapshot,isB35);
  const b35Unit = useB35Unit(b35.data,index);
  const graph = useMemo(() => researchGraph(archive.data, model?.model_snapshot, selected, layers, dims),
    [archive.data, model?.model_snapshot, selected, layers, dims]);
  const circuitQuery = showRelations || mode ? query : '';
  const censusKey=model?.census_run?model.census_key:'';
  const towerData=useApi(censusKey&&['graph','parameters'].includes(view)?`/api/neurons/tower?model_key=${encodeURIComponent(censusKey)}`:null);
  const layerData=useApi(censusKey&&module==='mlp'&&colorMode!=='index'?`/api/neurons/layer?model_key=${encodeURIComponent(censusKey)}&block=${block}&run=${encodeURIComponent(model.census_run)}`:null);
  const layerColumns=layerData.data?.block===block?layerData.data.columns:null;
  const typeColors=useMemo(()=>colorMode==='type'&&layerColumns?.type?layerColumns.type.map(type=>TYPE_COLORS[type]):null,[colorMode,layerColumns]);
  const activityValues=useMemo(()=>colorMode==='activity'&&layerColumns?.dens?layerColumns.dens.map(value=>Math.log10(1+1000*(value||0))):null,[colorMode,layerColumns]);
  const circuit = useCandidateCircuit(circuitQuery, model?.census_key);
  const paths = circuitQuery ? neighbors(circuit.data, model, block, index, dims.units) : null;
  const target = index != null && module!=='embedding' ? toPoint(selected, dims) : null;
  const showLinks = showRelations || mode;
  const edges = showLinks ? [...graph.lines] : [];
  if (showLinks && paths && target) {
    const point = row => toPoint({ layer: row.block + 1, module: 'mlp', index: row.unit }, dims);
    const strongest = Math.max(1e-9, ...[...paths.upstream, ...paths.downstream].map(row => row.score));
    const width = row => 1 + 3 * row.score / strongest;
    for (const row of paths.upstream) edges.push(line(point(row), target, '#b4531c', [4, 3], width(row)));
    for (const row of paths.downstream) edges.push(line(target, point(row), '#1f6fb0', [4, 3], width(row)));
  }
  if (showLinks && target && b35Unit.data) {
    const upstream = new Set(Object.values(b35Unit.data.upstream).flatMap(rows=>rows.map(row=>row.unit)));
    for(const unit of upstream)
      if(!paths?.upstream.some(row=>row.block===34&&row.unit===unit))
        edges.push(line(toPoint({layer:35,module:'mlp',index:unit},dims),target,'#bd6d42',[4,3]));
  }

  const latest = card?.explanations?.at(-1);
  const recursive = card?.explanations?.some(entry => String(entry.cause || '').includes('递归泛化'));
  const nativeValues=view==='graph'||view==='parameters'?
    (archiveOverlay?.model_snapshot===boundSnapshot&&archiveOverlay?.block===block&&module==='mlp'?archiveOverlay.values:
      module==='embedding'&&embedding?.snapshot===boundSnapshot&&embedding?.index===index?embedding.values:null)
    : trace.pointValues;
  const measuredPlot = isB35 && ['graph','parameters'].includes(view) ? b35.data?.plots?.[featurePlot] : null;
  const values = measuredPlot || activityValues || nativeValues;
  const pointValues=values&&values.length===(module==='embedding'?model?.d_model:n)?values:null;
  const headScale=pointValues&&module==='attn'?(trace.maxAbs||Math.max(1e-12,...pointValues.map(Math.abs))):1;
  const weights = index!=null && <ParameterExplorer key={`${model?.model}:${layer}:${module}:${index}`} researchModel={model?.model}
    snapshot={boundSnapshot} selection={selected} onEmbedding={module==='embedding'?setEmbedding:undefined}
    onCoordinate={i=>change({...selected,coordinate:i})}
    onSelectModel={id=>{setEmbedding(null);setArchiveOverlay(null);publishResearchContext({pmodel:id});}}
    onLocate={p=>change({...EMPTY,layer:p.block+1,module:p.kind==='head'?'attn':p.kind,index:p.kind==='head'?p.head:p.unit})}
    onResponse={setArchiveOverlay}/>;
  const hideScene = compact && view === 'response' && !sceneOpen && index == null;
  return <section className="v8-route v8-structure-view" aria-label="模型结构浏览">
    <header className="v8-overview-heading">
      <div>{!(compact && view === 'live') && <h2>{VIEW_TEXT[view]?.[0] || '空间'}</h2>}{!compact && <p>{VIEW_TEXT[view]?.[1]}</p>}{model && <p className="v8-note wb-pro">{shortModel(model.model)}{model.precision ? ` · ${model.precision}` : ''} · {layers} 个 Transformer 层</p>}</div>
      {models.length > 1 && <label>模型 <select aria-label="结构模型" value={model?.model || ''} onChange={event => {
        const next = models.find(row => row.model === event.target.value);
        if (!next) return;
        onModel?.(next.model);
        publishResearchContext({ snapshot: next.model_snapshot || '', run: '', trace: '', context_trace: '', pmodel: '' });
        setEmbedding(null);setArchiveOverlay(null);
        change(EMPTY);
      }}>
        {!model && <option value="">选择已有模型</option>}
        {models.map(row => <option key={row.model} value={row.model}>{shortModel(row.model)}{row.precision ? ` · ${row.precision}` : ''}</option>)}
      </select></label>}
      <details className="v8-spatial-graph-menu wb-pro"><summary>研究图谱</summary><div className="v6-toolbar">
        {[['internal','全部研发成果'],['relations','共享关系与映射']].map(([key,label])=>
          <button type="button" className="fw-tbtn" key={key} onClick={()=>onGraphMode?.(key)}>{label}</button>)}
      </div></details>
    </header>
    {['graph','parameters'].includes(view)&&<div className="v6-toolbar" role="group" aria-label="结构视图">
      <button type="button" className="fw-tbtn" aria-pressed={structureView==='3d'} onClick={()=>setStructureView('3d')}>3D 模型空间</button>
      <button type="button" className="fw-tbtn" aria-pressed={structureView==='list'} onClick={()=>setStructureView('list')}>二维列表</button>
      {!compact&&<button type="button" className="fw-tbtn" onClick={()=>onGraphMode?.('language')}>语言图谱</button>}
    </div>}
    {view==='live'&&!compact&&<><InfoFlowMap model={model?.model}/><AttributionGraph snapshot={boundSnapshot} onOpen={node=>{change({...EMPTY,layer:node.l+1,module:node.k==='m'?'mlp':'attn',index:node.i});onGraphMode?.('structure');}}/></>}
    {view==='response'&&<MinimalPairs model={model?.model}/>}
    {compact&&view==='response'&&<div className="ck-pills"><button type="button" aria-pressed={sceneOpen} aria-expanded={sceneOpen} onClick={()=>setSceneOpen(o=>!o)}>{sceneOpen?'收起 3D 层塔与当前数据':'在 3D 层塔中定位'}</button></div>}
    {!hideScene&&trace.controls}
    {['graph','parameters'].includes(view)&&structureView==='list'&&<LayerList tower={towerData.data&&!towerData.data.pending?towerData.data:null} onPick={hs=>{setStructureView('3d');pickLayer(hs);}}/>}
    {hideScene ? null : source.error&&!local ? <Empty>{source.error}<ExtraPackPrompt pack={source.missingPack} /></Empty>
      : !source.data || source.data.pending ? <Empty>读取模型骨架…</Empty>
        : !model || !layers ? <Empty>当前模型快照暂无可核对的层结构。</Empty>
          : <div className="v8-structure-layout" hidden={['graph','parameters'].includes(view)&&structureView==='list'}>
            <section className="v8-card v8-structure-scene">
              <div className="v6-toolbar">
                <label>定位层 <select aria-label="定位模型层" value={layer ?? ''} onChange={event => pickLayer(event.target.value === '' ? null : Number(event.target.value))}>
                  <option value="">全部层</option><option value="0">词嵌入</option>
                  {Array.from({ length: layers }, (_, b) => <option key={b} value={b + 1}>B{b}</option>)}
                </select></label>
                {censusKey&&<label>着色 <select aria-label="神经元着色" value={colorMode} onChange={event=>setColorMode(event.target.value)}>
                  <option value="type">按类型</option><option value="activity">按活跃度</option><option value="index">不着色</option>
                </select></label>}
                <button type="button" className="fw-tbtn" aria-pressed={showRelations} onClick={()=>setShowRelations(value=>!value)}>前后关系</button>
                <span className="v8-note">拖动旋转 · 平移按钮或 Shift 拖动 · 滚轮缩放</span>
              </div>
              <Tower key={model.model} nLayers={layers} selected={layer} onPick={pickLayer} allowEmbedding
                modules={MODULES} focusModules panControls fit={0.92} arch={dims.arch} onPickModule={pickModule}
                selModule={module ? { hs: layer, mod: module } : null} height={narrow ? 335 : 540}
                caption={`LLM 全部 ${layers} 层的 3D 层塔${n ? `，B${block} 显示全部 ${n} 个${module === 'attn' ? ' Head' : ' MLP 单元'}` : ''}`}
                cells={n > 0 ? { hs: layer, kind: module, count: n, values:measuredPlot||activityValues, colors:module==='mlp'&&!measuredPlot?typeColors:null } : null} onPickIndex={pickIndex} target={target} edges={edges} />
              <p className="v8-note">{layer == null ? '点击层板查看 Head 和 MLP。底部为词嵌入，顶部为输出头。'
                : showLinks && index != null ? '实线为已登记关系。橙色虚线是上游权重候选，蓝色虚线是下游权重候选；线越粗权重越强，候选不等于实际因果。'
                  : '板内位置按原始编号排列，选中点以红圈标出。'}</p>
              {module==='mlp'&&colorMode!=='index'&&layerData.missingPack&&<p className="v8-note">下载神经元数据后可以按类型或活跃度着色。<ExtraPackPrompt pack={layerData.missingPack}/></p>}
              {module==='mlp'&&colorMode==='type'&&<div className="v6-toolbar" aria-label="神经元类型图例">{TYPE_LABELS.map((name,type)=><span className="v8-note" key={name}><i className="cg-swatch" style={{background:TYPE_COLORS[type]}}/>{name}</span>)}</div>}
            </section>
            {layer == null ? <aside className="v8-spatial-empty">选择一层查看词嵌入、Head 或 MLP。</aside> :
              <section className="v8-card v8-structure-focus" aria-label="当前层详情">
              <div className="v8-overview-heading"><h3>{layer === 0 ? '词嵌入' : `B${block} · 层结构`}</h3>
                <button type="button" className="fw-tbtn" onClick={() => pickLayer(null)}>返回全层</button>
              </div>
              {layer === 0 ? <EmbeddingIndex total={vocab} index={index} coordinate={coordinate} dimensions={model.d_model||0}
                  values={pointValues} maxAbs={trace.maxAbs} onToken={pickIndex} onCoordinate={i=>change({...selected,coordinate:i})}/>
                : <div className="v8-structure-modules">
                  {[['attn', '注意力 Head', dims.heads], ['mlp', 'MLP 单元', dims.units]].map(([id, name, total]) =>
                    <button key={id} className="fw-tbtn" type="button" aria-pressed={module === id} onClick={() => pickModule(layer, id)}>
                      <i className={id} />{name}{total > 0 ? ` · ${format(total)}` : ''}
                    </button>)}
                </div>}
              {module && !(module==='embedding'?vocab:n) && <div className="v8-note">
                {arch.loading || (needLocalDimensions && localModels.loading) ? '读取架构维度…'
                  : localIdentity.loading ? '正在核验本机架构与研究快照…'
                    : '当前快照的模块索引数尚无可用记录。'}<ExtraPackPrompt pack={arch.missingPack} />
              </div>}
              {trace.colourControls}
              {module === 'attn' && n > 0 && <div className="v8-head-grid" aria-label="全部注意力 Head">
                {Array.from({ length: n }, (_, head) => <button type="button" key={head} className="fw-tbtn" aria-pressed={head === index}
                  style={pointValues?{backgroundColor:`rgba(${pointValues[head]<0?'235,104,52':'42,120,214'},${.08+.58*Math.abs(pointValues[head])/headScale})`}:undefined}
                  onClick={() => pickIndex(head)}>H{head}</button>)}
              </div>}
              {module === 'mlp' && n > 0 && <label>单元索引 <input aria-label="定位 MLP 单元" type="number" min={0} max={n - 1} value={index ?? ''}
                placeholder={`0–${n - 1}`} onChange={event => {
                  const v = event.target.value;
                  pickIndex(v === '' ? null : Math.max(0, Math.min(n - 1, Math.trunc(Number(v)))));
                }} /></label>}
              {module === 'mlp' && n > 0 && <>
                {b35.data && <div className="v6-toolbar">
                  <span>B35 特征覆盖 {format(b35.data.unit_count)} / {format(n)}</span>
                  {['graph','parameters'].includes(view)&&<label>特征叠加 <select aria-label="B35 特征叠加" value={featurePlot} onChange={e=>setFeaturePlot(e.target.value)}>
                    <option value="">原生索引</option><option value="natural_std">自然激活标准差</option>
                    <option value="lockbox_sign">长链方向一致率</option><option value="replicated">双留出复现</option>
                  </select></label>}
                </div>}
                <ArrayCanvas label="MLP 单元" count={n} selected={index} onPick={pickIndex} color="#adbacb" values={pointValues} colors={measuredPlot?null:typeColors} maxAbs={measuredPlot||activityValues?undefined:trace.maxAbs}/>
                {b35.stage ? <ResearchDisclosure title="递归泛化 · 六步留出" key={b35.stage.report_sha256}>
                  <B35LayerResults sha={b35.stage.report_sha256}/>
                </ResearchDisclosure> : model.census_key && <ResearchDisclosure title="递归泛化线索" key={`recursive:${block}:${index}`}>
                  <RecursiveExamples censusKey={model.census_key} layers={layers} units={n} onPick={change} />
                </ResearchDisclosure>}
              </>}
              {trace.pointNote&&pointValues&&<p className="v8-note">{trace.pointNote} · 原生索引顺序 · 蓝正、橙负</p>}
            </section>}
          </div>}
    {linked.error&&index==null&&<Empty>{linked.error}<ExtraPackPrompt pack={linked.missingPack}/></Empty>}
    {!hideScene&&(index!=null||view==='live'||view==='response')&&<section className="v8-card v8-spatial-detail" aria-label={index!=null?'选中索引':'当前数据'}>
      <h3>{index==null?'当前数据':module==='embedding'?`Token #${index}`:`B${block} · ${label(module,index)}`}</h3>
      {index!=null&&<div className="v6-toolbar" role="group" aria-label="对象详情">
        {[...(['live','response'].includes(view)?[['data','当前数据']]:[]),...(module==='embedding'?[]:[['feature','特征']]),['parameters','真实参数'],['evidence','研究证据'],['history','历史测试']].map(([key,title])=>
          <button type="button" className="fw-tbtn" key={key} aria-pressed={panel===key&&!mode} onClick={()=>{
            setPanel(key); if(mode)change({...selected,mode:''});
          }}>{title}</button>)}
        {module!=='embedding'&&<button type="button" className="fw-tbtn" aria-pressed={Boolean(mode)}
          onClick={()=>change({...selected,mode:mode?'':'relations'})}>关联模式</button>}
      </div>}
      {mode ? <Relations graph={graph} archive={archive} candidate={circuit} query={circuitQuery} paths={paths} recursive={recursive}
        onPick={location=>change({...location,mode})}/> :
        (panel==='data'||index==null) && (view==='live'||view==='response') ? trace.detail : null}
      {mode&&b35.data&&<B35UnitFeatures row={b35Unit} index={b35.data} recursive
        onUpstream={u=>change({layer:35,module:'mlp',index:u,mode:'relations'})}/>}
      {!mode&&index!=null&&(panel==='parameters'||(panel==='feature'&&module==='embedding'))&&weights}
      {!mode&&index!=null&&panel==='parameters'&&module==='attn'&&<HeadInternals snapshot={boundSnapshot} block={block} head={index}/>}
      {!mode&&index!=null&&panel==='feature'&&module==='attn'&&<HeadFeature snapshot={boundSnapshot} block={block} head={index}/>}
      {!mode&&index!=null&&panel==='feature'&&module==='mlp'&&(card?<NeuronUnitCardContent data={card}/>:<p className="v8-note">{unit.error||(unit.loading?'读取神经元卡片…':'暂无与当前快照匹配的神经元卡片。')}<ExtraPackPrompt pack={unit.missingPack}/></p>)}
      {!mode&&index!=null&&panel==='evidence'&&<>
        {b35.data&&<><B35UnitFeatures row={b35Unit} index={b35.data}/>
          <ResearchDisclosure title="本单元的递归泛化"><B35UnitFeatures row={b35Unit} index={b35.data} recursive
            onUpstream={u=>change({layer:35,module:'mlp',index:u,mode:'relations'})}/></ResearchDisclosure></>}
        {card ? <>
          <p>{latest?.explanation || '已归档自然语料统计，解释仍待检验。'}</p>
          {recursive&&<p className="v8-note">解释检验记录 · 递归泛化线索</p>}
          <ul className="v8-overview-tags">{[['读出','read'],['控制','control'],['解释','explain']].map(([name,key])=>
            <li key={key}>{name} · {card.evidence_axes?.[key] || '未登记'}</li>)}</ul>
        </> : query&&!b35.data ? <div className="v8-note">{unit.error || (unit.loading ? '读取单元检验…' : '暂无与当前快照匹配的单元卡。')}<ExtraPackPrompt pack={unit.missingPack}/></div> : null}
        <GraphCards key={`${module}:${block}:${index}`} graph={graph} archive={archive} snapshot={boundSnapshot}/>
      </>}
      {!mode&&index!=null&&panel==='history'&&<>
        {card ? <NeuronUnitCardContent data={card}/> : <p className="v8-note">该对象暂无匹配的单元测试档案。</p>}
        <GraphCards key={`history:${module}:${block}:${index}`} graph={graph} archive={archive} snapshot={boundSnapshot}/>
        {historical}
      </>}
      {!mode&&view==='response'&&<ResearchDisclosure title="模板响应汇总与历史对照" initiallyOpen={legacy.template}>
        <p className="v8-note">模板模型桶和层级统计在此查看。汇总结果不叠加到单个原生对象。</p>
        <Suspense fallback={<Empty>读取模板响应…</Empty>}><ResponseExplorer key={`template:${model?.model}`}
          initialTm={tm||undefined} initialBucket={model?.model} strictBucket
          syncLayer={layer!=null?`final@L${layer}`:undefined}
          onLayer={tag=>{const match=/@L(\d+)$/.exec(tag);if(match&&Number(match[1])!==layer)pickLayer(Number(match[1]))}}
          onContext={({tm:next})=>onTm?.(next)}/></Suspense>
        {historical}
      </ResearchDisclosure>}
    </section>}
    {!compact&&view==='response'&&<RelationBoards/>}
    {!compact&&['graph','parameters'].includes(view)&&<BasisCompare/>}
  </section>;
}
