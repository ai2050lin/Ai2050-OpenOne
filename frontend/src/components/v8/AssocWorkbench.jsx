/* 空间 · 模型分析工作台（Phase 346）。
   保留页面布局；3D 里显示完整 LLM 层塔，旁边放大选中层的 Head / HiddenState / MLP，
   点格子看具体数值；可全屏。下：控制台——输入文本运行、token 条、操作面板（全屏时变成浮窗，同一个面板）、插件面板。
   控制台保留四个任务页签；逐层变化与条件对照统一到运行页签，其他分析通过插件管理添加。
   数据：本机后端可读时，从 /api/traces/mainline/ma-h2/ 读 MA-H2 discovery 冻结样本的原生值（float16）；
   没有匹配的原生数据时，保持结构查询并说明缺失原因，不回退为其他模型的案例数据。 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SCENE_MOUSE_HINT } from '../scenePointer.js';
import Scene from './AssocLayerScene.jsx';
import { ASSOC_VIEWS as VIEWS } from './assocLayerGeometry.js';
import ModelStructure from './ModelStructure.jsx';
import { useSpatialObject, writeSpatialObject } from './spatialObjectState.js';
import { nativeGrid as grid, nativeTexture as tex, positiveDimension } from './nativeTexture.js';
import { publishResearchContext, useResearchContext } from '../app/researchContext.js';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import ObservationTask from './ObservationTask.jsx';
import { ArchiveHead, ArchiveHeadCondition, archiveHeadCase, archivedHeadPrompt, EXAMPLE_SITES, ExamplePositions, ExampleSource, loadGraphExample, NeuronExample, WORKSPACE_MODES } from './GraphWorkspaceEvidence.jsx';
import {WorkspaceAttention, WorkspaceGeometry} from './WorkspaceAnalysisTools.jsx';
import SparseDictionaryExplorer from './SparseDictionaryExplorer.jsx';


import { DESKTOP } from '../../config.js';
import { useApi } from '../app/rdc_fusion/api.js';
import { TestsPanel, TopTokens } from './AssocConsole.jsx';
import { liveSampler, runOnce, useLiveBase, useLiveModel, useLocalBase } from './liveRun.js';
import { request } from './MechanismTraceExplorer.jsx';
import {archiveShortcuts, readObservationArchive, useObservation, useWorkspaceCatalog, useWorkspaceResource} from './nativeObservationArchive.js';
import {useSceneHeadField} from './useSceneHeadField.js';
import {MlpAtlasControls, MlpAtlasReport, useMlpAtlas} from './MlpAtlas.jsx';
import {matchesReplayManifest, nativeAddress} from './workspaceRecordReplay.js';
import { shortModel } from './v8Util.js';
import './assoc_workbench.css';
import './graph_workspace.css';
const TestRecords = lazy(() => import('./TestRecords.jsx'));
const FullNativeField = lazy(() => import('./FullNativeField.jsx'));

const POSN = { ans: '答案位置', qname: '问题中的实体名', scope: '范围词', tname: '资料里的实体名', tcol: '资料里的颜色', lastcol: '最后一条记录的颜色' };
const CN = { red: '红', yellow: '黄', green: '绿', blue: '蓝', white: '白', black: '黑', brown: '棕', orange: '橙', purple: '紫', pink: '粉', gray: '灰' };
// Functional plugins share the same model, object and run; activation never executes a model.
const WORKSPACE_PLUGINS = [
  { id: 'structure', name: '结构与档案', panel: 'object', desc: '查询完整结构、真实参数、特征与历史证据' },
  { id: 'live', name: '运行响应', panel: 'run', desc: '输入文本执行一次前向，或回放已有运行；执行需手动点击' },
  { id: 'response', name: '条件对照', panel: 'run', desc: '回放三线索选源，或选择同快照样本对比较响应' },
];
const MODE_PLUGINS = [
  { id: 'langmeasure', name: '语言项实测', tag: 'on', desc: '一项语言能力在哪一层读得出' },
  { id: 'relations', name: '共享关系', tag: 'on', desc: '不同语言能力是否共用同一个读出方向' },
];
const PLUGINS = [
  ['分析视角', MODE_PLUGINS],
  ['观察', [
    { id: 'hs', name: 'HiddenState', tag: 'on', native: true, desc: '放大当前层全部 HiddenState 坐标，点格子看原生值与研究记录' },
    { id: 'mlp', name: 'MLP 神经元', tag: 'on', native: true, desc: '当前块全部 MLP 中间单元；点选查看真实激活、特征与前后层关联' },
    { id: 'heads', name: '注意力头', tag: 'on', desc: '当前块全部 Head；颜色为采集查询位置对目标 token 的注意力，缺原生数据时注明案例汇总' },
  ]],
  ['行业工具', [
    { id: 'lens', name: 'Logit Lens', tag: 'on', desc: '逐层读出：资料色与常识色的 logit（每层先做最终归一化再接输出头）' },
    { id: 'dla', name: '直接 Logit 归因', tag: 'todo', desc: '各头 / 单元写入在答案方向上的投影；需要本机权重' },
    { id: 'patch', name: '激活修补', tag: 'have', desc: 'Activation patching：项目 mech_bench exact' },
    { id: 'atp', name: '归因修补', tag: 'have', desc: 'AtP / AtP+IG：项目 mech_bench；atp_ig 在 Qwen3 上偏重前段（Phase 345）' },
    { id: 'tuned', name: 'Tuned Lens', tag: 'todo', desc: '每层训练一个仿射读出' },
    { id: 'attnviz', name: '注意力分布', tag: 'on', desc: '一个查询位置各来源的权重、熵、有效来源数与累计占比' },
    { id: 'geometry', name: '表征几何', tag: 'on', desc: '对齐同一输入的多个位置，用线性 CKA 和 RSA 比较两层及其距离矩阵' },
    { id: 'sae', name: 'SAE 特征字典', tag: 'on', desc: '按模型快照查看稀疏字典特征、留出复现与反例' },
    { id: 'tc', name: 'Transcoder 归因图', tag: 'have', desc: '特征到 logit 的归因图；项目 transcoder.v1（第 30 块）' },
    { id: 'probe', name: '线性探针', tag: 'have', desc: '逐层可读出检验；项目 analysis.v1' },
    { id: 'steer', name: '引导向量', tag: 'todo', desc: '在某层加方向，正负双向' },
  ]],
  ['检验', [{ id: 'tests', name: '三类检验', tag: 'on', desc: '查看冻结运行的响应预测、输出预测和原生机制检验报告，不自动执行新检验' }]],
];
const ALL = PLUGINS.flatMap(([, ps]) => ps);
const AVAILABLE = ALL.filter(p => p.tag === 'on');
const PENDING = ALL.filter(p => p.tag !== 'on');
const DEFAULT_PLUGINS = ['heads', 'hs', 'mlp', 'lens'];
const ANALYSIS_TOOL_IDS = new Set(['attnviz','geometry','sae']);
const PLUGIN_PREFERENCES = 'ai2050.model-explorer.plugins.v1';
function initialPlugins(mode) {
  let ids=DEFAULT_PLUGINS;
  try {
    const saved=JSON.parse(localStorage.getItem(PLUGIN_PREFERENCES));
    if(Array.isArray(saved))ids=saved.filter(id=>AVAILABLE.some(p=>p.id===id));
  } catch { /* Preferences may be unavailable in isolated browsers. */ }
  const preferred=new URLSearchParams(window.location.search).get('observePlugin');
  return new Set([...ids,...(AVAILABLE.some(p=>p.id===preferred)?[preferred]:[]),...(OPTIONAL_MODE_IDS.has(mode)?[mode]:[])]);
}
const OPTIONAL_MODE_IDS = new Set(MODE_PLUGINS.map(p => p.id));
const MODE_IDS = new Set(['live', 'response', ...OPTIONAL_MODE_IDS]);

/* ---------- 数据 ---------- */
function f16(o) {      // float16 小端 → Float32Array
  const bin = atob(o.b64), n = bin.length / 2, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8), s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return { a: out, shape: o.shape };
}
/* 实时运行：一次取一个 token 的全部层（/api/traces/{id}/bundle），查询位置固定为最后一个 token */
function useBundle(base, id, position) {
  const [s, set] = useState({});
  useEffect(() => {
    if (base == null || !id || position == null) return undefined;
    let dead = false;
    request(base, `/api/traces/${id}/bundle?position=${position}`).then(d => { if (!dead) set({ key: `${base}:${id}:${position}`, data: d }); })
      .catch(e => { if (!dead) set({ key: `${base}:${id}:${position}`, error: e.message }); });
    return () => { dead = true; };
  }, [base, id, position]);
  return s.key === `${base}:${id}:${position}` ? s : {};
}
function useSample(base, i) {
  const [s, set] = useState({});
  useEffect(() => {
    if (base == null || i == null) return undefined;
    let dead = false;
    request(base, `/api/traces/mainline/ma-h2/sample/${i}`).then(d => {
      if (dead) return;
      set({ key: `${base}:${i}`, data: { ...d, hs: f16(d.hs), mlp: f16(d.mlp), attn: f16(d.attn) } });
    }).catch(e => { if (!dead) set({ key: `${base}:${i}`, error: e.message }); });
    return () => { dead = true; };
  }, [base, i]);
  return s.key === `${base}:${i}` ? s : {};
}
const norm = v => { let s = 0; for (const x of v) s += x * x; return Math.sqrt(s); };
const cosv = (u, v) => { let a = 0, b = 0, c = 0; for (let i = 0; i < u.length; i++) { a += u[i] * v[i]; b += u[i] * u[i]; c += v[i] * v[i]; } return a / Math.sqrt(b * c); };
function pct(arrs, q) { const a = []; for (const v of arrs) for (const x of v) a.push(Math.abs(x)); a.sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(a.length * q))] || 1; }
const absMax = v => { let m = 0; for (const x of v) m = Math.max(m, Math.abs(x)); return m || 1; };
function divc(v, m) { const t = Math.max(-1, Math.min(1, v / (m || 1))), e = Math.abs(t) ** 0.6, c = t >= 0 ? [31, 95, 191] : [212, 98, 43]; return [255 + (c[0] - 255) * e, 255 + (c[1] - 255) * e, 255 + (c[2] - 255) * e]; }
function seqc(v, m) { const e = Math.max(0, Math.min(1, v / (m || 1))) ** 0.5; return [255 - 225 * e, 255 - 160 * e, 255 - 64 * e]; }
const put = (d, i, rgb) => { d[i * 4] = rgb[0]; d[i * 4 + 1] = rgb[1]; d[i * 4 + 2] = rgb[2]; d[i * 4 + 3] = 255; };
const rawNumber = v => Number.isFinite(v) ? String(v) : '未采集';
const scaleNumber = v => Number.isFinite(v) ? Number(v.toPrecision(5)).toString() : '—';

function heatTexture(width,height,count,values,scale,color){
  return tex(width,height,pixels=>{
    for(let i=0;i<width*height;i++){
      const value = values?.[i];
      const missing = (i%width+Math.floor(i/width))%2 ? [190,201,214] : [211,220,229];
      put(pixels,i,i>=count?[246,248,251]:Number.isFinite(value)?color(value,scale):missing);
    }
  });
}

/* 一条样本的取值器（层 l、位置 p） */
function sampler(d) {
  if (!d) return null;
  const [P, NL, DM] = d.hs.shape, [NB, NF] = d.mlp.shape, NH = d.attn.shape[1], T = d.attn.shape[2];
  return {
    P, NL, DM, NB, NF, NH, T, d,
    hs: (p, l) => d.hs.a.subarray((p * NL + l) * DM, (p * NL + l + 1) * DM),
    mlp: b => d.mlp.a.subarray(b * NF, (b + 1) * NF),
    att: (b, h) => d.attn.a.subarray((b * NH + h) * T, (b * NH + h + 1) * T),
  };
}

const VORDER = ['iso', 'side', 'top'];
const CONSOLE_TABS = [
  ['object','对象详情'], ['run','运行与对照'], ['plugin','插件分析'], ['research','成果与方法'],
];
const MODE_TABS = {
  live:['run','object','plugin'],
  archive:['run','plugin','research'],
  features:['object','plugin'],
};
// Index of the bundled Phase 2933 sample; its BF16 fields are loaded on demand.

function hasExplorerContext() {
  const query = new URLSearchParams(window.location.search);
  return ['sblock','smodule','sindex','pkind','pblock','phead','punit','prow','tblock','thead','tunit','ttab','rlayer',
    'block','unit','trace','context_trace','run','snapshot','observeMode','observeTab','observePlugin','headCase']
    .some(key => query.has(key) && query.get(key) !== '');
}

/* ---------- 主组件 ---------- */
const ThreeCueComparison = lazy(() => import('./ThreeCueComparison.jsx'));

export default function AssocWorkbench({ mode, onMode, contextBar, children, workspaceMode = 'analysis', modelKey, onModel, routeView, paramTab, archiveView, renderArchive, methodItems = [], historyContent }) {
  const consoleRef = useRef(null), pluginsRef = useRef(null), panelRef = useRef(null);
  const sceneRef = useRef(null), stageNavRef = useRef(null);
  const context = useResearchContext(), object = useSpatialObject();
  const traceId = context.trace || context.context_trace;
  const [workspaceSearch,setWorkspaceSearch]=useState(null);
  const [replayReference,setReplayReference]=useState(null);
  const [pinnedAnalysis,setPinnedAnalysis]=useState(null);
  const [observeMode, setObserveMode] = useState(() => {
    const requested = new URLSearchParams(window.location.search).get('observeMode');
    if (WORKSPACE_MODES.some(([id]) => id === requested)) return requested;
    if (routeView === 'parameters') return 'features';
    return mode === 'live' && !traceId ? 'live' : 'archive';
  });
  const browsing = observeMode === 'features' || observeMode === 'archive' && workspaceMode === 'browse' && !traceId;
  const [example, setExample] = useState(null);
  const [exampleError, setExampleError] = useState('');
  const [exampleActive, setExampleActive] = useState(false);
  const [exampleSite, setExampleSite] = useState(()=>{
    const site=new URLSearchParams(window.location.search).get('observeExampleSite');
    return /^[0-3]$/.test(site)?Number(site):3;
  });
  const [archiveId,setArchiveId]=useState(()=>new URLSearchParams(window.location.search).get('observeArchive')||'');
  const [bundleRow,setBundleRow]=useState(()=>new URLSearchParams(window.location.search).get('observeArchiveRow')||'');
  const [fieldChannel,setFieldChannel]=useState(()=>Number(new URLSearchParams(window.location.search).get('observeChannel'))||0);
  const [liveCase,setLiveCase] = useState(()=>new URLSearchParams(window.location.search).get('liveCase')||'');
  const workspaceCatalog = useWorkspaceCatalog();
  const models = workspaceCatalog.catalog?.models || [];
  const activeModel = context.snapshot ? models.find(model=>model.model_snapshot===context.snapshot)
    : modelKey ? models.find(model=>model.model===modelKey||model.model_key===modelKey) : models[0];
  const snapshot = activeModel?.model_snapshot || context.snapshot || '';
  const archives = workspaceCatalog.archives;
  const headExample = archives.find(entry=>entry.presentation==='selection');
  const selection = useWorkspaceResource(headExample&&(archiveId===headExample.id||context.run===headExample.run||liveCase)?'selection-cases':null).data;
  const pathExample = workspaceCatalog.catalog?.resources.find(resource=>resource.kind==='path-example');
  const activeArchive = observeMode==='archive'&&!traceId ? archives.find(entry=>entry.id===archiveId) ||
    (!archiveId&&context.run===headExample?.run?headExample:null) : null;
  const bundleId=activeArchive?.snapshot===snapshot&&activeArchive.presentation!=='path-example' ? activeArchive.id : null;
  const [featureKind, setFeatureKind] = useState('unit');
  const [headCase, setHeadCase] = useState(() => new URLSearchParams(window.location.search).get('headCase') || '');
  const modePanels = useRef({});
  const [sceneInfo, setSceneInfo] = useState(null);
  const receiveScene = useCallback(next => setSceneInfo(previous => previous?.keys?.length === next.keys.length && previous.keys.every((v, i) => v === next.keys[i]) ? previous : next), []);
  const changeObject = next => { writeSpatialObject(next); setPop(null); };
  const layer = object.layer ?? 24;
  const setLayer = value => { const next = typeof value === 'function' ? value(layer) : value; changeObject({ ...object, layer: next, ...(next === 0 || layer === 0 ? { module: '', index: null, coordinate: null } : {}) }); };
  const archiveBundle = useObservation(bundleId?activeArchive:null,activeArchive?.presentation==='selection'?(headCase||bundleRow):bundleRow,layer-1);
  const sel = { unit: object.module === 'mlp' ? object.index : null, head: object.module === 'attn' ? object.index : null, coord: object.module === 'hs' ? object.index : object.module === 'embedding' ? object.coordinate : null };
  const live = useLiveBase();                                   // 实时生成
  const lm = useLiveModel(live.base);                           // 本机模型的加载 / 卸载
  const nat = useLocalBase('/api/traces/mainline/ma-h2/samples'); // 冻结样本的原生数据（只在本机可读）
  const base = nat.base;
  const [samples, setSamples] = useState(null);
  const [run, setRun] = useState({ i: 0, other: null });
  const [fpos, setFpos] = useState(0);           // 冻结样本里的读出位置
  const [livePos, setLivePos] = useState(0);     // 实时运行里的 token 序号
  const [archiveSource, setArchiveSource] = useState(() => ({ trace:traceId, kind:traceId?'trace':'frozen' }));
  const acceptsTrace=useCallback(data=>data?.trace_id===traceId,[traceId]);
  const traceService = useLocalBase(traceId?`/api/traces/${encodeURIComponent(traceId)}/manifest`:null,acceptsTrace);
  const activeReference=replayReference?.trace===traceId&&replayReference?.run===context.run&&
    replayReference?.snapshot===context.snapshot?replayReference:null;
  const recordTraceMismatch=Boolean(activeReference&&traceService.status&&!matchesReplayManifest(traceService.status,activeReference));
  const traceManifest = recordTraceMismatch?null:traceService.status;
  const [archivePosition, setArchivePosition] = useState({});
  const savedPosition = new URLSearchParams(window.location.search).get('tpos');
  const requestedPosition = archivePosition.id === traceId ? archivePosition.position : savedPosition===null?null:Number(savedPosition);
  const archivePos = Number.isInteger(requestedPosition) && requestedPosition >= 0 && requestedPosition < (traceManifest?.tokens?.length || 0) ? requestedPosition : traceManifest?.readout_positions?.at(-1);
  const archivedRun = useMemo(() => traceManifest ? {id:traceManifest.trace_id,tokens:traceManifest.tokens,readout:traceManifest.readout_positions} : null,[traceManifest]);
  const [gen, setGen] = useState({ busy: false, res: null, err: '' });
  const [listed, setListed] = useState(() => initialPlugins(mode));
  const [tab, setTab] = useState(() => {
    const preferred = new URLSearchParams(window.location.search).get('observePlugin');
    if (AVAILABLE.some(p=>p.id===preferred)) return preferred;
    return OPTIONAL_MODE_IDS.has(mode) ? mode : object.index != null ? 'object' : mode || 'object';
  });
  const [requestedConsoleTab,setConsoleTab]=useState(()=>{
    const preferred = new URLSearchParams(window.location.search).get('observeTab');
    if (MODE_TABS[observeMode].includes(preferred)) return preferred;
    if (OPTIONAL_MODE_IDS.has(mode)) return 'plugin';
    return observeMode==='archive'&&archiveView&&!traceId&&mode!=='response'?'research':MODE_TABS[observeMode][0];
  });
  const modeTabs=MODE_TABS[observeMode];
  const visibleConsoleTabs = workspaceSearch&&!modeTabs.includes('research')?[...modeTabs,'research']:modeTabs;
  const consoleTab = visibleConsoleTabs.includes(requestedConsoleTab) ? requestedConsoleTab : visibleConsoleTabs[0];
  const featureAnalysisActive=observeMode==='features'||(observeMode==='live'&&consoleTab==='object');
  const [comparisonView, setComparisonView] = useState('three');
  const [recordTab,setRecordTab]=useState('library'),[libraryVisited,setLibraryVisited]=useState(Boolean(archiveView));
  const [recordVisited,setRecordVisited]=useState(()=>new Set()),[methodId,setMethodId]=useState(methodItems[0]?.id||'');
  useEffect(()=>{panelRef.current?.scrollTo({top:0});},[consoleTab,recordTab]);
  const [priorArchiveView,setPriorArchiveView]=useState(archiveView);
  if(priorArchiveView!==archiveView){setPriorArchiveView(archiveView);if(archiveView&&observeMode==='archive'){setConsoleTab('research');setRecordTab('library');setLibraryVisited(true);}}
  const [archiveLayer, setArchiveLayer] = useState(null);
  const libraryVisible = consoleTab==='research'&&recordTab==='library';
  const [priorMode, setPriorMode] = useState(mode);
  if (priorMode !== mode) {
    setPriorMode(mode);
    if (mode) {
      setTab(current => OPTIONAL_MODE_IDS.has(mode) || current === mode ? mode : !MODE_IDS.has(current) && listed.has(current) ? current : object.index != null ? 'object' : mode);
      if (OPTIONAL_MODE_IDS.has(mode)) {
        setListed(s => new Set([...s, mode])); setConsoleTab('plugin');
      }
    }
  }
  const [view, setView] = useState('iso');
  const [max, setMax] = useState(() => new URLSearchParams(window.location.search).get('observeFullscreen') === '1');
  const [fieldView,setFieldView]=useState(()=>new URLSearchParams(window.location.search).get('observeField')==='1');
  const fieldOpen=max&&fieldView;
  const fieldButton=useRef(null);
  const [stageHeight, setStageHeight] = useState(0);
  useEffect(() => {
    if (!max || !stageNavRef.current) return;
    const nav = stageNavRef.current;
    const measure = () => setStageHeight(Math.ceil(nav.getBoundingClientRect().bottom - sceneRef.current.getBoundingClientRect().top));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(nav);
    for (const element of sceneRef.current.querySelectorAll('.ow-mode-bar,.ow-object-locator')) observer.observe(element);
    return () => observer.disconnect();
  }, [max]);
  const [pop, setPop] = useState(null);
  const [locator, setLocator] = useState(false);
  const [indexQuery, setIndexQuery] = useState('');
  const [indexError, setIndexError] = useState('');
  const [findKind, setFindKind] = useState('mlp');
  const [displayOpen, setDisplayOpen] = useState(false);
  const [colorValues, setColorValues] = useState(true);
  const [headColor,setHeadColor] = useState('probability');
  const [scaleLock,setScaleLock] = useState(null);
  const [featureMarks, setFeatureMarks] = useState(true);
  const [opt, setOpt] = useState({ hsNorm: 'global', hsClip: 0.99, mlpClip: 0.99, headTarget: 'follow', headMax: 0.5, diffAlign: 'named', showDiff: true, showDelta: false });
  const [hide, setHide] = useState(() => new Set());
  const [cfg, setCfg] = useState(false);
  const [float, setFloat] = useState(() => ({ x: Math.max(8,window.innerWidth-446), y: 132 }));
  const focusMode = useRef(observeMode);
  const runInFlight = useRef(false);
  const [text, setText] = useState('');
  const livePrompt = liveCase ? archivedHeadPrompt(selection,liveCase) : null;
  useEffect(()=>{if(livePrompt)setText(livePrompt);},[livePrompt]);
  const [inputMode,setInputMode]=useState('manual');
  const priorSelections = useRef(new Map());
  const modelScene = sceneInfo?.model?.model === activeModel?.model && sceneInfo?.snapshot === snapshot ? sceneInfo : null;
  const localObject = modelScene?.layer === layer && modelScene?.module === object.module && modelScene?.index === object.index ? modelScene : null;
  const selectModel = m => {
    if(snapshot)priorSelections.current.set(snapshot,object);
    const saved = priorSelections.current.get(m.model_snapshot);
    const lastLayer = m.n_layers || 0;
    writeSpatialObject(saved || (fieldOpen&&lastLayer?{layer:Math.min(Math.max(1,layer),lastLayer),module:'attn',index:0}:
      {layer:Math.min(layer,lastLayer),module:'',index:null}));
    setArchiveId('');setBundleRow('');setFieldChannel(0);
    setReplayReference(null);
    saveView({observeArchive:null,observeArchiveRow:null,observeChannel:null,tpos:null});
    setIndexQuery(''); setIndexError(''); setLocator(false); setDisplayOpen(false);
    publishResearchContext({ snapshot: m.model_snapshot || '', run: '', trace: '', context_trace: '', pmodel: '' });
    onModel?.(m.model); setPop(null);
  };
  const serviceStatus = lm.status || live.status;
  const localModelName = serviceStatus?.model_name || '';
  const localSnapshot = serviceStatus?.model_snapshot;
  const localModel = localSnapshot && models.find(m=>m.model_snapshot===localSnapshot);
  const modelMismatch = Boolean(snapshot && localSnapshot && snapshot!==localSnapshot);
  const canLoad = live.base!=null&&serviceStatus?.available&&serviceStatus.snapshot_guard===true&&activeModel?.capabilities?.nativeLive===true&&
    ['idle','error'].includes(serviceStatus.status);
  const runReason = !live.checked ? '正在查找本机采集服务。' : live.base == null ? '未连接本机采集服务。' :
    !serviceStatus?.available ? serviceStatus?.reason || '本机模型权重不可用。' :
    serviceStatus.snapshot_guard!==true ? '请更新本机采集服务，启用中心快照核验。' :
    !activeModel?.model_snapshot ? '请选择有快照的模型。' :
    activeModel.capabilities?.nativeLive!==true ? '中心尚未登记该快照的实时原生采集能力。' :
    modelMismatch ? `本机 ${localModelName||'模型'} 的权重快照与当前选择不同。` :
    serviceStatus.status==='loading' ? '本机模型正在加载。' : serviceStatus.status!=='ready' ? '请先加载本机模型。' :
    localSnapshot!==snapshot ? '本机采集服务尚未确认已加载的权重快照。' : '';

  useEffect(() => {
    if (base == null) return undefined;
    let dead = false;
    Promise.resolve(nat.status).then(d => { if (dead || !d?.items) return; setSamples(d);
      const it = d.items[0], twin = d.items.find(x => x.entity === it.entity && x.cell[0] !== it.cell[0] && x.cell.slice(1).join() === it.cell.slice(1).join());
      setRun({ i: 0, other: twin?.i ?? null }); setFpos(d.positions.indexOf('ans')); })
      .catch(e => { if (!dead) setSamples({ error: e.message }); });
    return () => { dead = true; };
  }, [base, nat.status]);
  const A = useSample(samples?.items ? base : null, samples?.items ? run.i : null);
  const B = useSample(samples?.items && run.other != null ? base : null, run.other);
  const saF = useMemo(() => sampler(A.data), [A.data]), sbF = useMemo(() => sampler(B.data), [B.data]);
  const matchingExample = example?.snapshot === snapshot && activeModel?.model_snapshot === snapshot ? example : null;
  const archivedUnit = observeMode === 'archive' && !traceId && activeArchive?.presentation==='path-example' &&
    snapshot === activeArchive.snapshot && layer === activeArchive.location.layer && sel.unit != null;
  useEffect(() => {
    if (example || !pathExample || (!exampleActive && !(featureAnalysisActive&&snapshot===pathExample.snapshot) && !archivedUnit && activeArchive?.presentation!=='path-example')) return;
    let active = true;
    loadGraphExample(pathExample?.id).then(data => { if (active) setExample(data); })
      .catch(error => { if (active) setExampleError(error.message); });
    return () => { active = false; };
  }, [example, pathExample, exampleActive, featureAnalysisActive, snapshot, archivedUnit, activeArchive]);
  const exampleReplay = observeMode === 'live' && exampleActive ? matchingExample : null;
  const replay = observeMode === 'archive' && Boolean(traceId) && (archiveSource.trace !== traceId || archiveSource.kind === 'trace');
  const liveResult = observeMode === 'live' && !exampleActive && !gen.busy && gen.input===text && gen.res?.snapshot === snapshot ? gen.res : null;
  const activeRun = replay ? archivedRun : liveResult;
  const isLive = replay || Boolean(liveResult);
  const tokenPosition = replay ? archivePos : livePos;
  const capturedBundle = isLive && !replay && activeRun?.bundle?.position === tokenPosition ? activeRun.bundle : null;
  const LB = useBundle(replay ? traceService.base : activeRun?.base, isLive && !capturedBundle ? activeRun?.id : null, isLive ? tokenPosition : null);
  const incomingBundle = capturedBundle || LB.data;
  const bundleData = incomingBundle?.trace_id === activeRun?.id && incomingBundle?.model_snapshot === snapshot &&
    incomingBundle?.position === tokenPosition ? incomingBundle : null;
  const saL = useMemo(() => liveSampler(bundleData, activeRun), [bundleData, activeRun]);
  const runData = isLive ? saL : observeMode==='archive' ? saF : null;
  const sa = !browsing && activeModel && layer >= 0 && layer <= activeModel.n_layers && runData?.d.model_snapshot === snapshot && runData.NL === activeModel.n_layers + 1 && runData.DM === activeModel.d_model ? runData : null;
  const compatible = sa && sbF && sa.d.model_snapshot === sbF.d.model_snapshot && ['NL', 'DM', 'NF', 'NH', 'P'].every(key => sa[key] === sbF[key]);
  const sb = !isLive && compatible ? sbF : null;
  const native = Boolean(sa);
  const reason = recordTraceMismatch ? '逐题引用与轨迹清单的模型、运行、样本或位置不一致。' : browsing ? '在实时模式或测试记录中可查看对应输入的响应数据。' : runData && runData.d.model_snapshot !== snapshot ? '当前运行与所选模型快照不匹配。' : isLive ? (LB.error || (replay && traceService.checked && traceService.base == null ? '未找到此轨迹。请连接原采集节点，或手动选择冻结样本。' : !sa ? '读取运行原生数据…' : ''))
    : observeMode==='live' ? '运行一段文本或载入随包样例后可查看读数。'
    : !nat.checked ? '正在查找本机数据…' : samples?.error || A.error || (base == null ? (DESKTOP ? '没有找到提供冻结样本的本机后端（:5001 完整中心或 :5002 实时前向服务）：请确认本机服务已启动，原生全场在 AI2050_NATIVE_DIR 下' : '网页版不公开原生全场：原生数据保存在采集节点，桌面客户端或本机后端可用') : !sa ? '读取冻结样本…' : '');
  const positions = useMemo(()=>isLive ? activeRun?.tokens || [] : samples?.positions || ['ans'],
    [isLive,activeRun?.tokens,samples?.positions]);
  const pos = isLive ? tokenPosition : fpos;
  const setPos = replay ? position => { setArchivePosition({ id: traceId, position }); const url = new URL(window.location.href); url.searchParams.set('tpos', position); window.history.replaceState(window.history.state, '', url); window.dispatchEvent(new Event('research-context')); } : isLive ? setLivePos : setFpos;
  const ANS = isLive ? positions.length - 1 : positions.indexOf('ans');
  const mlpOk = isLive ? Boolean(sa?.hasMlp) : pos === ANS;     // 实时运行只在模块读出位置（首 / 中 / 末）采了 MLP
  const tokText = t => String(t ?? '').replace(/\n/g, '⏎') || '␣';
  const posName = i => (isLive ? `token ${i}「${tokText(positions[i])}」` : (POSN[positions[i]] || positions[i] || ''));
  const NL = (activeModel?.n_layers || 0) + 1, NB = NL - 1, NH = activeModel?.n_heads || modelScene?.dims.heads || 0, DM = activeModel?.d_model || 0, NF = activeModel?.d_ff || modelScene?.dims.units || 0;
  const dimensionsReady = [NB, NH, DM, NF].every(positiveDimension);
  const [hw, hh] = grid(DM), [mw, mh] = grid(NF);
  const aligned = sa && sb && sa.T === sb.T && sa.d.tokpos.join() === sb.d.tokpos.join();
  const threeCueActive = observeMode === 'archive' && mode === 'response' && tab === 'response' && comparisonView === 'three';
  const diffOn = mode === 'response' && !threeCueActive && opt.showDiff && sb && (aligned || opt.diffAlign === 'named');
  const shown = id => !hide.has(id);
  const headTarget = opt.headTarget === 'follow' || isLive ? pos : positions.indexOf(opt.headTarget);
  const scope = `${workspaceMode}:${snapshot}:${base}:${isLive ? activeRun?.id : run.i}:${sa?.d.model_snapshot || ''}:${pos}`;
  const visiblePop = pop?.scope === scope && pop.h.l === layer ? pop : null;

  /* 纹理 */
  const TX = useMemo(() => {
    const out = {};
    if (!dimensionsReady || !colorValues) return out;
    if (exampleReplay) {
      const field = exampleReplay.field('hs', layer, exampleSite);
      if (field && shown('hs')) {
        const scale = Math.max(...exampleReplay.positions.map((_, index) => absMax(exampleReplay.field('hs', layer, index))));
        out.hs = {[layer]: heatTexture(hw,hh,DM,field,scale,divc)};
      }
      return out;
    }
    if (browsing) {
      const field = localObject?.pointValues, colors = localObject?.typeColors;
      const kind = object.module === 'embedding' || object.module === 'hs' ? 'hs' : object.module === 'attn' ? 'heads' : 'mlp';
      const [w, h] = kind === 'hs' ? [hw, hh] : kind === 'heads' ? [8, Math.ceil(NH / 8)] : [mw, mh];
      if (field?.length || colors?.length) { const max = field ? absMax(field) : 1; out[kind] = []; out[kind][kind === 'hs' ? layer : layer - 1] = tex(w, h, d => {
        for (let i = 0; i < w * h; i++) { const hex = colors?.[i]; put(d, i, field && i < field.length ? divc(field[i], max) : hex ? [1, 3, 5].map(start => parseInt(hex.slice(start, start + 2), 16)) : [214, 222, 232]); }
      }); }
      return out;
    }
    if (shown('hs') && sa) { const v = l => { const a = sa.hs(pos, l); if (!diffOn) return a; const b = sb.hs(pos, l); return a.map((x, i) => x - b[i]); };
      const vals = Array.from({ length: NL }, (_, l) => v(l)), sc = pct(vals.slice(0, NL - 1), opt.hsClip);
      out.hs = vals.map(x => { const m = opt.hsNorm === 'layer' ? absMax(x) : sc; return tex(hw, hh, d => { for (let i = 0; i < hw * hh; i++) put(d, i, i < DM ? divc(x[i], m) : [246, 248, 251]); }); }); }
    return out;
  }, [sa, sb, pos, hide, opt, diffOn, browsing, localObject, object.module, layer, NH, DM, dimensionsReady, colorValues, exampleReplay, exampleSite]);   // eslint-disable-line react-hooks/exhaustive-deps

  const objs = [
    { id: 'llm', name: `${activeModel ? shortModel(activeModel.model) : '读取模型'} · ${NB} 层` },
    ...(shown('heads') && positiveDimension(NH) ? [{ id: 'heads', name: `Head · ${NH}`, grid: [8, Math.ceil(NH / 8)], count: NH }] : []),
    ...(shown('hs') && positiveDimension(DM) ? [{ id: 'hs', name: `${browsing && layer === 0 ? '词嵌入' : 'HiddenState'} · ${DM}${browsing && object.module === 'embedding' && localObject?.pointValues ? '（词表参数）' : sa || exampleReplay ? '' : '（结构）'}`, grid: [hw, hh], count: DM }] : []),
    ...(shown('mlp') && positiveDimension(NF) ? [{ id: 'mlp', name: `MLP · ${NF}${browsing && object.module === 'mlp' && localObject?.typeColors ? '（普查类型）' : ''}`, grid: [mw, mh], count: NF }] : []),
  ];

  const describe = h => {
    if (h.o === 'llm') return `第 ${h.l} 层 · ${h.l ? h.part : '词嵌入'}`;
    if (exampleReplay && ['hs','mlp'].includes(h.o)) {
      const field=exampleReplay.field(h.o,h.l,exampleSite);
      return `${h.o==='mlp'? 'U':'坐标 '}${h.idx} · ${field ? field[h.idx].toFixed(4) : '该层未采集'}`;
    }
    if (browsing) {
      const name = h.o === 'hs' ? layer === 0 ? `词嵌入 Token #${object.index ?? '未选择'} · 坐标 ${h.idx}` : `HiddenState L${h.l} · 坐标 ${h.idx}` : h.o === 'mlp' ? `MLP B${h.l - 1} · U${h.idx}` : `Head B${h.l - 1} · H${h.idx}`;
      const matching = (h.o === 'hs' ? ['embedding', 'hs'].includes(object.module) : h.o === 'mlp' ? object.module === 'mlp' : object.module === 'attn');
      const value = matching ? localObject?.pointValues?.[h.idx] : null;
      return `${name} · ${Number.isFinite(value) ? value.toPrecision(6) : '结构索引；点击查看档案'}`;
    }
    if (h.o === 'hs') return `HiddenState · 第 ${h.l} 层 · 坐标 ${h.idx} · ${sa && h.idx < DM ? sa.hs(pos, h.l)[h.idx].toFixed(4) : '未采集'}`;
    if (h.o === 'mlp') return `MLP · 块 ${h.l - 1} · 单元 ${h.idx}${sa && mlpOk && h.idx < NF ? ' · A ' + sa.mlp(h.l - 1)[h.idx].toFixed(4) : ' · 未采集'}`;
    if (h.o === 'heads') return h.idx < NH ? `注意力 · L${h.l - 1}H${h.idx} · ${sa ? sa.att(h.l - 1, h.idx)[sa.d.tokpos[headTarget]].toFixed(4) : '未采集'}` : '';
    return '';
  };
  const onPick = (h, x, y) => {
    if (!h) { setPop(null); return; }
    if (h.o==='headChannels') { changeChannel(h.idx); setPop(null); return; }
    if (h.o === 'llm') { changeObject({ layer: h.l, module: h.l === 0 && browsing ? 'embedding' : '', index: null }); return; }
    if (h.o === 'hs' && browsing && h.l === 0) changeObject({ layer: 0, module: 'embedding', index: object.module === 'embedding' ? object.index ?? 0 : 0, coordinate: h.idx });
    else changeObject({ layer: h.l, module: h.o === 'heads' ? 'attn' : h.o, index: h.idx });
    if (featureAnalysisActive) {
      if(h.o==='mlp')setFeatureKind('unit');
      setTab('object'); setConsoleTab('object');
    }
    if (atlas.run && h.o==='mlp') { setPop(null); return; }
    if (observeMode === 'live' || consoleTab==='plugin' && ANALYSIS_TOOL_IDS.has(tab)) return;
    setPop({ h, x, y, scope });
  };
  const selected = object.index != null ? object : null;
  const chooseFrozenSample = i => { const it=samples?.items.find(x=>x.i===i); if(!it)return;
    const twin=samples.items.find(x=>x.entity===it.entity&&x.cell[0]!==it.cell[0]&&x.cell.slice(1).join()===it.cell.slice(1).join());
    setRun({i,other:twin?.i??null});setArchiveSource({trace:traceId,kind:'frozen'});
  };
  const validObject = selected && layer >= 0 && layer < NL && selected.index >= 0 && (selected.module === 'hs' ? selected.index < DM : selected.module === 'embedding' ? layer === 0 : layer > 0 && selected.index < (selected.module === 'mlp' ? NF : NH));
  const rawValues = !sa || !validObject ? null : selected.module === 'hs' ? sa.hs(pos, layer) : selected.module === 'mlp' && mlpOk ? sa.mlp(layer - 1) : selected.module === 'attn' ? sa.att(layer - 1, selected.index) : null;
  const sceneLinks = localObject?.links || [];
  const locate = location => {
    const o = location.module === 'mlp' ? 'mlp' : location.module === 'attn' ? 'heads' : 'hs';
    setHide(s => { const next = new Set(s); next.delete(o); return next; });
    changeObject(location);
    if(observeMode==='features'){setTab('object');setConsoleTab('object');}
  };
  const locateArchive = entry => {
    const target = models.find(m=>m.model_snapshot===entry.snapshot);
    if(!target)return '中心尚无此快照的模型结构；请同步中心目录。';
    publishResearchContext({snapshot:entry.snapshot,run:entry.run||'',trace:entry.trace||'',context_trace:'',pmodel:''});
    onModel?.(target.model); locate(entry.location);
    document.querySelector('.aw-space')?.scrollIntoView({block:'start',behavior:'smooth'});
    return '';
  };
  const showObjectDetails = () => {
    if(observeMode!=='live')chooseObserveMode('features');
    setPop(null); setTab('object'); setConsoleTab('object');
    saveView({observeTab:'object'});
    if(!max)requestAnimationFrame(()=>consoleRef.current?.scrollIntoView({block:'start',behavior:'smooth'}));
  };
  const locateLibraryEntry = entry => {
    const error = locateArchive(entry);
    if (error) return error;
    if (entry.trace) {
      setArchiveId('');setArchivePosition({});saveView({observeArchive:null,observeArchiveRow:null,tpos:null});
      chooseObserveMode('archive');setArchiveSource({trace:entry.trace,kind:'trace'});
      setTab('live');setConsoleTab('run');saveView({observeTab:'run'});
      if(['attn','mlp'].includes(entry.location?.module))changeFieldView(true,entry.location);
    }
    else showObjectDetails();
    return '';
  };
  const objectEvidence = <button type="button" className="v6-linkbtn" onClick={showObjectDetails}>查看参数、特征、证据与历史 ↓</button>;
  const openPlugin = id => {
    const target = id === 'delta' ? 'live' : id === 'diff' ? 'response' : id;
    const functional = WORKSPACE_PLUGINS.find(p => p.id === target);
    if (functional) {
      setPop(null); setTab(target === 'structure' ? 'object' : target);
      setConsoleTab(functional.panel); onMode(target); return;
    }
    if (!['live', 'response', 'result'].includes(target) && !AVAILABLE.some(p => p.id === target)) return;
    const next = ['live','response','result'].includes(target)?'run':'plugin';
    setTab(target); selectConsoleTab(next,target);
    if (next==='plugin') saveView({observePlugin:target});
    if (MODE_IDS.has(target)) onMode(target);
  };
  const listPlugin = (id, checked) => {
    if (!AVAILABLE.some(p => p.id === id)) return;
    const next=new Set(listed);
    if(checked)next.add(id);else next.delete(id);
    resetPlugins([...next]);
  };
  const resetPlugins = ids => {
    const next = new Set(ids.filter(id => AVAILABLE.some(p => p.id === id)));
    setListed(next);
    try{localStorage.setItem(PLUGIN_PREFERENCES,JSON.stringify([...next]));}catch{ /* Keep session preferences. */ }
    if(!next.has(new URLSearchParams(window.location.search).get('observePlugin')))saveView({observePlugin:null});
    if (!next.has(tab) && !['object', 'live', 'response', 'result'].includes(tab)) setTab('object');
  };
  const selectConsoleTab = (value, selectedTool) => {
    if (!visibleConsoleTabs.includes(value)) return;
    setConsoleTab(value); setPop(null); saveView({observeTab:value});
    if (value==='plugin' && !selectedTool && !listed.has(tab)) {
      const matching = object.module==='attn'?'heads':object.module==='mlp'?'mlp':'hs';
      if(listed.has(matching)){setTab(matching);saveView({observePlugin:matching});}
    }
    if(value==='research'&&recordTab==='library')setLibraryVisited(true);
  };
  const selectRecordTab = value => {setRecordTab(value);if(value==='library')setLibraryVisited(true);else setRecordVisited(s=>new Set([...s,value]));};
  const saveView = values => {
    const url = new URL(window.location.href);
    for (const [key,value] of Object.entries(values)) {
      if(value == null) url.searchParams.delete(key);
      else url.searchParams.set(key,value);
    }
    window.history.replaceState(window.history.state,'',url);
  };
  const changeFullscreen = value => {
    setMax(value); setFieldView(false); setPop(null); setDisplayOpen(false); setLocator(false);
    saveView({observeFullscreen:value?'1':null,observeField:null});
    if(!value&&fieldOpen)fieldButton.current?.focus({preventScroll:true});
  };
  const changeFieldView = (value,location) => {
    if(value&&!location&&(!['attn','mlp'].includes(object.module)||!validObject)&&dimensionsReady){
      changeObject({layer:Math.min(NB,Math.max(1,layer)),module:'attn',index:0});
    }
    setMax(true);setFieldView(value);setDisplayOpen(false);setLocator(false);setPop(null);
    saveView({observeFullscreen:'1',observeField:value?'1':null});
  };
  const chooseObserveMode = value => {
    if (observeMode === value) return;
    modePanels.current[observeMode] = { consoleTab,tab };
    const previous = modePanels.current[value];
    focusMode.current = value;
    const nextConsole = previous?.consoleTab || MODE_TABS[value][0];
    const nextTool = previous?.tab || (value==='live'?'live':value==='archive'&&mode==='response'?'response':'object');
    setObserveMode(value); setConsoleTab(nextConsole); setTab(nextTool);
    if (!previous && value==='archive') setLibraryVisited(true);
    setPop(null); setLocator(false); setDisplayOpen(false); setCfg(false);
    saveView({observeMode:value,observeTab:nextConsole,observePlugin:nextTool});
  };
  const chooseDisplay = id => setHide(current => {
    const next=new Set(current);
    if(next.has(id))next.delete(id);
    else next.add(id);
    return next;
  });
  const moveFloatingPanel = event => {
    if(!max || event.button!==0)return;
    const handle=event.currentTarget, bounds=consoleRef.current.getBoundingClientRect();
    const dx=event.clientX-bounds.left,dy=event.clientY-bounds.top;
    handle.setPointerCapture(event.pointerId);
    const move=pointer=>setFloat({x:pointer.clientX-dx,y:pointer.clientY-dy});
    handle.addEventListener('pointermove',move);
    handle.addEventListener('lostpointercapture',()=>handle.removeEventListener('pointermove',move),{once:true});
  };
  useEffect(() => {
    const onEscape=event=>{
      if(event.key!=='Escape'||document.querySelector('dialog[open]'))return;
      if(locator)setLocator(false);
      else if(displayOpen)setDisplayOpen(false);
      else if(max)changeFullscreen(false);
    };
    window.addEventListener('keydown',onEscape);
    return ()=>window.removeEventListener('keydown',onEscape);
  },[locator,displayOpen,max,fieldOpen]);

  const locateExampleNeuron = (unit, data=matchingExample) => {
    if(!data)return;
    setHide(current=>{const next=new Set(current);next.delete('mlp');return next;});
    changeObject({layer:data.block+1,module:'mlp',index:unit ?? data.units[0]});
  };
  async function showExample() {
    setExampleError('');
    try {
      const data=await loadGraphExample(pathExample?.id);
      const target=models.find(model=>model.model_snapshot===data.snapshot);
      if(!target)throw Error('当前中心缺少此样例的模型结构。');
      if(focusMode.current!=='live')return;
      if(snapshot!==data.snapshot)selectModel(target);
      setExample(data); setExampleActive(true); setConsoleTab('run'); setTab('live');
      locateExampleNeuron(undefined,data);
    } catch(error) { setExampleError(error.message); }
  }

  const showRun = (r, input, condition) => {
    if(focusMode.current==='live'){setText(input);setConsoleTab('run');setTab('result');}
    setGen({ busy: false, res: r, input, condition, err: '' });
    setExampleActive(false);setLivePos(r.bundle.position);
  };
  const observeTask = (r, input, condition, targetLayer) => {
    showRun(r, input, condition);
    if (targetLayer != null) setLayer(targetLayer);
    const scene = sceneRef.current;
    scene?.focus({ preventScroll: true });
    scene?.scrollIntoView({ block: 'start' });
  };
  async function executeText(input, condition, promptFormat='chat') {
    if (runReason) throw Error(runReason);
    if (runInFlight.current) throw Error('正在采集，请等待本次运行完成。');
    runInFlight.current=true;
    setText(input);
    setConsoleTab('run');
    setGen({busy:true,res:null,err:''});
    try {
      const r = await runOnce(live.base, input, snapshot, promptFormat);
      showRun(r, input, condition);
      if(focusMode.current==='live'&&promptFormat==='chat')setLayer(r.hidden.length - 1);
      return r;
    } catch (e) {
      setGen({busy:false,res:null,err:condition?'':e.message});
      throw e;
    } finally {
      runInFlight.current=false;
    }
  }
  const generate = () => executeText(text,null,liveCase?'rendered':'chat').catch(()=>{});
  const releaseLiveCase = () => {setLiveCase('');setText('');saveView({liveCase:null});};
  const editInput = value => {setExampleActive(false);setText(value);};

  /* 叠加：层间变化（LLM 两侧的条）与逐层读出（右缘） */
  const overlays = useCallback(({ ctx, proj, LY, llm, alpha }) => {
    if (browsing || !colorValues) return;
    if (opt.showDelta && sa) { const v = []; for (let l = 1; l < NL - 1; l++) v.push(norm(sa.hs(pos, l).map((x, i) => x - sa.hs(pos, l - 1)[i])));
      const mx = Math.max(...v) || 1; v.forEach((x, k) => { const l = k + 1, p = proj(llm.x - llm.w / 2, LY(l), llm.d / 2); ctx.globalAlpha = alpha(l); ctx.fillStyle = '#7050a0'; ctx.fillRect(p[0] - 40 - 60 * x / mx, p[1] - 2, 60 * x / mx, 3); ctx.globalAlpha = 1; }); }
    if (opt.showLens && sa?.d.lens && !isLive) { const L = [sa.d.lens.map(r => r[0] - r[1]), sb ? sb.d.lens.map(r => r[0] - r[1]) : null];
      for (let l = 0; l < NL; l++) { const p = proj(llm.x + llm.w / 2, LY(l), -llm.d / 2); ctx.globalAlpha = alpha(l);
        ctx.fillStyle = '#2a78d6'; ctx.fillRect(p[0] + 8, p[1] - 4, Math.max(-60, Math.min(60, (L[0][l] || 0) * 4)), 3);
        if (L[1]) { ctx.fillStyle = '#d4622b'; ctx.fillRect(p[0] + 8, p[1], Math.max(-60, Math.min(60, (L[1][l] || 0) * 4)), 3); } ctx.globalAlpha = 1; } }
  }, [sa, sb, pos, opt.showDelta, opt.showLens, NL, browsing, colorValues, isLive]);   // eslint-disable-line react-hooks/exhaustive-deps

  const marks = validObject ? [{ id: selected.module === 'attn' ? 'heads' : selected.module === 'embedding' ? 'hs' : selected.module, idx: selected.module === 'embedding' ? selected.coordinate : selected.index }].filter(m => m.idx != null) : [];
  if(featureAnalysisActive && featureMarks && featureKind==='group' && matchingExample && layer===matchingExample.block+1)
    marks.push(...matchingExample.units.map(idx=>({id:'mlp',idx,feature:true})));
  const k = sa?.d, Bk = layer - 1;
  const atlas = useMlpAtlas({active:observeMode==='features',snapshot,block:Bk,width:NF,unit:sel.unit});
  const [atlasLock,setAtlasLock] = useState(null);
  const atlasScaleKey = [snapshot,atlas.isPole?'position':atlas.metric].join(':');
  const atlasLocked = atlasLock?.key===atlasScaleKey;
  const atlasScale = useMemo(()=>atlasLocked?atlasLock.scale:atlas.values?absMax(atlas.values):null,
    [atlasLocked,atlasLock,atlas.values]);
  const atlasControls = atlas.active && <MlpAtlasControls atlas={atlas} block={Bk} unit={sel.unit} scale={atlasScale}
    locked={atlasLocked} onLock={atlas.values?()=>setAtlasLock(atlasLocked?null:{key:atlasScaleKey,scale:atlasScale}):null}
    onBlock={block=>locate({layer:block+1,module:'mlp',index:sel.unit??0})}/>;
  const atlasReport = atlas.active && <MlpAtlasReport atlas={atlas} block={Bk} unit={sel.unit}
    onShowPole={pole=>{
      atlas.selectPole(pole);
      if(fieldOpen)changeFieldView(false);
      requestAnimationFrame(()=>sceneRef.current?.scrollIntoView({block:'start'}));
    }}/>;

  /* ---------- 操作面板（非全屏停靠在控制台，全屏时浮在 3D 上：同一个组件） ---------- */
  const Seg = ({ o, items }) => <span className="aw-seg">{items.map(([v, n]) => <button type="button" key={v} aria-pressed={String(opt[o]) === v} onClick={() => setOpt(x => ({ ...x, [o]: v }))}>{n}</button>)}</span>;
  const Sel = ({ o, items, num }) => <select value={String(opt[o])} onChange={e => setOpt(x => ({ ...x, [o]: num ? +e.target.value : e.target.value }))}>{items.map(([v, n]) => <option key={v} value={v}>{n}</option>)}</select>;
  const CLIP = [['0.95', '95% 分位'], ['0.99', '99% 分位'], ['0.999', '99.9% 分位'], ['1', '最大值']];
  const OPTS = {
    hs: () => <><div className="aw-r"><span>着色口径</span><Seg o="hsNorm" items={[['global', '全图'], ['layer', '逐层']]} /></div><div className="aw-r"><span>色标截断</span><Sel o="hsClip" items={CLIP} num /></div><div className="aw-r"><span>排列</span><span className="v8-note">原生编号 · {hw} × {hh}</span></div></>,
    mlp: () => <><div className="aw-r"><span>色标截断</span><Sel o="mlpClip" items={CLIP} num /></div>{!mlpOk && <p className="v8-note">{isLive ? '实时运行只在模块读出位置（首 / 中 / 末 token）采集 MLP，这个 token 显示为灰。' : '当前 token 不是答案位置：MA-H2 只在答案位置存 MLP，模型显示为灰。'}</p>}</>,
    heads: () => <><div className="aw-r"><span>看向</span><Sel o="headTarget" items={[['follow', '跟随所选 token'], ...positions.map(p => [p, POSN[p] || p])]} /></div><div className="aw-r"><span>色标上限</span><Sel o="headMax" items={[['0.25', '0.25'], ['0.5', '0.5'], ['1', '1.0']]} num /></div></>,
    diff: () => <><div className="aw-r"><span>对照运行</span><span>{run.other != null ? `#${run.other}` : '—'}</span></div><div className="aw-r"><span>对齐</span><Seg o="diffAlign" items={[['strict', '严格'], ['named', '命名位置']]} /></div>
      {sb && !aligned && <p className="v8-note">两句 token 数不同（{sa?.T} / {sb.T}）：{opt.diffAlign === 'strict' ? '严格对齐时不显示差值。' : '按命名读出位置比较，差值含前缀差异。'}</p>}</>,
  };
  const ops = <div ref={pluginsRef} className="ow-plugins">
    <div className="ow-tool-row" role="group" aria-label="分析插件">
      <span>分析插件</span>
      {[...listed].map(id=>{
        const plugin=AVAILABLE.find(item=>item.id===id);
        return plugin&&<button className="fw-tbtn" key={id} type="button" aria-pressed={consoleTab==='plugin'&&tab===id}
          onClick={()=>openPlugin(id)}>{id==='heads'?'Head':id==='mlp'?'MLP':plugin.name}</button>;
      })}
      <button className="fw-tbtn ow-plugin-manager" type="button" aria-expanded={cfg} onClick={()=>setCfg(value=>!value)}>管理插件</button>
    </div>
    {cfg && <div className="ow-plugin-options" role="group" aria-label="插件管理">
      {AVAILABLE.map(plugin=><label key={plugin.id} title={plugin.desc}>
        <input type="checkbox" checked={listed.has(plugin.id)} onChange={event=>listPlugin(plugin.id,event.target.checked)}/>{plugin.name}
      </label>)}
      <button className="fw-tbtn" onClick={()=>resetPlugins(DEFAULT_PLUGINS)}>恢复默认</button>
      <ResearchDisclosure title={'待接入工具（'+PENDING.length+'）'}><ul>{PENDING.map(plugin=><li key={plugin.id}>{plugin.name}</li>)}</ul></ResearchDisclosure>
    </div>}
  </div>;

  /* ---------- 插件面板 ---------- */
  const line = (series, mark, dp = 1) => { const W = 420, H = 130, all = series.flatMap(s => s.v.filter(Number.isFinite)); let lo = Math.min(0, ...all), hi = Math.max(...all); if (hi === lo) hi = lo + 1;
    const n = Math.max(...series.map(s => s.v.length)), X = i => 30 + (W - 38) * i / (n - 1), Y = v => 6 + (H - 22) * (1 - (v - lo) / (hi - lo));
    return <><svg viewBox={`0 0 ${W} ${H}`} className="aw-line" role="img" aria-label={series.map(s => s.name).join('、')}><line x1="30" x2={W - 8} y1={Y(0)} y2={Y(0)} stroke="#e2e7ef" />
      {mark != null && <line x1={X(mark)} x2={X(mark)} y1="4" y2={H - 16} stroke="#d29a00" strokeDasharray="3 3" />}
      {series.map(s => <path key={s.name} d={s.v.map((v, i) => Number.isFinite(v) ? `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}` : '').join('')} fill="none" stroke={s.color} strokeWidth="1.7" strokeDasharray={s.dash ? '4 3' : undefined} />)}
      <text x="0" y={Y(hi) + 4} fontSize="10" fill="#68748a">{hi.toFixed(dp)}</text><text x="0" y={Y(lo)} fontSize="10" fill="#68748a">{lo.toFixed(dp)}</text></svg>
      <div className="v8-note">{series.map(s => <span key={s.name} className="aw-key"><i style={{ background: s.color }} />{s.name}</span>)}</div></>; };
  const nameK = k ? `${CN[k.K] || k.K}` : '资料色', nameP = k ? `${CN[k.P] || k.P}` : '常识色';
  const needNative = <div><p className="v8-note">{reason}</p>{browsing&&<button type="button" className="fw-tbtn" onClick={()=>chooseObserveMode('live')}>打开实时模式</button>}</div>;
  const P = {
    hs: () => !sa ? [needNative] : (() => { const v = sa.hs(pos, layer), top = [...v].map((x, i) => [i, x]).sort((p, q) => Math.abs(q[1]) - Math.abs(p[1])).slice(0, 8);
      return [<><h4>第 {layer} 层 · {posName(pos)} · ‖h‖ {norm(v).toFixed(1)}</h4><table className="v6-table"><tbody>{top.map(([i, x]) => <tr key={i} className={sel.coord === i ? 'on' : ''} onClick={() => onPick({ o: 'hs', l: layer, idx: i }, 20, 100)}><td>坐标 {i}</td><td className="num">{x.toFixed(4)}</td></tr>)}</tbody></table></>,
        sel.coord != null ? <><h4>坐标 {sel.coord} 跨层</h4>{line([{ v: Array.from({ length: NL }, (_, l) => sa.hs(pos, l)[sel.coord]), color: '#2a78d6', name: '本运行' }, ...(sb ? [{ v: Array.from({ length: NL }, (_, l) => sb.hs(pos, l)[sel.coord]), color: '#d4622b', name: '对照运行', dash: true }] : [])], layer, 2)}</>
          : <><h4>每层 ‖h‖</h4>{line([{ v: Array.from({ length: NL }, (_, l) => norm(sa.hs(pos, l))), color: '#2a78d6', name: '本运行' }], layer)}<p className="v8-note">点旁边的 HiddenState 网格选择原生坐标。</p></>,
        '坐标按原生编号排成网格，网格位置没有含义；颜色按全图 |值| 的分位截断，悬停或点击看原值（float16）。']; })(),
    mlp: () => !sa ? [needNative] : !mlpOk ? [<p key="m" className="v8-note">{isLive ? '实时运行只在模块读出位置（首 / 中 / 末 token）采集 MLP；在 token 条里选这几个位置。' : 'MA-H2 只在答案位置存了 MLP 激活；把 token 切到答案位置。'}</p>] : layer === 0 ? [<p key="m" className="v8-note">词嵌入层没有 MLP。</p>] : (() => { const a = sa.mlp(Bk), top = [...a.keys()].sort((x, y) => Math.abs(a[y]) - Math.abs(a[x])).slice(0, 8); if (sel.unit != null && !top.includes(sel.unit)) top.push(sel.unit);
      return [<><h4>块 {Bk} · 最活跃的单元（A = silu(G)⊙V）</h4><table className="v6-table"><thead><tr><th>单元</th><th className="num">本运行</th><th className="num">对照</th></tr></thead><tbody>{top.map(u => <tr key={u} className={sel.unit === u ? 'on' : ''} onClick={() => onPick({ o: 'mlp', l: layer, idx: u }, 20, 100)}><td>U{u}</td><td className="num">{a[u].toFixed(3)}</td><td className="num">{sb ? sb.mlp(Bk)[u].toFixed(3) : '—'}</td></tr>)}</tbody></table></>,
        (selected?.module === 'mlp' && objectEvidence) || <p className="v8-note">点单元查看真实激活、已有研究特征和前后层关联。</p>, '单元按原生编号排成网格，网格位置没有含义；激活大小不说明它被使用。']; })(),
    heads: () => layer === 0 ? [<p key="h" className="v8-note">词嵌入层没有注意力头。</p>] : !sa ? [needNative]
      : (() => { const tp = sa.d.tokpos[headTarget], rank = [...Array(NH).keys()].sort((x, y) => sa.att(Bk, y)[tp] - sa.att(Bk, x)[tp]).slice(0, 8);
        return [<><h4>块 {Bk} · 采集查询位置看“{posName(headTarget)}”最多的头</h4><table className="v6-table"><thead><tr><th>头</th><th className="num">本运行</th><th className="num">对照</th></tr></thead><tbody>{rank.map(h => <tr key={h} className={sel.head === h ? 'on' : ''} onClick={() => onPick({ o: 'heads', l: layer, idx: h }, 20, 100)}><td>L{Bk}H{h}</td><td className="num">{sa.att(Bk, h)[tp].toFixed(3)}</td><td className="num">{sb ? sb.att(Bk, h)[sb.d.tokpos[headTarget]].toFixed(3) : '—'}</td></tr>)}</tbody></table></>,
          sel.head != null ? <><h4>L{Bk}H{sel.head} 对全部 {sa.T} 个 token</h4>{line([{ v: Array.from(sa.att(Bk, sel.head)), color: '#b07d00', name: '虚线 = 所选位置' }], tp, 2)}</> : <p className="v8-note">点旁边的 Head 网格或表中一行。</p>, '注意力去向不等于计算使用。']; })(),
    delta: () => !sa ? [needNative] : (() => { const dn = [], cs = []; for (let l = 1; l < NL; l++) { const a = sa.hs(pos, l), b = sa.hs(pos, l - 1); dn.push(l === NL - 1 ? NaN : norm(a.map((x, i) => x - b[i]))); cs.push(l === NL - 1 ? NaN : cosv(a, b)); }
      const split = sa.hasSplit ? [{ v: Array.from({ length: NB }, (_, b) => norm(sa.ao(b))), color: '#7050a0', name: '注意力写入', dash: true }, { v: Array.from({ length: NB }, (_, b) => norm(sa.mo(b))), color: '#2f8f5b', name: 'MLP 写入' }] : [];
      return [<><h4>每层总变化 ‖Δh‖{sa.hasSplit ? ' 与注意力 / MLP 写入（本次实时运行，真实采集）' : opt.showDelta ? '（3D：LLM 左侧紫条）' : ''}</h4>{line([{ v: dn, color: '#183f77', name: '‖h_l − h_(l−1)‖' }, ...split], layer - 1)}</>, <><h4>相邻层方向 cos(h_l, h_(l−1))</h4>{line([{ v: cs, color: '#183f77', name: '越接近 1 改写越少' }], layer - 1, 2)}</>,
        sa.hasSplit ? '注意力写入与 MLP 写入是 o_proj 与 down_proj 的真实输出（本次运行在模块读出位置采集）；最后一层的 hs 已过最终归一化。' : isLive ? '这个 token 不是模块读出位置，没有写入拆分；在 token 条里选首 / 中 / 末位置。' : '最后一层的 hs 已过最终归一化，不与前层比较。冻结样本的注意力 / MLP 写入拆分需要原生权重（待接入）。']; })(),
    diff: () => isLive ? [<p key="d" className="v8-note">实时运行没有对照运行：资料 / 现实成对的比较只在冻结样本里。切回“冻结样本”使用。</p>, '', ''] : !sa || !sb ? [<p key="d" className="v8-note">{sa && sbF && !compatible ? '对照模型快照或数组维度不匹配，已停止比较。' : reason || B.error || '没有可比较的对照样本。'}</p>] : (() => { if (!aligned && opt.diffAlign === 'strict') return [<p key="d">对齐检查：token 数 {sa.T} / {sb.T} ✗。严格对齐时不显示差值。</p>, '', '差值是观察，原因要靠干预。'];
      const rel = Array.from({ length: NL }, (_, l) => { const a = sa.hs(pos, l), b = sb.hs(pos, l); return norm(a.map((x, i) => x - b[i])) / norm(a); });
      return [<><p>对齐检查：同一快照 ✓ · token 数 {sa.T} / {sb.T} {aligned ? '✓' : '✗（按命名读出位置比较）'}</p><p className="v8-note">{diffOn ? '3D 中当前层网格显示本运行 − 对照运行的差值；单元详情保留原值。' : '3D 差值显示已关闭，网格显示本运行的原值。'}</p></>, <><h4>每层相对差异</h4>{line([{ v: rel, color: '#a33e43', name: POSN[positions[pos]] }], layer, 2)}</>, '差值是观察，原因要靠干预；不对齐时差值含前缀差异。']; })(),
    lens: () => !sa?.d.lens ? [browsing ? needNative : <p key="l" className="v8-note">{reason || '此轨迹未保存逐层 Logit Lens，未替换为其他样本。'}</p>] : (() => { const L = [{ v: sa.d.lens.map(r => r[0] - r[1]), color: '#2a78d6', name: '本运行' }, ...(sb ? [{ v: sb.d.lens.map(r => r[0] - r[1]), color: '#d4622b', name: '对照运行' }] : [])];
      return [<><h4>Logit Lens · 答案位置 · {nameK}色 − {nameP}色</h4>{line(L, layer)}</>, <p className="v8-note">3D：LLM 右缘的条。</p>, '读出近似，早层不可靠。']; })(),
    tests: () => [<TestsPanel key="t" />, '', ''],
    result: () => liveResult ? [<><h4>下一个词的预测 · {liveResult.tokens.length} 个 token · {liveResult.elapsed} ms</h4><TopTokens top={liveResult.top} /></>, <><h4>每层改写幅度（3D 主干上的金色圆点）</h4>{line([{ v: liveResult.curve, color: '#d29a00', name: '1 − cos(h_l, h_(l−1))，最后一个位置' }], layer, 2)}</>, '实时生成是一次前向的未登记观察。'] : [<p key="r" className="v8-note">还没有该模型的实时运行。</p>],
  };
  P.live = P.delta;
  P.response = P.diff;
  const pluginTabs = AVAILABLE.filter(p => listed.has(p.id));
  const pluginTab=pluginTabs.some(p=>p.id===tab)?tab:pluginTabs[0]?.id;
  const runTab=['live','response','result'].includes(tab)?tab:mode==='response'?'response':'live';
  const cur=consoleTab==='plugin'?pluginTab:consoleTab==='run'?runTab:'object';
  const archiveHead = activeArchive?.presentation==='selection' && snapshot===activeArchive.snapshot &&
    layer===activeArchive.location.layer && sel.head===activeArchive.location.index;
  const matchingHeadRun=archiveHead && Boolean(selection);
  const geometryExample=exampleReplay || (archivedUnit && (!context.run || context.run===matchingExample?.run) ? matchingExample : null);
  const analysisTrace=isLive&&sa&&activeRun&&tokenPosition!=null?{run:activeRun,base:replay?traceService.base:activeRun.base}:null;
  const frozenSample=!isLive&&!browsing&&!exampleReplay&&sa===saF?sa:null;
  const analysisSceneKey=[observeMode,snapshot,context.run,activeRun?.id||'',run.i,pos,layer,object.module,object.index,headCase,exampleSite].join(':');
  const sceneAnalysis=pinnedAnalysis?.key===analysisSceneKey?pinnedAnalysis:null;
  const pinAnalysis = result=>{
    setPinnedAnalysis({...result,key:analysisSceneKey,location:validObject?selected:null,
      readout:isLive?posName(pos):archiveHead?'末尾输入 token':posName(pos)});
    requestAnimationFrame(()=>{
      sceneRef.current?.focus({preventScroll:true});
      sceneRef.current?.scrollIntoView({block:'start'});
    });
  };
  const panel = !cur ? <p className="v8-note">从操作面板启用一个插件。</p>
    : cur==='attnviz'?<WorkspaceAttention snapshot={snapshot} block={Bk} head={sel.head} query={tokenPosition} headCase={headCase}
        archiveHead={matchingHeadRun} selection={selection} sample={archiveHead?null:frozenSample} trace={analysisTrace} reason={reason} onPin={pinAnalysis}/>
    : cur==='geometry'?<WorkspaceGeometry layer={layer} layerCount={NL} width={DM} snapshot={snapshot}
        example={geometryExample} sample={archiveHead?null:frozenSample} trace={analysisTrace}
        reason={archiveHead?'这道题缺少多位置 HiddenState。可回放中心实测样例，或选择保存了多个读出位置的轨迹。':reason} onPin={pinAnalysis}/>
    : cur==='sae'?<SparseDictionaryExplorer snapshot={snapshot} block={Bk} onLocate={block=>setLayer(block+1)}/>
    : cur === 'object' ? null : MODE_IDS.has(cur) && !P[cur] ? (mode === cur ? <div className="aw-mode">{contextBar}{children}</div> : <div className="aw-method-content"><button type="button" className="fw-tbtn" onClick={()=>openPlugin(cur)}>打开{MODE_PLUGINS.find(p => p.id === cur).name}</button></div>)
      : P[cur] ? (() => { const [a, b, bound] = P[cur](); return <div className="aw-pbody"><div>{a}</div><div>{b}</div>{bound && <div className="aw-bound">{bound}</div>}</div>; })()
        : (() => { const p = ALL.find(x => x.id === cur); return <div className="aw-pbody"><div><h4>{p.name}</h4><p>{p.desc}</p></div><div><p className="v8-note">{p.tag === 'have' ? '项目里已有实现，正式接成插件后读同一个观察对象，并在 3D 上画自己的图层。' : '待接入：按插件规范声明需要的数据、3D 图层与面板。'}</p></div></div>; })();

  const renderPopBody = () => { const { h } = visiblePop;
    if (h.o === 'hs' && sa && h.idx < DM) { const vals = Array.from({ length: NL }, (_, l) => sa.hs(pos, l)[h.idx]);
      return <><h4>HiddenState · 第 {h.l} 层 · 坐标 {h.idx}</h4><div className="aw-kv"><span>值</span><span><b>{vals[h.l].toFixed(4)}</b></span>{sb && <><span>对照运行</span><span>{sb.hs(pos, h.l)[h.idx].toFixed(4)}</span></>}<span>本层 ‖h‖</span><span>{norm(sa.hs(pos, h.l)).toFixed(1)}</span></div>
        {line([{ v: vals, color: '#2a78d6', name: '同编号坐标的跨层响应（非因果链）' }], h.l, 2)}<p className="v8-note">{h.l === NL - 1 ? '本层已过最终归一化。' : ''}相邻坐标：{[-2, -1, 1, 2].map(d => h.idx + d).filter(i => i >= 0 && i < DM).map(i => `${i}: ${sa.hs(pos, h.l)[i].toFixed(2)}`).join('　')}</p></>; }
    if (h.o === 'mlp' && sceneMeasures) return <><h4>{describeScene(h)}</h4><p>{mlpValues ? mlpField.source : mlpMissing}</p>
      {mlpOther&&<p>对照运行 A <output>{rawNumber(mlpOther[h.idx])}</output></p>}</>;
    if (h.o === 'heads' && h.idx < NH) return <><h4>注意力头 B{h.l - 1} · H{h.idx}</h4><p>{describeScene(h)}</p>
      <p className="v8-note">{headColor==='strength' ? headField?.source||headField?.missing : probabilityScope||probabilityMissing}</p></>;
    return <><h4>{describe(h)}</h4><p className="v8-note">{reason || '该位置没有原生采集值。'}</p></>;
  };

  const tabSelectors = visibleConsoleTabs.map(id=>CONSOLE_TABS.find(([key])=>key===id));
  const exampleUnit = layer===exampleReplay?.block+1 && sel.unit!=null ? sel.unit : exampleReplay?.units[0];
  const exampleValue = exampleReplay?.field('mlp', exampleReplay.block+1, exampleSite)?.[exampleUnit];
  const exampleField = exampleReplay && (pluginTab==='hs'||pluginTab==='mlp') ? exampleReplay.field(pluginTab,layer,exampleSite) : null;
  const exampleIndex = pluginTab==='mlp' ? sel.unit ?? exampleReplay?.units[0] : sel.coord ?? 0;
  const selectIndex = location => {
    if(fieldOpen || consoleTab==='plugin'&&ANALYSIS_TOOL_IDS.has(tab)){
      changeObject(location);setIndexQuery('');setIndexError('');return;
    }
    locate(location); setIndexQuery(''); setIndexError('');
    if (featureAnalysisActive) {
      if (location.module === 'mlp') setFeatureKind('unit');
    } else if (observeMode === 'archive' || consoleTab === 'plugin') {
      const plugin = location.module === 'attn' ? 'heads' : location.module;
      if (!listed.has(plugin)) listPlugin(plugin,true);
      openPlugin(plugin);
    }
  };
  const searchIndex = event => {
    event.preventDefault();
    const match = /^([BL])(\d+)\s*\/\s*([HUD])(\d+)(?:\s*\/\s*D(\d+))?$/i.exec(indexQuery.trim());
    const prefix = match?.[1].toUpperCase(), kind = match?.[3].toUpperCase();
    const nativeLayer = Number(match?.[2]) + (prefix === 'B' ? 1 : 0);
    const index = Number(match?.[4]);
    const limit = kind === 'H' ? NH : kind === 'U' ? NF : DM;
    const channel=match?.[5]==null?null:Number(match[5]);
    if (!match || (prefix === 'L') !== (kind === 'D') || nativeLayer >= NL || index >= limit ||
        (channel!==null&&(kind!=='H'||!(channel<headWidth))) || (fieldOpen&&kind==='D')) {
      setIndexError(fieldOpen?`使用 B0–B${NB-1}/H0–H${NH-1}/D0–D${headWidth-1} 或 B0–B${NB-1}/U0–U${NF-1}。`:
        `使用 B0–B${NB-1} 的 H0–H${NH-1} 或 U0–U${NF-1}；坐标使用 L0–L${NL-1}/D0–D${DM-1}。`);
      return;
    }
    if(channel!==null){changeChannel(channel);if(observeMode==='features'&&!fieldOpen)changeFieldView(true,{layer:nativeLayer});}
    selectIndex({layer:nativeLayer,module:kind==='H'?'attn':kind==='U'?'mlp':'hs',index});
  };
  const shortcuts = dimensionsReady ? archiveShortcuts(archives,activeModel) : [];
  const changeArchiveView = id => {
    setArchiveSource({trace:traceId,kind:id==='trace'?'trace':'frozen'});
    if(id==='trace'){setTab('live');return;}
    setComparisonView(id);setTab('response');
    if(id==='pair' && mode!=='response')onMode?.('response');
  };
  const checkLocalModel = () => lm.refresh();
  const showLayerStep = <button type="button" className="fw-tbtn" disabled={!dimensionsReady}
    onClick={()=>setLayer(previous=>previous < NL-1 ? previous+1 : 0)}>下一层</button>;
  const runtimeControls = <div className="aw-load" role="group" aria-label="本机模型与运行状态">
    <span className="v8-note" role="status">{localModelName || '本机模型'} · {serviceStatus?.status==='ready'?'已加载':
      serviceStatus?.status==='loading'?(serviceStatus.progress||'加载中'):'待准备'}</span>
    {modelMismatch&&localModel&&serviceStatus?.available ?
      <button type="button" className="fw-tbtn primary" disabled={gen.busy||lm.busy}
        onClick={()=>{selectModel(localModel);if(canLoad)lm.load(localModel.model_snapshot);}}>切换到 {shortModel(localModel.model)}{canLoad?' 并加载':''}</button> :
      <button type="button" className="fw-tbtn primary" disabled={gen.busy||lm.busy||modelMismatch||!canLoad}
        onClick={()=>lm.load(snapshot)}>加载模型</button>}
    {serviceStatus?.status==='ready'&&<button type="button" className="fw-tbtn" disabled={gen.busy||lm.busy} onClick={lm.unload}>卸载模型</button>}
    {live.base!=null&&<button type="button" className="fw-tbtn" onClick={checkLocalModel}>刷新状态</button>}
    {live.checked&&live.base==null&&<button type="button" className="fw-tbtn" onClick={live.reconnect}>重新查找本机服务</button>}
    {(serviceStatus?.error||lm.msg)&&<span role="status" className="v8-note">{serviceStatus?.error||lm.msg}</span>}
  </div>;
  const tokenPicker = isLive ? <label>Token<select aria-label="当前Token" value={pos??''} disabled={!positions.length}
    onChange={event=>setPos(+event.target.value)}>{positions.map((token,index)=><option key={index} value={index}>
      {index} · {tokText(token)}{activeRun?.readout?.includes(index)?' · 模块读出':''}</option>)}</select></label> :
    !browsing&&native&&<label>读出位置<select aria-label="当前Token" value={pos??''} onChange={event=>setPos(+event.target.value)}>
      {positions.map((name,index)=><option key={name} value={index}>{POSN[name]||name}</option>)}</select></label>;

  const fieldKind=object.module==='attn'&&sel.head!=null?'head':object.module==='mlp'&&sel.unit!=null?'ffn':null;
  const traceStructure=replay&&traceManifest?.model_snapshot===snapshot ? traceManifest : null;
  const replayIdentityOk=traceStructure?.trace_id===traceId&&traceStructure?.n_layers===NB&&traceStructure?.d_model===DM&&
    (!context.run||!traceStructure.run_id||traceStructure.run_id===(activeReference?.traceRun||context.run))&&
    (traceStructure.n_heads==null||traceStructure.n_heads===NH)&&(traceStructure.d_ff==null||traceStructure.d_ff===NF);
  const fieldTrace=replay&&activeRun&&replayIdentityOk ? {run:activeRun,base:traceService.base} :
    liveResult ? {run:liveResult,base:liveResult.base} : null;
  const fieldSample=!context.run||context.run===samples?.id?frozenSample:null;
  const headWidth=activeModel?.head_dim??traceStructure?.head_dim;
  const kvHeadCount=activeModel?.n_kv_heads??traceStructure?.n_kv_heads;
  const currentObject = !layer?'词嵌入':!validObject?'第 '+layer+' 层':selected.module==='attn'?
    'B'+(layer-1)+'/H'+selected.index+(fieldChannel>=0&&fieldChannel<headWidth&&(fieldOpen||observeMode!=='features')?'/D'+fieldChannel:''):
    selected.module==='mlp'?'B'+(layer-1)+'/U'+selected.index:'第 '+layer+' 层 · 坐标 '+selected.index;
  const archiveExample=activeArchive?.presentation==='path-example'&&matchingExample?.run===activeArchive.run?matchingExample:null;
  const fieldHeadArchive=activeArchive?.presentation==='selection';
  const fieldExample=activeArchive?archiveExample:geometryExample;
  const fieldSourceError=replay&&!fieldTrace?(!traceService.checked?'正在查找保存此轨迹的节点…':
    recordTraceMismatch?reason:
    traceService.base==null?'已尝试连接中心和本机，未找到此轨迹的清单。':
      '轨迹、运行或快照与当前选择不一致。'):
    activeArchive&&activeArchive.snapshot!==snapshot?'此记录属于另一个模型快照。':
    observeMode==='archive'&&archiveId&&!activeArchive?(workspaceCatalog.loading?'正在读取中心档案目录…':'当前中心或会话未提供此归档。'):
    bundleId&&!archiveBundle?.data?(archiveBundle?.error?`读取归档失败：${archiveBundle.error}`:'正在读取中心归档…'):
    activeArchive?.presentation==='path-example'&&!archiveExample?(exampleError?`读取归档失败：${exampleError}`:'正在读取中心路径归档…'):
    observeMode==='live'&&!liveResult&&!exampleReplay?(gen.busy?'正在采集当前输入…':gen.err||runReason||'当前输入尚未采集。请在实时模式开始运行。'):'';
  const pickFieldKind=(kind,block,idx=0)=>{
    changeObject({layer:block+1,module:kind==='head'?'attn':'mlp',index:idx});
  };
  const changeChannel=value=>{setFieldChannel(value);saveView({observeChannel:value||null});};
  const changeHeadCase=value=>{setHeadCase(value);saveView({headCase:value});};
  const changeExampleSite=value=>{setExampleSite(value);saveView({observeExampleSite:value});};
  const showRecordPanel=()=>{
    if(fieldOpen)changeFieldView(false);
    selectRecordTab('records');setConsoleTab('research');saveView({observeTab:'research'});
    if(!max)requestAnimationFrame(()=>consoleRef.current?.scrollIntoView({block:'nearest'}));
  };
  const searchFromScene=()=>{
    setWorkspaceSearch(previous=>({serial:(previous?.serial||0)+1,modelId:activeModel.model,
      modelLabel:shortModel(activeModel.model),snapshot:activeModel.model_snapshot,
      target:validObject?nativeAddress(selected):''}));
    showRecordPanel();
  };
  const openCaseReplay=entry=>{
    const target=models.find(m=>m.model_snapshot===entry.snapshot);
    if(!target)return '此题对应的模型结构尚未同步。';
    const {layer:targetLayer,module,index}=entry.location;
    const width=module==='attn'?target.n_heads:module==='mlp'?target.d_ff:target.d_model;
    if(targetLayer>target.n_layers||!(index<width))return '归档原生地址与模型结构不一致。';
    const error=locateArchive(entry);
    if(error)return error;
    setReplayReference(entry);setArchiveId('');setBundleRow('');setFieldChannel(0);
    setArchiveSource({trace:entry.trace,kind:'trace'});setArchivePosition({id:entry.trace,position:entry.position});
    chooseObserveMode('archive');setTab('live');setConsoleTab('run');
    if(fieldOpen)changeFieldView(false);
    saveView({observeArchive:null,observeArchiveRow:null,observeChannel:null,observeTab:'run',tpos:entry.position});
    return '';
  };
  const openObservationArchive=(id,{caseId='',scene=false,location,row,site}={})=>{
    if(id==='current')return;
    setReplayReference(null);
    const entry=archives.find(item=>item.id===id);
    if(!entry){
      setArchiveId('');setBundleRow('');setArchivePosition({});
      publishResearchContext({run:'',trace:'',context_trace:''});
      saveView({observeArchive:null,observeArchiveRow:null,tpos:null});
      return;
    }
    const target = location || entry.location;
    const error=locateArchive({...entry,trace:'',location:target});
    if(error){setIndexError(error);return;}
    chooseObserveMode('archive');
    const switching = id!==activeArchive?.id;
    setArchiveId(id);setIndexQuery('');setIndexError('');
    if(switching){setBundleRow('');setExampleSite(3);setFieldChannel(0);}
    if(row!=null)setBundleRow(row);
    if(Number.isInteger(site))setExampleSite(site);
    if(switching||caseId)changeHeadCase(caseId);
    if(scene){
      if(fieldOpen)changeFieldView(false);
      const plugin=target.module==='mlp'?'mlp':'heads';
      if(!listed.has(plugin))listPlugin(plugin,true);
      if(plugin==='heads')setHeadColor(entry.presentation==='selection'?'probability':'strength');
      setTab(plugin);selectConsoleTab('plugin',plugin);
    }else changeFieldView(true,target);
    saveView({observeArchive:id,tpos:null,...(switching?{observeArchiveRow:null,observeChannel:null,observeExampleSite:null}:{}),
      ...(row!=null?{observeArchiveRow:row}:{}),...(Number.isInteger(site)?{observeExampleSite:site}:{})});
  };
  const bundledHeadCase=fieldHeadArchive&&activeArchive.snapshot===snapshot?archiveHeadCase(selection,headCase):null;
  const recordOnScene=bundledHeadCase?{location:activeArchive.location,caseId:bundledHeadCase.id,readout:'末尾输入 token',
    output:bundledHeadCase.output}:activeReference&&replayIdentityOk&&replay?{location:activeReference.location,
      caseId:activeReference.caseId,readout:'token '+activeReference.position}:null;
  const recordPoint=recordOnScene?{...recordOnScene.location,id:recordOnScene.caseId,level:'observation'}:null;
  const libraryOverlay=libraryVisible&&archiveLayer?.snapshot===snapshot?archiveLayer:null;
  const analysisPoints=sceneAnalysis?.layers?.map(l=>({layer:l,layerOnly:true,level:'observation'}))||
    (sceneAnalysis?.location?[{...sceneAnalysis.location,level:'observation'}]:[]);
  const spatialOverlay=(recordPoint||analysisPoints.length)?{
    label:sceneAnalysis?'已标记的分析位置':'已核验的测试位置',
    points:[...(recordPoint?[recordPoint]:[]),...analysisPoints,...(libraryOverlay?.points||[])],
    onPick:entry=>{if(libraryOverlay?.points.includes(entry))libraryOverlay.onPick?.(entry);},
  }:libraryOverlay;

  const sceneHeadCase = bundledHeadCase;
  const sceneMeasures = observeMode!=='features' && (!browsing || Boolean(activeArchive||fieldExample||sceneHeadCase));
  const expandHead = observeMode!=='features' && shown('heads') && fieldKind==='head' && validObject && positiveDimension(headWidth);
  const channelIndex = Number.isInteger(fieldChannel) && fieldChannel>=0 && fieldChannel<headWidth ? fieldChannel : 0;
  const bundledArchiveData = archiveBundle.data;
  const selectedBundle = useMemo(()=>bundledArchiveData ? {data:bundledArchiveData,row:archiveBundle.row,
    field:archiveBundle.field,pending:archiveBundle.pending,fieldError:archiveBundle.fieldError} : null,
    [bundledArchiveData,archiveBundle.row,archiveBundle.field,archiveBundle.pending,archiveBundle.fieldError]);
  const headField = useSceneHeadField({enabled:!fieldOpen&&dimensionsReady&&Bk>=0&&positiveDimension(headWidth)&&shown('heads')&&
      (expandHead||sceneMeasures&&headColor==='strength'),trace:fieldTrace,snapshot,block:Bk,position:pos,width:headWidth,count:NH,
    bundle:selectedBundle,sourceError:fieldSourceError});
  const channelValues = headField?.vectors?.[sel.head];
  const rawHead = useMemo(()=>{
    if(sceneHeadCase && !fieldSourceError){
      if(layer!==activeArchive.location.layer)return null;
      return Array.from({length:NH},(_,i)=>i===activeArchive.location.index ? sceneHeadCase.mass[sceneHeadCase.gold] : null);
    }
    if(!sa || Bk<0 || activeArchive || fieldSourceError || !Number.isInteger(sa.d.tokpos[headTarget]))return null;
    return Array.from({length:NH},(_,i)=>sa.att(Bk,i)[sa.d.tokpos[headTarget]]);
  },[sceneHeadCase,fieldSourceError,layer,NH,sa,Bk,activeArchive,headTarget]);
  const otherHead = useMemo(()=>diffOn&&rawHead&&sb&&Number.isInteger(sb.d.tokpos[headTarget])&&!sceneHeadCase
    ? Array.from({length:NH},(_,i)=>sb.att(Bk,i)[sb.d.tokpos[headTarget]]) : null,
  [diffOn,rawHead,sb,headTarget,sceneHeadCase,NH,Bk]);
  const selectedHeadOverview = headColor==='strength' ? headField?.strengths : rawHead;
  const headDifference = headColor==='probability' && Boolean(otherHead);
  const headDisplay = useMemo(()=>headDifference ? rawHead.map((a,i)=>a-otherHead[i]) : selectedHeadOverview,
    [headDifference,rawHead,otherHead,selectedHeadOverview]);
  const mlpField = useMemo(()=>{
    if(Bk<0)return {};
    if(fieldSourceError)return {missing:fieldSourceError};
    if(selectedBundle){
      return readObservationArchive({...selectedBundle,block:Bk,measure:'mlp_activation',size:NF});
    }
    if(fieldExample){
      const site = fieldExample.positions[exampleSite];
      return {values:fieldExample.field('mlp',layer,exampleSite),source:site?`${fieldExample.caseId} · token ${site.index}`:'',
        missing:`这条记录只保存 B${fieldExample.block} 的 MLP 激活。`};
    }
    if(sceneHeadCase)return {missing:'选源原题未随包保存 MLP 激活。'};
    if(sa && mlpOk && !activeArchive){
      return {values:sa.mlp(Bk),other:diffOn?sb?.mlp(Bk):null,source:isLive?`轨迹 ${activeRun?.id} · token ${pos}`:POSN[positions[pos]]};
    }
    return {missing:sa ? '此位置未采集 MLP 激活。请选择模块读出位置。' : reason || '此记录未保存 MLP 原场。'};
  },[Bk,fieldSourceError,selectedBundle,NF,fieldExample,exampleSite,layer,sceneHeadCase,sa,mlpOk,
    activeArchive,diffOn,sb,isLive,activeRun?.id,pos,positions,reason]);
  const mlpValues = useMemo(()=>mlpField.values?.length===NF&&mlpField.values.every(Number.isFinite)?mlpField.values:null,[mlpField.values,NF]);
  const mlpOther = useMemo(()=>mlpField.other?.length===NF&&mlpField.other.every(Number.isFinite)?mlpField.other:null,[mlpField.other,NF]);
  const mlpMissing = mlpField.values ? 'MLP 原生向量长度或数值与当前模型不符。' : mlpField.missing || '当前对象没有 MLP 原生数据。';
  const mlpDisplay = useMemo(()=>mlpValues&&mlpOther ? mlpValues.map((value,i)=>value-mlpOther[i]) : mlpValues,[mlpValues,mlpOther]);
  const mlpAuto = useMemo(()=>mlpDisplay ? pct([mlpDisplay],opt.mlpClip) : null,[mlpDisplay,opt.mlpClip]);
  const headAuto = headField?.strengths ? absMax(headField.strengths) : null;
  const channelAuto = channelValues ? absMax(channelValues) : null;
  const locked = scaleLock?.snapshot===snapshot;
  useEffect(()=>{
    if(!locked)return;
    setScaleLock(previous=>{
      if(previous?.snapshot!==snapshot)return previous;
      const strength=previous.strength??headAuto, channel=previous.channel??channelAuto, mlp=previous.mlp??mlpAuto;
      return strength===previous.strength&&channel===previous.channel&&mlp===previous.mlp ? previous : {...previous,strength,channel,mlp};
    });
  },[locked,snapshot,headAuto,channelAuto,mlpAuto]);
  const probabilityAuto = opt.headMax*(headDifference ? .8 : 1);
  const probabilityScale = locked ? scaleLock.probability : probabilityAuto;
  const overviewScale = headColor==='strength' ? (locked?scaleLock.strength??headAuto:headAuto) : probabilityScale;
  const channelScale = locked?scaleLock.channel??channelAuto:channelAuto;
  const mlpScale = locked?scaleLock.mlp??mlpAuto:mlpAuto;
  const sceneTextures = useMemo(()=>{
    if(!colorValues||!dimensionsReady||Bk<0)return TX;
    const out={...TX};
    if(atlas.run&&!hide.has('mlp')){
      out.mlp={[Bk]:heatTexture(mw,mh,NF,atlas.values,atlasScale,atlas.signed?divc:seqc)};
    }
    if(!sceneMeasures)return out;
    if(!hide.has('heads'))out.heads={[Bk]:heatTexture(8,Math.ceil(NH/8),NH,headDisplay,overviewScale,headDifference?divc:seqc)};
    if(!hide.has('mlp'))out.mlp={[Bk]:heatTexture(mw,mh,NF,mlpDisplay,mlpScale,divc)};
    return out;
  },[colorValues,dimensionsReady,Bk,sceneMeasures,TX,hide,NH,headDisplay,overviewScale,headDifference,mw,mh,NF,
    mlpDisplay,mlpScale,atlas.run,atlas.values,atlasScale,atlas.signed]);
  const channelCols = Math.min(16,headWidth), channelRows = Math.ceil(headWidth/channelCols);
  const channelTexture = useMemo(()=>expandHead && !fieldOpen && colorValues
    ? heatTexture(channelCols,channelRows,headWidth,channelValues,channelScale,divc) : null,
  [expandHead,fieldOpen,colorValues,channelCols,channelRows,headWidth,channelValues,channelScale]);
  const headChannels = expandHead ? {head:sel.head, grid:[channelCols,channelRows],count:headWidth,selected:channelIndex,texture:channelTexture} : null;
  const probabilityScope = sceneHeadCase?'末位查询 → 目标记录（概率合计）':rawHead
    ? `查询 token ${bundleData?.query??sa.T-1} → 来源 token ${sa.d.tokpos[headTarget]}` : '';
  const probabilityMissing = fieldSourceError || (sceneHeadCase?'逐 token 的注意力未发布，归档只包含 B26/H36 的记录概率。':
    selectedBundle?'此中心归档未保存注意力概率矩阵。':reason || '此记录未保存注意力概率。');
  const overviewMissing = headColor==='strength' ? headField?.missing : !rawHead ? probabilityMissing :
    sceneHeadCase ? '其余 Head 无同题注意力记录。' : '';
  const describeScene = h => {
    if(atlas.run&&h.o==='mlp')return `B${Bk}/U${h.idx} · ${atlas.title} ${rawNumber(atlas.values?.[h.idx])}`;
    if(h.o==='headChannels')return `B${Bk}/H${sel.head}/D${h.idx} · Z ${rawNumber(channelValues?.[h.idx])}`;
    if(sceneMeasures&&h.o==='heads')return `B${Bk}/H${h.idx} · ${headColor==='strength'?'RMS(Z)':'注意力概率'} ${rawNumber(selectedHeadOverview?.[h.idx])}`;
    if(sceneMeasures&&h.o==='mlp')return `B${Bk}/U${h.idx} · A ${rawNumber(mlpValues?.[h.idx])}`;
    return describe(h);
  };
  const bundledPosition = observeMode==='archive' && fieldExample ? <label>读出位置<select aria-label={fieldOpen?'全屏归档读出位置':'3D 归档读出位置'} value={exampleSite}
    onChange={event=>changeExampleSite(Number(event.target.value))}>
    {fieldExample.positions.map((p,i)=><option key={p.site} value={i}>{EXAMPLE_SITES[i]} · token {p.index}</option>)}
  </select></label> : bundledArchiveData?.rows ? <label>{archiveBundle.row?.doc?'归档文档':'读出位置'}<select aria-label={fieldOpen?'全屏归档读出位置':'3D 归档读出位置'}
    value={archiveBundle.row?.key||''} onChange={event=>{setBundleRow(event.target.value);saveView({observeArchiveRow:event.target.value});}}>
    {bundledArchiveData.rows.map(row=><option key={row.key} value={row.key}>{row.doc?`${row.doc} · `:''}token {row.position}（ID {row.tokenId}）</option>)}
  </select></label> : null;

  return <div className="v8-route aw ow-graph" aria-label="模型分析">
    {!dimensionsReady && <p className="v8-note" role="status">{workspaceCatalog.loading?'正在读取中心模型目录…':workspaceCatalog.error||'中心尚未登记当前快照的完整模型结构。'}</p>}
    <section ref={sceneRef} tabIndex={-1} aria-label="模型运行 3D 场景" className={'aw-space'+(max?' max':'')+(fieldOpen?' ow-field-open':'')}>
        <div className="ow-mode-bar">
          <h2>模型分析</h2>
          <div role="group" aria-label="观察模式">{WORKSPACE_MODES.map(([id,label])=>
            <button type="button" key={id} aria-pressed={observeMode===id} onClick={()=>chooseObserveMode(id)}>{label}</button>)}</div>
        </div>
      <div className="ow-workspace-stage">
        <aside className="ow-object-locator" aria-label="对象定位">
          <h3>对象定位</h3>
          <form className="ow-index-search" onSubmit={searchIndex}>
            <input aria-label="输入原生编号" aria-describedby={indexError?'ow-index-error':undefined}
              placeholder={fieldOpen?'B0/H0/D0 或 B0/U0':'输入 B26/H36'} value={indexQuery}
              onChange={event=>{setIndexQuery(event.target.value);setIndexError('');}}/>
            <button type="submit" aria-label="定位对象" title="定位对象" disabled={!dimensionsReady}>↵</button>
          </form>
          {indexError&&<p className="ow-index-error" id="ow-index-error" role="status">{indexError}</p>}
          <div className="ow-archive-shortcuts" role="group" aria-label="有归档的对象">
            <span>有归档的对象</span>
            {shortcuts.map(({archive,label,location})=><button key={`${archive.id}:${label}`} type="button" aria-label={label} title={archive.title}
              aria-pressed={location.layer===layer&&location.module===object.module&&location.index===object.index&&
                (!activeArchive||activeArchive.id===archive.id)}
              onClick={()=>openObservationArchive(archive.id,{location,scene:true})}>
              <strong>{label}</strong><small>{location.module==='attn'?'注意力头':'MLP 单元'}</small></button>)}
            {!shortcuts.length&&<small>{workspaceCatalog.loading?'正在读取中心档案…':'当前快照没有可访问的归档入口。'}</small>}
            {workspaceCatalog.local&&<small role="status">中心暂不可达 · 上次同步的公开目录</small>}
            <button type="button" onClick={workspaceCatalog.reload}>同步中心目录</button>
          </div>
          <div className="ow-selection" aria-live="polite"><span>当前选择</span><strong>{currentObject}</strong></div>
          <button type="button" className="fw-tbtn ow-find-records" disabled={!activeModel} onClick={searchFromScene}>查相关测试</button>
        </aside>
        <div className="ow-viewport">
        <div className="ow-stage-nav" ref={stageNavRef}>
        <div className="ow-scene-bar">
          <label className="ow-model">模型<select aria-label="模型" value={activeModel?.model_snapshot||''} disabled={gen.busy}
            onChange={event=>{const target=models.find(item=>item.model_snapshot===event.target.value);if(target)selectModel(target);}}>
            {!activeModel&&<option value="">选择模型</option>}
            {models.map(model=><option key={model.model_snapshot} value={model.model_snapshot}>{shortModel(model.model)} · {model.model_snapshot.slice(0,8)}</option>)}
          </select></label>
          {dimensionsReady&&<span className="ow-dimensions">{NB} 个块 · 每块 {object.module==='mlp'?`${NF.toLocaleString('zh-CN')} 个 MLP 单元`:`${NH} 个注意力头`}</span>}
          {workspaceCatalog.local&&<span className="v8-note ow-bundled">中心离线缓存</span>}
          <span className="ow-view" hidden={fieldOpen}><select aria-label="空间视角" value={view} onChange={event=>setView(event.target.value)}>
            {VORDER.map(id=><option key={id} value={id}>{VIEWS[id][2]}</option>)}</select></span>
          <div className="ow-scene-actions">
            {max&&<button type="button" aria-pressed={!fieldOpen} onClick={()=>changeFieldView(false)}>3D 与控制台</button>}
            <button type="button" ref={fieldButton} aria-pressed={fieldOpen} title="在全屏中按原生编号查看全部 Head 通道与 FFN 单元" onClick={()=>changeFieldView(true)}>原生单元</button>
            {!fieldOpen&&<button type="button" aria-expanded={displayOpen} onClick={()=>{setDisplayOpen(value=>!value);setLocator(false);}}>显示插件</button>}
            {!fieldOpen&&<button type="button" aria-expanded={locator} onClick={()=>{setLocator(value=>!value);setDisplayOpen(false);setPop(null);}}>精确定位</button>}
            <button type="button" onClick={()=>changeFullscreen(!max)}>
              {max?'退出全屏（Esc）':'⛶ 全屏'}</button>
          </div>
        </div>
        {!fieldOpen&&atlasControls}
        {!fieldOpen&&sceneMeasures&&Bk>=0&&<div className="ow-heat-controls" role="group" aria-label="3D 数值着色">
          {shown('heads')&&<label>Head 着色<select aria-label="Head 着色量" value={headColor} onChange={event=>setHeadColor(event.target.value)}>
            <option value="probability">注意力概率</option><option value="strength">输出强度</option>
          </select></label>}
          <button type="button" className="fw-tbtn" aria-pressed={locked}
            onClick={()=>setScaleLock(locked?null:{snapshot,probability:probabilityAuto,strength:headAuto,channel:channelAuto,mlp:mlpAuto})}>锁定色标</button>
          {shown('heads')&&Number.isFinite(overviewScale)&&<span data-scene-scale="heads">Head {headDifference?'±':'0–'}{scaleNumber(overviewScale)}</span>}
          {expandHead&&Number.isFinite(channelScale)&&<span data-scene-scale="channels">通道 ±{scaleNumber(channelScale)}</span>}
          {shown('mlp')&&Number.isFinite(mlpScale)&&<span data-scene-scale="mlp">MLP ±{scaleNumber(mlpScale)}</span>}
          {bundledPosition}
          {sceneHeadCase&&!(cur==='heads'&&matchingHeadRun)&&<ArchiveHeadCondition data={selection} caseId={sceneHeadCase.id} onCase={changeHeadCase}/>}
        </div>}
        {!fieldOpen&&liveResult&&<div className="ow-scene-record" role="region" aria-label="3D 运行状态">
          <strong>本次运行</strong><span title={liveResult.id}>轨迹 {liveResult.id.slice(0,16)}</span><span>{posName(pos)}</span>
          {liveResult.readout?.includes(pos)&&<span>模块读出</span>}
          {liveResult.top?.[0]&&<span>下一 token <code>{tokText(liveResult.top[0].text)}</code></span>}
        </div>}
        {!fieldOpen&&(recordOnScene||activeReference&&replay)&&<div className="ow-scene-record" role="region" aria-label="3D 测试读数">
          {recordOnScene?<><strong>{nativeAddress(recordOnScene.location)}</strong><span>归档原题 {recordOnScene.caseId} · {recordOnScene.readout}</span>
            {recordOnScene.output!=null&&<span>实际输出 <code>{recordOnScene.output}</code></span>}</>:
            <span role="status">{fieldSourceError||'正在核对逐题轨迹。'}</span>}
          <button className="fw-tbtn" type="button" onClick={showRecordPanel}>查看测试记录</button>
        </div>}
        {!fieldOpen&&sceneAnalysis&&<div className="ow-scene-record" role="region" aria-label="3D 分析结果">
          <strong>{sceneAnalysis.title}</strong><span>{sceneAnalysis.scope}</span>
          {!sceneAnalysis.layers&&<span>{nativeAddress(sceneAnalysis.location||{})} · {sceneAnalysis.readout}</span>}
          {sceneAnalysis.values.map(value=><span key={value}>{value}</span>)}
          <button type="button" className="fw-tbtn" onClick={()=>setPinnedAnalysis(null)}>清除标记</button>
        </div>}
        {displayOpen&&<div className="aw-locator ow-display" role="group" aria-label="显示插件">
          {[['heads','Head'],['hs','HiddenState'],['mlp','MLP']].map(([id,name])=><label key={id}>
            <input type="checkbox" checked={shown(id)} onChange={()=>chooseDisplay(id)}/>{name} 网格</label>)}
          {featureAnalysisActive&&<label><input type="checkbox" checked={featureMarks}
            onChange={event=>setFeatureMarks(event.target.checked)}/>单元组标记</label>}
          <label><input type="checkbox" checked={colorValues} onChange={event=>setColorValues(event.target.checked)}/>数值着色</label>
          {native&&<>
            <label><input type="checkbox" checked={opt.showDelta} onChange={event=>setOpt(state=>({...state,showDelta:event.target.checked}))}/>层间变化</label>
            {sa?.d.lens&&<label><input type="checkbox" checked={Boolean(opt.showLens)} onChange={event=>setOpt(state=>({...state,showLens:event.target.checked}))}/>词表读出</label>}
            {sb&&<label><input type="checkbox" checked={opt.showDiff} onChange={event=>setOpt(state=>({...state,showDiff:event.target.checked}))}/>对照差值</label>}
            <ResearchDisclosure title="色标、目标和对齐">{OPTS.hs()}{OPTS.mlp()}{OPTS.heads()}{sb&&OPTS.diff()}</ResearchDisclosure>
          </>}
        </div>}
      </div>
      <Scene dims={{ NL, describe:describeScene }} objs={objs} layer={layer} view={view} tex={sceneTextures} marks={marks} onPick={onPick}
        overlays={overlays} live={colorValues&&sa?liveResult:null} links={sceneLinks}
        archive={spatialOverlay} headChannels={fieldOpen?null:headChannels}/>
      {!fieldOpen&&((sceneMeasures&&fieldKind==='ffn'&&validObject)||expandHead)&&<div className="ow-scene-record ow-raw-readout" role="region" aria-label="3D 原生读数">
        {expandHead?<>
          <strong>B{Bk}/H{sel.head}</strong>
          <label>通道<select aria-label="3D Head 通道" value={channelIndex} onChange={event=>changeChannel(Number(event.target.value))}>
            {Array.from({length:headWidth},(_,i)=><option key={i} value={i}>D{i}</option>)}
          </select></label>
          <span>Z 原值 <output aria-label="Head 通道原值">{rawNumber(channelValues?.[channelIndex])}</output></span>
          <span>{headColor==='strength'?'RMS(Z)':'注意力概率'} <output aria-label="Head 总览原值">{rawNumber(selectedHeadOverview?.[sel.head])}</output></span>
          {headDifference&&<span>对照 <output>{rawNumber(otherHead[sel.head])}</output> · 差值 <output>{rawNumber(headDisplay[sel.head])}</output></span>}
          {headField?.source&&<span>Z · {headField.source}</span>}
          {headColor==='probability'&&Number.isFinite(rawHead?.[sel.head])&&<span>{probabilityScope}</span>}
          {(!channelValues||!Number.isFinite(selectedHeadOverview?.[sel.head]))&&<span role="status">
            {!channelValues ? headField?.missing||'此位置未保存该头的 Z。' : probabilityMissing}</span>}
        </>:<>
          <strong>B{Bk}/U{sel.unit}</strong>
          <span>A 原值 <output aria-label="MLP 单元原值">{rawNumber(mlpValues?.[sel.unit])}</output></span>
          {mlpOther&&<span>对照 <output>{rawNumber(mlpOther[sel.unit])}</output> · 差值 <output>{rawNumber(mlpDisplay[sel.unit])}</output></span>}
          {mlpField.source&&<span>{mlpField.source}</span>}
          {!mlpValues&&<span role="status">{mlpMissing}</span>}
        </>}
      </div>}
      {fieldOpen&&<section className="ow-native-field" aria-label="全屏原生单元">
        <header><h3>{fieldKind==='head'?'Head 内部通道':fieldKind==='ffn'?'FFN 神经元':'原生单元'}</h3>
          <div className="ow-field-controls">
            {!(atlas.run&&fieldKind==='ffn')&&<label>读数来源<select aria-label="原生读数来源" value={activeArchive?.id||(replay||liveResult?'current':'')}
              onChange={e=>openObservationArchive(e.target.value)}>
              {!liveResult&&<option value="">当前工作区</option>}
              {(replay||liveResult)&&<option value="current">{replay?'当前测试轨迹':'本次本机采集'}</option>}
              {archives.map(entry=><option key={entry.id} value={entry.id}>{entry.title}</option>)}
            </select></label>}
            <label>观察对象<select aria-label="原生对象类型" value={fieldKind||''} disabled={!dimensionsReady}
              onChange={e=>pickFieldKind(e.target.value,Math.max(0,Bk))}>
              {!fieldKind&&<option value="">选择对象</option>}<option value="head">Head 内部通道</option><option value="ffn">FFN 神经元</option>
            </select></label>
            {!(atlas.run&&fieldKind==='ffn')&&<label>模型块<select aria-label="原生单元模型块" value={Bk<0?'':Bk} disabled={!dimensionsReady}
              onChange={e=>pickFieldKind(fieldKind||'head',Number(e.target.value),fieldKind==='ffn'?sel.unit:sel.head??0)}>
              {Bk<0&&<option value="">嵌入层</option>}{Array.from({length:NB},(_,i)=><option key={i} value={i}>B{i}</option>)}
            </select></label>}
            {fieldKind==='head'&&<label>注意力头<select aria-label="头编号" value={sel.head} onChange={e=>pickFieldKind('head',Bk,Number(e.target.value))}>
              {Array.from({length:NH},(_,i)=><option key={i} value={i}>H{i}</option>)}
            </select></label>}
            {fieldHeadArchive&&!fieldSourceError ? <ArchiveHeadCondition data={selection} caseId={headCase} onCase={changeHeadCase}/> : bundledPosition || (
              fieldExample ? <label>读出位置<select aria-label="全屏归档读出位置" value={exampleSite} onChange={e=>changeExampleSite(Number(e.target.value))}>
                {fieldExample.positions.map((p,i)=><option key={p.site} value={i}>{EXAMPLE_SITES[i]} · token {p.index}</option>)}
              </select></label> : activeArchive?null:tokenPicker)}
          </div>
          {activeArchive&&<p className="v8-note">覆盖范围 · {activeArchive.coverage}</p>}
          {atlasControls}
        </header>
        {fieldKind&&validObject&&layer>0&&dimensionsReady?<Suspense fallback={<p role="status">正在载入原生单元视图…</p>}>
          <FullNativeField kind={fieldKind} model={activeModel.model} snapshot={snapshot} block={Bk} head={sel.head} unit={sel.unit}
            ffnSize={NF} headWidth={headWidth} headCount={NH} kvHeadCount={kvHeadCount}
            example={fieldExample} site={exampleSite} sample={activeArchive||fieldHeadArchive?null:fieldSample} trace={fieldTrace}
            position={pos} mlpOk={mlpOk} sourceError={fieldSourceError}
            atlas={atlas.run&&fieldKind==='ffn'?atlas:null}
            bundled={selectedBundle}
            channel={fieldChannel} onChannel={changeChannel} onHead={i=>pickFieldKind('head',Bk,i)}
            onPick={i=>pickFieldKind('ffn',Bk,i)}/>
        </Suspense>:<p role="status">请选择一个模型块中的 Head 或 FFN，即可按编号查看全部分量。</p>}
        {atlasReport}
      </section>}
      {locator&&<form className="aw-locator" aria-label="原生索引定位" onSubmit={event=>{
        event.preventDefault();
        const form=new FormData(event.currentTarget),targetLayer=Number(form.get('layer')),index=Number(form.get('index'));
        const limit=findKind==='hs'?DM:findKind==='mlp'?NF:NH;
        if(!Number.isInteger(targetLayer)||!Number.isInteger(index)||targetLayer<(findKind==='hs'?0:1)||targetLayer>=NL||index<0||index>=limit)return;
        locate({layer:targetLayer,module:findKind,index});
        if(featureAnalysisActive&&findKind==='mlp')setFeatureKind('unit');
        setLocator(false);
      }}>
        <label>层（L0 = 词嵌入）<input name="layer" type="number" min={findKind==='hs'?0:1} max={NL-1} defaultValue={Math.max(findKind==='hs'?0:1,layer)} required/></label>
        <label>对象<select aria-label="定位对象类型" value={findKind} onChange={event=>setFindKind(event.target.value)}>
          <option value="mlp">MLP 单元</option><option value="attn">Head</option><option value="hs">HiddenState 坐标</option></select></label>
        <label>原生编号<input name="index" type="number" min="0" max={(findKind==='hs'?DM:findKind==='mlp'?NF:NH)-1} defaultValue="0" required/></label>
        <button type="submit" className="fw-tbtn primary" disabled={!dimensionsReady}>定位单元</button>
      </form>}
      <div className="aw-legend v8-note"><span><i style={{background:'#e5def3'}}/>注意力</span><span><i style={{background:'#dcefe4'}}/>MLP</span>
        <span><i style={{background:'#172033'}}/>残差主干</span>
        {colorValues&&(channelValues||mlpValues||exampleReplay||native)&&<span><i style={{background:'linear-gradient(90deg,#d4622b,#fff,#1f5fbf)'}}/>通道与 MLP · 蓝正橙负，白为零{mlpOther?' · MLP 显示对照差值':''}</span>}
        {colorValues&&shown('mlp')&&atlas.values&&<span><i style={{background:atlas.signed?'linear-gradient(90deg,#d4622b,#fff,#1f5fbf)':'linear-gradient(90deg,#fff,#1f5fbf)'}}/>
          MLP · {atlas.title} · {atlas.signed?'蓝正橙负，白为零':'由白到蓝，数值逐渐增大'}</span>}
        {colorValues&&shown('heads')&&selectedHeadOverview&&<span>{headColor==='strength'?'Head 越蓝，输出 Z 的均方根越大':headDifference?'Head 概率差值 · 蓝高橙低':'Head 越蓝，注意力概率越高'}</span>}
        {colorValues&&shown('heads')&&headColor==='strength'&&diffOn&&<span>对照没有 Head 输出场，显示本运行强度。</span>}
        {colorValues&&sceneMeasures&&Bk>=0&&shown('heads')&&overviewMissing&&<span>Head · {overviewMissing}</span>}
        {colorValues&&sceneMeasures&&Bk>=0&&shown('mlp')&&!mlpValues&&fieldKind!=='ffn'&&<span>MLP · {mlpMissing}</span>}
        {marks.some(mark=>mark.feature)&&<span><i style={{background:'#b87916'}}/>冻结单元组</span>}
        {colorValues&&browsing&&localObject?.legend?.map(entry=><span key={entry.name}><i style={{background:entry.color}}/>{entry.name}</span>)}
        {sceneLinks.length>0&&<span>紫虚线为共同响应，蓝线为登记关联</span>}
        <span>{SCENE_MOUSE_HINT} · 点击层或网格 · 灰色为结构或缺测</span>
      </div>
      {visiblePop&&<div className="aw-pop" role="dialog" aria-label="单元详情" style={{
        left:'clamp(8px, '+(visiblePop.x+16)+'px, calc(100% - 344px))',
        top:'clamp(80px, '+(visiblePop.y+24)+'px, calc(100% - 260px))'}}>
        <button type="button" className="x" aria-label="关闭" onClick={()=>setPop(null)}>×</button>
        {renderPopBody()}{objectEvidence}
      </div>}
        </div>
      </div>
    </section>

    <section hidden={fieldOpen} className={'v8-card aw-cons'+(max?' ow-float':'')} aria-label="控制台" ref={consoleRef}
      style={max?{left:'clamp(var(--ow-float-left, 8px), '+float.x+'px, calc(100% - var(--ow-float-width) - 8px))',
        '--ow-float-top':stageHeight ? stageHeight+8+'px' : undefined,
        '--ow-float-y':'clamp(var(--ow-float-top), '+float.y+'px, max(var(--ow-float-top), 45vh))'}:undefined}>
      {max&&<div className="ow-float-head">
        <div className="ow-grip" onPointerDown={moveFloatingPanel}><strong>控制台</strong><span>拖动标题栏移动</span></div>
        <label className="ow-layer-step">第 {layer} 层<input type="range" min="0" max={NL-1} value={layer}
          aria-label="全屏层" disabled={!dimensionsReady} onChange={event=>setLayer(+event.target.value)}/></label>
      </div>}
      <div className="aw-body"><div className="aw-right" ref={panelRef}>
        <div className="ow-console-nav">
          <div className="aw-tabs" role="tablist" aria-label="控制台功能">{tabSelectors.map(([id,title])=>
            <button key={id} type="button" role="tab" id={'console-tab-'+id} aria-controls={'console-panel-'+id}
              aria-selected={consoleTab===id} onClick={()=>selectConsoleTab(id)}>{id==='object'&&observeMode==='live'?'特征分析':title}</button>)}</div>
        </div>
        <div className="ow-console-layout">
        {consoleTab==='plugin'&&ops}
        <div className="ow-console-content">
        {visibleConsoleTabs.includes('run')&&<div id="console-panel-run" role="tabpanel" aria-labelledby="console-tab-run" hidden={consoleTab!=='run'}>
          {observeMode==='live'&&<section className="ow-mode-panel" aria-label="实时模式工作区">
            <div className="ow-section-head"><div><h3>实时模式</h3><p>提交文本进行一次本机前向，并在 3D 中查看采集结果。</p></div>
              <button className="fw-tbtn" disabled={gen.busy||!models.length} onClick={showExample}>回放中心实测样例</button></div>
            {exampleError&&<p role="alert">{exampleError}</p>}
            {runtimeControls}
            <div className="ow-step-list" role="group" aria-label="输入方式">
              {[['manual','自由输入'],['task','任务模式']].map(([id,label])=><button key={id} className="fw-tbtn" type="button"
                disabled={gen.busy||Boolean(liveCase&&id==='task')} aria-pressed={inputMode===id} onClick={()=>setInputMode(id)}>{label}</button>)}
            </div>
            {liveCase&&<div className="ow-step-list"><strong>归档原始输入 · {liveCase}</strong>
              <button className="fw-tbtn" type="button" disabled={gen.busy} onClick={releaseLiveCase}>清除归档输入</button></div>}
            {inputMode==='manual'&&<label className="ow-draft-input">运行文本<textarea rows={3} value={text} disabled={gen.busy} readOnly={Boolean(liveCase)}
              onChange={event=>editInput(event.target.value)} placeholder="输入要在本机模型上观察的文本"/></label>}
            <div className="ow-step-list">
              {inputMode==='manual'&&<button type="button" className="fw-tbtn primary" disabled={gen.busy||!text.trim()||Boolean(runReason)||(Boolean(liveCase)&&snapshot!==headExample?.snapshot)}
                onClick={generate}>{gen.busy?'正在采集…':'开始运行'}</button>}
              {showLayerStep}{tokenPicker}
            </div>
            {runReason&&<p className="v8-note">{runReason}</p>}
            {live.checked&&live.base==null&&<p className="v8-note">在项目目录运行 <code>.venv/Scripts/python.exe server/live_only_app.py</code> 后，点击「重新查找本机服务」。</p>}
            {gen.err&&<p role="alert">{gen.err}</p>}
            {inputMode==='task'&&<ObservationTask execute={executeText} observe={observeTask} preview={editInput}
              busy={gen.busy} disabled={runReason}/>}
            {exampleReplay&&<div className="ow-run-sample">
              <strong>Phase {exampleReplay.phase} · 中心归档样例</strong>
              <ExamplePositions sample={exampleReplay} site={exampleSite} onSite={changeExampleSite}/>
              <p>Token {exampleReplay.positions[exampleSite].index} · 逐层 HiddenState 与 B{exampleReplay.block} MLP</p>
              <p className="ow-inline-value">U{exampleUnit} 激活 <output>{exampleValue?.toFixed(4)}</output></p>
              <ExampleSource sample={exampleReplay}/>
              {gen.res?.snapshot===snapshot&&<button className="fw-tbtn" onClick={()=>{setExampleActive(false);setTab('result');}}>回到本次运行</button>}
            </div>}
            {!exampleReplay&&liveResult&&<>
              <div className="ow-run-nav"><span>本次轨迹 {liveResult.id} · {liveResult.elapsed} ms</span>
                <button type="button" className="fw-tbtn" onClick={()=>changeFieldView(true)}>查看本次原生读数</button>
                <span className="aw-seg" role="group" aria-label="实时结果">
                  <button type="button" aria-pressed={runTab==='result'} onClick={()=>setTab('result')}>预测结果</button>
                  <button type="button" aria-pressed={runTab!=='result'} onClick={()=>setTab('live')}>逐层变化</button>
                </span></div>{panel}
            </>}
          </section>}
          {consoleTab==='run'&&observeMode==='archive'&&<>
            <div className="ow-run-nav">
              <div className="aw-subtabs" role="tablist" aria-label="运行结果视图">
                {[['three','三线索对照'],['pair','冻结样本对'],...(traceId?[['trace','归档轨迹']]:[])].map(([id,title])=>
                  <button key={id} type="button" role="tab" aria-selected={id==='trace'?replay:!replay&&comparisonView===id}
                    onClick={()=>changeArchiveView(id)}>{title}</button>)}</div>
              <button className="fw-tbtn" disabled={!headExample} onClick={()=>openObservationArchive(headExample?.id,{scene:true})}>在 Head 中观察 B26/H36 →</button>
            </div>
            {replay?<section className="ow-mode-panel"><h3>归档轨迹</h3><p>{native?'轨迹 '+traceId:reason}</p>
              <div className="ow-step-list">{tokenPicker}{showLayerStep}</div>{panel}</section>:
              comparisonView==='three'?<Suspense fallback={<p role="status">正在读取归档对照…</p>}>
                {snapshot===headExample?.snapshot?<ThreeCueComparison/>:<p>三线索对照绑定 Qwen3-14B。上方可切换到对应模型。</p>}
              </Suspense>:<>
                <div className="aw-context-tools"><label>冻结样本对<select aria-label="条件对照样本" value={run.i}
                  disabled={!samples?.items} onChange={event=>chooseFrozenSample(+event.target.value)}>
                  {samples?.items?.map(row=><option key={row.i} value={row.i}>#{row.i} · {row.entity} · {row.cell.join('/')}</option>)}</select></label>{tokenPicker}</div>{panel}
              </>}
          </>}
        </div>}
        <div id="console-panel-plugin" role="tabpanel" aria-labelledby="console-tab-plugin" hidden={consoleTab!=='plugin'}>
          {consoleTab==='plugin'&&(matchingHeadRun&&pluginTab==='heads'?<ArchiveHead data={selection} caseId={headCase} model={activeModel?.model} fullscreen={max}
            onCase={changeHeadCase}/>:
            (archivedUnit || observeMode==='features') && pluginTab==='mlp' && layer===matchingExample?.block+1 && sel.unit!=null ?
              <section className="ow-mode-panel" aria-label="MLP 归档激活">
                <h3>B{layer-1}/U{sel.unit} 在四个位置的激活</h3>
                <p className="v8-note">Phase {matchingExample.phase} · {shortModel(activeModel.model)} · {matchingExample.caseId}</p>
                <table className="ow-value-table"><thead><tr><th>采集位置</th><th>Token</th><th>原生激活</th></tr></thead>
                  <tbody>{matchingExample.positions.map((entry,i)=><tr key={entry.site}><td>{EXAMPLE_SITES[i]}</td><td>{entry.index}</td>
                    <td>{matchingExample.field('mlp',layer,i)[sel.unit]}</td></tr>)}</tbody>
                </table><ExampleSource sample={matchingExample}/>
              </section>:
            bundleId&&DEFAULT_PLUGINS.includes(pluginTab)?<section className="ow-mode-panel" aria-label="原生归档来源">
              <h3>{activeArchive.title}</h3>
              <p>{activeArchive.coverage}</p>
              {fieldSourceError?<p role="status">{fieldSourceError}</p>:
                <p>{pluginTab==='hs'||pluginTab==='lens'?'此摘录未保存这项读数。':'在上方 3D 网格点选部件查看此归档的原值。'}</p>}
            </section>:
            exampleReplay&&DEFAULT_PLUGINS.includes(pluginTab)?<section className="ow-mode-panel">
              <h3>{pluginTab==='heads'?'Head':pluginTab==='mlp'?'MLP':pluginTab==='hs'?'HiddenState':'Logit Lens'} · 归档样例</h3>
              <ExamplePositions sample={exampleReplay} site={exampleSite} onSite={setExampleSite}/>
              {exampleField?<p className="ow-inline-value">{pluginTab==='mlp'?'原生单元 U':'原生坐标 '}{exampleIndex}
                <output>{exampleField[exampleIndex]?.toFixed(4)??'未采集'}</output></p>:
                <p>{pluginTab==='mlp'?'样例只采集了 B36 的 MLP 激活。':'该样例未保存此项读数。'}</p>}
            </section>:panel)}
          {consoleTab==='plugin'&&!activeArchive&&rawValues!=null&&<ResearchDisclosure
            key={[snapshot,layer,selected.module,selected.index,pos].join(':')} title="当前对象的完整响应与来源">
            <p>{posName(pos)} · 快照 {snapshot.slice(0,12)}</p>
            <pre className="v8-json">{JSON.stringify({
              snapshot,layer,module:selected.module,index:selected.index,position:pos,values:Array.from(rawValues),
            },null,2)}</pre>
          </ResearchDisclosure>}
        </div>
        {visibleConsoleTabs.includes('object')&&<div id="console-panel-object" role="tabpanel" aria-labelledby="console-tab-object" className="aw-object-host" hidden={consoleTab!=='object'}>
          {consoleTab==='object'&&<>
            {atlasReport}
            {observeMode==='live'&&liveResult&&<div className="ow-run-nav">
              <span>本次轨迹 {liveResult.id} · {posName(pos)}</span>
              <button type="button" className="fw-tbtn" onClick={()=>changeFieldView(true)}>查看本次原生读数</button>
            </div>}
            {!atlas.run&&<NeuronExample sample={matchingExample} kind={featureKind} onKind={setFeatureKind}
              unit={layer===matchingExample?.block+1?sel.unit:null} onLocate={locateExampleNeuron}
              onGroup={()=>{setFeatureMarks(true);locateExampleNeuron();}}/>}
            <ModelStructure key={snapshot||activeModel?.model||'missing'} detailOnly compact modelKey={activeModel?.model||modelKey}
              view={routeView==='parameters'?'parameters':'graph'} paramTab={paramTab} workspaceSelection={object} onSelection={locate} onSceneData={receiveScene}
              mlpAtlasRun={atlas.run?.run}
              responseContent={<p className="v8-note">{reason}</p>}/>
          </>}
        </div>}
        {visibleConsoleTabs.includes('research')&&<div id="console-panel-research" role="tabpanel" aria-labelledby="console-tab-research" hidden={consoleTab!=='research'}>
          <div className="aw-subtabs" role="tablist" aria-label="成果与方法分类">
            {[['library','成果库'],['records','测试记录'],['methods','方法与基准'],['history','历史对照']].map(([id,title])=>
              <button type="button" role="tab" key={id} aria-selected={recordTab===id} onClick={()=>selectRecordTab(id)}>{title}</button>)}</div>
          <div hidden={recordTab!=='records'}>{recordVisited.has('records')&&<Suspense fallback={<p>正在载入测试记录…</p>}>
            <TestRecords key={workspaceSearch?.serial||0} model={activeModel&&shortModel(activeModel.model)}
              workspaceSearch={workspaceSearch} onOpenNativeArchive={openObservationArchive} onReplay={openCaseReplay}/></Suspense>}</div>
          <div hidden={recordTab!=='library'}>{libraryVisited&&renderArchive?.({onLocate:locateLibraryEntry,onOverlay:setArchiveLayer})}</div>
          <div hidden={recordTab!=='methods'}>{recordVisited.has('methods')&&<>
            <div className="v6-toolbar aw-panel-toolbar"><label>研究工具<select aria-label="研究方法" value={methodId}
              onChange={event=>setMethodId(event.target.value)}>{methodItems.map(item=><option key={item.id} value={item.id}>{item.label}</option>)}</select></label></div>
            <div className="aw-method-content">{methodItems.find(item=>item.id===methodId)?.render()}</div></>}</div>
          <div className="aw-method-content" hidden={recordTab!=='history'}>{recordVisited.has('history')&&historyContent}</div>
        </div>}
        </div></div>
      </div></div>
    </section>
  </div>;
}
