/* 空间 · 模型探索工作台（Phase 346）。
   保留页面布局；3D 里显示完整 LLM 层塔，旁边放大选中层的 Head / HiddenState / MLP，
   点格子看具体数值；可全屏。下：控制台——输入文本运行、token 条、操作面板（全屏时变成浮窗，同一个面板）、插件面板。
   控制台保留四个任务页签；逐层变化与条件对照统一到运行页签，其他分析通过插件管理添加。
   数据：本机后端可读时，从 /api/traces/mainline/ma-h2/ 读 MA-H2 discovery 冻结样本的原生值（float16）；
   没有匹配的原生数据时，保持结构查询并说明缺失原因，不回退为其他模型的案例数据。 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SCENE_MOUSE_HINT } from '../scenePointer.js';
import Scene from './AssocLayerScene.jsx';
import { ASSOC_VIEWS as VIEWS } from './assocLayerGeometry.js';
import ModelStructure from './ModelStructure.jsx';
import { useSpatialObject, writeSpatialObject } from './spatialObjectState.js';
import { nativeGrid as grid, nativeTexture as tex, positiveDimension } from './nativeTexture.js';
import { publishResearchContext, useResearchContext } from '../app/researchContext.js';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import ObservationTask from './ObservationTask.jsx';


import { DESKTOP } from '../../config.js';
import { useApi } from '../app/rdc_fusion/api.js';
import { TestsPanel, TopTokens } from './AssocConsole.jsx';
import { liveSampler, runOnce, useLiveBase, useLiveModel, useLocalBase } from './liveRun.js';
import { request, useNative } from './MechanismTraceExplorer.jsx';
import { PageHead } from './CoreKit.jsx';
import { shortModel } from './v8Util.js';
import './assoc_workbench.css';

const POSN = { ans: '答案位置', qname: '问题中的实体名', scope: '范围词', tname: '资料里的实体名', tcol: '资料里的颜色', lastcol: '最后一条记录的颜色' };
const CN = { red: '红', yellow: '黄', green: '绿', blue: '蓝', white: '白', black: '黑', brown: '棕', orange: '橙', purple: '紫', pink: '粉', gray: '灰' };
// Functional plugins share the same model, object and run; activation never executes a model.
const WORKSPACE_PLUGINS = [
  { id: 'structure', name: '结构与档案', panel: 'object', desc: '查询完整结构、真实参数、特征与历史证据' },
  { id: 'live', name: '运行响应', panel: 'run', desc: '输入文本执行一次前向，或回放已有运行；执行需手动点击' },
  { id: 'response', name: '条件对照', panel: 'run', desc: '选择同快照样本对，核对读出位置并比较响应' },
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
    { id: 'attnviz', name: '注意力可视化', tag: 'todo', desc: '逐头的 token 对 token 注意力' },
    { id: 'sae', name: 'SAE 特征', tag: 'todo', desc: '稀疏自编码器特征与特征卡' },
    { id: 'tc', name: 'Transcoder 归因图', tag: 'have', desc: '特征到 logit 的归因图；项目 transcoder.v1（第 30 块）' },
    { id: 'probe', name: '线性探针', tag: 'have', desc: '逐层可读出检验；项目 analysis.v1' },
    { id: 'steer', name: '引导向量', tag: 'todo', desc: '在某层加方向，正负双向' },
  ]],
  ['检验', [{ id: 'tests', name: '三类检验', tag: 'on', desc: '查看冻结运行的响应预测、输出预测和原生机制检验报告，不自动执行新检验' }]],
];
const ALL = PLUGINS.flatMap(([, ps]) => ps);
const AVAILABLE = ALL.filter(p => p.tag === 'on');
const PENDING = ALL.filter(p => p.tag !== 'on');
const DEFAULT_PLUGINS = ['hs', 'mlp', 'heads', 'lens'];
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

/* ---------- 主组件 ---------- */
export default function AssocWorkbench({ mode, onMode, contextBar, children, workspaceMode = 'analysis', modelKey, onModel, routeView, paramTab, archiveView, renderArchive, methodItems = [], historyContent }) {
  const browsing = workspaceMode === 'browse';
  const consoleRef = useRef(null), pluginsRef = useRef(null), panelRef = useRef(null);
  const sceneRef = useRef(null);
  const context = useResearchContext(), object = useSpatialObject();
  const [sceneInfo, setSceneInfo] = useState(null);
  const receiveScene = useCallback(next => setSceneInfo(previous => previous?.keys?.length === next.keys.length && previous.keys.every((v, i) => v === next.keys[i]) ? previous : next), []);
  const changeObject = next => { writeSpatialObject(next); setPop(null); };
  const layer = object.layer ?? 24;
  const setLayer = value => { const next = typeof value === 'function' ? value(layer) : value; changeObject({ ...object, layer: next, ...(next === 0 || layer === 0 ? { module: '', index: null, coordinate: null } : {}) }); };
  const sel = { unit: object.module === 'mlp' ? object.index : null, head: object.module === 'attn' ? object.index : null, coord: object.module === 'hs' ? object.index : object.module === 'embedding' ? object.coordinate : null };
  const live = useLiveBase();                                   // 实时生成
  const lm = useLiveModel(live.base);                           // 本机模型的加载 / 卸载
  const nat = useLocalBase('/api/traces/mainline/ma-h2/samples'); // 冻结样本的原生数据（只在本机可读）
  const base = nat.base;
  const [samples, setSamples] = useState(null);
  const [run, setRun] = useState({ i: 0, other: null });
  const [fpos, setFpos] = useState(0);           // 冻结样本里的读出位置
  const [livePos, setLivePos] = useState(0);     // 实时运行里的 token 序号
  const [src, setSrc] = useState(() => context.trace || context.context_trace ? 'archive' : 'frozen');
  const traceId = context.trace || context.context_trace;
  const traceService = useLocalBase('/api/traces/catalog');
  const manifest = useNative(traceService.base, traceId ? `/api/traces/${traceId}/manifest` : null);
  const [archivePosition, setArchivePosition] = useState({});
  const requestedPosition = archivePosition.id === traceId ? archivePosition.position : Number(new URLSearchParams(window.location.search).get('tpos'));
  const archivePos = Number.isInteger(requestedPosition) && requestedPosition >= 0 && requestedPosition < (manifest.data?.tokens?.length || 0) ? requestedPosition : manifest.data?.readout_positions?.at(-1);
  const archivedRun = useMemo(() => manifest.data ? { id: manifest.data.trace_id, tokens: manifest.data.tokens, readout: manifest.data.readout_positions } : null, [manifest.data]);
  const [gen, setGen] = useState({ busy: false, res: null, err: '' });
  const [on, setOn] = useState(() => new Set([...DEFAULT_PLUGINS, ...(OPTIONAL_MODE_IDS.has(mode) ? [mode] : [])]));
  const [listed, setListed] = useState(() => new Set([...DEFAULT_PLUGINS, ...(OPTIONAL_MODE_IDS.has(mode) ? [mode] : [])]));
  const [tab, setTab] = useState(() => OPTIONAL_MODE_IDS.has(mode) ? mode : object.index != null ? 'object' : mode || 'object');
  const [consoleTab,setConsoleTab]=useState(()=>archiveView?'research':['langmeasure','relations'].includes(mode)?'plugin':browsing?'object':['live','response'].includes(mode)?'run':object.index!=null?'object':'run');
  const [recordTab,setRecordTab]=useState('library'),[libraryVisited,setLibraryVisited]=useState(Boolean(archiveView));
  const [recordVisited,setRecordVisited]=useState(()=>new Set()),[methodId,setMethodId]=useState(methodItems[0]?.id||'');
  useEffect(()=>{panelRef.current?.scrollTo({top:0});},[consoleTab,recordTab]);
  const [priorArchiveView,setPriorArchiveView]=useState(archiveView);
  if(priorArchiveView!==archiveView){setPriorArchiveView(archiveView);if(archiveView){setConsoleTab('research');setRecordTab('library');setLibraryVisited(true);}}
  const [archiveLayer, setArchiveLayer] = useState(null);
  const libraryVisible = consoleTab==='research'&&recordTab==='library';
  const [priorMode, setPriorMode] = useState(mode);
  if (priorMode !== mode) {
    setPriorMode(mode);
    if (mode) {
      setTab(current => OPTIONAL_MODE_IDS.has(mode) || current === mode ? mode : !MODE_IDS.has(current) && on.has(current) ? current : object.index != null ? 'object' : mode);
      if (OPTIONAL_MODE_IDS.has(mode)) {
        setOn(s => new Set([...s, mode])); setListed(s => new Set([...s, mode])); setConsoleTab('plugin');
      }
    }
  }
  const [view, setView] = useState('iso');
  const [max, setMax] = useState(false);
  const [pop, setPop] = useState(null);
  const [locator, setLocator] = useState(false);
  const [findKind, setFindKind] = useState('mlp');
  const [opt, setOpt] = useState({ hsNorm: 'global', hsClip: 0.99, mlpClip: 0.99, headTarget: 'follow', headMax: 0.5, diffAlign: 'named', showDiff: true, showDelta: false });
  const [hide, setHide] = useState(() => new Set());
  const [openOpt, setOpenOpt] = useState(() => new Set());
  const [cfg, setCfg] = useState(false);
  const [float, setFloat] = useState({ x: 16, y: 96 });
  const [text, setText] = useState('');
  const [inputMode,setInputMode]=useState('manual');
  const skel = useApi('/api/model/skeleton');
  const models = (skel.data?.models || []).filter(m => Number.isInteger(m.n_layers));
  const activeModel = context.snapshot ? models.find(m => m.model_snapshot === context.snapshot && (!modelKey || m.model === modelKey))
    : modelKey ? models.find(m => m.model === modelKey) : models.find(m => /qwen3-4b/i.test(m.model)) || models[0];
  const snapshot = context.snapshot || activeModel?.model_snapshot || '';
  const modelScene = sceneInfo?.model?.model === activeModel?.model && sceneInfo?.snapshot === snapshot ? sceneInfo : null;
  const localObject = modelScene?.layer === layer && modelScene?.module === object.module && modelScene?.index === object.index ? modelScene : null;
  const selectModel = m => {
    writeSpatialObject({ layer: Math.min(layer, m.n_layers), module: '', index: null });
    publishResearchContext({ snapshot: m.model_snapshot || '', run: '', trace: '', context_trace: '', pmodel: '' });
    onModel?.(m.model); setPop(null);
  };

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
  const replay = src === 'archive' && Boolean(traceId);
  const isLive = replay || src === 'live' && Boolean(gen.res);
  const activeRun = replay ? archivedRun : gen.res;
  const tokenPosition = replay ? archivePos : livePos;
  const capturedBundle = isLive && !replay && activeRun?.bundle?.position === tokenPosition ? activeRun.bundle : null;
  const LB = useBundle(replay ? traceService.base : activeRun?.base, isLive && !capturedBundle ? activeRun?.id : null, isLive ? tokenPosition : null);
  const bundleData = capturedBundle || LB.data;
  const saL = useMemo(() => liveSampler(bundleData, activeRun), [bundleData, activeRun]);
  const runData = isLive ? saL : saF;
  const sa = !browsing && activeModel && layer >= 0 && layer <= activeModel.n_layers && runData?.d.model_snapshot === snapshot && runData.NL === activeModel.n_layers + 1 && runData.DM === activeModel.d_model ? runData : null;
  const compatible = sa && sbF && sa.d.model_snapshot === sbF.d.model_snapshot && ['NL', 'DM', 'NF', 'NH', 'P'].every(key => sa[key] === sbF[key]);
  const sb = !isLive && compatible ? sbF : null;
  const native = Boolean(sa);
  const reason = browsing ? '图谱浏览展示结构与研究档案，切换运行分析查看具体样本响应。' : runData && runData.d.model_snapshot !== snapshot ? '当前运行与所选模型快照不匹配。请选匹配的运行或模型；未混用其他模型的激活。' : isLive ? ((replay && manifest.error) || LB.error || (replay && traceService.checked && traceService.base == null ? '未连接到保存此轨迹的节点。请连接原采集节点，或手动选择冻结样本。' : !sa ? '读取运行原生数据…' : ''))
    : !nat.checked ? '正在查找本机数据…' : samples?.error || A.error || (base == null ? (DESKTOP ? '没有找到提供冻结样本的本机后端（:5001 完整中心或 :5002 实时前向服务）：请确认本机服务已启动，原生全场在 AI2050_NATIVE_DIR 下' : '网页版不公开原生全场：原生数据保存在采集节点，桌面客户端或本机后端可用') : !sa ? '读取冻结样本…' : '');
  const positions = isLive ? activeRun?.tokens || [] : (samples?.positions || ['ans']);
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
  const diffOn = mode === 'response' && opt.showDiff && sb && (aligned || opt.diffAlign === 'named');
  const shown = id => on.has(id) && !hide.has(id);
  const headTarget = opt.headTarget === 'follow' || isLive ? pos : positions.indexOf(opt.headTarget);
  const scope = `${workspaceMode}:${snapshot}:${base}:${isLive ? activeRun?.id : run.i}:${sa?.d.model_snapshot || ''}:${pos}`;
  const visiblePop = pop?.scope === scope && pop.h.l === layer ? pop : null;

  /* 纹理 */
  const TX = useMemo(() => {
    const out = {};
    if (!dimensionsReady) return out;
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
    if (shown('mlp') && sa) { const ok = mlpOk, v = b => { const a = sa.mlp(b); if (!diffOn) return a; const c = sb.mlp(b); return a.map((x, i) => x - c[i]); };
      const vals = ok ? Array.from({ length: NB }, (_, b) => v(b)) : null, sc = ok ? pct(vals, opt.mlpClip) : 1;
      out.mlp = Array.from({ length: NB }, (_, b) => tex(mw, mh, d => { for (let i = 0; i < mw * mh; i++) put(d, i, i >= NF ? [246, 248, 251] : vals ? divc(vals[b][i], sc) : [214, 222, 232]); })); }
    if (shown('heads')) {
      const target = headTarget;
      out.heads = Array.from({ length: NB }, (_, b) => tex(8, Math.ceil(NH / 8), d => { for (let h = 0; h < NH; h++) {
        let val, rgb;
        if (sa) { val = sa.att(b, h)[sa.d.tokpos[target]]; rgb = diffOn ? divc(val - sb.att(b, h)[sb.d.tokpos[target]], opt.headMax * 0.8) : seqc(val, opt.headMax); }
        else { rgb = [214, 222, 232]; }       // 案例汇总：答案位置对资料颜色的注意力
        put(d, h, rgb); } }));
    }
    return out;
  }, [sa, sb, pos, on, hide, opt, diffOn, browsing, localObject, object.module, layer, NH, DM, NF, dimensionsReady]);   // eslint-disable-line react-hooks/exhaustive-deps

  const objs = [
    { id: 'llm', name: `${activeModel ? shortModel(activeModel.model) : '读取模型'} · ${NB} 层` },
    ...(shown('heads') && positiveDimension(NH) ? [{ id: 'heads', name: `Head · ${NH}${sa ? '' : '（结构）'}`, grid: [8, Math.ceil(NH / 8)], count: NH }] : []),
    ...(shown('hs') && positiveDimension(DM) ? [{ id: 'hs', name: `${browsing && layer === 0 ? '词嵌入' : 'HiddenState'} · ${DM}${browsing && object.module === 'embedding' && localObject?.pointValues ? '（词表参数）' : sa ? '' : '（结构）'}`, grid: [hw, hh], count: DM }] : []),
    ...(shown('mlp') && positiveDimension(NF) ? [{ id: 'mlp', name: `MLP · ${NF}${browsing && object.module === 'mlp' && localObject?.typeColors ? '（普查类型）' : sa && mlpOk ? '' : browsing ? '（档案）' : '（未采集）'}`, grid: [mw, mh], count: NF }] : []),
  ];

  const describe = h => {
    if (h.o === 'llm') return `第 ${h.l} 层 · ${h.l ? h.part : '词嵌入'}`;
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
    if (h.o === 'llm') { changeObject({ layer: h.l, module: h.l === 0 && browsing ? 'embedding' : '', index: null }); return; }
    if (h.o === 'hs' && browsing && h.l === 0) changeObject({ layer: 0, module: 'embedding', index: object.module === 'embedding' ? object.index ?? 0 : 0, coordinate: h.idx });
    else changeObject({ layer: h.l, module: h.o === 'heads' ? 'attn' : h.o, index: h.idx });
    setTab('object'); setConsoleTab('object'); setPop({ h, x, y, scope });
  };
  const selected = object.index != null ? object : null;
  const chooseFrozenSample = i => { const it=samples?.items.find(x=>x.i===i); if(!it)return;
    const twin=samples.items.find(x=>x.entity===it.entity&&x.cell[0]!==it.cell[0]&&x.cell.slice(1).join()===it.cell.slice(1).join());
    setRun({i,other:twin?.i??null});setSrc('frozen');
  };
  const validObject = selected && layer >= 0 && layer < NL && selected.index >= 0 && (selected.module === 'hs' ? selected.index < DM : selected.module === 'embedding' ? layer === 0 : layer > 0 && selected.index < (selected.module === 'mlp' ? NF : NH));
  const rawValues = !sa || !validObject ? null : selected.module === 'hs' ? sa.hs(pos, layer) : selected.module === 'mlp' && mlpOk ? sa.mlp(layer - 1) : selected.module === 'attn' ? sa.att(layer - 1, selected.index) : null;
  const sceneLinks = localObject?.links || [];
  const locate = location => {
    const o = location.module === 'mlp' ? 'mlp' : location.module === 'attn' ? 'heads' : 'hs';
    setOn(s => new Set([...s, o])); setHide(s => { const next = new Set(s); next.delete(o); return next; });
    changeObject(location); setTab('object'); setConsoleTab('object');
  };
  const locateArchive = entry => {
    const target = models.find(m=>m.model_snapshot===entry.snapshot);
    if(!target)return '本机尚无此快照的模型结构；请同步对应数据包。';
    publishResearchContext({snapshot:entry.snapshot,run:entry.run||'',trace:entry.trace||'',context_trace:'',pmodel:''});
    onModel?.(target.model); locate(entry.location);
    document.querySelector('.aw-space')?.scrollIntoView({block:'start',behavior:'smooth'});
    return '';
  };
  const objectEvidence = <button type="button" className="v6-linkbtn" onClick={() => { setPop(null); setTab('object'); setConsoleTab('object');requestAnimationFrame(()=>document.querySelector('.aw-object-host')?.scrollIntoView({ block: 'start', behavior: 'smooth' })); }}>查看参数、特征、证据与历史 ↓</button>;
  const openPlugin = id => {
    const target = id === 'delta' ? 'live' : id === 'diff' ? 'response' : id;
    const functional = WORKSPACE_PLUGINS.find(p => p.id === target);
    if (functional) {
      setPop(null); setTab(target === 'structure' ? 'object' : target);
      setConsoleTab(functional.panel); onMode(target); return;
    }
    if (!['live', 'response', 'result'].includes(target) && !AVAILABLE.some(p => p.id === target)) return;
    setTab(target);
    setConsoleTab(['live','response','result'].includes(target)?'run':'plugin');
    if (MODE_IDS.has(target)) onMode(target);
  };
  const toggle = id => {
    if (!AVAILABLE.some(p => p.id === id)) return;
    const next = new Set(on);
    if (next.has(id)) { next.delete(id); if(tab === id)setTab('object'); }
    else { next.add(id); openPlugin(id); }
    setOn(next);
  };
  const listPlugin = (id, checked) => {
    if (!AVAILABLE.some(p => p.id === id)) return;
    setListed(s => { const next = new Set(s); if (checked) next.add(id); else next.delete(id); return next; });
    setOn(s => { const next = new Set(s); if (checked) next.add(id); else next.delete(id); return next; });
    if (!checked && tab === id) setTab('object');
  };
  const resetPlugins = ids => {
    const next = new Set(ids.filter(id => AVAILABLE.some(p => p.id === id)));
    setListed(next); setOn(new Set(next)); setHide(new Set());
    if (!next.has(tab) && !['object', 'live', 'response', 'result'].includes(tab)) setTab('object');
  };
  const focusConsole = () => {
    requestAnimationFrame(()=>consoleRef.current?.scrollIntoView({block:'start',behavior:'smooth'}));
  };
  const selectSource = value => { setSrc(value); if(browsing)onMode('live'); };
  const selectConsoleTab = value => {setConsoleTab(value);setPop(null);if(value==='research'&&recordTab==='library')setLibraryVisited(true);};
  const showLibrary = () => {setConsoleTab('research');setRecordTab('library');setLibraryVisited(true);focusConsole();};
  const selectRecordTab = value => {setRecordTab(value);if(value==='library')setLibraryVisited(true);else setRecordVisited(s=>new Set([...s,value]));};

  /* 逐层播放 */
  useEffect(() => { const k = e => { if (e.key === 'Escape') setMax(false); }; window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k); }, []);

  const showRun = (r, input, condition) => {
    onMode('live'); setText(input); setConsoleTab('run'); setTab('result');
    setGen({ busy: false, res: r, condition, err: '' });
    setSrc('live'); setLivePos(r.bundle.position);
  };
  const observeTask = (r, input, condition, targetLayer) => {
    showRun(r, input, condition);
    if (targetLayer != null) setLayer(targetLayer);
    const scene = sceneRef.current;
    scene?.focus({ preventScroll: true });
    scene?.scrollIntoView({ block: 'start' });
  };
  async function executeText(input, condition) {
    onMode('live');
    setText(input);
    setConsoleTab('run');
    setGen(old => ({ ...old, busy: true, err: '' }));
    try {
      const r = await runOnce(live.base, input, snapshot);
      showRun(r, input, condition);
      setLayer(r.hidden.length - 1);
      return r;
    } catch (e) {
      setGen(old => ({ ...old, busy: false, err: condition ? '' : e.message }));
      throw e;
    }
  }
  const generate = () => executeText(text).catch(()=>{});

  /* 叠加：层间变化（LLM 两侧的条）与逐层读出（右缘） */
  const overlays = useCallback(({ ctx, proj, LY, llm, alpha }) => {
    if (browsing) return;
    if (opt.showDelta && sa) { const v = []; for (let l = 1; l < NL - 1; l++) v.push(norm(sa.hs(pos, l).map((x, i) => x - sa.hs(pos, l - 1)[i])));
      const mx = Math.max(...v) || 1; v.forEach((x, k) => { const l = k + 1, p = proj(llm.x - llm.w / 2, LY(l), llm.d / 2); ctx.globalAlpha = alpha(l); ctx.fillStyle = '#7050a0'; ctx.fillRect(p[0] - 40 - 60 * x / mx, p[1] - 2, 60 * x / mx, 3); ctx.globalAlpha = 1; }); }
    if (on.has('lens') && sa?.d.lens && !isLive) { const L = [sa.d.lens.map(r => r[0] - r[1]), sb ? sb.d.lens.map(r => r[0] - r[1]) : null];
      for (let l = 0; l < NL; l++) { const p = proj(llm.x + llm.w / 2, LY(l), -llm.d / 2); ctx.globalAlpha = alpha(l);
        ctx.fillStyle = '#2a78d6'; ctx.fillRect(p[0] + 8, p[1] - 4, Math.max(-60, Math.min(60, (L[0][l] || 0) * 4)), 3);
        if (L[1]) { ctx.fillStyle = '#d4622b'; ctx.fillRect(p[0] + 8, p[1], Math.max(-60, Math.min(60, (L[1][l] || 0) * 4)), 3); } ctx.globalAlpha = 1; } }
  }, [sa, sb, pos, on, opt.showDelta, NL, browsing]);   // eslint-disable-line react-hooks/exhaustive-deps

  const marks = selected ? [{ id: selected.module === 'attn' ? 'heads' : selected.module === 'embedding' ? 'hs' : selected.module, idx: selected.module === 'embedding' ? selected.coordinate : selected.index }].filter(m => m.idx != null) : [];
  const k = sa?.d, Bk = layer - 1;

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
  const ops = <aside ref={pluginsRef} className={`aw-ops${max ? ' float' : ' docked'}`} style={max ? { left: float.x, top: float.y } : undefined} aria-label="操作面板">
    <div className="aw-bar" onPointerDown={e => {     // 全屏时拖动标题栏移动浮窗
      if (!max || e.target.tagName === 'BUTTON') return;
      const s = { x0: e.clientX, y0: e.clientY, x: float.x, y: float.y };
      const move = ev => setFloat({ x: Math.max(0, s.x + ev.clientX - s.x0), y: Math.max(0, s.y + ev.clientY - s.y0) });
      const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up); }}>
      <b>插件系统</b>{max && <span className="v8-note">拖动标题栏移动</span>}
      <button type="button" className={cfg ? 'on' : ''} aria-expanded={cfg} title="选择在这里显示哪些插件" onClick={() => setCfg(c => !c)}>管理插件</button>
    </div>
    {cfg && <div className="aw-cfg" role="group" aria-label="插件管理">
      <div><h5>基础功能 · 常驻</h5><p className="v8-note">结构与档案、运行响应、条件对照共用当前对象；无需安装或启用，切换不会自动执行计算。</p></div>
      {PLUGINS.map(([g, ps]) => { const ready = ps.filter(p => p.tag === 'on'); return ready.length > 0 && <div key={g}><h5>{g}</h5>{ready.map(p => <label key={p.id} title={p.desc}><input type="checkbox" aria-label={`添加${p.name}`} checked={listed.has(p.id)} onChange={e => listPlugin(p.id, e.target.checked)} />{p.name}<small>{listed.has(p.id) ? '已添加' : '可添加'}</small></label>)}</div>; })}
      <div className="aw-acts"><button type="button" onClick={() => resetPlugins(DEFAULT_PLUGINS)}>恢复默认</button><button type="button" onClick={() => resetPlugins(AVAILABLE.map(p => p.id))}>添加全部可用</button><button type="button" onClick={() => setCfg(false)}>完成</button></div>
      <ResearchDisclosure title={`未接入插件（${PENDING.length}）`}><div className="aw-pending"><p className="v8-note">以下工具尚未接入本工作台，暂不能添加或执行。</p>{PENDING.map(p => <p key={p.id}><b>{p.name}</b><small>{p.tag === 'have' ? '项目已有实现，工作台待接入' : '待接入'}</small></p>)}</div></ResearchDisclosure>
    </div>}
    <div className="aw-in">
      {max&&<div className="aw-r"><span>层</span><input aria-label="全屏层" type="range" min="0" max={NL-1} value={layer} onChange={e=>setLayer(+e.target.value)}/></div>}
      <h4>功能插件</h4>
      <div className="aw-function-plugins" role="group" aria-label="功能插件">
        {WORKSPACE_PLUGINS.map(p => <button type="button" key={p.id} title={p.desc}
          aria-pressed={browsing ? p.id === 'structure' : mode === 'response' ? p.id === 'response' : p.id === 'live'}
          onClick={() => openPlugin(p.id)}><b>{p.name}</b><small>{p.desc}</small></button>)}
      </div>
      <h4>分析插件</h4>
      {AVAILABLE.filter(p => listed.has(p.id)).map(p => { const en = on.has(p.id), dis = p.native && !native && !browsing, vis = en && !hide.has(p.id);
        return <div key={p.id} className={`aw-pl${en ? '' : ' off'}`}>
          <div onClick={() => OPTS[p.id] && setOpenOpt(s => { const n = new Set(s); if (n.has(p.id)) n.delete(p.id); else n.add(p.id); return n; })}>
            <input type="checkbox" aria-label={`启用${p.name}`} checked={en} disabled={dis} title={dis ? reason : '启用插件'} onClick={e => e.stopPropagation()} onChange={() => toggle(p.id)} /><b>{p.name}</b>
            {en && <button type="button" className="aw-eye" aria-label={`查看${p.name}插件`} onClick={e=>{e.stopPropagation();openPlugin(p.id);}}>查看</button>}
            {en && ['hs', 'mlp', 'heads'].includes(p.id) && <button type="button" className="aw-eye" aria-pressed={vis} onClick={e => { e.stopPropagation(); setHide(s => { const n = new Set(s); if (n.has(p.id)) n.delete(p.id); else n.add(p.id); return n; }); }}>{vis ? '显示' : '隐藏'}</button>}
            {OPTS[p.id]&&<button type="button" className="aw-eye" aria-label={`${p.name}参数`} aria-expanded={openOpt.has(p.id)} onClick={e=>{e.stopPropagation();setOpenOpt(s=>{const n=new Set(s);if(n.has(p.id))n.delete(p.id);else n.add(p.id);return n;});}}>参数</button>}</div>
          {dis && <p className="v8-note aw-why">{reason}</p>}
          {OPTS[p.id] && openOpt.has(p.id) && en && <div className="aw-opt">{browsing?<p className="v8-note">这些参数用于运行响应；选择运行后可调整色标和对齐口径。</p>:OPTS[p.id]()}</div>}
        </div>; })}
      <p className="v8-note">已添加 {listed.size} / {AVAILABLE.length} 个插件；更多工具从“管理插件”添加。</p>
      {max && <p className="v8-note">Esc 或右上角按钮退出全屏。</p>}
    </div>
  </aside>;

  /* ---------- 插件面板 ---------- */
  const line = (series, mark, dp = 1) => { const W = 420, H = 130, all = series.flatMap(s => s.v.filter(Number.isFinite)); let lo = Math.min(0, ...all), hi = Math.max(...all); if (hi === lo) hi = lo + 1;
    const n = Math.max(...series.map(s => s.v.length)), X = i => 30 + (W - 38) * i / (n - 1), Y = v => 6 + (H - 22) * (1 - (v - lo) / (hi - lo));
    return <><svg viewBox={`0 0 ${W} ${H}`} className="aw-line" role="img" aria-label={series.map(s => s.name).join('、')}><line x1="30" x2={W - 8} y1={Y(0)} y2={Y(0)} stroke="#e2e7ef" />
      {mark != null && <line x1={X(mark)} x2={X(mark)} y1="4" y2={H - 16} stroke="#d29a00" strokeDasharray="3 3" />}
      {series.map(s => <path key={s.name} d={s.v.map((v, i) => Number.isFinite(v) ? `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}` : '').join('')} fill="none" stroke={s.color} strokeWidth="1.7" strokeDasharray={s.dash ? '4 3' : undefined} />)}
      <text x="0" y={Y(hi) + 4} fontSize="10" fill="#68748a">{hi.toFixed(dp)}</text><text x="0" y={Y(lo)} fontSize="10" fill="#68748a">{lo.toFixed(dp)}</text></svg>
      <div className="v8-note">{series.map(s => <span key={s.name} className="aw-key"><i style={{ background: s.color }} />{s.name}</span>)}</div></>; };
  const nameK = k ? `${CN[k.K] || k.K}` : '资料色', nameP = k ? `${CN[k.P] || k.P}` : '常识色';
  const needNative = <div><p className="v8-note">{reason}</p>{browsing&&<button type="button" className="fw-tbtn" onClick={()=>onMode('live')}>查看运行响应</button>}</div>;
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
    result: () => gen.res ? [<><h4>下一个词的预测 · {gen.res.tokens.length} 个 token · {gen.res.elapsed} ms</h4><TopTokens top={gen.res.top} /></>, <><h4>每层改写幅度（3D 主干上的金色圆点）</h4>{line([{ v: gen.res.curve, color: '#d29a00', name: '1 − cos(h_l, h_(l−1))，最后一个位置' }], layer, 2)}</>, '实时生成是一次前向的未登记观察，不产生研究结论。'] : [<p key="r" className="v8-note">还没有运行。</p>],
  };
  P.live = P.delta;
  P.response = P.diff;
  const pluginTabs = AVAILABLE.filter(p => listed.has(p.id) && on.has(p.id));
  const pluginTab=pluginTabs.some(p=>p.id===tab)?tab:pluginTabs[0]?.id;
  const runTab=['live','response','result'].includes(tab)?tab:mode==='response'?'response':'live';
  const cur=consoleTab==='plugin'?pluginTab:consoleTab==='run'?runTab:'object';
  const panel = !cur ? <p className="v8-note">从操作面板启用一个插件。</p>
    : cur === 'object' ? null : MODE_IDS.has(cur) && !P[cur] ? (mode === cur ? <div className="aw-mode">{contextBar}{children}</div> : <div className="aw-method-content"><button type="button" className="fw-tbtn" onClick={()=>openPlugin(cur)}>打开{MODE_PLUGINS.find(p => p.id === cur).name}</button></div>)
      : P[cur] ? (() => { const [a, b, bound] = P[cur](); return <div className="aw-pbody"><div>{a}</div><div>{b}</div>{bound && <div className="aw-bound">{bound}</div>}</div>; })()
        : (() => { const p = ALL.find(x => x.id === cur); return <div className="aw-pbody"><div><h4>{p.name}</h4><p>{p.desc}</p></div><div><p className="v8-note">{p.tag === 'have' ? '项目里已有实现，正式接成插件后读同一个观察对象，并在 3D 上画自己的图层。' : '待接入：按插件规范声明需要的数据、3D 图层与面板。'}</p></div></div>; })();

  const popBody = visiblePop && (() => { const { h } = visiblePop;
    if (h.o === 'hs' && sa && h.idx < DM) { const vals = Array.from({ length: NL }, (_, l) => sa.hs(pos, l)[h.idx]);
      return <><h4>HiddenState · 第 {h.l} 层 · 坐标 {h.idx}</h4><div className="aw-kv"><span>值</span><span><b>{vals[h.l].toFixed(4)}</b></span>{sb && <><span>对照运行</span><span>{sb.hs(pos, h.l)[h.idx].toFixed(4)}</span></>}<span>本层 ‖h‖</span><span>{norm(sa.hs(pos, h.l)).toFixed(1)}</span></div>
        {line([{ v: vals, color: '#2a78d6', name: '同编号坐标的跨层响应（非因果链）' }], h.l, 2)}<p className="v8-note">{h.l === NL - 1 ? '本层已过最终归一化。' : ''}相邻坐标：{[-2, -1, 1, 2].map(d => h.idx + d).filter(i => i >= 0 && i < DM).map(i => `${i}: ${sa.hs(pos, h.l)[i].toFixed(2)}`).join('　')}</p></>; }
    if (h.o === 'mlp' && sa && h.idx < NF) return !mlpOk ? <p className="v8-note">这个位置没有采集 MLP 激活。</p> : <><h4>MLP · 块 {h.l - 1} · 单元 {h.idx}</h4><div className="aw-kv"><span>A</span><span><b>{sa.mlp(h.l - 1)[h.idx].toFixed(4)}</b></span>{sb && <><span>对照运行</span><span>{sb.mlp(h.l - 1)[h.idx].toFixed(4)}</span></>}</div></>;
    if (h.o === 'heads' && h.idx < NH) return <><h4>注意力头 B{h.l - 1} · H{h.idx}</h4><p>{describe(h)}</p><p className="v8-note">{sa ? `查询 token ${sa.T - 1} → 目标 token ${sa.d.tokpos[headTarget]}（${posName(headTarget)}）` : '当前对象没有匹配的注意力采集值'}。注意力概率不是权重参数，也不代表因果重要性。</p></>;
    return <><h4>{describe(h)}</h4><p className="v8-note">{reason || '该位置没有原生采集值。'}</p></>; })();

  const modelControls = <div className="aw-models" role="group" aria-label="模型">{(models.length ? models : [{ model: 'qwen3-4b', n_layers: 36 }]).map(m => {
      const isMain = m.model === activeModel?.model;
      return <button type="button" key={m.model} aria-pressed={isMain} title={`选择 ${shortModel(m.model)} 的结构与研究档案`} onClick={() => selectModel(m)}>
        <b>{shortModel(m.model)}</b><small>{m.n_layers} 层{m.d_ff ? ` · MLP ${m.d_ff}` : ''}</small><span className={isMain ? 'ok' : 'warn'}>{isMain ? (browsing ? '当前模型 · 研究档案' : native ? '当前运行 · 原生数据' : '当前模型 · 无匹配运行') : '查看模型档案'}</span></button>; })}
      <div className="aw-load" aria-label="本机模型">{(() => {
        const s = lm.status, state = !live.checked ? 'checking' : !live.base ? 'none' : s?.status || 'checking';
        const label = { checking: '查找本机模型…', none: '没有本机模型服务', unavailable: '本机没有模型权重', idle: '未加载', loading: `加载中${s?.progress ? ` · ${s.progress}` : ''}…`, ready: `已加载${s?.vram_gb != null ? ` · 显存 ${s.vram_gb} GB` : ''}`, error: '加载出错' }[state] || state;
        return <>
          <span className="v8-note">本机模型</span>
          <b className={`aw-ls ${state}`}>{s?.model_name ? `${s.model_name} · ` : ''}{label}</b>
          <span className="aw-row">
            <button type="button" className="fw-tbtn primary" disabled={!live.base || lm.busy || !['idle', 'error'].includes(state)} title={live.base ? '把模型加载到 GPU（占用显存）' : '需要桌面节点或本机后端（:5001 / :5002）'} onClick={lm.load}>加载</button>
            <button type="button" className="fw-tbtn" disabled={!live.base || lm.busy || !['ready', 'loading'].includes(state)} title="卸载模型，释放 GPU 显存" onClick={lm.unload}>卸载</button>
          </span>
          {(lm.msg || s?.error || (state === 'unavailable' && s?.reason)) && <small className="v8-note">{lm.msg || s?.error || s?.reason}</small>}
        </>; })()}</div></div>;

  return <div className="v8-route aw" aria-label="统一空间工作台">
    <PageHead title="模型探索" sub="在同一个空间查询结构、参数、条件响应与研究证据。观察关联与因果检验分别标注。" />
    <div className="aw-workspace-mode"><span className="v8-note">当前功能：{browsing ? '结构与档案' : mode === 'response' ? '条件对照' : '运行响应'} · 在下方插件系统切换</span><div className="aw-mode-actions"><button type="button" className="v6-linkbtn" onClick={focusConsole}>控制台</button><button type="button" className="v6-linkbtn" aria-expanded={libraryVisible} onClick={showLibrary}>成果库</button><button type="button" className="v6-linkbtn" onClick={()=>pluginsRef.current?.scrollIntoView({block:'start',behavior:'smooth'})}>插件系统</button></div></div>
    {!dimensionsReady && <p className="v8-note" role="status">{skel.error ? `模型结构读取失败：${skel.error}` : '正在读取模型结构尺寸；尺寸就绪后显示单元网格。'}</p>}
    <section ref={sceneRef} tabIndex={-1} aria-label="模型运行 3D 场景" className={`aw-space${max ? ' max' : ''}`}>
      <Scene dims={{ NL, describe }} objs={objs} layer={layer} view={view} tex={TX} marks={marks} onPick={onPick} overlays={overlays} live={!browsing && sa && isLive && !replay ? gen.res : null} links={sceneLinks} archive={libraryVisible&&archiveLayer?.snapshot===snapshot?archiveLayer:null}/>
      <div className="aw-ov tr"><button type="button" aria-expanded={locator} onClick={() => { setLocator(v => !v); setPop(null); }}>精确定位</button><button type="button" onClick={() => { setMax(m => !m); setPop(null); }}>{max ? '退出全屏（Esc）' : '⛶ 全屏'}</button></div>
      {locator && <form className="aw-locator" aria-label="原生索引定位" onSubmit={event => {
        event.preventDefault(); const form = new FormData(event.currentTarget), l = +form.get('layer'), index = +form.get('index');
        if (!Number.isInteger(l) || !Number.isInteger(index) || l < (findKind === 'hs' ? 0 : 1) || l >= NL || index < 0 || index >= (findKind === 'hs' ? DM : findKind === 'mlp' ? NF : NH)) return;
        locate({ layer: l, module: findKind, index }); setLocator(false);
      }}>
        <label>层（L0 = 词嵌入）<input name="layer" type="number" min={findKind === 'hs' ? 0 : 1} max={NL - 1} defaultValue={Math.max(findKind === 'hs' ? 0 : 1, layer)} required/></label>
        <label>对象<select aria-label="定位对象类型" value={findKind} onChange={e => setFindKind(e.target.value)}><option value="mlp">MLP 单元</option><option value="attn">Head</option><option value="hs">HiddenState 坐标</option></select></label>
        <label>原生编号（从 0 开始）<input name="index" type="number" min="0" max={(findKind === 'hs' ? DM : findKind === 'mlp' ? NF : NH) - 1} defaultValue="0" required/></label>
        <button type="submit" className="fw-tbtn primary">定位单元</button><p className="v8-note">L1 对应块 B0。所有原生编号均可查询。</p>
      </form>}
      <div className="aw-big"><b>{layer ? `第 ${layer} 层` : '词嵌入'}</b><span className="v8-note">{browsing ? '结构与研究档案' : (replay ? '归档轨迹 · ' : isLive ? `${gen.condition ? `任务条件 ${gen.condition}` : '实时运行'} · ` : '冻结样本 · ') + posName(pos)}{k ? `（token ${k.tokpos[pos]}）· ‖h‖ ${norm(sa.hs(pos, layer)).toFixed(1)}` : ''}</span></div>
      <div className="aw-legend v8-note"><span><i style={{ background: '#e5def3' }} />注意力</span><span><i style={{ background: '#dcefe4' }} />MLP</span><span><i style={{ background: '#172033' }} />残差主干</span>
        {!browsing && (shown('hs') || shown('mlp')) && <span><i style={{ background: 'linear-gradient(90deg,#d4622b,#fff,#1f5fbf)' }} />{diffOn ? '差值：蓝 = 本运行大' : '蓝正橙负'}</span>}
        {!browsing && shown('heads') && <span><i style={{ background: 'linear-gradient(90deg,#fff,#1e5fbf)' }} />头越蓝，越看所选 token</span>}
        {browsing && localObject?.legend?.map(item => <span key={item.name}><i style={{ background: item.color }}/>{item.name}</span>)}
        {browsing && localObject?.pointValues && <span>蓝正橙负 · max|值| 色标</span>}
        <span>{sceneLinks.length > 0 && '紫虚线 = 留出共同响应 · 蓝线 = 登记关联 · '}{SCENE_MOUSE_HINT} · 点格子看数值 · 点层展开 · 灰色 = 未叠加数据 · 金虚线 = 局部放大</span></div>
      {visiblePop && <div className="aw-pop" role="dialog" aria-label="单元详情" style={{ left: `clamp(8px, ${visiblePop.x + 16}px, calc(100% - 344px))`, top: `clamp(64px, ${visiblePop.y + 10}px, calc(100% - 260px))` }}><button type="button" className="x" aria-label="关闭" onClick={() => setPop(null)}>×</button>{popBody}<p className="v8-note">{browsing ? `研究档案 · 快照 ${snapshot.slice(0, 12)}` : `${posName(pos)} · ${sa ? `快照 ${sa.d.model_snapshot.slice(0, 12)}` : '无匹配的原生数据'}`}</p>{objectEvidence}</div>}
      {max && ops}
    </section>
    <section className="v8-card aw-cons" aria-label="控制台" ref={consoleRef}>
      <div className="aw-chead"><h3>控制台</h3><span className="v8-note">{browsing ? '输入文本后执行，或选择已有运行；插件系统在下方直接使用' : '一次输入共用一个运行，所有分析读取同一对象'}</span></div>
      {modelControls}
      <div className="aw-task-switch aw-seg" role="group" aria-label="输入方式"><button type="button" disabled={gen.busy} aria-pressed={inputMode==='manual'} onClick={()=>setInputMode('manual')}>自由输入</button><button type="button" disabled={gen.busy} aria-pressed={inputMode==='task'} onClick={()=>{setInputMode('task');onMode('live');setConsoleTab('run');setTab('live');}}>任务模式</button></div>
      <div hidden={inputMode!=='task'}><ObservationTask execute={executeText} observe={observeTask} preview={setText} busy={gen.busy}
        disabled={!live.base?'未连接本机模型服务，请先连接。':!activeModel||!/qwen3-4b/i.test(activeModel.model)?'当前任务采集适配本机 Qwen3-4B，请先选择该模型。':lm.status?.status!=='ready'?'请先在上方加载本机模型，准备好后再运行。':''}/></div>
      <div className="aw-cin"><div><textarea aria-label="运行文本" readOnly={inputMode==='task'} value={text} onChange={e => setText(e.target.value)} placeholder="输入一段文字后执行；留空可查看当前已有样本，无需运行模型。“下一步”逐层查看。" />
        {mode==='response'&&<div className="aw-comparison" aria-label="选择可比运行"><label>冻结样本对 <select aria-label="条件对照样本" value={run.i} disabled={!samples?.items} onChange={e=>chooseFrozenSample(+e.target.value)}>{samples?.items?.map(row=><option key={row.i} value={row.i}>#{row.i} · {row.entity} · {row.cell.join('/')}</option>)}</select></label><button type="button" className="fw-tbtn" disabled={!samples?.items} onClick={()=>chooseFrozenSample(run.i)}>使用此样本对</button><p className="v8-note">{sb?`当前比较 #${run.i} / #${run.other}，模型快照一致；${aligned?'token 位置对齐':'按命名读出位置对齐，差值含前缀差异'}。`:isLive?'此运行尚无匹配对照，可选择上面的冻结样本对。':reason||'尚无合格对照，未生成差值。'}</p></div>}
        <p className="v8-note">{replay ? (native ? `正在回放归档轨迹 ${traceId.slice(0, 12)} · token ${pos} · ${positions.length} 个 token。` : reason) : isLive ? `现在显示的是本次实时运行（Qwen3-4B 真实前向，${gen.res.tokens.length} 个 token，用时 ${gen.res.elapsed} ms）：3D 与插件都读这次运行的原生值。` : native ? `现在显示的是冻结样本 #${run.i}：实体 ${k.entity}，资料里写的颜色 = ${nameK}，常识颜色 = ${nameP}。` : reason}</p>{gen.err && <p className="ac-err" role="alert">{gen.err}</p>}</div>
        <div className="aw-run">
          <button type="button" className="fw-tbtn primary" hidden={inputMode==='task'} disabled={gen.busy || (text.trim() && (!live.base || live.status?.available === false || !activeModel || !/qwen3-4b/i.test(activeModel.model)))}
            title={text.trim() ? (live.base ? '用本机模型运行这段文字' : '没有连接到可用的本机模型：运行需要桌面节点或本机后端') : '显示当前样本的完整运行（到最后一层）'}
            onClick={() => { if (text.trim()) generate(); else {onMode('live');setConsoleTab('run');setTab('live');setLayer(NL - 1);} }}>{gen.busy ? '执行中…' : '执行'}</button>
          <button type="button" className="fw-tbtn" title="前进一层；到最后一层后从词嵌入重新开始" onClick={() => setLayer(l => (l >= NL - 1 ? 0 : l + 1))}>下一步</button>
          <span className="v8-note">{layer ? `第 ${layer} / ${NL - 1} 层` : `词嵌入 / ${NL - 1} 层`}</span>
          <span className="aw-seg aw-src" role="group" aria-label="数据来源"><button type="button" aria-pressed={!isLive} onClick={() => selectSource('frozen')}>冻结样本</button>
            <button type="button" aria-pressed={isLive && !replay} disabled={!gen.res} title={gen.res ? '本次实时运行' : '先用“执行”运行一段文字'} onClick={() => selectSource('live')}>实时运行</button>{traceId && <button type="button" aria-pressed={replay} onClick={() => selectSource('archive')}>归档轨迹</button>}</span>
        </div></div>
      <div className="aw-context-tools" aria-label="公共定位与样本">
        {!isLive&&mode!=='response'&&<label>样本<select aria-label="当前样本" value={run.i} disabled={!samples?.items} onChange={e=>{chooseFrozenSample(+e.target.value);if(browsing)onMode('live');}}>{samples?.items?.map(row=><option key={row.i} value={row.i}>#{row.i} · {row.entity} · {row.cell.join('/')}</option>)}</select></label>}
        <label>Token<select aria-label="当前Token" value={pos??''} disabled={browsing||!native} onChange={e=>setPos(+e.target.value)}>{positions.map((p,i)=><option key={i} value={i}>{isLive?`${i} · ${tokText(p)}`:`${POSN[p]||p}${k?`（${k.tokpos[i]}）`:''}`}</option>)}</select></label>
        <label>层<input aria-label="当前层" type="number" min="0" max={NL-1} value={layer} onChange={e=>{const v=+e.target.value;if(Number.isInteger(v)&&v>=0&&v<NL)setLayer(v);}}/></label>
        <label>视角<select aria-label="空间视角" value={view} onChange={e=>setView(e.target.value)}>{VORDER.map(v=><option key={v} value={v}>{VIEWS[v][2]}</option>)}</select></label>
      </div>
      <div className="aw-body">{!max && ops}<div className="aw-right" ref={panelRef}>
        <div className="aw-tabs" role="tablist" aria-label="控制台功能">{[['object','对象详情'],['run','运行与对照'],['plugin','插件分析'],['research','成果与方法']].map(([id,name])=><button type="button" role="tab" key={id} id={`console-tab-${id}`} aria-controls={`console-panel-${id}`} aria-selected={consoleTab===id} onClick={()=>selectConsoleTab(id)}>{name}</button>)}</div>
        <div role="tabpanel" id="console-panel-run" aria-labelledby="console-tab-run" hidden={consoleTab!=='run'}>
          <div className="aw-subtabs" role="tablist" aria-label="运行结果视图">{[['live','逐层变化'],['response','对照结果'],['result','生成结果']].filter(([id])=>id!=='result'||gen.res).map(([id,name])=><button type="button" role="tab" key={id} aria-selected={runTab===id} onClick={()=>openPlugin(id)}>{name}</button>)}</div>
          {consoleTab==='run'&&panel}
          {runTab === 'live' && <ResearchDisclosure title="逐层变化显示设置"><label className="aw-run-setting"><input type="checkbox" checked={opt.showDelta} onChange={e => setOpt(s => ({...s, showDelta: e.target.checked}))}/>在3D空间显示层间变化</label></ResearchDisclosure>}
          {runTab === 'response' && <ResearchDisclosure title="对照显示与对齐设置"><div className="aw-opt"><label className="aw-run-setting"><input type="checkbox" checked={opt.showDiff} onChange={e => setOpt(s => ({...s, showDiff: e.target.checked}))}/>在3D空间显示差值</label>{OPTS.diff()}</div></ResearchDisclosure>}
          <ResearchDisclosure title="完整Token序列与读出位置">
      {!browsing && (isLive ? <div className="aw-toks live"><span className="v8-note">{replay ? '归档输入的' : '本次输入的'} {positions.length} 个 token（点一个看它的各层；带下划线的是模块读出位置，有 MLP 与写入拆分）</span>
        {positions.map((t, i) => <button type="button" key={i} className={`${i === pos ? 'on' : ''}${activeRun?.readout?.includes(i) ? ' ro' : ''}`} title={`token ${i}`} onClick={() => setPos(i)}>{tokText(t)}</button>)}</div>
        : k && <div className="aw-toks"><span className="v8-note">token（{k.ntok} 个；有名字的是读出位置）</span>{Array.from({ length: k.ntok }, (_, t) => { const i = k.tokpos.indexOf(t);
        return i >= 0 ? <button type="button" key={t} className={i === pos ? 'on' : ''} title={POSN[positions[i]]} onClick={() => setPos(i)}>{t}</button> : <span key={t} title={`token ${t}`}>·</span>; })}</div>)}
          </ResearchDisclosure>
        </div>
        <div role="tabpanel" id="console-panel-plugin" aria-labelledby="console-tab-plugin" hidden={consoleTab!=='plugin'}>
          <div className="v6-toolbar aw-panel-toolbar"><label>分析插件<select aria-label="当前分析插件" disabled={!pluginTabs.length} value={pluginTab||''} onChange={e=>openPlugin(e.target.value)}>{!pluginTabs.length && <option value="">暂无启用插件</option>}{pluginTabs.map(p=><option key={p.id} value={p.id}>{p.name}</option>)}</select></label><span className="v8-note">点击左侧“查看”切换；更多工具从“管理插件”添加。</span></div>
          {consoleTab==='plugin'&&panel}
        </div>
        <div role="tabpanel" id="console-panel-object" aria-labelledby="console-tab-object" className="aw-object-host" hidden={consoleTab!=='object'}><ModelStructure key={activeModel?.model || 'missing'} detailOnly compact modelKey={activeModel?.model || modelKey}
        view={routeView === 'parameters' ? 'parameters' : 'graph'} paramTab={paramTab} workspaceSelection={object} onSelection={locate} onSceneData={receiveScene}
        responseContent={sa && validObject ? <div><p>{posName(pos)} · 快照 {snapshot.slice(0, 12)}</p>{selected.module === 'embedding' ? <p>这是词表参数行；当前运行的输入状态请在 HiddenState 中查看。</p> : <p>{describe({ o: selected.module === 'attn' ? 'heads' : selected.module, l: layer, idx: selected.index })}</p>}
          {rawValues && <ResearchDisclosure title="完整响应数组与来源"><p className="v8-note">{isLive ? `轨迹 ${activeRun?.id}` : `MA-H2 冻结样本 #${run.i}`} · {rawValues.length} 个原生值；数组按原生索引排序，不表示因果贡献。</p><pre className="v8-json">{JSON.stringify({snapshot,layer,module:selected.module,index:selected.index,position:pos,values:Array.from(rawValues)},null,2)}</pre></ResearchDisclosure>}
        </div> : <p className="v8-note">{reason}</p>}/></div>
        <div role="tabpanel" id="console-panel-research" aria-labelledby="console-tab-research" hidden={consoleTab!=='research'}>
          <div className="aw-subtabs" role="tablist" aria-label="成果与方法分类">{[['library','成果库'],['methods','方法与基准'],['history','历史对照']].map(([id,name])=><button type="button" role="tab" key={id} aria-selected={recordTab===id} onClick={()=>selectRecordTab(id)}>{name}</button>)}</div>
          <div hidden={recordTab!=='library'}>{libraryVisited&&renderArchive?.({onLocate:locateArchive,onOverlay:setArchiveLayer})}</div>
          <div hidden={recordTab!=='methods'}>{recordVisited.has('methods')&&<><div className="v6-toolbar aw-panel-toolbar"><label>研究工具<select aria-label="研究方法" value={methodId} onChange={e=>setMethodId(e.target.value)}>{methodItems.map(item=><option key={item.id} value={item.id}>{item.label}</option>)}</select></label></div><div className="aw-method-content">{methodItems.find(item=>item.id===methodId)?.render()}</div></>}</div>
          <div className="aw-method-content" hidden={recordTab!=='history'}>{recordVisited.has('history')&&historyContent}</div>
        </div>
      </div></div>
      {!browsing&&<p className="v8-note aw-foot">运行响应是观察；参数来源、统计关联与干预证据分别回查。</p>}
    </section>
  </div>;
}
