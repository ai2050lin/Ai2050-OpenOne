/* 研究工作台 · 空间：需要层结构的 tab（神经元、语言图谱、响应点云）左侧显示 LLM 多层结构（3D 层塔），右侧按 tab 显示内容。
   层塔共用，含义属于 tab——每个 tab 自己决定「颜色是什么、标记是什么、点击做什么」，点击层塔从不切换 tab；
   跨 tab 只走显式按钮（如「在神经元中查看 →」）。各 tab 的选中层分开记录（block / lblock / rlayer）。
     神经元   颜色 = 神经元叠加量（选择性 / 已解释 / T1 / 强激活率）；点击层 = 结束「只看结构」、展开右侧详情；
              每层板面另有 Attn（蓝，后半）/ MLP（橙，前半）两个模块盒，点盒 = 选中 (块, 模块)
     语言图谱 颜色 = 当前语言项在各块的有效神经元（模板选择性单元数）；绿色边 = 未见词表上可读出（路线 T），粗边 = 峰值层；
              点击 = 在文章上方打开「该层」面板，并在 C / E 节高亮该层
     响应点云 颜色 = 所选模式族的逐层留出投影间隔；绿色边 = 可读出；点击 = 点云切到该层
     谱图关系、本地包：不显示层塔（谱图关系视图自带层选择与 3D 谱图；本地包在打开研究包之前没有模型数据），内容占满宽度
   层序号统一用 hs：hs[0] = 嵌入层，块 b 的输出 = hs[b+1]。全部读取来自本机数据包（只读数据包模式）。
   渐进披露：神经元 tab 默认只画结构（无选中层、不渲染详情），点一层或一个模块后才展开右栏——
   默认界面回答「模型长什么样」，操作之后才回答「这一层里有什么」。 */
import { Suspense, lazy, useEffect, useMemo, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import LangPage from './LangPage.jsx';
import LangAtlas from './LangAtlas.jsx';
import LivePane from './MechanismTraceExplorer.jsx';
import ResearchSpace from './ResearchSpace.jsx';
import NeuronPane from './NeuronPane.jsx';
import ParameterExplorer from './ParameterExplorer.jsx';
import ModelStructure from './ModelStructure.jsx';
import Tower from './Tower.jsx';
import { OVERLAYS, READOUT_LABEL, defaultModel, fmtP, hsLabel, pickItem, pickPiece, readState, shortModel, towerValues, writeState } from './v8Util.js';
import './v8.css';
import LanguageFrame from './LanguageFrame.jsx';
import { AssocIntro, EncodingLadder } from './SpaceGuides.jsx';
import AssocSpace from './AssocSpace.jsx';
import { BasisCompare, RelationBoards } from './ClientGuide.jsx';
import { PageHead, PageTabs, PillTools } from './CoreKit.jsx';

const ResponseExplorer = lazy(() => import('../app/rdc_fusion/ResponseExplorer.jsx'));
const AtlasView = lazy(() => import('../app/rdc_fusion/AtlasView.jsx'));
const HistoricalActivations=lazy(()=>import('../app/rdc_fusion/NeuronSpace.jsx'));
const HistoricalFidelity=lazy(()=>import('../app/rdc_fusion/FidelityMode.jsx'));

function HistoricalPanel({kind}){
  const [open,setOpen]=useState(false);
  return <details className="v8-card" open={open} onToggle={e=>setOpen(e.currentTarget.open)}><summary>{kind==='activation'?'历史封存激活（独立数据源）':'历史图谱预测保真度（封存口径）'}</summary><p className="v8-note">此面板保留原始实验与模型口径，不自动绑定上方当前权重，也不将历史指标当作整体破解率。</p>{open&&<Suspense fallback={<Empty>读取历史实测…</Empty>}>{kind==='activation'?<HistoricalActivations/>:<HistoricalFidelity/>}</Suspense>}</details>;
}

const hsOf = tag => { const m = /@L(\d+)$/.exec(tag || ''); return m ? Number(m[1]) : null; };
const toHs = (blockValues, n) => (blockValues ? [null, ...Array.from({ length: n }, (_, b) => blockValues[b] ?? null)] : null);

/* 层内模块的开关对象：引用稳定，避免每次渲染都重绘层塔 */
const MODS = { attn: 1, mlp: 1 };

function itemSelectivity(item, census, st) {
  const runs = (item?.selectivity || []).filter(s => s.census_key === census);
  const run = runs.find(s => s.file === st.sel) || runs[0];
  if (!run) return { run: null, ro: null, values: null };
  const ros = Object.keys(run.by_readout || {});
  const ro = ros.includes(st.ro) ? st.ro : ros[0];
  return { run, ro, values: run.by_readout[ro] || null };
}

/* 响应点云：所选模式族在该模型上的代表拼图（等级最高，同等级时峰值间隔最大） */
function familyRow(rows, tm) {
  const fam = String(tm || '').replace(/^TG-/, '');
  const own = rows.filter(r => r.family_id === fam);
  return [...own].sort((a, b) => b.level - a.level || ((b.peak?.sep || 0) - (a.peak?.sep || 0)))[0] || null;
}

function curveHs(row, n) {
  if (!row) return { values: null, marks: [], peak: null };
  const values = Array.from({ length: n + 1 }, () => null);
  const marks = [];
  row.layers.forEach((l, k) => { if (l <= n) { values[l] = row.sep[k]; if (row.readable[k]) marks.push(l); } });
  return { values, marks, peak: row.peak?.layer ?? null };
}

/* 关联机制的四种模式（Phase 336）：合并原来的四个子页签。路由不变，旧链接照常打开对应模式。 */
const ASSOC_MODES = [
  ['live', '运行与路径', '一段文字在全部层里怎样流动、怎样决定下一个词'],
  ['response', '响应对照', '只改句子里的一处，看模型内部哪里跟着变'],
  ['langmeasure', '语言项实测', '一项语言能力在哪一层读得出，按数据集批量测'],
  ['relations', '共享关系', '不同语言能力是否共用同一个读出方向'],
];
function assocModeOf(view, graphMode) {
  if (view === 'live' || view === 'response') return view;
  return view === 'graph' && ['langmeasure', 'relations'].includes(graphMode) ? graphMode : null;
}
function AssocHub({ mode, onMode, contextBar, children }) {
  return <div className="v8-route as-page" aria-label="关联机制">
    <PageHead title="关联机制" sub="把语言图谱里的能力，对到模型内部的层、注意力头和神经元。" />
    <PageTabs label="关联机制模式" items={ASSOC_MODES} value={mode} onChange={onMode} />
    {contextBar}
    {children}
  </div>;
}

/* LLM 编码图谱：结构与参数、全部研发成果 */
const ENC_MODES = [
  ['structure', '结构与参数', '词嵌入、36 层的注意力头和 MLP 单元；点一层看它的 Head 与 MLP，再看真实参数与证据'],
  ['internal', '全部研发成果', '全部研究对象的 3D 总图，按证据来源回查'],
];
function EncHub({ mode, onMode, children }) {
  return <div className="v8-route as-page" aria-label="LLM 编码图谱">
    <PageHead title="LLM 编码图谱" sub="模型里有什么：词嵌入、注意力头和 MLP 单元，以及它们各自的证据。" />
    <PageTabs label="LLM 编码图谱视图" items={ENC_MODES} value={mode} onChange={onMode} />
    {children}
  </div>;
}

export default function SpaceLens(props) {
  const assocMode = assocModeOf(props.view, props.graphMode);
  if (assocMode && !props.assocInner) {
    const onMode = id => (id === 'live' || id === 'response' ? props.setView?.(id) : props.onGraphMode?.(id));
    return <AssocHub mode={assocMode} onMode={onMode} contextBar={props.contextBar}><SpaceLens {...props} assocInner /></AssocHub>;
  }
  const encMode = props.view === 'parameters' || (props.view === 'graph' && (!props.graphMode || ['structure', 'internal'].includes(props.graphMode)))
    ? (props.view === 'graph' && props.graphMode === 'internal' ? 'internal' : 'structure') : null;
  if (encMode && !props.assocInner) {
    return <EncHub mode={encMode} onMode={id => props.onGraphMode?.(id)}><SpaceLens {...props} assocInner /></EncHub>;
  }
  if (props.view !== 'graph' || !props.graphMode || props.graphMode === 'structure') {
    /* 编码图谱（结构、参数）先给证据阶梯；关联机制（运行、对照）先说明三种运行方式的边界 */
    const structure = <ModelStructure modelKey={props.model} onModel={props.onModel} view={props.view} paramTab={props.paramTab}
      onGraphMode={props.onGraphMode} tm={props.tm} onTm={props.setTm} compact
      historical={<HistoricalPanel kind={props.view==='response'?'fidelity':'activation'}/>}/>;
    /* 关联机制 · 运行与路径：首屏只有 3D 空间，运行一段文字等功能收进操作台（Phase 332） */
    if (props.view === 'live') return <AssocSpace model={props.model} bare>{structure}</AssocSpace>;
    /* 其余结构页与响应对照：首屏只有核心模块（3D 层塔 / 最小对照），说明与看板收进按钮（Phase 333） */
    const tools = props.view === 'response' ? [
      { id: 'modes', label: '三种运行的区别', render: () => <AssocIntro/> },
      { id: 'map', label: '语言目录映射', render: () => <RelationBoards only="map"/> },
      { id: 'fidelity', label: '保真看板', render: () => <RelationBoards only="fidelity"/> },
      { id: 'bench', label: '方法基准', render: () => <RelationBoards only="bench"/> },
      { id: 'public', label: '公开基准', render: () => <RelationBoards only="public"/> },
    ] : [
      { id: 'ladder', label: '证据阶梯', render: () => <EncodingLadder open/> },
      { id: 'basis', label: '特征基底对比', render: () => <BasisCompare/> },
    ];
    return <>{structure}<PillTools label="更多内容" items={tools}/></>;
  }
  return <SpaceDetailLens {...props} />;
}

function SpaceDetailLens({ view: routeView, graphMode='language', paramTab='weights',onParamTab, setView, tm, setTm, localPack, model: gModel, onModel, ds = '', onDs }) {
  /* language = 语言图谱（只有外部能力框架）；langmeasure = 关联机制里的语言项实测（原语言图谱页：层塔、读出、神经元、数据集实测） */
  const view=routeView==='graph'?({language:'langframe',langmeasure:'lang',internal:'research',relations:'relations'}[graphMode]||'langframe'):routeView==='parameters'&&paramTab==='neurons'?'neuron':routeView;
  /* 图层切换已由工作台的三张图和视图条承担，这里不再重复一排图层按钮 */
  const modeBar=routeView==='graph'?null:routeView==='parameters'?<section className="v8-card wb-object-modes"><strong>结构与参数</strong><div className="v6-toolbar" role="group" aria-label="参数与原生特征">{[['weights','真实参数'],['neurons','原生单元与特征卡']].map(([id,label])=><button type="button" className={'fw-tbtn'+(paramTab===id?' primary':'')} aria-pressed={paramTab===id} key={id} onClick={()=>onParamTab?.(id)}>{label}</button>)}</div></section>:null;
  const treeApi = useApi('/api/lang/tree');
  const skApi = useApi('/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = useMemo(() => Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s])), [skApi.data]);
  const [st, setSt] = useState(readState);
  /* 右栏常驻展开：块细节是这一页的主体，不做「展开 / 收起」。
     层塔与细节必须同时可见——选中块在塔上是高亮的，右栏讲的就是那一块。 */
  useEffect(() => { writeState(st); }, [st]);
  const patchSt = patch => setSt(prev => ({ ...prev, ...patch }));

  // 实时分析：一次前向的观察数据与步进位置（0 = 嵌入层，1..L = 块，L+1 = 输出头）；数据留在会话内，不进地址栏
  const [liveTrace, setLiveTrace] = useState(null);
  const [liveStep, setLiveStep] = useState(0);

  /* 模型来源优先级：全局条（mdl）> 本页旧链接（model）> 自动挑一个普查最全的。
     全局条是权威：页顶换模型，空间里跟着换，不再出现「两处各选一个」的矛盾。 */
  const ownModel = st.model && tree?.models?.includes(st.model) ? st.model : '';
  const model = (gModel && tree?.models?.includes(gModel) ? gModel : '') || ownModel || defaultModel(tree, skeletons);
  const skeleton = skeletons[model];
  const census = skeleton?.census_run ? skeleton.census_key : null;
  /* 结构参数（三块：词嵌入 / 注意力 / MLP）：形状与参数量来自模型自己的 config.json，
     层塔按它把模块盒画成有体积的实体。取不到就退回平面色带，不猜数字。 */
  const archApi = useApi(model ? `/api/model/arch?model_key=${encodeURIComponent(model)}` : null);
  const arch = archApi.data && archApi.data.available ? archApi.data : null;
  const tw = useApi(census ? `/api/neurons/tower?model_key=${encodeURIComponent(census)}` : null);
  const tower = tw.data && !tw.data.pending ? tw.data : null;
  const nLayersCensus = tower?.n_layers || skeleton?.n_layers || 0;
  // 实时分析不依赖普查骨架：有 trace 就按 trace 的层数画层塔（本机模型 36 块）
  const nLayers = view === 'live' ? (liveTrace?.n_layers || 0) : nLayersCensus;
  const block = st.block != null && st.block < nLayers ? st.block : Math.min(18, Math.max(0, nLayersCensus - 1));

  // 各 tab 需要的数据（全部来自本机数据包；不在当前 tab 时不读取）
  const itemId = tree ? pickItem(tree, st, model) : null;
  const itemApi = useApi(view === 'lang' && itemId ? `/api/lang/item/${itemId}` : null);
  const ovApi = useApi(view === 'response' && model ? `/api/space/overlay?model_key=${encodeURIComponent(model)}` : null);
  const rows = ovApi.data?.route_T || [];
  const respTm = tm || (rows.length ? `TG-${[...new Set(rows.map(r => r.family_id))].sort()[0]}` : '');
  const respRow = view === 'response' ? familyRow(rows, respTm) : null;

  /* 面板里的显式跳转（page: lang / neuron / response）换算成空间 tab；其余是本 tab 的状态 */
  const set = patch => {
    const { page, tm: nextTm, ...rest } = patch;
    if (Object.keys(rest).length) patchSt(rest);
    if(page==='neuron'&&Number.isInteger(rest.block)){
      const url=new URL(window.location.href);
      url.searchParams.set('sblock',String(rest.block+1));url.searchParams.set('smodule','mlp');
      if(Number.isInteger(rest.unit))url.searchParams.set('sindex',String(rest.unit));else url.searchParams.delete('sindex');
      for(const key of ['smode','scoord','prow','pcol'])url.searchParams.delete(key);
      window.history.replaceState(window.history.state,'',url);
    }
    if (page === 'neuron' || page === 'lang') setView(page);
    if (page === 'response') { setTm(nextTm || ''); setView('response'); }
    if (page === 'relations') setView('relations');   // 谱图关系：语言图谱内的入口（不再是一级 tab）
  };

  /* 每个 tab 的层塔配置 */
  const cfg = useMemo(() => {
    const n = nLayers;
    if (view === 'neuron') {
      /* 点层或点模块都算「开始看这一层」：右栏展开。点层复位到 MLP（默认模块），点模块保留模块 */
      const pick = (hs, mod) => patchSt({ block: hs - 1, unit: null, ...(mod ? { mod } : {}) });
      return { values: tower ? toHs(towerValues(tower, st.ov, st.sel, st.ro), n) : null,
               selected: block + 1, allowEmbedding: false,
               onPick: hs => pick(hs, null),
               onPickModule: (hs, mod) => pick(hs, mod),
               selModule: { hs: block + 1, mod: st.mod },
               modules: MODS,
               valueLabel: (OVERLAYS.find(o => o[0] === st.ov) || [])[1] || '叠加量',
               color: `颜色 = ${(OVERLAYS.find(o => o[0] === st.ov) || [])[1] || '无'}（每块）`,
               click: '点击一层 = 展开该块；点板面上的 Attention（后半，蓝）/ MLP（前半，橙）盒 = 只看该模块' };
    }
    if (view === 'lang') {
      const d = itemApi.data;
      const sel = itemSelectivity(d, census, st);
      const piece = d && !d.pending ? pickPiece(d.pieces || [], itemId, model, st.piece) : null;
      const ro = piece?.readouts?.[piece.readout];
      const marks = (ro?.curve || []).filter(c => c.readable).map(c => c.layer);
      const name = tree?.items.find(i => i.id === itemId)?.name || '—';
      return { values: toHs(sel.values, n), marks, peak: ro?.peak?.layer ?? null, selected: st.lblock, allowEmbedding: true,
               onPick: hs => patchSt({ lblock: hs }), valueLabel: '有效神经元', markLabel: '未见词表上可读出（路线 T）',
               color: `颜色 = 「${name}」在各块的有效神经元（模板选择性单元数${sel.ro ? ` · ${READOUT_LABEL[sel.ro] || sel.ro}读出` : ''}）${!census ? '；该模型未普查' : !sel.run ? '；此模型上没有神经元选择性运行' : ''}`,
               mark: piece ? `绿色边 = ${piece.family_id}.${piece.factor} 在该层可读出，粗边 = 峰值层` : '',
               click: '点击 = 在右侧打开「该层」面板（读出与有效神经元），C / E 节同步高亮' };
    }
    if (view === 'response') {
      const c = curveHs(respRow, n);
      const sel = st.rlayer != null ? st.rlayer : (respRow?.peak?.layer ?? null);
      return { ...c, selected: sel, allowEmbedding: true, onPick: hs => patchSt({ rlayer: hs }), valueLabel: '留出投影间隔', markLabel: '未见词表上可读出',
               color: respRow ? `颜色 = ${respTm} · ${respRow.factor} 的逐层留出投影间隔（${READOUT_LABEL[respRow.readout] || respRow.readout}读出）` : '颜色：该模式族在此模型上没有拼图',
               mark: respRow ? '绿色边 = 未见词表上可读出' : '', click: '点击 = 右侧点云切到该层' };
    }
    if (view === 'live') {
      const tr = liveTrace;
      let values = null;
      if (tr?.layers) {
        // 每块活动量 = 平均每个 token 的注意力总写出能量（Σ_h 头输出范数）；嵌入层 = 残差流入范数
        const act = tr.layers.map(l => l.heads.out_norm.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0) / tr.n_tokens);
        const top = Math.max(...act, 1e-9);
        values = [tr.layers[0]?.resid_in?.reduce((a, b) => a + b, 0) / (tr.n_tokens || 1) || 0,
                  ...act.map(v => v / top)];
      }
      return { values, selected: liveStep, allowEmbedding: true, onPick: hs => setLiveStep(hs),
               valueLabel: '该块活动量',
               modules: MODS, onPickModule: hs => setLiveStep(hs),
               color: tr?.schema==='mechanism_trace.v1' ? '原生层序索引；右侧显示选定读出的完整计算阵列' : tr ? `颜色 = 「${tr.model_name}」对这段输入的逐块活动量（注意力总写出能量，归一化）`
                         : '颜色 = 运行实时分析后按各块活动量着色',
               click: '点击 = 跳到该层（嵌入层 / 块 / 输出头）' };
    }
    return { values: null, selected: null, onPick: null, color: '', click: '' };
  }, [view, nLayers, tower, st, block, itemApi.data, census, itemId, model, tree, respRow, respTm, liveTrace, liveStep]);

  const selRuns = tower?.selectivity || [];
  const selRun = selRuns.find(s => s.file === st.sel) || selRuns[0];
  const ros = Object.keys(selRun?.by_readout || {});
  const ro = ros.includes(st.ro) ? st.ro : ros[0];

  /* 上下游电路（Goodfire 口径的「前后一致」验证）：选中 MLP 单元后，按权重路径
     W_out→残差→(注意力头 OV)→W_in 找前一层最强写入者与后一层最强读取者，
     在层塔上画跨层连线；打分来自本机模型权重，是电路假设（确认需干预实验）。 */
  const circuitApi = useApi(view === 'neuron' && census && tower && st.unit != null && st.mod === 'mlp'
    ? `/api/neurons/circuit?block=${block}&unit=${st.unit}&k=5&model_key=${encodeURIComponent(census)}&run=${encodeURIComponent(tower.run || '')}` : null);
  const circuit = circuitApi.data && circuitApi.data.available ? circuitApi.data : null;
  /* 失败必须显式落到卡片上：以前只认 available===false，请求本身报错（服务重启、超时）时
     circuitNote 为空，卡片就永远停在「正在计算…」——错误被静默吞掉。 */
  const circuitNote = circuitApi.data && circuitApi.data.available === false
    ? circuitApi.data.reason
    : (circuitApi.error || null);

  const UP_C = '#d85a30', DOWN_C = '#378add';
  const dff = tower?.d_ff || 0;
  const BCOLS = 128;
  const unitPos = u => {
    const rows = Math.max(1, Math.ceil(dff / BCOLS));
    return [-1 + (2 * ((u % BCOLS) + 0.5)) / BCOLS, -0.6 + (1.2 * (Math.floor(u / BCOLS) + 0.5)) / rows];
  };
  let edges = null, target = null;
  if (view === 'neuron' && circuit && st.mod === 'mlp' && dff) {
    const tp = unitPos(circuit.unit);
    target = { hs: block + 1, x: tp[0], z: tp[1] };
    edges = [
      ...circuit.upstream.map(u => { const p = unitPos(u.unit); return { hs1: block, hs2: block + 1, p1: p, p2: tp, color: UP_C }; }),
      ...circuit.downstream.map(u => { const p = unitPos(u.unit); return { hs1: block + 1, hs2: block + 2, p1: tp, p2: p, color: DOWN_C, dash: [5, 3] }; }),
    ];
  }
  const syncTag = respRow && cfg.selected != null && view === 'response' ? `${respRow.readout}@L${cfg.selected}` : undefined;

  const showTower = view === 'neuron' || view === 'lang' || view === 'response';   // 原生运行页自带按真实快照绘制的3D结构

  /* 布局：需要层结构的 tab 一律左右分栏（塔在左，内容在右）。神经元页右栏常驻，
     不再有「只有塔」的单列态——单列时右边是空白，而块细节本来就是这一页要看的东西。 */
  const split = showTower;

  if (view === 'parameters') return <>{modeBar}<ParameterExplorer researchModel={gModel || model} /><HistoricalPanel kind="activation"/></>;
  if (view === 'live') return <LivePane/>;
  if (view === 'research') return <>{modeBar}<ResearchSpace/></>;
  if (view === 'langframe') return <LanguageFrame/>;
  return (
    <>{modeBar}{view === 'lang' && <p className="v8-note">语言项实测：语言图谱里的每一项能力，在模型上能不能读出、在哪一层、由哪些单元承载。外部框架本身在“空间 · 语言图谱”。</p>}<div className={'v8-space' + (showTower ? '' : ' wide') + (split ? ' split' : '')}>
      {showTower && (
      <aside className="v8-tower">
        <div className="v8-tower-hd">
          <div className="v8-tower-title">
            <b>LLM 多层结构</b>
            {/* 模型不在这里选：页顶的全局条是唯一入口，避免同一页两处各选一个 */}
            <span className="v8-tower-model" title={model}>{shortModel(model)}{skeletons[model]?.census_run ? '' : ' · 未普查'}</span>
          </div>
          {view === 'neuron' && census && tower && (
            <>
              <span className="v8-seg">{OVERLAYS.map(([k, t]) => <button key={k} type="button" className={st.ov === k ? 'on' : ''} onClick={() => patchSt({ ov: k })}>{t}</button>)}</span>
              {/* 选择性运行 / 读出轴是细节：展开详情后才给，默认界面只留一层着色口径 */}
              {st.ov === 'sel' && selRun && (
                <div className="v8-tools">
                  <select value={selRun.file} onChange={e => patchSt({ sel: e.target.value })} aria-label="选择性运行">
                    {selRuns.map(s => <option key={s.file} value={s.file}>{s.item || s.tm_id} · {s.factor}（{s.hi} 对 {s.lo}）</option>)}
                  </select>
                  <span className="v8-seg">{ros.map(x => <button key={x} type="button" className={x === ro ? 'on' : ''} onClick={() => patchSt({ ro: x })}>{READOUT_LABEL[x] || x}</button>)}</span>
                </div>
              )}
            </>
          )}
          <div className="v8-tower-legend">
            <span>{cfg.color}</span>
            {cfg.mark && <span><i aria-hidden="true" />{cfg.mark}</span>}
            {edges && <span><i aria-hidden="true" style={{ background: UP_C }} />橙线 = 前一层写入者（上游）</span>}
            {edges && <span><i aria-hidden="true" style={{ background: DOWN_C }} />蓝虚线 = 后一层读取者（下游）</span>}
            <span>{cfg.click}</span>
            {cfg.selected != null && <span>当前选中：{hsLabel(cfg.selected)}</span>}
          </div>
        </div>
        {/* 塔的读法放在塔上方：默认态整页要在一屏内说完，说明行不能掉到最底部被浮动按钮压住 */}
        <p className="v8-note v8-note-top">每块板 = 一层：下=嵌入层、上=输出头；颜色按最大值归一化（浅→深）。拖动旋转、滚轮缩放。
          {!census && skeleton ? ` ${shortModel(model)} 还没有自然语料普查：没有神经元数据。` : ''}</p>
        {nLayers ? (
          <Tower nLayers={nLayers} selected={cfg.selected} values={cfg.values} marks={cfg.marks} peak={cfg.peak} onPick={cfg.onPick}
                 modules={cfg.modules} selModule={cfg.selModule} onPickModule={cfg.onPickModule} arch={arch}
                 allowEmbedding={cfg.allowEmbedding} valueLabel={cfg.valueLabel} markLabel={cfg.markLabel}
                 edges={edges} target={target} height={edges ? 820 : 740}
                 focusKey={edges ? `${block}:${st.unit}` : ''}
                 caption={`${view === 'live' && liveTrace ? (liveTrace.model_name||'已核验快照') : shortModel(model)} 的 ${nLayers} 个块：${cfg.color}`} />
        ) : <Empty>没有模型骨架数据。</Empty>}
        {arch && (
          <div className="v8-archbox" title={arch.source ? `结构参数来源：${arch.source}` : ''}>
            <div className="v8-arch">
            <div className="v8-arch-c emb">
              <span className="v8-arch-k">词嵌入</span>
              <span className="v8-arch-p">{fmtP(arch.embed.params)}</span>
              <span className="v8-arch-s">{arch.embed.shape_text}</span>
              <span className="v8-arch-s">{arch.embed.tied ? '与输出头共享' : `输出头另算 ${fmtP(arch.embed.lm_head)}`}</span>
            </div>
            <div className="v8-arch-c attn">
              <span className="v8-arch-k">注意力 · 每层</span>
              <span className="v8-arch-p">{fmtP(arch.attn.params)}</span>
              <span className="v8-arch-s">{arch.attn.shape_text}</span>
              <span className="v8-arch-s">{arch.attn.heads}</span>
            </div>
            <div className="v8-arch-c mlp">
              <span className="v8-arch-k">MLP · 每层</span>
              <span className="v8-arch-p">{fmtP(arch.mlp.params)}</span>
              <span className="v8-arch-s">{arch.mlp.shape_text}</span>
              <span className="v8-arch-s">{arch.mlp.act} 激活</span>
            </div>
            </div>
            <p className="v8-arch-sum">
              全模型 <b>{fmtP(arch.total.params)}</b> 参数 · {arch.n_layers} 层 ×（注意力 + MLP + 归一化）+
              词嵌入{arch.embed.tied ? '（共享输出头）' : '与输出头'} · 每层 MLP 是注意力的 <b>{arch.total.mlp_over_attn?.toFixed(2)}×</b>
            </p>
          </div>
        )}
        {archApi.data && archApi.data.available === false && (
          <p className="v8-note">{archApi.data.reason}</p>
        )}
      </aside>
      )}
      <section className="v8-space-main">
        {view === 'neuron' && (
          <div className="v8-detail-hd">
            <b>{hsLabel(block + 1)} 的细节</b>
            <span title={arch ? (st.mod === 'attn' ? `${arch.attn.shape_text} · ${arch.attn.heads}` : st.mod === 'mlp' ? arch.mlp.shape_text : '') : ''}>
              {st.mod === 'attn' ? '注意力' : st.mod === 'norm' ? '归一化层' : 'MLP'}
              {arch && st.mod === 'attn' ? ` · ${fmtP(arch.attn.params)} 参数/层` : ''}
              {arch && st.mod === 'mlp' ? ` · ${fmtP(arch.mlp.params)} 参数/层` : ''}
              {st.mod === 'attn' ? ' · 本层暂无逐头分析数据' : ''}
              {st.unit != null ? ` · U${st.unit}` : ''}
            </span>
          </div>
        )}
        {!tree && <Empty>{treeApi.error ? `语言谱图不在本机数据包里：${treeApi.error}` : '读取语言谱图…'}</Empty>}
        <Suspense fallback={<Empty>加载中…</Empty>}>
          {view === 'neuron' && (census && tower ? <NeuronPane census={census} tower={tower} skeleton={skeleton} block={block} st={st} set={set} arch={arch} circuit={circuit} circuitNote={circuitNote} circuitLoading={st.unit != null ? circuitApi.loading : null} onRetryCircuit={circuitApi.reload} />
            : <Empty>{census ? (tw.error ? `层塔数据不在本机数据包里：${tw.error}` : '读取层塔…') : `${shortModel(model)} 还没有自然语料普查，没有逐单元数据；左侧只画层结构。`}</Empty>)}
          {view === 'lang' && tree && (
            <div className="v8-langhost">
              {/* 入口顺序：先实测（数据集 → 三类型 → 普遍规律），再看单项的模式卡 */}
              <div className="v8-langtabs">
                <button type="button" className={st.ltab === 'items' ? '' : 'on'} onClick={() => patchSt({ ltab: 'atlas' })}>数据集实测</button>
                <button type="button" className={st.ltab === 'items' ? 'on' : ''} onClick={() => patchSt({ ltab: 'items' })}>模式目录（A–F）</button>
              </div>
              {st.ltab === 'items'
                ? <LangPage compact tree={tree} model={model} skeleton={skeleton} st={st} set={set}
                            layer={st.lblock} onLayer={hs => patchSt({ lblock: hs })} />
                : <LangAtlas ds={ds} onDs={onDs} researchModel={model} />}
            </div>
          )}
          {view === 'live' && <LivePane census={census} skeleton={skeleton} tower={tower}
                                        trace={liveTrace} setTrace={setLiveTrace} step={liveStep} setStep={setLiveStep} />}
          {view === 'response' && (
            <><ResponseExplorer key={`${respTm}|${model}`} initialTm={respTm || undefined} initialBucket={model || undefined}
                              syncLayer={syncTag} onLayer={tag => { const h = hsOf(tag); if (h != null) patchSt({ rlayer: h }); }}
                              onContext={({ tm: t }) => { setTm(t); patchSt({ rlayer: null }); }} /><HistoricalPanel kind="fidelity"/></>
          )}
          {view === 'relations' && (
            <div className="v8-atlaswrap">
              <div className="v8-atlas-back">
                <button type="button" onClick={() => setView('lang')}>← 返回语言项实测</button>
                <b>谱图关系</b>
                <span>跨模式族的共享编码轴、概念假设与跨模型一致性；点击一个族回到响应点云看证据</span>
              </div>
              <AtlasView snapshot modelKey={model} onModelChange={onModel} onOpenFamily={({ tm: t }) => { setTm(t); setView('response'); }} />
            </div>
          )}
          {view === 'local' && localPack}
        </Suspense>
      </section>
    </div></>
  );
}
