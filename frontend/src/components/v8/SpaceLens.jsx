/* 研究工作台 · 空间：需要层结构的 tab（神经元、语言图谱、响应点云）左侧显示 LLM 多层结构（3D 层塔），右侧按 tab 显示内容。
   层塔共用，含义属于 tab——每个 tab 自己决定「颜色是什么、标记是什么、点击做什么」，点击层塔从不切换 tab；
   跨 tab 只走显式按钮（如「在神经元中查看 →」）。各 tab 的选中层分开记录（block / lblock / rlayer）。
     神经元   颜色 = 神经元叠加量（选择性 / 已解释 / T1 / 强激活率）；点击 = 选中该块（右侧块结构、单元阵列、单元卡）
     语言图谱 颜色 = 当前语言项在各块的有效神经元（模板选择性单元数）；绿色边 = 未见词表上可读出（路线 T），粗边 = 峰值层；
              点击 = 在文章上方打开「该层」面板，并在 C / E 节高亮该层
     响应点云 颜色 = 所选模式族的逐层留出投影间隔；绿色边 = 可读出；点击 = 点云切到该层
     谱图关系、本地包：不显示层塔（谱图关系视图自带层选择与 3D 谱图；本地包在打开研究包之前没有模型数据），内容占满宽度
   层序号统一用 hs：hs[0] = 嵌入层，块 b 的输出 = hs[b+1]。全部读取来自本机数据包（只读数据包模式）。 */
import { Suspense, lazy, useEffect, useMemo, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import LangPage from './LangPage.jsx';
import LivePane from './LivePane.jsx';
import NeuronPane from './NeuronPane.jsx';
import Tower from './Tower.jsx';
import { OVERLAYS, READOUT_LABEL, defaultModel, hsLabel, pickItem, pickPiece, readState, shortModel, towerValues, writeState } from './v8Util.js';
import './v8.css';

const ResponseExplorer = lazy(() => import('../app/rdc_fusion/ResponseExplorer.jsx'));
const AtlasView = lazy(() => import('../app/rdc_fusion/AtlasView.jsx'));

const hsOf = tag => { const m = /@L(\d+)$/.exec(tag || ''); return m ? Number(m[1]) : null; };
const toHs = (blockValues, n) => (blockValues ? [null, ...Array.from({ length: n }, (_, b) => blockValues[b] ?? null)] : null);

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

export default function SpaceLens({ view, setView, tm, setTm, localPack }) {
  const treeApi = useApi('/api/lang/tree');
  const skApi = useApi('/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = useMemo(() => Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s])), [skApi.data]);
  const [st, setSt] = useState(readState);
  useEffect(() => { writeState(st); }, [st]);
  const patchSt = patch => setSt(prev => ({ ...prev, ...patch }));

  // 实时分析：一次前向的观察数据与步进位置（0 = 嵌入层，1..L = 块，L+1 = 输出头）；数据留在会话内，不进地址栏
  const [liveTrace, setLiveTrace] = useState(null);
  const [liveStep, setLiveStep] = useState(0);

  const model = st.model && tree?.models?.includes(st.model) ? st.model : defaultModel(tree, skeletons);
  const skeleton = skeletons[model];
  const census = skeleton?.census_run ? skeleton.census_key : null;
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
    if (page === 'neuron' || page === 'lang') setView(page);
    if (page === 'response') { setTm(nextTm || ''); setView('response'); }
    if (page === 'relations') setView('relations');   // 谱图关系：语言图谱内的入口（不再是一级 tab）
  };

  /* 每个 tab 的层塔配置 */
  const cfg = useMemo(() => {
    const n = nLayers;
    if (view === 'neuron') {
      return { values: tower ? toHs(towerValues(tower, st.ov, st.sel, st.ro), n) : null, selected: block + 1, allowEmbedding: false,
               onPick: hs => patchSt({ block: hs - 1, unit: null, mod: 'mlp' }),
               valueLabel: (OVERLAYS.find(o => o[0] === st.ov) || [])[1] || '叠加量',
               color: `颜色 = ${(OVERLAYS.find(o => o[0] === st.ov) || [])[1] || '无'}（每块）`, click: '点击 = 选中该块：右侧显示块结构、单元阵列与单元卡' };
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
      if (tr) {
        // 每块活动量 = 平均每个 token 的注意力总写出能量（Σ_h 头输出范数）；嵌入层 = 残差流入范数
        const act = tr.layers.map(l => l.heads.out_norm.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0) / tr.n_tokens);
        const top = Math.max(...act, 1e-9);
        values = [tr.layers[0]?.resid_in?.reduce((a, b) => a + b, 0) / (tr.n_tokens || 1) || 0,
                  ...act.map(v => v / top)];
      }
      return { values, selected: liveStep, allowEmbedding: true, onPick: hs => setLiveStep(hs),
               valueLabel: '该块活动量',
               color: tr ? `颜色 = 「${tr.model_name}」对这段输入的逐块活动量（注意力总写出能量，归一化）`
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

  const showTower = view === 'neuron' || view === 'lang' || view === 'response' || view === 'live';   // 谱图关系、本地包不需要层结构

  return (
    <div className={'v8-space' + (showTower ? '' : ' wide')}>
      {showTower && (
      <aside className="v8-tower">
        <div className="v8-tower-hd">
          <div className="v8-tower-title">
            <b>LLM 多层结构</b>
            <select value={model} onChange={e => patchSt({ model: e.target.value, block: null, unit: null, lblock: null, rlayer: null })} aria-label="模型">
              {(tree?.models || []).map(m => <option key={m} value={m} title={m}>{shortModel(m)}{skeletons[m]?.census_run ? '' : '（未普查）'}</option>)}
            </select>
          </div>
          {view === 'neuron' && census && tower && (
            <>
              <span className="v8-seg">{OVERLAYS.map(([k, t]) => <button key={k} type="button" className={st.ov === k ? 'on' : ''} onClick={() => patchSt({ ov: k })}>{t}</button>)}</span>
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
        {nLayers ? (
          <Tower nLayers={nLayers} selected={cfg.selected} values={cfg.values} marks={cfg.marks} peak={cfg.peak} onPick={cfg.onPick}
                 allowEmbedding={cfg.allowEmbedding} valueLabel={cfg.valueLabel} markLabel={cfg.markLabel}
                 edges={edges} target={target} height={edges ? 780 : 600} focusKey={edges ? `${block}:${st.unit}` : ''}
                 caption={`${view === 'live' && liveTrace ? liveTrace.model_name : shortModel(model)} 的 ${nLayers} 个块：${cfg.color}`} />
        ) : <Empty>没有模型骨架数据。</Empty>}
        <p className="v8-note">每块板 = 一层：下=嵌入层、上=输出头；颜色按最大值归一化（浅→深）。拖动旋转、滚轮缩放。
          {!census && skeleton ? ` ${shortModel(model)} 还没有自然语料普查：没有神经元数据。` : ''}</p>
      </aside>
      )}
      <section className="v8-space-main">
        {!tree && <Empty>{treeApi.error ? `语言谱图不在本机数据包里：${treeApi.error}` : '读取语言谱图…'}</Empty>}
        <Suspense fallback={<Empty>加载中…</Empty>}>
          {view === 'neuron' && (census && tower ? <NeuronPane census={census} tower={tower} skeleton={skeleton} block={block} st={st} set={set} circuit={circuit} circuitNote={circuitNote} circuitLoading={st.unit != null ? circuitApi.loading : null} onRetryCircuit={circuitApi.reload} />
            : <Empty>{census ? (tw.error ? `层塔数据不在本机数据包里：${tw.error}` : '读取层塔…') : `${shortModel(model)} 还没有自然语料普查，没有逐单元数据；左侧只画层结构。`}</Empty>)}
          {view === 'lang' && tree && <LangPage compact tree={tree} model={model} skeleton={skeleton} st={st} set={set}
                                                layer={st.lblock} onLayer={hs => patchSt({ lblock: hs })} />}
          {view === 'live' && <LivePane census={census} skeleton={skeleton} tower={tower}
                                        trace={liveTrace} setTrace={setLiveTrace} step={liveStep} setStep={setLiveStep} />}
          {view === 'response' && (
            <ResponseExplorer key={`${respTm}|${model}`} initialTm={respTm || undefined} initialBucket={model || undefined}
                              syncLayer={syncTag} onLayer={tag => { const h = hsOf(tag); if (h != null) patchSt({ rlayer: h }); }}
                              onContext={({ tm: t }) => { setTm(t); patchSt({ rlayer: null }); }} />
          )}
          {view === 'relations' && (
            <div className="v8-atlaswrap">
              <div className="v8-atlas-back">
                <button type="button" onClick={() => setView('lang')}>← 返回语言图谱</button>
                <b>谱图关系</b>
                <span>跨模式族的共享编码轴、概念假设与跨模型一致性；点击一个族回到响应点云看证据</span>
              </div>
              <AtlasView snapshot onOpenFamily={({ tm: t }) => { setTm(t); setView('response'); }} />
            </div>
          )}
          {view === 'local' && localPack}
        </Suspense>
      </section>
    </div>
  );
}
