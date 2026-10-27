/* 研究工作台 · 空间：左侧常驻 LLM 的多层结构（3D 层塔），右侧按视图显示
   神经元（选中块的内部结构 → MLP 单元阵列 → 单元卡）、语言图谱（语言项六节）、响应点云、谱图关系、本地包。
   语言图谱视图里，层塔的颜色是当前语言项在各块的「有效神经元」（模板选择性单元数）；其他视图里是所选叠加量。
   点击任一块 → 进入该块的神经元视图。全部读取来自本机数据包（只读数据包模式）。 */
import { Suspense, lazy, useEffect, useMemo, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import LangPage from './LangPage.jsx';
import NeuronPane from './NeuronPane.jsx';
import Tower from './Tower.jsx';
import { OVERLAYS, READOUT_LABEL, defaultModel, pickItem, readState, shortModel, towerValues, writeState } from './v8Util.js';
import './v8.css';

const ResponseExplorer = lazy(() => import('../app/rdc_fusion/ResponseExplorer.jsx'));
const AtlasView = lazy(() => import('../app/rdc_fusion/AtlasView.jsx'));

/* 语言项在当前普查模型上的选择性运行（每块的有效神经元数） */
function itemSelectivity(item, census, st) {
  const runs = (item?.selectivity || []).filter(s => s.census_key === census);
  const run = runs.find(s => s.file === st.sel) || runs[0];
  if (!run) return { run: null, ro: null, values: null };
  const ros = Object.keys(run.by_readout || {});
  const ro = ros.includes(st.ro) ? st.ro : ros[0];
  return { run, ro, values: run.by_readout[ro] || null };
}

export default function SpaceLens({ view, setView, tm, setTm, localPack }) {
  const treeApi = useApi('/api/lang/tree');
  const skApi = useApi('/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = useMemo(() => Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s])), [skApi.data]);
  const [st, setSt] = useState(readState);
  useEffect(() => { writeState(st); }, [st]);
  const model = st.model && tree?.models?.includes(st.model) ? st.model : defaultModel(tree, skeletons);
  const skeleton = skeletons[model];
  const census = skeleton?.census_run ? skeleton.census_key : null;
  const tw = useApi(census ? `/api/neurons/tower?model_key=${encodeURIComponent(census)}` : null);
  const tower = tw.data && !tw.data.pending ? tw.data : null;
  const itemId = tree ? pickItem(tree, st, model) : null;
  const itemApi = useApi(view === 'lang' && itemId ? `/api/lang/item/${itemId}` : null);
  const nLayers = tower?.n_layers || skeleton?.n_layers || 0;
  const block = st.block != null && st.block < nLayers ? st.block : Math.min(18, Math.max(0, nLayers - 1));

  /* 面板里的跳转（page: lang / neuron / response）换算成空间视图 */
  const set = patch => {
    const { page, tm: nextTm, ...rest } = patch;
    if (Object.keys(rest).length) setSt(prev => ({ ...prev, ...rest }));
    if (page === 'neuron' || page === 'lang') setView(page);
    if (page === 'response') { setTm(nextTm || ''); setView('response'); }
  };

  const itemSel = view === 'lang' ? itemSelectivity(itemApi.data, census, st) : null;
  const values = useMemo(() => {
    if (view === 'lang') return itemSel?.values || null;
    return tower ? towerValues(tower, st.ov, st.sel, st.ro) : null;
  }, [view, itemSel?.values, tower, st.ov, st.sel, st.ro]);
  const selRuns = tower?.selectivity || [];
  const selRun = selRuns.find(s => s.file === st.sel) || selRuns[0];
  const ros = Object.keys(selRun?.by_readout || {});
  const ro = ros.includes(st.ro) ? st.ro : ros[0];
  const itemName = tree?.items.find(i => i.id === itemId)?.name;

  return (
    <div className="v8-space">
      <aside className="v8-tower">
        <div className="v8-tower-hd">
          <div className="v8-tower-title">
            <b>LLM 多层结构</b>
            <select value={model} onChange={e => setSt(prev => ({ ...prev, model: e.target.value, block: null, unit: null }))} aria-label="模型">
              {(tree?.models || []).map(m => <option key={m} value={m} title={m}>{shortModel(m)}{skeletons[m]?.census_run ? '' : '（未普查）'}</option>)}
            </select>
          </div>
          {view === 'lang' ? (
            <p className="v8-note">颜色 = 「{itemName || '—'}」在各块的有效神经元（模板选择性单元数{itemSel?.ro ? ` · ${READOUT_LABEL[itemSel.ro] || itemSel.ro}读出` : ''}）。
              {!census ? '该模型未普查，没有神经元数据。' : !itemSel?.run ? '这个语言项在此模型上还没有神经元选择性运行。' : ''}</p>
          ) : census && tower ? (
            <>
              <span className="v8-seg">{OVERLAYS.map(([k, t]) => <button key={k} type="button" className={st.ov === k ? 'on' : ''} onClick={() => setSt(p => ({ ...p, ov: k }))}>{t}</button>)}</span>
              {st.ov === 'sel' && selRun && (
                <div className="v8-tools">
                  <select value={selRun.file} onChange={e => setSt(p => ({ ...p, sel: e.target.value }))} aria-label="选择性运行">
                    {selRuns.map(s => <option key={s.file} value={s.file}>{s.item || s.tm_id} · {s.factor}（{s.hi} 对 {s.lo}）</option>)}
                  </select>
                  <span className="v8-seg">{ros.map(x => <button key={x} type="button" className={x === ro ? 'on' : ''} onClick={() => setSt(p => ({ ...p, ro: x }))}>{READOUT_LABEL[x] || x}</button>)}</span>
                </div>
              )}
            </>
          ) : <p className="v8-note">{!skeleton ? '读取模型骨架…' : !census ? `${shortModel(model)} 还没有自然语料普查：只画层结构，没有神经元数据。` : tw.error ? `层塔数据不在本机数据包里：${tw.error}` : '读取层塔…'}</p>}
        </div>
        {nLayers ? (
          <Tower nLayers={nLayers} block={block} values={values} onPick={b => { setSt(p => ({ ...p, block: b, unit: null, mod: 'mlp' })); setView('neuron'); }}
                 caption={`${shortModel(model)} 的 ${nLayers} 个块；点击进入该块的神经元`} />
        ) : <Empty>没有模型骨架数据。</Empty>}
        <p className="v8-note">每块板 = 一个 Transformer 块（下：嵌入层，上：输出头）；颜色按各块最大值归一化，浅 → 深。拖动旋转、滚轮缩放、点击进入该块的神经元。层板只是排列，不代表模型几何。
          {tower && tower.summaries_ready < tower.n_layers ? ` 逐层摘要已生成 ${tower.summaries_ready}/${tower.n_layers} 块。` : ''}</p>
      </aside>
      <section className="v8-space-main">
        {!tree && <Empty>{treeApi.error ? `语言谱图不在本机数据包里：${treeApi.error}` : '读取语言谱图…'}</Empty>}
        <Suspense fallback={<Empty>加载中…</Empty>}>
          {view === 'neuron' && (census && tower ? <NeuronPane census={census} tower={tower} skeleton={skeleton} block={block} st={st} set={set} />
            : <Empty>{census ? (tw.error ? `层塔数据不在本机数据包里：${tw.error}` : '读取层塔…') : `${shortModel(model)} 还没有自然语料普查，没有逐单元数据；左侧只画层结构。`}</Empty>)}
          {view === 'lang' && tree && <LangPage compact tree={tree} model={model} skeleton={skeleton} st={st} set={set} />}
          {view === 'response' && <ResponseExplorer key={`${tm}|${model}`} initialTm={tm || undefined} initialBucket={model || undefined} />}
          {view === 'relations' && <AtlasView snapshot onOpenFamily={({ tm: t }) => { setTm(t); setView('response'); }} />}
          {view === 'local' && localPack}
        </Suspense>
      </section>
    </div>
  );
}
