/* 3D 空间：默认显示一个模型的多层结构，并在其上叠加所选语言项的逐层编码（路线 T）；
   下方热力表是同一份数值的二维视图（精细比较与回查）。其他图形（响应点云、单元卡、谱图关系）切换后才加载。 */
import { Suspense, lazy, useMemo, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { Empty, seqColor } from '../app/rdc_fusion/ui.jsx';
import { LevelChip } from './AtlasPage.jsx';
import LayerStack3D from './LayerStack3D.jsx';
import { MAX_COMPARE, SPACE_VIEWS, compareColor, matchLang, segText, shortModel } from './v7Util.js';

const ResponseExplorer = lazy(() => import('../app/rdc_fusion/ResponseExplorer.jsx'));
const NeuronUnitPanel = lazy(() => import('../app/rdc_fusion/NeuronUnitPanel.jsx'));
const AtlasView = lazy(() => import('../app/rdc_fusion/AtlasView.jsx'));

const pieceLabel = r => `${r.family_id}.${r.factor} · ${r.readout}`;

/* 每个语言项默认用等级最高的拼图；同等级时优先自己的模式族，再看峰值间隔 */
function bestRow(rows) {
  const own = r => (r.family_id === String(r.item).replace('.', '-') ? 1 : 0);   // 同等级时优先该语言项自己的模式族（GR.Q → GR-Q）
  return [...rows].sort((a, b) => b.level - a.level || own(b) - own(a) || ((b.peak?.sep || 0) - (a.peak?.sep || 0)))[0];
}

function Skeleton({ sk }) {
  if (!sk) return null;
  const rows = [['层数', sk.n_layers], ['残差宽度 d_model', sk.d_model], ['MLP 宽度 d_ff', sk.d_ff ?? '未记录'],
                ['精度', sk.precision], ['最终归一化', sk.final_layer_norm], ['普查', sk.census_run || '无']];
  return (
    <div className="v7-skel">
      <h4 className="v7-h4">模型骨架 <small title={sk.model}>{shortModel(sk.model)}</small></h4>
      <dl>{rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{v ?? '—'}</dd></div>)}</dl>
      {sk.model_snapshot && <p className="v6-note">快照 <span className="mono">{sk.model_snapshot.slice(0, 16)}</span></p>}
      <p className="v6-note">{sk.note}</p>
    </div>
  );
}

function Heatmap({ nLayers, rows, selectedLayer, onPickLayer }) {
  const layers = Array.from({ length: (nLayers || 0) + 1 }, (_, i) => i);
  return (
    <div className="v7-heat" role="table" aria-label="逐层留出投影间隔（每行按自身最大值归一化着色）">
      <div className="v7-heat-row hd" role="row">
        <span className="lab" role="columnheader">语言项</span>
        {layers.map(i => (
          <button key={i} type="button" role="columnheader" className={'c' + (i === selectedLayer ? ' sel' : '')}
                  onClick={() => onPickLayer(i)} title={i === 0 ? 'hs[0] 嵌入层' : `L${i}`}>{i === 0 ? 'e' : i % 4 === 0 ? i : ''}</button>
        ))}
      </div>
      {rows.map(r => {
        const max = Math.max(1e-9, ...r.sep.filter(v => v != null));
        return (
          <div key={r.piece} className="v7-heat-row" role="row">
            <span className="lab" role="rowheader"><i style={{ background: r.color }} aria-hidden="true" />{r.name}</span>
            {layers.map(i => {
              const k = r.layers.indexOf(i);
              const v = k >= 0 ? r.sep[k] : null;
              const ok = k >= 0 && r.readable[k];
              return (
                <button key={i} type="button" role="cell" className={'c' + (i === selectedLayer ? ' sel' : '') + (v == null ? ' na' : '')}
                        style={v == null ? undefined : { background: seqColor(Math.max(0, v) / max), color: v / max > 0.55 ? '#fff' : '#1e293b' }}
                        onClick={() => onPickLayer(i)}
                        title={v == null ? `${i === 0 ? 'hs[0]' : `L${i}`}：无数据` : `${i === 0 ? 'hs[0]' : `L${i}`}：间隔 ${v.toFixed(3)}，${ok ? '可读出' : '未达判据'}`}>
                  {ok ? '•' : ''}
                </button>
              );
            })}
          </div>
        );
      })}
      <p className="v6-note">颜色 = 留出投影间隔 ÷ 该行最大值（单一色相，浅 → 深）；• = 该层在未见词表上达到可读出判据；e = 嵌入层 hs[0]。点击任一列选中该层。</p>
    </div>
  );
}

function LayerPanel({ layer, rows }) {
  if (layer == null) return <Empty>点击 3D 中的层板或热力表的列，查看该层的数值。</Empty>;
  return (
    <div className="v7-layer-panel">
      <h4 className="v7-h4">{layer === 0 ? 'hs[0] 嵌入层' : `L${layer} · 第 ${layer} 层之后的残差流`}</h4>
      <table className="v6-mini-table">
        <thead><tr><th>语言项</th><th>拼图</th><th className="num">间隔</th><th>判定</th><th>嵌入层基线</th></tr></thead>
        <tbody>
          {rows.map(r => {
            const k = r.layers.indexOf(layer);
            return (
              <tr key={r.piece}>
                <th><i className="v7-dot" style={{ background: r.color }} aria-hidden="true" />{r.name}</th>
                <td className="mono">{pieceLabel(r)}</td>
                <td className="num">{k >= 0 && r.sep[k] != null ? r.sep[k].toFixed(3) : '—'}</td>
                <td>{k < 0 ? '无数据' : r.readable[k] ? '● 可读出' : '○ 未达判据'}</td>
                <td>{r.token_identity ? '已可分（token 身份）' : '不可分或相同'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function LayersView({ tree, model, lang, st, set }) {
  const ov = useApi(model ? `/api/space/overlay?model_key=${encodeURIComponent(model)}` : null);
  const sk = useApi(model ? `/api/model/skeleton?model_key=${encodeURIComponent(model)}` : null);
  const [choice, setChoice] = useState({});
  const [layer, setLayer] = useState(null);
  const names = useMemo(() => Object.fromEntries(tree.items.map(it => [it.id, it.name])), [tree]);
  const byItem = useMemo(() => {
    const m = {};
    for (const r of ov.data?.route_T || []) (m[r.item] = m[r.item] || []).push(r);
    return m;
  }, [ov.data]);
  const avail = tree.items.filter(it => byItem[it.id] && matchLang(it, lang));
  const cmpValid = st.cmp.filter(id => byItem[id]);
  const cmp = cmpValid.length ? cmpValid
    : [...avail].sort((a, b) => (bestRow(byItem[b.id]).level - bestRow(byItem[a.id]).level)).slice(0, 3).map(it => it.id);
  const rows = cmp.map((id, k) => {
    const opts = byItem[id];
    const r = opts.find(x => x.piece === choice[id]) || bestRow(opts);
    return { ...r, name: names[id] || id, color: compareColor(k) };
  });
  const toggle = id => {
    const next = cmp.includes(id) ? cmp.filter(x => x !== id) : [...cmp, id].slice(-MAX_COMPARE);
    set({ cmp: next });
  };
  if (!model) return <Empty>还没有任何模型的数据。</Empty>;
  if (ov.error) return <Empty>3D 叠加数据未接入：{ov.error}</Empty>;
  if (!ov.data || ov.data.pending) return <Empty>{ov.data?.note || '读取中…'}</Empty>;
  const nLayers = sk.data?.n_layers ?? ov.data.n_layers;
  return (
    <div className="v7-space">
      <aside className="v7-space-side" aria-label="选择语言项">
        <h4 className="v7-h4">语言项 <small>最多对照 {MAX_COMPARE} 个</small></h4>
        {avail.length ? (
          <ul className="v7-pick">
            {avail.map(it => {
              const k = cmp.indexOf(it.id);
              const on = k >= 0;
              const opts = byItem[it.id];
              return (
                <li key={it.id} className={on ? 'on' : ''}>
                  <label>
                    <input type="checkbox" checked={on} onChange={() => toggle(it.id)} />
                    <i className="v7-dot" style={{ background: on ? compareColor(k) : 'transparent', borderColor: on ? compareColor(k) : 'var(--fw-border-strong)' }} aria-hidden="true" />
                    <span>{it.name}</span>
                    <LevelChip level={bestRow(opts).level} />
                  </label>
                  {on && opts.length > 1 && (
                    <select value={(opts.find(x => x.piece === choice[it.id]) || bestRow(opts)).piece}
                            onChange={e => setChoice(c => ({ ...c, [it.id]: e.target.value }))} aria-label={`${it.name} 使用哪张拼图`}>
                      {opts.map(o => <option key={o.piece} value={o.piece}>{pieceLabel(o)}（{o.level}）</option>)}
                    </select>
                  )}
                </li>
              );
            })}
          </ul>
        ) : <Empty>当前模型{lang ? '与语种' : ''}下没有语言项的拼图。</Empty>}
        <Skeleton sk={sk.data && !sk.data.pending ? sk.data : null} />
      </aside>
      <section className="v7-space-main">
        <div className="v7-legend" aria-label="图例">
          {rows.map(r => (
            <span key={r.piece}><i style={{ background: r.color }} aria-hidden="true" />{r.name}
              <small>{pieceLabel(r)} · 可读出 {segText(r.segments)}{r.peak ? ` · 峰值 L${r.peak.layer}` : ''}</small></span>
          ))}
        </div>
        {nLayers ? (
          <LayerStack3D nLayers={nLayers} rows={rows} selectedLayer={layer} onPickLayer={setLayer} />
        ) : <Empty>模型骨架缺少层数记录。</Empty>}
        <p className="v6-note">层序与层数来自 {shortModel(model)} 的结果摘要；层板内的位置只是通道分区，不代表模型几何。
          条长 = 该层留出投影间隔 ÷ 该语言项各层最大值；实心 = 未见词表上可读出，空心 = 未达判据，圆点 = 峰值层。
          {' '}路线 N（神经元）与两路线会合尚未接入，不画单元与连线。</p>
        {rows.length > 0 && <Heatmap nLayers={nLayers} rows={rows} selectedLayer={layer} onPickLayer={setLayer} />}
        <LayerPanel layer={layer} rows={rows} />
      </section>
    </div>
  );
}

export default function SpacePage({ tree, model, lang, st, set }) {
  return (
    <div className="v7-space-page">
      <nav className="v7-views" aria-label="3D 视图">
        {SPACE_VIEWS.map(v => (
          <button key={v.id} type="button" className={st.view === v.id ? 'on' : ''} title={v.desc} onClick={() => set({ view: v.id })}>
            {v.label}{v.id !== 'layers' && <small>次要</small>}
          </button>
        ))}
      </nav>
      <Suspense fallback={<Empty>加载中…</Empty>}>
        {st.view === 'layers' && <LayersView tree={tree} model={model} lang={lang} st={st} set={set} />}
        {st.view === 'response' && <ResponseExplorer key={`${st.tm}|${model}`} initialTm={st.tm || undefined} initialBucket={model || undefined} />}
        {st.view === 'units' && <NeuronUnitPanel />}
        {st.view === 'relations' && <AtlasView snapshot onOpenFamily={({ tm }) => set({ view: 'response', tm })} />}
      </Suspense>
    </div>
  );
}
