/* 空间 · 原生神经元（实测）：普查运行与选择性运行的选择、单元检索、按层浏览 → 单元卡。
   数据源：/api/neurons/index、/api/neurons/search、/api/neurons/selectivity、/api/neurons/unit。
   没有普查结果时显示空态；按 |d| 与按残差贡献两种排序口径都显式标注。 */
import { useMemo, useState } from 'react';

import { useApi } from './api.js';
import NeuronUnitCard from './NeuronUnitCard.jsx';
import { REGIME_META, fmtNum, parseUnitKey, unitKey } from './neuronUtil.js';
import './neuron_units.css';

function readUnitParam() {
  try {
    return parseUnitKey(new URLSearchParams(window.location.search).get('unit') || '');
  } catch {
    return null;
  }
}

function writeUnitParam(sel) {
  try {
    const url = new URL(window.location.href);
    if (sel) url.searchParams.set('unit', unitKey(sel.block, sel.unit)); else url.searchParams.delete('unit');
    window.history.replaceState(null, '', url);
  } catch { /* 地址栏不可写时只保留页面状态 */ }
}

export default function NeuronUnitPanel() {
  const index = useApi('/api/neurons/index');
  const census = useMemo(() => index.data?.census || [], [index.data]);
  const [runKey, setRunKey] = useState('');
  const run = census.find(r => `${r.model_key}/${r.run}` === runKey)
    || census.slice().sort((a, b) => (b.n_tokens || 0) - (a.n_tokens || 0))[0];
  const selRuns = (index.data?.selectivity || []).filter(r => run && r.model_key === run.model_key);
  const [selFile, setSelFile] = useState('');
  const selRun = selRuns.find(r => r.file === selFile) || selRuns.find(r => r.contribution) || selRuns[0];
  const [readout, setReadout] = useState('final');
  const [layer, setLayer] = useState(23);
  const [rank, setRank] = useState('contrib');
  const [q, setQ] = useState('');
  const [query, setQuery] = useState('');
  const [sel, setSel] = useState(readUnitParam);

  const search = useApi(run && query ? `/api/neurons/search?model_key=${encodeURIComponent(run.model_key)}&run=${encodeURIComponent(run.run)}&q=${encodeURIComponent(query)}` : null);
  const selectivity = useApi(selRun ? `/api/neurons/selectivity?run=${encodeURIComponent(selRun.file)}&top=20` : null);
  const tag = selectivity.data?.tags.find(t => t.readout === readout && t.block === layer);
  const list = tag ? (rank === 'contrib' && selectivity.data.contribution ? tag.top_by_contrib : tag.top_by_d) : [];

  const choose = (block, unit) => {
    const next = { block, unit };
    setSel(next);
    writeUnitParam(next);
  };

  if (index.error) return <div className="nu-empty nu-err">神经元端点不可用：{index.error}</div>;
  if (!index.data) return <div className="nu-empty">读取中…</div>;
  if (!run) {
    return <div className="nu-empty">未接入：还没有自然语料普查结果（scripts/run_mlp_census.py）。本页只显示实测数据。</div>;
  }
  return (
    <div className="nu-wrap">
      <aside className="nu-side">
        <div className="nu-badge">实测 · MLP 中间单元</div>
        <label className="nu-field">普查运行
          <select value={`${run.model_key}/${run.run}`} onChange={e => setRunKey(e.target.value)}>
            {census.map(r => <option key={`${r.model_key}/${r.run}`} value={`${r.model_key}/${r.run}`}>
              {r.model_key.slice(0, 8)} · {r.run} · {(r.n_tokens || 0).toLocaleString()} token</option>)}
          </select>
        </label>

        <form className="nu-field" onSubmit={e => { e.preventDefault(); const u = parseUnitKey(q); if (u) choose(u.block, u.unit); else setQuery(q.trim()); }}>
          检索（L23.U8056，或按读出 token，如「不确定」）
          <div className="nu-row"><input value={q} onChange={e => setQ(e.target.value)} placeholder="L23.U8056 / 不确定"/><button type="submit">查找</button></div>
        </form>
        {query && (
          <div className="nu-list">
            {search.error && <div className="nu-err">{search.error}</div>}
            {search.data?.note && <div className="nu-note">{search.data.note}</div>}
            {(search.data?.results || []).map(r => (
              <button key={unitKey(r.block, r.unit)} className={'nu-item' + (sel && sel.block === r.block && sel.unit === r.unit ? ' on' : '')}
                      onClick={() => choose(r.block, r.unit)}>
                <b>{unitKey(r.block, r.unit)}</b> {r.promotes && <span className="nu-note">{r.promotes.join(' ')}</span>}
              </button>
            ))}
            {search.data && !search.data.results.length && <div className="nu-note">无结果。</div>}
          </div>
        )}

        <div className="nu-field">按层浏览（模板选择性）
          {selRuns.length === 0 ? <div className="nu-note">该模型没有模板选择性运行。</div> : (<>
            <select value={selRun?.file || ''} onChange={e => setSelFile(e.target.value)}>
              {selRuns.map(r => <option key={r.file} value={r.file}>{r.file}</option>)}
            </select>
            <div className="nu-row">
              <select value={readout} onChange={e => setReadout(e.target.value)}>
                <option value="final">final</option><option value="probe_answer">probe_answer</option>
              </select>
              <input type="number" min={0} max={(run.n_layers || 1) - 1} value={layer} onChange={e => setLayer(Number(e.target.value))} aria-label="层"/>
              <select value={rank} onChange={e => setRank(e.target.value)} title="排序口径">
                <option value="contrib">按残差贡献</option><option value="d">按 |d|（偏向低方差单元）</option>
              </select>
            </div>
          </>)}
        </div>
        {tag && (
          <div className="nu-list">
            <div className="nu-note">B{tag.block} 入选 {tag.n_selected} / {tag.n_units}（阈值 |d| ≥ {fmtNum(tag.cut)}）</div>
            {list.map(u => (
              <button key={u.unit} className={'nu-item' + (sel && sel.block === layer && sel.unit === u.unit ? ' on' : '')} onClick={() => choose(layer, u.unit)}>
                <b>U{u.unit}</b> d {fmtNum(u.d_disc)} / {fmtNum(u.d_held)}
                {u.dominant_regime && <span className="nu-reg" style={{ background: REGIME_META[u.dominant_regime].color }}>{u.dominant_regime}</span>}
                <span className="nu-note">高极 {u.pos_level}{u.contrib != null && ` · 贡献 ${fmtNum(u.contrib)}`}</span>
              </button>
            ))}
          </div>
        )}
      </aside>
      <main className="nu-main">
        <NeuronUnitCard modelKey={run.model_key} run={run.run} block={sel?.block} unit={sel?.unit}/>
      </main>
    </div>
  );
}
