/* 主线 ①：语言图谱。左侧「大类 → 语言族 → 语言项」，右侧语言项页按固定六节从「是什么」讲到「由哪些神经元承载」。
   每节只用真实数据（lang_atlas.v1 端点）；某一节没有数据时显示「未测」与下一步，不用示意数据填充。 */
import { useMemo, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { LayerCurve, ScatterPlot, SelBars } from './Charts.jsx';
import { LANG_LABEL, LEVELS, READOUT_LABEL, hsLabel, itemLevel, levelMeta, matchLang, pickItem, pickPiece, pieceLabel, segText, shortModel } from './v8Util.js';

export function LevelChip({ level, prefix, none, title }) {
  if (none) return <span className="v8-chip none" title={title}>{prefix && <em>{prefix}</em>}{none}</span>;
  const m = levelMeta(level);
  return <span className="v8-chip" style={{ background: m.color, color: m.ink }} title={title || m.label}>{prefix && <em>{prefix}</em>}{m.sym} {m.label}</span>;
}

function Sidebar({ tree, model, st, set }) {
  const [ext, setExt] = useState(st.type === 'EXT');
  const main = tree.types.filter(t => t.id !== 'EXT');
  const items = tree.items.filter(it => matchLang(it, st.lang));
  const langs = [...new Set(tree.items.flatMap(it => [...(it.languages || []), ...(it.status?.languages_tested || [])]))].sort();
  const types = ext ? [st.type, 'EXT'].filter((v, i, a) => a.indexOf(v) === i) : [st.type === 'EXT' ? 'GR' : st.type];
  return (
    <aside className="v8-side" aria-label="语言分类">
      <div className="v8-bigcats">
        {main.map(t => {
          const its = items.filter(i => i.type === t.id);
          return (
            <button key={t.id} type="button" className={st.type === t.id ? 'on' : ''} title={t.desc} onClick={() => set({ type: t.id })}>
              <b>{t.name}</b><small>{its.filter(i => itemLevel(i, model) > 0).length}/{its.length} 有数据</small>
            </button>
          );
        })}
      </div>
      <div className="v8-side-tools">
        <button type="button" className="v8-ext" onClick={() => setExt(v => !v)} aria-expanded={ext}>{ext ? '－ 收起扩展' : '＋ 扩展（语种…）'}</button>
        <label>语种
          <select value={st.lang} onChange={e => set({ lang: e.target.value })}>
            <option value="">全部</option>
            {langs.map(l => <option key={l} value={l}>{LANG_LABEL[l] || l}</option>)}
          </select>
        </label>
      </div>
      {types.map(tid => tree.categories.filter(c => c.type === tid).map(c => {
        const its = items.filter(i => i.category === c.id);
        if (!its.length) return null;
        return (
          <div key={c.id} className="v8-fam">
            <div className="v8-fam-hd">{c.name}<small>{tid === 'EXT' ? '扩展' : '语言族'}</small></div>
            {its.map(i => {
              const m = levelMeta(itemLevel(i, model));
              return (
                <button key={i.id} type="button" className={'v8-itm' + (st.item === i.id ? ' on' : '')} onClick={() => set({ item: i.id, piece: '' })} title={`${i.name}：${m.label}`}>
                  <span className="s" style={{ color: m.level ? m.color : 'var(--fw-text-3)' }} aria-hidden="true">{m.sym}</span>{i.name}
                </button>
              );
            })}
          </div>
        );
      }))}
      <div className="v8-lvlegend">
        <b>状态（{shortModel(model)}，读出证据）</b>
        {LEVELS.map(l => <span key={l.level}><i style={{ color: l.level ? l.color : 'var(--fw-text-3)' }}>{l.sym}</i> {l.label}</span>)}
      </div>
    </aside>
  );
}

function Sentence({ row, readout }) {
  const t = row.text || '';
  const tc = row.target_char, tl = (row.target || '').length;
  const fin = t.length - 1;
  const parts = [];
  for (let i = 0; i < t.length; i += 1) {
    const inTarget = tc != null && i >= tc && i < tc + tl;
    const isRead = (readout === 'final' && i === fin) || (readout === 'target' && inTarget && i === tc + tl - 1);
    parts.push(<span key={i} className={(inTarget ? 'tg' : '') + (isRead ? ' ro' : '')}>{t[i]}</span>);
  }
  return <div className="v8-sent">{parts}</div>;
}

function SecA({ ex, piece }) {
  if (!ex?.rows?.length) return <Empty>这个语言项还没有冻结的模板材料。下一步：在「参与」领题，或在研究工作台设计模板。</Empty>;
  return (
    <>
      <div className="v8-pairs">
        {ex.levels.map((lv, k) => (
          <div key={lv} className="v8-pair-col">
            <h4><i style={{ background: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'][k % 4] }} />{ex.factor} = {lv}</h4>
            {ex.rows.filter(r => r.level === lv).map((r, i) => <Sentence key={i} row={r} readout={piece?.readout} />)}
          </div>
        ))}
      </div>
      <p className="v8-note"><span className="tg">下划线</span> = 目标词；<span className="ro">蓝框</span> = 代表读出位置（{READOUT_LABEL[piece?.readout] || piece?.readout || '—'}）。
        {ex.minimal_pairs === false ? '这个因子随填充词变化（每个词只属于一个水平），没有最小对：各水平各取几个不同填充的句子' : '同一行事实（填充词）在各水平上各取一两个句式'}，只展示发现集材料；模板 {ex.tm_id}。</p>
    </>
  );
}

function SecB({ piece, ex }) {
  const cells = Object.entries(piece?.behavior?.cells || {});
  return (
    <>
      {Object.entries(ex?.probes || {}).slice(0, 2).map(([lv, pr]) => (
        <div key={lv} className="v8-probe">
          <small>{ex.factor} = {lv} 的探针</small>
          <div>{pr.prompt}</div>
          <div className="v8-ans"><span className="ok">可接受：{(pr.answers || []).join(' / ')}</span>{pr.contrast && <span className="no">对照：{[].concat(pr.contrast).join(' / ')}</span>}</div>
        </div>
      ))}
      {cells.length ? (
        <table className="v6-mini-table v8-beh">
          <thead><tr><th>条件格</th><th className="num">样本</th><th className="num">准确率</th>{piece.behavior.mode === 'paired' && <th className="num">单句绝对准确率</th>}<th>行为门</th></tr></thead>
          <tbody>{cells.map(([c, v]) => (
            <tr key={c}><th className="mono">{c}</th><td className="num">{v.n}</td><td className="num">{v.acc != null ? v.acc.toFixed(3) : '—'}</td>
              {piece.behavior.mode === 'paired' && <td className="num">{v.absolute_acc != null ? v.absolute_acc.toFixed(3) : '—'}</td>}
              <td>{v.pass ? '✓ 通过' : '✗ 未通过（不进入内部分析）'}</td></tr>
          ))}</tbody>
        </table>
      ) : <Empty>没有行为门记录。</Empty>}
      {piece?.behavior?.mode === 'paired' && <p className="v8-note">成对口径：同一事实只在该因子上不同的两句组成一对，两句的强制选择间隔之和 &gt; 0 判对。它检验模型能否区分，不检验单句是否答对。</p>}
    </>
  );
}

function SecE({ sel, census, block, ro, set, st, onLayer }) {
  if (!census) return <Empty>当前模型没有自然语料普查，神经元主线不可用。切换到有普查的模型（如 qwen3-4b）查看。</Empty>;
  const runs = sel.filter(s => s.census_key === census);
  if (!runs.length) return <Empty>当前模型没有这个语言项的神经元选择性运行（需要在模板里声明 mlp_units 并重新采集）。</Empty>;
  const run = runs.find(s => s.file === st.sel) || runs[0];
  const ros = Object.keys(run.by_readout || {});
  const r = ros.includes(ro) ? ro : ros[0];
  const counts = run.by_readout[r] || [];
  const b = block != null && block < counts.length ? block : counts.indexOf(Math.max(...counts));
  const top = run.top?.[r]?.[String(b)] || [];
  return (
    <>
      <div className="v8-tools">
        {runs.length > 1 && <label>运行 <select value={run.file} onChange={e => set({ sel: e.target.value })}>{runs.map(s => <option key={s.file} value={s.file}>{s.tm_id} v{s.version} · {s.factor}</option>)}</select></label>}
        <span className="v8-seg">{ros.map(x => <button key={x} type="button" className={x === r ? 'on' : ''} onClick={() => set({ ro: x })}>{READOUT_LABEL[x] || x}读出</button>)}</span>
        <span className="v8-note">{run.factor}：{run.hi} 对 {run.lo}</span>
      </div>
      <SelBars counts={counts} block={b} onPick={nb => (onLayer ? onLayer(nb + 1) : set({ block: nb }))} />
      <div className="v8-two">
        <div>
          <b className="v8-sub">块 {b} · {counts[b] || 0} 个选择性单元（共 9,728），效应最大的 {top.length} 个</b>
          <ul className="v8-units">
            {top.map(([u, d, pos]) => (
              <li key={u}>
                <span className="mono">B{b}·U{u}</span><span>留出 d = {d}</span>
                <span className="v8-chip" style={{ background: pos === run.hi ? '#dbeafe' : '#ffedd5', color: pos === run.hi ? '#1e40af' : '#9a3412' }}>偏向 {pos}</span>
                <button type="button" className="v6-linkbtn" onClick={() => set({ page: 'neuron', block: b, unit: u, sel: run.file, ro: r, ov: 'sel', color: 'sel', mod: 'mlp' })}>看这个神经元 →</button>
              </li>
            ))}
          </ul>
        </div>
        <p className="v8-note">选择性：发现集拟合、留出同号复现、置换族错误阈值（neurons.v1）。它只说明「这个单元在两类句子上激活不同」，不说明模型用它来区分；后者需要消融（控制轴）。路线 N（神经元经模板确认后挂到语言项）实现后，这里还会列出已确认的单元。</p>
      </div>
    </>
  );
}

/* 该层面板：语言图谱 tab 里点层塔或曲线选中某层后，显示该语言项在这一层的读出与有效神经元（不切换 tab） */
function LayerPanel({ layer, piece, sel, census, set, onClose }) {
  const b = layer - 1;
  const runs = (sel || []).filter(s => s.census_key === census);
  return (
    <section className="v8-layerpanel" aria-label="该层">
      <div className="v8-layerpanel-hd"><b>该层：{hsLabel(layer)}</b><button type="button" className="v6-linkbtn" onClick={onClose}>收起</button></div>
      <div className="v8-two">
        <div>
          <b className="v8-sub">读出（{piece ? pieceLabel(piece) : '—'}）</b>
          {piece ? (
            <ul className="v8-lp-list">
              {Object.entries(piece.readouts).map(([ro, r]) => {
                const c = r.curve.find(x => x.layer === layer);
                return <li key={ro}>{READOUT_LABEL[ro] || ro}读出：{c ? <>间隔 <span className="mono">{c.sep ?? '—'}</span> · {c.readable ? '● 未见词表上可读出' : '○ 未达判据'}</> : '该层无记录（各样本向量相同或未采集）'}</li>;
              })}
            </ul>
          ) : <p className="v8-note">当前模型上没有这个语言项的拼图。</p>}
        </div>
        <div>
          <b className="v8-sub">有效神经元（块 {b >= 0 ? b : '—'} 的 MLP 单元）</b>
          {layer === 0 ? <p className="v8-note">嵌入层没有 MLP 单元；它是 token 身份基线。</p>
            : !census ? <p className="v8-note">该模型未普查，没有神经元数据。</p>
              : runs.length ? runs.map(run => Object.entries(run.by_readout || {}).map(([ro, counts]) => {
                const top = run.top?.[ro]?.[String(b)] || [];
                return (
                  <div key={run.file + ro} className="v8-lp-run">
                    <small>{run.tm_id} · {READOUT_LABEL[ro] || ro}读出：{counts[b] || 0} 个选择性单元</small>
                    {top.slice(0, 3).map(([u, d, pos]) => (
                      <div key={u} className="v8-lp-unit"><span className="mono">U{u}</span> d={d} · 偏向 {pos}
                        <button type="button" className="v6-linkbtn" onClick={() => set({ page: 'neuron', block: b, unit: u, sel: run.file, ro, ov: 'sel', color: 'sel', mod: 'mlp' })}>在神经元中查看 →</button></div>
                    ))}
                  </div>
                );
              })) : <p className="v8-note">这个语言项在此模型上没有神经元选择性运行。</p>}
        </div>
      </div>
    </section>
  );
}

function ItemPage({ tree, itemId, model, skeleton, st, set, layer, onLayer }) {
  const api = useApi(itemId ? `/api/lang/item/${itemId}` : null);
  const [allModels, setAllModels] = useState(false);
  const d = api.data && !api.data.pending ? api.data : null;
  const piece = useMemo(() => (d ? pickPiece(d.pieces || [], itemId, model, st.piece) : null), [d, itemId, model, st.piece]);
  if (!itemId) return <Empty>在左侧选一个语言项。</Empty>;
  if (api.error) return <Empty>该语言项的数据未接入：{api.error}</Empty>;
  if (!d) return <Empty>{api.data?.note || '读取中…'}</Empty>;
  const it = d.item, stt = d.status;
  const ps = (d.pieces || []).filter(p => p.model === model);
  const examples = d.examples || [];
  const oldCenter = d.examples === undefined;           // 1.0.8 及更早的中心没有例句 / 行为门 / 点云 / 选择性字段
  const ex = piece ? examples.find(e => e.tm_id === piece.tm_id && e.factor === piece.factor) : examples[0];
  const ro = piece ? piece.readouts[piece.readout] : null;
  const rules = (d.rules || []).filter(r => allModels || r.model === model);
  const typeName = tree.types.find(t => t.id === it.type)?.name || it.type;
  const catName = tree.categories.find(c => c.id === it.category)?.name || it.category;
  const secs = ['A 这是什么', 'B 模型能不能做到', 'C 在哪一层被编码', 'D 编码长什么样', 'E 由哪些神经元承载', 'F 证据与结论'];
  return (
    <article className="v8-article">
      <header className="v8-ihead">
        <div>
          <div className="v8-path">{typeName} › {catName}</div>
          <h2>{it.name}</h2>
          <p>{it.desc}{it.also_in?.length ? `（也属于 ${it.also_in.join('、')}）` : ''}</p>
        </div>
        <div className="v8-ihead-st">
          <LevelChip prefix="T 模板" level={itemLevel({ status: { by_model: Object.fromEntries(Object.entries(stt.by_model).map(([m, v]) => [m, v.read])) } }, model)} />
          <LevelChip prefix="N 神经元" none="— 待接入" title="路线 N（神经元经模板确认后入树）尚未实现" />
          <LevelChip prefix="控制" none="— 未检验" />
        </div>
      </header>
      <nav className="v8-toc">
        {secs.map((s, i) => <a key={s} href={`#v8sec${i}`}>{s}</a>)}
        {ps.length > 1 && (
          <label className="v8-piecesel">拼图
            <select value={piece?.id || ''} onChange={e => set({ piece: e.target.value })}>
              {ps.map(p => <option key={p.id} value={p.id}>{pieceLabel(p)}（{levelMeta(p.level).label}）</option>)}
            </select>
          </label>
        )}
      </nav>
      {oldCenter && <Empty>当前中心的数据版本较旧（1.0.8 及以前）：对照例句、行为门、点云与神经元选择性要在中心升级后显示；层曲线与规则照常可看。</Empty>}
      {layer != null && onLayer && <LayerPanel layer={layer} piece={piece} sel={d.selectivity} census={skeleton?.census_key && skeleton?.census_run ? skeleton.census_key : null} set={set} onClose={() => onLayer(null)} />}
      {!piece && <Empty>{stt.planned ? '这个语言项还没有模式族：下面各节都是「未测」。' : `${shortModel(model)} 上还没有这个语言项的拼图；其他模型：${Object.keys(stt.by_model).map(shortModel).join('、') || '无'}。`}</Empty>}
      <section className="v8-sec" id="v8sec0"><h3><span className="k">A</span>这是什么</h3>
        <p className="v8-q">用最小对把语言现象固定下来：同一事实只在这一个因子上不同。</p><SecA ex={ex} piece={piece} /></section>
      <section className="v8-sec" id="v8sec1"><h3><span className="k">B</span>模型能不能做到</h3>
        <p className="v8-q">先确认模型真的分得清；行为门未通过的条件格不进入内部分析，也不能作为机制正例。</p>
        {piece ? <SecB piece={piece} ex={ex} /> : <Empty>未测。</Empty>}</section>
      <section className="v8-sec" id="v8sec2"><h3><span className="k">C</span>在哪一层被编码</h3>
        <p className="v8-q">每层、每个读出位置上用发现集拟合方向，在没见过的词上检验能否把两类分开（留出投影间隔）。</p>
        {ro ? <><LayerCurve readouts={piece.readouts} nLayers={skeleton?.n_layers} highlight={layer} onPick={onLayer} />
          <p className="v8-note">{piece.readout ? `代表读出位置 ${READOUT_LABEL[piece.readout] || piece.readout}：可读出 ${segText(ro.segments)}${ro.peak ? `，峰值 L${ro.peak.layer}` : ''}。` : ''}
            {ro.token_identity ? '嵌入层 hs[0] 上已可分：差异可能来自读出位置的 token 身份，不计为留出复现。' : ro.baseline?.state === 'identical' ? '嵌入层 hs[0] 上各样本 token 相同：可读出来自上下文计算。' : ''}</p></> : <Empty>未测。</Empty>}</section>
      <section className="v8-sec" id="v8sec3"><h3><span className="k">D</span>编码长什么样</h3>
        <p className="v8-q">峰值层上每个点是一句话的 HiddenState 投影（discovery 拟合的 LDA 轴）；留出样本落在哪一侧，就是方向能否推广的直观检查。</p>
        {piece?.scatter ? <div className="v8-two"><ScatterPlot scatter={piece.scatter} />
          <p className="v8-note">{piece.scatter.tag} · {piece.scatter.axes}。点与点的距离只在这张图里有意义，不代表模型内部几何。悬停看原句。<br /><button type="button" className="v6-linkbtn" onClick={() => set({ page: 'response', tm: piece.tm_id })}>查看完整点云与规律分析（响应点云）→</button></p></div>
          : <Empty>{piece ? '这张拼图没有达到可读出的峰值层，或分析结果里没有点云坐标。' : '未测。'}</Empty>}</section>
      <section className="v8-sec" id="v8sec4"><h3><span className="k">E</span>由哪些神经元承载</h3>
        <p className="v8-q">每块 MLP 的中间单元里，对这一变量有显著选择性的有多少、最强的是哪些。点柱子换块，点单元进入「神经元」主线。</p>
        <SecE sel={d.selectivity || []} census={skeleton?.census_key && skeleton?.census_run ? skeleton.census_key : null} block={layer != null && layer > 0 ? layer - 1 : (onLayer ? null : st.block)} ro={st.ro} set={set} st={st} onLayer={onLayer} /></section>
      <section className="v8-sec" id="v8sec5"><h3><span className="k">F</span>证据与结论</h3>
        <div className="v8-axes">
          <div><small>读出轴（是否编码）</small><LevelChip level={stt.by_model[model]?.read || 0} /></div>
          <div><small>控制轴（是否被使用）</small><LevelChip none="未检验" /></div>
          <div><small>各模型</small><span>{Object.entries(stt.by_model).map(([m, v]) => `${shortModel(m)} ${levelMeta(v.read).sym}`).join(' · ') || '—'}</span></div>
        </div>
        <div className="v8-tools"><b className="v8-sub">规则（{rules.length} 条）</b>
          {Object.keys(stt.by_model).length > 1 && <label className="v8-toggle"><input type="checkbox" checked={allModels} onChange={e => setAllModels(e.target.checked)} />显示全部模型</label>}</div>
        {rules.length ? <ul className="v8-rules">{rules.map(r => <li key={r.piece}><LevelChip level={r.level} /><span>{allModels && <b>{shortModel(r.model)} · </b>}{r.text}</span></li>)}</ul> : <Empty>当前模型上没有规则。</Empty>}
        <details className="v8-criteria"><summary>等级判据（固定在 pieces.py，随结果返回）</summary>
          <ul>{LEVELS.map(l => <li key={l.level}><LevelChip level={l.level} /> {d.level_labels?.[l.level]}</li>)}</ul>
          <p className="v8-note">{d.criteria?.readable_rule}</p><p className="v8-note">{d.criteria?.baseline_rule}</p>
          <p className="v8-note">等级只描述可读出证据：不表示该方向被模型使用，也不是计算单元。</p></details>
      </section>
    </article>
  );
}

export default function LangPage({ tree, model, skeleton, st, set, compact = false, layer = null, onLayer }) {
  const itemId = pickItem(tree, st, model);
  return (
    <div className={'v8-lang' + (compact ? ' compact' : '')}>
      <Sidebar tree={tree} model={model} st={{ ...st, item: itemId }} set={set} />
      <ItemPage key={itemId} tree={tree} itemId={itemId} model={model} skeleton={skeleton} st={st} set={set} layer={layer} onLayer={onLayer} />
    </div>
  );
}
