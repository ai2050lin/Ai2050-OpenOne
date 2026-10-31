/* 语言图谱 · 数据集实测（lang_atlas）
   流程按「先选数据集 → 再对三个类型做注意力实测 → 最后跨类型归纳普遍规律」组织，
   与 lang_bench（只测正确率）分开：这里测的是「分数怎么来的」。

   三个概念：
     数据集   题库里的 source（C-Eval / CMMLU / CMExam），决定样本量与题型。
     类型     知识网络 / 功能推理 / 语法规则。类型 ↔ 科目是显式映射表，接口随数据返回，可审阅。
     普遍规律 同一 (层,头) 在 ≥2 个类型上同号且过 Bonferroni 阈值。判据在服务端写死，界面不重算。
*/
import { useEffect, useMemo, useState } from 'react';

import { apiUrl } from '../../config.js';
import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';

const CAT_ORDER = ['知识网络', '功能推理', '语法规则'];

/* 默认选中：沿用上次实测的数据集（这样一进来就能看到结果），否则第一个可测的。 */
function layoutsDefault(datasets, last) {
  if (last?.dataset && datasets.some(d => d.id === last.dataset)) return last.dataset;
  return datasets.find(d => d.testable)?.id ?? null;
}
const CAT_COLOR = { 知识网络: '#0d366b', 功能推理: '#7a4bb5', 语法规则: '#0f7b6c' };

/* t 值色阶：正 = 推高答案（蓝），负 = 压制（橙）。深浅按 |t|，4 以上满色。 */
function tStyle(t) {
  const a = Math.min(1, Math.abs(t) / 8);
  if (t >= 0) return { background: `rgba(13,54,107,${0.08 + 0.72 * a})`, color: a > 0.5 ? '#fff' : '#0d366b' };
  return { background: `rgba(216,90,48,${0.08 + 0.72 * a})`, color: a > 0.5 ? '#fff' : '#8a3416' };
}

function useStatusPoll() {
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const poll = async () => {
    for (let i = 0; i < 240; i += 1) {
      const r = await fetch(apiUrl('/api/lang/atlas/status')).then(x => x.json()).catch(() => null);
      if (!r) break;
      setStatus(r);
      if (r.state === 'done' || r.state === 'error') { setBusy(false); return r; }
      await new Promise(res => { setTimeout(res, 1500); });
    }
    setBusy(false);
    return null;
  };
  return { status, setStatus, busy, setBusy, poll };
}

function DatasetCard({ d, active, onPick }) {
  const testable = d.testable;
  return (
    <button type="button" className={'v8-atlas-ds' + (active ? ' on' : '') + (testable ? '' : ' off')}
            onClick={() => testable && onPick(d.id)} disabled={!testable}
            title={d.reason || '三个类型都可测'}>
      <b>{d.id}</b>
      <span>{d.n_questions.toLocaleString()} 题 · {d.n_subjects} 科目</span>
      <div className="v8-atlas-ds-cats">
        {CAT_ORDER.map(c => (
          <em key={c} style={{ borderColor: d.by_category[c].n ? CAT_COLOR[c] : '#cbd5e1', color: d.by_category[c].n ? CAT_COLOR[c] : '#94a3b8' }}>
            {c} {d.by_category[c].n}
          </em>
        ))}
      </div>
      <small>{testable ? `可测 ${CAT_ORDER.reduce((s, c) => s + d.by_category[c].n, 0)} 题` : '不含三大类型'}</small>
    </button>
  );
}

function LayerBar({ hist, nLayers }) {
  const total = nLayers || 36;
  const max = Math.max(1, ...Object.values(hist || {}));
  return (
    <div className="v8-atlas-layerbar">
      {Array.from({ length: total }, (_, l) => {
        const n = (hist || {})[l] || 0;
        return (
          <i key={l} title={`L${l}：${n} 个通用头`}
             style={{ background: n ? `rgba(13,54,107,${0.18 + 0.72 * (n / max)})` : '#eef2f7' }} />
        );
      })}
      <span>L0</span><span>L{total - 1}</span>
    </div>
  );
}

/* 全局量级取三类型平均：只显示第一个类型会让人以为那是唯一值。 */
function avgGlobal(h) {
  const vs = Object.values(h.net_global || {}).map(Number).filter(Number.isFinite);
  return vs.length ? (vs.reduce((a, b) => a + b, 0) / vs.length).toFixed(1) : '—';
}

function HeadRow({ h, cats, nLayers }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr className={'v8-atlas-row' + (open ? ' open' : '')} onClick={() => setOpen(v => !v)}>
        <td className="h">L{h.block}·H{h.head}</td>
        {cats.map(c => (
          <td key={c} className="t"><span className="v8-atlas-t" style={tStyle(h.t[c] ?? 0)}>{(h.t[c] ?? 0).toFixed(1)}</span></td>
        ))}
        <td className="g">{avgGlobal(h)}</td>
        <td className="v">{h.cats ? h.cats.length : cats.length} 类</td>
      </tr>
      {open && (
        <tr className="v8-atlas-detail"><td colSpan={cats.length + 4}>
          层内相对推力：{cats.map(c => `${c} ${(h.net?.[c] ?? 0).toFixed(2)}`).join(' · ')}；
          全局相对量级：{cats.map(c => `${c} ${(h.net_global?.[c] ?? 0).toFixed(1)}`).join(' · ')}。
          读法：t 是跨题 t 检验（同号且 |t| &gt; {4} 才算承载）；层内相对值把每层自己的平均头贡献当 1，
          所以跨层可比、不会被「深层天然贡献大」带偏；全局量级说明它在输出端的绝对分量。
          {h.block < (nLayers ? nLayers - 12 : 24) ? ' 注意：这个头不在最后 12 层，跨类型稳定性较弱。' : ''}
        </td></tr>
      )}
    </>
  );
}

export default function LangAtlas() {
  const dsApi = useApi('/api/lang/atlas/datasets');
  const { status, setStatus, busy, setBusy, poll } = useStatusPoll();
  const [picked, setPicked] = useState(null);
  const [cats, setCats] = useState(CAT_ORDER);
  const [nPer, setNPer] = useState(0);
  const [err, setErr] = useState('');

  // 进页面先把上次结果读出来：默认数据集要跟着它走，否则会默认落到题量最大的那个、结果区显示「无记录」
  useEffect(() => {
    let alive = true;
    fetch(apiUrl('/api/lang/atlas/status')).then(r => r.json()).then(r => { if (alive) setStatus(r); }).catch(() => {});
    return () => { alive = false; };
  }, [setStatus]);

  const datasets = dsApi.data?.datasets || [];
  const active = picked || (layoutsDefault(datasets, status?.last));
  const cur = datasets.find(d => d.id === active);
  const last = status?.last || null;
  const running = ['loading', 'running'].includes(status?.state) || busy;

  const usable = useMemo(() => (cur ? cats.filter(c => (cur.by_category[c]?.n || 0) > 0) : []), [cur, cats]);
  const totalPick = useMemo(() => (cur ? usable.reduce((s, c) => s + (nPer > 0 ? Math.min(nPer, cur.by_category[c].n) : cur.by_category[c].n), 0) : 0), [cur, usable, nPer]);

  const start = async () => {
    setErr('');
    if (!active || !usable.length) { setErr('先选一个含三大类型的数据集，并至少勾一个类型。'); return; }
    setBusy(true);
    const q = new URLSearchParams({ dataset: active, categories: usable.join(','), n_per_category: String(nPer) });
    const r = await fetch(apiUrl(`/api/lang/atlas/run?${q}`), { method: 'POST' }).then(x => x.json()).catch(() => null);
    if (!r || !r.accepted) {
      setErr(r?.reason || '启动失败：本机中心没有响应。');
      setBusy(false);
      return;
    }
    setStatus({ state: 'loading', done: 0, total: 0, elapsed_s: 0, phase: '准备' });
    poll();
  };

  const toggle = c => setCats(prev => (prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]));
  const show = last && last.dataset === active ? last : null;
  // 跨数据集复现：同一批头在另一个数据集上是否也过阈。这条比任何单次运行都更能说明「规律」。
  const repro = useMemo(() => {
    if (!show) return null;
    const runs = (status?.runs || []).filter(r => r.dataset !== show.dataset && r.model_name === show.model_name);
    if (!runs.length) return null;
    const o = runs[runs.length - 1];
    const mine = new Set((show.common || []).map(h => `L${h.block}H${h.head}`));
    const theirs = o.heads || [];
    const shared = theirs.filter(x => mine.has(x));
    return { o, shared, mine: mine.size, total: theirs.length };
  }, [show, status]);
  const catsInResult = show ? Object.keys(show.by_category) : [];
  const nLayers = show?.n_heads_scanned ? Math.round(show.n_heads_scanned / 32) : 36;

  return (
    <div className="v8-atlas-root">
      <header className="v8-atlas-hd">
        <div>
          <h2>语言图谱 · 数据集实测</h2>
          <p>先选数据集，再对知识网络 / 分析推理 / 语法规则分别测一遍：答案位置的注意力落在哪个选项，
            以及每个头对正确字母的 logit 贡献。最后跨类型比对，把「普遍规律」和「类型专属」分开。</p>
        </div>
      </header>

      {/* ① 数据集 */}
      <section className="v8-atlas-step">
        <h3><span className="k">1</span>选数据集</h3>
        <p className="v8-q">数据集决定样本量与题型。同一批头在不同数据集上的功效差别很大：
          三大类型在 C-Eval 只有 169 题，在 CMMLU 能映射出 2300 题。</p>
        {!datasets.length && <Empty>{dsApi.error ? `读取数据集失败：${dsApi.error}` : '读取数据集…'}</Empty>}
        <div className="v8-atlas-dsrow">
          {datasets.map(d => <DatasetCard key={d.id} d={d} active={d.id === active} onPick={setPicked} />)}
        </div>
        {cur && (
          <details className="v8-atlas-map">
            <summary>类型 ↔ 科目映射（显式表，可审阅）</summary>
            <ul>{CAT_ORDER.map(c => (
              <li key={c}><b style={{ color: CAT_COLOR[c] }}>{c}</b> = {(cur.by_category[c]?.subjects || []).map(s2 => `${s2.name}(${s2.n})`).join(' + ') || '无'}</li>
            ))}</ul>
            <p className="v8-note">改动映射必须改服务端的 CATEGORY_SUBJECTS，界面不推断类型归属。</p>
          </details>
        )}
      </section>

      {/* ② 选类型 + 跑 */}
      <section className="v8-atlas-step">
        <h3><span className="k">2</span>选类型与样本量</h3>
        <div className="v8-atlas-ctrl">
          <div className="v8-atlas-catpick">
            {CAT_ORDER.map(c => {
              const n = cur?.by_category[c]?.n || 0;
              const on = cats.includes(c) && n > 0;
              return (
                <label key={c} className={'v8-atlas-chk' + (on ? ' on' : '') + (n ? '' : ' off')} style={on ? { borderColor: CAT_COLOR[c] } : undefined}>
                  <input type="checkbox" checked={on} disabled={!n} onChange={() => toggle(c)} />
                  <b style={{ color: on ? CAT_COLOR[c] : undefined }}>{c}</b>
                  <small>{n} 题可用</small>
                </label>
              );
            })}
          </div>
          <label className="v8-atlas-nper">每类题数
            <select value={nPer} onChange={e => setNPer(Number(e.target.value))}>
              <option value={0}>全部</option>
              <option value={10}>10（约 5 秒）</option>
              <option value={20}>20</option>
              <option value={40}>40</option>
            </select>
          </label>
          <button type="button" className="v8-atlas-run" onClick={start} disabled={running || !totalPick}>
            {running ? '实测中…' : `开始实测${totalPick ? `（${totalPick} 题）` : ''}`}
          </button>
        </div>
        {err && <p className="v8-atlas-err">{err}</p>}
        {running && (
          <div className="v8-atlas-prog">
            <div style={{ width: `${status?.total ? (100 * status.done) / status.total : 4}%` }} />
            <span>{status?.phase || '准备'} · {status?.done || 0}/{status?.total || '—'}{status?.elapsed_s ? ` · ${status.elapsed_s}s` : ''}</span>
          </div>
        )}
        {status?.state === 'error' && <p className="v8-atlas-err">实测失败：{status.error}</p>}
      </section>

      {/* ③ 结果 */}
      <section className="v8-atlas-step">
        <h3><span className="k">3</span>结果与普遍规律</h3>
        {!show && <Empty>{last ? `下面显示的是另一次实测（${last.dataset}）。切换数据集或重新运行后这里会更新。` : '还没有实测记录：选好数据集与类型后点「开始实测」，169 题约 20 秒。'}</Empty>}
        {show && (
          <>
            <div className="v8-atlas-sum">
              <div><small>模型</small><b>{show.model_name}</b></div>
              <div><small>实测量</small><b>{show.n_measured} 题{show.n_skipped ? `（跳过 ${show.n_skipped}）` : ''}</b></div>
              <div><small>用时</small><b>{show.elapsed_s}s</b></div>
              <div><small>扫描头</small><b>{show.n_heads_scanned}</b></div>
              <div><small>普遍头</small><b>{show.n_common ?? show.common.length}</b></div>
              <div><small>类型专属</small><b>{show.n_specific ?? show.specific.length}</b></div>
            </div>

            <div className="v8-atlas-cats">
              {catsInResult.map(c => {
                const v = show.by_category[c];
                const top = (v.top_heads || []).slice(0, 3);
                return (
                  <div key={c} className="v8-atlas-cat" style={{ borderTopColor: CAT_COLOR[c] }}>
                    <b style={{ color: CAT_COLOR[c] }}>{c}</b>
                    <span className="v8-atlas-acc">{(v.acc * 100).toFixed(1)}%<small> n={v.n}</small></span>
                    <ul>{top.map(h => <li key={`${h.block}-${h.head}`}>L{h.block}·H{h.head} <em>t {h.t.toFixed(1)}</em> <small>落点 {h.hit.toFixed(2)}</small></li>)}</ul>
                  </div>
                );
              })}
            </div>

            <div className="v8-atlas-verdict">
              <h4>判据与结论</h4>
              <p>
                同一 (层, 头) 在 <b>≥2 个类型</b>上 t 同号且 |t| &gt; {show.t_threshold} → 记为「普遍」；
                只在 1 个类型过阈且另一方向无强头 → 记为「类型专属」。阈值按 1152 个头做 Bonferroni 修正。
                落点 = 该头注意力质量最大的选项恰是正确选项的比例（四选一随机基线 {show.n_options_baseline}），它只说明「有没有在看选项」，不作承载判据。
              </p>
              {show.common.length ? (
                <p className="v8-atlas-finding">
                  共 <b>{show.n_common ?? show.common.length}</b> 个普遍头，全部落在 <b>L{Math.min(...show.common.map(x => x.block))}–L{Math.max(...show.common.map(x => x.block))}</b>。
                  前 {Math.min(...show.common.map(x => x.block))} 层没有任何一个头在三个类型上一致——答案选择不是逐层累加出来的，
                  而是在最后约三分之一网络里由少数几个头收口。最强的是
                  {' '}{show.common.slice(0, 3).map(x => `L${x.block}·H${x.head}`).join('、')}，三类型 t 值分别为
                  {' '}{show.common.slice(0, 3).map(x => `${Math.min(...Object.values(x.t)).toFixed(1)}~${Math.max(...Object.values(x.t)).toFixed(1)}`).join('、')}。
                </p>
              ) : (
                <p className="v8-atlas-finding">这一轮没有头同时通过两个类型的阈值：样本量不足或该模型不存在跨类型共用头。加样本量再看。</p>
              )}
              <LayerBar hist={show.layer_hist_common} nLayers={nLayers} />
              <p className="v8-note">上排每格是一层（左 = 第 0 层，右 = 最后一层），颜色深浅 = 该层有几个普遍头。</p>
            </div>

            {repro && (
              <div className="v8-atlas-repro">
                <h4>跨数据集复现</h4>
                <p>
                  同模型另一次实测：<b>{repro.o.dataset}</b>（{repro.o.n_measured} 题）。
                  本轮 {repro.mine} 个普遍头里，有 <b>{repro.shared.length}</b> 个在那次也过阈
                  （占对方 {repro.total} 个普遍头的 {repro.total ? Math.round((100 * repro.shared.length) / repro.total) : 0}%）。
                </p>
                <p className="v8-atlas-shared">{repro.shared.slice(0, 12).join('、')}{repro.shared.length > 12 ? ' …' : ''}</p>
                <p className="v8-note">
                  读法：两次都过阈的头才算稳定规律；只在一处出现的，更可能是小样本涨落或该数据集的题型偏好。
                  两个数据集的题量与科目构成差别越大，这条交集的可信度越高。
                </p>
              </div>
            )}

            <div className="v8-atlas-tables">
              <div>
                <h4>普遍头（≥2 类型）{(show.n_common ?? 0) > show.common.length ? ` · 只列 t 值最高的 ${show.common.length} 个 / 共 ${show.n_common}` : ''}</h4>
                {show.common.length ? (
                  <table className="v8-atlas-tbl">
                    <thead><tr><th>头</th>{catsInResult.map(c => <th key={c}>{c.slice(0, 2)}</th>)}<th>全局量级</th><th>覆盖</th></tr></thead>
                    <tbody>{show.common.map(h => <HeadRow key={`c${h.block}-${h.head}`} h={h} cats={catsInResult} nLayers={nLayers} />)}</tbody>
                  </table>
                ) : <Empty>无。</Empty>}
                <p className="v8-note">点一行展开该头的两个量。t 的正负是方向：正 = 把这个字母推上来，负 = 压下去（相邻头一正一负，通常是「取证据」与「排除干扰」的分工）。</p>
              </div>
              <div>
                <h4>类型专属头（只在一个类型过阈）{(show.n_specific ?? 0) > show.specific.length ? ` · 只列 ${show.specific.length} 个 / 共 ${show.n_specific}` : ''}</h4>
                {show.specific.length ? (
                  <table className="v8-atlas-tbl">
                    <thead><tr><th>头</th>{catsInResult.map(c => <th key={c}>{c.slice(0, 2)}</th>)}<th>全局量级</th><th>归属</th></tr></thead>
                    <tbody>{show.specific.map(h => <HeadRow key={`s${h.block}-${h.head}`} h={h} cats={catsInResult} nLayers={nLayers} />)}</tbody>
                  </table>
                ) : <Empty>无：这一轮所有过阈的头都跨了类型。</Empty>}
                <p className="v8-note">专属头才是「破解某一种语言能力」的入口：它只在这一类题目上过阈，说明承载的是类型特有计算，不是通用的答案收口。</p>
              </div>
            </div>

            <details className="v8-atlas-crit">
              <summary>这个测量的边界（读结论前先看）</summary>
              <ul>
                <li><b>dla 是线性化量。</b>头输出经 o_proj 分块到残差，再按该位置 RMSNorm 的 rms 作常数与 unembed 方向点积。RMSNorm 严格非线性，这只是标准的近似，因此只看头间相对大小，不当绝对 logit 用。</li>
                <li><b>相关性不是因果。</b>贡献大不等于模型必须用它。要判定承载，需消融该头后看正确率与 Δlogit 是否掉——这一步还没做，界面上不写作「承载」。</li>
                <li><b>单选题只测一种能力用法。</b>选字母依赖「选项文本可读 + 与题干匹配」，不覆盖生成、翻译、多步推理等用法。</li>
                <li><b>题量与题型会影响结论。</b>C-Eval 三大类共 169 题；换成 CMMLU 切片后重跑，看普遍头是否稳定复现——复现的才算规律。</li>
              </ul>
            </details>
          </>
        )}
      </section>
    </div>
  );
}
