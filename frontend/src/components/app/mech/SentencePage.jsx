/* 例句（核心页）：选一句话 → 行为（答案对数概率差）→ 注意力路由（层 × 头：直接作用与总效应并列）→ 激活的神经元 → 头电路保真度。
   数据：机制案例报告（conflict_case.v1，经本机数据包）。报告存的是每种线索在全部过门句子上的统计；
   单句的答案概率与单句归因图没有进报告，显示「未接入」，不用集合统计冒充单句。
   「输入一句话」的实时前向在「例句 → 输入一句话」（本机模型，只在本机中心可用）。 */
import { useMemo } from 'react';

import { Empty } from '../rdc_fusion/ui.jsx';
import PanelBoundary from '../PanelBoundary.jsx';
import { HeadHeatmap, LayerBars } from './MechCharts.jsx';
import { CaseHeader, CaseMissing } from './MechCase.jsx';
import { useConflictCase } from './mechData.js';
import { GROUP_EXPECT, GROUP_LABEL, cueLabel, exampleFor, fmt, gateFor, markAnswers, negate, pct, v1Void } from './mechUtil.js';

function Sentence({ report, cue, fact }) {
  const ex = exampleFor(report, cue);
  const gate = fact ? gateFor(report, fact) : null;
  if (!ex) return <Empty>报告里没有例句。</Empty>;
  return (
    <div className="wb9-sentence">
      <p className="wb9-sent-text" lang="zh">
        {markAnswers(ex.prompt, ex.context_answer, ex.memory_answer).map((p, i) => (
          p.mark ? <mark key={i} className={p.mark}>{p.text}</mark> : <span key={i}>{p.text}</span>
        ))}
        <span className="wb9-sent-cursor" aria-hidden="true">▁</span>
      </p>
      <div className="wb9-legend">
        <span><i className="ctx" />上下文答案「{ex.context_answer}」</span>
        <span><i className="mem" />记忆答案「{ex.memory_answer}」</span>
        <span>线索：{cueLabel(ex.cue)}</span>
        <span>读出位置：句末（下一个 token）</span>
      </div>
      {fact && (gate ? (
        <p className="v6-note">所选事实「{gate.subject}」：记忆答案 {gate.answer} · 反事实 {gate.counterfactual} · {gate.split === 'heldout' ? '留出' : '发现'}划分 ·
          行为门间隔 {fmt(gate.gate_margin)} · {gate.gated ? '✓ 过门' : '✗ 未过门（不进入机制分析）'}。报告只给出第一条过门事实的例句，上面的句子不是这一条。</p>
      ) : <p className="v6-note warn">报告的行为门记录里没有「{fact}」。</p>)}
    </div>
  );
}

function Behaviour({ report, cue }) {
  const row = report?.priority_table?.[cue];
  if (!row) return <Empty>未接入：报告没有线索「{cueLabel(cue)}」的行为记录。</Empty>;
  return (
    <>
      <div className="wb9-tiles">
        <div><b>{pct(row.context_win_rate)}</b><span>上下文答案胜出</span><small>{row.n} 句中（对数概率差 &gt; 0）</small></div>
        <div><b>{fmt(row.mean_d)}</b><span>平均对数概率差</span><small>log P(上下文答案) − log P(记忆答案)，完整答案</small></div>
        <div><b>{fmt(row.decisive_layer_median, 0)}</b><span>决出层（中位数）</span><small>logit lens 差值最后一次越过 0 后保持同号的层</small></div>
      </div>
      <div className="wb9-winbar" role="img" aria-label={`上下文胜 ${pct(row.context_win_rate)}`}>
        <i className="ctx" style={{ width: pct(row.context_win_rate, 1) }} /><i className="mem" />
      </div>
      <p className="v6-note">以上是这种线索在全部 {row.n} 个过门句子上的统计。单句的答案概率没有写进报告：未接入。</p>
    </>
  );
}

function Routing({ report, cue }) {
  const h = report?.heads || {};
  const top = useMemo(() => new Set((h.top_dla || []).map(x => x.head)), [h.top_dla]);
  const cueRow = report?.priority_table?.[cue];
  if (!Array.isArray(h.dla_discovery)) return <Empty>未接入：报告里没有逐头直接作用。</Empty>;
  return (
    <>
      {cue !== 'plain' && cueRow?.top5_heads_dla?.length > 0 && (
        <p className="v6-note">「{cueLabel(cue)}」直接作用前五的头：{cueRow.top5_heads_dla.map(x => `${x.head}（${fmt(x.dla, 3)}）`).join('、')}。
          下面两张热力图来自直接陈述（plain）的发现划分，总效应只在这一种线索上做过。</p>
      )}
      <div className="wb9-pair">
        <HeadHeatmap matrix={h.dla_discovery} title="直接作用（DLA）"
                     legend="头输出经最终归一化投到（上下文 − 记忆）答案方向；蓝 = 推高上下文答案" highlight={top} />
        {Array.isArray(h.total_effect_discovery)
          ? <HeadHeatmap matrix={negate(h.total_effect_discovery)} title="总效应（逐头均值消融）"
                         legend="颜色 = −（消融后 − 消融前）的对数概率差，即去掉该头后上下文优势的下降量；蓝 = 支持上下文答案" highlight={top} />
          : <Empty>未接入：报告里没有逐头总效应。</Empty>}
      </div>
      <p className="v6-note">描边 = 直接作用前 {top.size} 的头。两种量排名的重合：{h.overlap_top_dla_vs_total ?? '—'} / {top.size} 个头；
        秩相关 {fmt(h.spearman_dla_vs_total)}。单组件消融会被下游补偿误导，所以两者并列，不互相替代。</p>
      {(h.top_dla || []).length > 0 && (
        <table className="wb9-table">
          <thead><tr><th>头</th><th>直接作用</th><th>总效应</th><th>两者同向？</th></tr></thead>
          <tbody>{h.top_dla.slice(0, 10).map(x => (
            <tr key={x.head}><td className="mono">{x.head}</td><td>{fmt(x.dla, 4)}</td><td>{fmt(x.total_effect, 4)}</td>
              <td>{Number.isFinite(x.dla) && Number.isFinite(x.total_effect) ? (x.dla * -x.total_effect > 0 ? '✓ 同向' : '✗ 反向') : '—'}</td></tr>
          ))}</tbody>
        </table>
      )}
    </>
  );
}

function Neurons({ report, onUnit }) {
  const n = report?.neurons;
  if (!n?.top?.length) return <Empty>未接入：报告里没有神经元直接作用。</Empty>;
  const share = report?.component_share_plain;
  return (
    <>
      {share && (
        <p className="v6-note">直接陈述上，上下文优势的直接作用合计：注意力 {fmt(share.attention, 2)}，MLP {fmt(share.mlp, 2)}
          {report.dla_closure ? `（加嵌入直连后与真实首 token 差值的最大偏差 ${fmt(report.dla_closure.max_abs_gap, 3)}）` : ''}。</p>
      )}
      <LayerBars rows={[
        { id: 'attn', label: '注意力', values: n.attn_total_by_layer },
        { id: 'mlp', label: 'MLP', values: n.mlp_total_by_layer },
      ]} />
      <table className="wb9-table">
        <thead><tr><th>块.单元</th><th>直接作用（发现）</th><th>直接作用（留出）</th><th>留出同号</th><th /></tr></thead>
        <tbody>{n.top.slice(0, 15).map(u => (
          <tr key={`${u.layer}.${u.unit}`}>
            <td className="mono">L{u.layer}.U{u.unit}</td><td>{fmt(u.dla_discovery, 4)}</td><td>{fmt(u.dla_heldout, 4)}</td>
            <td>{Math.sign(u.dla_discovery) === Math.sign(u.dla_heldout) ? '✓' : '✗'}</td>
            <td><button type="button" className="fw-tbtn ghost" onClick={() => onUnit(u.layer, u.unit)}>单元卡 →</button></td>
          </tr>
        ))}</tbody>
      </table>
      <p className="v6-note">按 |发现划分直接作用| 排序的前 {n.top.length} 个单元中，留出划分同号的比例 {pct(n.sign_replication_heldout)}。直接作用只是读出近似，控制要看干预。</p>
    </>
  );
}

/* v2 三组头：留出划分上消融每组后的平均 d 变化 Δ，与 R 组随机同数量头的 Δ 并列；「有作用」= 方向符合预期且 |Δ| 超过全部随机组 */
function GroupTests({ v2 }) {
  const cues = Object.keys(v2?.by_cue || {});
  if (!cues.length) return null;
  return (
    <>
      <table className="wb9-table">
        <thead><tr><th>头组</th><th>预期</th>{cues.map(c => <th key={c}>{cueLabel(c)}：Δ（随机组 Δ）</th>)}<th>成员（前 6）</th></tr></thead>
        <tbody>{Object.keys(GROUP_LABEL).map(g => (
          <tr key={g}>
            <td>{GROUP_LABEL[g]}</td><td>{GROUP_EXPECT[g] < 0 ? 'Δ < 0（更偏记忆）' : 'Δ > 0（更偏上下文）'}</td>
            {cues.map(c => {
              const r = v2.by_cue[c]?.[g];
              const rnd = v2.by_cue[c]?.random_delta || [];
              return (
                <td key={c} className={r?.effect ? 'good' : r ? 'bad' : 'muted'}>
                  {r ? `${r.effect ? '✓' : '✗'} ${fmt(r.delta)}` : '—'}
                  <small className="wb9-sub">（{rnd.length ? `${fmt(Math.min(...rnd))} ~ ${fmt(Math.max(...rnd))}` : '—'}）</small>
                </td>
              );
            })}
            <td className="mono">{(v2.groups?.[g] || []).slice(0, 6).join(' ')}</td>
          </tr>
        ))}</tbody>
      </table>
      <p className="v6-note">留出平均 d（不消融）：{cues.map(c => `${cueLabel(c)} ${fmt(v2.by_cue[c]?.heldout_mean_d)}`).join('，')}。三组只用发现划分（直接陈述）选出。</p>
    </>
  );
}

function Fidelity({ report }) {
  const h = report?.heads || {};
  const suf = h.sufficiency_last_position;
  const nec = h.necessity;
  const v2 = report?.v2;
  const hier = v2?.hierarchy;
  const why = v1Void(report);
  if (!suf && !nec && !v2) return <Empty>未接入：报告里没有头电路检验。</Empty>;
  const tone = ok => (why ? 'void' : ok ? 'good' : 'bad');
  return (
    <>
      {(suf || nec) && (
        <div className="wb9-tiles">
          <div className={tone(suf?.sufficient)}><b>{why ? '失效' : pct(suf?.keep_fraction)}</b><span>头电路保真度（v1）</span>
            <small>只保留所选 {report?.criteria?.K ?? ''} 个头（其余在句末均值消融）时留出集上下文优势的保留比例{why ? `；报告值 ${pct(suf?.keep_fraction)} 不可解读` : ` · ${suf?.sufficient ? '✓ 达到' : '✗ 未达到'}预注册线 ${pct(report?.criteria?.min_keep)}`}</small></div>
          <div className={tone(nec?.necessary)}><b>{why ? '失效' : pct(nec?.drop_top)}</b><span>去掉所选头后的下降（v1）</span>
            <small>{why ? `报告值 ${pct(nec?.drop_top)} 不可解读` : `随机同数量头：${(nec?.drop_random || []).map(v => pct(v)).join(' / ') || '—'} · ${nec?.necessary ? '✓ 必要' : '✗ 未判为必要'}`}</small></div>
          {hier && (
            <div className={hier.holds ? 'good' : 'bad'}><b>{pct(hier.drop)}</b><span>层级 U → C（v2）</span>
              <small>消融 U 组后 C 组直接作用合计 {fmt(hier.C_dla_clean)} → {fmt(hier.C_dla_after_U_ablation)}；随机组后 {(hier.C_dla_after_random || []).map(v => fmt(v)).join(' / ')} · {hier.holds ? '✓ 上下游成立' : '✗ 未成立'}（线 {pct(v2?.criteria?.hier_min)}）</small></div>
          )}
        </div>
      )}
      {why && <p className="v6-note warn">{why}。v1 的两条判据在进展页记为「判据失效」，不算通过也不算未通过。</p>}
      <GroupTests v2={v2} />
      <p className="v6-note">这是「头电路」检验：只检验注意力头。图谱保真度（神经元与特征都接入的归因图）还没有；单句归因图：未接入。</p>
    </>
  );
}

export default function SentencePage({ st, set, onUnit }) {
  const { index, cases, caseId, api, report } = useConflictCase(st.case);
  if (!cases.length) return <CaseMissing index={index} />;
  const cues = (report?.examples || []).map(e => e.cue);
  const cue = cues.includes(st.cue) ? st.cue : (cues[0] || 'plain');
  const sec = (title, sub, body) => (
    <section className="wb9-sec" aria-label={title}>
      <header><h3>{title}</h3><small>{sub}</small></header>
      <PanelBoundary name={title} resetKey={`${caseId}|${cue}`}>{body}</PanelBoundary>
    </section>
  );
  return (
    <div className="wb9-page">
      <CaseHeader cases={cases} caseId={caseId} onCase={id => set({ case: id, cue: '', fact: '' })} report={report} />
      {api.error && <Empty>案例报告不在本机数据包里：{api.error}</Empty>}
      {report && (
        <>
          <nav className="wb9-chips" aria-label="线索">
            {cues.map(c => <button key={c} type="button" className={c === cue ? 'on' : ''} onClick={() => set({ cue: c })}>{cueLabel(c)}</button>)}
          </nav>
          {sec('① 句子', '报告里的例句；冲突 = 上下文给出反事实答案', <Sentence report={report} cue={cue} fact={st.fact} />)}
          {sec('② 行为', '模型更信上下文还是记忆', <Behaviour report={report} cue={cue} />)}
          {sec('③ 注意力路由', '层 × 头：直接作用与总效应并列', <Routing report={report} cue={cue} />)}
          {sec('④ 激活的神经元', 'MLP 单元对上下文优势的直接作用', <Neurons report={report} onUnit={onUnit} />)}
          {sec('⑤ 头电路检验', '去掉或只保留所选头能留下多少行为；三组头与上下游层级', <Fidelity report={report} />)}
        </>
      )}
    </div>
  );
}
