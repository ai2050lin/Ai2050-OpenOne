/* 模式与优先级：每种冲突一节——逐层赛跑曲线（logit lens 上 上下文答案 − 记忆答案）+ 优先级表
   （每种线索一行：上下文胜率、决出层、负责的头、相对直接陈述的变化、是否翻转胜负）。
   优先级 = 两种模式冲突时的胜者与决出层；层号先后不等于层级（设计 §1）。数据只来自冲突案例报告。 */
import { Empty } from '../rdc_fusion/ui.jsx';
import PanelBoundary from '../PanelBoundary.jsx';
import { CheckBoard, RaceChart } from './MechCharts.jsx';
import { useMechCase, useMechIndex } from './mechData.js';
import { CaseMissing } from './MechCase.jsx';
import { fmt, isConflict, pct, priorityRows, schemaLabel } from './mechUtil.js';

function ConflictMode({ summary, onSentence }) {
  const { report, error, loading } = useMechCase(summary.id);
  if (error) return <Empty>案例报告不在本机数据包里：{error}</Empty>;
  if (loading || !report) return <Empty>读取 {summary.id}…</Empty>;
  const rows = priorityRows(report);
  const series = rows.filter(r => r.race).map(r => ({ id: r.cue, label: r.label, values: r.race, bold: r.cue === 'plain', dashed: r.cue === 'negated' }));
  const neg = report.negated_control;
  return (
    <>
      {series.length ? <RaceChart series={series} /> : <Empty>未接入：报告里没有逐层赛跑曲线。</Empty>}
      <p className="v6-note">每条线 = 一种线索在全部过门句子上的平均 logit lens 读数（句末位置，中间层先经最终归一化再接 unembedding；横轴 0 = 嵌入层）。
        logit lens 是读出近似，不等于该层的完整计算作用。</p>
      <table className="wb9-table">
        <thead><tr><th>线索</th><th>句数</th><th>上下文胜率</th><th>相对直接陈述</th><th>决出层（中位）</th><th>负责的头（直接作用前五）</th><th>翻转胜负</th><th /></tr></thead>
        <tbody>{rows.map(r => (
          <tr key={r.cue} className={r.flips ? 'flip' : ''}>
            <td>{r.label}</td><td>{r.n ?? '—'}</td>
            <td><span className="wb9-minibar"><i style={{ width: pct(r.win, 1) }} /></span>{pct(r.win)}</td>
            <td>{r.delta == null ? '基准' : `${r.delta > 0 ? '+' : ''}${(r.delta * 100).toFixed(0)} 个百分点`}</td>
            <td>{fmt(r.decisive, 0)}</td>
            <td className="mono">{r.heads.map(h => h.head).join(' ') || '—'}</td>
            <td>{r.cue === 'plain' ? '—' : r.flips ? '⇄ 翻转' : '不翻转'}</td>
            <td><button type="button" className="fw-tbtn ghost" onClick={() => onSentence(summary.id, r.cue)}>例句 →</button></td>
          </tr>
        ))}</tbody>
      </table>
      {neg && (
        <p className={'v6-note' + (neg.copying_candidate ? ' warn' : '')}>
          否定对照（上下文出现反事实答案词但语义是否定）：上下文胜率 {pct(neg.context_win_rate)}。
          {neg.copying_candidate ? '≥ 50%：上下文胜出可能来自复制答案词而不是理解，上面的胜率不能直接读成「模型更信上下文」。' : '< 50%：没有出现只靠复制取胜的迹象。'}
        </p>
      )}
      {report.v2?.hierarchy && (
        <p className={'v6-note' + (report.v2.hierarchy.holds ? '' : ' warn')}>
          层级（模式 A 的输出成为模式 B 的输入）：消融 U 组（上游头）后，C 组（上下文写入头）的直接作用下降 {pct(report.v2.hierarchy.drop)}，
          {report.v2.hierarchy.holds ? '超过预注册线且低于全部随机组——U → C 上下游关系成立。' : '未达到预注册判据。'}三组头的消融检验见「例句 ⑤」。
        </p>
      )}
      <CheckBoard checks={summary.checks} voids={summary.void_checks} />
    </>
  );
}

export default function ModesPage({ onSentence }) {
  const index = useMechIndex();
  const conflicts = index.cases.filter(c => isConflict(c) && !c.error);
  if (!conflicts.length) return <div className="wb9-page"><CaseMissing index={index} /></div>;
  return (
    <div className="wb9-page">
      <p className="wb9-lede">每种冲突回答三件事：谁赢、在哪一层决出、哪些线索能翻转胜负。负责的头只列直接作用；总效应与保真度见「例句」。</p>
      {conflicts.map(c => (
        <section key={c.id} className="wb9-sec">
          <header><h3>{schemaLabel(c.schema)} · {c.set}</h3><small>{c.id} · 过行为门 {c.n_gated ?? '—'} / {c.n_facts ?? '—'}</small></header>
          <PanelBoundary name={c.id} resetKey={c.id}><ConflictMode summary={c} onSentence={onSentence} /></PanelBoundary>
        </section>
      ))}
      <p className="v6-note">尚未接入的冲突类型（语法与语义、否定与疑问覆盖、指令与常识）在报告写出之前不显示。</p>
    </div>
  );
}
