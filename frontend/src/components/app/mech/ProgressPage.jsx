/* 进展：只放三类数——各模式的保真度、完成的完整案例数、未通过的检验。失败与成功同样醒目（同尺寸并排）。
   数据：机制案例与方法校准报告的索引摘要（/api/mechanism/index）。保真度目前只有「头电路保真度」（只检验注意力头）；
   图谱保真度（神经元 / 特征接入后的归因图，AGENTS §5.4）在算出之前显示「未接入」，不用其他数代替。
   完成案例只表示报告字段齐全；检验通过只描述预注册判据，都不改变证据等级。 */
import { Suspense, lazy } from 'react';

import { Empty } from '../rdc_fusion/ui.jsx';
import PanelBoundary from '../PanelBoundary.jsx';
import { CheckBoard } from './MechCharts.jsx';
import { useMechIndex } from './mechData.js';
import { CASE_PARTS, caseComplete, checkLabel, fmt, isConflict, pct, schemaLabel } from './mechUtil.js';

const RuleLedgerPage = lazy(() => import('../../v8/RuleLedgerPage.jsx'));

function Fidelity({ cases }) {
  const conflicts = cases.filter(isConflict);
  if (!conflicts.length) return <Empty>未接入：还没有机制案例报告。</Empty>;
  return (
    <table className="wb9-table">
      <thead><tr><th>模式</th><th>案例</th><th>头电路保真度</th><th>去掉所选头后的下降</th><th>图谱保真度</th></tr></thead>
      <tbody>{conflicts.map(c => (
        <tr key={c.id}>
          <td>{schemaLabel(c.schema)}</td><td className="mono">{c.id}</td>
          {c.fidelity_void ? (
            <td colSpan={2} className="muted">⊘ 失效（报告值 {pct(c.head_circuit_keep_fraction)} / {pct(c.head_necessity_drop)} 不可解读）：{c.fidelity_void}</td>
          ) : (
            <>
              <td className={c.checks?.head_sufficiency_last_position === false ? 'bad' : c.checks?.head_sufficiency_last_position ? 'good' : ''}>
                {pct(c.head_circuit_keep_fraction)} {c.checks?.head_sufficiency_last_position === false ? '✗' : c.checks?.head_sufficiency_last_position ? '✓' : ''}</td>
              <td className={c.checks?.head_necessity === false ? 'bad' : c.checks?.head_necessity ? 'good' : ''}>
                {pct(c.head_necessity_drop)} {c.checks?.head_necessity === false ? '✗' : c.checks?.head_necessity ? '✓' : ''}</td>
            </>
          )}
          <td className="muted">未接入</td>
        </tr>
      ))}</tbody>
    </table>
  );
}

function Cases({ cases }) {
  const conflicts = cases.filter(isConflict);
  const done = conflicts.filter(caseComplete);
  return (
    <>
      <div className="wb9-tiles">
        <div><b>{done.length}</b><span>完整案例</span><small>共 {conflicts.length} 个机制案例；完整 = 下表五项都在报告里</small></div>
      </div>
      {conflicts.length > 0 && (
        <table className="wb9-table">
          <thead><tr><th>案例</th>{CASE_PARTS.map(([k, t]) => <th key={k}>{t}</th>)}</tr></thead>
          <tbody>{conflicts.map(c => (
            <tr key={c.id}><td className="mono">{c.id}</td>
              {CASE_PARTS.map(([k]) => <td key={k} className={c.has?.[k] ? 'good' : 'bad'}>{c.has?.[k] ? '✓ 有' : '✗ 缺'}</td>)}</tr>
          ))}</tbody>
        </table>
      )}
    </>
  );
}

function Checks({ cases }) {
  const all = {};
  const voids = {};
  for (const c of cases) {
    for (const [k, v] of Object.entries(c.checks || {})) all[`${c.id} · ${k}`] = v;
    for (const [k, v] of Object.entries(c.void_checks || {})) voids[`${c.id} · ${k}`] = v;
  }
  const errors = cases.filter(c => c.error);
  return (
    <>
      <CheckBoard checks={all} voids={voids} empty="未接入：报告里还没有判据。" />
      {errors.length > 0 && <p className="v6-note warn">无法解析的报告：{errors.map(c => c.id).join('、')}（不计入上面的检验）。</p>}
      <table className="wb9-table">
        <thead><tr><th>报告</th><th>类型</th><th>模型</th><th>整体</th><th>判据</th></tr></thead>
        <tbody>{cases.filter(c => !c.error).map(c => (
          <tr key={c.id}>
            <td className="mono">{c.id}</td><td>{c.kind === 'calibration' ? '方法校准' : '机制案例'} · {schemaLabel(c.schema)}</td>
            <td>{String(c.model || '—').split(/[\\/]/).pop()}</td>
            <td className={c.passed === false ? 'bad' : c.passed ? 'good' : ''}>{c.passed == null ? '—' : c.passed ? '✓ 通过' : '✗ 未通过'}</td>
            <td>{Object.entries(c.checks || {}).map(([k, v]) => <span key={k} className={'wb9-ck ' + (v ? 'good' : 'bad')} title={checkLabel(k)}>{v ? '✓' : '✗'} {k}</span>)}</td>
          </tr>
        ))}</tbody>
      </table>
      <p className="v6-note">「方法校准」在答案已知的 GPT-2 上检验方法本身（设计 §2「校准先行」）；校准未通过的检验方法，在主模型上的同类结论要降级看待。
        秒数：{cases.filter(c => Number.isFinite(c.seconds)).map(c => `${c.id} ${fmt(c.seconds, 0)} s`).join('、') || '—'}。</p>
    </>
  );
}

export default function ProgressPage() {
  const index = useMechIndex();
  const sec = (title, sub, body) => (
    <section className="wb9-sec" aria-label={title}>
      <header><h3>{title}</h3><small>{sub}</small></header>
      <PanelBoundary name={title}>{body}</PanelBoundary>
    </section>
  );
  return (
    <div className="wb9-page">
      {index.error && <Empty>未接入：机制报告索引不在本机数据包里（{index.error}）。</Empty>}
      {index.loading && <Empty>读取…</Empty>}
      {!index.loading && !index.error && (
        <>
          {sec('各模式保真度', '只保留所选成分时留下多少行为（留出材料）', <Fidelity cases={index.cases} />)}
          {sec('完成的完整案例', '报告五项齐全的机制案例数', <Cases cases={index.cases} />)}
          {sec('检验：未通过与通过', '机制案例与方法校准的全部预注册判据；未通过排在前面', <Checks cases={index.cases} />)}
        </>
      )}
      <details className="wb9-sec wb9-details">
        <summary>已测规则账本（含被否证的规则；只在本机中心可读）</summary>
        <PanelBoundary name="规则账本"><Suspense fallback={<Empty>加载中…</Empty>}><RuleLedgerPage /></Suspense></PanelBoundary>
      </details>
    </div>
  );
}
