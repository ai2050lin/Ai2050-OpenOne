/* 研发主线的进度部件：进度条、进度块、阶段条、过门清单、冻结向导与锁箱看管。
   约束：百分比只在分母明确时显示；进度只表示执行到哪里，不表示结论成立。 */
import { useState } from 'react';

import { LOCKBOXES, STAGES, achievedPower, nRequired, pct, stageProgress } from './data/mainline.js';
import './client_guide.css';

export function PBar({ done, total, label }) {
  const p = pct(done, total);
  const width = p == null ? 0 : Math.max(Number(p) > 0 ? 1 : 0, Number(p));
  return <div className="mp-bar" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={total || 0} aria-valuenow={done ?? 0}>
    <i style={{ width: `${width}%` }} /></div>;
}

export function PTile({ title, done, total, unit, on, onClick, empty }) {
  const p = pct(done, total);
  const body = <>
    <span className="mp-t">{title}</span>
    <span className={'mp-v' + (p == null ? ' muted' : '')}>{p == null ? (empty || '未统计') : `${p}%`}</span>
    {p != null && <PBar done={done} total={total} label={`${title} ${p}%`} />}
    <span className="mp-n">{p == null ? ' ' : `${done.toLocaleString()} / ${total.toLocaleString()} ${unit}`}</span>
  </>;
  return onClick ? <button type="button" className={'mp-tile' + (on ? ' on' : '')} aria-expanded={!!on} onClick={onClick}>{body}</button>
    : <div className="mp-tile">{body}</div>;
}

export function Fold({ title, children, open = false }) {
  const [shown, setShown] = useState(open);
  return <details className="mp-fold" open={shown} onToggle={event => setShown(event.currentTarget.open)}>
    <summary>{title}</summary>{shown && <div className="mp-fold-body">{children}</div>}</details>;
}

export function GateList({ stage }) {
  return <>
    <p className="v8-note"><b>{stage.id} 完成定义：</b>{stage.def}</p>
    <ul className="mp-gates">{stage.gates.map((gate, i) => <li key={gate} className={stage.done.includes(i) ? 'ok' : ''}>
      <span aria-hidden="true" className="mp-box">{stage.done.includes(i) ? '✓' : ''}</span>{gate}
      <span className="mp-sr">{stage.done.includes(i) ? '已满足' : '未满足'}</span></li>)}</ul>
    <p className="v8-note">全部满足才进入下一阶段；每一项都要链接到运行记录。</p>
  </>;
}

/* 主线总进度：H0–H7 完成阶段数。 */
export function MainlineCard() {
  const [sel, setSel] = useState(null);
  const { done, total, next } = stageProgress();
  const stage = sel == null ? null : STAGES[sel];
  return <section className="v8-card mp-main" aria-label="主线总进度">
    <span className="mp-t">主线总进度</span>
    <div className="mp-hero">{pct(done, total)}%</div>
    <PBar done={done} total={total} label="主线完成阶段" />
    <p className="v8-note">{done} / {total} 阶段完成{next ? <> · 下一步 <b>{next.id} {next.name}</b></> : ''}</p>
    <div className="mp-stages" role="group" aria-label="主线阶段">{STAGES.map((item, i) => {
      const finished = item.done.length === item.gates.length;
      return <button type="button" key={item.id} className={(finished ? 'done' : item === next ? 'next' : '') + (sel === i ? ' on' : '')}
        aria-expanded={sel === i} title={item.name} onClick={() => setSel(sel === i ? null : i)}>{item.id}</button>;
    })}</div>
    {stage && <div className="mp-detail"><b>{stage.id} {stage.name}</b> · {stage.done.length === stage.gates.length ? '已完成' : stage === next ? '下一步' : '未开始'}<GateList stage={stage} /></div>}
    <p className="v8-note">进度只表示执行到哪里，不表示结论成立。</p>
  </section>;
}

/* 冻结向导：AGENTS 第 13 节的功效三栏。只做计算，不写入任何冻结记录。 */
export function PowerWizard() {
  const [rows, setRows] = useState([['主效应', '0.05', ''], ['两因素交互', '0.05', ''], ['未见组合', '0.05', '']]);
  const [pD, setPD] = useState('0.10');
  const edit = (i, k, v) => setRows(rows.map((row, j) => j === i ? row.map((x, m) => m === k ? v : x) : row));
  const p = Number(pD);
  return <div>
    <div className="v8-weight-scroll"><table className="v6-table">
      <thead><tr><th>检验</th><th>预期效应量 Δ</th><th>所需 n</th><th>计划 n</th><th>检验力</th></tr></thead>
      <tbody>{rows.map(([name, delta, n], i) => {
        const d = Number(delta), need = nRequired(d, p), pw = achievedPower(Number(n), d, p);
        return <tr key={name}><td>{name}</td>
          <td><input aria-label={`${name}预期效应量`} className="mp-num" value={delta} onChange={e => edit(i, 1, e.target.value)} /></td>
          <td>{need ?? '—'}</td>
          <td><input aria-label={`${name}计划样本数`} className="mp-num" value={n} placeholder="填写" onChange={e => edit(i, 2, e.target.value)} /></td>
          <td>{pw == null ? '—' : <span className={'mp-tag ' + (pw >= 0.8 ? 'ok' : 'warn')}>{pw.toFixed(3)} {pw >= 0.8 ? '通过' : '不足'}</span>}</td></tr>;
      })}</tbody></table></div>
    <label className="v8-note">预期翻转率 p_d <input aria-label="预期翻转率" className="mp-num" value={pD} onChange={e => setPD(e.target.value)} /></label>
    <p className="v8-note">公式与 tools/power_analysis.py 相同：n = (z₁₋α/₂ + z_power)² · p_d / Δ²，α = 0.05。检验力 ≥ 0.8 才能冻结；冻结后阈值只读，正式数字以 Python 工具的输出为准。</p>
  </div>;
}

const LOCK_STATE = { opened: ['已开封', 'warn'], sealed: ['封存中', 'ok'], not_built: ['未建', 'gap'] };
export function LockboxTable() {
  return <div className="v8-weight-scroll"><table className="v6-table">
    <thead><tr><th>锁箱</th><th>状态</th><th>记录</th></tr></thead>
    <tbody>{LOCKBOXES.map(box => <tr key={box.id}><td>{box.id}</td>
      <td><span className={`mp-tag ${LOCK_STATE[box.state][1]}`}>{LOCK_STATE[box.state][0]}</span></td>
      <td>{box.note}{box.phase ? `（Phase ${box.phase}）` : ''}</td></tr>)}</tbody></table>
    <p className="v8-note">已开封的锁箱不能再用于调参后的确认。</p></div>;
}
