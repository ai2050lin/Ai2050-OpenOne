import { useMainline } from './data/mainlineState.js';
/* 能力矩阵（Phase 340，ResearchPlan_Final）：行 = 能力操作，按三种能力来源分组；列 = 六级证据状态。
   每格只读 data/mainline.js 的登记表；等级只描述证据走到哪一级，不代表机制已破解。点一行看来源与主张。 */
import { useState } from 'react';


import './client_guide.css';

export default function CapabilityMatrix({ onOpenClaim }) {
  const { CAPABILITIES, CAP_LEVELS, CLAIMS } = useMainline();
  const [sel, setSel] = useState('');
  const rows = CAPABILITIES.flatMap(([, ops]) => ops);
  const row = rows.find(item => item.op === sel);
  const claim = row?.claim ? CLAIMS.find(item => item.id === row.claim) : null;
  return <section className="v8-card cm" aria-label="能力矩阵">
    <h3>能力矩阵</h3>
    <p className="v8-note">行 = 能力操作（按三种能力来源分组），列 = 证据走到哪一级。点一行看来源和对应的主张。</p>
    <div className="v8-weight-scroll"><table className="cm-table">
      <thead><tr><th scope="col">能力操作</th><th scope="col">模式族</th>{CAP_LEVELS.map(level => <th scope="col" key={level}>{level}</th>)}</tr></thead>
      <tbody>{CAPABILITIES.map(([group, ops]) => [
        <tr className="cm-group" key={group}><td colSpan={CAP_LEVELS.length + 2}>{group}</td></tr>,
        ...ops.map(item => <tr key={item.op} className={sel === item.op ? 'on' : ''} tabIndex={0} aria-selected={sel === item.op}
          onClick={() => setSel(sel === item.op ? '' : item.op)} onKeyDown={event => { if (event.key === 'Enter') setSel(sel === item.op ? '' : item.op); }}>
          <td>{item.op}{item.claim && <span className="mp-tag">{item.claim}</span>}</td><td className="v8-note">{item.family}</td>
          {CAP_LEVELS.map((level, i) => <td key={level}><i className={`cm-cell${i <= item.level ? ' on' : ''}`} aria-label={i <= item.level ? `已到${level}` : `未到${level}`} /></td>)}
        </tr>),
      ])}</tbody>
    </table></div>
    {row ? <div className="mp-detail"><b>{row.op}</b> · {CAP_LEVELS[row.level]}<p className="v8-note">来源：{row.src}</p>
      {claim ? <p>主张 {claim.id}：{claim.title} · 正在 {claim.stage}{onOpenClaim && <> <button type="button" className="v6-linkbtn" onClick={onOpenClaim}>打开当前研发</button></>}</p>
        : <p className="v8-note">这一行还没有主张卡。机制检验要先写主张卡，再走 H0–H7。</p>}</div>
      : <p className="v8-note">模式族报告里的“未见词表可读出”只算到内部已观察；干预、选择性与自然生成都还没做，所以没有一行到“机制已检验”。</p>}
  </section>;
}
