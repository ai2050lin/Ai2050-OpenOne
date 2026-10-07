import { useEffect, useState } from 'react';
import { AlertTriangle, ChevronDown, GraduationCap, Layers } from 'lucide-react';

import { WorldNav } from './WorldNav';
import { EvidenceBadge, EVIDENCE_LEVELS } from './EvidenceBadge';
import './LearnHub.css';

/**
 * 学习世界（M2）：
 * 数据源 = researchctl export-client 投影的 registry 只读副本：
 *   /research_data/current/cases.json    —— 10 个封存案例（CASE-001..010）
 *   /research_data/current/industry.json —— 12 个领域方法节点
 * 纪律：案例数字与结论均来自 registry 投影，证据等级强制展示；
 *       本页是"解释身份"，不产生新的事实。
 */
const DATA_BASE = '/research_data/current';

function useLedger(path) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch(path)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j) => { if (alive) setData(j); })
      .catch((e) => { if (alive) setError(String(e)); });
    return () => { alive = false; };
  }, [path]);
  return { data, error };
}

function MethodCard({ method }) {
  return (
    <article className="learn-hub__method">
      <header>
        <strong>{method.name || method.title}</strong>
        <span className="learn-hub__era">{method.era}</span>
        <EvidenceBadge level={method.evidence_grade} title={`方法证据级：${method.id}`} />
      </header>
      <p className="learn-hub__idea">{method.core_idea}</p>
      <p className="learn-hub__established"><b>已确立：</b>{method.established}</p>
      {Array.isArray(method.limits) && method.limits.length > 0 && (
        <ul className="learn-hub__limits">
          {method.limits.map((lim, i) => <li key={i}>{lim}</li>)}
        </ul>
      )}
      {Array.isArray(method.key_works) && method.key_works.length > 0 && (
        <p className="learn-hub__works">{method.key_works.join(' · ')}</p>
      )}
    </article>
  );
}

function CaseCard({ item, open, onToggle }) {
  return (
    <article className={`learn-hub__case${open ? ' learn-hub__case--open' : ''}`}>
      <button type="button" className="learn-hub__case-head" onClick={onToggle} aria-expanded={open}>
        <span className="learn-hub__case-id">{item.id}</span>
        <span className="learn-hub__case-title">{item.title}</span>
        <EvidenceBadge level={item.evidence_level} title={`案例证据级：${item.id}`} />
        <ChevronDown size={15} className="learn-hub__chev" />
      </button>
      {open && (
        <div className="learn-hub__case-body">
          <p className="learn-hub__question"><b>问题：</b>{item.question}</p>
          <p className="learn-hub__narrative">{item.narrative}</p>
          {Array.isArray(item.limitations) && item.limitations.length > 0 && (
            <div className="learn-hub__limits-box">
              <b>局限与适用范围</b>
              <ul>{item.limitations.map((lim, i) => <li key={i}>{lim}</li>)}</ul>
            </div>
          )}
          <p className="learn-hub__meta">
            <span>来源：{item.phase_ref}</span>
            {item.run_ref ? <span>产物：{item.run_ref}</span> : null}
            {Array.isArray(item.source_refs) && item.source_refs.length > 0 && (
              <span>回查：{item.source_refs.join(' ; ')}</span>
            )}
          </p>
        </div>
      )}
    </article>
  );
}

export function LearnHub() {
  const cases = useLedger(`${DATA_BASE}/cases.json`);
  const industry = useLedger(`${DATA_BASE}/industry.json`);
  const [openId, setOpenId] = useState('CASE-001');

  const caseList = Array.isArray(cases.data) ? cases.data : [];
  const methods = Array.isArray(industry.data)
    ? industry.data.filter((r) => r.kind === 'method')
    : [];

  return (
    <main className="learn-hub" aria-label="学习世界">
      <WorldNav active="learn" />

      <header className="learn-hub__hero">
        <p className="learn-hub__eyebrow"><GraduationCap size={13} /> 学习世界 · Learn</p>
        <h1>从"结论是如何被证出来的"开始学机制可解释性。</h1>
        <p className="learn-hub__sub">
          每个案例都来自一个已封存的研究 Phase：问题 → 假设 → 实验 → 结论与局限。
          数字不是教学化简，是注册在案的封存结果；证据等级标在每张卡上。
        </p>
      </header>

      {(cases.error || industry.error) && (
        <div className="learn-hub__warn" role="alert">
          <AlertTriangle size={14} />
          数据投影加载失败（{cases.error || industry.error}）。请先在仓库根目录运行：
          <code>python ai2050_research_os/scripts/researchctl.py export-client</code>
        </div>
      )}

      <section aria-label="领域方法图谱">
        <h2><Layers size={15} /> 领域方法图谱（{methods.length} 个方法节点）</h2>
        <p className="learn-hub__section-note">
          行业方法按"证明了什么 / 在哪失效"登记，证据级是本平台对行业声明的标定（多数停留在"已有观察"级）。
          数据来源：registry/industry.json（candidate 状态，逐条核对原文后升格 verified）。
        </p>
        <div className="learn-hub__methods">
          {methods.map((m) => <MethodCard key={m.id} method={m} />)}
        </div>
      </section>

      <section aria-label="封存案例">
        <h2>封存案例（{caseList.length} 个，点击展开）</h2>
        <p className="learn-hub__section-note">
          全部案例从 AGI_DEEPSEEK_MEMO 已封存 Phase 派生；引用数字先经 MEMO 核对再入库。
        </p>
        <div className="learn-hub__cases">
          {caseList.map((c) => (
            <CaseCard
              key={c.id}
              item={c}
              open={openId === c.id}
              onToggle={() => setOpenId(openId === c.id ? null : c.id)}
            />
          ))}
        </div>
      </section>

      <section className="learn-hub__legend" aria-label="证据标尺图例">
        <h2>四级证据标尺</h2>
        <div className="learn-hub__legend-row">
          {Object.entries(EVIDENCE_LEVELS).map(([key, meta]) => (
            <EvidenceBadge key={key} level={key} title={`证据等级：${key}`} />
          ))}
        </div>
        <p className="learn-hub__dim">
          行业方法与自产证据用同一把尺子量；"有机制证据"是最高级，本页所有条目均未达到——这正是可以一起补的空白。
        </p>
      </section>
    </main>
  );
}

export default LearnHub;
