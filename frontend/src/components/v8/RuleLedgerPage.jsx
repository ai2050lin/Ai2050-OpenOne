import { useEffect, useRef, useState } from 'react';

import { useResearchContext } from '../app/researchContext.js';
import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import { checkRules, ruleScope, SCOPE_FIELDS } from './ruleCheck.js';
import { workbenchHref } from './v8Util.js';

const PAGE_SIZE = 6;
const GROUPS = [
  { id: 'refuted', label: '负面规则', note: '在已测范围内失败的假设与做法' },
  { id: 'confirmed', label: '有效规则', note: '有实测支持的规律与方法' },
];
const RETEST = { block: '同范围直接引用', need_n: '复测需补样本', allow: '按范围判断复测' };
const DOMAINS = { measurement: '指标与对照', attention: '注意力头', 'attention-free static': '静态结构' };
const ruleTitle = rule => rule.title || rule.claim;

function RuleCheck({ rules, snapshot, disabled, onRule, prepared }) {
  const [query, setQuery] = useState('');
  const [scope, setScope] = useState({ model: snapshot, data: '', protocol: '' });
  const [submitted, setSubmitted] = useState(null);
  const criteria = useRef(null);
  const queryInput = useRef(null);

  useEffect(() => {
    if (!prepared) return;
    const known = ruleScope(prepared);
    setQuery(prepared.id);
    setScope({ model: known.modelId || snapshot, data: known.dataId, protocol: known.protocolId });
    setSubmitted(null);
    criteria.current.open = true;
    queryInput.current.focus();
  }, [prepared, snapshot]);

  const hits = submitted ? checkRules(rules, submitted.query, submitted.scope) : [];
  return <section className="v8-rules-check" aria-label="实验查重">
    <form onSubmit={event => {
      event.preventDefault();
      if (query.trim() && !disabled) setSubmitted({ query: query.trim(), scope });
    }}>
      <label htmlFor="rule-query">实验查重</label>
      <div className="v8-rules-checkrow">
        <input id="rule-query" ref={queryInput} value={query} maxLength={300}
          placeholder="输入实验、关键词或规则编号"
          onChange={event => { setQuery(event.target.value); setSubmitted(null); }} />
        <button type="submit" disabled={disabled || !query.trim()}>查重</button>
      </div>
      <details ref={criteria} className="v8-rules-criteria">
        <summary>核对模型、数据与实验口径</summary>
        <div className="v8-rules-fields">
          {SCOPE_FIELDS.map(({ key, label, placeholder }) => <label key={key}>{label}
            <input value={scope[key]} placeholder={placeholder} maxLength={220} spellCheck={false}
              onChange={event => { setScope({ ...scope, [key]: event.target.value }); setSubmitted(null); }} />
          </label>)}
        </div>
        <p className="v8-note">只填关键词也可查找。完整身份可从有登记的规则详情带入。</p>
      </details>
    </form>
    {submitted && <div className="v8-rules-results" aria-label="查重结果">
      <p role="status">{hits.length ? `找到 ${hits.length} 条相关记录。展开核对项可查看差异。` : '没有找到相近记录。可换关键词或按规则编号查询。'}</p>
      {hits.map(({ rule, decision, comparison }) => <article key={rule.id} className="v8-rules-hit">
        <div className="v8-rules-hithead">
          <button type="button" className="v6-linkbtn" onClick={() => onRule(rule)}>{rule.id} · {ruleTitle(rule)}</button>
          <strong>{decision}</strong>
        </div>
        <details>
          <summary>查看核对项</summary>
          <dl className="v8-rules-compare">
            {comparison.map(item => <div key={item.key}>
              <dt>{item.label} · {item.outcome}</dt>
              <dd>账本 {item.description || '未登记'}{item.registered && <code>{item.registered}</code>}</dd>
              <dd>本次 {item.supplied || '未填写'}</dd>
            </div>)}
          </dl>
          {rule.retest?.note && <p className="v8-note">复测条件 · {rule.retest.note}</p>}
        </details>
      </article>)}
    </div>}
  </section>;
}

function RuleCard({ rule, labels, open, onToggle, onRule, byId, onPrepare }) {
  const scope = ruleScope(rule);
  const retest = rule.retest || {};
  const source = rule.evidence || [];
  const related = [...new Set([...(rule.related || []), ...(rule.supersedes || [])])].filter(id => byId.has(id));
  const measure = Object.entries(rule.measure || {});
  return <article className="v8-rules-card">
    <button id={`rule-head-${rule.id}`} type="button" className="v8-rules-head" aria-expanded={open}
      aria-controls={open ? `rule-detail-${rule.id}` : undefined} onClick={onToggle}>
      <span className="v8-rules-id">{rule.id}</span>
      <span className="v8-rules-claim">{rule.status === 'refuted' && !rule.title && <small>已否证 · </small>}{ruleTitle(rule)}</span>
      <span className="v8-rules-cardmeta">{scope.model || '模型未登记'} · {labels[rule.domain] || DOMAINS[rule.domain] || rule.domain}</span>
      <span className="v8-rules-caret">{open ? '收起' : '展开'}</span>
    </button>
    {open && <div id={`rule-detail-${rule.id}`} className="v8-rules-body">
      {rule.title && <p className="v8-note">原命题 · {rule.claim}</p>}
      <div>
        <h4>适用范围</h4>
        <dl className="v8-rules-scope">
          {SCOPE_FIELDS.map(({ key, label }) => <div key={key}>
            <dt>{label}</dt><dd>{scope[key] || '未登记'}</dd>
          </div>)}
        </dl>
        <button type="button" className="fw-tbtn" onClick={() => onPrepare(rule)}>带入查重</button>
      </div>
      <div>
        <h4>实测结果</h4>
        {rule.criterion && <p className="v8-note">判据 · {rule.criterion}</p>}
        {measure.length ? <dl className="v8-rules-chips">
          {measure.map(([key, value]) => <div key={key}>
            <dt>{key}</dt><dd>{typeof value === 'object' ? JSON.stringify(value) : String(value)}</dd>
          </div>)}
        </dl> : <p className="v8-note">暂无独立数值摘要。</p>}
      </div>
      <div>
        <h4>反例与判读</h4>
        {rule.why && <p className="v8-note">{rule.why}</p>}
        {rule.counterexamples?.length > 0 && <ul>{rule.counterexamples.map((item, index) => <li key={index}>{item}</li>)}</ul>}
        {!rule.why && !rule.counterexamples?.length && <p className="v8-note">暂无单列记录，查看证据来源。</p>}
      </div>
      <div>
        <h4>复测条件</h4>
        <p className="v8-note">{[RETEST[retest.status], retest.note, retest.need_n ? `建议样本量 ${retest.need_n}` : ''].filter(Boolean).join(' · ') || '请结合适用范围和证据核对。'}</p>
      </div>
      {source.length > 0 && <ResearchDisclosure title="原始证据来源">
        <ul className="v8-rules-sources">
          {source.map((item, index) => <li key={index}>
            <b>{item.phase}</b> {item.note}
            {item.script && <code>{item.script}</code>}{item.data && <code>{item.data}</code>}
            {item.run && <a href={workbenchHref(new URLSearchParams({ lens: 'process', station: 'execution', run: item.run, ...(scope.modelId ? { snapshot: scope.modelId } : {}) }).toString())}>打开研究报告</a>}
            {item.artifact_sha256 && <code>SHA256 {item.artifact_sha256}</code>}
          </li>)}
        </ul>
      </ResearchDisclosure>}
      {related.length > 0 && <div className="v6-toolbar" aria-label="关联规则">
        {related.map(id => <button className="fw-tbtn" type="button" key={id} onClick={() => onRule(byId.get(id))}>{id} · {byId.get(id).topic}</button>)}
      </div>}
    </div>}
  </article>;
}

export default function RuleLedgerPage({ children }) {
  const ledger = useApi('/api/rules/overview');
  const { snapshot } = useResearchContext();
  const [group, setGroup] = useState('');
  const [selected, setSelected] = useState('');
  const [prepared, setPrepared] = useState(null);
  const [pendingOpen, setPendingOpen] = useState(false);
  const [limit, setLimit] = useState(PAGE_SIZE);
  const [domain, setDomain] = useState('');
  const focusRule = useRef('');
  const rules = ledger.data?.rules || [];
  const byId = new Map(rules.map(rule => [rule.id, rule]));
  const labels = ledger.data?.labels?.domain || {};
  const pending = rules.filter(rule => !GROUPS.some(item => item.id === rule.status));
  const active = rules.filter(rule => rule.status === group && (!domain || rule.domain === domain));
  const ordered = [...active].sort((a, b) => Number(Boolean(b.title)) - Number(Boolean(a.title)));
  const visible = ordered.slice(0, limit);
  const selectedRule = byId.get(selected);
  if (selectedRule?.status === group && !visible.includes(selectedRule)) visible.unshift(selectedRule);

  useEffect(() => {
    if (!focusRule.current) return;
    const target = document.getElementById(`rule-head-${focusRule.current}`);
    if (target) {
      focusRule.current = '';
      target.focus();
    }
  });

  function choose(next, ruleId = '') {
    setGroup(next);
    setSelected(ruleId);
    setDomain('');
    setLimit(PAGE_SIZE);
  }

  function openRule(rule) {
    if (pending.includes(rule)) setPendingOpen(true);
    focusRule.current = rule.id;
    choose(GROUPS.some(item => item.id === rule.status) ? rule.status : '', rule.id);
  }

  function renderRule(rule) {
    return <RuleCard key={rule.id} rule={rule} labels={labels} open={selected === rule.id} byId={byId}
      onToggle={() => setSelected(selected === rule.id ? '' : rule.id)} onRule={openRule}
      onPrepare={item => setPrepared({ item: { ...item }, snapshot })} />;
  }

  return <section className="v8-route v8-rule-ledger" aria-labelledby="rules-title">
    <header className="v8-overview-heading"><h2 id="rules-title">成果与证据</h2></header>
    <RuleCheck key={snapshot} rules={rules} snapshot={snapshot} disabled={!rules.length} onRule={openRule}
      prepared={prepared?.snapshot === snapshot ? prepared.item : null} />
    <div className="v8-rules-groups" role="group" aria-label="规则分类">
      {GROUPS.map(item => <button key={item.id} type="button" className="v8-card v8-direction v8-rule-group"
        aria-expanded={group === item.id} aria-controls={group === item.id ? 'rule-results' : undefined}
        onClick={() => choose(group === item.id ? '' : item.id)}>
        <strong>{item.label}</strong>
        <span>{ledger.data ? rules.filter(rule => rule.status === item.id).length : '—'} 条</span>
        <span className="v8-rule-group-caret" aria-hidden="true">{group === item.id ? '▴' : '▾'}</span>
        <small>{item.note}</small>
      </button>)}
    </div>
    {ledger.error ? <Empty>规则账本读取失败 · {ledger.error} <button className="fw-tbtn" type="button" onClick={ledger.reload}>重试</button></Empty>
      : !ledger.data ? <Empty>读取规则账本…</Empty> : null}
    {group && <section id="rule-results" className="v8-rules-list" aria-label={GROUPS.find(item => item.id === group).label}>
      <details className="v8-rules-criteria">
        <summary>筛选规则</summary>
        <label>研究领域 <select value={domain} onChange={event => { setDomain(event.target.value); setSelected(''); setLimit(PAGE_SIZE); }}>
          <option value="">全部领域</option>
          {[...new Set(rules.filter(rule => rule.status === group).map(rule => rule.domain))].map(key =>
            <option key={key} value={key}>{labels[key] || DOMAINS[key] || key}</option>)}
        </select></label>
      </details>
      {visible.length ? visible.map(renderRule) : <Empty>当前分类暂无规则。</Empty>}
      {active.length > limit && <button type="button" className="fw-tbtn" onClick={() => setLimit(limit + PAGE_SIZE)}>
        显示更多规则（剩 {active.length - limit} 条）
      </button>}
    </section>}
    {pending.length > 0 && <details className="v8-card v8-overview-fold" open={pendingOpen}
      onToggle={event => setPendingOpen(event.currentTarget.open)}>
      <summary>未决与待验证 · {pending.length} 条</summary>
      {pendingOpen && <div className="v8-rules-list">{pending.map(renderRule)}</div>}
    </details>}
    {children}
  </section>;
}
