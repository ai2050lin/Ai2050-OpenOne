/* 研究工作台 · 进展 · 规则账本
   目的不是"再展示一次结论"，而是防止重复劳动：每条规则给结论状态、关键数字、证据链，
   以及明确的重测判定（不要重测 / 重测需加样本 / 可以重测）。
   查重框可以贴一句"我准备测什么"，先看这件事有没有已经做过。
   数据来自服务器 /api/rules/overview（本机 data/rules/ledger.json）。 */
import { useEffect, useMemo, useState } from 'react';

import { API_BASE } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import '../app/workbench.css';
import './v8.css';

const STATUS_STYLE = {
  confirmed: { text: '已证实', ink: '#0f6e56', bg: '#e1f5ee', line: '#9fe1cb', sym: '✓' },
  refuted: { text: '已否证', ink: '#993c1d', bg: '#faece7', line: '#f5c4b3', sym: '✗' },
  inconclusive: { text: '未决', ink: '#854f0b', bg: '#faeeda', line: '#fac775', sym: '◐' },
  superseded: { text: '已被修正', ink: '#5f5e5a', bg: '#f1efe8', line: '#d3d1c7', sym: '↻' },
  open: { text: '待测', ink: '#185fa5', bg: '#e6f1fb', line: '#b5d4f4', sym: '○' },
};

const RETEST_STYLE = {
  block: { text: '不要重测', ink: '#993c1d', bg: '#faece7' },
  need_n: { text: '重测需加样本', ink: '#854f0b', bg: '#faeeda' },
  allow: { text: '可以重测', ink: '#0f6e56', bg: '#e1f5ee' },
};

const DOMAINS = [
  ['', '全部'],
  ['mlp', 'MLP 单元'],
  ['attn', '注意力头'],
  ['metric', '指标与对照'],
  ['method', '实验方法'],
];

function Pill({ style, children }) {
  if (!style) return null;
  return (
    <span className="v8-rules-pill" style={{ color: style.ink, background: style.bg, borderColor: style.line || style.bg }}>
      {children}
    </span>
  );
}

function MeasureChips({ measure }) {
  const entries = Object.entries(measure || {});
  if (!entries.length) return null;
  return (
    <div className="v8-rules-chips">
      {entries.map(([k, v]) => (
        <span key={k}>
          <i>{k}</i>
          <b>{Array.isArray(v) ? v.join(' / ') : String(v)}</b>
        </span>
      ))}
    </div>
  );
}

function RuleCard({ rule, labels }) {
  const [open, setOpen] = useState(false);
  const st = STATUS_STYLE[rule.status] || STATUS_STYLE.open;
  const rt = rule.retest || {};
  const rts = RETEST_STYLE[rt.status] || RETEST_STYLE.allow;
  return (
    <div className={'v8-rules-card' + (rt.status === 'block' ? ' blocked' : '')}>
      <button type="button" className="v8-rules-head" onClick={() => setOpen(o => !o)}>
        <span className="v8-rules-id">{rule.id}</span>
        <span className="v8-rules-claim">{rule.claim}</span>
        <span className="v8-rules-tags">
          <Pill style={st}>{st.sym} {st.text}</Pill>
          <Pill style={rts}>{rts.text}</Pill>
          <span className="v8-rules-count">测过 {rt.count || 0} 次</span>
          <span className="v8-rules-caret">{open ? '收起' : '展开'}</span>
        </span>
      </button>
      {open && (
        <div className="v8-rules-body">
          <MeasureChips measure={rule.measure} />
          {rule.criterion && (
            <p className="v8-rules-line"><b>判据</b>{rule.criterion}</p>
          )}
          {rule.why && <p className="v8-rules-line"><b>判读</b>{rule.why}</p>}
          {rt.note && (
            <p className="v8-rules-line"><b>复查</b>{rt.note}{rt.need_n ? `（重测至少要 ${rt.need_n} 题量级）` : ''}</p>
          )}
          {(rule.evidence || []).length > 0 && (
            <div className="v8-rules-ev">
              <b>证据</b>
              <ul>
                {rule.evidence.map((e, i) => (
                  <li key={i}>
                    <span className="v8-rules-ev-phase">{e.phase}</span>
                    {e.note}
                    <code>{e.script}</code>
                    <code>{e.data}</code>
                    {e.model && <small>{e.model}</small>}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {labels?.domain?.[rule.domain] && (
            <p className="v8-rules-line dim"><b>领域</b>{labels.domain[rule.domain]} · 置信 {rule.confidence}</p>
          )}
        </div>
      )}
    </div>
  );
}

function RuleCheck() {
  const [q, setQ] = useState('');
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    if (!q.trim()) return;
    setBusy(true);
    try {
      const r = await fetch(`${API_BASE}/api/rules/check?q=${encodeURIComponent(q)}`, { cache: 'no-store' });
      setRes(await r.json());
    } catch (e) {
      setRes({ advice: `查询失败：${e.message || e}`, hits: [] });
    } finally {
      setBusy(false);
    }
  };
  const blocked = res?.advice_kind === 'block';
  return (
    <div className="v8-rules-check">
      <div className="v8-rules-checkrow">
        <input
          value={q}
          onChange={e => setQ(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') run(); }}
          placeholder="我准备测：删掉某层 top-1% 单元看正确率是否下降……"
        />
        <button type="button" disabled={busy || !q.trim()} onClick={run}>{busy ? '查…' : '查重'}</button>
      </div>
      {res && (
        <>
          <p className={'v8-rules-advice' + (blocked ? ' bad' : res.advice_kind === 'new' ? ' new' : '')}>{res.advice}</p>
          {(res.hits || []).map(h => {
            const st = STATUS_STYLE[h.status] || STATUS_STYLE.open;
            return (
              <div key={h.id} className="v8-rules-hit">
                <span className="v8-rules-id">{h.id}</span>
                <span className="v8-rules-hit-claim">{h.claim}</span>
                <Pill style={st}>{st.text}</Pill>
                <span className="v8-rules-sim">相似 {Math.round((h.similarity || 0) * 100)}%</span>
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}

export default function RuleLedgerPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState('');
  const [domain, setDomain] = useState('');

  useEffect(() => {
    let dead = false;
    fetch(`${API_BASE}/api/rules/overview`, { cache: 'no-store' })
      .then(r => r.json())
      .then(d => { if (!dead) setData(d); })
      .catch(e => { if (!dead) setErr(e.message || String(e)); });
    return () => { dead = true; };
  }, []);

  const rules = useMemo(() => {
    const all = (data?.rules || []).slice();
    const order = { refuted: 0, confirmed: 1, inconclusive: 2, open: 3, superseded: 4 };
    all.sort((a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9) || String(a.id).localeCompare(String(b.id)));
    return domain ? all.filter(r => r.domain === domain) : all;
  }, [data, domain]);

  if (err) return <Empty>规则账本读取失败：{err}</Empty>;
  if (!data) return <Empty>读取规则账本…</Empty>;
  if (!data.available) {
    return <p className="v8-note">规则账本还是空的：在本机运行 <code>python tools/rules_ingest.py</code> 写入初始条目。</p>;
  }

  const c = data.counts || {};
  const bs = c.by_status || {};
  const scan = data.scan || {};

  return (
    <div className="v8-rules">
      <p className="v8-nprog-say">
        已登记 <b>{c.total}</b> 条规则，其中 <b>{bs.refuted || 0}</b> 条已被实测否证、<b>{bs.confirmed || 0}</b> 条已证实、
        <b>{bs.inconclusive || 0}</b> 条未决。<b>{c.blocked || 0}</b> 条标注「不要重测」——口径不变时再做一次属于重复劳动，
        账本里已有结论与证据。累计独立测试 <b>{c.recorded_retests}</b> 次。
      </p>

      <div className="v8-nprog-tiles">
        <div><b>{c.total}</b><span>条规则已登记</span><small>每条 = 一条可判定命题 + 结论 + 证据链</small></div>
        <div><b>{bs.refuted || 0}</b><span>条已被否证</span><small>最有价值的一类：直接告诉后来者这条路不通</small></div>
        <div><b>{c.blocked || 0}</b><span>条不要重测</span><small>口径不变时再测不会得到新信息</small></div>
        <div><b>{c.need_n || 0}</b><span>条需加样本才能定论</span><small>方向已知但题量不足，重测必须上到指定规模</small></div>
      </div>

      <div>
        <h4 className="v8-h4">编码破解现状 <small>把规则按"机制假设是否还活着"分三堆：目标不是攒结论，是收缩候选空间</small></h4>
        <div className="v8-rules-split">
          <div style={{ borderLeftColor: '#1d9e75' }}>
            <b>仍然成立的机制事实</b>
            <span>{bs.confirmed || 0} 条</span>
            <ul>
              {(data.rules || []).filter(r => r.status === 'confirmed').map(r => (
                <li key={r.id}><code>{r.id}</code>{r.topic || r.claim.slice(0, 26)}</li>
              ))}
            </ul>
          </div>
          <div style={{ borderLeftColor: '#d85a30' }}>
            <b>已被排除的机制</b>
            <span>{bs.refuted || 0} 条</span>
            <ul>
              {(data.rules || []).filter(r => r.status === 'refuted').map(r => (
                <li key={r.id}><code>{r.id}</code>{r.topic || r.claim.slice(0, 26)}</li>
              ))}
            </ul>
          </div>
          <div style={{ borderLeftColor: '#ba7517' }}>
            <b>仍在候选里的机制</b>
            <span>{(bs.inconclusive || 0) + (bs.open || 0)} 条</span>
            <ul>
              {(data.rules || []).filter(r => r.status === 'inconclusive' || r.status === 'open').map(r => (
                <li key={r.id}><code>{r.id}</code>{r.topic || r.claim.slice(0, 26)}</li>
              ))}
            </ul>
            <small>这几条是下一轮该做的实验；其余 24 条口径不变时不必再测。</small>
          </div>
        </div>
      </div>

      <div>
        <h4 className="v8-h4">先查重：这件事是不是已经测过 <small>贴一句你准备做的测试，账本会给出最相近的已有结论</small></h4>
        <RuleCheck />
      </div>

      {scan.memory_files ? (
        <p className="v8-note">
          日志扫描：{scan.memory_files} 篇工作日志（{scan.memory_lines} 行）与 {scan.design_files} 篇设计文档中，
          共 <b>{data.scan_total_records || 0}</b> 条测试记录命中主题词（下面只列最集中的 5 类）。
          重复最多的前几类：
          {(data.scan_topics || []).slice(0, 5).map(t => ` ${t.name} ${t.n_records} 条`).join(' ·')}。
          这些主题已收敛成上面的 {c.total} 条规则，不再重开。
        </p>
      ) : null}

      <div>
        <h4 className="v8-h4">
          规则清单 <small>否证排在最前 · 点任意一条展开数字与证据</small>
        </h4>
        <div className="v8-tools">
          <span className="v8-seg">
            {DOMAINS.map(([id, label]) => (
              <button key={id || 'all'} type="button" className={domain === id ? 'on' : ''} onClick={() => setDomain(id)}>
                {label}{id ? ` ${(c.by_domain || {})[id] || 0}` : ` ${c.total}`}
              </button>
            ))}
          </span>
        </div>
        <div className="v8-rules-list">
          {rules.map(r => <RuleCard key={r.id} rule={r} labels={data.labels} />)}
        </div>
      </div>

      <p className="v8-note">{data.policy} 模型范围：{data.model_scope}。更新时间：{data.updated_at || '—'}。</p>
    </div>
  );
}
