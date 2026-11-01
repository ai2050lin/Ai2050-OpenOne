/* 研究工作台 · 进展 · 规则账本
   目的不是"再展示一次结论"，而是防止重复劳动：每条规则给结论状态、关键数字、证据链，
   以及明确的重测判定（不要重测 / 重测需加样本 / 可以重测）。
   查重框可以贴一句"我准备测什么"，先看这件事有没有已经做过。
   数据来自服务器 /api/rules/overview（本机 data/rules/ledger.json）。 */
import { useEffect, useMemo, useRef, useState } from 'react';

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

/* 折叠块：默认只给「数量 + 前三条示例」，点开才铺全部。避免一屏密密麻麻。 */
function Fold({ tone, title, count, hint, items, preview }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="v8-rules-fold" style={{ borderLeftColor: tone }}>
      <button type="button" className="v8-rules-foldhead" onClick={() => setOpen(o => !o)}>
        <b>{title}</b>
        <span className="v8-rules-foldcount" style={{ color: tone }}>{count}</span>
        <span className="v8-rules-foldcaret">{open ? '收起' : '展开'}</span>
      </button>
      {open ? (
        <ul className="v8-rules-foldlist">
          {items.map(r => (
            <li key={r.id}><code>{r.id}</code><span>{r.topic || r.claim.slice(0, 30)}</span></li>
          ))}
        </ul>
      ) : (
        <p className="v8-rules-foldprev">{preview}</p>
      )}
      {hint && <small className="v8-rules-foldhint">{hint}</small>}
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
  const listRef = useRef(null);

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

  /* 类型概览：每类一条，带「证实 / 否证 / 待定」堆叠条。顶部先给这个，一屏看完全局。
     类型从数据里聚合（不写死），名字优先用服务端标签，未知类型退回原 id。 */
  const domainRows = useMemo(() => {
    const all = data?.rules || [];
    const labels = data?.labels?.domain || {};
    const rank = id => {
      const i = DOMAINS.findIndex(([k]) => k === id);
      return i < 0 ? 99 : i;
    };
    const seen = new Map();
    all.forEach(r => {
      const id = r.domain || 'mlp';
      if (!seen.has(id)) seen.set(id, { id, label: labels[id] || id, total: 0, confirmed: 0, refuted: 0 });
      const row = seen.get(id);
      row.total += 1;
      if (r.status === 'confirmed') row.confirmed += 1;
      else if (r.status === 'refuted') row.refuted += 1;
    });
    return [...seen.values()]
      .map(row => ({ ...row, pending: row.total - row.confirmed - row.refuted }))
      .sort((a, b) => rank(a.id) - rank(b.id) || b.total - a.total);
  }, [data]);

  const alive = useMemo(() => (data?.rules || []).filter(r => r.status === 'confirmed'), [data]);
  const dead = useMemo(() => (data?.rules || []).filter(r => r.status === 'refuted'), [data]);
  const cand = useMemo(() => (data?.rules || []).filter(r => r.status === 'inconclusive' || r.status === 'open'), [data]);

  const pick = id => {
    setDomain(id);
    if (listRef.current) listRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

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
        已登记 <b>{c.total}</b> 条规则，其中 <b>{bs.refuted || 0}</b> 条已被实测否证。
        否证最值钱——它直接告诉后来者这条路不通。
      </p>

      <div className="v8-rules-ov">
        <div className="v8-rules-ovlead">
          <span>按类型</span>
          <small>点一行筛选下面的清单</small>
        </div>
        {domainRows.map(row => (
          <button
            key={row.id}
            type="button"
            className={'v8-rules-ovrow' + (domain === row.id ? ' on' : '')}
            onClick={() => pick(domain === row.id ? '' : row.id)}
          >
            <span className="v8-rules-ovname">{row.label}</span>
            <span className="v8-rules-ovbar">
              <i style={{ flexGrow: row.confirmed, background: '#1d9e75' }} />
              <i style={{ flexGrow: row.refuted, background: '#d85a30' }} />
              <i style={{ flexGrow: row.pending, background: '#d3d1c7' }} />
            </span>
            <span className="v8-rules-ovnum">{row.total}</span>
            <span className="v8-rules-ovsub">
              <b style={{ color: '#0f6e56' }}>{row.confirmed}</b>证实
              <b style={{ color: '#993c1d' }}>{row.refuted}</b>否证
              <b style={{ color: '#8a6d3b' }}>{row.pending}</b>待定
            </span>
          </button>
        ))}
      </div>

      <div className="v8-nprog-tiles">
        <div><b>{c.total}</b><span>条规则已登记</span><small>每条 = 一条可判定命题 + 结论 + 证据链</small></div>
        <div><b>{bs.refuted || 0}</b><span>条已被否证</span><small>最有价值的一类：直接告诉后来者这条路不通</small></div>
        <div><b>{c.blocked || 0}</b><span>条不要重测</span><small>口径不变时再测不会得到新信息</small></div>
        <div><b>{(bs.open || 0) + (c.need_n || 0)}</b><span>条仍待测</span><small>已预注册但还没跑，或需加样本才能定论</small></div>
      </div>

      <div>
        <h4 className="v8-h4">编码破解现状 <small>按「机制假设是否还活着」分三堆：目标不是攒结论，是收缩候选空间</small></h4>
        <div className="v8-rules-folds">
          <Fold
            tone="#1d9e75" title="仍然成立的机制事实" count={alive.length} items={alive}
            preview={alive.slice(0, 3).map(r => r.topic || r.id).join(' · ') + (alive.length > 3 ? ` 等 ${alive.length} 条` : '')}
          />
          <Fold
            tone="#d85a30" title="已被排除的机制" count={dead.length} items={dead}
            preview={dead.slice(0, 3).map(r => r.topic || r.id).join(' · ') + (dead.length > 3 ? ` 等 ${dead.length} 条` : '')}
            hint="这一堆越长，后面的弯路越短。口径不变时不要再测其中任何一条。"
          />
          <Fold
            tone="#ba7517" title="仍在候选里的机制" count={cand.length} items={cand}
            preview={cand.slice(0, 3).map(r => r.topic || r.id).join(' · ') + (cand.length > 3 ? ` 等 ${cand.length} 条` : '')}
            hint="这几条是下一轮该做的实验；其余条目口径不变时不必再测。"
          />
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

      <div ref={listRef} className="v8-rules-anchor">
        <h4 className="v8-h4">
          规则清单 <small>否证排在最前 · 点任意一条展开数字与证据</small>
        </h4>
        <p className="v8-rules-filter">
          {domain
            ? <>只看 <b>{(domainRows.find(r => r.id === domain) || {}).label || domain}</b> · {rules.length} 条</>
            : <>全部 {rules.length} 条</>}
          {domain ? <button type="button" onClick={() => setDomain('')}>显示全部</button> : null}
        </p>
        <div className="v8-rules-list">
          {rules.map(r => <RuleCard key={r.id} rule={r} labels={data.labels} />)}
        </div>
      </div>

      <p className="v8-note">{data.policy} 模型范围：{data.model_scope}。更新时间：{data.updated_at || '—'}。</p>
    </div>
  );
}
