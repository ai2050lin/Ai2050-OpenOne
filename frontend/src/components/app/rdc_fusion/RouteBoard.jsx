/* 路线 · 证据门链与预测记分板（scoreboard.v1 快照）：每个 v3 模板走到了哪一道证据门、卡在哪里，以及留出预测分数。
   自动门由分布式协议判定；选择性 / 必要性 / 自然生成需要干预或生成实验，显示「未运行」。
   记分板按「模板 × 模型桶 × 因子 × 划分」汇总为最佳层一行；逐层明细按模板展开，避免几千行同时渲染。 */
import { useMemo, useState } from 'react';

import { fmtNum, useSnapshot } from './api.js';
import { Card, Empty, GateChain, StatusPill } from './ui.jsx';

const SPLIT_LABEL = { lockbox: '未见词表', lockbox_frame: '未见句式', confirm: '留出复现' };

function Verdict({ r }) {
  if (r.passed == null) return <StatusPill status="na" label="只作记录" />;
  return <StatusPill status={r.passed ? 'ok' : 'fail'} label={r.passed ? '达标' : '未达'} />;
}

export default function RouteBoard({ onGo }) {
  const sb = useSnapshot('scoreboard', { interval: 120000 });
  const [open, setOpen] = useState('');
  const summary = useMemo(() => {
    const best = new Map();
    for (const r of sb.data?.scores || []) {
      if (r.score == null) continue;
      const k = `${r.tm_id}|${r.bucket}|${r.factor}|${r.split}`;
      const cur = best.get(k);
      if (!cur || r.score > cur.score) best.set(k, r);
    }
    const rows = [...best.values()];
    rows.sort((a, b) => a.tm_id.localeCompare(b.tm_id) || a.bucket.localeCompare(b.bucket) || a.factor.localeCompare(b.factor) || a.split.localeCompare(b.split));
    return rows;
  }, [sb.data]);
  if (sb.error && !sb.data) return <Empty>记分板不可用：{sb.error}</Empty>;
  if (!sb.data) return <Empty>{sb.loading ? '记分板快照计算中（中心刚启动时需要几分钟）…' : '读取记分板…'}</Empty>;
  const { templates, scores, gate_chain: chain } = sb.data;
  const detail = open ? scores.filter(r => r.tm_id === open) : [];
  return (
    <div className="v6-route">
      <Card title="证据门链" sub="10 道门：前 7 道由协议自动判定，后 3 道需要预注册干预或自然生成实验">
        <ol className="v6-route-legend">
          {chain.map((g, i) => <li key={g.id} title={g.desc}><b>{i + 1}</b> {g.label}{g.auto ? '' : '（需实验）'}</li>)}
        </ol>
      </Card>
      <Card title="模板族" sub="每行一个生成式模板；点击进入 空间 · 实验响应">
        {!templates.length && <Empty>还没有生成式（v3）模板的结果。去 研发 · 模板工作室 设计并冻结模板，或在 参与 页贡献算力。</Empty>}
        {templates.map(t => (
          <div key={t.tm_id} className="v6-route-row">
            <button type="button" className="v6-linkbtn" onClick={() => onGo('spatial', null, null, { tm: t.tm_id })}>
              <b>{t.family_id || t.tm_id}</b> <span>{t.name}</span>
            </button>
            <GateChain gates={t.gates} compact />
            <span className="v6-note">{t.blocking ? `卡在「${t.blocking.label}」：${t.blocking.note || ''}` : '全部自动门通过'}</span>
          </div>
        ))}
      </Card>
      <Card title="预测记分板" sub={`每个模板 × 模型桶 × 因子 × 划分取最佳层（共 ${summary.length} 行，逐层明细 ${scores.length} 行可按模板展开）。${sb.data.note}`}>
        {!summary.length && <Empty>暂无留出预测分数。</Empty>}
        {summary.length > 0 && (
          <table className="v6-table">
            <thead><tr><th>模板</th><th>模型桶</th><th>因子</th><th>划分</th><th>最佳层</th><th>平均 cos</th><th>阈值</th><th>判定</th><th /></tr></thead>
            <tbody>{summary.map((r, i) => (
              <tr key={i}>
                <td>{r.family_id || r.tm_id}</td><td className="mono">{r.bucket}</td><td>{r.factor}</td>
                <td>{SPLIT_LABEL[r.split] || r.split}</td><td className="mono">{r.layer}</td>
                <td>{fmtNum(r.score, 3)}</td><td>{r.threshold == null ? '—' : fmtNum(r.threshold, 2)}</td><td><Verdict r={r} /></td>
                <td>{(i === 0 || summary[i - 1].tm_id !== r.tm_id) && (
                  <button type="button" className="v6-linkbtn" onClick={() => setOpen(o => (o === r.tm_id ? '' : r.tm_id))}>{open === r.tm_id ? '收起明细' : '逐层明细'}</button>
                )}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </Card>
      {open && (
        <Card title={`逐层明细 · ${open}`} sub={`${detail.length} 行`} extra={<button type="button" className="fw-tbtn" onClick={() => setOpen('')}>收起</button>}>
          <table className="v6-table">
            <thead><tr><th>模型桶</th><th>因子</th><th>层</th><th>划分</th><th>平均 cos</th><th>阈值</th><th>判定</th></tr></thead>
            <tbody>{detail.slice(0, 600).map((r, i) => (
              <tr key={i}>
                <td className="mono">{r.bucket}</td><td>{r.factor}</td><td className="mono">{r.layer}</td><td>{SPLIT_LABEL[r.split] || r.split}</td>
                <td>{fmtNum(r.score, 3)}</td><td>{r.threshold == null ? '—' : fmtNum(r.threshold, 2)}</td><td><Verdict r={r} /></td>
              </tr>
            ))}</tbody>
          </table>
          {detail.length > 600 && <p className="v6-note">只显示前 600 行；完整数据见 /api/scoreboard。</p>}
        </Card>
      )}
    </div>
  );
}
