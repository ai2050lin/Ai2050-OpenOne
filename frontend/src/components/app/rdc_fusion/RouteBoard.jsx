/* 路线 · 证据门链与预测记分板（scoreboard.v1 快照）：每个 v3 模板走到了哪一道证据门、卡在哪里，以及留出预测分数。
   自动门由分布式协议判定；选择性 / 必要性 / 自然生成需要干预或生成实验，显示「未运行」。 */
import { fmtNum, useSnapshot } from './api.js';
import { Card, Empty, GateChain, StatusPill } from './ui.jsx';

export default function RouteBoard({ onGo }) {
  const sb = useSnapshot('scoreboard', { interval: 120000 });
  if (sb.error && !sb.data) return <Empty>记分板不可用：{sb.error}</Empty>;
  if (!sb.data) return <Empty>{sb.loading ? '记分板快照计算中（中心刚启动时需要几分钟）…' : '读取记分板…'}</Empty>;
  const { templates, scores, gate_chain: chain } = sb.data;
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
      <Card title="预测记分板" sub={sb.data.note}>
        {!scores.length && <Empty>暂无留出预测分数。</Empty>}
        {scores.length > 0 && (
          <table className="v6-table">
            <thead><tr><th>模板</th><th>模型桶</th><th>因子</th><th>层</th><th>划分</th><th>平均 cos</th><th>阈值</th><th>判定</th></tr></thead>
            <tbody>{scores.map((r, i) => (
              <tr key={i}>
                <td>{r.family_id || r.tm_id}</td><td className="mono">{r.bucket}</td><td>{r.factor}</td><td>{r.layer}</td><td>{r.split}</td>
                <td>{fmtNum(r.score, 3)}</td><td>{r.threshold == null ? '—' : fmtNum(r.threshold, 2)}</td>
                <td>{r.passed == null ? <StatusPill status="na" label="只作记录" /> : <StatusPill status={r.passed ? 'ok' : 'fail'} label={r.passed ? '达标' : '未达'} />}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
