/* 研发 · 模板流水线 ③ 分发：生成式模板的分片覆盖 + 节点命令 + 节点与租约（/api/distributed/summary、/api/tasks，只显示实时数据）。 */
import { useState } from 'react';
import { fmtAgoTs, useApi } from './api.js';
import { Card, Empty, RefreshButton, StatusPill } from './ui.jsx';
import { NodeCommand } from './FreezePanel.jsx';

function NodesAndLeases() {
  const summary = useApi('/api/distributed/summary', { online: true });
  const tasks = useApi('/api/tasks', { online: true });
  const s = summary.data;
  const t = tasks.data;
  return (
    <>
      <Card title="节点" sub={s ? `在线 ${s.nodes_online} / 全部 ${s.nodes_total}（10 分钟内有心跳算在线）` : ''}
        extra={<RefreshButton apis={[summary, tasks]} />}>
        {!s && <Empty>{summary.error ? `节点信息不可用（${summary.error}）` : '读取…'}</Empty>}
        {s && !s.nodes.length && <Empty>当前没有在线节点。</Empty>}
        {s?.nodes.map(n => (
          <div key={n.node_id} className="v6-node-row">
            <b>{n.name}</b><span className="mono">{n.node_id}</span><span>{n.gpu || '—'}</span><span>{fmtAgoTs(n.last_seen)}</span>
          </div>
        ))}
      </Card>
      <Card title="租约" sub={t ? `单个租约 ${t.lease_hours} 小时；过期未完成的分片会重新派发` : ''}>
        {!t && <Empty>{tasks.error ? `调度信息不可用（${tasks.error}）` : '读取…'}</Empty>}
        {t && (
          <ul className="v6-health">
            {Object.entries(t.stats).map(([k, v]) => <li key={k}><span>{{ done: '完成', failed: '失败', expired: '过期', claimed: '执行中' }[k] || k}</span><b>{v}</b></li>)}
          </ul>
        )}
        {t?.active?.length > 0 && t.active.map(a => (
          <div key={a.task_id} className="v6-node-row"><span className="mono">{a.task_id}</span><span>{a.tm_id}</span><span>{a.node_name || '?'}</span></div>
        ))}
      </Card>
    </>
  );
}

function ShardCoverage({ tm }) {
  const report = useApi(tm ? `/api/grammar/${encodeURIComponent(tm)}/report` : null, { online: true });
  if (!report.data) return <Empty>{report.error ? `报告不可用（${report.error}）` : '读取…'}</Empty>;
  const r = report.data;
  const shards = Array.from({ length: r.shards }, (_, i) => `${i}/${r.shards}`);
  return (
    <div>
      <div className="v6-toolbar"><RefreshButton apis={[report]} label="刷新分片" /></div>
      {!r.buckets.length && <Empty>还没有 real 结果；任务：{Object.entries(r.tasks).map(([k, v]) => `${k} ${v}`).join(' · ') || '无'}</Empty>}
      {r.buckets.map(b => (
        <div key={b.bucket} className="v6-shards">
          <span className="mono">{b.bucket}</span>
          <div>
            {shards.map(sh => {
              const done = b.shards_present.includes(sh);
              const reps = b.replicates.filter(x => x.shard === sh).length;
              return <span key={sh} className={'v6-shard ' + (done ? 'done' : 'todo')} title={`${sh}${done ? ' 已回传' : ' 未回传'}${reps ? ` · 独立重跑 ${reps}` : ''}`}>{done ? '✓' : '○'} {sh}{reps ? ` ×${reps + 1}` : ''}</span>;
            })}
          </div>
          <StatusPill status={b.complete ? 'done' : 'partial'} label={b.complete ? '分片齐全' : `${b.shards_present.length}/${b.shards_expected}`} />
        </div>
      ))}
    </div>
  );
}

export default function DispatchPanel({ onGo }) {
  const templates = useApi('/api/templates', { online: true });
  const v3 = (templates.data?.templates || []).filter(t => t.protocol === 'v3');
  const [pick, setPick] = useState('');
  const tm = pick || v3[0]?.tm_id || '';
  return (
    <div className="v6-dispatch">
      <div className="v6-dispatch-main">
        <Card title="生成式模板 · 分片覆盖" sub="中心按「最少 done+claimed 次数」分配分片；同一分片被不同节点重跑用于一致性检查"
          extra={v3.length > 0 && (
            <select value={tm} onChange={event => setPick(event.target.value)}>
              {v3.map(t => <option key={t.tm_id} value={t.tm_id}>{t.tm_id} v{t.version}{t.mode === 'pilot' ? '（试跑）' : ''}</option>)}
            </select>
          )}>
          {!templates.data && <Empty>{templates.error ? `未接入（${templates.error}）` : '读取…'}</Empty>}
          {templates.data && !v3.length && <Empty action={<button type="button" className="fw-tbtn" onClick={() => onGo('studio')}>去模板工作室 →</button>}>还没有冻结的生成式模板。</Empty>}
          {tm && <ShardCoverage tm={tm} />}
          <NodeCommand />
        </Card>
      </div>
      <aside className="v6-dispatch-side">
        <NodesAndLeases />
      </aside>
    </div>
  );
}
