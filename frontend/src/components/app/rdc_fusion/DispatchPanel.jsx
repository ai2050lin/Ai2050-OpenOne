/* 研发 · ③ 分发与节点：生成式模板的分片覆盖 + 节点命令 + 平台执行监控（PlatformOps）
   + Research OS Work Unit 执行链（DistributedPuzzlePipeline，原在路线页）。 */
import { useState } from 'react';
import { useApi } from './api.js';
import { Card, Empty, StatusPill } from './ui.jsx';
import PlatformOps from './PlatformOps.jsx';
import DistributedPuzzlePipeline from './DistributedPuzzlePipeline.jsx';
import { NodeCommand } from './FreezePanel.jsx';

function ShardCoverage({ tm }) {
  const report = useApi(tm ? `/api/grammar/${encodeURIComponent(tm)}/report` : null, { interval: 20000 });
  if (!report.data) return <Empty>{report.error ? `报告不可用（${report.error}）` : '读取…'}</Empty>;
  const r = report.data;
  const shards = Array.from({ length: r.shards }, (_, i) => `${i}/${r.shards}`);
  return (
    <div>
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
  const templates = useApi('/api/templates', { interval: 30000 });
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
        <Card title="Research OS 执行链" sub="合同 → Work Unit → Worker → 产物审计">
          <DistributedPuzzlePipeline onShowResults={() => onGo('audit')} />
        </Card>
      </div>
      <aside className="v6-dispatch-side">
        <PlatformOps onOpenDist={() => onGo('audit')} />
      </aside>
    </div>
  );
}
