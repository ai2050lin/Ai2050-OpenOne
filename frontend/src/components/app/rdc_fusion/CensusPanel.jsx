/* 研发 · 普查作业（census_job.v1）：作业进度、逐分片状态、复算不一致队列；管理员可开关作业与手动合并。
   数据源：/api/census/jobs、/api/census/jobs/{id}、/api/census/review；写操作带管理令牌。 */
import { useState } from 'react';

import { CensusJobBar, CensusRuns } from '../../open/OpenHome.jsx';
import { SHARD_STATES, fmtAgo, fmtTokens } from '../../open/openUtil.js';
import { postJson, useAdminToken, useApi } from './api.js';
import { AdminTokenField, Card, Empty, RefreshButton } from './ui.jsx';

const STATE_META = Object.fromEntries(SHARD_STATES.map(s => [s.id, s]));

function ShardGrid({ job }) {
  const groups = job.layer_groups || [];
  return (
    <table className="v6-census-grid">
      <thead>
        <tr><th>切片 \ 层组</th>{groups.map((g, i) => <th key={i}>G{i}<small>L{g[0]}–{g[g.length - 1]}</small></th>)}</tr>
      </thead>
      <tbody>
        {job.slices.map(sl => (
          <tr key={sl.slice}>
            <th>S{sl.slice}<small>{fmtTokens(sl.expected_n)}</small></th>
            {groups.map((_, g) => {
              const sh = job.shards.find(x => x.slice === sl.slice && x.group === g);
              const m = STATE_META[sh?.state] || STATE_META.queued;
              return (
                <td key={g} style={{ background: m.color }} title={`${sh?.shard} · ${m.label} · 已接受 ${sh?.accepted}/${sh?.replicas}${sh?.rejected ? ` · 拒收 ${sh.rejected}` : ''}`}>
                  {sh?.replicas > 1 ? '×2' : ''}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function CensusPanel() {
  const jobs = useApi('/api/census/jobs', { online: true });
  const review = useApi('/api/census/review', { online: true });
  const runs = useApi('/api/neurons/index');
  const [pick, setPick] = useState('');
  const [token, setToken] = useAdminToken();
  const [msg, setMsg] = useState('');
  const list = jobs.data?.jobs || [];
  const cur = list.find(j => j.job_id === pick) || list[0];
  const detail = useApi(cur ? `/api/census/jobs/${encodeURIComponent(cur.job_id)}` : null, { online: true });

  const act = async (path, body, label) => {
    setMsg('');
    try {
      await postJson(path, body, { adminToken: token });
      setMsg(`${label}：完成`);
      jobs.reload(); detail.reload(); review.reload();
    } catch (e) {
      setMsg(`${label}：${e.message}`);
    }
  };

  if (jobs.error) return <Empty>普查端点不可用：{jobs.error}</Empty>;
  if (!jobs.data) return <Empty>读取普查作业…</Empty>;
  return (
    <div className="v6-census">
      <div className="v6-toolbar">
        {list.length > 0 && (
          <label>作业
            <select value={cur?.job_id || ''} onChange={e => setPick(e.target.value)}>
              {list.map(j => <option key={j.job_id} value={j.job_id}>{j.job_id} · {j.status}</option>)}
            </select>
          </label>
        )}
        <RefreshButton apis={[jobs, detail, review, runs]} />
        <AdminTokenField token={token} onChange={setToken} />
        {msg && <span className="v6-note">{msg}</span>}
      </div>
      {(runs.data?.census || []).length > 0 && (
        <Card title="已合并的普查结果" sub="单元卡（空间 · 原生神经元）读取这些结果；分布式作业合并后也会出现在这里">
          <CensusRuns runs={runs.data.census} title={false} />
        </Card>
      )}
      {!list.length && (
        <Empty>还没有分布式普查作业。管理员在中心服务器上用 deploy/admin_tool.py census-create 建作业（见 deploy/README_DEPLOY.md）。</Empty>
      )}
      {cur && (
        <Card title={`${cur.job_id} · ${cur.model?.name || cur.model?.model_key}`}
              sub={`语料 ${cur.corpus_fp} · 每片 ${fmtTokens(cur.tokens_per_slice)} token · 快照 ${String(cur.model?.snapshot_sha256 || '').slice(0, 16)}`}
              extra={token ? (
                <>
                  {cur.status === 'open'
                    ? <button type="button" className="fw-tbtn" onClick={() => act(`/api/census/jobs/${cur.job_id}/status`, { status: 'paused' }, '暂停')}>暂停派发</button>
                    : <button type="button" className="fw-tbtn" onClick={() => act(`/api/census/jobs/${cur.job_id}/status`, { status: 'open' }, '开放')}>开放派发</button>}
                  <button type="button" className="fw-tbtn primary" onClick={() => act(`/api/census/jobs/${cur.job_id}/merge`, {}, '合并')}>合并已完成切片</button>
                </>
              ) : null}>
          <CensusJobBar job={cur} />
          {detail.data && <ShardGrid job={detail.data} />}
          {cur.merged && <p className="v6-note">最近合并：{cur.merged.slices.length} 个切片 · {fmtTokens(cur.merged.n_tokens)} token · {fmtAgo(cur.merged.merged_at)}；单元卡见 空间 · 原生神经元。中心每 5 分钟自动合并新完成的切片。</p>}
          <p className="v6-note">格子颜色 = 分片状态；×2 = 抽中复算（需另一个账号的节点再跑一次）。校验与复算通过只表示产物可信、可以合并，不是证据晋级。</p>
        </Card>
      )}
      <Card title="复算不一致" sub="两份独立结果超出容差时都标为 disputed，暂停合并，由管理员接受其中一份">
        {review.error && <Empty>{review.error}</Empty>}
        {review.data && !review.data.disputed.length && <Empty>没有待处理的不一致。</Empty>}
        {review.data?.disputed.map(r => (
          <div key={r.result_id} className="v6-census-dispute">
            <span className="mono">{r.job_id} / {r.shard}</span>
            <span className="mono">{r.result_id} · {r.sha.slice(0, 12)}</span>
            <span>节点 {r.node_id} · {fmtAgo(r.created_at)}</span>
            {token && <button type="button" className="fw-tbtn" onClick={() => act(`/api/census/results/${r.result_id}/review`, { action: 'accept' }, '接受')}>接受这一份</button>}
          </div>
        ))}
      </Card>
    </div>
  );
}
