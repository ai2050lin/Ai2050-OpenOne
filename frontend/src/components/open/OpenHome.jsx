/* 开放平台 · 首页：研究进展快照（/api/platform/overview）。
   纪律：各项计数来自不同数据表，分开呈现，不拼成转化率；没有数据时显示空态，不用演示数字。 */
import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { GRADES } from '../app/rdc_fusion/neuronUtil.js';
import { SHARD_STATES, fmtAgo, fmtInt, fmtTokens } from './openUtil.js';

function Stat({ value, label, sub, onClick }) {
  return (
    <button type="button" className="op-stat" onClick={onClick} disabled={!onClick}>
      <b>{value}</b>
      <span>{label}</span>
      {sub && <small>{sub}</small>}
    </button>
  );
}

export function CensusRuns({ runs, onGo, title = true }) {
  return (
    <div className="op-runs">
      {title && <h3>已合并的普查结果</h3>}
      <table className="op-table">
        <thead><tr><th>模型</th><th>运行</th><th>计入 token</th><th>层数</th><th /></tr></thead>
        <tbody>{runs.map(r => (
          <tr key={`${r.model_key}/${r.run}`}>
            <td className="op-mono">{r.model_key}</td><td className="op-mono">{r.run}</td><td>{fmtTokens(r.n_tokens)}</td><td>{r.n_layers ?? '—'}</td>
            <td>{onGo && <button type="button" className="op-btn" onClick={() => onGo('atlas')}>看单元卡 →</button>}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export function CensusJobBar({ job }) {
  const total = job.n_shards || 1;
  return (
    <div className="op-job">
      <div className="op-job-hd">
        <b>{job.model?.name || job.model?.model_key}</b>
        <span className="op-mono">{job.job_id}</span>
        <span>{fmtTokens(job.tokens_done)} / {fmtTokens(job.tokens_per_slice * job.n_slices)} token 已完成</span>
        {job.merged && <span className="op-tag">已合并 {fmtTokens(job.merged.n_tokens)} · {fmtAgo(job.merged.merged_at)}</span>}
      </div>
      <div className="op-bar" role="img" aria-label="分片状态分布">
        {SHARD_STATES.map(s => {
          const n = (job.counts || {})[s.id] || 0;
          return n ? <i key={s.id} style={{ width: `${(100 * n) / total}%`, background: s.color }} title={`${s.label} ${n}`} /> : null;
        })}
      </div>
      <div className="op-legend">
        {SHARD_STATES.map(s => ((job.counts || {})[s.id] ? (
          <span key={s.id}><i style={{ background: s.color }} />{s.label} {job.counts[s.id]}</span>
        ) : null))}
        <span className="op-muted">共 {total} 个分片（{job.n_slices} 个 token 切片 × {job.layer_groups?.length} 个层组）· 复算抽样 {Math.round(100 * job.replica_rate)}%</span>
      </div>
    </div>
  );
}

export default function OpenHome({ config, onGo }) {
  const ov = useApi('/api/platform/overview');
  const d = ov.data;
  if (ov.error) return <Empty>中心服务未接入：{ov.error}</Empty>;
  if (!d) return <Empty>读取研究进展…</Empty>;
  const grades = d.neurons?.explanation_grades || {};
  const census = d.neurons?.census_runs || [];
  const nTok = census.reduce((s, r) => s + (r.n_tokens || 0), 0);
  return (
    <div className="op-page">
      <section className="op-hero">
        <h1>{config?.title || 'AI2050 开源 LLM 逆向研究平台'}</h1>
        <p>{config?.tagline}</p>
        <div className="op-hero-actions">
          <button type="button" className="op-btn primary" onClick={() => onGo('join')}>参与测试 →</button>
          <button type="button" className="op-btn" onClick={() => onGo('learn')}>先了解方法</button>
          <button type="button" className="op-btn" onClick={() => onGo('atlas')}>浏览图谱</button>
        </div>
      </section>

      <section className="op-stats">
        <Stat value={fmtInt(d.families?.results_real)} label="模式族测试结果" sub={`${fmtInt(d.families?.templates_open)} 个开放模板 · ${(d.families?.models || []).length} 个模型`} onClick={() => onGo('atlas')} />
        <Stat value={d.atlas ? fmtInt(d.atlas.n_nodes) : '—'} label="谱图特征" sub={d.atlas ? `${fmtInt(d.atlas.n_edges)} 条跨族边 · ${(d.atlas.models || []).length} 个模型` : '快照计算中或尚无谱图'} onClick={() => onGo('atlas')} />
        <Stat value={nTok ? fmtTokens(nTok) : '—'} label="神经元普查 token" sub={`${census.length} 次普查`} onClick={() => onGo('atlas')} />
        <Stat value={fmtInt(Object.values(grades).reduce((a, b) => a + b, 0))} label="已检验解释的神经元"
              sub={GRADES.filter(g => grades[g]).map(g => `${g} ${grades[g]}`).join(' · ') || '尚无'} onClick={() => onGo('atlas')} />
        <Stat value={`${fmtInt(d.people?.nodes_online)} / ${fmtInt(d.people?.nodes_total)}`} label="在线节点 / 全部" sub={`${fmtInt(d.people?.contributors)} 位贡献者 · ${fmtInt(d.people?.users)} 个账号`} onClick={() => onGo('join')} />
      </section>

      <section className="op-section">
        <h2>神经元普查进度</h2>
        <p className="op-lead">每个分片 = 一段自然语料（token 切片）× 一组层。节点在本机 GPU 上统计每个 MLP 神经元的激活分布与最强激活片段，中心逐项校验、抽样复算，再合并成单元卡。</p>
        {(d.census_jobs || []).map(j => <CensusJobBar key={j.job_id} job={j} />)}
        {census.length > 0 && <CensusRuns runs={census} onGo={onGo} />}
        {(d.census_jobs || []).length === 0 && census.length === 0 && <Empty>还没有普查结果，也没有开放的普查作业。</Empty>}
      </section>

      <section className="op-section op-grid2">
        <div>
          <h2>这个平台在做什么</h2>
          <ul className="op-list">
            <li><b>外部语言模式</b>：用可复现的句式模板（模式族）描述词义、角色、语法、语气等语言变量。</li>
            <li><b>内部特征</b>：在模型的 HiddenState 与 MLP 神经元里找能读出、能复现、能干预的特征。</li>
            <li><b>两者之间的边</b>：每条边都带证据等级、运行 ID 与反例，可以回到原始数据核对。</li>
          </ul>
        </div>
        <div>
          <h2>证据边界</h2>
          <p className="op-lead">{d.note}</p>
          <p className="op-lead">快照更新于 {fmtAgo(d.computed_at)}；公开内容：{(config?.public_scope || []).join('、')}。</p>
        </div>
      </section>
    </div>
  );
}
