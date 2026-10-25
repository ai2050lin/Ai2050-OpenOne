/* 总览 v6：闭环漏斗 + 题库概览 + 谱图进展 + 覆盖矩阵 + 数据健康 + 待我决策 + 贡献者 + 参与测试。
   数据：/api/pipeline/overview · /api/coverage · /api/catalog · /api/atlas · /api/atlas/hypotheses（中心服务）+ 研究账本。
   纪律：各阶段计数来自独立数据表，不连成转化率；服务未接入显示空态，不填演示数字。 */
import { fmtAgoTs, fmtNum, useApi, useSnapshot } from './api.js';
import { CensusJobBar } from '../../open/OpenHome.jsx';
import { Card, Empty, SourceTag, StatusPill } from './ui.jsx';
import { MISSION } from './distributedData.js';
import { STATUS_META, STATUS_ORDER } from './catalogUtil.js';
import { HYP_STATUS } from './atlasUtil.js';
import { ParticipateSteps } from './CatalogPanel.jsx';

/* 题库概览：每个类别一条堆叠条（已完成 / 测试中 / 待测试 / 试跑中 / 未冻结），点击进入 研发 · 题库 */
function CatalogOverview({ data, error, loading, onGo }) {
  if (!data) return <Empty>{loading ? '读取题库…' : `题库未接入${error ? `（${error}）` : ''}`}</Empty>;
  const cats = data.categories;
  const max = Math.max(1, ...cats.map(c => c.total));
  return (
    <div className="v6-catov">
      <div className="v6-catov-legend">
        {STATUS_ORDER.map(k => <span key={k}><i className={'k ' + k} />{STATUS_META[k][1]}</span>)}
      </div>
      {cats.map(c => (
        <button key={c.category} type="button" className="v6-catov-row" onClick={() => onGo('process', 'catalog')}
          title={STATUS_ORDER.map(k => `${STATUS_META[k][1]} ${c[k]}`).join(' · ')}>
          <span className="v6-catov-name">{c.category}</span>
          <span className="v6-catov-bar" style={{ width: `${(c.total / max) * 100}%` }}>
            {STATUS_ORDER.map(k => (c[k] ? <i key={k} className={k} style={{ flex: c[k] }} /> : null))}
          </span>
          <span className="v6-catov-n"><b>{c.done}</b>/{c.total} 已完成{c.running ? ` · ${c.running} 测试中` : ''}</span>
        </button>
      ))}
      <p className="v6-note">「已完成」= 至少一个模型桶的分片齐全、可以合并；不是证据等级。</p>
    </div>
  );
}

/* 谱图进展：每个模型的共享编码轴、预注册概念假设；最近 AI 假设（未检验） */
function AtlasProgress({ atlas, hyps, onGo }) {
  if (!atlas.data) return <Empty>{atlas.loading ? '读取语言谱图…' : `语言谱图未接入${atlas.error ? `（${atlas.error}）` : ''}`}</Empty>;
  const graphs = atlas.data.graphs || [];
  if (!graphs.length) return <Empty>谱图需要同一模型下至少两个模式族分片齐全。</Empty>;
  const recent = (hyps.data?.items || []).slice(0, 2);
  return (
    <div className="v6-atov">
      {graphs.map(g => {
        const sup = g.axes.filter(a => a.supported);
        const concepts = [...new Set(sup.flatMap(a => (a.concepts.length ? a.concepts : ['新发现'])))];
        return (
          <button key={g.model} type="button" className="v6-atov-model" onClick={() => onGo('spatial', null, undefined, { view: 'atlas' })}>
            <span className="v6-atov-hd"><b className="mono">{g.model}</b><small>{g.families.length} 族 · {g.edges.length} 边</small></span>
            <span className="v6-atov-row">
              <span>共享轴 <b>{sup.length}</b>/{g.axes.length} 通过迁移{concepts.length ? `（${concepts.join('、')}）` : ''}</span>
            </span>
            <span className="v6-atov-row">
              {g.concept_hypotheses.length
                ? g.concept_hypotheses.map(h => <StatusPill key={h.concept} status={(HYP_STATUS[h.status] || ['na'])[0]} label={`${h.concept} · ${(HYP_STATUS[h.status] || ['', h.status])[1]}`} />)
                : <small className="muted">没有可检验的预注册概念</small>}
            </span>
          </button>
        );
      })}
      {atlas.data.cross_model && <p className="v6-note">跨模型：可检验边 {atlas.data.cross_model.edges_testable}，复现率 {fmtNum(atlas.data.cross_model.replication_rate, 2)}</p>}
      {recent.length > 0 && (
        <div className="v6-atov-ai">
          <h5 className="v6-h5">最近 AI 假设</h5>
          {recent.flatMap(r => (r.hypotheses || []).slice(0, 1).map((h, i) => (
            <div key={`${r.id}-${i}`} className="v6-atov-hyp"><StatusPill status="pending" label="未检验" /><span>{h.statement}</span><small>{fmtAgoTs(r.created_at)}</small></div>
          )))}
        </div>
      )}
    </div>
  );
}

function Contributors({ data }) {
  if (!data) return <Empty>题库未接入。</Empty>;
  const list = data.contributors || [];
  if (!list.length) return <Empty>还没有节点注册。</Empty>;
  return (
    <ul className="v6-contrib">
      {list.slice(0, 8).map(c => (
        <li key={c.node_id}>
          <StatusPill status={c.online ? 'ok' : 'na'} label={c.online ? '在线' : '离线'} />
          <span><b>{c.name || c.node_id}</b><small>{[c.owner && `账号 ${c.owner}`, c.gpu, c.model, c.dtype].filter(Boolean).join(' · ') || '—'}</small></span>
          <span className="v6-contrib-n"><b>{c.results}</b> 份</span>
        </li>
      ))}
    </ul>
  );
}

const CELL_LABEL = { done: '完成', partial: '部分', fail: '未通过', empty: '空格', na: '不适用' };

function Funnel({ data, error, loading }) {
  if (!data) {
    return <Empty>{loading ? '总览快照计算中（中心刚启动时需要几分钟）…' : `闭环漏斗暂不可用${error ? `（${error}）` : ''}。`}</Empty>;
  }
  return (
    <ol className="v6-funnel" aria-label="研究闭环各阶段">
      {data.stages.map((stage, index) => (
        <li key={stage.id}>
          <span className="v6-funnel-n">{String(index + 1).padStart(2, '0')}</span>
          <small>{stage.label}</small>
          <b>{stage.value}</b>
          <span className="v6-funnel-d">{stage.detail}</span>
        </li>
      ))}
    </ol>
  );
}

function CoverageMatrix({ data, error, loading, onGo }) {
  if (!data) return <Empty>{loading ? '读取覆盖矩阵…' : `覆盖矩阵未接入${error ? `（${error}）` : ''}`}</Empty>;
  if (!data.rows.length) return <Empty action={<button type="button" className="fw-tbtn" onClick={() => onGo('process', 'studio')}>去模板工作室 →</button>}>还没有注册任何模板。</Empty>;
  return (
    <div className="v6-cov-wrap">
      <table className="v6-cov">
        <thead>
          <tr><th>模板</th>{data.columns.map(col => <th key={col.id}>{col.label}</th>)}</tr>
        </thead>
        <tbody>
          {data.rows.map(row => (
            <tr key={row.tm_id}>
              <th scope="row">
                <span className="mono">{row.tm_id}</span>
                <small>{row.name}{row.mode === 'pilot' ? ' · 试跑' : ''}</small>
                <i className={'v6-proto ' + row.protocol}>{row.protocol === 'v3' ? '生成式 v3' : 'legacy'}</i>
              </th>
              {data.columns.map(col => {
                const cell = row.cells[col.id] || { status: 'na' };
                return (
                  <td key={col.id} className={'v6-cov-cell ' + cell.status} title={cell.note}>
                    <button type="button" onClick={() => {
                      if (cell.status === 'empty' || cell.status === 'fail') onGo('process', col.id === 'audit' ? 'audit' : 'studio');
                      else onGo('spatial', null, null, { tm: row.tm_id });
                    }}>
                      <StatusPill status={cell.status} label={CELL_LABEL[cell.status]} />
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="v6-note">{data.note} 点击空格/未通过 → 研发；点击已有数据 → 空间·实验响应。</p>
    </div>
  );
}

export default function HomeOverview({ onGo }) {
  const overview = useSnapshot('pipeline_overview', { interval: 60000 });
  const coverage = useSnapshot('coverage', { interval: 60000 });
  const catalog = useApi('/api/catalog', { interval: 60000 });
  const atlas = useSnapshot('atlas', { interval: 120000 });
  const census = useApi('/api/census/jobs', { interval: 60000 });
  const hyps = useApi('/api/atlas/hypotheses?limit=5', { interval: 120000 });
  const live = overview.data ? 'live' : overview.loading ? 'loading' : 'none';
  return (
    <div className="fw-view on v6-home">
      <header className="v6-home-hero">
        <div>
          <h2><b className="fw-brand">{MISSION.brand}</b>{MISSION.title}</h2>
          <p>闭环：生成式模板 → 行为门 → 分布式采集 → 充分统计量合并 → 候选结构 → 留出预测 → 审计 → 图谱</p>
        </div>
        <div className="v6-home-actions">
          <SourceTag state={live} label="中心服务" />
          <button type="button" className="fw-tbtn primary" onClick={() => onGo('process', 'studio')}>设计模板 →</button>
        </div>
      </header>

      {overview.data?.sandbox_synthetic && (
        <div className="v6-synthetic v6-span2">
          <b>沙盒中心 · 接受合成结果</b>
          <span>本中心以 AI2050_DIST_ALLOW_SYNTHETIC=1 启动，计数中包含 runner 合成模式的结果（模型桶名带「合成·」前缀）。这些数字只用于验证流程，不代表真实模型。</span>
        </div>
      )}

      <Card title="闭环漏斗" sub="各阶段计数来自不同数据表，彼此独立，不构成转化率" className="v6-span2">
        <Funnel {...overview} />
      </Card>

      <div className="v6-home-pair">
        <Card title="题库概览" sub="按类别：每个题目 = 一个模式族；点击进入 研发 · 题库" extra={<SourceTag state={catalog.data ? 'live' : catalog.loading ? 'loading' : 'none'} />}>
          <CatalogOverview {...catalog} onGo={onGo} />
        </Card>
        <Card title="谱图进展" sub="跨模式族共享编码轴与预注册概念；点击进入 空间 · 语言谱图" extra={<SourceTag state={atlas.data ? 'live' : atlas.loading ? 'loading' : 'none'} />}>
          <AtlasProgress atlas={atlas} hyps={hyps} onGo={onGo} />
        </Card>
      </div>

      <div className="v6-home-grid">
        <Card title="覆盖矩阵" sub="模板 × 闭环阶段；每格由结果库与审计表现算" extra={<SourceTag state={coverage.data ? 'live' : coverage.loading ? 'loading' : 'none'} />}>
          <CoverageMatrix {...coverage} onGo={onGo} />
        </Card>

        <div className="v6-home-side">
          <Card title="待我决策" sub="需要人工处理的事项">
            {!overview.data && <Empty>{overview.loading ? '总览快照计算中（中心刚启动时需要几分钟）…' : `中心服务未接入${overview.error ? `（${overview.error}）` : ''}。`}</Empty>}
            {overview.data && !overview.data.decisions.length && <Empty>当前没有待处理事项。</Empty>}
            {overview.data?.decisions.map(item => (
              <button key={item.id} type="button" className="v6-decision" onClick={() => onGo(item.go, item.tab)}>
                <span>{item.label}</span><i aria-hidden="true">→</i>
              </button>
            ))}
          </Card>

          <Card title="数据健康">
            {!overview.data && <Empty>{overview.loading ? '总览快照计算中（中心刚启动时需要几分钟）…' : `中心服务未接入${overview.error ? `（${overview.error}）` : ''}。`}</Empty>}
            {overview.data && (
              <ul className="v6-health">
                {overview.data.health.map(item => (
                  <li key={item.id} className={item.level}>
                    <span>{item.label}</span>
                    <b>{item.value}</b>
                    <StatusPill status={item.level === 'bad' ? 'fail' : item.level === 'warn' ? 'warn' : item.level === 'info' ? 'na' : 'ok'}
                      label={item.level === 'bad' ? '需处理' : item.level === 'warn' ? '注意' : item.level === 'info' ? '信息' : '正常'} />
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="贡献者" sub="注册节点与回传结果数" extra={<button type="button" className="fw-home-mini-link" onClick={() => onGo('process', 'catalog')}>去题库 →</button>}>
            <Contributors data={catalog.data} />
          </Card>

          <Card title="参与测试" sub="任何用户：注册节点 → 领题 → 本地运行回传">
            <ParticipateSteps compact />
          </Card>

          <Card title="神经元普查" sub="分片状态：完成 / 等复算 / 运行中 / 待领取" extra={<button type="button" className="fw-home-mini-link" onClick={() => onGo('census')}>普查作业 →</button>}>
            {census.error && <Empty>{census.error}</Empty>}
            {census.data && !census.data.jobs.length && <Empty>还没有开放的普查作业。</Empty>}
            {census.data?.jobs.map(j => <CensusJobBar key={j.job_id} job={j} />)}
          </Card>
        </div>
      </div>
    </div>
  );
}
