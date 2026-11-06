import { useCallback, useEffect } from 'react';
import { ADMIN_LIVE, API_BASE, useApi } from '../app/rdc_fusion/api.js';
import { useResearchContext } from '../app/researchContext.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import DatasetAtlasPlan from './DatasetAtlasPlan.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import ResearchInterventionReport from './ResearchInterventionReport.jsx';
import ResearchParticipation from './ResearchParticipation.jsx';
import ResearchBundleUpload from './ResearchBundleUpload.jsx';
import { orderedStages } from './researchFragmentUtil.js';

const STAGE = {
  freeze: '冻结材料', collection: '原场采集', client_acceptance: '客户端回查', validation: '留出验证',
  capability: '能力保持', registry: '数据与图谱登记', registration_validated: '成果登记', pilot: '行为试跑',
  discovery: '候选发现', intervention: '预注册组干预', native_audit: '生成全场审计', ai_review: '外部 AI 盲评',
  interface: '统一采集接口', calibration: '行为校准', prediction: '未见组合预测', native_mlp_plan: 'MLP 主线规划',
  registration_discovery: '登记观察', dataset_atlas_plan: '数据与图谱方案',
  semantic_mechanism: '语义机制验证', semantic_causal_closure: '语义因果闭环',
};
const STAGE_STATE = {
  done: '已完成', completed: '已完成', passed: '通过', not_supported: '△ 未支持', not_tested: '未测试',
  running: '执行中', proposed: '方案待实施', partial: '部分已完成', ready: '已冻结，待执行',
  inconclusive: '尚未证实', behavior_failed: '行为门失败', noninferior: '非劣效通过', inferior: '超过允许损失',
  scope_limited: '结论有适用范围', failed: '执行失败',
};
const REMAINING = new Set(['running', 'ready', 'proposed', 'partial', 'not_tested', 'pending', 'awaiting_ai_reviews', 'awaiting_human_reviews']);
const TASK_STATE = { claimed: '运行中', queued: '排队', failed: '失败', expired: '租约过期', done: '已完成' };
const workpackagePath = '/api/research-library/workpackages';
// Plain-language summaries are pinned to the reviewed report revision.
const RESEARCH_BRIEFS = {
  'ce7acdafa21579c42e1c125bf06c5e4fbff8149c68d73ee5561b0d1dc3ec42c1': {
    label: 'Qwen3-4B · 神经元跨深度复用',
    question: '模型把推理链从三步延长到六步时，会复用哪些神经元？',
    finding: '在 Qwen3-4B 最后一层，9,728 个 MLP 单元中有 2,001 个在短链确认和长链留出上复现了贡献方向。',
  },
  '2eac0a9c54680ee04d35747d495fea8f92955e84adf9a2ea0e62b3cb6178d24c': {
    label: '原子序数来源选择',
    question: '当两份原子序数记录互相冲突，模型会采信哪一份？',
    finding: 'Qwen3-4B 在有参考资料的两种场景中通过来源选择测试。无参考问答的校准未通过，候选 MLP 组的效应也未达到预注册标准。',
    sample: true,
  },
};
const dateTime = seconds => new Date(seconds * 1000).toLocaleString();
const stageState = stage => stage.state ?? stage.status;

function planStatus(body) {
  const stages = orderedStages(body.stages ?? {});
  const pending = stages.find(([, stage]) => stageState(stage) === 'running') ||
    stages.find(([, stage]) => REMAINING.has(stageState(stage)));
  return {
    status: pending ? STAGE_STATE[stageState(pending[1])] ?? '待处理'
      : body.stages?.registration_validated?.report_sha256 ? '已有归档' : '已登记',
    next: pending ? STAGE[pending[0]] ?? pending[0] : '',
  };
}

function PlanStages({ stages }) {
  return <div className="v8-workpackage-stages">{orderedStages(stages ?? {}).map(([key, stage]) => {
    const state = stageState(stage);
    const status = state === 'not_supported' && stage.baseline_gate === false ? '测量门未通过' : STAGE_STATE[state] ?? state;
    return <span key={key}><b>{STAGE[key] ?? key}</b>{status ?? '已登记'}</span>;
  })}</div>;
}

function SourceNotice({ source, loading }) {
  return source.error ? <div role="status"><Empty>{source.error}</Empty><ExtraPackPrompt pack={source.missingPack}/></div>
    : !source.data ? <p className="v8-note">{loading}</p> : null;
}

function PlanRow({ entry, focused, children = [], onGo }) {
  const body = entry.body;
  const { status, next } = planStatus(body);
  const context = { run: body.run_id, snapshot: body.model_snapshot };
  const datasetPlan = body.stages?.dataset_atlas_plan;
  const report = body.stages?.registration_validated?.report_sha256;
  const brief = RESEARCH_BRIEFS[report];
  return <ResearchDisclosure initiallyOpen={focused} title={<span className="v8-rnd-row">
    <strong>{brief?.label || body.title || body.run_id}</strong><span className="v8-chip">{status}</span>
    <small>{next ? `下一步 · ${next}` : '下一步待登记'}</small>
  </span>}>
    <p className="v8-note">{body.run_id}</p>
    {(brief?.question || body.question) && <h3>{brief?.question || body.question}</h3>}
    {(brief?.finding || body.finding) && <p>{brief?.finding || body.finding}</p>}
    <div className="v6-toolbar">
      <button className="fw-tbtn" type="button" onClick={() => onGo('catalog', null, null, { ...context, mrun: body.run_id, tm: '' })}>查看所用材料</button>
      <button className="fw-tbtn" type="button" onClick={() => onGo('execution', null, null, context)}>查看实验</button>
    </div>
    <ResearchDisclosure title="阶段与验收"><PlanStages stages={body.stages}/></ResearchDisclosure>
    {brief?.sample && <ResearchDisclosure title="查看一条真实样本"><ResearchSample sha={report}/></ResearchDisclosure>}
    {report && <ResearchDisclosure title="查看结果与反例">
      <ResearchInterventionReport sha={report} planSha={body.stages.native_mlp_plan?.plan_sha256}
        capabilityInputsSha={body.stages.client_acceptance?.capability_inputs_sha256} rawFallback/>
    </ResearchDisclosure>}
    {datasetPlan?.plan_sha256 && <DatasetAtlasPlan sha={datasetPlan.plan_sha256} implementation={stageState(datasetPlan) === 'partial'}/>}
    <ResearchDisclosure title="预注册与原始记录"><pre className="v8-artifact-json">{JSON.stringify(body.stages, null, 2)}</pre></ResearchDisclosure>
    {children.length > 0 && <ResearchDisclosure title={`前置校准与试跑（${children.length}）`}>
      {children.map(child => <PlanRow key={child.sha} entry={child} onGo={onGo}/>)}
    </ResearchDisclosure>}
  </ResearchDisclosure>;
}

function ResearchSample({ sha }) {
  const { data, error } = useApi(`/api/research-library/artifacts/${sha}`);
  if (error) return <Empty>{error}</Empty>;
  if (!data) return <p className="v8-note">读取已归档的样本…</p>;
  const sample = data.examples?.find(item => /^[a-f0-9]{64}$/.test(item.trace_id));
  if (!sample) return <Empty>这份报告没有登记可回查的样本。</Empty>;
  const query = new URLSearchParams({ lens: 'spatial', view: 'live', trace: sample.trace_id,
    run: data.run_id, snapshot: data.model_snapshot });
  if (window.location.protocol === 'file:') query.set('wb', '1');
  return <p>{sample.entity} · {sample.scope} · {sample.length} token{' '}
    <a className="v6-linkbtn" href={`?${query}`}>查看原文与模型响应</a></p>;
}

function currentWork(census, queue) {
  const jobs = (census?.jobs ?? []).filter(job => job.status === 'open' && job.n_shards > (job.counts?.done ?? 0))
    .map(job => ({ id: job.job_id, name: [job.model?.name, '神经元普查'].filter(Boolean).join(' · '),
      done: job.counts?.done ?? 0, total: job.n_shards, active: job.counts?.running ?? 0,
      queued: (job.counts?.queued ?? 0) + (job.counts?.replica_pending ?? 0), unit: '分片' }));
  const listed = queue?.queued ?? [];
  const allQueuedListed = (queue?.stats?.queued ?? 0) === listed.length;
  for (const group of queue?.by_template ?? []) {
    const listedForTemplate = listed.filter(task => task.tm_id === group.tm_id);
    const queued = group.queued ?? (allQueuedListed ? listedForTemplate.length : null);
    if (!group.active && !queued && !listedForTemplate.length) continue;
    const total = group.total ?? (queued === null ? null : group.done + group.active + group.failed + group.expired + queued);
    jobs.push({ id: group.tm_id, name: group.template_name || listedForTemplate[0]?.template_name || group.tm_id,
      done: group.done, total, active: group.active, queued, unit: '任务' });
  }
  return jobs;
}

function ResearchPlans({ onGo }) {
  const packages = useApi(workpackagePath);
  const census = useApi('/api/census/jobs', { online: true });
  const queue = useApi('/api/tasks', { online: true });
  const { reload: reloadCensus } = census;
  const { reload: reloadQueue } = queue;
  const refreshJobs = useCallback(() => {
    if (ADMIN_LIVE) { reloadCensus(); reloadQueue(); }
  }, [reloadCensus, reloadQueue]);
  useEffect(() => {
    if (!ADMIN_LIVE) return undefined;
    const timer = window.setInterval(() => {
      if (!document.hidden) refreshJobs();
    }, 12000);
    return () => window.clearInterval(timer);
  }, [refreshJobs]);

  const { run } = useResearchContext();
  const all = packages.data?.workpackages ?? [];
  const ids = new Set(all.map(({ body }) => body.run_id));
  const focused = all.find(({ body }) => body.run_id === run);
  const roots = all.filter(entry => entry === focused || !ids.has(entry.body.parent_run_id));
  const ongoing = all.filter(({ body }) => body.kind !== 'synthetic' &&
    Object.values(body.stages ?? {}).some(stage => stageState(stage) === 'running'));
  const archived = roots.filter(entry => {
    const registration = entry.body.stages?.registration_validated;
    return !ongoing.includes(entry) && registration &&
      (registration.report_sha256 || ['done', 'completed', 'passed'].includes(stageState(registration)));
  });
  const plans = roots.filter(entry => !ongoing.includes(entry) && !archived.includes(entry));
  const linkedArchive = archived.includes(focused);
  const archiveLabel = linkedArchive && (RESEARCH_BRIEFS[focused.body.stages.registration_validated.report_sha256]?.label || focused.body.title || run);
  const archiveRows = linkedArchive ? [focused, ...archived.filter(entry => entry !== focused)] : archived;
  const jobs = currentWork(census.data, queue.data);
  const row = entry => <PlanRow key={`${run}:${entry.sha}`} entry={entry} focused={entry === focused} onGo={onGo}
    children={all.filter(child => child !== focused && !ongoing.includes(child) && child.body.parent_run_id === entry.body.run_id)}/>;

  return <section className="v8-card v8-rnd v8-rnd-current" aria-label="当前研发">
    <h2>当前研发</h2>
    <div className="v8-rnd-dashboard">
      <section className="v8-rnd-progress" aria-label="研究进度">
        <h3>研究进度</h3>
        {ongoing.map(({ body }) => <div className="v8-rnd-progress-title" key={body.run_id}>
          <strong>{body.title || body.run_id}</strong><span className="v8-chip">执行中</span>
        </div>)}
        {jobs.map(job => <div className="v8-rnd-work" key={`${job.unit}:${job.id}`}>
          <div className="v8-rnd-progress-title"><strong>{job.name}</strong>
            <span className="v8-chip">{job.active > 0 ? '运行中' : job.queued > 0 ? '待领取' : '开放'}</span></div>
          <div className="v8-rnd-task-count"><b>{job.done}<span> / {job.total ?? '—'}</span></b> {job.unit}已完成</div>
          {job.total > 0 && <progress aria-label={`${job.name}已完成${job.unit}`} value={job.done} max={job.total}/>}
          <small>{job.active} 运行中{job.queued != null && ` · ${job.queued} 待领取`}</small>
        </div>)}
        {!ongoing.length && !jobs.length && <p className="v8-note">{census.loading || queue.loading ? '正在读取作业…'
          : census.error || queue.error ? '任务进度暂不可用' : '暂无进行中的研究或开放作业'}</p>}
        {(census.local || queue.local) && <small>任务统计来自已下载的数据包</small>}
        {jobs.length > 0 && (census.error || queue.error) && <small>部分任务统计暂未更新</small>}
      </section>
      <ResearchParticipation onGo={onGo} publicWork={jobs.find(job => job.queued > 0)?.name}
        publicSnapshot={census.local || queue.local} publicLoaded={Boolean(census.data && queue.data)}
        onTaskChanged={refreshJobs}/>
    </div>
    {ongoing.length > 0 && <ResearchDisclosure title="进行中的研究详情"><div className="v8-rnd-list">{ongoing.map(row)}</div></ResearchDisclosure>}
    {archiveRows.length > 0 && <ResearchDisclosure key={run} title={archiveLabel ? `研究历史 · ${archiveLabel}` : `研究历史（${archiveRows.length}）`}>
      <div className="v8-rnd-list">{archiveRows.map(row)}</div>
    </ResearchDisclosure>}
    {plans.length > 0 && <ResearchDisclosure title={`研究计划（${plans.length}）`}><div className="v8-rnd-list">{plans.map(row)}</div></ResearchDisclosure>}
    {packages.error && <ResearchDisclosure title="研究历史"><SourceNotice source={packages} loading="读取研究…"/></ResearchDisclosure>}
    {packages.data && run && !focused && <p className="v8-note">当前资料中未找到链接指定的研究。</p>}
  </section>;
}

function ArchivedExperiment({ run, onGo }) {
  const packages = useApi(workpackagePath);
  const body = packages.data?.workpackages?.find(entry => entry.body.run_id === run)?.body;
  const report = body?.stages?.registration_validated?.report_sha256;
  return <section className="v8-rnd-linked" aria-label="链接指定的实验">
    <h3>{body?.title || run}</h3>
    {body?.title && <p className="v8-note">{run}</p>}
    <SourceNotice source={packages} loading="读取关联实验…"/>
    {packages.data && !body && <p className="v8-note">当前资料中没有此运行的归档记录。</p>}
    {body && <>
      {report ? <ResearchDisclosure title="研究报告与反例">
        <ResearchInterventionReport sha={report} planSha={body.stages.native_mlp_plan?.plan_sha256}
          capabilityInputsSha={body.stages.client_acceptance?.capability_inputs_sha256} rawFallback/>
      </ResearchDisclosure> : <p className="v8-note">此运行尚无已登记报告。</p>}
      <div className="v6-toolbar">
        <button type="button" className="fw-tbtn" onClick={() => onGo('plan')}>返回当前研发</button>
        <button type="button" className="fw-tbtn" onClick={() => onGo('route', null, null, { rtab: 'rules' })}>查看成果与证据</button>
      </div>
    </>}
  </section>;
}

function TaskRow({ task, name, onGo }) {
  const ref = task.research_ref ?? {};
  return <ResearchDisclosure title={<span className="v8-rnd-row">
    <strong>{name || task.template_name || task.tm_id || task.task_id}</strong>
    <span className="v8-chip">{TASK_STATE[task.status] ?? task.status}</span><small>{task.task_id}</small>
  </span>}>
    <dl className="v8-rnd-meta">
      <div><dt>模板</dt><dd>{task.tm_id || '未登记'}</dd></div>
      <div><dt>节点</dt><dd>{task.node_name || '未分配'}</dd></div>
      <div><dt>分片</dt><dd>{task.shard || (ref.shard_id ?? ref.shard_index ?? '未登记')}</dd></div>
      {task.work_unit_id && <div><dt>工作单元</dt><dd>{task.work_unit_id}</dd></div>}
      {task.model && <div><dt>模型</dt><dd>{task.model}</dd></div>}
      {task.done_at > 0 && <div><dt>结束时间</dt><dd>{dateTime(task.done_at)}</dd></div>}
      {task.result_sha && <div><dt>结果校验值</dt><dd>{task.result_sha}</dd></div>}
    </dl>
    {task.note && <p className="v8-note">{task.note}</p>}
    <div className="v6-toolbar">
      {task.tm_id && <button className="fw-tbtn" type="button"
        onClick={() => onGo('catalog', null, null, { tm: task.tm_id, mrun: '' })}>查看所用模板</button>}
      {task.result_sha && <a className="fw-tbtn" target="_blank" rel="noreferrer"
        href={`${API_BASE}/api/results/${encodeURIComponent(task.result_sha)}`}>查看原始结果</a>}
    </div>
    {Object.keys(ref).length > 0 && <ResearchDisclosure title="执行来源"><pre className="v8-artifact-json">{JSON.stringify(ref, null, 2)}</pre></ResearchDisclosure>}
  </ResearchDisclosure>;
}

function ResearchQueue({ onGo, tools }) {
  const tasks = useApi('/api/tasks', { online: true });
  const catalog = useApi('/api/catalog');
  const { run } = useResearchContext();
  const items = catalog.data?.items ?? [];
  const names = new Map(items.flatMap(item => [item.tm_id, item.pilot_tm_id].filter(Boolean).map(id => [id, item.name])));
  const audits = items.filter(item => item.tm_id && item.audit_pending > 0);
  const stats = tasks.data?.stats ?? {};
  const recent = tasks.data?.recent ?? [];
  const failed = tasks.data?.attention ?? recent.filter(task => task.status === 'failed');
  const failureGroups = tasks.data?.attention ? [] : (tasks.data?.by_template ?? []).flatMap(template => {
    const shown = failed.filter(task => task.tm_id === template.tm_id).length;
    const remaining = (template.failed ?? 0) + (template.expired ?? 0) - shown;
    return remaining > 0 ? [{ ...template, remaining }] : [];
  });
  const active = (tasks.data?.active ?? []).map(task => ({ ...task, status: 'claimed' }));
  const queued = (tasks.data?.queued ?? []).map(task => ({ ...task, status: 'queued' }));
  const completed = recent.filter(task => task.status === 'done');
  const row = task => <TaskRow key={task.task_id} task={task} name={names.get(task.tm_id)} onGo={onGo}/>;
  const issueRow = item => <ResearchDisclosure key={item.tm_id} title={<span className="v8-rnd-row">
    <strong>{item.name || names.get(item.tm_id) || item.tm_id}</strong>
    <span className="v8-chip">{item.remaining ? '异常任务' : '待审'}</span>
    <small>{[item.remaining && `${item.remaining} 条异常`, item.audit_pending && `${item.audit_pending} 份待审`].filter(Boolean).join(' · ')}</small>
  </span>}>
    {item.remaining > 0 && <p className="v8-note">当前数据只提供该模板的汇总。可更新队列来源查看任务明细。</p>}
    <div className="v6-toolbar">
      <button className="fw-tbtn" type="button" onClick={() => onGo('catalog', null, null, { tm: item.tm_id, mrun: '' })}>查看所用模板</button>
      {item.audit_pending > 0 && tools.some(tool => tool.id === 'pipeline') &&
        <button className="fw-tbtn" type="button" onClick={() => onGo('audit')}>打开审计</button>}
    </div>
  </ResearchDisclosure>;
  const byTemplate = new Map(failureGroups.map(item => [item.tm_id, item]));
  for (const audit of audits) byTemplate.set(audit.tm_id, { ...byTemplate.get(audit.tm_id), ...audit });
  const groups = [...byTemplate.values()];
  const highlighted = new Set([
    ...groups.filter(item => item.remaining).slice(0, 2),
    ...groups.filter(item => item.audit_pending).slice(0, 2),
  ]);
  const otherGroups = groups.filter(item => !highlighted.has(item));
  const pending = catalog.data ? `${audits.reduce((sum, item) => sum + item.audit_pending, 0)} 份待审`
    : catalog.error ? '待审暂不可用' : '待审读取中';
  const failures = tasks.data ? `${stats.failed ?? 0} 个失败` : '失败未接入';
  const current = [...active, ...queued];

  return <section className="v8-card v8-rnd" aria-label="实验与队列">
    <h2>实验与队列</h2>
    <dl className="v8-rnd-metrics">
      <div><dt>运行中</dt><dd>{tasks.data ? stats.claimed ?? 0 : '—'}</dd></div>
      <div><dt>排队</dt><dd>{tasks.data ? stats.queued ?? 0 : '—'}</dd></div>
      <div><dt>需处理</dt><dd>{failures} · {pending}{stats.expired > 0 ? ` · ${stats.expired} 个租约过期` : ''}</dd></div>
    </dl>
    <div className="v8-rnd-caption">
      {tasks.data?.server_time > 0 && <span>{tasks.local ? '数据包' : '队列'} · {dateTime(tasks.data.server_time)}</span>}
      <button className="v6-linkbtn" type="button" onClick={() => { tasks.reload(); catalog.reload(); }}>刷新</button>
    </div>
    <SourceNotice source={tasks} loading="读取调度队列…"/>
    {catalog.error && <p className="v8-note">{catalog.error}</p>}
    <ResearchBundleUpload/>
    {run && <ArchivedExperiment key={run} run={run} onGo={onGo}/>}
    <div className="v8-rnd-list" aria-label="当前任务">
      {failed.slice(0, 2).map(row)}
      {[...highlighted].map(issueRow)}
      {current.slice(0, 3).map(row)}
    </div>
    {tasks.data && !current.length && <p className="v8-note">当前无运行或排队任务。</p>}
    {tasks.data?.attention && (stats.failed ?? 0) + (stats.expired ?? 0) > failed.length && <p className="v8-note">
      队列显示最近 {failed.length} 条待处理任务明细。
    </p>}
    {(failed.length > 2 || otherGroups.length > 0 || current.length > 3) && <ResearchDisclosure title="更多任务与待审模板">
      <div className="v8-rnd-list">{failed.slice(2).map(row)}{otherGroups.map(issueRow)}{current.slice(3).map(row)}</div>
    </ResearchDisclosure>}
    <ResearchDisclosure title="历史任务">
      {completed.length ? <div className="v8-rnd-list">{completed.map(row)}</div> : <Empty>当前资料中没有已完成任务记录。</Empty>}
    </ResearchDisclosure>
    {tools.length > 0 && <ResearchDisclosure title="专业工具">
      <div className="v6-toolbar">{tools.map(tool => <button key={tool.id} type="button" className="fw-tbtn"
        onClick={() => onGo(tool.id)}>{tool.label}</button>)}</div>
    </ResearchDisclosure>}
  </section>;
}

export default function ResearchExecution({ plan = false, onGo, tools = [] }) {
  return plan ? <ResearchPlans onGo={onGo}/> : <ResearchQueue onGo={onGo} tools={tools}/>;
}
