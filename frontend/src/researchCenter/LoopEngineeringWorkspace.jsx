import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowUp,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  Clock3,
  Code2,
  Cpu,
  Database,
  FileCode2,
  FlaskConical,
  Goal,
  ListChecks,
  Loader2,
  Paperclip,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings2,
  ShieldCheck,
  SkipForward,
  Square,
  Terminal,
  Trash2,
  X,
} from 'lucide-react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const GATES = [
  { id: 'gap', label: '证据缺口', phases: ['analyze'] },
  { id: 'contract', label: '冻结契约', phases: ['plan'] },
  { id: 'execute', label: '执行实验', phases: ['generate', 'execute'] },
  { id: 'review', label: '独立复核', phases: ['summarize'] },
  { id: 'writeback', label: '证据回写', phases: [] },
];

const INSPECTOR_TABS = [
  { id: 'plan', label: '计划', icon: ListChecks },
  { id: 'contract', label: '契约', icon: ShieldCheck },
  { id: 'artifacts', label: '产物', icon: FileCode2 },
  { id: 'tests', label: '测试', icon: FlaskConical },
  { id: 'evidence', label: '证据', icon: Database },
  { id: 'models', label: '模型', icon: Bot },
];

const EMPTY_MASTER = {
  name: '主研发模型', model_type: 'master', api_type: 'openai', api_base: 'https://api.openai.com/v1',
  api_key: '', model_id: 'gpt-5', analysis_prompt: '', planning_prompt: '', code_gen_prompt: '', summary_prompt: '',
};

const EMPTY_ANALYST = {
  name: '独立分析模型', model_type: 'analyst', api_type: 'openai', api_base: 'https://api.openai.com/v1',
  api_key: '', model_id: 'gpt-5-mini', analysis_prompt: '', planning_prompt: '', code_gen_prompt: '', summary_prompt: '',
};

const STATUS_LABEL = {
  idle: '未运行',
  running: '运行中',
  paused: '已暂停',
  stopped: '已停止',
  waiting_step: '等待确认',
  waiting_approval: '等待确认',
  completed: '已完成',
  blocked: '已阻塞',
  review_required: '待复核',
  plan_completed: '计划完成',
};

const DEFAULT_PROJECT_AGENT_FORM = {
  project_goal: '',
  max_loops: 3,
  execution_mode: 'auto',
  stop_on_accepted: true,
  stop_on_rejected: true,
  max_consecutive_inconclusive: 3,
};

const EVENT_META = {
  objective: { label: '研究目标', tone: 'blue' },
  project_agent_status: { label: '项目 Agent', tone: 'blue' },
  project_agent_progress: { label: '任务推进', tone: 'blue' },
  phase_change: { label: '研究门', tone: 'neutral' },
  round_change: { label: '新一轮', tone: 'neutral' },
  status_change: { label: '运行状态', tone: 'neutral' },
  mode_change: { label: '执行模式', tone: 'neutral' },
  analysis: { label: '独立分析', tone: 'violet' },
  planning: { label: '计划', tone: 'blue' },
  generation: { label: '代码生成', tone: 'violet' },
  code_generated: { label: '代码产物', tone: 'violet' },
  execution: { label: '执行', tone: 'amber' },
  execution_result: { label: '执行结果', tone: 'amber' },
  review: { label: '复核', tone: 'violet' },
  summary: { label: '综合', tone: 'blue' },
  finding: { label: '研究发现', tone: 'green' },
  database_writeback: { label: '数据回写', tone: 'green' },
  error: { label: '错误', tone: 'red' },
};

async function readJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function normalizeConfig(config) {
  return {
    master_model: { ...EMPTY_MASTER, ...(config?.master_model || {}) },
    analyst_models: Array.isArray(config?.analyst_models)
      ? config.analyst_models.map((item) => ({ ...EMPTY_ANALYST, ...item, model_type: 'analyst' }))
      : [],
  };
}

function currentGateId(phase, status) {
  if (!phase) return status === 'stopped' ? 'writeback' : 'gap';
  return GATES.find((gate) => gate.phases.includes(phase))?.id || 'gap';
}

function formatTime(value, includeDate = false) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat('zh-CN', includeDate
    ? { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }
    : { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(date);
}

function eventText(event) {
  if (event.content) return event.content;
  if (event.message) return event.message;
  if (event.objective) return event.objective;
  if (event.finding?.summary) return event.finding.summary;
  if (event.finding?.decision) return `裁决：${event.finding.decision}`;
  if (event.run_id) return `工件已保存至 ${event.run_id}`;
  if (event.phase) return `进入 ${GATES.find((gate) => gate.phases.includes(event.phase))?.label || event.phase}`;
  if (event.status) return STATUS_LABEL[event.status] || event.status;
  if (event.mode) return event.mode === 'auto' ? '切换为自动运行' : '切换为手动执行';
  if (event.round) return `开始 Loop ${event.round}`;
  if (event.type === 'code_generated') return '主模型已生成待验证代码';
  if (event.type === 'execution_result') return '执行完成，原始结果已进入测试检查器';
  return '运行状态已更新';
}

function decisionLabel(value) {
  if (value === 'accepted') return '支持';
  if (value === 'rejected') return '反证';
  if (value === 'inconclusive') return '未决';
  if (value === 'completed') return '完成';
  if (value === 'running') return '执行中';
  return value || '待执行';
}

function GateStrip({ phase, status }) {
  const active = currentGateId(phase, status);
  const activeIndex = GATES.findIndex((gate) => gate.id === active);
  return (
    <div className="loop-workspace__gates" aria-label="研究证据门">
      {GATES.map((gate, index) => (
        <article key={gate.id} className={index < activeIndex ? 'is-done' : gate.id === active ? 'is-active' : ''}>
          <span>{index < activeIndex ? <Check size={12} /> : index + 1}</span>
          <strong>{gate.label}</strong>
        </article>
      ))}
    </div>
  );
}

function ModelFields({ model, onChange, master = false }) {
  const set = (key, value) => onChange({ ...model, [key]: value });
  return (
    <div className="loop-workspace__model-fields">
      <div className="loop-workspace__form-grid">
        <label>显示名称<input value={model.name || ''} onChange={(event) => set('name', event.target.value)} /></label>
        <label>模型 ID<input value={model.model_id || ''} onChange={(event) => set('model_id', event.target.value)} /></label>
        <label>API 类型<select value={model.api_type || 'openai'} onChange={(event) => set('api_type', event.target.value)}><option value="openai">OpenAI</option><option value="nownextai">NowNextAI</option><option value="zhipu">智谱兼容</option><option value="deepseek">DeepSeek</option><option value="dashscope">DashScope</option><option value="openai-compatible">OpenAI 兼容</option></select></label>
        <label>API 地址<input value={model.api_base || ''} onChange={(event) => set('api_base', event.target.value)} /></label>
      </div>
      <label>API Key<input type="password" autoComplete="off" value={model.api_key || ''} onChange={(event) => set('api_key', event.target.value)} placeholder="只保存在本地配置文件" /></label>
      {master ? (
        <div className="loop-workspace__prompts">
          <label>分析提示词<textarea value={model.analysis_prompt || ''} onChange={(event) => set('analysis_prompt', event.target.value)} /></label>
          <label>规划提示词<textarea value={model.planning_prompt || ''} onChange={(event) => set('planning_prompt', event.target.value)} /></label>
          <label>编程提示词<textarea value={model.code_gen_prompt || ''} onChange={(event) => set('code_gen_prompt', event.target.value)} /></label>
          <label>裁决提示词<textarea value={model.summary_prompt || ''} onChange={(event) => set('summary_prompt', event.target.value)} /></label>
        </div>
      ) : <label>独立分析提示词<textarea value={model.analysis_prompt || ''} onChange={(event) => set('analysis_prompt', event.target.value)} /></label>}
    </div>
  );
}

function ThreadRail({ agent, plan, runs, status, selectedId, onSelect }) {
  const [query, setQuery] = useState('');
  const tasks = Array.isArray(plan?.tasks) ? plan.tasks : [];
  const normalizedQuery = query.trim().toLowerCase();
  const visibleRuns = runs.filter((run) => {
    if (!normalizedQuery) return true;
    return `${run.objective || ''} ${run.run_id || ''}`.toLowerCase().includes(normalizedQuery);
  });
  const projectTitle = plan?.project_goal || agent.project_goal || '当前项目研发';

  return (
    <aside className="agent-thread-rail" aria-label="研究任务线程">
      <div className="agent-thread-rail__heading">
        <div><strong>研究任务</strong><span>PROJECT THREADS</span></div>
        <button type="button" title="创建新研究任务" onClick={() => onSelect('project')}><Plus size={16} /></button>
      </div>
      <label className="agent-thread-search"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索任务或 Run" /></label>

      <div className="agent-thread-section">
        <span className="agent-thread-section__label">当前项目</span>
        <button type="button" className={`agent-thread-card ${selectedId === 'project' ? 'is-active' : ''}`} onClick={() => onSelect('project')}>
          <span className={`agent-status-dot is-${status.status || 'idle'}`} />
          <div><strong>{projectTitle}</strong><small>{STATUS_LABEL[agent.status] || STATUS_LABEL[status.status] || '未启动'} · {tasks.length} 个任务</small></div>
          <ChevronRight size={14} />
        </button>
        {tasks.length ? (
          <div className="agent-thread-queue">
            {tasks.slice(0, 6).map((task, index) => (
              <div key={task.id}><span>{task.status === 'completed' ? <Check size={10} /> : index + 1}</span><p>{task.title}</p></div>
            ))}
          </div>
        ) : null}
      </div>

      <div className="agent-thread-section agent-thread-section--runs">
        <span className="agent-thread-section__label">最近运行</span>
        {visibleRuns.length ? visibleRuns.map((run) => (
          <button type="button" key={run.run_id} className={`agent-thread-card ${selectedId === `run:${run.run_id}` ? 'is-active' : ''}`} onClick={() => onSelect(`run:${run.run_id}`)}>
            <span className={`agent-run-icon is-${run.decision || 'pending'}`}><FlaskConical size={13} /></span>
            <div><strong>{run.objective || '未命名研究运行'}</strong><small>{formatTime(run.created_at || run.started_at, true)} · {decisionLabel(run.decision)}</small></div>
          </button>
        )) : <p className="agent-thread-empty">没有匹配的运行记录</p>}
      </div>

      <div className="agent-thread-rail__footer"><ShieldCheck size={13} /><span>所有结论等待确定性验证</span></div>
    </aside>
  );
}

function TimelineEvent({ event }) {
  const meta = EVENT_META[event.type] || { label: '运行记录', tone: 'neutral' };
  const text = eventText(event);
  const isBusy = /中\.{0,3}$/.test(text) || event.type === 'execution';
  return (
    <article className={`agent-timeline-event is-${meta.tone}`}>
      <div className="agent-timeline-event__marker">{isBusy ? <Loader2 size={14} className="is-spinning" /> : <Circle size={9} fill="currentColor" />}</div>
      <div className="agent-timeline-event__body">
        <header><strong>{meta.label}</strong><time>{formatTime(event.timestamp || event.created_at)}</time></header>
        <p>{text}</p>
        {event.run_id ? <span className="agent-inline-link"><Database size={12} />{event.run_id}</span> : null}
      </div>
    </article>
  );
}

function TaskThread({ agent, plan, status, events, selectedRun }) {
  if (selectedRun) {
    return (
      <div className="agent-task-thread">
        <header className="agent-thread-header">
          <div><span>历史运行</span><h2>{selectedRun.objective || selectedRun.run_id}</h2><p>{selectedRun.run_id}</p></div>
          <span className={`agent-status-pill is-${selectedRun.decision || 'pending'}`}>{decisionLabel(selectedRun.decision)}</span>
        </header>
        <div className="agent-thread-history">
          <article className="agent-message is-user"><div className="agent-message__avatar"><Goal size={15} /></div><div><header><strong>研究目标</strong><time>{formatTime(selectedRun.created_at || selectedRun.started_at, true)}</time></header><p>{selectedRun.objective || '该运行没有保存目标摘要。'}</p></div></article>
          <article className="agent-message is-system"><div className="agent-message__avatar"><Database size={15} /></div><div><header><strong>运行工件</strong></header><p>这是已保存的历史 Run。完整命令、测试和证据将从工件接口读取；当前界面不会根据摘要补造过程。</p></div></article>
        </div>
      </div>
    );
  }

  const tasks = Array.isArray(plan?.tasks) ? plan.tasks : [];
  const currentTaskIndex = Number(agent.current_task_index || 0);
  const goal = plan?.project_goal || agent.project_goal || '尚未设置研究目标';
  return (
    <div className="agent-task-thread">
      <header className="agent-thread-header">
        <div><span>当前研究线程</span><h2>{goal}</h2><p>Loop {status.round || 0} · 当前研究门：{GATES.find((gate) => gate.id === currentGateId(status.current_phase, status.status))?.label}</p></div>
        <span className={`agent-status-pill is-${agent.status || status.status || 'idle'}`}><span className={`agent-status-dot is-${status.status || 'idle'}`} />{STATUS_LABEL[agent.status] || STATUS_LABEL[status.status] || '未启动'}</span>
      </header>

      <div className="agent-thread-history">
        <article className="agent-message is-user">
          <div className="agent-message__avatar"><Goal size={15} /></div>
          <div><header><strong>研究目标</strong><time>{formatTime(agent.started_at, true)}</time></header><p>{goal}</p></div>
        </article>

        {plan ? (
          <article className="agent-message is-agent">
            <div className="agent-message__avatar"><Bot size={16} /></div>
            <div className="agent-message__content">
              <header><strong>研究代理</strong><span>已生成有界计划</span></header>
              <p>计划以证据缺口为入口，执行结果必须经过完整工件审计与独立复核。</p>
              <div className="agent-plan-steps">
                {tasks.map((task, index) => {
                  const current = Boolean(agent.enabled) && index === currentTaskIndex;
                  const completed = task.status === 'completed';
                  return (
                    <article key={task.id} className={current ? 'is-current' : completed ? 'is-completed' : ''}>
                      <span>{completed ? <Check size={12} /> : index + 1}</span>
                      <div><strong>{task.title}</strong><p>{task.objective}</p><small>完成条件：{task.completion_rule}</small></div>
                      <em>{decisionLabel(task.decision || (current ? 'running' : null))}</em>
                    </article>
                  );
                })}
              </div>
            </div>
          </article>
        ) : (
          <article className="agent-message is-system">
            <div className="agent-message__avatar"><ListChecks size={15} /></div>
            <div><header><strong>等待任务</strong></header><p>在下方输入项目目标，先生成计划进行检查，或直接启动自动研发。</p></div>
          </article>
        )}

        {events.map((event) => <TimelineEvent key={event.id} event={event} />)}

        {agent.stop_reason ? (
          <article className="agent-message is-warning"><div className="agent-message__avatar"><AlertTriangle size={15} /></div><div><header><strong>停止原因</strong></header><p>{agent.stop_reason}</p></div></article>
        ) : null}
      </div>
    </div>
  );
}

function EvidenceInspector({ orchestrator }) {
  const evidence = orchestrator.evidence_context || {};
  const roles = orchestrator.roles || {};
  return (
    <div className="agent-inspector-stack">
      <div className="agent-inspector-metrics"><article><span>Runs</span><strong>{evidence.run_count || 0}</strong></article><article><span>模型</span><strong>{evidence.models?.length || 0}</strong></article><article><span>缺口</span><strong>{evidence.open_gaps?.length || 0}</strong></article></div>
      <section><h3>证据来源</h3><p>{evidence.source || '尚无 Evidence Kernel'}</p></section>
      <section><h3>独立角色</h3>{Object.entries(roles).map(([id, description]) => <article className="agent-role-card" key={id}><strong>{id}</strong><p>{description}</p></article>)}</section>
      <div className="agent-boundary-note"><ShieldCheck size={15} /><p>多模型一致意见只作为评论，不能代替实验复现和因果证据。</p></div>
    </div>
  );
}

function ModelsInspector({ config, setConfig, saving, onSave }) {
  const analysts = config.analyst_models || [];
  const updateAnalyst = (index, value) => setConfig({ ...config, analyst_models: analysts.map((item, itemIndex) => itemIndex === index ? value : item) });
  return (
    <div className="agent-model-config">
      <section><div className="agent-inspector-heading"><div><Cpu size={15} /><h3>主研发模型</h3></div><span>{config.master_model?.model_id}</span></div><ModelFields master model={config.master_model} onChange={(value) => setConfig({ ...config, master_model: { ...value, model_type: 'master' } })} /></section>
      <section>
        <div className="agent-inspector-heading"><div><Bot size={15} /><h3>辅助分析模型</h3></div><button type="button" onClick={() => setConfig({ ...config, analyst_models: [...analysts, { ...EMPTY_ANALYST, name: `独立分析模型 ${analysts.length + 1}` }] })}><Plus size={13} />添加</button></div>
        {analysts.length ? analysts.map((analyst, index) => (
          <details className="agent-analyst-config" key={`${analyst.name}-${index}`}>
            <summary><span>{analyst.name || `辅助模型 ${index + 1}`}</span><small>{analyst.model_id}</small><ChevronDown size={13} /></summary>
            <button type="button" className="agent-remove-model" onClick={() => setConfig({ ...config, analyst_models: analysts.filter((_, itemIndex) => itemIndex !== index) })}><Trash2 size={13} />删除模型</button>
            <ModelFields model={analyst} onChange={(value) => updateAnalyst(index, value)} />
          </details>
        )) : <div className="agent-blank-state">至少添加一个独立分析模型。</div>}
      </section>
      <button type="button" className="agent-primary-button agent-save-models" disabled={saving} onClick={onSave}><Save size={14} />保存模型与提示词</button>
    </div>
  );
}

function RunInspector({ tab, setTab, status, agent, plan, orchestrator, config, setConfig, saving, onSave, events, selectedRun }) {
  const tasks = Array.isArray(plan?.tasks) ? plan.tasks : [];
  const currentTask = tasks[Number(agent.current_task_index || 0)] || tasks[0];
  const lastResult = [...events].reverse().find((event) => event.type === 'execution_result');
  const recentRuns = orchestrator.recent_runs || [];
  return (
    <aside className="agent-run-inspector" aria-label="运行检查器">
      <nav>{INSPECTOR_TABS.map((item) => { const Icon = item.icon; return <button type="button" key={item.id} className={tab === item.id ? 'is-active' : ''} onClick={() => setTab(item.id)} title={item.label}><Icon size={15} /><span>{item.label}</span></button>; })}</nav>
      <div className="agent-run-inspector__content">
        {tab === 'plan' ? <div className="agent-inspector-stack"><section><div className="agent-inspector-heading"><h3>证据门</h3><span>Loop {status.round || 0}</span></div><GateStrip phase={status.current_phase} status={status.status} /></section><section><div className="agent-inspector-heading"><h3>任务队列</h3><span>{tasks.filter((task) => task.status === 'completed').length}/{tasks.length}</span></div>{tasks.length ? tasks.map((task, index) => <article className={`agent-inspector-task ${index === Number(agent.current_task_index || 0) && agent.enabled ? 'is-current' : ''}`} key={task.id}><span>{task.status === 'completed' ? <CheckCircle2 size={14} /> : index + 1}</span><div><strong>{task.title}</strong><small>{decisionLabel(task.decision)}</small></div></article>) : <div className="agent-blank-state">尚未生成计划。</div>}</section></div> : null}
        {tab === 'contract' ? <div className="agent-inspector-stack"><section><h3>当前完成条件</h3><p>{currentTask?.completion_rule || '生成计划后冻结完成条件。'}</p></section><section className="agent-contract-grid"><article><span>运行方式</span><strong>{status.mode === 'manual' ? '手动逐门' : '自动连续'}</strong></article><article><span>任务上限</span><strong>{agent.config?.max_loops || 3} Loops</strong></article><article><span>支持即停止</span><strong>{agent.config?.stop_on_accepted === false ? '否' : '是'}</strong></article><article><span>反证即停止</span><strong>{agent.config?.stop_on_rejected === false ? '否' : '是'}</strong></article></section><div className="agent-boundary-note"><ShieldCheck size={15} /><p>计划完成不等于理论成立；Agent 不能修改 lockbox 或批准闭合。</p></div></div> : null}
        {tab === 'artifacts' ? <div className="agent-inspector-stack"><section><h3>当前 Run</h3><code>{selectedRun?.run_id || status.active_run_id || '尚未产生运行工件'}</code></section><section><div className="agent-inspector-heading"><h3>最近工件</h3><span>{recentRuns.length}</span></div>{recentRuns.slice(0, 8).map((run) => <article className="agent-artifact-card" key={run.run_id}><Database size={14} /><div><strong>{run.objective || run.run_id}</strong><small>{run.run_id}</small></div><span className={`is-${run.decision || 'pending'}`}>{decisionLabel(run.decision)}</span></article>)}</section></div> : null}
        {tab === 'tests' ? <div className="agent-inspector-stack"><section><h3>最新执行结果</h3>{lastResult ? <pre>{JSON.stringify(lastResult.result, null, 2)}</pre> : <div className="agent-blank-state">本次连接尚未收到执行结果。</div>}</section><div className="agent-boundary-note"><FlaskConical size={15} /><p>测试结果必须保留退出码、标准输出、输入合同和工件哈希。</p></div></div> : null}
        {tab === 'evidence' ? <EvidenceInspector orchestrator={orchestrator} /> : null}
        {tab === 'models' ? <ModelsInspector config={config} setConfig={setConfig} saving={saving} onSave={onSave} /> : null}
      </div>
    </aside>
  );
}

function TaskComposer({ form, setForm, status, agent, ready, analystCount, busy, onProjectAction, onSessionAction, onOpenModels }) {
  const active = Boolean(agent.enabled);
  const set = (key, value) => setForm((current) => ({ ...current, [key]: value }));
  const setMode = (value) => {
    set('execution_mode', value);
    if (active) onSessionAction('mode', value);
  };
  return (
    <div className="agent-task-composer">
      <textarea value={form.project_goal} disabled={active} onChange={(event) => set('project_goal', event.target.value)} placeholder="描述要完成的项目研发目标；留空时从最高优先级证据缺口开始……" />
      <div className="agent-task-composer__context">
        <span><Paperclip size={13} />当前项目</span>
        <span><ShieldCheck size={12} />Evidence-first</span>
        <span><Cpu size={12} />1 主模型 + {analystCount} 复核模型</span>
      </div>
      <div className="agent-task-composer__actions">
        <div className="agent-mode-switch" aria-label="执行模式"><button type="button" className={(active ? status.mode : form.execution_mode) === 'auto' ? 'is-active' : ''} onClick={() => setMode('auto')}>自动</button><button type="button" className={(active ? status.mode : form.execution_mode) === 'manual' ? 'is-active' : ''} onClick={() => setMode('manual')}>手动</button></div>
        {!active ? <label className="agent-loop-limit">最多<input type="number" min="1" max="12" value={form.max_loops} onChange={(event) => set('max_loops', Number(event.target.value))} />轮</label> : null}
        <div className="agent-task-composer__primary-actions">
          {!active ? <button type="button" disabled={busy} onClick={() => onProjectAction('plan')}><ListChecks size={14} />生成计划</button> : null}
          {!active && !ready ? <button type="button" onClick={onOpenModels}><Settings2 size={14} />配置模型</button> : null}
          {!active ? <button type="button" className="agent-primary-button" disabled={busy || !ready} onClick={() => onProjectAction('start')}><Play size={14} />开始研发</button> : null}
          {active && status.status === 'running' ? <button type="button" onClick={() => onSessionAction('pause')} disabled={busy}><Pause size={14} />暂停</button> : null}
          {active && status.status === 'paused' ? <button type="button" className="agent-primary-button" onClick={() => onSessionAction('resume')} disabled={busy}><Play size={14} />继续</button> : null}
          {active && status.status === 'waiting_step' ? <button type="button" className="agent-primary-button" onClick={() => onSessionAction('step')} disabled={busy}><SkipForward size={14} />确认下一门</button> : null}
          {active ? <button type="button" className="agent-stop-button" onClick={() => onProjectAction('stop')} disabled={busy}><Square size={13} />停止</button> : null}
        </div>
      </div>
    </div>
  );
}

function TerminalDrawer({ open, setOpen, tab, setTab, events }) {
  const codeEvent = [...events].reverse().find((event) => event.type === 'code_generated');
  const resultEvent = [...events].reverse().find((event) => event.type === 'execution_result');
  return (
    <section className={`agent-terminal-drawer ${open ? 'is-open' : ''}`}>
      <header>
        <nav><button type="button" className={tab === 'logs' ? 'is-active' : ''} onClick={() => { setTab('logs'); setOpen(true); }}><Terminal size={13} />实时日志</button><button type="button" className={tab === 'code' ? 'is-active' : ''} onClick={() => { setTab('code'); setOpen(true); }}><Code2 size={13} />生成代码</button><button type="button" className={tab === 'result' ? 'is-active' : ''} onClick={() => { setTab('result'); setOpen(true); }}><FlaskConical size={13} />测试输出</button></nav>
        <button type="button" className="agent-drawer-toggle" onClick={() => setOpen((value) => !value)} aria-label={open ? '收起运行面板' : '展开运行面板'}><ChevronDown size={15} /></button>
      </header>
      {open ? <div className="agent-terminal-drawer__body">
        {tab === 'logs' ? (events.length ? events.slice(-80).map((event) => <div className="agent-log-row" key={event.id}><time>{formatTime(event.timestamp)}</time><span>{EVENT_META[event.type]?.label || event.type}</span><p>{eventText(event)}</p></div>) : <div className="agent-drawer-empty">启动任务后，结构化命令和实时事件将显示在这里。</div>) : null}
        {tab === 'code' ? (codeEvent?.code ? <pre>{codeEvent.code}</pre> : <div className="agent-drawer-empty">尚未收到代码产物。</div>) : null}
        {tab === 'result' ? (resultEvent ? <pre>{JSON.stringify(resultEvent.result, null, 2)}</pre> : <div className="agent-drawer-empty">尚未收到测试输出。</div>) : null}
      </div> : null}
    </section>
  );
}

export function LoopEngineeringWorkspace({ mode = 'sidebar', onClose }) {
  const [status, setStatus] = useState({ status: 'idle', mode: 'auto', round: 0 });
  const [orchestrator, setOrchestrator] = useState({});
  const [config, setConfig] = useState(() => normalizeConfig({}));
  const [projectAgentForm, setProjectAgentForm] = useState(DEFAULT_PROJECT_AGENT_FORM);
  const [projectAgentPlan, setProjectAgentPlan] = useState(null);
  const [events, setEvents] = useState([]);
  const [selectedThreadId, setSelectedThreadId] = useState('project');
  const [inspectorTab, setInspectorTab] = useState('plan');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState('logs');
  const [streamConnected, setStreamConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const eventId = useRef(0);

  const load = useCallback(async ({ quiet = false, includeConfig = true } = {}) => {
    if (!quiet) setBusy(true);
    try {
      const requests = [
        fetch(`${API_BASE}/api/ai-rnd/session/status`, { cache: 'no-store' }).then(readJson),
        fetch(`${API_BASE}/api/ai-rnd/orchestrator/status`, { cache: 'no-store' }).then(readJson),
      ];
      if (includeConfig) requests.push(fetch(`${API_BASE}/api/ai-rnd/config`, { cache: 'no-store' }).then(readJson));
      const [statusPayload, orchestratorPayload, configPayload] = await Promise.all(requests);
      setStatus(statusPayload);
      setOrchestrator(orchestratorPayload);
      if (includeConfig) setConfig(normalizeConfig(configPayload));
      const savedAgent = statusPayload.project_agent || {};
      const savedAgentConfig = savedAgent.config || {};
      setProjectAgentForm((current) => ({
        ...current,
        project_goal: savedAgent.project_goal || current.project_goal,
        max_loops: Number(savedAgentConfig.max_loops || current.max_loops),
        execution_mode: savedAgentConfig.execution_mode || current.execution_mode,
        stop_on_accepted: savedAgentConfig.stop_on_accepted ?? current.stop_on_accepted,
        stop_on_rejected: savedAgentConfig.stop_on_rejected ?? current.stop_on_rejected,
        max_consecutive_inconclusive: Number(savedAgentConfig.max_consecutive_inconclusive || current.max_consecutive_inconclusive),
      }));
      if (savedAgent.plan) setProjectAgentPlan(savedAgent.plan);
      setError('');
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      if (!quiet) setBusy(false);
    }
  }, []);

  useEffect(() => {
    load();
    const timer = window.setInterval(() => load({ quiet: true, includeConfig: false }), 5000);
    return () => window.clearInterval(timer);
  }, [load]);

  useEffect(() => {
    const stream = new EventSource(`${API_BASE}/api/ai-rnd/session/events`);
    stream.onopen = () => setStreamConnected(true);
    stream.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data);
        eventId.current += 1;
        setEvents((current) => [...current.slice(-299), { ...event, id: `${event.timestamp || Date.now()}-${eventId.current}` }]);
        if (['status_change', 'project_agent_status', 'project_agent_progress', 'database_writeback', 'finding'].includes(event.type)) {
          load({ quiet: true, includeConfig: false });
        }
      } catch {
        // Keep the UI alive when a malformed optional event is received.
      }
    };
    stream.onerror = () => setStreamConnected(false);
    return () => stream.close();
  }, [load]);

  const action = useCallback(async (type, value) => {
    setBusy(true);
    setError('');
    try {
      const requests = {
        resume: () => fetch(`${API_BASE}/api/ai-rnd/session/start`, { method: 'POST' }),
        pause: () => fetch(`${API_BASE}/api/ai-rnd/session/pause`, { method: 'POST' }),
        stop: () => fetch(`${API_BASE}/api/ai-rnd/session/stop`, { method: 'POST' }),
        step: () => fetch(`${API_BASE}/api/ai-rnd/session/step`, { method: 'POST' }),
        mode: () => fetch(`${API_BASE}/api/ai-rnd/session/mode`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: value }) }),
      };
      await readJson(await requests[type]());
      await load({ quiet: true, includeConfig: false });
    } catch (actionError) {
      setError(actionError.message);
    } finally {
      setBusy(false);
    }
  }, [load]);

  const projectAgentAction = useCallback(async (type) => {
    setBusy(true);
    setError('');
    try {
      if (type === 'plan') {
        const planPayload = await readJson(await fetch(`${API_BASE}/api/ai-rnd/project-agent/plan`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ project_goal: projectAgentForm.project_goal.trim(), max_tasks: projectAgentForm.max_loops }),
        }));
        setProjectAgentPlan(planPayload);
        setSelectedThreadId('project');
        setInspectorTab('plan');
      } else if (type === 'start') {
        const payload = await readJson(await fetch(`${API_BASE}/api/ai-rnd/project-agent/start`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...projectAgentForm, project_goal: projectAgentForm.project_goal.trim() }),
        }));
        if (payload.project_agent?.plan) setProjectAgentPlan(payload.project_agent.plan);
        setSelectedThreadId('project');
        setDrawerOpen(true);
        await load({ quiet: true, includeConfig: false });
      } else if (type === 'stop') {
        await readJson(await fetch(`${API_BASE}/api/ai-rnd/project-agent/stop`, { method: 'POST' }));
        await load({ quiet: true, includeConfig: false });
      }
    } catch (actionError) {
      setError(actionError.message);
    } finally {
      setBusy(false);
    }
  }, [load, projectAgentForm]);

  const saveConfig = useCallback(async () => {
    setBusy(true);
    setError('');
    try {
      await readJson(await fetch(`${API_BASE}/api/ai-rnd/config`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) }));
    } catch (saveError) {
      setError(saveError.message);
    } finally {
      setBusy(false);
    }
  }, [config]);

  const agent = status.project_agent || {};
  const plan = agent.plan || projectAgentPlan;
  const runs = useMemo(() => Array.isArray(orchestrator.recent_runs) ? orchestrator.recent_runs : [], [orchestrator.recent_runs]);
  const selectedRun = selectedThreadId.startsWith('run:') ? runs.find((run) => `run:${run.run_id}` === selectedThreadId) : null;
  const masterReady = Boolean(config.master_model?.api_key?.trim());
  const analystCount = (config.analyst_models || []).filter((model) => model.api_key?.trim()).length;
  const ready = masterReady && analystCount > 0;

  return (
    <aside className={`simple-research-center loop-workspace agent-workbench is-${mode}`} aria-label="AI 自动研发">
      <header className="agent-workbench__topbar">
        <div className="agent-workbench__brand"><span><Bot size={18} /></span><div><h1>AI 自动研发</h1><p>项目研究代理 · 可审计任务工作台</p></div></div>
        <div className="agent-workbench__runtime">
          <span><span className={`agent-status-dot is-${status.status || 'idle'}`} />{STATUS_LABEL[agent.status] || STATUS_LABEL[status.status] || '未运行'}</span>
          <span><Cpu size={13} />{config.master_model?.model_id || '未配置主模型'}</span>
          <span><Database size={13} />{status.active_run_id || '无活动 Run'}</span>
          <span title={streamConnected ? '实时事件已连接' : '使用状态轮询'}><span className={`agent-stream-dot ${streamConnected ? 'is-live' : ''}`} />{streamConnected ? '实时' : '轮询'}</span>
        </div>
        <div className="agent-workbench__top-actions">
          <button type="button" onClick={() => { setInspectorTab('models'); setSelectedThreadId('project'); }} title="模型与提示词"><Settings2 size={16} /></button>
          <button type="button" onClick={() => load()} title="刷新"><RefreshCw size={16} className={busy ? 'is-spinning' : ''} /></button>
          {mode === 'page' ? <button type="button" className="agent-back-button" onClick={onClose}><ArrowUp size={15} /><span>返回 3D</span></button> : <button type="button" onClick={onClose} title="关闭"><X size={17} /></button>}
        </div>
      </header>

      {error ? <div className="agent-workbench__error"><AlertTriangle size={15} /><span>{error}</span><button type="button" onClick={() => setError('')}><X size={14} /></button></div> : null}

      <div className="agent-workbench__layout">
        <ThreadRail agent={agent} plan={plan} runs={runs} status={status} selectedId={selectedThreadId} onSelect={setSelectedThreadId} />
        <main className="agent-workbench__main">
          <TaskThread agent={agent} plan={plan} status={status} events={events} selectedRun={selectedRun} />
          {!selectedRun ? <TaskComposer form={projectAgentForm} setForm={setProjectAgentForm} status={status} agent={agent} ready={ready} analystCount={analystCount} busy={busy} onProjectAction={projectAgentAction} onSessionAction={action} onOpenModels={() => setInspectorTab('models')} /> : null}
          <TerminalDrawer open={drawerOpen} setOpen={setDrawerOpen} tab={drawerTab} setTab={setDrawerTab} events={events} />
        </main>
        <RunInspector tab={inspectorTab} setTab={setInspectorTab} status={status} agent={agent} plan={plan} orchestrator={orchestrator} config={config} setConfig={setConfig} saving={busy} onSave={saveConfig} events={events} selectedRun={selectedRun} />
      </div>

      <footer className="agent-workbench__footer"><ShieldCheck size={13} /><span>Agent 可以执行有界任务并整理证据，但不能自动批准理论闭合。</span><span><Clock3 size={12} />{formatTime(agent.started_at, true)}</span></footer>
    </aside>
  );
}
