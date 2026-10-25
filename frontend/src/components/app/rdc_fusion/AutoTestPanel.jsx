/* 研发 · 自动测试（autotest.v1）：登记多个被测模型 → 勾选模式族 → 一键运行。
   中心为每个「模型 × 模式族」组合生成一个作业，在中心所在机器上顺序执行完整流程：
   文法预览 → 试跑与行为门 → 冻结正式模板 → 分片采集 → 合并 → 规律分析 → 报告。
   被测模型必须是本机权重（只有 API 的模型拿不到内部状态）；合成模型只在沙盒中心可用并醒目标注。
   数据：/api/autotest/state（轮询）· /api/grammar/library · /api/models/local；写操作需管理令牌。 */
import { useMemo, useState } from 'react';
import { fmtAgoTs, postJson, useAdminToken, useApi } from './api.js';
import { AdminTokenField, Card, Empty, SourceTag, StatusPill } from './ui.jsx';
import { STATUS_META } from './catalogUtil.js';

const JOB_STATUS = {
  queued: ['…', '排队', 'muted'], running: ['▶', '运行中', 'info'], done: ['✓', '完成', 'good'],
  failed: ['✗', '失败', 'bad'], stopped: ['■', '行为门停止', 'warn'], cancelled: ['–', '已取消', 'muted'],
  interrupted: ['!', '中断', 'warn'],
};
const FINDING = [['supported', '支持', 'good'], ['negative', '未支持', 'bad'], ['warning', '注意', 'warn']];
const EMPTY_FORM = { kind: 'local', label: '', model_id: '', model_path: '', dtype: 'bf16', gpu: '', model_rev: '' };

function JobPill({ status }) {
  const [icon, text, tone] = JOB_STATUS[status] || ['·', status, 'muted'];
  return <span className={'v6-pill ' + tone}><i aria-hidden="true">{icon}</i>{text}</span>;
}

function fmtDur(job) {
  if (!job.started_at) return '—';
  const s = Math.max(0, Math.round((job.ended_at || Date.now() / 1000) - job.started_at));
  if (s < 60) return `${s} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
  return `${Math.floor(s / 3600)} 时 ${Math.floor((s % 3600) / 60)} 分`;
}

function fmtBytes(n) {
  if (!n) return '';
  return n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(1)} GB` : `${(n / (1 << 20)).toFixed(0)} MB`;
}

function slug(text) {
  return String(text || '').trim().split(/[\\/]/).filter(Boolean).pop()?.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^[-.]+/, '').slice(0, 64).toLowerCase() || '';
}

/* 7 步进度条：完成段实色，当前段描边，失败/停止段标红/黄 */
function StepBar({ job, steps }) {
  return (
    <ol className="v6-at-steps" aria-label="流程进度">
      {steps.map(st => {
        let cls = 'todo';
        if (job.status === 'done' || st.n < job.step) cls = 'done';
        else if (st.n === job.step) cls = job.status === 'running' ? 'now' : job.status === 'failed' || job.status === 'interrupted' ? 'bad' : job.status === 'stopped' ? 'warn' : 'todo';
        return <li key={st.n} className={cls} title={`${st.n}. ${st.label}`}><span>{st.label}</span></li>;
      })}
    </ol>
  );
}

function FindingCounts({ result }) {
  if (!result?.findings) return null;
  return (
    <span className="v6-at-finds">
      {FINDING.map(([k, label, tone]) => (result.findings[k] ? <span key={k} className={'v6-pill ' + tone}>{label} {result.findings[k]}</span> : null))}
    </span>
  );
}

function ModelForm({ initial, sandbox, token, onSaved, onCancel }) {
  const [form, setForm] = useState(initial || EMPTY_FORM);
  const [probe, setProbe] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const local = useApi(form.kind === 'local' ? '/api/models/local' : null);
  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));
  const editing = Boolean(initial);

  const check = async path => {
    setProbe({ loading: true });
    try { setProbe(await postJson('/api/autotest/probe', { model_path: path })); }
    catch (e) { setProbe({ ok: false, error: e.message }); }
  };
  const pick = path => {
    const m = (local.data?.models || []).find(x => x.path === path);
    if (!m) return;
    setForm(f => ({ ...f, model_path: m.path, model_id: editing ? f.model_id : (f.model_id || slug(m.name)), label: f.label || m.name }));
    check(m.path);
  };
  const save = async () => {
    setBusy(true); setErr('');
    try {
      const body = { ...form, model_id: form.model_id || slug(form.model_path) || slug(form.label) };
      const res = await postJson('/api/autotest/models', body, { adminToken: token });
      onSaved(res.model);
    } catch (e) { setErr(e.message); }
    setBusy(false);
  };

  return (
    <div className="v6-at-form">
      <div className="v6-at-form-row">
        <label>类型
          <select value={form.kind} onChange={e => set('kind', e.target.value)} disabled={editing}>
            <option value="local">本机权重（HF 目录）</option>
            {(sandbox || form.kind === 'synthetic') && <option value="synthetic">合成 · 沙盒验证</option>}
          </select>
        </label>
        <label>显示名称<input value={form.label} placeholder="如 Qwen3-4B" onChange={e => set('label', e.target.value)} /></label>
        <label>model_id<input value={form.model_id} disabled={editing} placeholder="qwen3-4b（进入模型桶名）" onChange={e => set('model_id', e.target.value.trim())} /></label>
      </div>
      {form.kind === 'local' && (
        <>
          {(local.data?.models || []).length > 0 && (
            <label>从本机模型库选择
              <select value="" onChange={e => pick(e.target.value)}>
                <option value="">— 已下载 {local.data.models.length} 个（{local.data.root}）—</option>
                {local.data.models.map(m => <option key={m.path} value={m.path}>{m.name} · {m.layers ?? '?'} 层 · {fmtBytes(m.bytes)}</option>)}
              </select>
            </label>
          )}
          <label>权重目录
            <span className="v6-at-path">
              <input value={form.model_path} placeholder="D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b" onChange={e => { set('model_path', e.target.value); setProbe(null); }} />
              <button type="button" className="fw-tbtn" disabled={!form.model_path} onClick={() => check(form.model_path)}>检查</button>
            </span>
          </label>
          {probe && !probe.loading && (probe.ok
            ? <p className="v6-note"><StatusPill status="pass" label="权重就绪" /> {probe.arch || '未知架构'} · {probe.layers ?? '?'} 层 · 隐藏维 {probe.hidden ?? '?'} · {probe.files} 个 safetensors · {fmtBytes(probe.bytes)}</p>
            : <p className="v6-note warn"><StatusPill status="fail" label="不可用" /> {probe.error}</p>)}
          <div className="v6-at-form-row">
            <label>精度<select value={form.dtype} onChange={e => set('dtype', e.target.value)}>{['bf16', 'fp16', 'fp32'].map(d => <option key={d}>{d}</option>)}</select></label>
            <label>GPU（可选）<input value={form.gpu} placeholder="RTX 5080" onChange={e => set('gpu', e.target.value)} /></label>
            <label>快照（可选）<input value={form.model_rev} placeholder="缺省 = config.json 摘要" onChange={e => set('model_rev', e.target.value.trim())} /></label>
          </div>
        </>
      )}
      {form.kind === 'synthetic' && <p className="v6-note warn">合成模型由 runner 按已知规律生成数据，只用于验证流程与分析方法，结果不代表任何真实模型。</p>}
      {err && <div className="fw-ai-err">{err}</div>}
      <div className="v6-at-form-act">
        <button type="button" className="fw-tbtn primary" disabled={busy || !token || (form.kind === 'local' && !form.model_path)} onClick={save}>{busy ? '保存中…' : editing ? '保存修改' : '登记模型'}</button>
        <button type="button" className="fw-tbtn" onClick={onCancel}>取消</button>
        {!token && <small className="muted">需要管理令牌</small>}
      </div>
    </div>
  );
}

function JobLog({ job }) {
  const log = useApi(`/api/autotest/jobs/${job.id}/log?tail=400`, { interval: job.status === 'running' ? 2000 : 0 });
  return (
    <pre className="v6-at-log" aria-label="作业日志">
      {log.data ? (log.data.lines.join('\n') || '（暂无输出）') : log.error || '读取日志…'}
    </pre>
  );
}

export default function AutoTestPanel({ onGo, onStation, presetFamilies }) {
  const [token, setToken] = useAdminToken();
  const state = useApi('/api/autotest/state', { interval: 3000 });
  const lib = useApi('/api/grammar/library');
  const catalog = useApi('/api/catalog', { interval: 20000 });
  const [selModels, setSelModels] = useState(() => new Set());
  const [selFams, setSelFams] = useState(() => new Set(presetFamilies || []));
  const [opts, setOpts] = useState({ pilot: true, pilot_per_cell: 3, min_pilot_pass: 0.5 });
  const [editing, setEditing] = useState(null);   // null | 'new' | model
  const [openLog, setOpenLog] = useState('');
  const [busy, setBusy] = useState('');
  const [msg, setMsg] = useState(null);

  const st = state.data;
  const models = st?.models || [];
  const families = (lib.data?.grammars || []).filter(g => g.family_id);
  const jobs = useMemo(() => st?.jobs || [], [st]);
  const steps = st?.steps || [];

  const readyModels = [...selModels].filter(id => models.find(m => m.model_id === id)?.check?.ok);
  /* 题库状态：每个模式族的总体状态，以及哪些模型已有分片齐全的结果（模型桶名含 model_id|） */
  const catItem = Object.fromEntries((catalog.data?.items || []).map(it => [it.id, it]));
  const doneFor = (fam, mid) => (catItem[fam]?.models || []).some(m => m.complete && m.has_agg && m.bucket.includes(`${mid}|`));
  const pickUnfinished = () => setSelFams(new Set(families.filter(g => catItem[g.family_id]?.status !== 'done').map(g => g.family_id)));
  const pickMissing = () => setSelFams(new Set(families.filter(g => readyModels.some(mid => !doneFor(g.family_id, mid))).map(g => g.family_id)));
  const nJobs = readyModels.length * selFams.size;
  const toggle = (setter, id) => setter(prev => { const n = new Set(prev); if (n.has(id)) n.delete(id); else n.add(id); return n; });

  const act = async (key, fn, okText) => {
    setBusy(key); setMsg(null);
    try { const r = await fn(); setMsg({ ok: true, text: typeof okText === 'function' ? okText(r) : okText }); state.reload(); }
    catch (e) { setMsg({ ok: false, text: e.message }); }
    setBusy('');
  };
  const run = () => act('run', () => postJson('/api/autotest/batches', {
    families: [...selFams], models: readyModels,
    options: { skip_pilot: !opts.pilot, pilot_per_cell: opts.pilot_per_cell, min_pilot_pass: opts.min_pilot_pass },
  }, { adminToken: token }), r => `已加入队列 ${r.created.length} 个作业（批次 ${r.batch}）` + (r.skipped_duplicates.length ? `；跳过已在队列中的 ${r.skipped_duplicates.join('、')}` : ''));

  /* 结果矩阵：每个「模型 × 模式族」取最新一次作业 */
  const latest = useMemo(() => {
    const out = {};
    for (const j of jobs) { const k = `${j.model_id}\u0000${j.family}`; if (!out[k]) out[k] = j; }
    return out;
  }, [jobs]);
  const mxModels = [...new Set(jobs.map(j => j.model_id))];
  const mxFams = [...new Set(jobs.map(j => j.family))].sort();
  const viewResult = job => onGo && onGo('spatial', null, undefined, { tm: job.tm_id, bucket: job.bucket });

  if (!st) {
    return <Empty>{state.error ? `中心服务未接入（${state.error}）。自动测试只在研究者本机的中心上可用。` : '读取自动测试状态…'}</Empty>;
  }

  return (
    <div className="v6-at">
      <header className="v6-at-hd">
        <div>
          <h3>自动测试 · 被测模型 × 模式族</h3>
          <p>勾选模型和模式族后点击运行：每个组合自动走完 试跑行为门 → 冻结 → 分片采集 → 合并 → 规律分析 → 报告，作业在中心所在机器上逐个执行（一张 GPU 同时只跑一个模型）。</p>
        </div>
        <div className="v6-at-hd-r">
          <SourceTag state="live" label="作业队列" />
          <AdminTokenField token={token} onChange={setToken} />
        </div>
      </header>
      {st.sandbox_synthetic && (
        <div className="v6-synthetic"><b>沙盒中心</b><span>可登记「合成」模型验证流程；合成结果的模型桶带「合成·」前缀，不代表任何真实模型。</span></div>
      )}

      <div className="v6-at-grid">
        <Card title="① 被测模型" sub="本机权重；只有 API 的模型拿不到内部状态，不能作为被测对象"
          extra={<button type="button" className="fw-tbtn" onClick={() => setEditing('new')} disabled={editing === 'new'}>＋ 添加模型</button>}>
          {editing && (
            <ModelForm key={editing === 'new' ? 'new' : editing.model_id} initial={editing === 'new' ? null : editing} sandbox={st.sandbox_synthetic} token={token}
              onCancel={() => setEditing(null)}
              onSaved={m => { setEditing(null); setSelModels(prev => new Set(prev).add(m.model_id)); state.reload(); }} />
          )}
          {!models.length && !editing && <Empty action={<button type="button" className="fw-tbtn primary" onClick={() => setEditing('new')}>添加第一个模型</button>}>还没有登记被测模型。</Empty>}
          <ul className="v6-at-list">
            {models.map(m => (
              <li key={m.model_id} className={selModels.has(m.model_id) ? 'on' : ''}>
                <label className="v6-at-check">
                  <input type="checkbox" checked={selModels.has(m.model_id) && Boolean(m.check?.ok)} disabled={!m.check?.ok} onChange={() => toggle(setSelModels, m.model_id)} />
                  <span>
                    <b>{m.label || m.model_id}</b>
                    <small className="mono">{m.model_id} · {m.dtype}{m.gpu ? ` · ${m.gpu}` : ''}</small>
                    {m.kind === 'synthetic'
                      ? <small className="warn">合成 · 沙盒验证</small>
                      : <small className="mono v6-at-ellipsis" title={m.model_path}>{m.model_path}</small>}
                  </span>
                </label>
                <span className="v6-at-li-r">
                  {m.check?.ok
                    ? <StatusPill status="pass" label={m.kind === 'synthetic' ? '可用' : `${m.check.layers ?? '?'} 层 · ${fmtBytes(m.check.bytes)}`} />
                    : <StatusPill status="fail" label={m.check?.error || '不可用'} title={m.model_path} />}
                  <button type="button" className="fw-tbtn ghost" onClick={() => setEditing(m)}>编辑</button>
                  <button type="button" className="fw-tbtn ghost" disabled={!token || busy === 'del' + m.model_id}
                    onClick={() => act('del' + m.model_id, () => postJson(`/api/autotest/models/${encodeURIComponent(m.model_id)}`, null, { adminToken: token, method: 'DELETE' }), `已删除 ${m.model_id}`)}>删除</button>
                </span>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="② 模式族" sub="来自文法库；每个模式族 = 句式框架 × 填充词表 × 因子"
          extra={families.length > 0 && (
            <span className="v6-at-quick">
              <button type="button" className="fw-tbtn ghost" onClick={() => setSelFams(selFams.size === families.length ? new Set() : new Set(families.map(f => f.family_id)))}>{selFams.size === families.length ? '全不选' : '全选'}</button>
              {catalog.data && <button type="button" className="fw-tbtn ghost" title="题库状态不是「已完成」的模式族" onClick={pickUnfinished}>只选未完成</button>}
              {catalog.data && readyModels.length > 0 && <button type="button" className="fw-tbtn ghost" title="所选模型中至少一个还没有分片齐全结果的模式族" onClick={pickMissing}>所选模型未测的</button>}
            </span>
          )}>
          {!lib.data && <Empty>{lib.error ? `文法库不可用（${lib.error}）` : '读取文法库…'}</Empty>}
          <ul className="v6-at-list">
            {families.map(g => (
              <li key={g.family_id} className={selFams.has(g.family_id) ? 'on' : ''}>
                <label className="v6-at-check">
                  <input type="checkbox" checked={selFams.has(g.family_id)} onChange={() => toggle(setSelFams, g.family_id)} />
                  <span>
                    <b><span className="mono">{g.family_id}</span> {g.name}</b>
                    <small>{(g.factors || []).map(f => f.name || f).join(' × ')} · {g.n_frames} 句式 × {g.n_fillers} 填充</small>
                  </span>
                </label>
                <span className="v6-at-li-r">
                  {catItem[g.family_id] && <StatusPill status={(STATUS_META[catItem[g.family_id].status] || ['na'])[0]} label={catItem[g.family_id].status_label} />}
                  {readyModels.length > 0 && catItem[g.family_id] && (
                    <small className="muted" title="所选模型在该模式族上是否已有分片齐全的结果">
                      {readyModels.map(mid => `${mid} ${doneFor(g.family_id, mid) ? '✓' : '○'}`).join(' · ')}
                    </small>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="③ 运行" className="v6-at-run">
          <label className="v6-at-opt"><input type="checkbox" checked={opts.pilot} onChange={e => setOpts(o => ({ ...o, pilot: e.target.checked }))} /> 先试跑行为门（推荐）</label>
          {opts.pilot && (
            <div className="v6-at-form-row">
              <label>每格试跑条数<input type="number" min="1" max="20" value={opts.pilot_per_cell} onChange={e => setOpts(o => ({ ...o, pilot_per_cell: Number(e.target.value) || 3 }))} /></label>
              <label>最低达标比例<input type="number" min="0" max="1" step="0.05" value={opts.min_pilot_pass} onChange={e => setOpts(o => ({ ...o, min_pilot_pass: Number(e.target.value) }))} /></label>
            </div>
          )}
          <p className="v6-note">试跑达标格比例低于阈值时，该组合停止在行为门，不冻结正式模板：模型答不对的材料不进入内部分析。</p>
          <div className="v6-at-sum">
            <b>{readyModels.length}</b> 个模型 × <b>{selFams.size}</b> 个模式族 = <b>{nJobs}</b> 个作业
          </div>
          <button type="button" className="fw-tbtn primary v6-at-go" disabled={!token || !nJobs || busy === 'run'} onClick={run}>
            {busy === 'run' ? '提交中…' : `▶ 运行 ${nJobs || ''} 个测试`}
          </button>
          {!token && <small className="muted">填写右上角管理令牌后可运行</small>}
          {msg && <div className={msg.ok ? 'v6-ok' : 'fw-ai-err'}>{msg.text}</div>}
          <p className="v6-note">队列：排队 {st.queue.queued} · 运行中 {st.queue.running ? 1 : 0}。中心回连地址 <span className="mono">{st.center}</span>。</p>
        </Card>
      </div>

      {mxModels.length > 0 && (
        <Card title="结果矩阵" sub="每格 = 该模型在该模式族上的最新一次作业；完成后点击进入 空间 · 3D 响应空间 / 规律分析">
          <div className="v6-at-mx-wrap">
            <table className="v6-at-mx">
              <thead><tr><th scope="col">模型 \ 模式族</th>{mxFams.map(f => <th key={f} scope="col" className="mono">{f}</th>)}</tr></thead>
              <tbody>
                {mxModels.map(mid => (
                  <tr key={mid}>
                    <th scope="row" className="mono">{mid}</th>
                    {mxFams.map(f => {
                      const j = latest[`${mid}\u0000${f}`];
                      if (!j) return <td key={f}><span className="muted">—</span></td>;
                      return (
                        <td key={f}>
                          {j.status === 'done' && j.tm_id
                            ? <button type="button" className="v6-at-cell" onClick={() => viewResult(j)} title="在 空间 · 实验响应 中查看"><JobPill status={j.status} /><FindingCounts result={j.result} /></button>
                            : <span className="v6-at-cell"><JobPill status={j.status} />{j.status === 'running' && <small>{j.step_label}</small>}</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="运行队列" sub="按提交顺序逐个执行；失败或中断可重跑，日志保存在中心数据目录"
        extra={jobs.some(j => !['queued', 'running'].includes(j.status)) && (
          <button type="button" className="fw-tbtn ghost" disabled={!token || busy === 'clear'} onClick={() => act('clear', () => postJson('/api/autotest/clear', {}, { adminToken: token }), r => `已清除 ${r.removed} 条已结束的作业记录`)}>清除已结束</button>
        )}>
        {!jobs.length && <Empty>还没有作业。勾选上方模型与模式族后点击运行。</Empty>}
        <ul className="v6-at-jobs">
          {jobs.map(j => (
            <li key={j.id} className={'v6-at-job ' + j.status}>
              <div className="v6-at-job-hd">
                <JobPill status={j.status} />
                <b><span className="mono">{j.family}</span> × {j.model?.label || j.model_id}</b>
                {j.model?.kind === 'synthetic' && <span className="v6-pill warn">合成</span>}
                <small className="muted">{j.id} · {fmtAgoTs(j.created_at)} · 用时 {fmtDur(j)}</small>
                <span className="v6-at-job-act">
                  {j.status === 'done' && j.tm_id && <button type="button" className="fw-tbtn primary" onClick={() => viewResult(j)}>查看 3D 与规律 →</button>}
                  {j.status === 'done' && <button type="button" className="fw-tbtn ghost" onClick={() => onStation && onStation('audit')}>审计队列</button>}
                  {(j.status === 'queued' || j.status === 'running') && (
                    <button type="button" className="fw-tbtn ghost" disabled={!token || busy === 'cancel' + j.id} onClick={() => act('cancel' + j.id, () => postJson(`/api/autotest/jobs/${j.id}/cancel`, {}, { adminToken: token }), '已请求取消')}>取消</button>
                  )}
                  {['failed', 'stopped', 'cancelled', 'interrupted', 'done'].includes(j.status) && (
                    <button type="button" className="fw-tbtn ghost" disabled={!token || busy === 'retry' + j.id} onClick={() => act('retry' + j.id, () => postJson(`/api/autotest/jobs/${j.id}/retry`, {}, { adminToken: token }), '已重新加入队列')}>重跑</button>
                  )}
                  <button type="button" className="fw-tbtn ghost" aria-expanded={openLog === j.id} onClick={() => setOpenLog(openLog === j.id ? '' : j.id)}>{openLog === j.id ? '收起日志' : '日志'}</button>
                </span>
              </div>
              <StepBar job={j} steps={steps} />
              <div className="v6-at-job-detail">
                {j.status === 'running' && <span><b>{j.step_label}</b>{j.detail ? ` · ${j.detail}` : ''}</span>}
                {j.status === 'done' && j.result && (
                  <span>
                    {j.result.n_items} 条进入分析（行为门排除 {j.result.n_excluded}）· 模型桶 <span className="mono">{j.bucket}</span> <FindingCounts result={j.result} />
                  </span>
                )}
                {['failed', 'stopped', 'interrupted'].includes(j.status) && <span className={j.status === 'failed' ? 'bad' : 'warn'}>{j.status === 'stopped' ? (j.detail || '试跑行为门未达标') : j.error}</span>}
              </div>
              {j.status === 'done' && j.result?.headlines?.length > 0 && (
                <ul className="v6-at-heads">
                  {j.result.headlines.filter(h => h.kind !== 'info').slice(0, 5).map((h, i) => (
                    <li key={i}><span className={'v6-at-dot ' + h.kind} aria-hidden="true" />{h.title}</li>
                  ))}
                </ul>
              )}
              {j.status === 'done' && j.result?.report && <small className="muted mono v6-at-ellipsis" title={j.result.report}>报告：{j.result.report}</small>}
              {openLog === j.id && <JobLog job={j} />}
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
