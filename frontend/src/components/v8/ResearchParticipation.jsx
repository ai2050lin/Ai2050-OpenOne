import { useEffect, useState } from 'react';

import { useAuth } from '../app/rdc_fusion/authContext.js';
import { fmtTime, nodeCall, TASK_KIND_LABEL, taskSlice, useNode } from '../app/rdc_fusion/contributeUtil.js';
import { downloadKit } from '../open/openUtil.js';
import ResearchDisclosure from './ResearchDisclosure.jsx';

const PREPARE = {
  service: '检查节点服务', runtime: '准备计算组件', account: '注册或绑定节点',
  device: '检查显卡', models: '准备模型', resources: '检查资源与任务设置',
};
const AUDIT = { passed: '审核通过', rejected: '审核驳回', needs_rerun: '需要复测' };
const RECEIPT = { accepted: '中心校验通过', disputed: '复算不一致，待审', rejected: '中心拒收',
  failed: '执行失败', expired: '任务已过期' };
const taskName = task => task.name || TASK_KIND_LABEL[task.kind] || task.tm_id || task.job || '研究任务';

function receiptLabel(receipt, synced) {
  if (receipt.audit_status) return AUDIT[receipt.audit_status] || '审核待更新';
  if (receipt.status === 'done') return synced ? '已上传，待审核' : '已上传，审核状态未同步';
  return RECEIPT[receipt.status] || '回执待更新';
}

export default function ResearchParticipation({ onGo, publicWork, publicSnapshot, publicLoaded, onTaskChanged }) {
  const { user, guest } = useAuth();
  const { bridge, sup, live, status, error: nodeError, setStatus } = useNode({ withLogs: false });
  const [participation, setParticipation] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const counts = status?.counts;
  const completed = counts ? counts.done + counts.rejected + counts.failed : 0;
  const taskId = status?.current?.task;

  useEffect(() => {
    if (taskId || completed) onTaskChanged();
  }, [taskId, completed, onTaskChanged]);

  useEffect(() => {
    setParticipation(null);
  }, [live, sup?.port, status?.bound]);

  useEffect(() => {
    if (!live) return undefined;
    let cancelled = false;
    let timer;
    const refresh = async () => {
      try {
        const data = await nodeCall('GET', '/participation');
        if (!cancelled) setParticipation(data);
      } catch {
        const history = await nodeCall('GET', '/history?limit=5').catch(() => null);
        if (!cancelled) setParticipation({ receipts: history?.items ?? [], center_error: '当前节点服务无法读取实时派发与审核状态。' });
      } finally {
        if (!cancelled) timer = setTimeout(refresh, 12000);
      }
    };
    refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [live, sup?.port, status?.bound, completed]);

  const perform = async action => {
    setBusy(true);
    setMessage('');
    try { await action(); }
    catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  };
  const prepare = step => onGo('config', null, null, { csec: 'node', cstep: step });
  const start = () => perform(async () => {
    const latest = await nodeCall('GET', '/participation');
    setParticipation(latest);
    if (latest.step) return prepare(latest.step);
    if (!latest.available) return;
    setStatus(await nodeCall('POST', status.state === 'paused' ? '/resume' : '/start'));
  });

  const current = taskId ? status.current : null;
  const progress = current?.progress;
  const measured = Number.isFinite(progress?.done) && Number.isFinite(progress.total) &&
    progress.total > 0 && progress.done >= 0 && progress.done <= progress.total;
  const active = ['running', 'waiting'].includes(status?.state);
  const receipts = participation?.receipts ?? [];
  const latestReceipt = receipts[0];
  let description;
  let action;
  let scope = null;
  if (!bridge) {
    description = publicWork ? `${publicSnapshot ? '数据包中的开放作业' : '开放作业'} · ${publicWork}`
      : publicLoaded ? '暂无公开排队任务' : '网页端 · 下载后在本机运行';
    action = <button type="button" className="fw-tbtn primary" disabled={busy} onClick={() => perform(async () => {
      await downloadKit(Boolean(user) && !guest);
      setMessage('参与工具已下载。解压后按包内说明启动。');
    })}>下载参与工具</button>;
  } else if (!sup || sup.state === 'starting' || (live && !status && !nodeError)) {
    description = '正在检查本机节点…';
  } else if (!live || nodeError) {
    description = '本机节点服务需要检查。';
    action = <button type="button" className="fw-tbtn primary" onClick={() => prepare('service')}>{PREPARE.service}</button>;
  } else if (current) {
    description = `${progress?.stage === 'uploading' ? '正在提交' : '正在运行'} · ${taskName(current)}`;
  } else if (participation?.step || !status.bound) {
    const step = participation?.step || 'account';
    description = participation?.reason || '请先注册本机节点，可绑定账号署名。';
    action = <button type="button" className="fw-tbtn primary" onClick={() => prepare(step)}>{PREPARE[step]}</button>;
  } else if (active) {
    description = status.last_error ? '节点正在等待中心连接恢复。' : status.reasons?.[0] || '本机已开启贡献，正在等待中心派发任务。';
  } else if (participation?.available) {
    const available = participation.available;
    description = ['可领取', taskName(available), available.shard].filter(Boolean).join(' · ');
    scope = available;
    action = <button type="button" className="fw-tbtn primary" disabled={busy} onClick={start}>
      {busy ? '正在启动…' : status.state === 'paused' ? '继续贡献算力' : '一键参与'}
    </button>;
  } else {
    description = participation?.center_error ? '暂时无法确认中心有哪些可领取的任务。' : !participation
      ? '正在核对可领取的任务…' : participation.reason || '当前没有与本机匹配的开放任务。';
  }

  return <section className="v8-rnd-participation" aria-label="参与研发">
    <div className="v8-rnd-participation-main">
      <h3>{current ? '我的任务' : '参与研发'}{status?.synthetic && <small> · 沙盒联调</small>}</h3>
      <div className="v8-rnd-claim">
        <p role="status">{description}</p>
        {action || (active && <button type="button" className="v6-linkbtn" onClick={() => prepare('')}>管理贡献</button>)}
      </div>
      {scope && <dl className="mp-join-scope" aria-label="开始前说明"><dt>任务</dt><dd>{taskName(scope)}{scope.shard ? ` · ${scope.shard}` : ''}</dd>
        <dt>所属</dt><dd>{taskSlice(scope)}</dd>
        <dt>资源</dt><dd>本机 GPU，按任务包限制；领取前再检查电量、显存和磁盘</dd>
        <dt>上传</dt><dd>统计结果与回执；原生全场留在本机</dd></dl>}
      {scope && <p className="v8-note">点击后才开始，可随时暂停或停止。</p>}
      {current && <div className="v8-rnd-task-progress">
        <small>{current.shard && `分片 ${current.shard} · `}{measured ? `${progress.done} / ${progress.total}`
          : progress?.stage === 'uploading' ? '结果上传中' : '等待节点上报计数'}</small>
        <progress aria-label="本机任务进度" value={measured ? progress.done : undefined} max={measured ? progress.total : undefined}/>
      </div>}
    </div>
    {message && <p role="status" className="v8-note">{message}</p>}
    {latestReceipt && <ResearchDisclosure title={`最近回执 · ${receiptLabel(latestReceipt, participation.center_current)}`}>
      <ol className="v8-rnd-receipts">
        {receipts.map(receipt => <li key={receipt.task}>
          <strong>{taskName(receipt)}</strong> · {receiptLabel(receipt, participation.center_current)}
          <small>{fmtTime(receipt.audited_at || receipt.t)} · {receipt.task}{receipt.shard ? ` · ${receipt.shard}` : ''}</small>
          {(receipt.audit_note || receipt.note) && <p>{receipt.audit_note || receipt.note}</p>}
          {receipt.sha && <small>结果校验值 {receipt.sha}</small>}
        </li>)}
      </ol>
    </ResearchDisclosure>}
  </section>;
}
