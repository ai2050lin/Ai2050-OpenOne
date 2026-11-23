import { useEffect, useRef, useState } from 'react';
import { getSession } from '../app/rdc_fusion/api.js';
import { useAuth } from '../app/rdc_fusion/authContext.js';
import { useWorkspaceResource } from './nativeObservationArchive.js';

const STAGES = {
  checking: '正在正式查重',
  preflight: '已保存正式预检回执',
  registered: '已在正式中心登记，正在采集或归档',
  duplicate: '发现重复或可复用采集',
  discovery_archived: '发现采集已归档',
  prediction_checking: '正在对留出验证查重',
  prediction_preflight: '已保存留出验证回执',
  prediction_registered: '留出验证已登记，正在运行',
  prediction_archived: '留出验证已归档',
  causal_checking: '正在对匹配干预查重',
  causal_preflight: '已保存匹配干预回执',
  causal_registered: '匹配干预已登记，正在运行',
  causal_archived: '匹配干预已归档',
  archive_retry: '正在恢复原运行的归档',
  incomplete: '发现采集或归档待完成',
  failed: '归档流程已停止',
  unavailable: '本机归档入口不可用',
};
const FOLLOW_KEY = 'ai2050-first-record-follow-run';

export default function FirstRecordArchive() {
  const verifiedResult = useWorkspaceResource('first-record').data || {};
  const { user } = useAuth();
  const archive = window.ai2050Desktop?.firstRecordArchive;
  const [status, setStatus] = useState(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');
  const [followRun, setFollowRun] = useState(() => {
    try { return sessionStorage.getItem(FOLLOW_KEY) || ''; } catch { return ''; }
  });
  const followed = useRef('');
  const followKey = status?.runId && user?.user_id ? `${status.runId}:${user.user_id}` : '';

  useEffect(() => {
    if (!archive) return undefined;
    let mounted = true;
    const refresh = () => archive.status().then(value => {
      if (mounted) setStatus(value);
    }).catch(() => {
      if (mounted) setStatus({ stage: 'unavailable', message: '无法读取本机归档状态。' });
    });
    void refresh();
    const timer = setInterval(refresh, 2500);
    return () => { mounted = false; clearInterval(timer); };
  }, [archive]);

  useEffect(() => {
    if (!archive || status?.working || status?.stage !== 'discovery_archived'
        || !followKey || followRun !== followKey || followed.current === followKey || user?.role !== 'admin') return;
    const session = getSession();
    if (!session) return;
    followed.current = followKey;
    try { sessionStorage.removeItem(FOLLOW_KEY); } catch { /* keep the in-memory state */ }
    void archive.start(session).then(response => {
      setFollowRun('');
      if (!response.accepted) setError(response.message);
      else setStatus({ stage: 'prediction_checking', working: true });
    }).catch(() => {
      setFollowRun('');
      setError('自动启动留出验证失败。');
    });
  }, [archive, followKey, followRun, status, user?.role]);

  async function start() {
    setStarting(true);
    setError('');
    try {
      const session = getSession();
      if (!session || user?.role !== 'admin') throw new Error('请使用正式中心管理员账户登录。');
      const response = await archive.start(session);
      if (!response.accepted) throw new Error(response.message);
      setStatus({ stage: 'checking', working: true });
    } catch (reason) {
      setError(reason.message);
    } finally {
      setStarting(false);
    }
  }

  function queuePrediction() {
    setFollowRun(followKey);
    try { sessionStorage.setItem(FOLLOW_KEY, followKey); } catch { /* optional persistence */ }
  }

  const included = verifiedResult.runId && [status?.runId, status?.predictionRunId].includes(verifiedResult.runId) ? verifiedResult : {};
  const archived = status?.archiveState === 'server_verified' && status.conclusion
    ? { ...included, ...status, conclusion: { ...included.conclusion, ...status.conclusion } }
    : verifiedResult;
  const report = archived.conclusion;
  const readout = status?.readout || report?.readout;
  const prediction = status?.prediction || report?.prediction;
  const causal = report?.causal;
  const progress = status?.stage !== 'ready' ? status : null;
  const percent = value => (value * 100).toFixed(1) + '%';
  const logit = value => value.toFixed(2);
  const metrics = [['source_log_mass', '记录注意力质量'], ['vo_write', 'O 写入'], ['mlp_units', 'MLP 激活']];
  const predictorText = prediction
    ? `五记录预测一致率 ${percent(prediction.scores.selected.agreement)}，${prediction.gate_passed ? '通过' : '未通过'}预注册门槛。`
    : report?.confirm ? '确认材料未通过行为或复算检查，无法判定预测门。'
      : '校准未通过，确认集未运行。';

  return <details className="cue-details">
    <summary>首行并行路径实验 · {progress?.preflightId || archived.preflightId ? '归档与运行状态' : '未取得中心预检回执'}</summary>
    {archive && user?.role === 'admin' && <p>使用当前管理员会话查重。正式登记通过后，在本机运行已冻结的实验。</p>}
    {archive && status?.stage !== 'unavailable' && user?.role === 'admin' &&
      <button type="button" className="fw-tbtn primary" disabled={starting || status?.working || ['prediction_archived', 'causal_archived'].includes(status?.stage)} onClick={start}>
        {status?.stage === 'discovery_archived' ? '继续留出验证' : '查重、登记并运行首行实验'}
      </button>}
    {archive && user?.role === 'admin' && status?.working && status.stage === 'registered' && followKey &&
      <button type="button" className="fw-tbtn" disabled={followRun === followKey} onClick={queuePrediction}>
        {followRun === followKey ? '已安排自动继续' : '归档核验后自动继续'}
      </button>}
    {progress && <p role="status">
      {STAGES[progress.stage] || progress.stage}。{progress.message}
      {progress.preflightId && <> 预检 <code>{progress.preflightId}</code>，匹配 {progress.matchCount} 项。</>}
      {(progress.runId || progress.matchedRunId) && <> 运行 <code>{progress.runId || progress.matchedRunId}</code>。</>}
    </p>}
    {(report || readout || prediction) && <>
      {report && <p>Qwen3-14B BF16 · B{report.block}/H{report.head} · 中心已读回核验</p>}
      <ul>
        <li><strong>可读出</strong> {readout ? `正确记录的头内首选率：姓名 ${percent(readout[0])}、印刷标签 ${percent(readout[1])}、物理首行 ${percent(readout[2])}。` : '暂无可登记的确认结果。'}</li>
        <li><strong>可预测</strong> {predictorText}</li>
        <li><strong>因果作用</strong> {causal
          ? `等剂量匹配率 ${percent(causal.dose_match)}，目标与随机干预造成的答案优势降幅相差 ${logit(causal.margin_drop_over_random)} logit，${causal.causal_gate_passed ? '达到' : '未达到'}因果验收条件。`
          : (prediction?.gate_passed ? '等待匹配干预核验。' : '未执行匹配干预。')}</li>
      </ul>
      <details>
        <summary>核对口径与正式回执</summary>
        <p>预测给定同层其他头与原残差。留出门同时要求五记录一致率至少 90%，并且选源、O 写入与 MLP 的误差较两类对照各下降至少 20%，按家族计算的置信下界为正。</p>
        {prediction && <p>同预算随机列的五记录一致率为 {percent(prediction.scores.random.agreement)}。一致率衡量首选记录，下表衡量数值复算。</p>}
        {prediction && <div className="cue-table-wrap"><table>
          <caption>原生坐标预测较对照的误差下降</caption>
          <thead><tr><th scope="col">指标</th><th scope="col">同预算随机列</th><th scope="col">发现集均值</th></tr></thead>
          <tbody>{metrics.map(([key, label]) => <tr key={key}>
            <th scope="row">{label}</th>
            {['random', 'mean'].map(control => <td key={control}>{percent(prediction.contrasts[`${control}/${key}`].gain)}</td>)}
          </tr>)}</tbody>
        </table></div>}
        {causal && <p>答案优势降幅差的 97.5% 家族区间为 [{causal.interval.map(logit).join(', ')}]。同输入头恢复 {causal.exact_sham_and_restore.restore ? '逐位复现完整 logits' : '未逐位复现完整 logits'}。单独恢复所选 MLP 单元的平均答案优势变化为 {logit(causal.separate_mlp_restore_margin_gain)} logit。</p>}
        {report?.sample_native_scalars && <p>原生输入列示例 Q {report.sample_native_scalars.q.col}、K {report.sample_native_scalars.k.col}、V {report.sample_native_scalars.v.col}。MLP 单元示例 {report.sample_native_scalars.mlp_unit}。这些是预测器的索引。</p>}
        {archived.preflightId && <p>正式预检 <code>{archived.preflightId}</code> · 运行 <code>{archived.runId}</code></p>}
        {archived.predictionRunId && <p>留出验证运行 <code>{archived.predictionRunId}</code></p>}
        {archived.discovery && <p>发现采集运行 <code>{archived.discovery.runId}</code></p>}
      </details>
    </>}
    {error && <p role="alert">{error}</p>}
    <p>发现采集用于确定候选。可读出、留出预测与因果干预按核验结果登记。</p>
  </details>;
}
