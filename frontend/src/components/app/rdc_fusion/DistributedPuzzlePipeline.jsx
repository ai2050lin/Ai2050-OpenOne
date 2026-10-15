import { useEffect, useState } from 'react';
import useCookiePreference from './useCookiePreference.js';

const STATUS = {
  queued: ['等待领取', 'queued'], claimed: ['租约中', 'claimed'], done: ['结果已上传', 'done'],
  failed: ['执行失败', 'failed'], expired: ['租约过期', 'expired'],
};

function Stage({ number, label, value, note, tone = '' }) {
  return (
    <div className={'at-pipeline-stage ' + tone}>
      <span className="at-pipeline-index">{number}</span>
      <div><b>{label}</b><strong>{value}</strong><small>{note}</small></div>
    </div>
  );
}

export default function DistributedPuzzlePipeline({ onShowResults, onGo, onData, onError }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [display, setDisplay] = useCookiePreference('rdc-route-pipeline-v1', { expanded: false, detailsOpen: false });
  const expanded = display?.expanded === true;
  const detailsOpen = display?.detailsOpen === true;
  const setExpanded = value => setDisplay(previous => ({ ...previous, expanded: value }));
  const setDetailsOpen = value => setDisplay(previous => ({ ...previous, detailsOpen: value }));

  useEffect(() => {
    let dead = false;
    const controller = new AbortController();
    const load = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/research/work-units`, { signal: controller.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const value = await response.json();
        if (!dead && value.schema_version === 'research_distributed_view.v1') {
          setData(value); setError(''); onData?.(value); onError?.('');
        } else if (!dead) throw new Error('服务端未返回 research_distributed_view.v1');
      } catch (reason) {
        if (!dead && reason.name !== 'AbortError') {
          const message = String(reason.message || reason);
          setError(message); onError?.(message);
        }
      }
    };
    load();
    const timer = window.setInterval(load, 30000);
    return () => { dead = true; controller.abort(); window.clearInterval(timer); };
  }, [onData, onError]);

  const counts = data?.counts || {};
  const units = data?.work_units || [];
  const puzzles = Object.fromEntries((data?.puzzles || []).map(p => [p.puzzle_id, p]));
  const queued = counts.by_status?.queued || 0;
  const claimed = counts.by_status?.claimed || 0;
  const terminal = (counts.by_status?.done || 0) + (counts.by_status?.failed || 0) + (counts.by_status?.expired || 0);

  return (
    <section className="at-pipeline" aria-label="分布式拼图执行链">
      <div className="at-pipeline-head">
        <div>
          <span className="at-eyebrow">DISTRIBUTED EVIDENCE PATH</span>
          <h4>拼图 → 合同 → Work Unit → Worker → 产物审计</h4>
          {expanded && <p>任务身份从 Research OS 合同读取；节点回传只形成待审计产物，不直接成为 Evidence。</p>}
        </div>
        <div className="at-pipeline-head-actions">
          <div className={'at-pipeline-source ' + (data ? 'live' : 'idle')}>
            {data ? '执行视图 LIVE' : error ? '执行视图未连接' : '读取执行视图…'}
          </div>
          <button type="button" className="at-pipeline-toggle" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>{expanded ? '收起执行详情' : '展开执行详情'}<span>{expanded ? '−' : '+'}</span></button>
        </div>
      </div>

      {error && <div className="at-pipeline-empty">分布式图谱接口不可用（{error}）。当前不显示演示任务。</div>}
      {data && !expanded && <div className="at-pipeline-compact-summary">
        <span><b>{counts.work_units ?? '—'}</b><small>Work Unit</small></span>
        <span><b>{counts.linked_results ?? '—'}</b><small>结果回传</small></span>
        <span><b>{counts.awaiting_audit ?? '—'}</b><small>等待审计</small></span>
        <span><b>{terminal}</b><small>已结束任务</small></span>
      </div>}
      {data && expanded && (
        <>
          <div className="at-pipeline-flow">
            <Stage number="01" label="研究 Puzzle" value={counts.puzzles ?? '—'} note="Registry 事实" />
            <span className="at-pipeline-arrow">→</span>
            <Stage number="02" label="可运行合同" value={counts.runnable_contracts ?? '—'} note="冻结、run_ready、哈希有效" tone={counts.runnable_contracts ? 'ready' : 'blocked'} />
            <span className="at-pipeline-arrow">→</span>
            <Stage number="03" label="分布式切片" value={counts.work_units ?? '—'} note={`${queued} 等待 · ${claimed} 租约中`} />
            <span className="at-pipeline-arrow">→</span>
            <Stage number="04" label="回传产物" value={counts.linked_results ?? '—'} note="文件哈希已入结果库" />
            <span className="at-pipeline-arrow">→</span>
            <Stage number="05" label="独立审计" value={counts.awaiting_audit ?? '—'} note="审计前不升证据" tone={counts.awaiting_audit ? 'review' : ''} />
          </div>

          <div className="at-pipeline-foot">
            <span>仅绑定 Work Unit 的任务进入这条证据路径</span>
            <span>未绑定的模板任务 {counts.unlinked_platform_tasks ?? 0} 条，保留在平台结果库，不混入 Puzzle 证据</span>
            {(counts.invalid_runnable_contracts || 0) > 0 && <span>无效合同 {counts.invalid_runnable_contracts} 条，未计入可运行数</span>}
            {onGo && <button type="button" className="at-link" onClick={() => onGo('process')}>节点与租约 →</button>}
          </div>

          <div className="at-pipeline-details">
            <button type="button" className="at-pipeline-details-toggle" aria-expanded={detailsOpen} onClick={() => setDetailsOpen(!detailsOpen)}>
              {detailsOpen ? '收起逐条任务来源' : '查看任务拼图与逐条来源'} · {counts.work_units ?? units.length} 个已绑定 Work Unit
            </button>
            {detailsOpen && (units.length === 0 ? (
              <div className="at-pipeline-empty">
                {(counts.runnable_contracts || 0) === 0
                  ? '目前没有通过冻结状态、run_ready 与合同哈希核验的可运行合同。现有 TM 模板任务不会自动归到 Puzzle；需先准备可运行合同，再注册哈希绑定的 research_adapter.v1。'
                  : `目前有 ${counts.runnable_contracts} 个可运行合同，但还没有已绑定的分布式 Work Unit。现有 TM 模板任务不会自动归到 Puzzle；需注册哈希绑定的 research_adapter.v1 后再排队。`}
              </div>
            ) : (
              <div className="at-pipeline-table-wrap">
                <table className="at-pipeline-table">
                  <thead><tr><th>Puzzle / 合同</th><th>分区与采集</th><th>模型 / 节点</th><th>执行状态</th><th>结果 / 审计</th></tr></thead>
                  <tbody>
                    {units.slice(0, 30).map(unit => {
                      const status = STATUS[unit.task_status] || [unit.task_status || '未知', ''];
                      const puzzle = puzzles[unit.puzzle_id];
                      return (
                        <tr key={unit.work_unit_id}>
                          <td>
                            <b>{unit.puzzle_id} · {puzzle?.name || 'Puzzle'}</b>
                            <small className="at-pipeline-target">语言图谱目标：{unit.language_graph_target_id || unit.graph_target_id || '未绑定'}</small>
                            <small className="mono">{unit.contract_id} · {String(unit.contract_sha256 || '').slice(0, 12)}</small>
                            <small className="mono">metrics {String(unit.metric_version || '').replace('sha256:', '').slice(0, 12)}</small>
                            <small className="mono">{unit.work_unit_id}</small>
                          </td>
                          <td>{unit.case_family_id}<small>{unit.split_id} · {unit.capture_tier} · seed {unit.seed}</small></td>
                          <td>{unit.model_id}<small>{unit.model_snapshot_id}</small><small>{unit.precision} · tokenizer {String(unit.tokenizer_sha256 || '').slice(0, 12)}</small><small>{unit.node_name || '尚未分配节点'}</small></td>
                          <td><span className={'at-pipeline-status ' + status[1]}>{status[0]}</span><small>{unit.result_kind || '等待执行'}</small></td>
                          <td>
                            {unit.result_sha ? <><small className="mono">sha256:{unit.result_sha.slice(0, 16)}</small><small>{unit.artifacts?.length || 0} 个 Artifact · {unit.evidence_state}</small>
                              {onShowResults && <button type="button" className="at-link" onClick={onShowResults}>运行产物 →</button>}</>
                              : <small>尚无结果回执</small>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ))}
            {detailsOpen && <div className="at-pipeline-audit-note">执行结束 ≠ Run 已裁决。独立审计、manifest 校验和 Canonical Registry 导入仍是证据晋级门。</div>}
          </div>
          {terminal > 0 && <span className="at-pipeline-terminal">已结束任务 {terminal}</span>}
        </>
      )}
    </section>
  );
}
