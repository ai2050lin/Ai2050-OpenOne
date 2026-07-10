import { AlertTriangle, Cpu, Database, ShieldCheck } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const PIPELINE = [
  '选择 Evidence Kernel 缺口',
  '登记可证伪假设',
  '冻结 ExperimentSpec',
  '方法与对照审核',
  '生成受约束脚本',
  'AST 与目录安全检查',
  'Smoke test',
  '三模型 GPU 串行执行',
  '原始产物与哈希审计',
  '反例和副作用审核',
  'accept / reject / inconclusive 裁决',
  '发布主张、缺口、图谱与备忘录',
];

export function AIRnDOrchestratorTab() {
  const [status, setStatus] = useState(null);
  const [selectedRun, setSelectedRun] = useState('');
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/ai-rnd/orchestrator/status`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setStatus(await response.json());
      setError('');
    } catch (reason) {
      setError(reason?.message || '编排器状态读取失败');
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  const selected = status?.recent_runs?.find((run) => run.run_id === selectedRun);
  const evidence = status?.evidence_context;

  return (
    <div style={{ flex: 1, overflow: 'auto', padding: 28, color: '#e5e7eb' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, marginBottom: 18 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 20 }}>证据驱动研究编排器</h2>
          <div style={{ color: '#9ca3af', fontSize: 12, marginTop: 5 }}>AI 意见只参与审核；只有校验后的实验产物可以改变机制主张。</div>
        </div>
        <button type="button" onClick={refresh} style={{ border: '1px solid rgba(167,139,250,0.3)', background: 'rgba(99,102,241,0.12)', color: '#c4b5fd', borderRadius: 6, padding: '7px 12px', cursor: 'pointer' }}>
          刷新状态
        </button>
      </div>

      {error && <div style={{ color: '#fb7185', fontSize: 12, marginBottom: 12 }}>{error}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 10, marginBottom: 18 }}>
        <StatusBox icon={Database} title="Evidence Kernel" value={evidence?.available ? `${evidence.run_count} runs` : '未连接'} detail={evidence?.source || evidence?.reason} />
        <StatusBox icon={Cpu} title="GPU 策略" value={status?.gpu_policy || 'single_process_serial'} detail={(status?.model_order || []).join(' -> ')} />
        <StatusBox icon={ShieldCheck} title="当前运行" value={status?.active_run_id || '无'} detail="运行状态持久化到 tests/result/auto_rnd" />
      </div>

      <section style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 16, marginBottom: 20 }}>
        <h3 style={{ fontSize: 14, margin: '0 0 10px' }}>十二阶段证据循环</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 6 }}>
          {PIPELINE.map((stage, index) => (
            <div key={stage} style={{ display: 'flex', gap: 8, alignItems: 'center', padding: 8, border: '1px solid rgba(255,255,255,0.07)', borderRadius: 5, background: 'rgba(17,24,39,0.5)', fontSize: 11 }}>
              <span style={{ color: '#a78bfa', fontFamily: 'monospace' }}>{String(index + 1).padStart(2, '0')}</span>
              {stage}
            </div>
          ))}
        </div>
      </section>

      <section style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 16, marginBottom: 20 }}>
        <h3 style={{ fontSize: 14, margin: '0 0 10px' }}>独立审核角色</h3>
        <div style={{ display: 'grid', gap: 7 }}>
          {Object.entries(status?.roles || {}).map(([role, description]) => (
            <div key={role} style={{ padding: 9, borderLeft: '2px solid #6366f1', background: 'rgba(17,24,39,0.45)' }}>
              <div style={{ color: '#c4b5fd', fontSize: 11, fontWeight: 700 }}>{role}</div>
              <div style={{ color: '#9ca3af', fontSize: 11, marginTop: 3 }}>{description}</div>
            </div>
          ))}
        </div>
      </section>

      <section style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 16 }}>
        <h3 style={{ fontSize: 14, margin: '0 0 10px' }}>最近运行与裁决</h3>
        <div style={{ display: 'grid', gap: 6 }}>
          {(status?.recent_runs || []).map((run) => {
            const open = selectedRun === run.run_id;
            return (
              <div key={run.run_id}>
                <button
                  type="button"
                  onClick={() => setSelectedRun(open ? '' : run.run_id)}
                  style={{ width: '100%', display: 'grid', gridTemplateColumns: '1fr auto', gap: 12, textAlign: 'left', padding: 9, border: '1px solid rgba(255,255,255,0.08)', background: open ? 'rgba(99,102,241,0.13)' : 'rgba(17,24,39,0.45)', borderRadius: 5, color: '#e5e7eb', cursor: 'pointer' }}
                >
                  <span style={{ overflowWrap: 'anywhere', fontSize: 11 }}>{run.run_id}</span>
                  <span style={{ color: run.decision === 'inconclusive' ? '#fbbf24' : '#86efac', fontSize: 10 }}>{run.decision}</span>
                </button>
                {open && selected && (
                  <div style={{ marginTop: 4, padding: 10, background: 'rgba(2,6,23,0.62)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 5, fontSize: 11 }}>
                    <div>目标：{selected.objective || '未登记'}</div>
                    <div style={{ color: '#9ca3af', marginTop: 5 }}>创建：{selected.created_at || '-'}</div>
                    <div style={{ color: selected.missing_required_artifacts?.length ? '#fda4af' : '#86efac', marginTop: 5 }}>
                      {selected.missing_required_artifacts?.length
                        ? `缺少产物：${selected.missing_required_artifacts.join(', ')}`
                        : '规定产物齐全或尚未完成审计'}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
          {!status?.recent_runs?.length && (
            <div style={{ display: 'flex', gap: 7, alignItems: 'center', color: '#9ca3af', fontSize: 12 }}>
              <AlertTriangle size={14} /> 尚无自动研发运行记录
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function StatusBox({ icon: Icon, title, value, detail }) {
  return (
    <div style={{ border: '1px solid rgba(255,255,255,0.09)', borderRadius: 6, padding: 12, background: 'rgba(17,24,39,0.52)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, color: '#9ca3af', fontSize: 10 }}><Icon size={14} />{title}</div>
      <div style={{ color: '#f3f4f6', fontSize: 12, marginTop: 8, overflowWrap: 'anywhere' }}>{value}</div>
      <div style={{ color: '#6b7280', fontSize: 10, marginTop: 5, overflowWrap: 'anywhere' }}>{detail || '-'}</div>
    </div>
  );
}
