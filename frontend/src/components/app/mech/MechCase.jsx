/* 机制案例的共用页头与空态 */
import { Empty } from '../rdc_fusion/ui.jsx';
import { schemaLabel } from './mechUtil.js';

export function CaseMissing({ index }) {
  if (index.loading) return <Empty>读取机制案例…</Empty>;
  return (
    <Empty>
      未接入：{index.error ? `机制案例索引不在本机数据包里（${index.error}）。` : '还没有机制案例报告。'}
      案例由研究脚本写到 research/mechanism/&lt;案例&gt;/report.json（如 scripts/conflict_case.py → CF-v1-qwen3-4b），
      下一版数据包会带上；在那之前这里不显示任何数字。
    </Empty>
  );
}

export function CaseHeader({ cases, caseId, onCase, report }) {
  const cur = cases.find(c => c.id === caseId);
  return (
    <header className="wb9-casehd">
      <label>案例
        <select value={caseId} onChange={e => onCase(e.target.value)}>
          {cases.map(c => <option key={c.id} value={c.id}>{c.id} · {schemaLabel(c.schema)}</option>)}
        </select>
      </label>
      {cur && (
        <span className="wb9-casemeta">
          材料 {cur.set || '—'} · 模型 {String(cur.model || '—').split(/[\\/]/).pop()}
          {cur.model_snapshot ? ` · 快照 ${String(cur.model_snapshot).slice(0, 10)}` : ''}
          {report ? ` · 过行为门 ${report.n_gated ?? '—'} / ${report.n_facts ?? '—'}（发现 ${report.n_discovery ?? '—'}、留出 ${report.n_heldout ?? '—'}）` : ''}
          {' · '}<b title="证据等级在研究记录登记；本页不晋级">证据等级：候选稳定观察</b>
        </span>
      )}
    </header>
  );
}
