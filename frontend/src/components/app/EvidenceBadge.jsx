import { ShieldCheck } from 'lucide-react';

import { useResearchSnapshot } from '../../researchKernel/useResearchSnapshot';
import './EvidenceBadge.css';

/**
 * 四级证据标尺（对齐 AGENTS.md §10.1 与 registry 证据纪律）：
 *   has_data              有数据（仅落盘，未形成观察）
 *   observed              已有观察（自然前向/干预观察成立）
 *   generalization_checked通过推广检查（未见样本/跨条件复现）
 *   mechanism_evidence    有机制证据（因果/参数级闭环）
 */
export const EVIDENCE_LEVELS = {
  has_data: { label: '有数据', color: '#64748b', bg: '#f1f5f9', border: '#cbd5e1' },
  observed: { label: '已有观察', color: '#1d4ed8', bg: '#eff6ff', border: '#93c5fd' },
  generalization_checked: { label: '通过推广检查', color: '#047857', bg: '#ecfdf5', border: '#6ee7b7' },
  mechanism_evidence: { label: '有机制证据', color: '#6d28d9', bg: '#f5f3ff', border: '#c4b5fd' },
};

/**
 * 单图证据徽章：任何进入主界面的可视化都必须挂。
 * 未传 level 时渲染为"无证据"警示态（灰色斜纹），提示该图未挂证据卡。
 */
export function EvidenceBadge({ level, runId, sha8, n, title }) {
  const meta = EVIDENCE_LEVELS[level];
  if (!meta) {
    return (
      <span className="evidence-badge evidence-badge--missing" title="该视图未挂证据卡：不得作为研究事实引用">
        <ShieldCheck size={11} />
        无证据卡
      </span>
    );
  }
  const tip = [
    title || null,
    `证据等级：${meta.label}（${level}）`,
    runId ? `run: ${runId}` : null,
    sha8 ? `产物 sha8: ${sha8}` : null,
    n != null ? `样本量: ${n}` : null,
  ].filter(Boolean).join('\n');
  return (
    <span
      className="evidence-badge"
      style={{ color: meta.color, background: meta.bg, borderColor: meta.border }}
      title={tip}
    >
      <ShieldCheck size={11} />
      {meta.label}
    </span>
  );
}

/**
 * 全局快照戳：读取 Canonical Snapshot 的身份元数据（snapshot_id / as_of / sha8），
 * 用于证据总览页头，标明"本页所有数字的可回查版本"。
 */
export function EvidenceStamp() {
  const { snapshot, error } = useResearchSnapshot();
  const sha8 = snapshot?.source_sha256 ? String(snapshot.source_sha256).slice(0, 8) : null;
  if (error) {
    return <span className="evidence-stamp evidence-stamp--error">Canonical Snapshot 读取失败：{error}</span>;
  }
  return (
    <span className="evidence-stamp" title={`source_sha256=${snapshot?.source_sha256 || ''}`}>
      <ShieldCheck size={12} />
      {snapshot?.snapshot_id || 'snapshot 加载中…'}
      {sha8 ? <code>sha8:{sha8}</code> : null}
      {snapshot?.as_of ? <time>{snapshot.as_of}</time> : null}
    </span>
  );
}

export default EvidenceBadge;
