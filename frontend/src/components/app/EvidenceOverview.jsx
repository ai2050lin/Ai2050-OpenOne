import { ArrowRight, BookOpen, Compass, FlaskConical, Map, Network } from 'lucide-react';

import { useResearchSnapshot } from '../../researchKernel/useResearchSnapshot';
import { ResearchEvidenceCockpit } from './ResearchEvidenceCockpit';
import { EvidenceBadge, EvidenceStamp, EVIDENCE_LEVELS } from './EvidenceBadge';
import { WorldNav } from './WorldNav';
import './EvidenceOverview.css';

/** main.jsx 已分发的 A 档证据路径（RdcAtlas 系列，registry 驱动）。 */
const EVIDENCE_ROUTES = [
  { path: '/rdc', label: 'RdcFeatureAtlas', desc: '特征图谱' },
  { path: '/rdc-prefix', label: 'RdcPrefixAtlas', desc: '前缀图谱' },
  { path: '/rdc-relation', label: 'RdcRelationStudy', desc: '关系研究' },
  { path: '/rdc-joint', label: 'RdcJointAtlas', desc: '联合图谱' },
  { path: '/rdc-operator', label: 'RdcOperatorAtlas', desc: '算子图谱' },
  { path: '/rdc-law', label: 'RdcLawAtlas', desc: '规律图谱' },
  { path: '/rdc-binding', label: 'RdcBindingAtlas', desc: '绑定图谱' },
  { path: '/rdc-update', label: 'RdcUpdateAtlas', desc: '更新图谱' },
  { path: '/rdc-query', label: 'RdcQueryAtlas', desc: '条件查询与有序来源图谱' },
  { path: '/rdc-construction', label: 'RdcConstructionAtlas', desc: '构造图谱' },
];

/**
 * 证据总览（新默认落地页）：
 * 打开客户端第一眼是 Canonical Snapshot 驱动的真实证据，而不是概念演示。
 * 数据源：frontend/public/research_data/current/snapshot.json（researchctl export-client 产物）。
 */
export function EvidenceOverview() {
  const { snapshot } = useResearchSnapshot();
  const counts = snapshot?.counts || {};
  const ia = snapshot?.information_architecture;

  return (
    <main className="evidence-overview" aria-label="证据总览">
      <WorldNav active="evidence" />
      <header className="evidence-overview__hero">
        <p className="evidence-overview__eyebrow">Mechanistic Interpretability · 机制可解释性公开学习平台</p>
        <h1>这里的每一条结论，都能回查到它是如何被证出来的。</h1>
        <p className="evidence-overview__sub">
          从原始激活数据到封存结论的全链路：Canonical Snapshot 单一事实源 · 四级证据标尺 · 预注册合同纪律。
        </p>
        <EvidenceStamp />
      </header>

      <section className="evidence-overview__cockpit" aria-label="当前证据状态">
        <ResearchEvidenceCockpit />
      </section>

      <section className="evidence-overview__routes" aria-label="证据图谱入口">
        <h2><Map size={15} /> 证据图谱（Registry 驱动）</h2>
        <div className="evidence-overview__grid">
          {EVIDENCE_ROUTES.map((route) => (
            <a key={route.path} className="evidence-overview__card" href={route.path}>
              <Network size={14} />
              <div>
                <strong>{route.label}</strong>
                <span>{route.desc}</span>
              </div>
              <ArrowRight size={13} />
            </a>
          ))}
        </div>
      </section>

      <section className="evidence-overview__meta" aria-label="快照统计">
        <h2><FlaskConical size={15} /> 本版快照范围</h2>
        <div className="evidence-overview__counts">
          {Object.entries(counts).length === 0 && <span className="evidence-overview__dim">快照未提供 counts。</span>}
          {Object.entries(counts).map(([key, value]) => (
            <span key={key} className="evidence-overview__count"><b>{key}</b>{String(value)}</span>
          ))}
        </div>
        {ia ? (
          <p className="evidence-overview__dim">{typeof ia === 'string' ? ia : JSON.stringify(ia).slice(0, 200)}</p>
        ) : null}
      </section>

      <section className="evidence-overview__legend" aria-label="证据标尺图例">
        <h2><Compass size={15} /> 四级证据标尺</h2>
        <div className="evidence-overview__levels">
          {Object.entries(EVIDENCE_LEVELS).map(([level, meta]) => (
            <EvidenceBadge key={level} level={level} title={`level: ${level}`} />
          ))}
        </div>
        <p className="evidence-overview__dim">
          视图分档审计（2026-10-03）：357 个组件中 A 档 10 / B 档 69 / C 档 175（概念示意，已隔离至
          <a href="/gallery"> 概念画廊</a>）/ D 档 103。
        </p>
      </section>

      <footer className="evidence-overview__footer">
        <a href="/app"><BookOpen size={14} /> 打开完整研究工作台（含 logit lens / glass matrix 等分析视图）</a>
        <a href="/rdc-query"><Network size={14} /> RDC 条件查询与有序来源图谱</a>
      </footer>
    </main>
  );
}

export default EvidenceOverview;
