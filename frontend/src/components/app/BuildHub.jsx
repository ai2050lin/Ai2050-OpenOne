import { useEffect, useState } from 'react';
import { AlertTriangle, Terminal, Wrench } from 'lucide-react';

import { WorldNav } from './WorldNav';
import './BuildHub.css';

/**
 * 研发世界（M2）：
 * 数据源 = registry/industry.json 投影（工具对照 + 空白雷达）。
 * 研发闭环与复现指引为静态说明（对齐 AGENTS.md §8.2 与 ai2050_research_os 纪律），
 * 队列/合同/复核状态的数据化接线属 M3（不造假数据）。
 */
const DATA_INDUSTRY = '/research_data/current/industry.json';

const RELATION_LABELS = {
  in_house: '自产装置（差异化资产）',
  reusable: '可直接复用',
  reference: '参照/借鉴',
  contrast: '对照位（差异化定位）',
};

const RELATION_ORDER = ['in_house', 'reusable', 'reference', 'contrast'];

const CHAIN_STEPS = [
  ['提案', '队列登记 proposed 条目，绑定预算与授权边界（atlas/phase_queue）'],
  ['冻结合同', '预注册 design 冻结于任何观测之前（sha 锚定，如 Q06 ebf960cf）'],
  ['执行', 'SMOKE 必过门 → 正式测量 → 资源守卫（GPU 串行逐模型）'],
  ['独立复核', '独立进程 re-hash + 重算校验 + 数字必看（47/0、42/0 惯例）'],
  ['封存回流', 'closeout 五写 → 队列 sealed → 派生教学案例 → 空白雷达接回新提案'],
];

const REPRO_STEPS = [
  ['1', '克隆仓库并安装依赖', 'pip install -e .（或按 pyproject.toml）；模型权重放本地 models/，不入 git'],
  ['2', '校验研究账本', 'python ai2050_research_os/scripts/researchctl.py validate'],
  ['3', '构建并导出快照', 'researchctl.py build-snapshot → validate-snapshot → export-client'],
  ['4', '重跑实验装置', 'tests/deepseek/ 下按冻结合同（q03/q04/q05/q06_prereg_*.json）重跑脚本'],
  ['5', '对照封存数字', '与 research/deepseek/docs/AGI_DEEPSEEK_MEMO.md 的封存记录逐项核对（drift 应为 0）'],
];

function useIndustry() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch(DATA_INDUSTRY)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j) => { if (alive) setData(j); })
      .catch((e) => { if (alive) setError(String(e)); });
    return () => { alive = false; };
  }, []);
  return { data, error };
}

export function BuildHub() {
  const { data, error } = useIndustry();
  const records = Array.isArray(data) ? data : [];
  const tools = records.filter((r) => r.kind === 'tool');
  const gaps = records.filter((r) => r.kind === 'gap');
  const gapsByPriority = ['high', 'medium', 'low']
    .map((p) => ({ priority: p, items: gaps.filter((g) => g.priority === p) }))
    .filter((g) => g.items.length > 0);

  return (
    <main className="build-hub" aria-label="研发世界">
      <WorldNav active="build" />

      <header className="build-hub__hero">
        <p className="build-hub__eyebrow"><Wrench size={13} /> 研发世界 · Build</p>
        <h1>研发闭环在平台上显性运行：预注册 → SMOKE → 正式 → 复核 → 封存回流。</h1>
        <p className="build-hub__sub">
          每一步都有可检查的产物：合同哈希、SMOKE 门、独立复核计数、快照 id。
          这里同时登记行业工具对照与研究空白雷达——空白直接接回研发队列。
        </p>
      </header>

      {error && (
        <div className="build-hub__warn" role="alert">
          <AlertTriangle size={14} />
          行业数据投影加载失败（{error}）。请先运行
          <code>researchctl.py export-client</code>
        </div>
      )}

      <section aria-label="AI 自动研发闭环">
        <h2><Terminal size={15} /> AI 自动研发闭环（五阶段）</h2>
        <ol className="build-hub__chain">
          {CHAIN_STEPS.map(([name, desc], i) => (
            <li key={name}>
              <b>{i + 1}. {name}</b>
              <span>{desc}</span>
            </li>
          ))}
        </ol>
        <div className="build-hub__cmd">
          <code>python ai2050_research_os/scripts/researchctl.py validate</code>
          <code>… build-snapshot → validate-snapshot → export-client</code>
          <code>… build-snapshot-v2 → validate-snapshot-v2 → drift-audit</code>
        </div>
      </section>

      <section aria-label="行业工具对照">
        <h2>行业工具对照（{tools.length} 项）</h2>
        {RELATION_ORDER.map((rel) => {
          const group = tools.filter((t) => t.relation_to_project === rel);
          if (group.length === 0) return null;
          return (
            <div key={rel} className="build-hub__tool-group">
              <h3>{RELATION_LABELS[rel] || rel}（{group.length}）</h3>
              <div className="build-hub__tools">
                {group.map((t) => (
                  <article key={t.id} className="build-hub__tool">
                    <header>
                      <strong>{t.name || t.title}</strong>
                      <span>{t.org}</span>
                    </header>
                    <p>{t.role}</p>
                    {t.project_note && <p className="build-hub__tool-note">{t.project_note}</p>}
                  </article>
                ))}
              </div>
            </div>
          );
        })}
      </section>

      <section aria-label="研究空白雷达">
        <h2>研究空白雷达（{gaps.length} 条）</h2>
        <p className="build-hub__section-note">
          空白 = 行业没做、本平台可差异化的位置；每条登记优先级与候选队列接续。
        </p>
        <div className="build-hub__gaps">
          {gapsByPriority.map(({ priority, items }) => (
            items.map((g) => (
              <article key={g.id} className={`build-hub__gap build-hub__gap--${priority}`}>
                <header>
                  <span className="build-hub__gap-id">{g.id}</span>
                  <strong>{g.title}</strong>
                  <span className={`build-hub__prio build-hub__prio--${priority}`}>
                    {priority === 'high' ? '高优先' : priority === 'medium' ? '中优先' : '低优先'}
                  </span>
                </header>
                <p><b>最近行业工作：</b>{g.nearest_industry_work}</p>
                <p><b>缺失：</b>{g.missing}</p>
                {g.candidate_queue_ref && (
                  <p className="build-hub__queue-ref"><b>队列接续：</b>{g.candidate_queue_ref}</p>
                )}
              </article>
            ))
          ))}
        </div>
      </section>

      <section aria-label="本地复现指引">
        <h2>本地复现指引（结果数据不入 git，脚本 + 冻结合同入库）</h2>
        <ol className="build-hub__repro">
          {REPRO_STEPS.map(([n, title, desc]) => (
            <li key={n}>
              <b>{title}</b>
              <span>{desc}</span>
            </li>
          ))}
        </ol>
        <p className="build-hub__dim">
          冻结合同（≤512KB JSON）随仓库发布；run bundle / npz 留在本地，按合同重跑后与 MEMO 封存数字逐项对照。
        </p>
      </section>
    </main>
  );
}

export default BuildHub;
