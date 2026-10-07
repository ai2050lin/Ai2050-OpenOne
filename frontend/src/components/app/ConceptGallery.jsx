import { ArrowLeft, ImageOff } from 'lucide-react';

import './ConceptGallery.css';

/**
 * 概念画廊（M1 归档骨架）：
 * 承接 2026-10-03 审计中 175 个 C 档"概念示意/演示数据"组件与自创理论叙事的隔离区。
 * M1 只立容器与名录；组件逐个挂接与回查原始数值属 M3。
 * 纪律依据：AGENTS.md §10.2——界面演示数据与真实实验数据必须明确区分。
 */
const GALLERY_SECTIONS = [
  {
    title: '3D 概念视觉件（无数据源）',
    note: '渲染自内置演示常量，不对应任何实验 run。视觉风格保留，证据卡缺失。',
    items: ['BrainVis3D', 'ResonanceField3D', 'TDAVisualization3D', 'HolonomyLoopVisualizer', 'LayerFirstNeuronScene', 'TrainingDynamics3D（概念模式）'],
  },
  {
    title: '自创理论叙事（与注册理论 RDC 并存，未入 Registry）',
    note: 'NFB 纤维丛、GUT 大统一智能、玻璃矩阵等叙事来自早期科普页；未按 candidate-theory 登记前不作为研究事实展示。',
    items: ['App.jsx: agi / gut_relationship / model_generation 等叙事 tab', 'FlowTubesVisualizer（叙事模式）', 'GlassMatrix3D（叙事模式）'],
  },
  {
    title: 'blueprint/ 概念 dashboard（recharts 演示数据）',
    note: '约 60+ 个 TwoLayerLaw / ConceptVector / GateLaw 类 dashboard，数据为内置常量。',
    items: ['AppleOrthogonalityDashboard', 'ConceptVectorAlgebraGraph', 'GateLawDynamicsDashboard', 'DProblemAtlasDashboard', 'FirstPrinciplesTheoryDashboard', '…（全量名录见审计文档）'],
  },
  {
    title: '旧叙事残骸（废弃项目阶段）',
    note: 'SNN / ICSPB / DNN 模块定位等 tab 与 AGI Central 系列来自其他阶段；已从主导航摘除。',
    items: ['App.jsx: snn_system / icspb_system / main_workspace / main_system tab', 'AGICentralCommand', 'HLAIBlueprint', 'GlobalTopologyDashboard', '输入面板 SNN / ICSPB 旧叙事页（FiberNetPanel / SNNResearchDashboard，M2 起主导航摘除）', '结构面板死分支 agi / glass_matrix / flow_tubes / global_topology（M2 起无导航入口，M3 拆分时清理）'],
  },
];

export function ConceptGallery() {
  return (
    <main className="concept-gallery" aria-label="概念画廊（非证据区）">
      <header className="concept-gallery__header">
        <a className="concept-gallery__back" href="/"><ArrowLeft size={14} /> 返回证据总览</a>
        <h1><ImageOff size={18} /> 概念画廊</h1>
        <p className="concept-gallery__banner">
          本区收录<strong>概念示意与演示数据</strong>组件。它们不来自实验 run、未挂证据卡，
          <strong>不得作为研究事实引用</strong>。分档依据：ai2050_research_os/docs/CLIENT_ASSET_AUDIT_2026-10-03.md（C 档 175 项）。
        </p>
      </header>

      {GALLERY_SECTIONS.map((section) => (
        <section key={section.title} className="concept-gallery__section">
          <h2>{section.title}</h2>
          <p className="concept-gallery__note">{section.note}</p>
          <ul className="concept-gallery__list">
            {section.items.map((item) => (
              <li key={item}>
                <span className="concept-gallery__tag">示意</span>
                {item}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </main>
  );
}

export default ConceptGallery;
