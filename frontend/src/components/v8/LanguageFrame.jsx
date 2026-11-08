/* 空间 · 语言图谱：外部能力框架。只描述语言本身（词、语言项、关系与材料覆盖），不含任何模型信息。
   三个视图：结构图谱（领域 → 子领域 → 语言项 → 词，lang_map.v1，Phase 341）、词汇关系环图、关系种子。
   原“语言目录”69 项已全部并入结构图谱（61 项进树，8 项转为横切维度）。 */
import { useState } from 'react';

import LangGraph3D from './LangGraph3D.jsx';
import LangMap, { LangMapDims } from './LangMap.jsx';
import WordRing from './WordRing.jsx';
import { PageHead, PageTabs, PillTools } from './CoreKit.jsx';

const VIEWS = [['map', '结构图谱', '知识网络、推理功能、语法规则三个领域，逐层到子领域、语言项和词'], ['ring', '词汇关系环图', '大量词按类别排成环，看它们之间的层级与属性关系'], ['seed', '关系种子', '语言项之间的人工关系（3D）']];

export default function LanguageFrame() {
  const [view, setView] = useState('map');
  /* 首屏只有当前视图（默认结构图谱，Phase 341）；说明文字收进按钮（Phase 333） */
  return <div className="v8-route" aria-label="语言图谱">
    <PageHead title="语言图谱" sub="外部能力框架：语言里有哪些词和语言项、它们之间是什么关系、哪些已经有测试材料。这一页只描述语言本身，不含模型信息。" />
    <PageTabs label="语言图谱视图" items={VIEWS} value={view} onChange={setView} />
    {view === 'ring' && <WordRing/>}
    {view === 'seed' && <LangGraph3D/>}
    {view === 'map' && <LangMap/>}
    <PillTools label="语言图谱说明" items={[
      ...(view === 'map' ? [{ id: 'dims', label: '横切维度 · 8（语种、语码混用、方言、语域、修辞、表述变换……作为节点标签，不单独成域）', render: () => <LangMapDims/> }] : []),{ id: 'about', label: '这一页是什么', render: () => <>
      <p>语言里有上百万种物体的颜色、大小、用途等知识，也有复杂的推理和语法。模型能表现出这些能力，内部就必然有对应的编码机制。
        语言图谱先把这些能力整理成一个外部框架：有哪些词和语言项、它们之间是什么关系、哪些已经有测试材料。</p>
      <p className="v8-note">这一页只描述语言本身。每一项能力在模型里怎样编码，在“关联机制”里破解。</p>
    </> }]} />
  </div>;
}
