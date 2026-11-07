/* 空间 · 语言图谱：外部能力框架。只描述语言本身（词、语言项、关系与材料覆盖），不含任何模型信息。
   三个视图：词汇关系环图（大量词的相对关系）、关系种子（语言项之间的人工关系）、语言目录（69 项）。 */
import { useState } from 'react';

import LangGraph3D from './LangGraph3D.jsx';
import LanguageCatalog from './LanguageCatalog.jsx';
import WordRing from './WordRing.jsx';
import { PageHead, PillTools } from './CoreKit.jsx';

const VIEWS = [['ring', '词汇关系环图'], ['seed', '关系种子'], ['catalog', '语言目录']];

export default function LanguageFrame() {
  const [view, setView] = useState('ring');
  /* 首屏只有当前视图（默认词汇关系环图）；说明文字收进按钮（Phase 333） */
  return <div className="v8-route" aria-label="语言图谱">
    <PageHead title="语言图谱：外部能力框架" sub="语言里有哪些词和语言项、它们之间是什么关系、哪些已经有测试材料。这一页只描述语言本身，不含模型信息。" />
    <div className="wb-space-tabs lang-tabs" role="tablist" aria-label="语言图谱视图">{VIEWS.map(([id, label]) =>
      <button type="button" role="tab" key={id} aria-pressed={view === id} aria-selected={view === id} onClick={() => setView(id)}>{label}</button>)}</div>
    {view === 'ring' && <WordRing/>}
    {view === 'seed' && <LangGraph3D/>}
    {view === 'catalog' && <section className="v8-card cg-section" aria-label="语言目录">
      <p className="v8-note">知识网络、推理与功能、语法体系三大类，共 69 项，每项附一个例子。</p>
      <LanguageCatalog/>
    </section>}
    <PillTools label="语言图谱说明" items={[{ id: 'about', label: '这一页是什么', render: () => <>
      <p>语言里有上百万种物体的颜色、大小、用途等知识，也有复杂的推理和语法。模型能表现出这些能力，内部就必然有对应的编码机制。
        语言图谱先把这些能力整理成一个外部框架：有哪些词和语言项、它们之间是什么关系、哪些已经有测试材料。</p>
      <p className="v8-note">这一页只描述语言本身。每一项能力在模型里怎样编码，在“关联机制”里破解。</p>
    </> }]} />
  </div>;
}
