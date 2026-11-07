/* 空间 · 语言图谱：外部能力框架。只描述语言本身（语言项、关系与材料覆盖），不含任何模型信息。
   语言项在模型上能否读出、在哪一层、由哪些单元承载，放在“关联机制 · 语言项实测”。 */
import LangGraph3D from './LangGraph3D.jsx';
import LanguageCatalog from './LanguageCatalog.jsx';

export default function LanguageFrame() {
  return <div className="v8-route" aria-label="语言图谱">
    <header className="v8-overview-heading"><div>
      <h2>语言图谱：外部能力框架</h2>
      <p>语言里有上百万种物体的颜色、大小、用途等知识，也有复杂的推理和语法。模型能表现出这些能力，内部就必然有对应的编码机制。
        语言图谱先把这些能力整理成一个外部框架：有哪些语言项、它们之间是什么关系、哪些已经有测试材料。</p>
      <p className="v8-note">这一页只描述语言本身。每一项能力在模型里怎样编码，在“关联机制”里破解。</p>
    </div></header>
    <LangGraph3D/>
    <section className="v8-card cg-section" aria-label="语言目录">
      <h3>语言目录</h3>
      <p className="v8-note">知识网络、推理与功能、语法体系三大类，共 69 项，每项附一个例子。</p>
      <LanguageCatalog/>
    </section>
  </div>;
}
