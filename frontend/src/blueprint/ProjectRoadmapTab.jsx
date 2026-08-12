import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Circle,
  Compass,
  Network,
  Repeat2,
  Target,
} from 'lucide-react';

import {
  CURRENT_RESEARCH_STATE,
  RESEARCH_EVIDENCE_GATES,
} from '../researchKernel/currentResearchState';

import './ProjectRoadmapTab.css';

const CONJECTURE_CHAIN = [
  {
    icon: <BrainCircuit size={18} />,
    title: '大脑可塑性',
    body: '工作猜想认为，高度可塑的神经网络能够被经验持续塑造；语言能力可能是这种塑造长期积累后的系统结果。',
  },
  {
    icon: <Network size={18} />,
    title: '语言数学结构',
    body: '注意力、自回归训练与大规模语言材料，使深度神经网络部分恢复了支撑语义、逻辑和语法的内部计算结构。',
  },
  {
    icon: <Target size={18} />,
    title: '智能理论',
    body: '若能恢复语言编码的可执行机制，便可能反推神经网络的塑造规律，并为可验证、可控制的 AGI 架构提供基础。',
  },
];

const WORKING_HYPOTHESES = [
  ['模式本体', '知识网络、分析推理和语法规则可以暂时统一为不同尺度、不同功能的模式。'],
  ['相对编码', '概念的含义由其在关系网络中的位置和作用共同决定，而不是一个孤立、固定的坐标。'],
  ['复用与差分', '可共享结构应尽量复用，无法复用的部分以较小差异表达独特特征。'],
  ['多尺度组织', '族内结构、族间关系和全局模式族分布可能共同平衡能力、容量与计算效率。'],
  ['动态生态位', '上下文和训练持续微调词的含义与用法，使每个模式形成条件化、相对独特的功能位置。'],
  ['多场竞争', '风格、逻辑、语法、知识和输出协议共同参与下一个 token 的候选竞争。'],
  ['实现粗糙度', '模型规模、训练数据和架构限制会引入噪声、捷径与模型特异实现，不能把小模型局部现象直接外推。'],
];

const CRACKING_TARGETS = [
  {
    index: '01',
    title: '具体模式',
    body: '恢复特殊词、翻译、转折、标点和终止等模式从输入条件到输出选择的完整计算链。',
    question: '一个模式在何时形成、由什么组件推进、怎样改变候选竞争？',
  },
  {
    index: '02',
    title: '模式族',
    body: '分析水果、动物、颜色、翻译和语法等模式族内部及族间的复用、差分和条件化路由。',
    question: '哪些结构被共享，哪些最小差异决定了模式身份？',
  },
  {
    index: '03',
    title: '整体网络',
    body: '重建模式族在层、位置和组件间的全局分布，解释其如何同时获得表达能力、效率与稳定性。',
    question: '是否存在可跨任务复用的最小计算单元或不变量？',
  },
];

const CORE_DIFFICULTIES = [
  '语言任务同时混合语义、语法、知识、风格和协议，难以获得只改变一个因素的纯净实验。',
  '现有数学工具通常只能描述局部投影，尚不能直接表达完整的条件化、分布式计算结构。',
  '编码可能同时包含相对关系、复用、差分、抽象与精确化，静态距离容易产生误导。',
  '规律跨越大量层、位置和组件；可读热点、强激活或高相似度不一定是模型实际使用的机制。',
  '紧凑结构、模型噪声和小模型粗糙度会产生大量虚影；当前仍缺少最小计算单元或稳定不变量。',
];

function EvidenceIcon({ status }) {
  if (status === 'passed') return <CheckCircle2 size={13} />;
  if (status === 'blocked') return <AlertTriangle size={13} />;
  return <Circle size={12} />;
}

export const ProjectRoadmapTab = () => (
  <div className="project-outline">
    <header className="project-outline__header">
      <div>
        <span>PROJECT OUTLINE</span>
        <h1>破解语言编码机制</h1>
        <p>从可证伪的语言行为出发，恢复深度神经网络内部可重复、可干预、可预测的计算结构，寻找支撑智能的最小计算单元或不变量。</p>
      </div>
      <div className="project-outline__phase">
        <span>当前研究</span>
        <strong>Phase {CURRENT_RESEARCH_STATE.phase}</strong>
        <small>{CURRENT_RESEARCH_STATE.statusLabel}</small>
      </div>
    </header>

    <section className="project-outline__section">
      <div className="project-outline__section-heading">
        <span>01</span>
        <div><h2>核心猜想</h2><p>这是研究方向，不是已经证明的结论。</p></div>
      </div>
      <div className="project-outline__chain">
        {CONJECTURE_CHAIN.map(({ icon, title, body }, index) => (
          <article key={title}>
            <div>{icon}<span>0{index + 1}</span></div>
            <h3>{title}</h3>
            <p>{body}</p>
          </article>
        ))}
      </div>
    </section>

    <section className="project-outline__section">
      <div className="project-outline__section-heading">
        <span>02</span>
        <div><h2>系统工作假设</h2><p>每一项都需要独立实验检验，不能相互代替证据。</p></div>
      </div>
      <div className="project-outline__hypotheses">
        {WORKING_HYPOTHESES.map(([title, body], index) => (
          <article key={title}><span>{String(index + 1).padStart(2, '0')}</span><h3>{title}</h3><p>{body}</p></article>
        ))}
      </div>
    </section>

    <section className="project-outline__section">
      <div className="project-outline__section-heading">
        <span>03</span>
        <div><h2>三级破解目标</h2><p>从单个模式逐步推进到模式族和整体网络。</p></div>
      </div>
      <div className="project-outline__targets">
        {CRACKING_TARGETS.map((target) => (
          <article key={target.index}>
            <span>{target.index}</span><h3>{target.title}</h3><p>{target.body}</p><strong>{target.question}</strong>
          </article>
        ))}
      </div>
    </section>

    <section className="project-outline__section project-outline__split">
      <div>
        <div className="project-outline__section-heading">
          <span>04</span>
          <div><h2>核心难点</h2><p>为什么大量局部规律仍不能自动组成理论。</p></div>
        </div>
        <ol className="project-outline__difficulties">
          {CORE_DIFFICULTIES.map((item) => <li key={item}>{item}</li>)}
        </ol>
      </div>
      <aside className="project-outline__goal">
        <Compass size={24} />
        <span>CORE OBJECTIVE</span>
        <h2>找到能够贯穿整个网络的算法或不变量</h2>
        <p>它应当能够解释运行机制、预测状态转移，并支持精确修改和控制。只有完成跨材料、跨任务和跨模型的因果闭合，才有资格上升为智能理论。</p>
      </aside>
    </section>

    <section className="project-outline__section">
      <div className="project-outline__section-heading">
        <span>05</span>
        <div><h2>当前证据边界</h2><p>{CURRENT_RESEARCH_STATE.summary}</p></div>
      </div>
      <div className="project-outline__gates">
        {RESEARCH_EVIDENCE_GATES.map((gate) => (
          <article key={gate.id} className={`is-${gate.status}`}>
            <div><EvidenceIcon status={gate.status} /><span>{gate.shortLabel}</span></div>
            <strong>{gate.value}</strong>
            <small>{gate.phase}</small>
          </article>
        ))}
      </div>
      <div className="project-outline__next">
        <Repeat2 size={17} />
        <div><strong>下一项关键实验</strong><p>{CURRENT_RESEARCH_STATE.nextTask}</p></div>
      </div>
    </section>
  </div>
);
