import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Crosshair,
  FlaskConical,
  Gauge,
  Route,
} from 'lucide-react';
import { useState } from 'react';

import { CURRENT_RESEARCH_STATE } from '../researchKernel/currentResearchState';

import './ResearchProgressTab.css';

const RESEARCH_STAGES = [
  {
    id: 'static-search',
    phase: 'Phase 901-1106',
    title: '静态语义实体搜索',
    objective: '在颜色、属性、关系等受控任务上寻找固定语义方向、单神经元和局部因果组件。',
    result: '积累了候选竞争、内容条件化、复用拓扑和地址路由等局部拼图，但固定跨材料执行方向、单点通用运输器等假说未通过项目测试。',
    shift: '从寻找静态坐标转向追踪条件化计算过程。',
  },
  {
    id: 'behavior-gates',
    phase: 'Phase 1107-1125',
    title: '自然语义行为门',
    objective: '用 WordNet 四象限、形容词双正交材料和受控训练区分行为能力、线性可读与真实使用。',
    result: '形容词双正交行为在三模型中稳定通过；自然语境义项方向与定义方向的固定物理桥在 0/3 模型中闭合。',
    shift: '确立“先行为、后内部；可读不等于使用”的证据顺序。',
  },
  {
    id: 'instrument',
    phase: 'Phase 1126-1134',
    title: '数值、路线与材料资格',
    objective: '排除 FP16 溢出、运行路径和材料质量对机制结论的污染。',
    result: '定位 GLM4、DS7B 数值问题，建立模型与材料入口契约，并形成时效关系反事实研究对象。',
    shift: '确认数值健康和研究对象资格都是机制实验的前置门。',
  },
  {
    id: 'temporal-binding',
    phase: 'Phase 1135-1138',
    title: '时效绑定与状态跃迁',
    objective: '用四状态反事实与整残差替换定位答案状态何时获得可搬运性。',
    result: 'Qwen3-4B 的行为在 Qwen3-14B 同族复验；两个尺寸均在相对深度 0.6-0.7 出现可搬运性增强，但没有共同通过的充分深度。',
    shift: '研究对象从“语义方向”推进到“状态转移与充分性”。',
  },
  {
    id: 'matched-path',
    phase: 'Phase 1139',
    title: '同路径插值校准',
    objective: '消除跨批次执行漂移，并判断后半程状态变化是平滑调制还是相变。',
    result: 'live-state 同路径插值使 α=0 漂移精确为零；深度 0.7 是强调制 donor，但约三成样本不足以翻转答案。',
    shift: '任何因果结论必须先通过身份等价和执行路径等价。',
  },
  {
    id: 'sequence-alignment',
    phase: 'Phase 1140',
    title: '多 token 决策路径对齐',
    objective: '修复“单点补丁”与“序列级候选评分”之间的位置错配。',
    result: '覆盖 candidate prediction span 后，12/12 条共享前缀曲线被恢复；统一充分状态仍未获得双模型授权。',
    shift: '下一干预目标转向候选首次分叉的真实决策边界。',
  },
];

const CONFIRMED = [
  '项目测试中的语言信息更符合上下文条件化、分布式展开的过程，而不是一个跨材料固定不变的执行向量。',
  '输出表现为候选竞争；表示、控制、执行与最终生成必须分别记账。',
  'Qwen3-4B 与 Qwen3-14B 在时效绑定任务上重复出现后半程状态可搬运性增强。',
  '同路径 α=0 校准可以消除自补丁漂移，说明干预仪器必须先验证身份等价。',
  '共享首 token 的候选需要覆盖其预测路径，答案开头并不总是唯一决策位置。',
];

const NOT_CONFIRMED = [
  '尚未找到跨模型、跨任务稳定成立的最小计算单元或语言编码不变量。',
  '相对深度 0.6-0.7 的跃迁不等于存在统一语义层，也不等于完整机制已经形成。',
  '整残差 donor 没有同时获得必要性、充分性、特异性、预测性与跨材料重复。',
  '当前结果不能直接外推到所有大模型，更不能直接证明人脑采用相同编码机制。',
  '真实生成闭合、精确修改控制和完整智能理论仍未完成。',
];

const FORMULAS = [
  {
    label: '四状态绑定效应',
    formula: 'I_bind = 1/2 [(m_orig,post - m_orig,pre) - (m_swap,post - m_swap,pre)]',
    note: '消除候选静态偏好、时间方向偏好和档案格式偏好。',
  },
  {
    label: '自然状态转移',
    formula: 'X_(l+1) = F_l(X_l; q, c, Θ)',
    note: '研究重点是状态如何在条件 q、上下文 c 和参数 Θ 下逐层变化。',
  },
  {
    label: '同路径干预响应',
    formula: 'ρ_l(α) = M(F_>l((1-α)X_l^a + αX_l^b))',
    note: '在同一执行路径内插值状态，并测量候选 margin 或生成结果。',
  },
  {
    label: '机制边闭合门',
    formula: 'E_(u→v) = I ∧ S ∧ N ∧ C ∧ P ∧ R',
    note: '身份等价、充分性、必要性、特异性、独立预测和跨材料重复必须同时成立。',
  },
];

export const ResearchProgressTab = () => {
  const [activeStage, setActiveStage] = useState('sequence-alignment');
  const selectedStage = RESEARCH_STAGES.find((stage) => stage.id === activeStage);

  return (
    <div className="research-progress">
      <header className="research-progress__header">
        <div>
          <span>CURRENT RESEARCH · PHASE {CURRENT_RESEARCH_STATE.phase}</span>
          <h1>从静态表征搜索转向条件化状态转移</h1>
          <p>{CURRENT_RESEARCH_STATE.summary}</p>
        </div>
        <div className="research-progress__verdict">
          <AlertTriangle size={17} />
          <span>当前判决</span>
          <strong>{CURRENT_RESEARCH_STATE.statusLabel}</strong>
        </div>
      </header>

      <section className="research-progress__section">
        <div className="research-progress__heading">
          <Route size={19} />
          <div><h2>六阶段推进逻辑</h2><p>点击阶段查看目标、结果和方法论转换。</p></div>
        </div>
        <div className="research-progress__stages">
          {RESEARCH_STAGES.map((stage) => {
            const active = stage.id === activeStage;
            return (
              <button key={stage.id} type="button" className={active ? 'is-active' : ''} onClick={() => setActiveStage(active ? '' : stage.id)} aria-expanded={active}>
                <span>{stage.phase}</span>
                <strong>{stage.title}</strong>
                {active ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
            );
          })}
        </div>
        {selectedStage && (
          <div className="research-progress__stage-detail">
            <div><span>核心问题</span><p>{selectedStage.objective}</p></div>
            <div><span>实验结果</span><p>{selectedStage.result}</p></div>
            <div><span>推进意义</span><p>{selectedStage.shift}</p></div>
          </div>
        )}
      </section>

      <section className="research-progress__section research-progress__evidence">
        <div>
          <div className="research-progress__heading"><CheckCircle2 size={19} /><div><h2>当前可以说什么</h2><p>项目内部已经获得的正结果与稳定约束。</p></div></div>
          <ul>{CONFIRMED.map((item) => <li key={item}>{item}</li>)}</ul>
        </div>
        <div>
          <div className="research-progress__heading"><AlertTriangle size={19} /><div><h2>当前不能说什么</h2><p>仍未通过严格证据门的关键主张。</p></div></div>
          <ul>{NOT_CONFIRMED.map((item) => <li key={item}>{item}</li>)}</ul>
        </div>
      </section>

      <section className="research-progress__section">
        <div className="research-progress__heading"><Gauge size={19} /><div><h2>当前测试原理</h2><p>公式用于约束实验判决，不替代机制解释。</p></div></div>
        <div className="research-progress__formulas">
          {FORMULAS.map((item) => (
            <article key={item.label}><span>{item.label}</span><code>{item.formula}</code><p>{item.note}</p></article>
          ))}
        </div>
      </section>

      <section className="research-progress__next">
        <Crosshair size={23} />
        <div>
          <span>NEXT FALSIFIABLE TARGET</span>
          <h2>候选首次分叉决策边界</h2>
          <p>{CURRENT_RESEARCH_STATE.nextTask}</p>
        </div>
        <FlaskConical size={20} />
      </section>
    </div>
  );
};
