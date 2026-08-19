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
    id: 'behavior-and-instrument',
    phase: 'Phase 1107-1140',
    title: '行为门、仪器资格与序列路径',
    objective: '先确认模型稳定执行目标行为，再校准数值、材料、同路径干预和多 token 决策相机。',
    result: '建立了“可读不等于使用”和同路径身份等价标准；后半程状态可搬运性增强得到同族重复，但没有统一充分状态。',
    shift: '从固定方向读取转向有明确行为对象、位置合同和执行等价的状态转移实验。',
  },
  {
    id: 'known-truth-cameras',
    phase: 'Phase 1151-1201',
    title: '已知真值相机与因果联盟校准',
    objective: '在机制真值可知的网络中检验算法能否识别双路径、冗余、协同、形成事件、响应商和救援结构。',
    result: '相机能够识别部分已知真值结构，也系统暴露学习后外部效度、组合泛化、长时身份和粗联盟救援的失败边界。',
    shift: '确认真正可迁移的对象应是条件化未来响应与功能等价类，而非固定坐标或单点热点。',
  },
  {
    id: 'functional-objects',
    phase: 'Phase 1202-1228',
    title: '功能商、类型化行为与原子对象',
    objective: '把对象—属性绑定、操作行为、生成时间和内部响应拆成可分别授权的功能对象。',
    result: '获得五个行为原子和若干局部 Qwen3 因果拼图，同时证明单点答案边界、固定数值域和教师强制相机都存在明确适用边界。',
    shift: '从“某层是什么”转向“同一功能在不同条件和读出下保持哪些可观测关系”。',
  },
  {
    id: 'typed-generation',
    phase: 'Phase 1229-1235',
    title: '程序可识别行为门与七读出边界',
    objective: '用去答案载荷、非双射世界、密封材料和七种读出区分内容选择、格式服从与自然生成。',
    result: 'Phase1235 四门为 (1,0,1,1)：候选评分、固定句式和自然内容稳定，严格短字符串合同失败；总门因此为 0。',
    shift: '同一对象的行为必须记为类型化响应向量，不能再用一个总准确率或“知道但说不出”替代机制分析。',
  },
  {
    id: 'research-os-wp00',
    phase: 'Phase 1236 · C001-WP00',
    title: '全局结构辨识研究操作系统',
    objective: '停止围绕局部热点无限续写 Phase，把证据、假说、对象、构念、测试、裁决和权限编译成机器可拒绝的系统。',
    result: 'WP00 校验通过；9 个假说、14 个拼图、13 类测试、14 条关键证据、2 个对象和 4 个构念已登记。WP01 合同已预注册但 run_ready=false。',
    shift: 'Phase 只保留为执行批次，科学主线改为响应张量、候选机制竞赛、主动实验和封存裁决。',
  },
];

const CONFIRMED = [
  '项目测试中的语言信息更符合上下文条件化、分布式展开的过程，而不是一个跨材料固定不变的执行向量。',
  '内容选择、严格格式、自然完整生成和停止缓存属于不同构念，必须分别取证和授权。',
  'Phase1235 在 96 个 axis-world 上确认：完整候选排序稳定不蕴含所有字符串读出合同稳定。',
  'K210 支持类型化行为响应边界，但明确不支持隐藏内容模块或生成编译模块主张。',
  'C001-WP00 已把关键证据和权限迁移为可校验注册表，能够阻止从局部正结果越权启动内部扫描。',
];

const NOT_CONFIRMED = [
  '尚未找到跨模型、跨任务稳定成立的最小计算单元或语言编码不变量。',
  '没有在当前合格自然对象上获得 hidden、attention、head、MLP、neuron、必要性、救援或中介证据。',
  'WP00 工程闭合不等于 WP01 科学实验通过；当前 run_ready=false，模型运行数为 0。',
  '人工英语 registry 和单模型 Qwen3 结果不能外推到开放语言、其他模型或人脑编码机制。',
  '真实生成闭合、跨模型功能同构、精确修改控制和完整智能理论均未完成。',
];

const FORMULAS = [
  {
    label: '类型化行为向量',
    formula: 'b(x,q) = (B_choice, B_bare-score, B_trie, B_bare, B_cued, B_sentence, B_natural)',
    note: '同一内容查询在七种读出合同中分别记账，不把异质成功和失败压成一个分数。',
  },
  {
    label: 'Phase1235 类型门',
    formula: 'G_1235 = G_program ∧ G_candidate ∧ G_short ∧ G_sentence ∧ G_natural = 0',
    note: '实测核心四门为 (1,0,1,1)，冻结合取失败，因此不能追溯授权 hidden。',
  },
  {
    label: '未来响应候选对象',
    formula: 'R_T(H_e) = { Q(F_≥e(do(I), H_e)) - Q(F_≥e(H_e)) }_(I,Q)',
    note: '这是下一阶段候选测量对象，当前尚未获准采集，不能写成已经发现的内部结构。',
  },
  {
    label: 'WP01 运行权限',
    formula: 'Authorize_run ⇔ contract_frozen ∧ material_audit ∧ leakage_controls ∧ environment_manifest ∧ independent_audit',
    note: '任一先决项未通过，机器必须保持 run_ready=false。',
  },
];

export const ResearchProgressTab = () => {
  const [activeStage, setActiveStage] = useState('research-os-wp00');
  const selectedStage = RESEARCH_STAGES.find((stage) => stage.id === activeStage);

  return (
    <div className="research-progress">
      <header className="research-progress__header">
        <div>
          <span>CURRENT RESEARCH · PHASE {CURRENT_RESEARCH_STATE.phase}</span>
          <h1>从局部 Phase 转向全局结构辨识战役</h1>
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
          <div><h2>七阶段推进逻辑</h2><p>点击阶段查看目标、结果和方法论转换。</p></div>
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
          <h2>WP01 无模型冻结与反泄漏预审计</h2>
          <p>{CURRENT_RESEARCH_STATE.nextTask}</p>
        </div>
        <FlaskConical size={20} />
      </section>
    </div>
  );
};
