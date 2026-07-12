import { useEffect, useMemo, useState } from 'react';
import { EvidenceKernelDashboard } from './EvidenceKernelDashboard';

const ATLAS_BASE = '/vis_data/pattern_family_atlas/v2';

const pct = (value) => `${Math.round(Number(value || 0) * 100)}%`;
const clamp01 = (value) => Math.max(0, Math.min(1, Number(value || 0)));

const cardStyle = {
  border: '1px solid rgba(148, 163, 184, 0.16)',
  borderRadius: 8,
  background: 'rgba(15, 23, 42, 0.68)',
  boxShadow: '0 18px 42px rgba(0, 0, 0, 0.22)',
};

const planCards = [
  {
    id: 'atlas_position',
    title: '方案定位',
    summary: '语言模式图谱不是结果展示表，而是逆向分析语言机制的主数据结构。',
    value: 'PatternPath(x,m)',
    detail: {
      title: '方案定位：从单点解释到 PatternPath',
      text: '语言模式图谱不是把实验结果堆在一起，而是把每个语言能力拆成可记录、可比较、可干预、可预测的内部机制路径。它要回答：属于哪个模式族、走什么物理路径、哪些证据有因果必要性、能否在自然生成中干净闭合。',
      value: '逆向分析主数据结构',
      valueHint: '基本对象是 PatternPath(x,m)：x 是样本，m 是模型。它比单神经元、单 head、单 token margin 更适合描述语言机制。',
      items: [
        '核心问题一：某类语言能力属于哪个模式族。',
        '核心问题二：这个模式族在模型内部走什么物理路径。',
        '核心问题三：哪些路径只是相关信号，哪些路径具有因果必要性。',
        '核心问题四：这条路径能否在自然生成中形成干净闭合。',
        '完整链路：prompt/context -> state variables -> route variables -> component path -> readout competition -> protocol/rollout gate -> closure gate -> evidence level -> claim registry。'
      ],
      formulas: [
        'PatternPath(x,m)',
        'LanguageMechanism(x,t)=sum_i alpha_i(x,t) P_i(x,t)',
        'P_i=Trigger_i o Route_i o StateWrite_i o ComponentPath_i o ReadoutCompetition_i o Rollout_i o Closure_i'
      ],
    },
  },
  {
    id: 'family_path_schema',
    title: '模式族与路径结构',
    summary: '用九大语言模式族和统一路径签名，把语言能力拆成可累计的图谱节点。',
    value: '9 families / full trace',
    detail: {
      title: '模式族与路径结构',
      text: '图谱以九大语言模式族为工作分类，并要求每个 case-model 逐步补齐行为、读出、层路径、组件路径、因果、补偿、rollout、闭合、语义评估、跨模型一致性和非线性耦合字段。',
      value: 'Family -> PathSignature -> Claim',
      valueHint: '九大族是工作分类，不是已经证明的完备分类；后续需要 open-set 检验 unknown / mixed / known。',
      items: [
        '九大模式族：content_knowledge、output_protocol、reasoning_constraint、syntax_structure、language_action、cross_lingual、readout_competition、state_drift、closure。',
        '全链路字段：behavior、readout、layer_path、component_path、causal、compensation、rollout、closure、semantic_eval、cross_model_consistency、nonlinear_coupling、claim_id、evidence_level。',
        'PathSignature 记录：State、DominantLayers、AttentionRoute、MLPWrite、Compensation、ReadoutWinner、ProtocolGate、ClosureQuality。',
        'Claim Registry 记录每条机制主张的适用范围、证据等级、正证据、反证据、反例和下一步测试。',
        '如果 unknown_family_rate > 5%，九大族不能冻结；如果 mixed_family_rate > 15%，说明模式族过粗或存在强耦合。'
      ],
      formulas: [
        'Atlas(f,m)=[Cases, Behavior, Readout, LayerPath, ComponentPath, CausalAudit, CompensationPath, Rollout, Closure]',
        'PathSignature(x,m)=[State, DominantLayers, AttentionRoute, MLPWrite, Compensation, ReadoutWinner, ProtocolGate, ClosureQuality]',
        'Claim(c)=[ClaimID, Scope, EvidenceLevel, PositiveEvidence, NegativeEvidence, CounterExamples, NextTest]'
      ],
    },
  },
  {
    id: 'evidence_closure',
    title: '证据、评分与闭合',
    summary: '结论必须区分观察、路径、组件、因果、rollout 和 clean closure，不能把中等工程分误判为闭合。',
    value: 'Score / Gate / Evidence',
    detail: {
      title: '证据、评分与闭合标准',
      text: '图谱完成度评分只是工程指标，不是智能理论闭合公式。当前方案使用加权评分和封顶规则，防止 behavior/readout 高分掩盖 causal、component_path 和 closure 的关键缺口。',
      value: '闭合是硬判据',
      valueHint: '只有 SemanticDone、StopWins、ContinueSuppressed、RolloutStable 同时成立，才能标记为 closed。',
      items: [
        '证据等级：L0 假设、L1 行为现象、L2 readout 可读出、L3 层路径稳定、L4 组件归因、L5 低副作用因果必要性、L6 受控充分性、L7 rollout 稳定、L8 clean closure。',
        '封顶规则：I(x)<0.30 时总分不超过 0.50；C(x)<0.30 时总分不超过 0.60；K(x)<0.30 时总分不超过 0.65。',
        'semantic_eval 未通过，不能标记 high_quality_candidate。',
        'cross_model_conflict 为 true 时，不能汇总为 global mechanism，只能标记 model_specific_mechanism。',
        '非线性耦合为 true 时，不能把单 writer 的线性贡献解释为完整机制，必须进入 compensation_path_audit。'
      ],
      formulas: [
        'Score_base=0.10B+0.10R+0.15L+0.20C+0.25I+0.10G+0.10K',
        'Score_atlas=min(Score_base, Cap_I, Cap_C, Cap_K)',
        'Closure=SemanticDone and StopWins and ContinueSuppressed and RolloutStable',
        'X(f)=1-N_disagree(f)/N_models(f)',
        'NonlinearCoupling=1[abs(Delta_actual-Delta_linear)>epsilon]'
      ],
    },
  },
  {
    id: 'visualization_execution',
    title: '可视化与执行流程',
    summary: '客户端要先看全局图谱，再按需展开路径、证据、反例、预测和 case 详情。',
    value: 'Overview -> Detail',
    detail: {
      title: '可视化与执行流程',
      text: '可视化客户端不是附属展示层，而是图谱研究的操作界面。它应支持 summary-first、detail-on-demand：先看整体结构和缺口，再点击进入具体路径、证据、反例、公式和原始 JSON。',
      value: '分层展开 + 预测验证',
      valueHint: '图谱不能只回顾性整理历史数据，还要用 heldout 样本验证 dominant_layers、component_path、readout_winner、closure_gate 和 failure_type 的预测。',
      items: [
        'Overview：总 path signatures、高质量候选、平均分、最近 Phase、未完成字段比例。',
        'Family Matrix：family x model，每格显示 weighted_score、score_cap_reason、causal、closure、cross_model_conflict。',
        'Path Explorer：prompt -> state -> dominant layers -> component route -> readout -> protocol gate -> closure quality。',
        'Component View：attention_route_score、mlp_write_score、dominant_layers、compensation_score、nonlinear_coupling。',
        'Causal Audit：zero、half、mean_replace、random_same_norm、permutation、cross-sample patch、direction add/remove、side effect。',
        'Claim Registry：claim_id、scope、evidence_level、positive_files、negative_files、counterexamples、next_test。',
        'Case Detail：raw prompt、target、path signature、behavior row、readout row、component row、causal rows、semantic eval、raw JSON。',
        '交互方式：右侧不固定放说明窗口，说明在对应模块下方按需展开。'
      ],
      formulas: [
        'P_hat(x,m)=AtlasPredict(family(x), mode(x), model(m), prompt_features(x))',
        'Acc_path=N(P_hat_path=P_path)/N_heldout'
      ],
    },
  },
];

const progressRows = [
  {
    id: 'pattern_family_atlas',
    label: '语言模式族谱',
    hint: '语言模式分类、样本组织和路径框架的完成度。',
    fallback: 0,
  },
  {
    id: 'physical_path_atlas',
    label: '物理路径图谱',
    hint: '逐层路径、内部状态和读出轨迹的追踪完成度。',
    fallback: 0,
  },
  {
    id: 'component_path_atlas',
    label: '组件路径图谱',
    hint: '注意力、MLP、残差等组件路径的定位进度。',
    fallback: 0,
  },
  {
    id: 'readout_competition_trace',
    label: '读出竞争追踪',
    hint: '目标、继续、停止和错误输出之间的竞争解释进度。',
    fallback: 0,
  },
  {
    id: 'stepwise_rollout_trace',
    label: '生成轨迹追踪',
    hint: '后续生成、状态漂移、协议延续和停止行为的记录进度。',
    fallback: 0,
  },
  {
    id: 'causal_closure',
    label: '机制闭合',
    hint: '因果必要性、竞争胜出、低副作用和自然闭合的综合进度。',
    fallback: 0,
  },
];

const latestCards = [
  {
    id: 'overall_latest',
    title: '总体最新判断',
    summary: '研究已经从颜色特征和单点读出，推进到语言模式、路径图谱和闭合验证。',
    value: '机制图谱阶段',
    detail: {
      title: '总体最新判断',
      text: '当前最重要的成果，是研究对象已经从“答案 token 如何出现”升级为“语言模式如何形成、竞争、生成和闭合”。这意味着项目不再只追逐单点解释，而是开始构建可复用的内部机制地图。',
      value: 'Pattern -> Path -> Closure',
      valueHint: '最新主线可以概括为：语言模式族谱已经成型，物理路径图谱正在深化，闭合验证仍是最关键终检。',
      items: [
        '语言能力被理解为多个模式族在上下文中的竞争 and 组合，而不是单个输出神经元的结果。',
        '停止机制被独立出来，说明“答对”和“知道停止”不是同一个问题。',
        'MLP 写入、注意力路由、读出竞争和补偿路径成为当前机制分析重点。',
        '后续重点是把可观察路径升级为因果路径，再把因果路径升级为稳定闭合。'
      ],
    },
  },
  {
    id: 'engineering_latest',
    title: '图谱工程进展',
    summary: '研究记录正在转化为统一图谱结构，支持进度、证据、缺口和下一步任务的自动组织。',
    value: 'Atlas v2',
    detail: {
      title: '图谱工程进展',
      text: '图谱工程的意义，是把每个实验从一次性结论变成可持续积累的节点、边和证据。当前已经具备模式族、路径、组件、读出、生成和闭合六个核心模块。',
      value: '6 个核心模块',
      valueHint: '模块进度来自客户端数据文件，读取失败时使用保守默认值显示。',
      items: [],
    },
  },
  {
    id: 'theory_latest',
    title: '机制理论进展',
    summary: '最新理论把语言机制表达为触发、路由、状态写入、读出、竞争、生成和闭合的组合链。',
    value: '综合机制链',
    detail: {
      title: '机制理论进展',
      text: '语言机制不再被描述为某一个单独公式，而是被描述为一条综合机制链。不同语言模式在上下文中被触发，通过内部路径改变状态，再进入输出竞争，最后由闭合机制决定是否结束。',
      value: 'LanguageMechanism',
      valueHint: '这条理论链可以直接转化为图谱结构：每个环节都是节点，每个证据都是边，每个缺口都是下一步任务。',
      items: [
        'Trigger：上下文如何激活某个语言模式。',
        'Route：注意力和残差路径如何把信息送到关键位置。',
        'StateWrite：MLP 或其他组件如何写入状态变量。',
        'Readout / Competition：目标、解释、重复、继续和停止之间如何竞争。',
        'Rollout / Closure：自然生成是否稳定，停止是否真正执行。'
      ],
      formulas: [
        'LanguageMechanism(x,t)=sum_i alpha_i(x,t)[Trigger_i o Route_i o StateWrite_i o Readout_i o Competition_i o Rollout_i o Closure_i]'
      ],
    },
  },
  {
    id: 'gap_latest',
    title: '关键缺口',
    summary: '最难的部分不是发现相关路径，而是证明路径具有因果必要性、低副作用和跨样本稳定性。',
    value: '闭合仍是终检',
    detail: {
      title: '关键缺口',
      text: '当前图谱需要继续补齐最有价值的缺口：哪些路径是必要路径，哪些只是相关信号；模型如何通过补偿路径绕过干预；停止和继续竞争如何在自然 rollout 中稳定闭合。',
      value: '因果 / 补偿 / 闭合',
      valueHint: '下一阶段应该优先补“高缺口、高证据价值、跨模型价值高”的节点。',
      items: [
        '把读出路径和真实因果路径区分开。',
        '识别干预后的补偿路径，避免把失败干预误判为机制不存在。',
        '验证 done、stop、continue 三类状态的竞争关系。',
        '建立跨模型、跨任务、跨样本的稳定性证据。',
        '把每个缺口转化为可执行测试脚本和结构化结果文件。'
      ],
    },
  },
];

const historyStages = [
  {
    id: 'phase_195_203',
    range: 'Phase 195-203',
    title: '全链路闭合 Trace',
    summary: '从单点神经元记录升级为完整链路记录，发现候选集合闭合不等于全词表输出闭合。',
    problem: '单个最大神经元不足以解释颜色能力和语言输出，需要记录从输入到输出的完整计算链路。',
    principle: '逐层记录内部状态、候选 token 分数、margin、熵和闭合强度，判断目标答案何时真正获得输出优势。',
    conclusion: '颜色候选空间中的闭合，只说明局部候选可分；真正输出还必须通过全词表竞争和自然生成验证。',
    formulas: [
      'score_l(token)=h_l · W_U[token]',
      'p_l(c)=softmax(score_l(c))',
      'K_l=score_l(target)-max(score_l(other))',
      'M_l=1-H(p_l)/log|Colors|'
    ],
    phases: [
      'Phase 195-198：追踪颜色特征从输入到候选答案的逐层变化。',
      'Phase 199-201：比较候选空间和全词表空间，发现局部闭合和真实输出之间存在断层。',
      'Phase 202-203：把 Trace 从结果记录推进为机制验证入口。'
    ],
  },
  {
    id: 'phase_204_208',
    range: 'Phase 204-208',
    title: '停止机制与输出协议',
    summary: '证明输出句号不等于真正停止，停止执行是独立机制。',
    problem: '模型为什么有时答对了还继续解释？为什么输出句号后仍可能延续？',
    principle: '分离 prose、stop、continue 等输出协议，比较停止方向和继续方向在内部状态中的竞争优势。',
    conclusion: '停止机制不是语义答案的附属物，而是一个独立执行问题；答对只解决语义，停止还需要压制继续路径。',
    formulas: [
      'm_stop(t)=max V_stop z_t - max V_prose z_t',
      'm_prose(t)=max V_prose z_t - max V_stop z_t',
      'StopExecuted = StopReadoutWins and ContinueSuppressed'
    ],
    phases: [
      'Phase 204-205：区分答案输出、句号输出和真正停止执行。',
      'Phase 206-208：追踪 stop/prose/continue 的竞争，确认停止是独立闭合环节。'
    ],
  },
  {
    id: 'phase_209_218',
    range: 'Phase 209-218',
    title: '语言模式 Pattern 建模',
    summary: '从答案 token 研究转向语言模式族研究，提出 answer、explain、repeat、list、chatty 等模式竞争。',
    problem: '同一个问题为什么会触发不同回答风格 and 延续方式？语言能力如何从 token 输出升级为模式选择？',
    principle: '把语言输出视为多个模式的动态竞争，每个模式包含触发条件、状态变量、特征轨迹、优先级代理、输出约束和失败模式。',
    conclusion: '语言机制应该被研究为 Pattern Family，而不是孤立 token；这是后续图谱化的理论入口。',
    formulas: [
      'Pattern=(Trigger, StateVariables, FeatureTrajectory, PriorityProxy, OutputConstraint, FailureModes)',
      'h_{t+1}=sum_k alpha_k T_k(h_t)+eps',
      'P_k=[alpha, phi, Delta h, b, o]'
    ],
    phases: [
      'Phase 209-212：提出语言模式结构，把输出行为拆成多个可竞争模式。',
      'Phase 213-216：观察 answer/explain/repeat/continue 等模式的切换和干扰。',
      'Phase 217-218：把模式建模和内部状态轨迹连接起来。'
    ],
  },
  {
    id: 'phase_219_233',
    range: 'Phase 219-233',
    title: 'StateWrite / Readout / Competitor 拆解',
    summary: '追问谁写入状态、写在哪里、如何被读出、竞争源来自哪里。',
    problem: '语言模式不是抽象标签，必须定位内部状态如何被写入、由哪些组件写入、如何进入读出竞争。',
    principle: '构造成功/漂移方向，追踪 MLP、attention 和 residual 的贡献，并通过方向增强或移除进行因果测试。',
    conclusion: '很多关键写入来自 MLP，attention 更多承担路由和条件化作用；竞争机制来自多条状态路径的读出差异。',
    formulas: [
      'v_{l,t}^{S-D}=E[h_{l,t}|success]-E[h_{l,t}|drift]',
      'u_{m,l,t}^{S-D}=E[o_{m,l,t}|success]-E[o_{m,l,t}|drift]',
      'SourceAlign=cos(u_{m,l,t}^{S-D}, v_{l,t}^{S-D})',
      "h'=h ± lambda v_hat"
    ],
    phases: [
      'Phase 219-224：定位状态写入方向和读出方向。',
      'Phase 225-229：拆解 MLP、attention、residual 在写入和路由中的差异。',
      'Phase 230-233：用竞争源和干预测试验证模式漂移原因。'
    ],
  },
  {
    id: 'phase_234_245',
    range: 'Phase 234-245',
    title: 'Pattern Family Atlas v1',
    summary: '把研究从实验记录升级为图谱工程，开始定义统一数据结构。',
    problem: '实验越来越多，如果没有统一格式，很难积累、比较和复用。',
    principle: '每个 Pattern 必须记录完整链路：触发、门控、残差写入、读出、竞争、生成、闭合。',
    conclusion: '这是从“做实验”转向“建图谱”的关键阶段，研究开始具备长期积累性。',
    formulas: [
      'LanguageMechanism=sum_i alpha_i P_i',
      'P_i=TriggerTrace o GateProductTrace o ResidualWriteTrace o ReadoutTrace o CompetitorTrace o RolloutTrace o ClosureTrace'
    ],
    phases: [
      'Phase 234-238：定义 Pattern Family Atlas 的基本字段和证据链。',
      'Phase 239-242：把触发、读出、竞争、rollout 和 closure 纳入同一结构。',
      'Phase 243-245：形成可以被客户端和脚本共同使用的数据格式。'
    ],
  },
  {
    id: 'phase_246_253',
    range: 'Phase 246-253',
    title: '机制方向库与共享子空间',
    summary: '验证不同语言模式是否共享内部子空间，机制方向能否复用。',
    problem: 'answer、explain、repeat、continue、stop 等模式是否完全独立，还是共享某些底层方向？',
    principle: '从正负样本提取 mechanism direction，再进行方向增强、方向移除和正交干预。',
    conclusion: '语言模式不是完全独立的，它们可能共享底层子空间，只是在不同边界条件下被读出为不同模式。',
    formulas: [
      'v_mech=mu_positive-mu_negative',
      's_mech(h)=h · v_hat_mech',
      "h'=h+lambda v_hat_mech",
      "h'=h-(h · v_hat_mech)v_hat_mech"
    ],
    phases: [
      'Phase 246-249：提取可复用机制方向，测试方向增强和移除效果。',
      'Phase 250-253：比较不同语言模式之间的共享子空间和边界条件。'
    ],
  },
  {
    id: 'phase_254_263',
    range: 'Phase 254-263',
    title: 'Done / Stop / Continue 竞争机制',
    summary: '把闭合问题拆成语义完成、模板完成、停止读出和继续读出。',
    problem: '模型为什么知道答案已经完成？为什么有时仍继续解释或延伸？',
    principle: '比较 close candidate 和 non-close candidate 的内部轨迹，寻找 done state 方向，再追踪 stop/continue 竞争。',
    conclusion: '真正闭合必须同时满足语义完成、停止获胜、继续被抑制和后续生成稳定。',
    formulas: [
      'v_done=mu_closed-mu_open',
      'S_done(h)=h · v_hat_done',
      'M_stop/continue=R_stop(h)-R_continue(h)',
      'Closure=SemanticDone and StopWins and ContinueSuppressed and RolloutStable'
    ],
    phases: [
      'Phase 254-257：区分 semantic done、template done、stop readout 和 continue readout。',
      'Phase 258-263：建立 done state 与 stop/continue 竞争之间的关系。'
    ],
  },
  {
    id: 'phase_264_278',
    range: 'Phase 264-278',
    title: '物理路径图谱 v2',
    summary: '把机制研究升级为可预测、可验证、可复用的物理路径图谱。',
    problem: '前面已经证明多个机制存在，但仍需要系统地把每个模式拆成可验证路径。',
    principle: '对每个语言模式记录 embedding bias、attention route、MLP writer set、compensation path、readout competition、rollout effect、side effect 和 cross-model consistency。',
    conclusion: '研究重点从提出单个公式，升级为用工程评分和缺口队列管理图谱进度。',
    formulas: [
      'M(h)=R_continue(h)-R_stop(h)',
      'Delta M_attn^(l)=M(h_l+a_l)-M(h_l)',
      'Delta M_mlp^(l)=M(h_l+a_l+m_l)-M(h_l+a_l)',
      'ContinuePath=B_embed + AttentionRoute + MLPWriterSet + CompensationPath + ReadoutCompetition + RolloutEffect',
      'Score(x)=w1B(x)+w2R(x)+w3L(x)+w4C(x)+w5I(x)+w6G(x)+w7K(x)',
      'Priority=Gap * EvidenceValue * CrossModelValue * RiskControl'
    ],
    phases: [
      'Phase 264-269：启动物理路径、组件路径、因果路径和补偿路径追踪。',
      'Phase 270-273：把路径证据和闭合验证合并为图谱评分。',
      'Phase 274-278：进入 Atlas v2，用缺口驱动系统决定下一步研究优先级。'
    ],
  },
];

function readProgressValue(progress, id, fallback = 0) {
  return progress?.[id] ?? progress?.progress?.[id] ?? progress?.global_progress?.[id] ?? fallback;
}

function ProgressBar({ value, color = '#67e8f9' }) {
  const width = pct(clamp01(value));
  const barGradient = color === '#67e8f9'
    ? 'linear-gradient(90deg, #06b6d4, #67e8f9)'
    : color === '#a7f3d0'
      ? 'linear-gradient(90deg, #059669, #34d399)'
      : `linear-gradient(90deg, ${color}, ${color})`;
  return (
    <div style={{
      height: 6,
      borderRadius: 999,
      background: 'rgba(15, 23, 42, 0.65)',
      border: '1px solid rgba(255, 255, 255, 0.03)',
      padding: 0,
      overflow: 'hidden'
    }}>
      <div style={{
        width,
        height: '100%',
        borderRadius: 999,
        background: barGradient,
        transition: 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
      }} />
    </div>
  );
}

function SectionHeader({ eyebrow, title, subtitle }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{
        color: '#67e8f9',
        fontSize: 10,
        fontWeight: 800,
        letterSpacing: 1.8,
        textTransform: 'uppercase',
        display: 'flex',
        alignItems: 'center',
        gap: 6
      }}>
        <span style={{ display: 'inline-block', width: 10, height: 1, background: '#67e8f9' }} />
        {eyebrow}
      </div>
      <h3 style={{ margin: '2px 0 0 0', color: '#f8fafc', fontSize: 22, fontWeight: 800, lineHeight: 1.2, letterSpacing: -0.2 }}>
        {title}
      </h3>
      <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.6, maxWidth: 960, marginTop: 2 }}>
        {subtitle}
      </div>
    </div>
  );
}

function ClickableCard({ children, active, onClick, style, className = '', activeColor = '#22d3ee' }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`clickable-card ${className}`}
      style={{
        ...cardStyle,
        ...style,
        width: '100%',
        textAlign: 'left',
        cursor: 'pointer',
        color: 'inherit',
        fontFamily: 'inherit',
        outline: 'none',
        transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
        border: active
          ? `1px solid ${activeColor}`
          : '1px solid rgba(255, 255, 255, 0.05)',
        borderTop: active
          ? `3px solid ${activeColor}`
          : '1px solid rgba(255, 255, 255, 0.05)',
        background: active
          ? 'rgba(15, 23, 42, 0.5)'
          : 'rgba(15, 23, 42, 0.18)',
        boxShadow: active
          ? `0 10px 30px rgba(0, 0, 0, 0.35), 0 0 10px ${activeColor}15`
          : 'none',
        backdropFilter: 'blur(8px)',
        position: 'relative',
      }}
    >
      {children}
    </button>
  );
}

function formatValueHint(hint) {
  if (!hint) return null;
  if (hint.includes('核心问题：') && hint.includes('测试原理：')) {
    const problemMatch = hint.match(/核心问题：(.*?)(?=测试原理：|$)/);
    const principleMatch = hint.match(/测试原理：(.*?)(?=阶段结论：|$)/);
    const conclusionMatch = hint.match(/阶段结论：(.*?)$/);

    const problem = problemMatch ? problemMatch[1].trim() : '';
    const principle = principleMatch ? principleMatch[1].trim() : '';
    const conclusion = conclusionMatch ? conclusionMatch[1].trim() : '';

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%' }}>
        {problem && (
          <div style={{ padding: '10px 14px', borderRadius: 8, background: 'rgba(239, 68, 68, 0.03)', border: '1px solid rgba(239, 68, 68, 0.08)', borderLeft: '3px solid #f87171' }}>
            <span style={{ color: '#f87171', fontWeight: 'bold', fontSize: 10, display: 'block', letterSpacing: 0.5, marginBottom: 2 }}>
              核心问题 KEY PROBLEM
            </span>
            <span style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.5 }}>{problem}</span>
          </div>
        )}
        {principle && (
          <div style={{ padding: '10px 14px', borderRadius: 8, background: 'rgba(56, 189, 248, 0.03)', border: '1px solid rgba(56, 189, 248, 0.08)', borderLeft: '3px solid #38bdf8' }}>
            <span style={{ color: '#38bdf8', fontWeight: 'bold', fontSize: 10, display: 'block', letterSpacing: 0.5, marginBottom: 2 }}>
              测试原理 TESTING PRINCIPLE
            </span>
            <span style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.5 }}>{principle}</span>
          </div>
        )}
        {conclusion && (
          <div style={{ padding: '10px 14px', borderRadius: 8, background: 'rgba(52, 211, 153, 0.03)', border: '1px solid rgba(52, 211, 153, 0.08)', borderLeft: '3px solid #34d399' }}>
            <span style={{ color: '#34d399', fontWeight: 'bold', fontSize: 10, display: 'block', letterSpacing: 0.5, marginBottom: 2 }}>
              阶段结论 STAGE CONCLUSION
            </span>
            <span style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.5 }}>{conclusion}</span>
          </div>
        )}
      </div>
    );
  }
  return <div style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.6 }}>{hint}</div>;
}

function DetailLayer({ detail, onClose }) {
  if (!detail) return null;

  return (
    <section
      style={{
        ...cardStyle,
        padding: 24,
        background: 'rgba(15, 23, 42, 0.55)',
        border: '1px solid rgba(103, 232, 249, 0.2)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
        backdropFilter: 'blur(16px)',
        borderRadius: 12,
        animation: 'slideDown 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        marginTop: 6
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 18, alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{
            color: '#22d3ee',
            fontSize: 9,
            fontWeight: 800,
            letterSpacing: 1.5,
            textTransform: 'uppercase',
            background: 'rgba(34, 211, 238, 0.08)',
            padding: '2px 6px',
            borderRadius: 4,
            display: 'inline-block',
            marginBottom: 6
          }}>
            研究详情 EXPLORATION DETAILS
          </div>
          <h3 style={{ margin: '2px 0 6px', color: '#fff', fontSize: 20, fontWeight: 800, lineHeight: 1.3, letterSpacing: -0.2 }}>{detail.title}</h3>
          {detail.text && <div style={{ color: '#cbd5e1', fontSize: 13.5, lineHeight: 1.7, maxWidth: 1020 }}>{detail.text}</div>}
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="detail-close-btn"
            style={{
              border: 'none',
              background: 'transparent',
              color: '#94a3b8',
              cursor: 'pointer',
              fontSize: 12,
              fontWeight: 'bold',
              padding: '4px 8px',
              transition: 'all 0.2s ease',
              borderBottom: '1px solid transparent'
            }}
          >
            收起详情 ↑
          </button>
        )}
      </div>

      {(detail.value || detail.valueHint) && (
        <div style={{
          marginTop: 14,
          display: 'grid',
          gridTemplateColumns: detail.value ? 'minmax(180px, 0.22fr) minmax(0, 1fr)' : '1fr',
          gap: 14
        }}>
          {detail.value && (
            <div style={{
              padding: '12px 16px',
              borderRadius: 8,
              background: 'rgba(15, 23, 42, 0.35)',
              border: '1px solid rgba(255, 255, 255, 0.04)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center'
            }}>
              <div style={{ color: '#64748b', fontSize: 10, fontWeight: 'bold', letterSpacing: 0.5, marginBottom: 4 }}>
                {detail.title?.includes('历史') || detail.value?.startsWith('Phase') ? '阶段 RANGE' : '当前状态 STATUS'}
              </div>
              <div style={{ color: '#fff', fontSize: 18, lineHeight: 1.2, fontWeight: 800, fontFamily: 'monospace' }}>{detail.value}</div>
            </div>
          )}
          {detail.valueHint && (
            <div style={{
              padding: '12px 16px',
              borderRadius: 8,
              background: 'rgba(15, 23, 42, 0.2)',
              border: '1px solid rgba(255, 255, 255, 0.03)',
              display: 'flex',
              alignItems: 'center',
              width: '100%'
            }}>
              {formatValueHint(detail.valueHint)}
            </div>
          )}
        </div>
      )}

      {(detail.items || []).length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10, marginTop: 14 }}>
          {detail.items.map((item, idx) => (
            <div key={item} style={{
              color: '#cbd5e1',
              fontSize: 13,
              lineHeight: 1.6,
              padding: '12px 14px',
              borderRadius: 8,
              background: 'rgba(15, 23, 42, 0.25)',
              border: '1px solid rgba(255, 255, 255, 0.03)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
            }}>
              <span style={{
                color: '#22d3ee',
                fontWeight: 'bold',
                fontFamily: 'monospace',
                fontSize: 11,
                background: 'rgba(34, 211, 238, 0.06)',
                padding: '2px 5px',
                borderRadius: 4,
                marginTop: 1
              }}>
                {String(idx + 1).padStart(2, '0')}
              </span>
              <span style={{ flex: 1 }}>{item}</span>
            </div>
          ))}
        </div>
      )}

      {(detail.formulas || []).length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 10, marginTop: 14 }}>
          {detail.formulas.map((formula) => (
            <div
              key={formula}
              style={{
                color: '#38bdf8',
                fontFamily: '"Fira Code", Monaco, Consolas, monospace',
                fontSize: 12,
                lineHeight: 1.6,
                padding: '12px 16px',
                borderRadius: 8,
                background: 'rgba(2, 6, 23, 0.5)',
                border: '1px solid rgba(56, 189, 248, 0.15)',
                position: 'relative',
                boxShadow: 'inset 0 1px 4px rgba(0,0,0,0.4)',
                overflowX: 'auto'
              }}
            >
              <div style={{
                position: 'absolute',
                top: 4,
                right: 10,
                color: 'rgba(56, 189, 248, 0.25)',
                fontSize: 8,
                fontWeight: 'bold',
                letterSpacing: 1,
                textTransform: 'uppercase'
              }}>
                Formula
              </div>
              {formula}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
export function AtlasControlDashboard({ lang = 'zh' }) {
  const isEnglish = lang === 'en';
  const [progress, setProgress] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [clientIndex, setClientIndex] = useState(null);
  const [error, setError] = useState('');
  const [activePlan, setActivePlan] = useState(null);
  const [activeLatest, setActiveLatest] = useState(null);
  const [activeHistory, setActiveHistory] = useState(null);

  useEffect(() => {
    let mounted = true;
    Promise.all([
      fetch(`${ATLAS_BASE}/progress.json`, { cache: 'no-store' }).then(async (res) => {
        if (!res.ok) throw new Error(`progress.json ${res.status}`);
        const summary = await res.json();
        const coverage = summary.client_stage_coverage || {};
        return {
          ...summary,
          ...coverage,
          last_phase: summary.last_phase,
          latest_phase: summary.last_phase,
          single_global_progress_percentage_valid: summary.single_global_progress_percentage_valid,
          source: `${ATLAS_BASE}/progress.json`,
        };
      }),
      fetch(`${ATLAS_BASE}/manifest.json`, { cache: 'no-store' }).then((res) => {
        if (!res.ok) throw new Error(`manifest.json ${res.status}`);
        return res.json();
      }),
      fetch(`${ATLAS_BASE}/client_index.json`, { cache: 'no-store' }).then((res) => (res.ok ? res.json() : null)),
    ])
      .then(([nextProgress, nextManifest, nextClientIndex]) => {
        if (!mounted) return;
        setProgress(nextProgress);
        setManifest(nextManifest);
        setClientIndex(nextClientIndex);
        setError('');
      })
      .catch((err) => {
        if (!mounted) return;
        setError(err?.message || '图谱进度数据读取失败');
      });
    return () => {
      mounted = false;
    };
  }, []);

  const progressMap = useMemo(() => {
    const map = {};
    progressRows.forEach((row) => {
      map[row.id] = clamp01(readProgressValue(progress, row.id, row.fallback));
    });
    return map;
  }, [progress]);

  const overall = useMemo(() => {
    const pattern = progressMap.pattern_family_atlas || 0;
    const physical = progressMap.physical_path_atlas || 0;
    const component = progressMap.component_path_atlas || 0;
    const readout = progressMap.readout_competition_trace || 0;
    const rollout = progressMap.stepwise_rollout_trace || 0;
    const closure = progressMap.causal_closure || 0;
    return (pattern * 0.18) + (physical * 0.18) + (component * 0.16) + (readout * 0.14) + (rollout * 0.14) + (closure * 0.2);
  }, [progressMap]);
  const overallValid = progress?.single_global_progress_percentage_valid !== false;

  const exactEventRows = useMemo(() => {
    const stage = progress?.exact_event_stage || {};
    const definitions = [
      ['audited_families', '精确事件覆盖模式族'],
      ['audited_registered_mechanisms', '精确事件覆盖注册机制'],
      ['single_path_qualified_models', '单样本路径合格模型'],
      ['instrument_conservation_models', '事件守恒合格模型'],
      ['signed_event_calibration_candidates', '有符号事件复制候选'],
      ['upstream_signed_event_candidates', '上游有符号事件候选'],
      ['function_specific_upstream_patterns', '功能特异上游模式'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const relationRows = useMemo(() => {
    const causalStage = progress?.causal_order_kv_stage;
    const stage = causalStage || progress?.multitime_relation_stage || {};
    const definitions = causalStage ? [
      ['physical_predictive_trajectories', '物理预测轨迹'],
      ['direct_computational_edges', '直接计算边'],
      ['causal_intervention_directions', '因果干预方向'],
      ['strict_answer_switches', '严格答案切换'],
      ['models_passing_query_gate', '查询状态门合格模型'],
      ['models_passing_margin_gate', '目标差值门合格模型'],
      ['models_passing_behavior_gate', '行为门合格模型'],
      ['crossmodel_source_specific_head_routes', '跨模型来源特异头路径'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ] : [
      ['behavior_eligible_mechanisms', '行为分母合格机制'],
      ['incremental_discovery_cases', '增量发现样本'],
      ['incremental_event_ledger_models', '增量事件账本模型'],
      ['physical_holdout_cases', '物理留出样本'],
      ['calibrated_predictive_relations', '校准预测关系'],
      ['physical_predictive_relations', '物理预测关系'],
      ['upstream_descriptive_predictive_relations', '上游描述预测关系'],
      ['physical_mlp_channel_relations', '物理 MLP 通道关系'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const latestSections = useMemo(() => {
    const sections = latestCards.map((item) => ({ ...item }));
    const latest = progress?.last_phase || progress?.latest_phase || manifest?.phase || 'Atlas v2';
    const overallLatest = sections.find((item) => item.id === 'overall_latest');
    if (overallLatest && progress?.exact_event_stage) {
      overallLatest.summary = '精确组件事件已可守恒回放，但功能特异的上游语言路径仍未出现。';
      overallLatest.value = '0/72 complete paths';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：精确事件图谱的严格边界`,
        text: '当前已建立单样本运行合同、精确注意力来源事件和 MLP 通道事件账本。复制结果仍集中在晚段终端接口；两个上游相消质量模式均未通过匹配特异性对照，因此不能提升为语言路径。',
        value: 'Exact events available / upstream path absent',
        valueHint: '九族是工程注册分母，不等于九族物理机制已完成。物理留出仍密封，CUDA 因果干预尚未授权。',
        items: exactEventRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.multitime_relation_stage) {
      overallLatest.summary = '真实增量生成路径中出现跨模型、跨词汇、物理留出可预测的关系，但证据仍以终端注意力关系为主。';
      overallLatest.value = '10 physical relations / 0 causal paths';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：增量多坐标关系图谱`,
        text: '五个语义坐标已按真实 KV-cache 生成路径采集。10 条关系通过一次性物理留出和错组、错时刻、错深度对照；其中 9 条属于目标词到后续词的终端延续，只有 1 条是晚深度的来源到查询注意力关系。当前没有 MLP 神经元通道关系，也没有因果必要性或完整语言路径。',
        value: 'Predictive descriptive relations / causal path absent',
        valueHint: '物理预测强于静态相关，但不能替代干预。物理留出已使用一次，不能重新挑选候选或阈值。',
        items: relationRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.causal_order_kv_stage) {
      overallLatest.summary = '预测轨迹已完成计算方向与 K/V 干预审计，未升级为因果路径。';
      overallLatest.value = '0 edges / 0 switches';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：计算方向与来源 K/V 因果审计`,
        text: 'Phase386 的 10 条关系保留为预测轨迹，但同层计算依赖审计判定 0 条是按原节点定义的直接边。全新的 96 条双向来源 K/V 干预没有一次切换到 donor 答案；头和来源分解虽复制出广泛活动，却没有三模型共同的来源锚点特异头路径。当前不新增神经元节点。',
        value: 'Predictive trajectories retained / causal route rejected',
        valueHint: '单位置粗 K/V 路线已经关闭。下一阶段需要新的多位置、多头、跨层联合状态合同，不能继续调层号或扩大单神经元扫描。',
        items: relationRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    const engineering = sections.find((item) => item.id === 'engineering_latest');
    if (engineering) {
      engineering.detail = {
        ...engineering.detail,
        items: overallValid
          ? progressRows.map((row) => `${row.label}：${pct(progressMap[row.id])}。${row.hint}`)
          : (relationRows.length ? relationRows : exactEventRows)
            .map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
        valueHint: `客户端视图数量：${clientIndex?.views?.length || 0}。最新阶段：${latest}。单一全局完成百分比：${overallValid ? '有效' : '无效'}。`,
      };
    }
    return sections;
  }, [clientIndex, exactEventRows, manifest, overallValid, progress, progressMap, relationRows]);

  const planDetail = useMemo(() => {
    return planCards.find((item) => item.id === activePlan)?.detail || null;
  }, [activePlan]);

  const latestDetail = useMemo(() => {
    return latestSections.find((item) => item.id === activeLatest)?.detail || null;
  }, [activeLatest, latestSections]);

  const historyDetail = useMemo(() => {
    const stage = historyStages.find((item) => item.id === activeHistory);
    if (!stage) return null;
    return {
      title: `${stage.range}：${stage.title}`,
      text: stage.summary,
      value: stage.range,
      valueHint: `核心问题：${stage.problem} 测试原理：${stage.principle} 阶段结论：${stage.conclusion}`,
      items: stage.phases,
      formulas: stage.formulas,
    };
  }, [activeHistory]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28, maxWidth: 1320, margin: '0 auto' }}>
      <style>{`
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-8px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .clickable-card {
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        .clickable-card:hover {
          transform: translateY(-2px) !important;
          border-color: rgba(255, 255, 255, 0.12) !important;
          box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25) !important;
        }
        .clickable-card:active {
          transform: translateY(0) !important;
        }
        .timeline-card {
          transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        .timeline-card:hover {
          transform: translateX(4px) !important;
          border-color: rgba(34, 211, 238, 0.35) !important;
        }
        .detail-close-btn:hover {
          color: #fff !important;
          border-bottom-color: #fff !important;
        }
        .progress-row-hover {
          transition: all 0.2s ease;
        }
        .progress-row-hover:hover {
          background: rgba(255, 255, 255, 0.04) !important;
          border-color: rgba(255, 255, 255, 0.08) !important;
          padding-left: 10px !important;
        }
      `}</style>

      {/* Main Header Board */}
      <section
        style={{
          ...cardStyle,
          padding: '24px 30px',
          background: 'rgba(15, 23, 42, 0.25)',
          border: '1px solid rgba(255, 255, 255, 0.04)',
          backdropFilter: 'blur(8px)',
          boxShadow: '0 12px 36px rgba(0,0,0,0.2)',
          borderRadius: 12,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
          <div style={{ maxWidth: 860 }}>
            <div style={{ color: '#22d3ee', fontSize: 10, fontWeight: 900, letterSpacing: 1.5, textTransform: 'uppercase' }}>
              {isEnglish ? 'Language Pattern Atlas Research' : 'Language Analysis / 语言分析'}
            </div>
            <h2 style={{ margin: '6px 0 6px', color: '#f8fafc', fontSize: 26, fontWeight: 800, lineHeight: 1.2, letterSpacing: -0.2 }}>
              语言模式图谱研究
            </h2>
            <div style={{ color: '#94a3b8', fontSize: 13.5, lineHeight: 1.6 }}>
            </div>
          </div>
          <div style={{
            minWidth: 200,
            padding: '12px 16px',
            borderRadius: 10,
            background: 'rgba(15, 23, 42, 0.35)',
            border: '1px solid rgba(255, 255, 255, 0.04)'
          }}>
            <div style={{ color: '#67e8f9', fontSize: 10, fontWeight: 'bold', letterSpacing: 0.8, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{ display: 'inline-block', width: 5, height: 5, borderRadius: '50%', background: '#67e8f9', boxShadow: '0 0 5px #67e8f9' }} />
              综合推进度 OVERALL
            </div>
            <div style={{ color: '#fff', fontSize: 32, fontWeight: 900, fontFamily: 'monospace', marginBottom: 6 }}>
              {overallValid ? pct(overall) : 'N/A'}
            </div>
            {overallValid && <ProgressBar value={overall} color="#67e8f9" />}
          </div>
        </div>
        {error && (
          <div style={{ marginTop: 12, color: '#fecaca', fontSize: 12 }}>
            {error}
          </div>
        )}
      </section>

      {/* Section 01: 图谱研究方案 */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <SectionHeader
          eyebrow="01"
          title="图谱研究方案"
          subtitle="根据《智能理论》中的语言模式图谱方案，说明 PatternPath、模式族、证据等级、闭合门、机制主张和可视化执行流程。点击卡片可查看对应规划详情。"
        />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
          {planCards.map((plan, index) => (
            <ClickableCard
              key={plan.id}
              active={activePlan === plan.id}
              onClick={() => setActivePlan(activePlan === plan.id ? null : plan.id)}
              activeColor="#22d3ee"
              style={{ padding: 18, minHeight: 136, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', borderRadius: 10 }}
            >
              <div>
                <div style={{ color: '#22d3ee', fontSize: 10, fontWeight: 900, fontFamily: 'monospace', opacity: 0.8, marginBottom: 6 }}>
                  STEP 0{index + 1}
                </div>
                <div style={{ color: '#f8fafc', fontSize: 15, fontWeight: 700, marginBottom: 4 }}>{plan.title}</div>
                <div style={{ color: '#cbd5e1', fontSize: 12.5, lineHeight: 1.55 }}>{plan.summary}</div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 10 }}>
                <span style={{ color: '#22d3ee', fontSize: 11, opacity: activePlan === plan.id ? 1 : 0.6, transition: 'opacity 0.2s' }}>
                  {activePlan === plan.id ? '收起详情 ▲' : '查看详情 ▼'}
                </span>
              </div>
            </ClickableCard>
          ))}
        </div>
        <DetailLayer detail={planDetail} onClose={() => setActivePlan(null)} />
      </section>

      {/* Section 02: 最新成果 */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <SectionHeader
          eyebrow="02"
          title="最新成果"
          subtitle="按当前合格科学分母显示精确事件覆盖、复制结果与未通过门槛；工程注册不等于物理路径完成。"
        />

        {/* Spacious Dashboard Control Center Board */}
        <div
          style={{
            ...cardStyle,
            padding: 22,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: 24,
            alignItems: 'center',
            background: 'rgba(15, 23, 42, 0.2)',
            border: '1px solid rgba(255, 255, 255, 0.04)',
            borderRadius: 12
          }}
        >
          {/* Left Overall Gauge */}
          <div style={{
            padding: '16px 20px',
            borderRadius: 10,
            background: 'rgba(15, 23, 42, 0.35)',
            border: '1px solid rgba(255, 255, 255, 0.03)',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            textAlign: 'center'
          }}>
            <div style={{ color: '#94a3b8', fontSize: 10, fontWeight: 'bold', letterSpacing: 1, marginBottom: 4 }}>OVERALL PROGRESS</div>
            <div style={{ color: '#67e8f9', fontSize: 38, lineHeight: 1, fontWeight: 900, fontFamily: 'monospace' }}>
              {overallValid ? pct(overall) : 'N/A'}
            </div>
            {overallValid && (
              <div style={{ marginTop: 10 }}>
                <ProgressBar value={overall} color="#67e8f9" />
              </div>
            )}
          </div>

          {/* Right Progress bars in 3 columns */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, gridColumn: 'span 2' }}>
            {(overallValid ? progressRows : (relationRows.length ? relationRows : exactEventRows)).map((row) => (
              <div
                key={row.id}
                className="progress-row-hover"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                  padding: '8px 10px',
                  borderRadius: 8,
                  background: 'rgba(255, 255, 255, 0.01)',
                  border: '1px solid rgba(255, 255, 255, 0.02)',
                }}
                title={row.hint}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
                  <span style={{ color: '#cbd5e1', fontSize: 12, fontWeight: 700 }}>{row.label}</span>
                  <span style={{ color: '#34d399', fontSize: 12, fontWeight: 'bold', fontFamily: 'monospace' }}>
                    {overallValid ? pct(progressMap[row.id]) : `${row.numerator}/${row.denominator}`}
                  </span>
                </div>
                {overallValid && <ProgressBar value={progressMap[row.id]} color="#a7f3d0" />}
              </div>
            ))}
          </div>
        </div>

        <EvidenceKernelDashboard />

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {latestSections.map((section) => (
            <ClickableCard
              key={section.id}
              active={activeLatest === section.id}
              onClick={() => setActiveLatest(activeLatest === section.id ? null : section.id)}
              activeColor="#34d399"
              style={{
                padding: '18px 20px',
                minHeight: 0,
                display: 'grid',
                gridTemplateColumns: '92px minmax(160px, 0.28fr) minmax(0, 1fr) minmax(120px, 0.16fr)',
                gap: 18,
                alignItems: 'center',
                borderRadius: 10,
              }}
            >
              <span style={{
                justifySelf: 'start',
                fontSize: 9,
                fontWeight: 'bold',
                textTransform: 'uppercase',
                color: section.id === 'gap_latest' ? '#fbbf24' : section.id === 'theory_latest' ? '#38bdf8' : '#34d399',
                background: section.id === 'gap_latest' ? 'rgba(251, 191, 36, 0.08)' : section.id === 'theory_latest' ? 'rgba(56, 189, 248, 0.08)' : 'rgba(52, 211, 153, 0.08)',
                padding: '4px 8px',
                borderRadius: 999
              }}>
                {section.id === 'gap_latest' ? 'GAP' : section.id === 'theory_latest' ? 'THEORY' : 'RESULT'}
              </span>
              <div style={{ color: '#f8fafc', fontSize: 15, fontWeight: 700 }}>{section.title}</div>
              <div style={{ color: '#cbd5e1', fontSize: 12.5, lineHeight: 1.55 }}>{section.summary}</div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 14 }}>
                <div style={{ color: '#bae6fd', fontSize: 11, fontWeight: 800, fontFamily: 'monospace', whiteSpace: 'nowrap' }}>{section.value}</div>
                <span style={{ color: '#34d399', fontSize: 11, opacity: activeLatest === section.id ? 1 : 0.6, transition: 'opacity 0.2s', whiteSpace: 'nowrap' }}>
                  {activeLatest === section.id ? '收起 ▲' : '详情 ▼'}
                </span>
              </div>
            </ClickableCard>
          ))}
        </div>
        <DetailLayer detail={latestDetail} onClose={() => setActiveLatest(null)} />
      </section>

      {/* Section 03: 历史记录 */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <SectionHeader
          eyebrow="03"
          title="历史记录"
          subtitle="按阶段摘要格式展示核心进展；点击阶段后展开具体问题、测试原理、公式和结论。"
        />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0, borderLeft: '2px solid rgba(34, 211, 238, 0.15)', marginLeft: 10 }}>
          {historyStages.map((stage, index) => (
            <ClickableCard
              key={stage.id}
              active={activeHistory === stage.id}
              onClick={() => setActiveHistory(activeHistory === stage.id ? null : stage.id)}
              className="timeline-card"
              style={{
                marginLeft: 28,
                marginBottom: index === historyStages.length - 1 ? 0 : 14,
                padding: 0,
                minHeight: 0,
                display: 'grid',
                gridTemplateColumns: '130px minmax(0, 1fr)',
                gap: 20,
                background: activeHistory === stage.id
                  ? 'rgba(15, 23, 42, 0.5)'
                  : 'rgba(15, 23, 42, 0.15)',
                border: activeHistory === stage.id
                  ? '1px solid #22d3ee'
                  : '1px solid rgba(255, 255, 255, 0.05)',
                boxShadow: activeHistory === stage.id
                  ? '0 8px 30px rgba(0, 0, 0, 0.3)'
                  : 'none',
                position: 'relative',
                borderRadius: 10,
                overflow: 'hidden'
              }}
            >
              <div style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: activeHistory === stage.id ? 3 : 1,
                background: activeHistory === stage.id
                  ? 'linear-gradient(to bottom, #22d3ee, #06b6d4)'
                  : 'linear-gradient(to bottom, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.02))',
                boxShadow: activeHistory === stage.id ? '0 0 6px #22d3ee' : 'none',
                transition: 'all 0.3s ease'
              }} />

              <span
                style={{
                  position: 'absolute',
                  left: -37,
                  top: 26,
                  width: 14,
                  height: 14,
                  borderRadius: '50%',
                  background: activeHistory === stage.id ? '#22d3ee' : '#0f172a',
                  border: activeHistory === stage.id ? '3px solid rgba(34, 211, 238, 0.3)' : '2px solid rgba(34, 211, 238, 0.25)',
                  boxShadow: activeHistory === stage.id ? '0 0 8px #22d3ee' : 'none',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  zIndex: 2,
                }}
              />

              <div style={{
                padding: '18px 0 18px 20px',
                color: '#34d399',
                fontSize: 12.5,
                fontWeight: 900,
                lineHeight: 1.4,
                fontFamily: 'monospace',
                letterSpacing: 0.5
              }}>
                {stage.range}
              </div>

              <div style={{ padding: '18px 20px 18px 0', display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ color: '#f8fafc', fontSize: 16, fontWeight: 700, lineHeight: 1.3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>{stage.title}</span>
                  <span style={{ color: '#22d3ee', fontSize: 11, fontWeight: 'normal', opacity: activeHistory === stage.id ? 1 : 0.6 }}>
                    {activeHistory === stage.id ? '收起 ▲' : '分析详情 ▼'}
                  </span>
                </div>
                <div style={{ color: '#cbd5e1', fontSize: 12.5, lineHeight: 1.6 }}>{stage.summary}</div>
                <div style={{
                  color: '#bae6fd',
                  fontSize: 12,
                  lineHeight: 1.5,
                  background: 'rgba(186, 230, 253, 0.02)',
                  padding: '6px 10px',
                  borderRadius: 6,
                  borderLeft: '2px solid rgba(186, 230, 253, 0.15)',
                  marginTop: 2
                }}>
                  <strong style={{ color: '#38bdf8', fontSize: 10.5, display: 'block', marginBottom: 1 }}>结论 CONCLUSION:</strong>
                  {stage.conclusion}
                </div>
              </div>
            </ClickableCard>
          ))}
        </div>
        <DetailLayer detail={historyDetail} onClose={() => setActiveHistory(null)} />
      </section>
    </div>
  );
}
