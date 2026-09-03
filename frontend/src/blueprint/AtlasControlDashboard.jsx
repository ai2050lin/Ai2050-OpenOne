import { useEffect, useMemo, useState } from 'react';
import { EvidenceKernelDashboard } from './EvidenceKernelDashboard';
import { researchAssetUrl } from '../config/researchAssets';

const ATLAS_BASE = researchAssetUrl('pattern_family_atlas/v2');

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
        <div className="atlas-detail-value-grid" style={{
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

  const jointFormationRows = useMemo(() => {
    const stage = progress?.joint_formation_stage || {};
    const definitions = [
      ['behavior_cases', '冻结行为样本'],
      ['behavior_eligible_mechanisms', '行为分母合格机制'],
      ['global_terminal_alignment_candidates', '全局终端方向候选'],
      ['physical_local_parent_layouts', '物理留出局部父节点布局'],
      ['joint_answer_switches', '联合父边界答案切换'],
      ['models_passing_joint_specificity', '联合路径特异性合格模型'],
      ['attribute_answer_switches', '独立属性内容答案切换'],
      ['structure_answer_switches', '结构位置答案切换'],
      ['random_answer_switches', '随机位置答案切换'],
      ['wrong_depth_attribute_switches', '错误深度属性答案切换'],
      ['models_passing_attribute_transport', '属性内容搬运合格模型'],
      ['models_passing_depth_specificity', '深度特异性合格模型'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const bindingSeparationRows = useMemo(() => {
    const stage = progress?.binding_separation_stage || {};
    const definitions = [
      ['formal_pointer_qualified_groups', '形式指针合格组'],
      ['natural_behavior_qualified_groups', '自然同词元集合合格组'],
      ['eligible_natural_surfaces', '行为合格任务面'],
      ['local_context_transport_cells', '局部上下文搬运单元'],
      ['crosssurface_shared_states', '跨任务共享静态状态'],
      ['field_physical_model_cells', '字段抽取物理复现模型'],
      ['field_same_literal_answer_switches', '同字面值上下文答案切换'],
      ['same_position_content_switches', '同位置内容答案切换'],
      ['abstract_binding_algorithms', '抽象绑定算法'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const factorSeparatedBindingRows = useMemo(() => {
    const stage = progress?.factor_separated_binding_stage || {};
    const definitions = [
      ['behavior_cases', '冻结行为案例'],
      ['qualified_parallel_groups', '三模型十条件完整合格组'],
      ['eligible_surfaces', '行为合格任务面'],
      ['discovery_observational_cells', '发现集关系签名单元'],
      ['calibration_observational_cells', '校准集关系签名单元'],
      ['physical_observational_cells', '物理留出关系签名单元'],
      ['causal_relation_context_cells', '因果关系载体单元'],
      ['relation_answer_switches', '纯关系上下文答案切换'],
      ['abstract_binding_algorithms', '抽象绑定算法'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const orderConditionedBindingRows = useMemo(() => {
    const stage = progress?.order_conditioned_joint_binding_stage || {};
    const definitions = [
      ['behavior_cases', '联合析因行为案例'],
      ['qualified_parallel_groups', '三模型十六条件完整合格组'],
      ['frozen_trace_cases', '冻结全深度追踪案例'],
      ['order_invariant_RQ_cells', '顺序不变 RQ 候选单元'],
      ['ROQ_discovery_cells', 'ROQ 发现单元'],
      ['ROQ_calibration_cells', 'ROQ 校准单元'],
      ['ROQ_physical_cells', 'ROQ 物理留出单元'],
      ['single_query_position_causal_cells', '单查询位置因果单元'],
      ['same_order_answer_switches', '同顺序联合状态答案切换'],
      ['abstract_binding_algorithms', '抽象绑定算法'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const multiPositionDynamicBindingRows = useMemo(() => {
    const stage = progress?.multi_position_dynamic_binding_stage || {};
    const definitions = [
      ['behavior_candidate_cases', '四任务候选行为案例'],
      ['qualified_parallel_groups', '三模型十六条件完整合格组'],
      ['eligible_surfaces', '行为合格任务面'],
      ['selected_trace_cases', '冻结动态追踪案例'],
      ['quality_trace_group_model_cells', '守恒合格组模型单元'],
      ['required_event_discovery_cells', '必需事件发现单元'],
      ['required_event_calibration_cells', '必需事件校准单元'],
      ['required_event_physical_cells', '必需事件物理留出单元'],
      ['ordered_chain_discovery_cells', '有序链发现单元'],
      ['ordered_chain_calibration_cells', '有序链校准单元'],
      ['ordered_chain_physical_cells', '有序链物理留出单元'],
      ['crossmodel_ordered_chain_surfaces', '跨模型有序链任务面'],
      ['joint_causal_intervention_authorized', '联合因果干预授权'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const dynamicPartialOrderRows = useMemo(() => {
    const stage = progress?.dynamic_partial_order_stage || {};
    const definitions = [
      ['behavior_candidate_cases', '冻结行为候选案例'],
      ['eligible_surfaces', '行为合格任务面'],
      ['discovery_quality_group_model_cells', '发现集守恒合格组模型单元'],
      ['discovery_partial_order_graph_cells', '发现集部分序图合格单元'],
      ['discovery_crossmodel_isomorphism_surfaces', '发现集跨模型同构任务面'],
      ['discovery_prediction_cells', '发现集预测门合格单元'],
      ['calibration_quality_group_model_cells', '校准集守恒合格组模型单元'],
      ['validated_crossmodel_partial_order_surfaces', '校准后跨模型部分序任务面'],
      ['physical_holdout_cases_consumed', '已使用物理留出案例'],
      ['joint_causal_intervention_authorized', '联合因果干预授权'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const localEdgeRows = useMemo(() => {
    const stage = progress?.local_edge_stage || {};
    const definitions = [
      ['behavior_candidate_cases', '冻结行为候选案例'],
      ['semantic_correct_cases', '语义答案正确案例'],
      ['complete_three_model_groups', '三模型完整行为组'],
      ['eligible_surfaces', '行为合格任务面'],
      ['batch_shape_all_field_matches', '批形状全字段一致案例'],
      ['instrument_quality_cases', '同形状账本合格案例'],
      ['discovery_model_cases', '局部边发现案例'],
      ['strict_local_edge_layers', '严格局部边合格层'],
      ['direct_local_physical_model_surface_cells', '特异直接物理边单元'],
      ['crossmodel_local_edge_surfaces', '跨模型局部边任务面'],
      ['calibration_cases_consumed', '已使用校准案例'],
      ['physical_holdout_cases_consumed', '已使用物理留出案例'],
      ['joint_causal_intervention_authorized', '联合因果干预授权'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const multiParentRows = useMemo(() => {
    const stage = progress?.multiparent_direct_child_stage || {};
    const definitions = [
      ['behavior_candidate_cases', '冻结行为候选案例'],
      ['semantic_correct_cases', '语义答案正确案例'],
      ['complete_three_model_groups', '三模型完整行为组'],
      ['eligible_surfaces', '行为合格任务面'],
      ['instrument_rows', '多父分区账本合格行'],
      ['discovery_model_cases', '多父发现案例'],
      ['true_base_joint_cells', '真实响应合格联合单元'],
      ['above_best_singleton_cells', '超过最佳单父联合单元'],
      ['all_control_specific_cells', '独立排除全部控制单元'],
      ['strict_local_joint_cells', '严格局部联合单元'],
      ['model_level_joint_parent_candidates', '模型级多父候选'],
      ['crossmodel_joint_parent_surfaces', '跨模型多父任务面'],
      ['calibration_cases_consumed', '已使用校准案例'],
      ['physical_holdout_cases_consumed', '已使用物理留出案例'],
      ['joint_causal_intervention_authorized', '联合因果干预授权'],
      ['complete_language_paths', '完整语言路径'],
      ['single_neuron_causal_paths', '单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const predictiveStateRows = useMemo(() => {
    const stage = progress?.predictive_state_stage || {};
    const definitions = [
      ['phase403_semantic_correct_cases', 'Phase403 状态操作正确案例'],
      ['phase403_crossmodel_candidate_families', 'Phase403 跨模型状态族'],
      ['phase404_finite_candidate_correct_cases', 'Phase404 有限候选正确案例'],
      ['phase404_natural_top_target_cases', 'Phase404 全词表目标首词'],
      ['phase404_crossmodel_candidate_families', 'Phase404 跨模型状态族'],
      ['phase405_finite_candidate_correct_cases', 'Phase405 有限候选正确案例'],
      ['phase405_natural_top_target_cases', 'Phase405 自然目标首词'],
      ['phase405_model_family_natural_group_pass_cells', 'Phase405 严格模型族单元'],
      ['phase405_crossmodel_candidate_families', 'Phase405 跨模型状态族'],
      ['physical_holdout_cases_consumed', '已使用物理留出'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const conditionedSequenceRows = useMemo(() => {
    const stage = progress?.conditioned_sequence_stage || {};
    const definitions = [
      ['formal_discovery_cases', 'Phase406 正式发现案例'],
      ['first_step_candidate_correct_cases', '首步候选集正确案例'],
      ['first_step_global_top_target_cases', '首步全词表目标案例'],
      ['H12_short_sequence_semantic_correct_cases', 'H12 短序列语义正确案例'],
      ['formal_group_passes', '正式组通过数'],
      ['crossmodel_candidate_families', '跨模型条件状态族'],
      ['calibration_cases_consumed', '已使用校准案例'],
      ['behavioral_holdout_cases_consumed', '已使用行为留出'],
      ['physical_holdout_cases_consumed', '已使用物理留出'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const eventHorizonRows = useMemo(() => {
    const stage = progress?.event_horizon_condition_response_stage || {};
    const definitions = [
      ['formal_discovery_cases', 'Phase407 正式发现案例'],
      ['semantic_correct_cases', '注册语义正确案例'],
      ['complete_response_cases', '完整响应案例'],
      ['fully_semantic_gated_groups', '表面/接口/历史/序列四门完整组'],
      ['crossmodel_candidate_families', '跨模型条件状态族'],
      ['calibration_cases_consumed', '已使用校准案例'],
      ['behavioral_holdout_cases_consumed', '已使用行为留出'],
      ['physical_holdout_cases_consumed', '已使用物理留出'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const partitionInterfaceRows = useMemo(() => {
    const stage = progress?.partition_interface_stage || {};
    const definitions = [
      ['formal_discovery_cases', 'Phase408 正式发现案例'],
      ['registered_response_cases', '注册响应案例'],
      ['allowed_response_cases', '允许响应案例'],
      ['condition_separation_groups', '全部条件可分组'],
      ['surface_lexical_stability_groups', '表面与词汇稳定组'],
      ['functional_partition_groups', '功能响应分区组'],
      ['discovery_crossmodel_candidate_families', '发现跨模型分区族'],
      ['calibration_crossmodel_candidate_families', '校准跨模型分区族'],
      ['behavioral_crossmodel_candidate_families', '行为留出跨模型分区族'],
      ['calibration_cases_consumed', '已使用校准案例'],
      ['behavioral_holdout_cases_consumed', '已使用行为留出'],
      ['physical_holdout_cases_consumed', '已使用物理留出'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const dynamicResponseProtocolRows = useMemo(() => {
    const stage = progress?.dynamic_response_protocol_stage || {};
    const definitions = [
      ['registered_abstract_cases', 'Phase409 协议注册案例'],
      ['future_model_prompt_hashes', '未来三模型提示哈希'],
      ['dual_rule_engine_scenarios', '双机器规则一致场景'],
      ['query_contracts', '冻结查询合同'],
      ['independent_human_rule_review', '独立外部规则复核'],
      ['incremental_collector_token_equivalence', '增量采集器逐词元等价'],
      ['model_qualification_cases_consumed', '已使用模型资格案例'],
      ['physical_holdout_cases_consumed', '已使用物理留出'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const orthogonalDynamicPreflightRows = useMemo(() => {
    const stage = progress?.orthogonal_dynamic_preflight_stage || {};
    const definitions = [
      ['orthogonal_state_axes', 'Phase410 正交状态轴'],
      ['h3_order_variants_machine_audited', 'h3 顺序反转机器审计'],
      ['finite_grammar_cases_machine_audited', '语法有限全集机器审计'],
      ['independent_external_reviewers', '独立外部规则审阅者'],
      ['sealed_model_collector_equivalence', '密封模型采集器等价'],
      ['model_cases_consumed', '已使用模型案例'],
      ['physical_cases_consumed', '已使用物理案例'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const finiteSemanticOperationRows = useMemo(() => {
    const stage = progress?.finite_semantic_operation_preflight_stage || {};
    const definitions = [
      ['finite_semantic_cases_machine_audited', 'Phase411 有限语义合同机器审计'],
      ['semantic_only_registered_resolutions', '仅有限语义通道解析案例'],
      ['registered_operations_with_inverse', '具有双侧逆的注册操作'],
      ['operation_compositions_machine_audited', '操作组合闭包机器审计'],
      ['history_covariance_cases_machine_audited', '历史规则协变机器审计'],
      ['independent_external_reviewers', '独立外部规则审阅者'],
      ['accepted_external_review_items', '双人一致且符合注册表的条目'],
      ['sealed_model_collector_equivalence', '密封模型采集器等价'],
      ['model_cases_consumed', '已使用模型案例'],
      ['physical_cases_consumed', '已使用物理案例'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const typedObserverQuotientRows = useMemo(() => {
    const stage = progress?.typed_observer_quotient_preflight_stage || {};
    const definitions = [
      ['typed_observer_covariance_cells', 'Phase412 类型化观察者协变'],
      ['fixed_failures_explained_by_role_transport', '固定角色失败的角色运输解释'],
      ['observer_action_compositions', '观察者作用组合审计'],
      ['finite_partitions_exhausted', '有限状态分区穷举'],
      ['global_nontrivial_qualifying_partitions', '全局合格非平凡商'],
      ['external_role_indexed_partition_bundles', '外部角色索引分区束'],
      ['registered_irreversible_operations', '已注册不可逆操作'],
      ['registered_cross_family_bridges', '已注册跨族桥'],
      ['independent_external_reviewers', '独立外部规则审阅者'],
      ['sealed_model_collector_equivalence', '密封模型采集器等价'],
      ['model_cases_consumed', '已使用模型案例'],
      ['physical_cases_consumed', '已使用物理案例'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const predictionKernelMeasurementRows = useMemo(() => {
    const stage = progress?.prediction_kernel_measurement_preflight_stage || {};
    const definitions = [
      ['source_claims_audited', 'Phase413 材料主张审计'],
      ['terminal_identical_synthetic_paths', '终端相同的有限轨迹'],
      ['endpoint_identical_internal_distinct_pairs', '端点相同但中间不同的轨迹对'],
      ['one_step_equal_future_different_pairs', '一步相同但未来不同的状态对'],
      ['native_output_channel_permutation_invariance', '通道置换下原生输出不变'],
      ['fixed_coordinate_probe_counterexamples', '固定通道读数反例'],
      ['candidate_panel_contract_cases', '定长多轴候选面板合同'],
      ['qualified_direct_layer_local_probability_readouts', '合格的层内局部概率读出'],
      ['independent_external_reviewers', '独立外部规则审阅者'],
      ['sealed_model_collector_equivalence', '密封模型采集器等价'],
      ['model_cases_consumed', '已使用模型案例'],
      ['physical_cases_consumed', '已使用物理案例'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
    ];
    return definitions.flatMap(([id, label]) => {
      const value = stage[id];
      if (!value || !Number.isFinite(value.numerator) || !Number.isFinite(value.denominator)) return [];
      return [{ id, label, numerator: value.numerator, denominator: value.denominator }];
    });
  }, [progress]);

  const observerIndexedEventRows = useMemo(() => {
    const stage = progress?.observer_indexed_event_preflight_stage || {};
    const definitions = [
      ['supplied_claims_audited', 'Phase414 材料主张审计'],
      ['mixed_catalog_items_classified', '混合证据目录分类'],
      ['natural_complete_state_replay_identity', '完整自然状态续跑恒等'],
      ['layerwise_terminal_kernel_variation', '层间终端核变化'],
      ['incomplete_state_counterexamples', '不完整状态续跑反例'],
      ['observer_indexed_cells', '观察者索引读数单元'],
      ['variable_length_event_panel', '可变长度事件面板合同'],
      ['invalid_prefix_panel_rejected', '前缀冲突面板拒绝'],
      ['cross_tokenizer_semantic_alignment', '跨 tokenizer 语义事件对齐'],
      ['qualified_observers', '合格中间观察者'],
      ['independent_external_reviewers', '独立外部规则审阅者'],
      ['sealed_model_collector_equivalence', '密封模型采集器等价'],
      ['model_cases_consumed', '已使用模型案例'],
      ['new_physical_paths', '新增物理路径'],
      ['new_single_neuron_causal_paths', '新增单神经元因果路径'],
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
    if (overallLatest && progress?.joint_formation_stage) {
      overallLatest.summary = '局部父节点布局已跨模型复现，属性内容可受控搬运；联合形成与深度特异路径仍未建立。';
      overallLatest.value = '71/72 transport / 0/72 paths';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：结构布局与内容搬运已分离`,
        text: 'Phase390 否定了跨层方向必须保持到终端的线性假设。Phase391 在三模型发现、校准和物理留出中复制出一个图合法的局部父节点布局。Phase392 的联合干预虽产生 139/144 次答案切换，却没有通过属性单独干预和错误深度对照。Phase393 在完全独立组上确认属性位置可搬运内容，但错误深度同样全部切换，因此它不是特定中层的自然语言路径。',
        value: 'Local structure replicated / content transport causal / depth route absent',
        valueHint: '客户端只提升“受控属性内容搬运”因果边，不提升多来源联合路径、深度特异路径、完整语言路径或神经元路径。当前内部测试只覆盖通过严格行为分母的 field_extraction，不能外推到九族。',
        items: jointFormationRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.binding_separation_stage) {
      overallLatest.summary = '字段抽取的同字面值上下文载体已在三模型物理留出复现；跨任务共享绑定规则仍未建立。';
      overallLatest.value = '46/72 context switches / 0 binding rules';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：内容、上下文与绑定规则分离`,
        text: 'Phase394 的形式指针接口没有形成三模型共同分母。Phase395 改用同一词元集合、仅交换关系排列的自然样本，在约 57% 相对深度发现并校准了上下文差分；同字面值状态搬运仅有 4/6 模型任务单元通过，跨任务共享门失败。Phase396 在未见字段抽取组上取得三模型物理复现，但同位置内容搬运仍更强。',
        value: 'Field-specific contextual carrier replicated / abstract binding absent',
        valueHint: '3D 只新增聚合词元状态锚点，不新增注意力头或 MLP 神经元。供体充分性不等于自然必要性，字段抽取结果不能外推到九族。',
        items: bindingSeparationRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.factor_separated_binding_stage) {
      overallLatest.summary = '关系相关签名已跨三任务、三模型和三分割复现，但隔离关系上下文不是可移植因果载体。';
      overallLatest.value = '27/27 observed cells / 0/9 causal cells';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：关系签名与关系载体分离`,
        text: 'Phase397 冻结六类自然任务、十个独立控制和 4,320 个三模型行为案例。拥有关系、角色填槽和指代解析通过严格行为门；同位置值状态的关系差异在发现、校准、物理留出三份数据中均为 9/9，但 144 个隔离关系上下文搬运方向没有一次切换答案，9 个因果单元全部失败。内容身份和整句顺序状态的作用明显更大。',
        value: 'Stable relation signature / portable causal carrier rejected',
        valueHint: '3D 蓝色锚点只表示聚合关系签名，不是神经元。Phase396 字段结果保留为旧对比下的局部充分性，但不能外推为纯关系载体；因果校准、物理因果和单神经元扫描均关闭。',
        items: factorSeparatedBindingRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.order_conditioned_joint_binding_stage) {
      overallLatest.summary = '顺序条件化 ROQ 联合轨迹完成三分割复制；单查询位置状态不是充分因果载体。';
      overallLatest.value = '27/27 observed cells / 0/9 causal cells';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：联合轨迹与单位置载体分离`,
        text: 'Phase398 冻结三个任务面、三模型、16 条件完整析因和 3,456 个行为案例。顺序不变 RQ 候选为 0/9；中层查询末端 ROQ 轨迹在发现、校准、物理留出均为 9/9，且跨两套词汇方向复现。随后 432 个父边界方向只产生 10 次同顺序答案切换，单位置因果单元为 0/9。',
        value: 'Order-conditioned trajectory replicated / single-position carrier rejected',
        valueHint: '3D 粉色锚点表示聚合顺序条件化轨迹，不是神经元或完整绑定规则。下一阶段需要多位置、注意力来源边和 MLP 写入的动态联合路径合同。',
        items: orderConditionedBindingRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.multi_position_dynamic_binding_stage) {
      overallLatest.summary = '三类聚合动态事件完成三分割复制；冻结有序链仅在 DS7B 角色填槽成立，跨模型与因果门关闭。';
      overallLatest.value = '3/27 ordered cells / 0/3 crossmodel surfaces';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：多位置动态事件与模型特异链`,
        text: 'Phase399 在四个新任务面上运行 5,376 个三模型行为案例，字段抽取因只有 1/28 个完整组而在追踪前排除。其余三个任务面的 2,880 个冻结案例覆盖来源角色、查询末端、首答案、目标完成和答案后时刻。三类必需事件在发现、校准、物理留出均为 9/9；但“来源路由峰值不晚于查询整合、再不晚于终端”的整链只在 DS7B 角色填槽以第 10、10、20 层三次复现。',
        value: 'Aggregate events replicated / one model-specific ordered chain / causal gate closed',
        valueHint: '3D 仅显示三枚 DS7B 聚合事件锚点，均不是头、通道或神经元。跨模型任务为 0/3，因此没有运行联合损伤、分层恢复或细粒度扫描；完整语言路径仍为 0/72。',
        items: multiPositionDynamicBindingRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.dynamic_partial_order_stage) {
      overallLatest.summary = '区间部分序候选在发现集出现，但未通过答案预测门；校准集又因批形状重放失败失去资格。';
      overallLatest.value = '5/6 discovery graphs / 0/6 prediction cells';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：区间部分序候选与严格门控结果`,
        text: 'Phase400 在全新 4,608 个三模型行为案例上，以事件区间、相对起点、持续时间和部分序边替代单峰层号。发现集有 5/6 个模型任务单元形成冻结图，拥有关系在三模型间出现一个仅限发现集的同构候选；但 6/6 单元均未通过答案预测和对照门。校准收集的 384 个案例中有 1 个 DS7B 案例因 batch=8 与 batch=1 的首格式词元不同而违反冻结重放合同，因此校准分析、物理留出、因果干预和神经元扫描全部关闭。',
        value: 'Discovery-only partial order / prediction negative / calibration invalid',
        valueHint: '3D 不新增神经元、头或通道节点。发现图只用于显示聚合候选及失败边界，不能标记为已验证路径；物理留出仍为 0/384，完整语言路径仍为 0/72。',
        items: dynamicPartialOrderRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.local_edge_stage) {
      overallLatest.summary = '同执行形状账本已经通过；真实关系替换产生局部响应，但未超过全部反事实控制。';
      overallLatest.value = '96/96 ledger / 0/208 local-edge layers';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：执行同构、语义跨度与局部边审计`,
        text: 'Phase401 在全新 4,608 个三模型行为案例上统一使用 batch=1，并把格式前缀、语义答案、格式后缀和停止分开记录。独立批敏感性样本有 7/192 个案例发生至少一个字段变化；同形状内部账本为 96/96。三模型共采集 239,616 条真实关系替换与八类控制响应。真实替换能在部分早层恢复注意力状态，但三模型两个任务的 208 个模型任务层都没有同时通过组门、语义门和八类控制，直接特异物理边为 0/6。',
        value: 'Exact ledger / observable response / functional edge rejected',
        valueHint: '同目标控制与原语义差值门存在预注册矛盾；严格审计和不具放行权的 N/A 敏感性审计均为 0 层。校准与物理留出均未使用，3D 不新增头、通道或神经元节点。',
        items: localEdgeRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.multiparent_direct_child_stage) {
      overallLatest.summary = '四类真实 K/V 父分区已完成全子集审计；出现 8 个局部联合单元，但没有模型级或跨模型候选。';
      overallLatest.value = '8/13,728 local cells / 0 model candidates';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：多父条件状态与直接子状态审计`,
        text: 'Phase402 冻结六个任务面、6,912 个三模型行为案例，其中 5,585 例语义正确，只有角色填槽和条件存在满足三分割行为门。四类互斥且守恒的真实注意力 K/V 父分区通过 3,328/3,328 条重放账本。发现集保留 2,875,392 条私有配对指标，并把 13,728 个联合组层子集依次与真实响应门、最佳包含单父和每项适用控制比较。最终只有 8 个早层 source_structure + query_local 局部单元同时通过，Qwen3 为 0、GLM4 为 1、DS7B 为 7；模型级、双模型和跨模型候选均为 0。',
        value: 'Exact parent ledger / local hint only / model candidate rejected',
        valueHint: '冻结小数阈值 0.666666667 在 6 组分母上实际要求 5/6，本轮按原合同执行且不事后放宽。校准和物理留出均未使用，终端生成、自然必要性、因果闭合及神经元扫描全部关闭；3D 只显示聚合局部迹象，不新增具体头、通道或神经元。',
        items: multiParentRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.predictive_state_stage) {
      overallLatest.summary = '三种预测状态协议已完成三模型发现集；行为响应丰富，但没有稳定跨模型状态族，物理与神经元门保持关闭。';
      overallLatest.value = '0/3 state families / 0 physical paths';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：预测状态新范式与自然未来分支审计`,
        text: 'Phase403 将有限状态与更新操作放入统一未来面板，发现更新执行会混淆状态表示。Phase404 改为直接终态，在 2,880 例中有 2,597 例有限候选正确，但只有 1,529 例全词表首词为目标。Phase405 删除选择指令并使用自然未完成分支，有限候选正确 2,076/2,880、自然目标首词 1,266/2,880；九个模型×语言族单元的严格组通过数全部为零。',
        value: 'Response ledgers available / predictive equivalence rejected',
        valueHint: '语义真值边只表示测试世界规则，不是模型内部算子。校准、行为留出和物理留出均未消耗；3D 不新增层、头、通道或神经元路径。',
        items: predictiveStateRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.conditioned_sequence_stage) {
      overallLatest.summary = '条件化短序列修复了大量首步误判，但正式组门和跨模型门仍未通过。';
      overallLatest.value = '3,512/5,760 sequences / 0/3 cross-model';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：条件化短序列响应表与接口迁移审计`,
        text: 'Phase406 在三类语言族、三模型、四个表面、六种未来条件上完成 5,760 个正式发现案例。首步候选集正确 4,490 例、首步全词表目标正确 1,722 例，而 H12 短序列语义正确达到 3,512 例，其中 1,791 例属于首步全词表错误但短序列恢复。72 个正式组仅 Qwen3 知识绑定通过 5 个，未达到预注册的 6/8 模型族门；其余模型族全部为零，跨模型候选为 0/3。',
        value: 'Sequence recovery observed / conditioned state rejected',
        valueHint: '宽松词汇上界和 H48 诊断都没有改变停止决定。校准、行为留出、物理留出、直接算子、因果干预和神经元扫描全部保持关闭；3D 不新增物理路径或神经元节点。',
        items: conditionedSequenceRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.event_horizon_condition_response_stage) {
      overallLatest.summary = '语义、句界和停止事件已经分离；观测响应分区存在，但九个模型族单元均未形成稳定状态。';
      overallLatest.value = '3,935/5,760 semantic / 0/3 cross-model';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：事件地平线条件响应核与接口迁移审计`,
        text: 'Phase407 使用全新语义组，在三类语言族、三模型、四个表面、两种族特异接口和两种历史携带方式上完成 5,760 个 batch=1 正式发现案例。注册语义正确 3,935 例、完整响应 3,933 例；1,728 个条件响应单元中有 894 个恒等映射、43 个完整非恒等置换、302 个状态塌缩和 489 个缺失。只有 10/108 个正式组同时通过表面、接口、历史和序列门，九个模型×语言族单元均未达到 9/12 门，跨模型候选为 0/3。',
        value: 'Event ledger available / conditioned state rejected',
        valueHint: 'GLM4 有 129 条 FP16 非有限感叹号退化路径，全部到达 H48，已作为运行时警告而非语言机制解释。校准、行为留出和物理留出均未消费；3D 不新增物理路径、头、通道或神经元。',
        items: eventHorizonRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.partition_interface_stage) {
      const stage = progress.partition_interface_stage;
      const discoveryCases = stage.formal_discovery_cases?.denominator || 0;
      const functionalGroups = stage.functional_partition_groups?.numerator || 0;
      const discoveryFamilies = stage.discovery_crossmodel_candidate_families?.numerator || 0;
      const calibrationFamilies = stage.calibration_crossmodel_candidate_families?.numerator || 0;
      const behavioralFamilies = stage.behavioral_crossmodel_candidate_families?.numerator || 0;
      overallLatest.summary = '排他响应合同已经冻结并完成三模型审计；行为响应分区仍未被晋升为内部状态或物理路径。';
      overallLatest.value = `${functionalGroups}/108 partitions / ${behavioralFamilies}/3 behavioral`;
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：排他响应分区与接口协变审计`,
        text: `Phase408 使用三类语言族、三个模型、三种族特异接口、四个结构表面和两个独立词汇复本，完成 ${discoveryCases} 个 batch=1 正式发现案例。语义类别、运行时数值、响应事件和停止右删失分别记账；功能分区通过 ${functionalGroups}/108 个组，发现、校准和行为留出的严格跨模型候选依次为 ${discoveryFamilies}/3、${calibrationFamilies}/3、${behavioralFamilies}/3。`,
        value: 'Exclusive response ledger / observational partitions only',
        valueHint: '接口坐标图是测试协议定义，不是已发现的模型内部算子；闭环映射复用同一批状态行，不作为独立因果证据。历史固定为空，因此物理留出、因果干预和神经元扫描不开放，3D 保留原有网络形状且不新增具体神经元。',
        items: partitionInterfaceRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.dynamic_response_protocol_stage) {
      const stage = progress.dynamic_response_protocol_stage;
      const abstractCases = stage.registered_abstract_cases?.numerator || 0;
      const promptHashes = stage.future_model_prompt_hashes?.numerator || 0;
      const ruleAgreement = stage.dual_rule_engine_scenarios?.numerator || 0;
      const ruleDenominator = stage.dual_rule_engine_scenarios?.denominator || 0;
      const externalReview = stage.independent_human_rule_review?.numerator || 0;
      const collectorEquivalence = stage.incremental_collector_token_equivalence?.numerator || 0;
      const modelCases = stage.model_qualification_cases_consumed?.numerator || 0;
      overallLatest.summary = '动态响应与历史条件协议已经冻结；当前展示的是可审计协议准备度，不是模型行为或内部物理机制。';
      overallLatest.value = `${ruleAgreement}/${ruleDenominator} machine rules / ${modelCases} model cases`;
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：动态响应过程与历史条件协议`,
        text: `Phase409 注册 ${abstractCases} 个抽象案例和 ${promptHashes} 个未来三模型提示哈希，覆盖五种历史条件、完整句格式以及知识族三次单实体查询的联合签名。闭式规则与有限世界穷举在 ${ruleAgreement}/${ruleDenominator} 个场景上一致。`,
        value: 'Protocol frozen / model execution closed',
        valueHint: `机器双求解不等于独立规则复核；外部复核为 ${externalReview}/1，增量采集器逐词元等价为 ${collectorEquivalence}/1，模型案例为 ${modelCases}。3D 保留原网络形状，不新增物理路径、头、通道或神经元。`,
        items: dynamicResponseProtocolRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.orthogonal_dynamic_preflight_stage) {
      const stage = progress.orthogonal_dynamic_preflight_stage;
      const axes = stage.orthogonal_state_axes?.numerator || 0;
      const h3 = stage.h3_order_variants_machine_audited || {};
      const grammar = stage.finite_grammar_cases_machine_audited || {};
      const reviewers = stage.independent_external_reviewers || {};
      const collector = stage.sealed_model_collector_equivalence || {};
      const modelCases = stage.model_cases_consumed?.numerator || 0;
      overallLatest.summary = '动态响应已改为六轴正交账本；机器预检通过，但外部复核和真实模型采集器门仍关闭。';
      overallLatest.value = `${h3.numerator || 0}/${h3.denominator || 0} h3 / ${modelCases} model cases`;
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：正交动态状态与执行资格预检`,
        text: `Phase410 将语义、格式、句界、停止、数值和响应角色拆成 ${axes} 个可并存状态轴；完成 ${h3.numerator || 0}/${h3.denominator || 0} 个 h3 顺序变体与 ${grammar.numerator || 0}/${grammar.denominator || 0} 个冻结语法响应案例的机器审计。`,
        value: 'Machine preflight passed / scientific execution closed',
        valueHint: `独立外部审阅者为 ${reviewers.numerator || 0}/${reviewers.denominator || 0}，密封真实模型采集器等价为 ${collector.numerator || 0}/${collector.denominator || 0}。当前模型、物理、因果和神经元案例均为零；3D 保留原网络形状。`,
        items: orthogonalDynamicPreflightRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.finite_semantic_operation_preflight_stage) {
      const stage = progress.finite_semantic_operation_preflight_stage;
      const semantic = stage.finite_semantic_cases_machine_audited || {};
      const semanticOnly = stage.semantic_only_registered_resolutions || {};
      const operations = stage.registered_operations_with_inverse || {};
      const compositions = stage.operation_compositions_machine_audited || {};
      const covariance = stage.history_covariance_cases_machine_audited || {};
      const reviewers = stage.independent_external_reviewers || {};
      const accepted = stage.accepted_external_review_items || {};
      const collector = stage.sealed_model_collector_equivalence || {};
      overallLatest.summary = '有限语义与操作合同已冻结；固定观察者粗分区出现失败，执行门关闭。';
      overallLatest.value = '1.21M finite / 0 model';
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：有限语义、操作闭包与独立复核门`,
        text: `Phase411 保留严格整段精确通道，并在 ${semantic.numerator || 0}/${semantic.denominator || 0} 个冻结模板案例上增加有限注册语义通道，其中 ${semanticOnly.numerator || 0} 例只能由该有限通道解析。三族 ${operations.numerator || 0}/${operations.denominator || 0} 个外部状态操作均有双侧逆；操作组合 ${compositions.numerator || 0}/${compositions.denominator || 0}、历史规则协变 ${covariance.numerator || 0}/${covariance.denominator || 0} 均通过机器审计。`,
        value: 'Finite protocol closed / model semantics and physics untested',
        valueHint: `这些操作属于测试世界，不是模型内部算子。独立审阅者为 ${reviewers.numerator || 0}/${reviewers.denominator || 0}，双人接受条目为 ${accepted.numerator || 0}/${accepted.denominator || 0}，密封真实模型采集器等价为 ${collector.numerator || 0}/${collector.denominator || 0}；3D 保留原网络形状，不新增物理路径或神经元。`,
        items: finiteSemanticOperationRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.typed_observer_quotient_preflight_stage) {
      const stage = progress.typed_observer_quotient_preflight_stage;
      const covariance = stage.typed_observer_covariance_cells || {};
      const explained = stage.fixed_failures_explained_by_role_transport || {};
      const compositions = stage.observer_action_compositions || {};
      const partitions = stage.finite_partitions_exhausted || {};
      const globalQuotients = stage.global_nontrivial_qualifying_partitions || {};
      const bundles = stage.external_role_indexed_partition_bundles || {};
      const irreversible = stage.registered_irreversible_operations || {};
      const bridges = stage.registered_cross_family_bridges || {};
      const reviewers = stage.independent_external_reviewers || {};
      const collector = stage.sealed_model_collector_equivalence || {};
      overallLatest.summary = '固定查询角色的失败已校准为角色运输问题；全局非平凡状态商仍未建立。';
      overallLatest.value = `${partitions.numerator || 0} partitions / 0 model`;
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：类型化观察者协变与非平凡状态商穷举`,
        text: `Phase412 复现 Phase411 的 ${explained.denominator || 0} 个固定查询角色失败，并确认其中 ${explained.numerator || 0} 个都由实体重命名时未同步运输查询角色造成。状态、查询角色和响应类联合变换后，协变单元通过 ${covariance.numerator || 0}/${covariance.denominator || 0}，观察者作用组合通过 ${compositions.numerator || 0}/${compositions.denominator || 0}。有限世界的 ${partitions.numerator || 0}/${partitions.denominator || 0} 个分区已穷举。`,
        value: `Global nontrivial quotient ${globalQuotients.numerator || 0}/${globalQuotients.denominator || 0}`,
        valueHint: `知识族有 ${bundles.numerator || 0} 个外部角色索引分区束，但它不是模型状态。不可逆操作 ${irreversible.numerator || 0}/${irreversible.denominator || 0}、跨族桥 ${bridges.numerator || 0}/${bridges.denominator || 0}、外部审阅者 ${reviewers.numerator || 0}/${reviewers.denominator || 0}、密封采集器 ${collector.numerator || 0}/${collector.denominator || 0}；3D 形状不变，不新增物理路径或神经元。`,
        items: typedObserverQuotientRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.prediction_kernel_measurement_preflight_stage) {
      const stage = progress.prediction_kernel_measurement_preflight_stage;
      const claims = stage.source_claims_audited || {};
      const paths = stage.terminal_identical_synthetic_paths || {};
      const pairs = stage.endpoint_identical_internal_distinct_pairs || {};
      const future = stage.one_step_equal_future_different_pairs || {};
      const nativeOutput = stage.native_output_channel_permutation_invariance || {};
      const fixedProbe = stage.fixed_coordinate_probe_counterexamples || {};
      const localReadouts = stage.qualified_direct_layer_local_probability_readouts || {};
      const reviewers = stage.independent_external_reviewers || {};
      const collector = stage.sealed_model_collector_equivalence || {};
      overallLatest.summary = '终端预测核可以精确定义，但中间候选轨迹尚无合格的层内局部概率读出。';
      overallLatest.value = `${paths.numerator || 0} terminal paths / ${localReadouts.numerator || 0} local readout`;
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：终端预测核与中间候选轨迹测量资格`,
        text: `Phase413 审计 ${claims.numerator || 0}/${claims.denominator || 0} 条材料主张。构造的 ${paths.numerator || 0}/${paths.denominator || 0} 条轨迹拥有相同起点和终端分布，但 ${pairs.numerator || 0}/${pairs.denominator || 0} 条轨迹对的中间过程不同；另有 ${future.numerator || 0}/${future.denominator || 0} 个有限状态对在当前一步分布相同而两步未来不同。`,
        value: `Qualified direct layer-local probability readout ${localReadouts.numerator || 0}/${localReadouts.denominator || 0}`,
        valueHint: `MLP 通道置换保持原生输出 ${nativeOutput.numerator || 0}/${nativeOutput.denominator || 0}，却产生固定通道读数反例 ${fixedProbe.numerator || 0}/${fixedProbe.denominator || 0}。这些都是协议和有限反例，不是模型物理路径。外部审阅者 ${reviewers.numerator || 0}/${reviewers.denominator || 0}、密封采集器 ${collector.numerator || 0}/${collector.denominator || 0}；3D 形状不变。`,
        items: predictionKernelMeasurementRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    if (overallLatest && progress?.observer_indexed_event_preflight_stage) {
      const stage = progress.observer_indexed_event_preflight_stage;
      const catalog = stage.mixed_catalog_items_classified || {};
      const replay = stage.natural_complete_state_replay_identity || {};
      const variation = stage.layerwise_terminal_kernel_variation || {};
      const incomplete = stage.incomplete_state_counterexamples || {};
      const observerCells = stage.observer_indexed_cells || {};
      const events = stage.variable_length_event_panel || {};
      const semanticAlignment = stage.cross_tokenizer_semantic_alignment || {};
      const qualified = stage.qualified_observers || {};
      const reviewers = stage.independent_external_reviewers || {};
      const collector = stage.sealed_model_collector_equivalence || {};
      overallLatest.summary = '完整自然状态续跑是终端恒等检查，不是层间候选收缩曲线；观察者索引与事件合同已冻结。';
      overallLatest.value = `${replay.numerator || 0}/${replay.denominator || 0} identity / ${qualified.numerator || 0} observer`;
      overallLatest.detail = {
        ...overallLatest.detail,
        title: `${latest}：自然状态续跑恒等、观察者索引与事件合同`,
        text: `Phase414 在 ${replay.numerator || 0}/${replay.denominator || 0} 个有限完整状态续跑单元中复现同一终端核，${variation.numerator || 0}/${variation.denominator || 0} 个案例出现层间终端核变化；删去角色、历史或缓存后则有 ${incomplete.numerator || 0}/${incomplete.denominator || 0} 个反例。完整状态续跑因此只用于验证采集器和 checkpoint 一致性。观察者索引读数记录 ${observerCells.numerator || 0}/${observerCells.denominator || 0}，不能称为模型原生中间概率。`,
        value: `Qualified observer ${qualified.numerator || 0}/${qualified.denominator || 0}`,
        valueHint: `${catalog.numerator || 0}/${catalog.denominator || 0} 项目录已按证据类型分类，但不是完成百分比分母。可变长度事件合同 ${events.numerator || 0}/${events.denominator || 0}，跨 tokenizer 语义事件对齐 ${semanticAlignment.numerator || 0}/${semanticAlignment.denominator || 0}。外部审阅者 ${reviewers.numerator || 0}/${reviewers.denominator || 0}、密封采集器 ${collector.numerator || 0}/${collector.denominator || 0}；3D 保留原网络形状，不新增物理路径或神经元。`,
        items: observerIndexedEventRows.map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
      };
    }
    const engineering = sections.find((item) => item.id === 'engineering_latest');
    if (engineering) {
      engineering.detail = {
        ...engineering.detail,
        items: overallValid
          ? progressRows.map((row) => `${row.label}：${pct(progressMap[row.id])}。${row.hint}`)
          : (observerIndexedEventRows.length ? observerIndexedEventRows : predictionKernelMeasurementRows.length ? predictionKernelMeasurementRows : typedObserverQuotientRows.length ? typedObserverQuotientRows : finiteSemanticOperationRows.length ? finiteSemanticOperationRows : orthogonalDynamicPreflightRows.length ? orthogonalDynamicPreflightRows : dynamicResponseProtocolRows.length ? dynamicResponseProtocolRows : partitionInterfaceRows.length ? partitionInterfaceRows : eventHorizonRows.length ? eventHorizonRows : conditionedSequenceRows.length ? conditionedSequenceRows : predictiveStateRows.length ? predictiveStateRows : multiParentRows.length ? multiParentRows : localEdgeRows.length ? localEdgeRows : dynamicPartialOrderRows.length ? dynamicPartialOrderRows : multiPositionDynamicBindingRows.length ? multiPositionDynamicBindingRows : orderConditionedBindingRows.length ? orderConditionedBindingRows : factorSeparatedBindingRows.length ? factorSeparatedBindingRows : bindingSeparationRows.length ? bindingSeparationRows : jointFormationRows.length ? jointFormationRows : relationRows.length ? relationRows : exactEventRows)
            .map((row) => `${row.label}：${row.numerator}/${row.denominator}。`),
        valueHint: `客户端视图数量：${clientIndex?.views?.length || 0}。最新阶段：${latest}。单一全局完成百分比：${overallValid ? '有效' : '无效'}。`,
      };
    }
    return sections;
  }, [bindingSeparationRows, clientIndex, conditionedSequenceRows, dynamicPartialOrderRows, dynamicResponseProtocolRows, eventHorizonRows, exactEventRows, factorSeparatedBindingRows, finiteSemanticOperationRows, jointFormationRows, localEdgeRows, manifest, multiParentRows, multiPositionDynamicBindingRows, observerIndexedEventRows, orderConditionedBindingRows, orthogonalDynamicPreflightRows, overallValid, partitionInterfaceRows, predictionKernelMeasurementRows, predictiveStateRows, progress, progressMap, relationRows, typedObserverQuotientRows]);

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
        @media (max-width: 640px) {
          .atlas-latest-card {
            grid-template-columns: minmax(0, 1fr) !important;
            gap: 8px !important;
            align-items: start !important;
          }
          .atlas-latest-card > span {
            justify-self: start !important;
          }
          .atlas-latest-actions {
            justify-content: flex-start !important;
            flex-wrap: wrap !important;
          }
          .atlas-latest-value {
            white-space: normal !important;
            overflow-wrap: anywhere;
          }
          .atlas-detail-value-grid {
            grid-template-columns: minmax(0, 1fr) !important;
          }
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
            {(overallValid ? progressRows : (dynamicPartialOrderRows.length ? dynamicPartialOrderRows : multiPositionDynamicBindingRows.length ? multiPositionDynamicBindingRows : orderConditionedBindingRows.length ? orderConditionedBindingRows : factorSeparatedBindingRows.length ? factorSeparatedBindingRows : bindingSeparationRows.length ? bindingSeparationRows : jointFormationRows.length ? jointFormationRows : relationRows.length ? relationRows : exactEventRows)).map((row) => (
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
              className="atlas-latest-card"
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
              <div className="atlas-latest-actions" style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 14 }}>
                <div className="atlas-latest-value" style={{ color: '#bae6fd', fontSize: 11, fontWeight: 800, fontFamily: 'monospace', whiteSpace: 'nowrap' }}>{section.value}</div>
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
