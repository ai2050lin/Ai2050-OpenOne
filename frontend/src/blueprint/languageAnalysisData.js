export function tr(value, lang = 'zh') {
  if (typeof value === 'string') return value;
  return value?.[lang] || value?.zh || '';
}

export const LANGUAGE_FRAMEWORK_COLUMNS = [
  {
    id: 'patterns',
    title: { zh: '语言模式系统', en: 'Language Pattern System' },
    icon: 'database',
    accent: '#22d3ee',
    items: [
      {
        title: { zh: '知识模式', en: 'Knowledge Patterns' },
        body: {
          zh: '实体、属性、关系和抽象层级组成知识网络中的模式。',
          en: 'Entities, properties, relations, and abstraction levels form patterns in a knowledge network.',
        },
      },
      {
        title: { zh: '推理模式', en: 'Reasoning Patterns' },
        body: {
          zh: '条件、比较、因果、转折和多步分析可以看作状态转移模式。',
          en: 'Conditioning, comparison, causality, contrast, and multistep analysis can be treated as transition patterns.',
        },
      },
      {
        title: { zh: '语法与协议模式', en: 'Syntax and Protocol Patterns' },
        body: {
          zh: '语序、指代、标点、终止和输出格式共同约束 token 生成。',
          en: 'Word order, reference, punctuation, stopping, and output format jointly constrain token generation.',
        },
      },
      {
        title: { zh: '模式是工作本体', en: 'Patterns Are a Working Ontology' },
        body: {
          zh: '“模式”用于统一研究对象，但模式边界、组合规则和最小单元仍需实验定义。',
          en: 'Patterns unify the research objects, while their boundaries, composition rules, and minimal units remain experimental questions.',
        },
      },
    ],
  },
  {
    id: 'relative-encoding',
    title: { zh: '相对编码机制', en: 'Relational Encoding' },
    icon: 'target',
    accent: '#f59e0b',
    items: [
      {
        title: { zh: '网络位置', en: 'Network Position' },
        body: {
          zh: '概念身份可能来自与其他概念、关系和功能的相对连接，而不是孤立坐标。',
          en: 'Concept identity may come from relations to other concepts and functions rather than an isolated coordinate.',
        },
      },
      {
        title: { zh: '复用与差分', en: 'Reuse and Difference' },
        body: {
          zh: '共享结构被反复复用，较小差异负责表达模式的独特特征。',
          en: 'Shared structures are reused, while smaller differences express pattern-specific properties.',
        },
      },
      {
        title: { zh: '抽象与精确', en: 'Abstraction and Precision' },
        body: {
          zh: '同一结构需要同时支持族级抽象和具体词、关系、位置的精确选择。',
          en: 'The same structure must support family-level abstraction and precise selection of words, relations, and positions.',
        },
      },
      {
        title: { zh: '动态生态位', en: 'Dynamic Niche' },
        body: {
          zh: '训练与上下文持续改变模式的实际含义和用法，使编码表现为条件化过程。',
          en: 'Training and context continually shape actual meaning and use, making encoding a conditioned process.',
        },
      },
    ],
  },
  {
    id: 'multiscale',
    title: { zh: '多尺度条件计算', en: 'Multiscale Conditional Computation' },
    icon: 'layers',
    accent: '#34d399',
    items: [
      {
        title: { zh: '族内结构', en: 'Within-Family Structure' },
        body: {
          zh: '比较同一模式族内共享结构、差异特征和候选竞争。',
          en: 'Compare shared structure, differential features, and candidate competition within a pattern family.',
        },
      },
      {
        title: { zh: '族间路由', en: 'Cross-Family Routing' },
        body: {
          zh: '观察知识、逻辑、语法和风格模式如何在不同组件间协同或竞争。',
          en: 'Observe how knowledge, logic, syntax, and style patterns cooperate or compete across components.',
        },
      },
      {
        title: { zh: '全局分布', en: 'Global Distribution' },
        body: {
          zh: '研究模式族在位置、层、Attention、MLP 和 residual 中的整体组织。',
          en: 'Study how pattern families are organized across positions, layers, attention, MLPs, and residual streams.',
        },
      },
      {
        title: { zh: '模型粗糙度', en: 'Model Imperfection' },
        body: {
          zh: '规模、训练数据和架构限制会产生噪声与模型特异捷径，跨模型比较应优先对齐功能角色。',
          en: 'Scale, data, and architecture create noise and model-specific shortcuts; cross-model comparisons should align functional roles first.',
        },
      },
    ],
  },
];

export const TRACE_EVIDENCE_STAGES = [
  { id: 'observed', label: { zh: '已观察', en: 'Observed' } },
  { id: 'repeated', label: { zh: '重复', en: 'Repeated' } },
  { id: 'crossContext', label: { zh: '跨上下文', en: 'Context' } },
  { id: 'crossTemplate', label: { zh: '跨模板', en: 'Template' } },
  { id: 'crossModel', label: { zh: '跨模型', en: 'Model' } },
  { id: 'causal', label: { zh: '局部因果', en: 'Causal' } },
  { id: 'closed', label: { zh: '闭合', en: 'Closed' } },
];

export const TRACE_DIMENSIONS = [
  { id: 'operation', label: { zh: '语言操作', en: 'Operation' } },
  { id: 'topology', label: { zh: '网络位置', en: 'Topology' } },
  { id: 'evidence', label: { zh: '证据等级', en: 'Evidence' } },
  { id: 'model', label: { zh: '模型覆盖', en: 'Models' } },
];

export const LANGUAGE_TRACES = [
  {
    id: 'binding-value-route',
    title: { zh: '事实绑定值词脉络', en: 'Fact-Binding Value Route' },
    phase: 'Phase 1007',
    accent: '#22d3ee',
    status: { zh: '双模型重复 · GLM4 局部因果', en: 'Repeated in two models · local GLM4 causality' },
    summary: {
      zh: '交换两个实体的双词代码时，无标签搜索在 Qwen3 与 GLM4 中都恢复了四个事实值词位置。',
      en: 'When two entity-code bindings are swapped, blind search recovers the four fact-value token positions in Qwen3 and GLM4.',
    },
    models: ['Qwen3', 'GLM4'],
    dimensions: {
      operation: { zh: '事实与绑定', en: 'Facts & binding' },
      topology: { zh: '早层输入源', en: 'Early source' },
      evidence: { zh: '局部因果', en: 'Local causal' },
      model: { zh: '双模型结构', en: 'Two-model structure' },
    },
    evidence: ['observed', 'repeated', 'crossContext', 'crossModel', 'causal'],
    mechanism: {
      zh: '同世界交换四个值词状态可以交换答案。GLM4 中，跨名字世界的整向量与匹配差分也能稳定交换答案，同时通过同答案与 no-op 控制。',
      en: 'Replacing the four value-token states within a world swaps the answer. In GLM4, whole-state and matched-delta transfer also work across name worlds while preserving same-answer and no-op controls.',
    },
    networkFeatures: [
      { label: { zh: '物理位置', en: 'Physical positions' }, value: { zh: 'Qwen3: 9/10/18/19 · GLM4: 10/11/19/20', en: 'Qwen3: 9/10/18/19 · GLM4: 10/11/19/20' } },
      { label: { zh: '层级', en: 'Depth' }, value: { zh: 'Residual depth 1', en: 'Residual depth 1' } },
      { label: { zh: '组件粒度', en: 'Granularity' }, value: { zh: '四个完整 residual 向量', en: 'Four whole residual vectors' } },
      { label: { zh: '输出表现', en: 'Output effect' }, value: { zh: '双词候选同时翻转', en: 'Both answer tokens switch' } },
    ],
    trajectory: [
      { stage: '01', title: { zh: '事实写入', en: 'Fact entry' }, detail: { zh: '两个事实中的四个值词发生绑定交换。', en: 'Four value tokens swap their entity bindings.' } },
      { stage: '02', title: { zh: '联合源形成', en: 'Joint source' }, detail: { zh: '四个位置共同形成充分的早层输入源。', en: 'The four positions jointly form an early sufficient source.' } },
      { stage: '03', title: { zh: '候选切换', en: 'Candidate switch' }, detail: { zh: '两个语义步的 gold/foil 竞争同步改变。', en: 'Gold/foil competition changes at both semantic steps.' } },
      { stage: '04', title: { zh: '双词输出', en: 'Two-token output' }, detail: { zh: '模型生成新的完整双词代码。', en: 'The model emits the new two-token code.' } },
    ],
    findings: [
      { zh: '两个模型恢复相同的四值词功能结构。', en: 'Both models recover the same four-value-token functional structure.' },
      { zh: 'GLM4 的六类正式条件均达到或接近 1.000。', en: 'All six formal GLM4 conditions reach or approach 1.000.' },
    ],
    limits: [
      { zh: 'depth 1 仍保留强词元内容，不能据此宣称已经发现抽象关系元组。', en: 'Depth 1 retains strong lexical content, so this does not establish an abstract relational tuple.' },
      { zh: 'Qwen3 no-op 为 31/32，存在 8bit 数值稳定性警告。', en: 'Qwen3 no-op is 31/32, exposing an 8-bit numerical stability warning.' },
    ],
    source: 'Phase1007 · discovery/t1 · binding_flip',
  },
  {
    id: 'query-selection-route',
    title: { zh: '查询条件选择脉络', en: 'Query-Condition Selection Route' },
    phase: 'Phase 1007',
    accent: '#f59e0b',
    status: { zh: '双模型重复 · 世界内因果', en: 'Repeated in two models · within-world causality' },
    summary: {
      zh: '只改变查询对象时，Qwen3 与 GLM4 都只需查询句中的一个名字位置即可在同一事实世界内切换答案。',
      en: 'When only the queried entity changes, one query-name position is sufficient to switch the answer within the same fact world in both models.',
    },
    models: ['Qwen3', 'GLM4'],
    dimensions: {
      operation: { zh: '查询与选择', en: 'Query & selection' },
      topology: { zh: '查询条件源', en: 'Query source' },
      evidence: { zh: '重复结构', en: 'Repeated structure' },
      model: { zh: '双模型结构', en: 'Two-model structure' },
    },
    evidence: ['observed', 'repeated', 'crossModel', 'causal'],
    mechanism: {
      zh: '查询名字状态在当前事实世界中控制模型读取哪一个事实槽。它不是一个已经证明可脱离上下文搬运的统一“查询向量”。',
      en: 'The query-name state controls which fact slot is read within the current world. It is not a proven context-free query vector.',
    },
    networkFeatures: [
      { label: { zh: '物理位置', en: 'Physical positions' }, value: { zh: 'Qwen3: 25 · GLM4: 26', en: 'Qwen3: 25 · GLM4: 26' } },
      { label: { zh: '层级', en: 'Depth' }, value: { zh: 'Residual depth 1', en: 'Residual depth 1' } },
      { label: { zh: '世界内效应', en: 'Within-world effect' }, value: { zh: '0.96875-1.000 donor sequence', en: '0.96875-1.000 donor sequence' } },
      { label: { zh: '跨世界效应', en: 'Cross-world effect' }, value: { zh: '整向量约 0.469，差分 0.219-0.375', en: 'Whole state about 0.469; delta 0.219-0.375' } },
    ],
    trajectory: [
      { stage: '01', title: { zh: '查询改变', en: 'Query change' }, detail: { zh: '事实不变，只替换问题中的实体名。', en: 'Facts stay fixed while the queried name changes.' } },
      { stage: '02', title: { zh: '事实选择', en: 'Fact selection' }, detail: { zh: '查询状态使答案转向另一个事实槽。', en: 'The query state redirects the answer to the other fact slot.' } },
      { stage: '03', title: { zh: '候选竞争', en: 'Competition' }, detail: { zh: '两个语义词的候选 margin 同步切换。', en: 'Candidate margins switch at both semantic steps.' } },
      { stage: '04', title: { zh: '上下文依赖', en: 'Context dependence' }, detail: { zh: '不同名字世界的查询状态不能直接交换。', en: 'Query states do not directly transfer between name worlds.' } },
    ],
    findings: [
      { zh: '两个模型都无标签选中唯一的 query_name 位置。', en: 'Blind search selects the single query_name position in both models.' },
      { zh: '查询条件与事实内容形成条件化关系，而不是独立绝对坐标。', en: 'The query condition is relational to fact content rather than an independent absolute coordinate.' },
    ],
    limits: [
      { zh: '跨世界 donor 同时改变实体身份，没有建立严格实体双射。', en: 'Cross-world donors change entity identity without a strict entity bijection.' },
      { zh: '当前失败不能区分上下文相对编码和 donor 设计混杂。', en: 'The failure cannot yet separate contextual coding from donor-design confounding.' },
    ],
    source: 'Phase1007 · discovery/t1 · query_flip',
  },
  {
    id: 'role-constellation',
    title: { zh: '多位置关系角色星座', en: 'Multi-Position Relation Constellation' },
    phase: 'Phase 1004-1006',
    accent: '#34d399',
    status: { zh: '跨运行重复 · 局部留出通过', en: 'Repeated across runs · local holdout pass' },
    summary: {
      zh: '无标签搜索反复找到查询、实体与候选值共同参与的多位置结构，说明简单关系检索不是单点神经元功能。',
      en: 'Blind searches repeatedly recover query, entity, and value positions, showing that simple relation retrieval is not a single-point neuron function.',
    },
    models: ['GLM4', 'DeepSeek7B'],
    dimensions: {
      operation: { zh: '关系结构', en: 'Relation structure' },
      topology: { zh: '多位置输入源', en: 'Multi-position source' },
      evidence: { zh: '重复结构', en: 'Repeated structure' },
      model: { zh: '双模型结构', en: 'Two-model structure' },
    },
    evidence: ['observed', 'repeated', 'crossContext', 'crossModel', 'causal'],
    mechanism: {
      zh: '查询槽、事实实体槽和值词槽共同构成任务局部的充分状态。不同位置的联合效应显著强于单位置搜索。',
      en: 'Query, fact-entity, and value slots jointly form a task-local sufficient state. Their joint effect is much stronger than single-position search.',
    },
    networkFeatures: [
      { label: { zh: '典型结构', en: 'Typical structure' }, value: { zh: '查询名 + 两个实体槽 + 四个值词槽', en: 'Query name + two entity slots + four value-token slots' } },
      { label: { zh: '搜索方式', en: 'Search method' }, value: { zh: '无标签 single/LOO + greedy + reverse delete', en: 'Blind single/LOO + greedy + reverse delete' } },
      { label: { zh: '局部复现', en: 'Local replication' }, value: { zh: 'GLM4 discovery/t0 留出通过', en: 'GLM4 discovery/t0 holdout pass' } },
      { label: { zh: '父门状态', en: 'Parent gate' }, value: { zh: '跨模型未闭合', en: 'Not closed across models' } },
    ],
    trajectory: [
      { stage: '01', title: { zh: '角色分布', en: 'Role distribution' }, detail: { zh: '事实与查询信息分散在多个提示位置。', en: 'Fact and query information is distributed over prompt positions.' } },
      { stage: '02', title: { zh: '联合状态', en: 'Joint state' }, detail: { zh: '多个位置共同改变输出，单位置通常不足。', en: 'Multiple positions jointly change output; single positions are usually insufficient.' } },
      { stage: '03', title: { zh: '输出读取', en: 'Output readout' }, detail: { zh: '联合源影响双词候选的选择。', en: 'The joint source changes two-token candidate selection.' } },
    ],
    findings: [
      { zh: '不同运行恢复相似角色计数，结构不是最高激活排序的产物。', en: 'Different runs recover similar role counts without activation ranking.' },
      { zh: '关系检索更接近联合条件化状态，而非概念神经元。', en: 'Relation retrieval resembles a joint conditional state rather than a concept neuron.' },
    ],
    limits: [
      { zh: '固定模板可能造成物理位置重复。', en: 'Fixed templates may create physical-position recurrence.' },
      { zh: '角色标签在冻结后解释结构，不等于模型内部使用相同符号。', en: 'Post-hoc role labels do not imply the model uses the same symbols internally.' },
    ],
    source: 'Phase1004-1006 · blind source search and holdout audit',
  },
  {
    id: 'residual-value-transport',
    title: { zh: '残差与 Value 运输候选脉络', en: 'Residual and Value Transport Candidate' },
    phase: 'Phase 1001-1003',
    accent: '#60a5fa',
    status: { zh: '局部路径候选', en: 'Local path candidate' },
    summary: {
      zh: '部分任务中，早层联合源的答案效应沿残差流向后传播，并在局部 KV 分解中更多表现于 Value 路径。',
      en: 'In selected tasks, answer effects from early joint sources propagate through the residual stream and appear more strongly in local Value-path decompositions.',
    },
    models: ['Qwen3', 'GLM4', 'DeepSeek7B'],
    dimensions: {
      operation: { zh: '状态运输', en: 'State transport' },
      topology: { zh: '中层运输', en: 'Middle transport' },
      evidence: { zh: '候选脉络', en: 'Candidate route' },
      model: { zh: '三模型观测', en: 'Three-model observations' },
    },
    evidence: ['observed', 'repeated', 'crossModel', 'causal'],
    mechanism: {
      zh: '提示源产生的差异沿 residual stream 保留，部分注意力事件把与答案相关的状态写入后续位置。Value 优势是局部观测，不是全局定律。',
      en: 'Prompt-source differences persist in the residual stream, while selected attention events write answer-related state to later positions. Value dominance is a local observation, not a global law.',
    },
    networkFeatures: [
      { label: { zh: '源', en: 'Source' }, value: { zh: '早层多位置联合状态', en: 'Early multi-position joint state' } },
      { label: { zh: '载体候选', en: 'Carrier candidate' }, value: { zh: 'Residual + attention Value', en: 'Residual + attention Value' } },
      { label: { zh: '接收区域', en: 'Receiver region' }, value: { zh: '答案前位置与晚层边界', en: 'Pre-answer positions and late boundary' } },
      { label: { zh: '守恒状态', en: 'Conservation status' }, value: { zh: '功能角色比物理坐标更稳定', en: 'Functional roles are more stable than physical coordinates' } },
    ],
    trajectory: [
      { stage: '01', title: { zh: '源差异', en: 'Source difference' }, detail: { zh: '早层提示位置产生任务相关变化。', en: 'Task-related changes arise at early prompt positions.' } },
      { stage: '02', title: { zh: '残差保留', en: 'Residual retention' }, detail: { zh: '差异在后续层持续存在并发生重组。', en: 'Differences persist and reorganize through later layers.' } },
      { stage: '03', title: { zh: '注意力写入', en: 'Attention write' }, detail: { zh: '局部 Value 事件向答案位置写入状态。', en: 'Local Value events write state toward answer positions.' } },
      { stage: '04', title: { zh: '边界汇聚', en: 'Boundary convergence' }, detail: { zh: '效应在输出竞争附近变得可读。', en: 'Effects become readable near output competition.' } },
    ],
    findings: [
      { zh: '局部路径支持“源、运输、边界”三段结构。', en: 'Local paths support a source, transport, and boundary organization.' },
      { zh: '不同模型的具体层和 head 坐标并不守恒。', en: 'Exact layers and head coordinates are not conserved across models.' },
    ],
    limits: [
      { zh: '尚未得到跨任务、跨模板的统一运输拓扑。', en: 'No unified transport topology has been established across tasks and templates.' },
      { zh: '不能把 Value 优势提升为所有语言功能的载体理论。', en: 'Value-path advantages cannot be generalized to all language functions.' },
    ],
    source: 'Phase1001-1003 · local attention/KV decomposition',
  },
  {
    id: 'candidate-competition-boundary',
    title: { zh: '候选竞争与输出边界', en: 'Candidate Competition and Output Boundary' },
    phase: 'Phase 998-1003',
    accent: '#fb7185',
    status: { zh: '重复输出结构 · 局部因果不足', en: 'Repeated output structure · partial local causality' },
    summary: {
      zh: '关系选择最终表现为候选 token 之间的 margin 竞争；晚层注意力可以改变竞争裕量，但单独贡献通常不足。',
      en: 'Relation selection ultimately appears as margin competition between candidate tokens; late attention can change that margin but is usually insufficient alone.',
    },
    models: ['Qwen3', 'GLM4', 'DeepSeek7B'],
    dimensions: {
      operation: { zh: '答案选择', en: 'Answer selection' },
      topology: { zh: '晚层输出边界', en: 'Late output boundary' },
      evidence: { zh: '局部因果', en: 'Local causal' },
      model: { zh: '三模型观测', en: 'Three-model observations' },
    },
    evidence: ['observed', 'repeated', 'crossContext', 'crossModel', 'causal'],
    mechanism: {
      zh: '早中层条件化状态在答案边界转化为 gold 与 foil 的 logit margin。局部 attention 事件对 margin 有真实作用，但仍依赖残差背景和其他组件。',
      en: 'Conditioned early and middle states become a gold-versus-foil logit margin at the answer boundary. Local attention events have real effects but remain dependent on residual background and other components.',
    },
    networkFeatures: [
      { label: { zh: '测量对象', en: 'Measurement' }, value: { zh: 'gold logit - foil logit', en: 'gold logit - foil logit' } },
      { label: { zh: '主要区域', en: 'Primary region' }, value: { zh: '答案前 token 的晚层状态', en: 'Late-layer state at the pre-answer token' } },
      { label: { zh: '组件作用', en: 'Component role' }, value: { zh: 'attention 局部必要但通常不充分', en: 'attention locally necessary but usually insufficient' } },
      { label: { zh: '竞争背景', en: 'Competition context' }, value: { zh: '多候选与输出协议共同参与', en: 'Multiple candidates and output protocol participate' } },
    ],
    trajectory: [
      { stage: '01', title: { zh: '条件汇聚', en: 'Condition convergence' }, detail: { zh: '事实和查询状态进入答案边界。', en: 'Fact and query states reach the answer boundary.' } },
      { stage: '02', title: { zh: '候选展开', en: 'Candidate field' }, detail: { zh: '多个可生成 token 同时进入竞争。', en: 'Multiple possible tokens enter competition.' } },
      { stage: '03', title: { zh: '裕量形成', en: 'Margin formation' }, detail: { zh: 'gold 相对 foil 的 logit 裕量逐层形成。', en: 'The gold-versus-foil margin forms across layers.' } },
      { stage: '04', title: { zh: '下一 token', en: 'Next token' }, detail: { zh: '输出协议与候选竞争共同决定发射。', en: 'Protocol and candidate competition determine emission.' } },
    ],
    findings: [
      { zh: '候选 margin 是跨阶段可复用的输出测量接口。', en: 'Candidate margin is a reusable output measurement interface.' },
      { zh: '单组件效应不等于完整决策机制。', en: 'A single component effect is not the full decision mechanism.' },
    ],
    limits: [
      { zh: '候选集合由实验者定义，不能代表完整词表场。', en: 'Experimenter-defined candidate sets do not represent the full vocabulary field.' },
      { zh: '局部因果贡献尚未形成自然 rollout 的完整充分链。', en: 'Local causal contributions do not yet form a sufficient natural-rollout chain.' },
    ],
    source: 'Phase998-1003 · local causal thread and boundary decomposition',
  },
  {
    id: 'autoregressive-boundary',
    title: { zh: '双词自回归时间边界', en: 'Two-Token Autoregressive Boundary' },
    phase: 'Phase 1002 / 1006',
    accent: '#a78bfa',
    status: { zh: '时间方向已校正 · 汇聚待临摹', en: 'Temporal direction corrected · aggregation unmapped' },
    summary: {
      zh: '双词任务明确区分第一语义词、第二语义词和终止动作，纠正了“生成后的 token 状态解释自身”的时间因果错误。',
      en: 'Two-token tasks separate the first semantic token, second token, and termination, correcting the error of using a generated token state to explain itself.',
    },
    models: ['Qwen3', 'GLM4', 'DeepSeek7B'],
    dimensions: {
      operation: { zh: '自回归时间', en: 'Autoregressive time' },
      topology: { zh: '生成时间轴', en: 'Generation timeline' },
      evidence: { zh: '时间边界', en: 'Temporal boundary' },
      model: { zh: '三模型行为', en: 'Three-model behavior' },
    },
    evidence: ['observed', 'repeated', 'crossModel'],
    mechanism: {
      zh: '第一个词由其发射前状态决定；生成第一个词后的新状态只能影响第二个词；第二个词后的状态再影响终止。完整时间汇聚路径尚未定位。',
      en: 'The first token is determined by its pre-emission state; the post-first-token state can only affect the second token, and the post-second-token state affects termination. The full aggregation path remains unmapped.',
    },
    networkFeatures: [
      { label: { zh: '时间步 0', en: 'Step 0' }, value: { zh: 'Answer 前缀 -> 第一个语义词', en: 'Answer prefix -> first semantic token' } },
      { label: { zh: '时间步 1', en: 'Step 1' }, value: { zh: '第一个词反馈 -> 第二个语义词', en: 'First-token feedback -> second semantic token' } },
      { label: { zh: '时间步 2', en: 'Step 2' }, value: { zh: '第二个词反馈 -> 终止动作', en: 'Second-token feedback -> termination' } },
      { label: { zh: '当前缺口', en: 'Current gap' }, value: { zh: 'prompt、KV 与生成反馈尚未完整分账', en: 'Prompt, KV, and generated feedback are not fully separated' } },
    ],
    trajectory: [
      { stage: '01', title: { zh: '首词预测', en: 'First-token prediction' }, detail: { zh: '提示状态形成第一个语义词竞争。', en: 'Prompt state forms first-token competition.' } },
      { stage: '02', title: { zh: '反馈写入', en: 'Feedback write' }, detail: { zh: '已生成首词进入下一时间步上下文。', en: 'The emitted first token enters the next-step context.' } },
      { stage: '03', title: { zh: '次词预测', en: 'Second-token prediction' }, detail: { zh: '提示与反馈共同支持第二个语义词。', en: 'Prompt and feedback jointly support the second token.' } },
      { stage: '04', title: { zh: '终止竞争', en: 'Termination' }, detail: { zh: '完成答案后模型选择结束或继续。', en: 'After the answer, the model chooses to stop or continue.' } },
    ],
    findings: [
      { zh: '教师强制可答不等于自然自回归路径稳定。', en: 'Teacher-forced readability does not imply stable natural autoregression.' },
      { zh: '时间因果方向已经成为后续图谱的硬约束。', en: 'Temporal causal order is now a hard atlas constraint.' },
    ],
    limits: [
      { zh: '尚未完成 prompt 源、KV cache 和生成反馈的 2x2 分解。', en: 'Prompt source, KV cache, and generated feedback have not been factorized.' },
      { zh: '不能把第一语义步直接称为完整决策点。', en: 'The first semantic step cannot yet be called the complete decision point.' },
    ],
    source: 'Phase1002 / Phase1006 · multi-token behavior and temporal audit',
  },
];

export const RESEARCH_MILESTONES = [
  {
    id: 'foundation',
    phase: 'Phase 20-300',
    tone: '#60a5fa',
    title: { zh: '从激活观察建立基础探针', en: 'Foundation Probes from Activation Observation' },
    summary: {
      zh: '建立隐藏状态、层级读出、候选概率和基础干预工具，确认语言信息广泛分布。',
      en: 'Established hidden-state, layer-readout, candidate-probability, and basic intervention tools, confirming broad distribution of language information.',
    },
    objective: { zh: '回答模型内部是否存在可读的语义、关系和层级信号。', en: 'Determine whether semantic, relational, and hierarchical signals are readable internally.' },
    result: { zh: '大量局部信号可以解码，但“可读”不能证明模型自然使用。', en: 'Many local signals are decodable, but readability does not prove natural use.' },
    lesson: { zh: '最高激活、线性可分和 probe 准确率不能单独定义机制。', en: 'Maximum activation, linear separability, and probe accuracy cannot define mechanism alone.' },
    impact: { zh: '为后续因果干预和物理路径定位建立测量基础。', en: 'Provided measurement foundations for causal intervention and physical path localization.' },
  },
  {
    id: 'local-theories',
    phase: 'Phase 301-594',
    tone: '#a78bfa',
    title: { zh: '局部结构与理论候选扩展', en: 'Expansion of Local Structures and Theory Candidates' },
    summary: {
      zh: '围绕商空间、流形、状态更新和编码结构提出多类候选解释，并积累大量局部拼图。',
      en: 'Explored quotient, manifold, state-update, and encoding candidates while accumulating many local observations.',
    },
    objective: { zh: '寻找能够统一描述语言状态变化的数学对象。', en: 'Search for mathematical objects that unify language-state changes.' },
    result: { zh: '局部规律丰富，但统一公式的外推性不足。', en: 'Local regularities were rich, but unified formulas did not generalize.' },
    lesson: { zh: '理论名称增长快于可重复物理证据，容易形成公式优先。', en: 'Theory labels grew faster than repeatable physical evidence, encouraging formula-first research.' },
    impact: { zh: '研究逐步转向以反事实和可干预结果约束理论。', en: 'Research shifted toward constraining theory with counterfactual and intervention evidence.' },
  },
  {
    id: 'closure-era',
    phase: 'Phase 595-826',
    tone: '#f59e0b',
    title: { zh: '闭合纤维与全路径验证', en: 'Closure Fibers and Full-Path Validation' },
    summary: {
      zh: '建立 source、transport、competition、rollout 和 control 等闭合维度，显著提高证据纪律。',
      en: 'Established source, transport, competition, rollout, and control dimensions, greatly improving evidence discipline.',
    },
    objective: { zh: '要求局部候选同时解释源、路径、输出和副作用。', en: 'Require local candidates to explain sources, paths, outputs, and side effects together.' },
    result: { zh: '发现多个局部因果部件，也反复暴露补偿、模板敏感和自然生成失败。', en: 'Found local causal parts while repeatedly exposing compensation, template sensitivity, and rollout failure.' },
    lesson: { zh: '闭合适合验证，却不适合作为全局发现阶段的停止条件。', en: 'Closure is valuable for verification but harmful as a stop condition during global discovery.' },
    impact: { zh: '保留严格控制，同时需要将闭合从图谱扩展的父门降为证据等级。', en: 'Strict controls remain, but closure must become an evidence level rather than an atlas-growth parent gate.' },
  },
  {
    id: 'global-atlas',
    phase: 'Phase 827-982',
    tone: '#22d3ee',
    title: { zh: '全局图谱、颜色特征与三模型观测', en: 'Global Atlas, Color Features, and Three-Model Observation' },
    summary: {
      zh: '扩展到全词表、颜色属性、模式族和三模型物理分布，开始形成统一数据格式和可视化图谱。',
      en: 'Expanded toward vocabulary-wide, color, pattern-family, and three-model physical distributions with unified data formats and visualization.',
    },
    objective: { zh: '把分散测试压缩为可预测、可验证和可复用的图谱。', en: 'Compress dispersed experiments into a predictive, testable, reusable atlas.' },
    result: { zh: '获得大量候选地址和模式分布，但最大神经元并不等于关键机制。', en: 'Produced many candidate addresses and pattern distributions, while showing that top neurons are not necessarily mechanistic.' },
    lesson: { zh: '图谱需要记录完整计算脉络、证据来源和失败边界。', en: 'An atlas must retain full computational traces, evidence sources, and failure boundaries.' },
    impact: { zh: '推动统一证据内核和主 3D 空间的数据合同。', en: 'Drove the unified evidence kernel and main 3D-space data contract.' },
  },
  {
    id: 'causal-subgraph',
    phase: 'Phase 983-1003',
    tone: '#fb7185',
    title: { zh: '从外部行为进入局部内部因果子图', en: 'From External Behavior to Local Internal Causal Subgraphs' },
    summary: {
      zh: '使用最小反事实、局部源、接收事件和注意力分解，定位早层联合源与晚层答案边界。',
      en: 'Used minimal counterfactuals, local sources, receiver events, and attention decomposition to locate early joint sources and late answer boundaries.',
    },
    objective: { zh: '从可读特征推进到实际改变输出的内部事件。', en: 'Move from readable features to internal events that change output.' },
    result: { zh: '对象身份、关系选择和候选竞争出现局部因果结构，但完整路径仍不充分。', en: 'Object identity, relation selection, and candidate competition showed local causal structures, while full paths remained insufficient.' },
    lesson: { zh: '物理坐标不守恒，功能拓扑可能比单层、单 head 更稳定。', en: 'Physical coordinates are not conserved; functional topology may be more stable than a single layer or head.' },
    impact: { zh: '研究对象从“关键神经元”转为“条件化事件图”。', en: 'The research object shifted from key neurons to conditioned event graphs.' },
  },
  {
    id: 'blind-structure',
    phase: 'Phase 1004-1007',
    tone: '#34d399',
    title: { zh: '无标签结构发现与角色对齐审计', en: 'Blind Structure Discovery and Role-Alignment Audit' },
    summary: {
      zh: '无标签搜索恢复多位置角色结构，并通过最小绑定与查询反事实区分值内容和查询条件。',
      en: 'Blind search recovered multi-position role structures and separated value content from query conditions through minimal binding and query counterfactuals.',
    },
    objective: { zh: '先发现重复内部结构，再讨论理论和公式。', en: 'Discover repeated internal structures before proposing theory and formulas.' },
    result: { zh: '四值词源在 GLM4 局部干净，查询名在世界内有效但不能直接跨世界交换。', en: 'The four-value source is locally clean in GLM4; the query name works within-world but does not directly transfer across worlds.' },
    lesson: { zh: '失败可能来自上下文相对编码，也可能来自 donor 身份没有真正对齐。', en: 'Failure may reflect contextual coding or donor identity misalignment.' },
    impact: { zh: '证明“每次发现立即追求闭合”会阻塞整体结构临摹。', en: 'Showed that demanding immediate closure after each discovery blocks global cartography.' },
  },
  {
    id: 'cartography-shift',
    phase: 'Phase 1008 方向',
    tone: '#fbbf24',
    title: { zh: '转向全局脉络临摹，闭合后置', en: 'Shift to Global Trace Cartography with Deferred Closure' },
    summary: {
      zh: '用多因子反事实持续绘制“出现、扩散、汇聚、输出”的全局响应脉络，闭合失败只降低等级，不删除结构。',
      en: 'Use multi-factor counterfactuals to map global emergence, spread, convergence, and output traces; closure failure lowers evidence level without deleting structure.',
    },
    objective: { zh: '先看到网络处理语言变化的整体形状，再选择少数关键路径做严格验证。', en: 'See the global shape of language processing before selecting a few critical paths for strict verification.' },
    result: { zh: '路线已确定，完整响应场数据尚待生成。', en: 'The route is defined; full response-field data remains to be generated.' },
    lesson: { zh: '临摹路线和验证路线应并行，不能串成单一失败即停止的流水线。', en: 'Cartography and verification should run in parallel rather than as one fail-stop pipeline.' },
    impact: { zh: '图谱成为持续增长的证据地图，闭合成为其中最高等级而非唯一目标。', en: 'The atlas becomes a continuously growing evidence map, with closure as its highest level rather than its sole objective.' },
  },
];

export const CURRENT_RESEARCH_MILESTONES = [
  {
    id: 'static-entity-search', phase: 'Phase 901-1106', tone: '#60a5fa',
    title: { zh: '静态语义实体搜索', en: 'Static Semantic Entity Search' },
    summary: { zh: '从固定方向、神经元和局部组件搜索出发，积累候选竞争、条件化内容、复用与路由拼图。', en: 'Started from fixed directions, neurons, and local components, accumulating evidence about competition, conditioning, reuse, and routing.' },
    objective: { zh: '检验语言模式是否对应跨材料稳定的静态物理实体。', en: 'Test whether language patterns correspond to static physical entities stable across materials.' },
    result: { zh: '项目测试未支持固定跨材料执行方向或单点通用运输器。', en: 'Project tests did not support a fixed cross-material execution direction or universal single-point transporter.' },
    lesson: { zh: '静态几何可以产生候选，但不能直接代表计算机制。', en: 'Static geometry can generate candidates but cannot directly define computation.' },
    impact: { zh: '研究对象转向条件化、分布式的计算过程。', en: 'The research object shifted toward conditioned distributed computation.' },
  },
  {
    id: 'natural-behavior', phase: 'Phase 1107-1125', tone: '#a78bfa',
    title: { zh: '自然语义行为门', en: 'Natural Semantic Behavior Gates' },
    summary: { zh: '用自然语义与双正交材料区分行为成功、线性可读和真实使用。', en: 'Used natural semantics and orthogonal materials to separate behavior, readability, and use.' },
    objective: { zh: '先确认模型稳定执行目标能力，再观察内部结构。', en: 'Verify stable target behavior before inspecting internals.' },
    result: { zh: '形容词双正交行为通过；固定物理方向桥未获得三模型闭合。', en: 'Adjective behavior passed, while fixed physical-direction bridges did not close across three models.' },
    lesson: { zh: '可读不等于使用，强制建立的桥也不等于自然机制。', en: 'Readability is not use, and an engineered bridge is not a natural mechanism.' },
    impact: { zh: '建立行为门到内部观测再到因果干预的顺序。', en: 'Established behavior, observation, then intervention as the required order.' },
  },
  {
    id: 'instrument-qualification', phase: 'Phase 1126-1134', tone: '#f59e0b',
    title: { zh: '仪器与材料资格', en: 'Instrument and Material Qualification' },
    summary: { zh: '解决 FP16、运行路径和材料质量问题，避免把仪器误差写成机制。', en: 'Addressed FP16, execution-path, and material issues before mechanism claims.' },
    objective: { zh: '建立可复现模型运行、研究对象和证据入口。', en: 'Establish reproducible model runs, research objects, and evidence entry gates.' },
    result: { zh: '定位数值形成问题并建立时效关系反事实对象。', en: 'Localized numerical failures and established temporal-relation counterfactual objects.' },
    lesson: { zh: '仪器资格是机制研究的前置条件。', en: 'Instrument qualification is a prerequisite for mechanism research.' },
    impact: { zh: '主线从热点搜索推进到可识别状态转移。', en: 'The main route moved from hotspot search to identifiable state transitions.' },
  },
  {
    id: 'temporal-state-transition', phase: 'Phase 1135-1138', tone: '#22d3ee',
    title: { zh: '时效绑定与状态跃迁', en: 'Temporal Binding and State Transition' },
    summary: { zh: '四状态反事实与同族尺寸复验显示后半程状态可搬运性增强。', en: 'Four-state counterfactuals and same-family scale replication showed stronger late-state transportability.' },
    objective: { zh: '定位答案状态从不可搬运到可搬运的深度变化。', en: 'Locate where answer states become transportable across depth.' },
    result: { zh: '两个 Qwen3 尺寸在相对深度 0.6-0.7 出现跃迁，但没有共同充分层。', en: 'Two Qwen3 sizes showed a transition around relative depth 0.6-0.7, without a shared sufficient layer.' },
    lesson: { zh: '功能现象可以重复，绝对层和统一充分状态却不一定守恒。', en: 'A functional phenomenon may repeat while absolute layers and sufficient states do not.' },
    impact: { zh: '从方向读取推进到状态充分性检验。', en: 'Advanced from direction readout to state-sufficiency tests.' },
  },
  {
    id: 'matched-path-interpolation', phase: 'Phase 1139', tone: '#34d399',
    title: { zh: '同路径插值校准', en: 'Matched-Path Interpolation' },
    summary: { zh: 'live-state 同路径插值把 α=0 自写回漂移降为零。', en: 'Live-state matched-path interpolation reduced alpha-zero self-write drift to zero.' },
    objective: { zh: '区分真实状态响应与跨批次执行漂移。', en: 'Separate real state response from cross-run execution drift.' },
    result: { zh: '深度 0.7 是强调制 donor，但约三成样本不足以翻转。', en: 'Depth 0.7 is a strong modulator, but roughly thirty percent of samples still do not flip.' },
    lesson: { zh: '身份等价必须先于充分性和相变判断。', en: 'Identity equivalence must precede sufficiency or phase-transition claims.' },
    impact: { zh: '建立同路径干预作为后续因果实验标准。', en: 'Established matched-path intervention as the causal standard.' },
  },
  {
    id: 'sequence-decision-path', phase: 'Phase 1140', tone: '#fb7185',
    title: { zh: '多 token 决策路径对齐', en: 'Multi-Token Decision-Path Alignment' },
    summary: { zh: '序列级读出必须覆盖候选预测路径，单点答案边界不足以代表共享前缀候选。', en: 'Sequence-level readout must cover the candidate prediction path; a single answer boundary is insufficient for shared-prefix candidates.' },
    objective: { zh: '修复单点补丁与多 token 候选评分的位置错配。', en: 'Repair the mismatch between single-point patching and multi-token candidate scoring.' },
    result: { zh: '12/12 条共享前缀曲线被恢复，但双模型统一充分状态仍未授权。', en: 'All 12 shared-prefix curves were recovered, while a cross-size sufficient state remains unauthorized.' },
    lesson: { zh: '真正决策位置可能位于候选首次分叉处。', en: 'The true decision location may be the first candidate divergence.' },
    impact: { zh: '下一步转向分叉边界的必要性、充分性和预测闭合。', en: 'Next work targets necessity, sufficiency, and prediction at the divergence boundary.' },
  },
];
