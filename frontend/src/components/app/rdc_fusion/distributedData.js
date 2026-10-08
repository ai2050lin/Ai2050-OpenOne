/* 分布式研发平台 demo 数据
   叙事：大量分布式机器各领一个「模板测试」（受控最小对/扫描句族），
   跑完把测试结果上传服务器；其他人可下载；服务器综合全部模板结果，
   分析 LLM 的语言编码机制（坐标级 η² 热图 + 方向级 cos 矩阵）。
   接入点（后端 API 未实现，当前全部为演示数据）：
   - GET  /api/distributed/summary    → 平台统计 + 节点列表
   - GET  /api/distributed/templates  → 模板矩阵 + 聚合状态
   - GET  /api/distributed/news       → 行业进展（RSS / arXiv 抓取）
   - POST /api/distributed/upload     → 本机结果上传（npz + 指纹 + execution.json） */

/* 本机节点（machine tab 顶部横幅） */
export const LOCAL_NODE = {
  id: 'NODE-A3F7', gpu: 'RTX 5080 16G', model: 'qwen3-4b · bf16',
  task: 'TM-07 · is-a 关系族（四臂消融）', progress: 0.67,
  cells: 312, cells_total: 462, uploaded: 312, downloaded: 1204,
  server: '握手 OK · 23ms', status: 'running',
};

/* 平台统计（platform tab 顶部横排；demo 数值，接 /summary 后自动刷新） */
export const PLATFORM_STATS = [
  { n: '47', l: '在线节点', s: '注册 63' },
  { n: '12', l: '模板测试', s: '进行中 9' },
  { n: '6,241', l: '已上传 cells', s: '今日 +388' },
  { n: '12,884', l: '结果下载', s: '他人引用 217' },
  { n: 'AGG-v3', l: '聚合分析', s: '8 维热图已出' },
];

/* 节点列表（显示 8 个代表，完整 63 节点接 /summary 后分页） */
export const NODES = [
  { id: 'NODE-A3F7', gpu: 'RTX 5080', model: '4b bf16', tm: 'TM-07 is-a 关系族', prog: 0.67, up: 312, cls: 'run', me: true },
  { id: 'NODE-E6A0', gpu: 'M3 Max', model: '4b bf16', tm: 'TM-10 位置扫描', prog: 1.00, up: 738, cls: 'agg' },
  { id: 'NODE-C881', gpu: 'H100 80G', model: '9B NF4', tm: 'TM-03 逻辑连接词', prog: 0.88, up: 441, cls: 'run' },
  { id: 'NODE-2K44', gpu: 'L4 24G', model: '4b bf16', tm: 'TM-12 否定转折', prog: 0.76, up: 283, cls: 'run' },
  { id: 'NODE-B2C9', gpu: 'A100 80G', model: '14B NF4', tm: 'TM-01 语言最小对', prog: 0.42, up: 198, cls: 'run' },
  { id: 'NODE-F5D2', gpu: 'A6000 48G', model: 'GLM4', tm: 'TM-02 风格最小对', prog: 0.51, up: 155, cls: 'run' },
  { id: 'NODE-0B77', gpu: 'V100 32G', model: '4b bf16', tm: 'TM-11 多语混合', prog: 0.33, up: 91, cls: 'idle' },
  { id: 'NODE-D14E', gpu: 'RTX 4090', model: '4b bf16', tm: 'TM-09 句法扰动', prog: 0.15, up: 26, cls: 'idle' },
];
/* 另有 39 个节点滚动执行同一批模板（接 /summary 后展示全量） */

/* 模板测试矩阵：每个模板 = 一类受控语料 + 固定采集口径，可独立分发给任意机器
   agg 状态：collecting 收集中 / done 已聚合 / pending 待分发 */
export const TEMPLATES = [
  { id: 'TM-01', name: '语言最小对（中 ↔ 英）', dim: '语言', nodes: 6, cells: 812, agg: 'collecting' },
  { id: 'TM-02', name: '风格最小对（论文 ↔ 聊天）', dim: '风格', nodes: 5, cells: 533, agg: 'collecting' },
  { id: 'TM-03', name: '逻辑连接词（因为 / 但是 / 而且）', dim: '逻辑', nodes: 4, cells: 1204, agg: 'collecting' },
  { id: 'TM-04', name: '句式×知识 因子网格（反词嵌入指纹）', dim: '内容×句式', nodes: 1, cells: 16, agg: 'collecting' },
  { id: 'TM-05', name: '实体×位置 等变网格（内容寻址解引用）', dim: '实体×位置', nodes: 1, cells: 6, agg: 'collecting' },
  { id: 'TM-06', name: '数字与计数', dim: '内容', nodes: 3, cells: 214, agg: 'pending' },
  { id: 'TM-07', name: 'is-a 关系族（四臂消融）', dim: '内容', nodes: 5, cells: 312, agg: 'collecting', me: true },
  { id: 'TM-08', name: '属性关系族', dim: '内容', nodes: 4, cells: 388, agg: 'collecting' },
  { id: 'TM-09', name: '句法结构扰动', dim: '句法', nodes: 4, cells: 96, agg: 'pending' },
  { id: 'TM-10', name: '重复填充句（纯位置响应 p_ℓ(pos)）', dim: '位置', nodes: 2, cells: 738, agg: 'done' },
  { id: 'TM-11', name: '多语言混合句', dim: '语言', nodes: 3, cells: 261, agg: 'collecting' },
  { id: 'TM-12', name: '否定与转折', dim: '逻辑', nodes: 3, cells: 305, agg: 'collecting' },
];

/* 聚合分析流水线（server 端）：节点上传 → 指纹校验去重 → 按模板分桶
   → 坐标级 η² 汇总 / 方向级 cos 矩阵 → 语言编码机制地图 */
export const AGG_STEPS = [
  ['节点上传', 'npz + 指纹 + execution.json'],
  ['指纹校验', '去重 · drift 断言'],
  ['模板分桶', '12 桶 · 按维度归组'],
  ['效应汇总', 'η² 热图 · cos 矩阵'],
  ['机制地图', 'AGG-v3 · 8 维已出'],
];

/* 聚合维度进展（对应「多条件分解」五个条件轴） */
export const AGG_DIMS = [
  { dim: '语言', src: 'TM-01 / 11', state: 'done', note: 'β^L 热图 36×2560 已出 · 6 节点贡献' },
  { dim: '位置', src: 'TM-04 / 10', state: 'done', note: 'p_ℓ(pos) 剔除版 v2 · 738 cells' },
  { dim: '逻辑', src: 'TM-03 / 12', state: 'partial', note: '收集中 83% · 预计 2 天封桶' },
  { dim: '风格', src: 'TM-02', state: 'partial', note: '收集中 51%' },
  { dim: '内容', src: 'TM-05~08', state: 'partial', note: '置换家族保证内容同分布' },
  { dim: '句法', src: 'TM-09', state: 'pending', note: '待分发' },
];

/* 结果浏览器演示行（LensData 离线回退；行=一次上传，字段=协议键 D1-3 平铺结构）
   覆盖三种 analysis 协议形态（factorial / oneway / heads）——演示组件的协议通用性，
   数值参考 qwen3-4b 首轮真实测量（2026-10-07，登记用） */
export const DEMO_RESULTS = [
  { sha: 'demo-a1f04c8e77b2', tm_id: 'TM-04', node_id: 'NODE-A3F7', model_id: 'qwen3-4b', seed: 0, kind: 'real', size: 36540, downloads: 2,
    summary_digest: { status: 'real', analysis: 'factorial', layers: ['first', 'mid', 'last'],
      eta2_by_factor: { A: 0.171, B: 0.525, interaction: 0.304 },
      fingerprint: { spread_within: { 水果: 0.021, 金属: 0.010 }, sep_out: { 水果: 0.057, 金属: 0.046 } } } },
  { sha: 'demo-b7d22e91cc04', tm_id: 'TM-05', node_id: 'NODE-A3F7', model_id: 'qwen3-4b', seed: 0, kind: 'real', size: 28211, downloads: 5,
    summary_digest: { status: 'real', analysis: 'factorial', layers: ['first', 'mid', 'last'],
      eta2_by_factor: { A: 0.766, B: 0.152, interaction: 0.082 },
      per_layer: { first: { A: 0.766, B: 0.152, interaction: 0.082 }, mid: { A: 0.279, B: 0.524, interaction: 0.197 }, last: { A: 0.292, B: 0.515, interaction: 0.193 } } } },
  { sha: 'demo-cec6669501aa', tm_id: 'TM-06', node_id: 'NODE-CD13', model_id: 'qwen3-4b', seed: 0, kind: 'real', size: 37390, downloads: 1,
    summary_digest: { status: 'real', analysis: 'heads',
      topk_jaccard: { 'none~space': 0.939, 'none~filler': 0.730, 'space~filler': 0.781 },
      prefix_pull_mean: { none: 0.000, space: 0.021, filler: 0.731 } } },
  { sha: 'demo-d9c31a54ef18', tm_id: 'TM-01', node_id: 'NODE-E6A0', model_id: 'qwen3-4b', seed: 0, kind: 'smoke', size: 8120, downloads: 0,
    summary_digest: { status: 'smoke', analysis: 'oneway', eta2_by_cond: { zh: 0.412, en: 0.389 } } },
];

/* 行业进展（demo 数据；接 /news 后由 RSS / arXiv / 官方博客抓取自动更新）
   日期为发布月份级别；每条必须与「LLM 内部机制分析」相关 */
export const NEWS = [
  { date: '2025-05', src: 'Anthropic', tag: '机制可解释性', title: 'Circuit Tracing：用归因图追踪大模型内部回路',
    p: '公开 Claude 3.5 Haiku 上 20 个案例（含签名循环、跨语言特征共享），把「特征 → 回路 → 行为」的因果链做成可审查的图——与本项目「条件齿轮」对象路由同构。' },
  { date: '2025-03', src: 'DeepMind', tag: '稀疏自编码器', title: 'Gemma Scope 扩展至 Gemma 3 全层开放',
    p: '全系跳层 SAE + 释出训练代码，为分布式社区提供可直接消融的公开基座模型——本平台模板矩阵兼容其口径。' },
  { date: '2024-11', src: 'InterPLM / Stanford', tag: '跨域对齐', title: '把 SAE 带到蛋白质语言模型 ESM-2',
    p: 'SAE 结论跨域同构的旁证：ESM-2 上叠加/分布式结论与 LLM 一致 → 支持本平台「多模板聚合出普适机制」的策略。' },
  { date: '2024-05', src: '开源社区', tag: '开放平台', title: 'Neuronpedia：SAE 特征的开放百科',
    p: '对象路由 + 激活示例的标准交互范式已被广泛复用；其开放上传接口与本平台「节点上传结果」模式可互操作。' },
  { date: '2024-01', src: 'SAEBench', tag: '评测基准', title: 'SAE 质量的系统评测基准发布',
    p: '提出稀疏性-重建-下游效应多维评测；本平台模板测试的「四臂对照 + 独立复核」属于同类证据分级思想。' },
  { date: '2022-09', src: 'Anthropic', tag: '理论基础', title: 'Toy Models of Superposition：叠加与特征相变',
    p: '「特征数 > 维度时以近正交方向叠加存储」——解释了模板矩阵里单坐标多条件混合（η² 中等）的普遍现象。' },
  { date: '2022-11', src: 'OpenAI', tag: '可视化', title: 'Microscope 归档：神经元可视化集合',
    p: '已停止更新，但「对象 → 多视图」范式被后续工具继承，是本平台三透镜路由的直系源头之一。' },
];

/* 机制可解释性领域 · 阶段时间轴（行业进展 tab 顶部）
   五个阶段：years / title / goal（该阶段的目标）/ outcome（关键成果）/ papers（代表性论文）
   论文链接均为真实官方源（distill.pub / transformer-circuits.pub / arXiv / deepmind.google / neuronpedia.org）；
   最右「最新节点」不在此登记——由 GET /api/news 实时流填充（LensProgress.jsx EraTimeline） */
export const MI_ERAS = [
  {
    id: 'era1', years: '2014–2019', title: '打开黑箱',
    goal: '对 CNN 与早期神经网络建立「看进去」的工具：神经元在看什么、哪些输入区域决定输出。确立「内部表征可以被系统性提问」的研究立场。',
    outcome: '特征可视化、显著图、探针（probing）三件套成型；Distill 特刊把可解释性做成独立发表载体，「对象 → 多视图」交互范式沿用至今。',
    papers: [
      { date: '2014-11', title: 'Visualizing and Understanding Convolutional Networks', src: 'arXiv', tag: '可视化', url: 'https://arxiv.org/abs/1311.2901', p: '反卷积把每层学到的模式显影——「逐层问网络学到了什么」的起点。' },
      { date: '2017-06', title: 'Feature Visualization（Distill 特刊）', src: 'Distill', tag: '可视化', url: 'https://distill.pub/2017/feature-visualization/', p: '激活最大化 + 正则化系统化：单个神经元/通道/层各在看什么，交互式可复现出版。' },
      { date: '2018-06', title: 'The Building Blocks of Interpretability', src: 'Distill', tag: '交互范式', url: 'https://distill.pub/2018/building-blocks/', p: '把「特征 → 人可读界面」组装成标准积木；本平台三透镜的对象路由直系源头之一。' },
    ],
  },
  {
    id: 'era2', years: '2020–2021', title: '电路假说',
    goal: '超越单神经元：提出「回路（circuit）」为理解单位——权重里能否找到可读、可人工验证的算法，而非事后统计相关。',
    outcome: '视觉回路人肉验证成功（曲线检测器）；Transformer Circuits 数学框架给出 residual stream / QK·OV 分解语言；induction heads 把「回路」接上 in-context learning 行为。',
    papers: [
      { date: '2020-04', title: 'Zoom In: An Introduction to Circuits', src: 'Distill', tag: '理论基础', url: 'https://distill.pub/2020/circuits/zoom-in/', p: 'Circuits 纲领宣言：特征由权重实现、回路由特征组成——主张「重要性权重里的真算法可被读出」。' },
      { date: '2020-06', title: 'Curve Detectors（Curve Circuits）', src: 'Distill', tag: '人工验证', url: 'https://distill.pub/2020/circuits/curve-detectors/', p: '在 InceptionV1 里逐 head 拼出完整的曲线检测回路——「可读算法」从假说变成实证。' },
      { date: '2021-09', title: 'A Mathematical Framework for Transformer Circuits', src: 'Anthropic', tag: '理论基础', url: 'https://transformer-circuits.pub/2021/framework/index.html', p: 'residual stream 视角 + QK/OV 电路分解：LLM 机制分析的标准坐标系，本项目残差流全场采集同源。' },
    ],
  },
  {
    id: 'era3', years: '2022–2023', title: '叠加与稀疏字典',
    goal: '回答「为什么单个神经元多义」：特征数超过维度时被压缩叠加存储 → 需要一种把叠加解开的工具，恢复单义特征。',
    outcome: 'Toy Models of Superposition 确立叠加理论与特征相变；SAE（稀疏自编码器）把 MLP 层分解成单义特征字典；grokking、Othello 世界模型证明内部算法可以整体逆向。',
    papers: [
      { date: '2022-09', title: 'Toy Models of Superposition', src: 'Anthropic', tag: '理论基础', url: 'https://transformer-circuits.pub/2022/toy_model/index.html', p: '特征数 > 维度时以近正交方向叠加存储，相变分相——解释了单坐标多条件混合（η² 中等）的普遍现象。' },
      { date: '2022-01', title: 'Progress Measures for Grokking via Mechanistic Interpretability', src: 'arXiv', tag: '内部算法', url: 'https://arxiv.org/abs/2201.02177', p: '把 modular addition 逆向成傅里叶三角恒等式：完整内部算法可被人类读懂的第一个 LLM 级案例。' },
      { date: '2022-11', title: 'Emergent World Representations (Othello-GPT)', src: 'arXiv', tag: '内部算法', url: 'https://arxiv.org/abs/2210.13382', p: '内部线性表征对应棋盘世界状态而非合法着法序列——「模型内部有可解释世界模型」的标志性证据。' },
      { date: '2023-10', title: 'Towards Monosemanticity', src: 'Anthropic', tag: '稀疏自编码器', url: 'https://transformer-circuits.pub/2023/monosemantic-features/index.html', p: 'SAE 把一层 MLP 分解成数千单义特征字典——多义性问题的工程解，SAE 时代开幕。' },
    ],
  },
  {
    id: 'era4', years: '2023–2024', title: '规模化与开放生态',
    goal: '把 SAE 从玩具模型推到生产级大模型：能否在真实前沿 LLM 上以工业规模提取可读特征，并作为公共基础设施开放。',
    outcome: 'Claude 3 Sonnet 上千万级特征字典（Scaling Monosemanticity）；Gemma Scope 全层 SAE 开源；Neuronpedia 特征百科、SAEBench 评测基准、InterPLM 跨域同构（ESM-2）——开放生态成型。',
    papers: [
      { date: '2024-05', title: 'Scaling Monosemanticity', src: 'Anthropic', tag: '稀疏自编码器', url: 'https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html', p: 'Claude 3 Sonnet 上提取数千万特征，含安全相关特征——SAE 正式进入生产级前沿模型。' },
      { date: '2024-09', title: 'Gemma Scope', src: 'DeepMind', tag: '开放资源', url: 'https://deepmind.google/technologies/gemma-scope/', p: 'Gemma-2 全层全宽 SAE 权重开源 + 在线演示——分布式社区可直接消融的公开基座。' },
      { date: '2024-05', title: 'Neuronpedia', src: '开源社区', tag: '开放平台', url: 'https://neuronpedia.org/', p: 'SAE 特征开放百科：对象路由 + 激活示例的标准交互范式被广泛复用。' },
      { date: '2024-11', title: 'InterPLM：蛋白质语言模型的 SAE 对齐', src: 'Stanford', tag: '跨域对齐', url: 'https://www.biorxiv.org/', p: 'ESM-2 特征与 UniProt 注释对齐：叠加/分布式结论跨域同构，支持多模型比较策略。' },
    ],
  },
  {
    id: 'era5', years: '2024–2025', title: '归因图与机制生物学',
    goal: '把「特征字典」与「电路」两大成果接起来：给定一条具体行为，自动生成特征级因果图（归因图），并对真实模型做系统的机制描述。',
    outcome: 'Circuit Tracing / Attribution Graphs 方法论发布；「LLM 生物学」20 个案例（签名循环、跨语言特征共享、加法内部计算）——机制可解释性第一次对前沿模型行为给出成体系的特征级因果解释。',
    papers: [
      { date: '2025-05', title: 'Circuit Tracing: Attribution Graphs', src: 'Anthropic', tag: '方法', url: 'https://transformer-circuits.pub/2025/attribution-graphs/methods.html', p: '替换模型 + 跨层编码边把激活图编译为可读电路——与本项目「条件齿轮」对象路由同构。' },
      { date: '2025-05', title: 'On the Biology of a Large Language Model', src: 'Anthropic', tag: '案例研究', url: 'https://transformer-circuits.pub/2025/attribution-graphs/biology.html', p: 'Claude 3.5 Haiku 上 20 个机制案例：签名循环、心智理论、跨语言共享特征。' },
    ],
  },
];

/* 表征相似性分析（RSA）· 领域阶段时间轴（行业进展 tab，第二条时间轴）
   格式与 MI_ERAS 完全一致：years / title / goal / outcome / papers（真实 DOI 或 arXiv ID）；
   relation 为与本项目证据链的「对比」注脚（对齐同行对比卡纪律：登记视角差），
   由组件以独立注脚卡渲染 */
export const RSA_ERAS = [
  {
    id: 'rsa1', years: '2008–2014', title: '认知神经科学起源',
    goal: '在 fMRI / 电生理里找一种不依赖坐标对齐的表征比较方法：MVPA 只能做分类判断，需要一种能描述「表征的关系结构」的度量。',
    outcome: 'RSA 开山论文把人 fMRI、猴电生理与行为数据放进同一 RDM 比较空间；方法纲领确立二阶同构主张；RSA Toolbox 工程化定型——RDM 成为神经科学标准工具。',
    papers: [
      { date: '2008-10', title: 'Matching Categorical Object Representations in Inferior Temporal Cortex of Man and Monkey', src: 'Neuron', tag: '开山之作', url: 'https://doi.org/10.1016/j.neuron.2008.10.043', p: 'RSA 创始论文：用 RDM 把人 fMRI、猴电生理与行为数据放进同一比较空间——「表征的关系结构可比」自此成为标准范式。' },
      { date: '2008-11', title: 'Representational Similarity Analysis — Connecting the Branches of Systems Neuroscience', src: 'Frontiers', tag: '方法纲领', url: 'https://doi.org/10.3389/neuro.06.004.2008', p: 'RSA 方法宣言：二阶同构（second-order isomorphism）主张——系统间比较不必坐标对齐，比较关系结构即可。' },
      { date: '2014-03', title: 'A Toolbox for Representational Similarity Analysis', src: 'PLOS Comput Biol', tag: '工具箱', url: 'https://doi.org/10.1371/journal.pcbi.1003553', p: 'MATLAB RSA 工具箱 + 统计推断流程（bootstrap/置换检验），RSA 方法的工程化定型。' },
    ],
  },
  {
    id: 'rsa2', years: '2014–2017', title: '深度网络对齐',
    goal: '回答「DNN 是不是好的大脑模型」：把 CNN 的表征空间放进 RDM，与灵长类腹侧视觉通路（IT 皮层）直接比较。',
    outcome: '目标驱动优化出的层级模型预测高阶视觉皮层响应；监督 CNN（而非无监督模型）解释 IT 表征——RSA 成为「神经网络↔大脑」比较的标准桥，Brain-Score 等基准随后成型。',
    papers: [
      { date: '2014-05', title: 'Performance-optimized Hierarchical Models Predict Neural Responses in Higher Visual Cortex', src: 'Nature Neurosci', tag: '深度网络', url: 'https://doi.org/10.1038/nn.3895', p: '目标驱动层级模型对齐 IT 皮层——「用任务优化出的 DNN 表征直接预测神经响应」的奠基工作。' },
      { date: '2014-11', title: 'Deep Supervised, Not Trained, Models Explain IT Representations', src: 'PLOS Comput Biol', tag: '深度网络', url: 'https://doi.org/10.1371/journal.pcbi.1003963', p: '首个把 CNN 的 RDM 与灵长类 IT 皮层对齐的经典研究：监督训练是关键，自监督/无监督不够。' },
    ],
  },
  {
    id: 'rsa3', years: '2017–2020', title: 'DNN 相似度方法学',
    goal: '为「比较两个网络」发展专门方法学：RSA 之外，子空间、核函数与不变性权衡——模型间比较需要处理各向异性缩放、正交变换等自由度。',
    outcome: 'SVCCA（子空间主角度）与 CKA（中心核对齐）相继提出，CKA 因对正交变换与各向同性缩放不变、并指出线性回归/CCA 的缺陷，成为模型间比较的事实标准——与 RSA 并列构成表征比较工具箱。',
    papers: [
      { date: '2017-06', title: 'SVCCA: Singular Vector Canonical Correlation Analysis', src: 'arXiv', tag: '相似度方法', url: 'https://arxiv.org/abs/1706.05806', p: '子空间层面比较两网络的表征：层间收敛曲线、跨训练相似度——RSA 思想在 DNN 内部的直接推广。' },
      { date: '2019-06', title: 'Similarity of Neural Network Representations Revisited (CKA)', src: 'arXiv', tag: '相似度方法', url: 'https://arxiv.org/abs/1905.00414', p: '中心核对齐 CKA：对正交变换与各向同性缩放不变——指出线性回归/CCA 的缺陷，成为模型间比较事实标准。' },
    ],
  },
  {
    id: 'rsa4', years: '2021–至今', title: '跨模态与规模化',
    goal: '从「网络↔大脑」扩展到「模型↔模型」的规模化比较：不同架构、模态、规模的模型表征是否收敛到同一结构。',
    outcome: 'feature-reweighted RSA 用脑/行为数据重加权模型特征提升拟合；Platonic Representation Hypothesis 用表征相似性度量发现跨模态模型随规模收敛到彼此对齐的表征空间——表征比较成为大模型时代的核心证据类型。',
    papers: [
      { date: '2024-05', title: 'The Platonic Representation Hypothesis', src: 'arXiv', tag: '跨模型', url: 'https://arxiv.org/abs/2405.07987', p: '用表征相似性度量发现：不同模态/架构的模型随规模收敛到彼此对齐的表征空间——「关系结构可比」的规模化验证。' },
    ],
  },
];
export const RSA_NOTE = '对比：本项目 TM 协议已逐 cell 保存 cos 方向矩阵（means.npz），RDM 可由 cell-cos 直接构造——RSA 是 AGG-v1 跨节点聚合的候选补充指标（跨模型/跨模板比较）；差异在于 RSA 比较整体关系结构、η² 方差分解按因子归位，二者正交而非替代。';
