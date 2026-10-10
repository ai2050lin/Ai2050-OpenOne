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
  { date: '2025-05', src: 'Anthropic', tag: '机制可解释性', cat: 'causal', title: 'Circuit Tracing：用归因图追踪大模型内部回路',
    p: '公开 Claude 3.5 Haiku 上 20 个案例（含签名循环、跨语言特征共享），把「特征 → 回路 → 行为」的因果链做成可审查的图——与本项目「条件齿轮」对象路由同构。' },
  { date: '2025-03', src: 'DeepMind', tag: '稀疏自编码器', cat: 'sparse', title: 'Gemma Scope 扩展至 Gemma 3 全层开放',
    p: '全系跳层 SAE + 释出训练代码，为分布式社区提供可直接消融的公开基座模型——本平台模板矩阵兼容其口径。' },
  { date: '2024-11', src: 'InterPLM / Stanford', tag: '跨域对齐', cat: 'sparse', title: '把 SAE 带到蛋白质语言模型 ESM-2',
    p: 'SAE 结论跨域同构的旁证：ESM-2 上叠加/分布式结论与 LLM 一致 → 支持本平台「多模板聚合出普适机制」的策略。' },
  { date: '2024-05', src: '开源社区', tag: '开放平台', cat: 'interp', title: 'Neuronpedia：SAE 特征的开放百科',
    p: '对象路由 + 激活示例的标准交互范式已被广泛复用；其开放上传接口与本平台「节点上传结果」模式可互操作。' },
  { date: '2024-01', src: 'SAEBench', tag: '评测基准', cat: 'sparse', title: 'SAE 质量的系统评测基准发布',
    p: '提出稀疏性-重建-下游效应多维评测；本平台模板测试的「四臂对照 + 独立复核」属于同类证据分级思想。' },
  { date: '2022-09', src: 'Anthropic', tag: '理论基础', cat: 'sparse', title: 'Toy Models of Superposition：叠加与特征相变',
    p: '「特征数 > 维度时以近正交方向叠加存储」——解释了模板矩阵里单坐标多条件混合（η² 中等）的普遍现象。' },
  { date: '2022-11', src: 'OpenAI', tag: '可视化', cat: 'interp', title: 'Microscope 归档：神经元可视化集合',
    p: '已停止更新，但「对象 → 多视图」范式被后续工具继承，是本平台三透镜路由的直系源头之一。' },
];

/* 机制可解释性领域 · 阶段时间轴（行业进展 tab 顶部）
   五个阶段：years / title / goal（该阶段的目标）/ outcome（关键成果）/ papers（代表性论文）
   论文链接均为真实官方源（distill.pub / transformer-circuits.pub / arXiv / deepmind.google / neuronpedia.org）；
   最右「最新节点」不在此登记——由 GET /api/news 实时流填充（LensProgress.jsx EraTimeline） */
export const MI_ERAS = [
  {
    id: 'era1', years: '2014–2019', title: '打开黑箱', cat: 'interp',
    goal: '对 CNN 与早期神经网络建立「看进去」的工具：神经元在看什么、哪些输入区域决定输出。确立「内部表征可以被系统性提问」的研究立场。',
    outcome: '特征可视化、显著图、探针（probing）三件套成型；Distill 特刊把可解释性做成独立发表载体，「对象 → 多视图」交互范式沿用至今。',
    papers: [
      { date: '2014-11', title: 'Visualizing and Understanding Convolutional Networks', src: 'arXiv', tag: '可视化', url: 'https://arxiv.org/abs/1311.2901', p: '反卷积把每层学到的模式显影——「逐层问网络学到了什么」的起点。' },
      { date: '2017-06', title: 'Feature Visualization（Distill 特刊）', src: 'Distill', tag: '可视化', url: 'https://distill.pub/2017/feature-visualization/', p: '激活最大化 + 正则化系统化：单个神经元/通道/层各在看什么，交互式可复现出版。' },
      { date: '2018-06', title: 'The Building Blocks of Interpretability', src: 'Distill', tag: '交互范式', url: 'https://distill.pub/2018/building-blocks/', p: '把「特征 → 人可读界面」组装成标准积木；本平台三透镜的对象路由直系源头之一。' },
    ],
  },
  {
    id: 'era2', years: '2020–2021', title: '电路假说', cat: 'causal',
    goal: '超越单神经元：提出「回路（circuit）」为理解单位——权重里能否找到可读、可人工验证的算法，而非事后统计相关。',
    outcome: '视觉回路人肉验证成功（曲线检测器）；Transformer Circuits 数学框架给出 residual stream / QK·OV 分解语言；induction heads 把「回路」接上 in-context learning 行为。',
    papers: [
      { date: '2020-04', title: 'Zoom In: An Introduction to Circuits', src: 'Distill', tag: '理论基础', url: 'https://distill.pub/2020/circuits/zoom-in/', p: 'Circuits 纲领宣言：特征由权重实现、回路由特征组成——主张「重要性权重里的真算法可被读出」。' },
      { date: '2020-06', title: 'Curve Detectors（Curve Circuits）', src: 'Distill', tag: '人工验证', url: 'https://distill.pub/2020/circuits/curve-detectors/', p: '在 InceptionV1 里逐 head 拼出完整的曲线检测回路——「可读算法」从假说变成实证。' },
      { date: '2021-09', title: 'A Mathematical Framework for Transformer Circuits', src: 'Anthropic', tag: '理论基础', url: 'https://transformer-circuits.pub/2021/framework/index.html', p: 'residual stream 视角 + QK/OV 电路分解：LLM 机制分析的标准坐标系，本项目残差流全场采集同源。' },
    ],
  },
  {
    id: 'era3', years: '2022–2023', title: '叠加与稀疏字典', cat: 'sparse',
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
    id: 'era4', years: '2023–2024', title: '规模化与开放生态', cat: 'sparse',
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
    id: 'era5', years: '2024–2025', title: '归因图与机制生物学', cat: 'causal',
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
    id: 'rsa1', years: '2008–2014', title: '认知神经科学起源', cat: 'repr',
    goal: '在 fMRI / 电生理里找一种不依赖坐标对齐的表征比较方法：MVPA 只能做分类判断，需要一种能描述「表征的关系结构」的度量。',
    outcome: 'RSA 开山论文把人 fMRI、猴电生理与行为数据放进同一 RDM 比较空间；方法纲领确立二阶同构主张；RSA Toolbox 工程化定型——RDM 成为神经科学标准工具。',
    papers: [
      { date: '2008-10', title: 'Matching Categorical Object Representations in Inferior Temporal Cortex of Man and Monkey', src: 'Neuron', tag: '开山之作', url: 'https://doi.org/10.1016/j.neuron.2008.10.043', p: 'RSA 创始论文：用 RDM 把人 fMRI、猴电生理与行为数据放进同一比较空间——「表征的关系结构可比」自此成为标准范式。' },
      { date: '2008-11', title: 'Representational Similarity Analysis — Connecting the Branches of Systems Neuroscience', src: 'Frontiers', tag: '方法纲领', url: 'https://doi.org/10.3389/neuro.06.004.2008', p: 'RSA 方法宣言：二阶同构（second-order isomorphism）主张——系统间比较不必坐标对齐，比较关系结构即可。' },
      { date: '2014-03', title: 'A Toolbox for Representational Similarity Analysis', src: 'PLOS Comput Biol', tag: '工具箱', url: 'https://doi.org/10.1371/journal.pcbi.1003553', p: 'MATLAB RSA 工具箱 + 统计推断流程（bootstrap/置换检验），RSA 方法的工程化定型。' },
    ],
  },
  {
    id: 'rsa2', years: '2014–2017', title: '深度网络对齐', cat: 'repr',
    goal: '回答「DNN 是不是好的大脑模型」：把 CNN 的表征空间放进 RDM，与灵长类腹侧视觉通路（IT 皮层）直接比较。',
    outcome: '目标驱动优化出的层级模型预测高阶视觉皮层响应；监督 CNN（而非无监督模型）解释 IT 表征——RSA 成为「神经网络↔大脑」比较的标准桥，Brain-Score 等基准随后成型。',
    papers: [
      { date: '2014-05', title: 'Performance-optimized Hierarchical Models Predict Neural Responses in Higher Visual Cortex', src: 'Nature Neurosci', tag: '深度网络', url: 'https://doi.org/10.1038/nn.3895', p: '目标驱动层级模型对齐 IT 皮层——「用任务优化出的 DNN 表征直接预测神经响应」的奠基工作。' },
      { date: '2014-11', title: 'Deep Supervised, Not Trained, Models Explain IT Representations', src: 'PLOS Comput Biol', tag: '深度网络', url: 'https://doi.org/10.1371/journal.pcbi.1003963', p: '首个把 CNN 的 RDM 与灵长类 IT 皮层对齐的经典研究：监督训练是关键，自监督/无监督不够。' },
    ],
  },
  {
    id: 'rsa3', years: '2017–2020', title: 'DNN 相似度方法学', cat: 'repr',
    goal: '为「比较两个网络」发展专门方法学：RSA 之外，子空间、核函数与不变性权衡——模型间比较需要处理各向异性缩放、正交变换等自由度。',
    outcome: 'SVCCA（子空间主角度）与 CKA（中心核对齐）相继提出，CKA 因对正交变换与各向同性缩放不变、并指出线性回归/CCA 的缺陷，成为模型间比较的事实标准——与 RSA 并列构成表征比较工具箱。',
    papers: [
      { date: '2017-06', title: 'SVCCA: Singular Vector Canonical Correlation Analysis', src: 'arXiv', tag: '相似度方法', url: 'https://arxiv.org/abs/1706.05806', p: '子空间层面比较两网络的表征：层间收敛曲线、跨训练相似度——RSA 思想在 DNN 内部的直接推广。' },
      { date: '2019-06', title: 'Similarity of Neural Network Representations Revisited (CKA)', src: 'arXiv', tag: '相似度方法', url: 'https://arxiv.org/abs/1905.00414', p: '中心核对齐 CKA：对正交变换与各向同性缩放不变——指出线性回归/CCA 的缺陷，成为模型间比较事实标准。' },
    ],
  },
  {
    id: 'rsa4', years: '2021–至今', title: '跨模态与规模化', cat: 'repr',
    goal: '从「网络↔大脑」扩展到「模型↔模型」的规模化比较：不同架构、模态、规模的模型表征是否收敛到同一结构。',
    outcome: 'feature-reweighted RSA 用脑/行为数据重加权模型特征提升拟合；Platonic Representation Hypothesis 用表征相似性度量发现跨模态模型随规模收敛到彼此对齐的表征空间——表征比较成为大模型时代的核心证据类型。',
    papers: [
      { date: '2024-05', title: 'The Platonic Representation Hypothesis', src: 'arXiv', tag: '跨模型', url: 'https://arxiv.org/abs/2405.07987', p: '用表征相似性度量发现：不同模态/架构的模型随规模收敛到彼此对齐的表征空间——「关系结构可比」的规模化验证。' },
    ],
  },
];
export const RSA_NOTE = '对比：本项目 TM 协议已逐 cell 保存 cos 方向矩阵（means.npz），RDM 可由 cell-cos 直接构造——RSA 是 AGG-v1 跨节点聚合的候选补充指标（跨模型/跨模板比较）；差异在于 RSA 比较整体关系结构、η² 方差分解按因子归位，二者正交而非替代。';

/* 分析技术注册表（design/client_analysis_tech_plan_v1.md §2，协议第六件；M7 扩容四分类）
   每技术：id / category（四类归属）/ name / input（数据契约，决定可用性）/ output（协议渲染形态）/
   metric_version（口径登记，禁止跨量纲平均 13.1）/ evidence_level / status / source / note。
   status 与 Q04/Q06 装置状态机同构：available=可运行 · planned=装置待建 · disabled=缺输入（disabled_reason 登记）。
   可用性由 LensData TechPanel 按 input 契约对当前选中 detail 匹配得出，不硬编码；
   JSx 中不得出现技术字面量以外的注册信息（红线扩展到技术层）。 */
export const ANALYSES = [
  {
    id: 'rsa-rdm', category: 'repr', name: 'RSA / RDM 相似性结构', input: ['result.cos'], output: 'rdm-matrix',
    metric_version: 'client-v0(1−cos)', evidence_level: 'observed', status: 'available', source: 'builtin',
    note: '从 cell 方向 cos 矩阵现算 RDM（相异度=1−cos，取值 0–2）：比较「条件间关系结构」而非坐标——跨模板可比的候选口径。',
  },
  {
    id: 'eta2-decompose', category: 'repr', name: 'η² 方差分解', input: ['result.summary.eta2_by_factor'], output: 'eta2-bars',
    metric_version: 'summary_digest-v2', evidence_level: 'observed', status: 'available', source: 'builtin',
    note: '按因子归位的逐维方差分解视图：η²(A)+η²(B)+η²(interaction)=1（双因子模板），交互项=组合特征领地。',
  },
  {
    id: 'cka-linear', category: 'repr', name: 'CKA 中心核对齐', input: ['result.cos', 'result2.cos'], output: 'similarity',
    metric_version: 'client-v0', evidence_level: 'candidate', status: 'available', source: 'builtin',
    disabled_reason: 'P2 开放：需第二个结果做对齐比较（多选两行）',
    note: '对正交变换与各向同性缩放不变的表征相似度——跨模型比较的事实标准；需双结果输入。',
  },
  {
    id: 'pca-proj', category: 'repr', name: '主轴 PCA 投影', input: ['result.activations'], output: 'projection',
    metric_version: 'planned-v0', evidence_level: 'candidate', status: 'planned', source: 'builtin',
    note: '残差流激活的主成分投影坐标（有效秩/主轴三段分析的前置产物）；需激活级采集，装置待建。',
  },
  {
    id: 'sae-topk', category: 'sparse', name: 'SAE top-k 特征编码', input: ['result.activations'], output: 'feature-list',
    metric_version: 'planned-v0', evidence_level: 'candidate', status: 'planned', source: 'builtin',
    note: '稀疏自编码器把残差流分解为单义特征字典，取 top-k 激活特征——「多义→单义」的工程解（TM 结果库暂无激活级 npz）。',
  },
  {
    id: 'probe-linear', category: 'causal', name: '线性探测', input: ['result.activations'], output: 'probe-score',
    metric_version: 'planned-v0', evidence_level: 'candidate', status: 'planned', source: 'builtin',
    note: '线性探针读出条件可分性（准确率/惩罚化得分）：表示「信息在不在」，因果性弱——与干预类技术配对使用。',
  },
  {
    id: 'patching-swap', category: 'causal', name: '激活修补（patching）', input: ['result.activations_pairs'], output: 'delta-rate',
    metric_version: 'planned-v0', evidence_level: 'candidate', status: 'planned', source: 'builtin',
    note: '最小对激活互换后观测 logits 变化：η² 定编码、修补定因果——「干预定因果」元法则的技术落地；需成对激活采集。',
  },
  {
    id: 'autointerp-notes', category: 'interp', name: '自动解释草稿', input: ['result.prev_sha'], output: 'interp-notes',
    metric_version: 'planned-v0', evidence_level: 'candidate', status: 'planned', source: 'builtin',
    note: '对前序技术产物（特征/簇/方向）生成 LLM 解释草稿 + 激活例证：人审后降级为 descriptive 注记，不直接升级证据等级。',
  },
];

/* 平台定位（总览 hero，M8）：品牌 + 标题 + 「手段→路径→目标」三段链 */
export const MISSION = {
  brand: 'AI2050',
  title: 'LLM 逆向工程平台',
  chain: [
    { k: '手段', v: '语言模板 × 分析技术' },
    { k: '路径', v: '从行为反推机制' },
    { k: '目标', v: '原理 → 智能理论 → AGI' },
  ],
};

/* 分析技术四大分类（M7-P0，design/tech_categories_four_views_plan_v1.md §P0.1）
   平台目标 = 多语言模板 × 多类技术提取特征；四类是一级分类，四界面围绕它组织。
   status: measured=结果库已有该类产物 · device_built=装置已建待测 · planned=未实现 */
export const TECH_CATEGORIES = [
  { id: 'sparse', name: '稀疏分解与字典学习', short: '稀疏分解', color: '#6d28d9', status: 'planned', cost: '激活采集 + SAE 训练 · ~2h/节点',
    methods: ['SAE 稀疏自编码器', '字典学习', 'NMF'], input: 'result.activations',
    note: '把叠加的多义激活解开成单义特征字典——特征提取的主力技术（Gemma Scope / Towards Monosemanticity 同源）。' },
  { id: 'causal', name: '探测与因果干预', short: '探测干预', color: '#d85a30', status: 'planned', cost: '成对激活采集 · ~1h/节点',
    methods: ['线性探测', '激活修补', '定向消融'], input: 'result.activations_pairs',
    note: '探测答「信息在不在」，修补答「是不是它在起作用」——平台元法则「干预定因果」的技术落地。' },
  { id: 'repr', name: '表示空间分析', short: '表示分析', color: '#0284c7', status: 'measured', cost: '结果库现算 · 秒级',
    methods: ['RSA/RDM', 'CKA', 'η² 分解', 'PCA'], input: 'result.cos',
    note: '比较「关系结构」而非坐标：跨模板/跨模型可比，平台现有结果库直接支撑的三项全在此类。' },
  { id: 'interp', name: '可视化与自动化解释', short: '可视解释', color: '#059669', status: 'device_built', cost: 'LLM 调用 · 按产物量',
    methods: ['特征看板', '聚类标注', 'LLM 自动解释'], input: 'result.prev_sha',
    note: '把技术产物翻译成人可读界面与草稿解释；可视化已随四透镜建成，自动解释待 LLM 接入。' },
];

/* 空间透镜 · 投影层注册表（M7-P1）：3D 点云按技术产物切换投影——数据契约驱动可用性
   render: cloud-cos=当前点云视图 · panel=选中后显示说明面板（含跳转） */
export const PROJECTIONS = [
  { id: 'cos-struct', cat: 'repr', name: '方向结构', render: 'cloud-cos', input: 'result.cos', status: 'available',
    note: '点云按 cell 方向 cos 聚簇（当前默认视图）：簇=关系族，颜色=模板 demo 包 cloud 族。' },
  { id: 'rdm-mds', cat: 'repr', name: 'RDM-MDS 布局', render: 'panel', input: 'result.cos', status: 'available',
    note: 'RDM（1−cos）经 MDS 压到 3D 的「关系结构布局」——数据透镜已可现算 RDM 热图；3D MDS 坐标待 P2 激活级重算。' },
  { id: 'sae-heat', cat: 'sparse', name: 'SAE 特征热图', render: 'panel', input: 'result.activations', status: 'planned',
    note: '点按 SAE 特征激活值着色 + top-k 特征清单——需激活级采集，装置待建（sae-topk 转可用后自动点亮）。' },
  { id: 'probe-axis', cat: 'causal', name: '探测判别轴', render: 'panel', input: 'result.activations', status: 'planned',
    note: '沿线性探针法向投影 + 类间分离度——需激活级采集（probe-linear 转可用后自动点亮）。' },
  { id: 'patch-compare', cat: 'causal', name: '干预前后对比', render: 'panel', input: 'result.activations_pairs', status: 'planned',
    note: '最小对激活互换前后双点云 + 位移箭头——需成对激活采集（patching-swap 转可用后自动点亮）。' },
  { id: 'interp-label', cat: 'interp', name: '自动解释标注', render: 'panel', input: 'result.prev_sha', status: 'planned',
    note: '簇质心挂自动解释标签——待 autointerp-notes 接入 LLM 后开放。' },
];

/* 覆盖度矩阵 DEMO 叙事（M7-P0）：对象 × 语言模板 × 技术四类 → 结果空间占用
   status: done=已有该类结果 · pending=结果部分到位/装置已建 · empty=空格（→ 缺口任务）
   真源（P2）= GET /api/coverage（结果库按 sha→(对象,模板,技术) 聚合）；前端先 DEMO 后 LIVE。
   现状依据：repr 类三项可用且 TM-04/05/07 有实测结果；其余类 planned → pending/empty */
export const DEMO_COVERAGE = [
  { tpl: 'is_a', cat: 'repr', status: 'done' },
  { tpl: 'is_a', cat: 'sparse', status: 'pending' },
  { tpl: 'is_a', cat: 'causal', status: 'pending' },
  { tpl: 'is_a', cat: 'interp', status: 'empty' },
  { tpl: 'attr', cat: 'repr', status: 'pending' },
  { tpl: 'attr', cat: 'sparse', status: 'empty' },
  { tpl: 'attr', cat: 'causal', status: 'empty' },
  { tpl: 'attr', cat: 'interp', status: 'empty' },
  { tpl: 'syntax', cat: 'repr', status: 'empty' },
  { tpl: 'syntax', cat: 'sparse', status: 'empty' },
  { tpl: 'syntax', cat: 'causal', status: 'empty' },
  { tpl: 'syntax', cat: 'interp', status: 'empty' },
  { tpl: 'part_of', cat: 'repr', status: 'empty' },
  { tpl: 'part_of', cat: 'sparse', status: 'empty' },
  { tpl: 'part_of', cat: 'causal', status: 'empty' },
  { tpl: 'part_of', cat: 'interp', status: 'empty' },
];

/* RSA/RDM 技术的 DEMO 回退输出（离线或 detail 无 cos 时演示协议结构；6 条件演示矩阵） */
export const DEMO_RDM = {
  keys: ['content=苹果', 'content=香蕉', 'prefix=甲', 'prefix=乙', 'position=首', 'position=尾'],
  matrix: [
    [0.00, 0.32, 0.91, 0.87, 1.12, 1.08],
    [0.32, 0.00, 0.88, 0.94, 1.05, 1.15],
    [0.91, 0.88, 0.00, 0.29, 1.31, 1.22],
    [0.87, 0.94, 0.29, 0.00, 1.28, 1.19],
    [1.12, 1.05, 1.31, 1.28, 0.00, 0.41],
    [1.08, 1.15, 1.22, 1.19, 0.41, 0.00],
  ],
};

/* 语言模板注册表（design/client_template_tech_plan_v1.md §2，协议第六·二件，M5-P0）
   「is-a」不是平台常量而是一个可选模板：每个模板登记 prompt 句式 / 神经元与点云两模式的
   demo 包（token 序列、叙事峰位、注意力回看、焦点特征、点云族、激活示例）与指标覆盖。
   status: measured=已有实测指标挂接（is_a ← TM-07/Q03）；demo=演示占位（接入 Q20 扩族后转 LIVE）。
   红线：模板叙事内容只允许存在于本注册表——任何组件内不得出现 is-a/水果等模板字面量；
   切换模板 = 换 demo 种子，四个透镜渲染器零改动。
   未测量的模板 metrics 为空 → 指标位显式显示「未测量」，禁止复用 is-a 数字（量纲纪律模板版）。 */
export const LANG_TEMPLATES = [
  {
    id: 'is_a', name: 'is-a 上位/下位', pattern: '{X} 是一种 {Y}', status: 'measured',
    desc: '关系族四臂消融（TM-07）：上位词方向已获机制证据（mech_evidence）。',
    metrics: { E_read: 0.331615 }, metric_version: 'v4',
    demo: {
      tokens: ['苹果', '是', '一种', '水果', '，', '富含', '维生素', '。'],
      peaks: { mlp: 3, res: 4, attn: 3 },
      attn_lookback: { from: 2, to: 0, note: '完成关系绑定' },
      focus: { id: 'F#3734', unit: 3734, label: 'is-a 上位关系：水果族', short: 'is-a 水果族', pos: [0.42, -1.18, 0.73] },
      neighbors: ['F#1092', 'F#2210', 'F#3655', 'F#0821'],
      cloud: [
        { c: [0.9, -0.6, 0.4], col: '#10b981', n: 120, s: 0.85, label: '水果族 · is-a' },
        { c: [-1.0, 0.5, -0.5], col: '#0ea5e9', n: 90, s: 0.95, label: '属性轴 · size/speed' },
        { c: [-0.2, -0.9, 1.0], col: '#f59e0b', n: 70, s: 0.75, label: '语法功能' },
      ],
      act_lines: [
        [['苹果', 0], ['是', 0], ['一种', 0], ['水果', 0.95], ['，', 0], ['富含', 0.45], ['维生素', 0]],
        [['香蕉', 0], ['、', 0], ['菠萝', 0], ['都', 0], ['属于', 0], ['水果', 0.95]],
        [['The', 0], ['crinoletta', 0], ['is', 0], ['a', 0], ['hybrid', 0.55], ['apple', 0.75], ['variety', 0.35]],
      ],
    },
  },
  {
    id: 'attr', name: '属性关系', pattern: '{X} 是 {ADJ} 的', status: 'demo',
    desc: '属性谓词方向（TM-08 关系族）：size/speed/材质属性轴，接入 Q20 扩 target 族后转 LIVE。',
    metrics: {}, metric_version: null,
    demo: {
      tokens: ['石头', '很', '坚硬', '，', '海绵', '却', '柔软', '。'],
      peaks: { mlp: 2, res: 3, attn: 2 },
      attn_lookback: { from: 3, to: 0, note: '完成属性主体绑定' },
      focus: { id: 'F#2150', unit: 2150, label: '属性关系：材质轴（DEMO）', short: 'attr 材质轴', pos: [-0.55, 0.4, -0.62] },
      neighbors: ['F#0917', 'F#1442', 'F#3025', 'F#3388'],
      cloud: [
        { c: [-0.8, -0.5, 0.5], col: '#0ea5e9', n: 120, s: 0.85, label: '材质轴 · attr（DEMO）' },
        { c: [0.7, 0.4, -0.5], col: '#10b981', n: 90, s: 0.95, label: '尺寸轴 · size（DEMO）' },
        { c: [0.1, -0.7, 0.9], col: '#f59e0b', n: 70, s: 0.75, label: '速度轴 · speed（DEMO）' },
      ],
      act_lines: [
        [['石头', 0], ['很', 0], ['坚硬', 0.9], ['，', 0], ['海绵', 0.1], ['却', 0], ['柔软', 0.7]],
        [['这块', 0], ['金属', 0], ['十分', 0], ['坚硬', 0.88]],
        [['The', 0], ['sponge', 0.1], ['is', 0], ['soft', 0.85], ['and', 0], ['porous', 0.4]],
      ],
    },
  },
  {
    id: 'syntax', name: '句法角色', pattern: '{X} {V} {Y}（语序消融）', status: 'demo',
    desc: '句法结构扰动（TM-09）：主谓宾角色指派与语序置换家族对照，演示占位。',
    metrics: {}, metric_version: null,
    demo: {
      tokens: ['猫', '追', '狗', '，', '狗', '却', '躲开', '。'],
      peaks: { mlp: 2, res: 3, attn: 1 },
      attn_lookback: { from: 2, to: 0, note: '完成施事角色指派' },
      focus: { id: 'F#0487', unit: 487, label: '句法角色：施事标记（DEMO）', short: 'syntax 施事', pos: [0.3, 0.7, -0.4] },
      neighbors: ['F#0211', 'F#1056', 'F#2703', 'F#3199'],
      cloud: [
        { c: [0.6, 0.8, -0.3], col: '#f59e0b', n: 120, s: 0.85, label: '施事标记 · syntax（DEMO）' },
        { c: [-0.6, 0.3, 0.8], col: '#0ea5e9', n: 90, s: 0.95, label: '受事标记（DEMO）' },
        { c: [0.1, -0.6, 0.7], col: '#10b981', n: 70, s: 0.75, label: '语序敏感簇（DEMO）' },
      ],
      act_lines: [
        [['猫', 0.85], ['追', 0.3], ['狗', 0.1], ['，', 0], ['狗', 0.85], ['却', 0], ['躲开', 0.4]],
        [['狗', 0.85], ['被', 0], ['猫', 0.1], ['追', 0.3]],
        [['The', 0], ['cat', 0.85], ['chases', 0.3], ['the', 0], ['dog', 0.1]],
      ],
    },
  },
  {
    id: 'part_of', name: '部分-整体', pattern: '{X} 的 {Y}', status: 'demo',
    desc: '部分-整体关系（meronymy）：整体激活带动部件指向，演示占位。',
    metrics: {}, metric_version: null,
    demo: {
      tokens: ['车轮', '是', '汽车', '的', '部件', '，', '可', '更换', '。'],
      peaks: { mlp: 2, res: 4, attn: 2 },
      attn_lookback: { from: 3, to: 2, note: '回看整体词完成部分-整体绑定' },
      focus: { id: 'F#1290', unit: 1290, label: '部分-整体：车轮→汽车（DEMO）', short: 'part-of 车轮', pos: [0.5, -0.3, 0.6] },
      neighbors: ['F#0066', 'F#1178', 'F#2451', 'F#3010'],
      cloud: [
        { c: [0.4, -0.4, 0.7], col: '#10b981', n: 120, s: 0.85, label: '部件指向 · part-of（DEMO）' },
        { c: [-0.7, 0.6, -0.2], col: '#0ea5e9', n: 90, s: 0.95, label: '整体指向（DEMO）' },
        { c: [-0.1, 0.8, 0.4], col: '#f59e0b', n: 70, s: 0.75, label: '可拆卸性簇（DEMO）' },
      ],
      act_lines: [
        [['车轮', 0], ['是', 0], ['汽车', 0.8], ['的', 0.3], ['部件', 0.9]],
        [['汽车', 0.8], ['的', 0.3], ['车轮', 0.9], ['可', 0], ['更换', 0.4]],
        [['A', 0], ['wheel', 0.9], ['is', 0], ['part', 0.3], ['of', 0], ['a', 0], ['car', 0.8]],
      ],
    },
  },
];

/* 研究包注册表（协议第七件，design/distributed_flow_plan_v1.md §2.2，M6-P0 DEMO 占位）
   语言模板 × 分析技术 的一等组合；P1 服务端对应 GET /api/kits · /api/kits/{kit_id}/bundle；
   status: available=结果库已有满足 input 契约的结果（P1 由服务端判定）| pending_results=等首个结果 */
export const RESEARCH_KITS = [
  { kit_id: 'is_a__rsa-rdm', lang_tpl: 'is_a', analysis: 'rsa-rdm', status: 'available' },
  { kit_id: 'is_a__eta2-decompose', lang_tpl: 'is_a', analysis: 'eta2-decompose', status: 'available' },
  { kit_id: 'is_a__cka-linear', lang_tpl: 'is_a', analysis: 'cka-linear', status: 'pending_results' },
  { kit_id: 'attr__rsa-rdm', lang_tpl: 'attr', analysis: 'rsa-rdm', status: 'pending_results' },
  { kit_id: 'syntax__rsa-rdm', lang_tpl: 'syntax', analysis: 'rsa-rdm', status: 'pending_results' },
  { kit_id: 'part_of__rsa-rdm', lang_tpl: 'part_of', analysis: 'rsa-rdm', status: 'pending_results' },
  /* M7-P0：四类技术各补一个起步 kit（× is_a 模板）；status=pending_results=等首个满足契约的结果 */
  { kit_id: 'is_a__sae-topk', lang_tpl: 'is_a', analysis: 'sae-topk', status: 'pending_results' },
  { kit_id: 'is_a__probe-linear', lang_tpl: 'is_a', analysis: 'probe-linear', status: 'pending_results' },
  { kit_id: 'is_a__pca-proj', lang_tpl: 'is_a', analysis: 'pca-proj', status: 'pending_results' },
  { kit_id: 'is_a__autointerp-notes', lang_tpl: 'is_a', analysis: 'autointerp-notes', status: 'pending_results' },
];

/* 新人教程 · 按技术类别的四条上手路径（M7-P1）：复用 TUTORIAL_STEPS 骨架，
   差异只在第 4 步 download 的 --kit 选择；kit 必须在 RESEARCH_KITS 中登记 */
export const TUTORIAL_PATHS = [
  { cat: 'repr', kit: 'is_a__rsa-rdm', note: '唯一已有实测结果的类别——四步照做即可，第 4 步 kit 用左侧 ID。' },
  { cat: 'sparse', kit: 'is_a__sae-topk', note: '等待首个激活级结果（装置待建）；可先跑步骤 1–2 熟悉节点接入。' },
  { cat: 'causal', kit: 'is_a__probe-linear', note: '需成对激活采集；建议先在缺格任务里领 repr 类熟悉口径，再切本类。' },
  { cat: 'interp', kit: 'is_a__autointerp-notes', note: '输入是前序技术产物（prev_sha）——先完成任一前序 kit 再做本类。' },
];

/* 新人教程四步卡（使用说明 tab，M6-P0；2026-10-08 自研发平台移入）：接入→领任务→上传→下载研究
   命令属接入文档（非实验数据），登记于数据层；P1 后第 4 步 download 子命令转真。
   server 占位 <center> 在卡片内说明替换为中心节点地址 */
export const TUTORIAL_STEPS = [
  { n: 1, t: '接入节点', cmd: 'python node_agent.py register --server http://<center>:5010 --gpu <gpu> --model <model>',
    d: '向中心节点注册本机（GPU/模型登记，获取节点凭据）。详见 deploy/README_CENTOS.md。' },
  { n: 2, t: '领取任务', cmd: 'python node_agent.py run',
    d: '循环 claim 任务（租约 6h，heartbeat 自动续期）；按模板 corpus+runner 口径在本机执行。' },
  { n: 3, t: '完成上传', cmd: '（自动）sha256 分块上传 → /api/results/upload/init|chunk|finish',
    d: '结果按内容寻址落库；design_sha 与预注册冻结不符会被拒收——口径改动必须先注册新模板。' },
  { n: 4, t: '下载研究', cmd: 'python node_agent.py download --kit TM-04__rsa-rdm --out ./kit && cd kit/TM-04__rsa-rdm && python reproduce.py',
    d: '取走「模板 × 分析技术」研究包（corpus/runner/contract + 结果清单 + README + 一键复现脚本 + AGG-v1 跨节点 η²）；reproduce.py 自动完成 design_sha 冻结校验 → runner 复现 → 输出对账（加 --server <center>:5010 联网逐文件比对结果库）→ reproduce_report.txt（真实口径加 --model-path <HF目录>）。' },
];

/* 使用说明 tab（M6-P0 后续调整：新人相关内容从研发平台移至此处）
   intro = 平台一句话介绍；docs = 文档索引（真实存在的仓库文件/端点）；
   cli = node_agent 子命令速查（deploy/node_agent.py 实际注册的五个子命令） */
export const USAGE_DOC = {
  intro: 'AI2050 是一个分布式机制可解释性研究平台：语言模板（受控语料 + 固定口径）分发给任意节点执行，结果以 sha256 内容寻址上传中心节点；任何人可下载「模板 × 分析技术」研究包，在本地校验冻结一致后复现与观察。四条路线 tab 共享同一套证据口径（本机 → 平台 → 行业）。',
  docs: [
    { t: '节点接入指南', s: 'deploy/README_CENTOS.md', d: 'register → run 全流程部署说明（系统依赖、凭据落盘、后台运行）。' },
    { t: '分布式研发全流程设计', s: 'design/distributed_flow_plan_v1.md', d: '目录可见 / 新人教程 / 研究包下载三能力的方案与落地注记（M6，P0–P3 已落地）。' },
    { t: '分析技术注册表设计', s: 'design/client_analysis_tech_plan_v1.md', d: '「技术可插拔」协议第六件：输入契约 / 口径版本 / 证据级（M4）。' },
    { t: '语言模板注册表设计', s: 'design/client_template_tech_plan_v1.md', d: '「is-a → 可选语言模板」去硬编码方案（M5）。' },
    { t: '研究包清单', s: 'GET /api/kits', d: '服务端实时枚举「模板 × 分析技术」组合与可用性（平台进度 · 研究包组合同样可查）。' },
  ],
  cli: [
    ['register', '注册本机节点（获取凭据）'],
    ['run', '循环领取并执行任务（租约 6h 自动续）'],
    ['status', '本机节点状态 + 最近任务（需凭据）'],
    ['queue', '全队列快照（公开只读，免凭据）'],
    ['download', '下载研究包并本地复现（--kit <tm>__<analysis>）'],
  ],
};

/* 调度队列只读视图 DEMO 快照（M6-P2，流程透镜「调度」tab 离线回退）
   真源 = GET /api/tasks（S8 公开只读：活跃租约 + 完成/失败/过期统计）；
   概念消歧：调度队列=节点租约实时状态（谁在领什么），研究队列=phase_queue 依赖序——两层不同 */
export const DEMO_SCHED = {
  stats: { claimed: 2, done: 14, failed: 1, expired: 3 },
  lease_hours: 6,
  active: [
    { task_id: 'T-demo4f21a0c8', tm_id: 'TM-04', node_name: 'node-a3f7', model: 'Qwen3-4B', dtype: 'bf16', remain_h: 4.7 },
    { task_id: 'T-demo9b17e3d2', tm_id: 'TM-06', node_name: 'node-c2e9', model: 'GLM4-9B', dtype: '4-bit NF4', remain_h: 1.3 },
  ],
};

/* 空间透镜 · 实时分析模式的 DEMO 快照（LIVE 真源 = 服务端 /api/live/connect：读 config+权重头、不加载权重）
   schema 与服务端 _scan_live_model 输出一致：meta/config/total_params/dtypes/files/groups[]
   结构数字为 Qwen3-4B 公开架构（demo 叙事，非本机测量）；groups 仅展开 layers.0，其余层同构略 */
export const LIVE_MODEL_DEMO = {
  model_path: '<本机模型目录>（DEMO 叙事 · 未连接）',
  arch: 'Qwen3ForCausalLM',
  config: { layers: 36, hidden: 2560, heads: 32, kv_heads: 8, vocab: 151936, tie_word_embeddings: true, torch_dtype: 'bfloat16' },
  total_params: 4022689280,
  dtypes: { bf16: 4022689280 },
  files: [{ name: 'model-00001-of-00002.safetensors（demo）', bytes: 4998827008 }, { name: 'model-00002-of-00002.safetensors（demo）', bytes: 3066141696 }],
  demo_note: 'DEMO 快照：结构数字为 Qwen3-4B 公开架构，非本机权重。连接本地模型后此处显示真实参数表（config + safetensors 权重头，秒级、不加载权重）。',
  groups: [
    { prefix: 'model.embed_tokens', tensor_count: 1, params: 388956160, tensors: [
      { name: 'model.embed_tokens.weight', shape: [151936, 2560], dtype: 'bf16', params: 388956160 },
    ] },
    { prefix: 'model.layers.0', tensor_count: 14, params: 100936960, tensors: [
      { name: 'model.layers.0.input_layernorm.weight', shape: [2560], dtype: 'bf16', params: 2560 },
      { name: 'model.layers.0.self_attn.q_proj.weight', shape: [4096, 2560], dtype: 'bf16', params: 10485760 },
      { name: 'model.layers.0.self_attn.q_proj.bias', shape: [4096], dtype: 'bf16', params: 4096 },
      { name: 'model.layers.0.self_attn.q_norm.weight', shape: [128], dtype: 'bf16', params: 128 },
      { name: 'model.layers.0.self_attn.k_proj.weight', shape: [1024, 2560], dtype: 'bf16', params: 2621440 },
      { name: 'model.layers.0.self_attn.k_proj.bias', shape: [1024], dtype: 'bf16', params: 1024 },
      { name: 'model.layers.0.self_attn.k_norm.weight', shape: [128], dtype: 'bf16', params: 128 },
      { name: 'model.layers.0.self_attn.v_proj.weight', shape: [1024, 2560], dtype: 'bf16', params: 2621440 },
      { name: 'model.layers.0.self_attn.v_proj.bias', shape: [1024], dtype: 'bf16', params: 1024 },
      { name: 'model.layers.0.self_attn.o_proj.weight', shape: [2560, 4096], dtype: 'bf16', params: 10485760 },
      { name: 'model.layers.0.mlp.gate_proj.weight', shape: [9728, 2560], dtype: 'bf16', params: 24903680 },
      { name: 'model.layers.0.mlp.up_proj.weight', shape: [9728, 2560], dtype: 'bf16', params: 24903680 },
      { name: 'model.layers.0.mlp.down_proj.weight', shape: [2560, 9728], dtype: 'bf16', params: 24903680 },
      { name: 'model.layers.0.post_attention_layernorm.weight', shape: [2560], dtype: 'bf16', params: 2560 },
    ] },
    { prefix: 'model.norm', tensor_count: 1, params: 2560, tensors: [
      { name: 'model.norm.weight', shape: [2560], dtype: 'bf16', params: 2560 },
    ] },
  ],
};
