/* 分布式研发平台 demo 数据
   叙事：大量分布式机器各领一个「模板测试」（受控最小对/扫描句族），
   跑完把测试结果上传服务器；其他人可下载；服务器综合全部模板结果，
   分析 LLM 的语言编码机制（坐标级 η² 热图 + 方向级 cos 矩阵）。
   接入点（后端 API 未实现，当前全部为演示数据）：
   - GET  /api/distributed/summary    → 平台统计 + 节点列表
   - GET  /api/distributed/templates  → 模板矩阵 + 聚合状态
   - GET  /api/distributed/news       → 业界新闻（RSS / arXiv 抓取）
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
  { id: 'TM-04', name: '距离扫描（句首 / 中 / 尾）', dim: '位置', nodes: 5, cells: 962, agg: 'collecting' },
  { id: 'TM-05', name: '实体置换家族（同义互换）', dim: '内容', nodes: 7, cells: 1418, agg: 'collecting' },
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

/* 业界新闻（demo 数据；接 /news 后由 RSS / arXiv / 官方博客抓取自动更新）
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
