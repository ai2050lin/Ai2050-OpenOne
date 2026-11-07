/* 主线研发状态（ResearchPlan v6，2026-11-07 登记）。
   这是研究记录的登记表，不是推算结果：阶段完成、锁箱开封、证据晋级都要在研究记录里有对应 Phase 后才改这里。
   界面只读本文件和接口；百分比只在分母明确时显示。 */

/* H0–H7：v6 第 8 节的阶段、完成定义与过门清单。done 为已满足的过门项序号。 */
export const STAGES = [
  { id: 'H0', name: '明确操作与备择机制', def: '每种结果会改变哪项判断写入主张卡，完成行为门和简单基线', gates: ['主张卡写好至少两个可区分的假设', '行为门通过', '简单基线已跑'], done: [0],
    note: 'MA-H0-v3（Phase 321）：主张卡与预注册已冻结。行为门未过：资料范围全部答对，现实范围未过中文 5 格、英文 13 格（主要是“第几条记录里的东西”式提问和英文带草稿）。预注册的基线判据按格取最好启发式，恒为 1.0，设计有误，需在下一版重做。' },
  { id: 'H1', name: '冻结材料与统计', def: '主效应、交互、保护、排除及多重比较口径完整，确认材料保持独立', gates: ['discovery、confirm、lockbox 已建', '功效三栏通过（检验力 ≥ 0.8）', '确认与锁箱已封存'], done: [] },
  { id: 'H2', name: '全场采集与响应发现', def: '原始与标准化视图可重算，先给出单因素及组合的响应脉络', gates: ['采集清单与 manifest 哈希', '原始和标准化视图可重算', '单因素与组合响应脉络'], done: [] },
  { id: 'H3', name: '冻结载体与预测规则', def: '在新材料上预测因素响应和角色绑定，并保留未解释部分', gates: ['方法同题比较', '载体与规则已冻结', '新材料预测与残差'], done: [] },
  { id: 'H4', name: '原生使用与交互验证', def: '明确哪条条件规则受支持，哪些仍是观察或注入效应', gates: ['QK、V/O、MLP 分项干预', '剂量与匹配负对照', '能力保持'], done: [] },
  { id: 'H5', name: '未见组合独立预测', def: '超过同信息预算基线，报告具体错误、首个分叉与条件失效', gates: ['未见组合预测', '全词表 KL 与生成', '超过同预算基线'], done: [] },
  { id: 'H6', name: '历史与分支验证', def: '同历史和不同历史的结果分别可复算，正确归属后续差异', gates: ['直接效应', '公共续写路径', '各自生成历史'], done: [] },
  { id: 'H7', name: '复算、登记与推广', def: '模型、材料、运行、证据身份齐全，可按因素、角色和路径反查', gates: ['第二物理节点复算', '正式登记', '二维与 3D 可反查'], done: [] },
];

export const SLICES = [
  ['A 共同底座', '长文属性绑定，文档与现实范围、所选记录、错误草稿', '行为与反例基准'],
  ['B 两因素交互', '逐批加入输出语言、表达方式、条件或否定、句法角色', '响应脉络、交互热力图'],
  ['C 原生路径', '读取来源、MLP 条件变换、下游输出', '干预、恢复、剂量'],
  ['D 组合预测', '新实体、未见因素对、1 / 3 / 6 步生成', '分布预测、错误复现、首个分叉'],
  ['E 正式积累', '上传、去重、独立复算、线上回查', '关联图谱碎片'],
].map(([name, task, output]) => ({ name, task, output, done: false }));

export const PAIRS = [
  ['风格 × 风格', '代码与说明文字切换', '门控链'],
  ['逻辑 × 知识', '反事实前提与事实先验', 'CF-v2、属性绑定'],
  ['风格 × 逻辑', '押韵与语义通顺，正式语气与转折', 'FU-v2'],
  ['语言 × 语法', '中英语序、翻译中的语序承诺', '新增'],
  ['语法 × 逻辑', '语法合法但逻辑矛盾', '新增'],
  ['语言 × 风格', '中文诗与英文诗、中文注释与英文代码', '新增'],
].map(([pair, design, base]) => ({ pair, design, base, built: false }));

export const SPLITS = [
  { id: 'discovery', use: '候选筛选、曲线拟合和定位', built: false },
  { id: 'confirm', use: '确认', built: false },
  { id: 'lockbox', use: '一次性留出', built: false },
];

/* 四组成绩：v6 第 11 节。全部未测。 */
export const SCORES = [
  ['新条件的响应和绑定预测', '只看前缀，预测内部响应和实体与属性的绑定'],
  ['全词表分布与生成', '全词表 KL、预测损失、自由生成'],
  ['干预方向与副作用', '事前预测方向和幅度，再看恢复和副作用'],
  ['未见组合与第二节点', '未见因素组合，以及第二物理节点复算'],
].map(([name, desc]) => ({ name, desc, value: null }));

/* 已完成研究运行的四关状态。evidence：未登记正式等级时为 null。 */
export const RUNS = [
  { id: 'MB-v1', phase: 315, title: '统一机制基准', executed: true, exec: '已完成，lockbox 已开封', checked: true, check: '身份门、行为门通过', evidence: null, evidenceText: '候选观测，未登记正式等级', replicated: false, replicaText: '未做' },
  { id: 'SP-v1', phase: 316, title: '特异性细化', executed: true, exec: '已完成，封存划分已开封', checked: true, check: '身份门、行为门通过，两格手工复算一致', evidence: null, evidenceText: '候选观测，未登记正式等级', replicated: false, replicaText: '第二节点未做' },
];

/* 锁箱看管：每个封存划分的状态。opened 后不能用于调参后的确认。 */
export const LOCKBOXES = [
  { id: 'MB-v1 lockbox', phase: 315, state: 'opened', note: 'Phase 315 评估时一次性开封' },
  { id: 'SP-v1 封存划分', phase: 316, state: 'opened', note: 'Phase 316 评估时一次性开封；材料不与 MB-v1 重叠' },
  { id: '主线 lockbox', phase: null, state: 'not_built', note: 'H1 冻结时建立' },
];

/* 方法基准：lockbox 的 cf 节点 auc_f（Phase 315），1% 预算特异性判定（Phase 316）。 */
export const BENCH = {
  mb: { head: ['任务', '随机', 'act_diff', 'atp_ig', 'exact'], rows: [
    ['GPT-2 首都', '.087', '.289', '.362', '.369'], ['GPT-2 IOI', '.014', '.006', '.185', '.195'], ['GPT-2 主谓', '.057', '.249', '.315', '.298'],
    ['Qwen3-4B 首都', '.050', '.053', '.402', '.262'], ['Qwen3-4B IOI', '.050', '.049', '.260', '.163'], ['Qwen3-4B 主谓', '.050', '.050', '.430', '.053']] },
  sp: { head: ['排序（1% 预算）', 'GPT-2', 'Qwen3-4B'], rows: [
    ['atp_ig、exact、exact_iter', '未支持', '未支持'], ['ig_contrast', '未支持', '未支持'], ['exact_contrast', '支持 3 / 3', '未支持'], ['ig_nobb', '支持 2 / 3', '支持 2 / 3']] },
  source: 'research/mech_bench/MB-v1、research/mech_bench/SP-v1',
};

export const pct = (done, total) => {
  if (!Number.isFinite(done) || !Number.isFinite(total) || total <= 0) return null;
  const value = 100 * done / total;
  return value === 0 || value === 100 ? value.toFixed(0) : value < 1 ? value.toFixed(2) : value.toFixed(0);
};

export function stageProgress(stages = STAGES) {
  const doneStages = stages.filter(stage => stage.done.length === stage.gates.length).length;
  const next = stages.find(stage => stage.done.length < stage.gates.length) || null;
  return { done: doneStages, total: stages.length, next };
}

/* AGENTS 第 13 节功效公式，与 tools/power_analysis.py 一致（McNemar 配对，p_d 缺省 0.10）。 */
function invNorm(p) {
  // Acklam 有理逼近，误差 < 1.2e-9，足够计算所需 n。
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425, hi = 1 - lo;
  if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > hi) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
function normCdf(x) {
  // Abramowitz–Stegun 7.1.26 的误差函数逼近。
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2);
  return x >= 0 ? (1 + y) / 2 : (1 - y) / 2;
}
export function nRequired(delta, pD = 0.1, alpha = 0.05, power = 0.8) {
  if (!(delta > 0 && delta < 1)) return null;
  const z = invNorm(1 - alpha / 2) + invNorm(power);
  return Math.ceil(z * z * pD / (delta * delta));
}
export function achievedPower(n, delta, pD = 0.1, alpha = 0.05) {
  if (!(n > 0) || !(delta > 0 && delta < 1)) return null;
  return normCdf(Math.abs(delta) * Math.sqrt(n / pD) - invNorm(1 - alpha / 2));
}
