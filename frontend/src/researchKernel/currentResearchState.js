export const CURRENT_RESEARCH_STATE = {
  phase: 1140,
  title: '多 token 决策路径完成对齐',
  status: 'blocked',
  statusLabel: '统一充分状态未通过',
  summary: 'Qwen3-4B 与 Qwen3-14B 在相对深度 0.6-0.7 出现状态可搬运性跃迁；共享前缀样本经序列路径对齐后恢复，但尚无两个模型共同通过的充分状态。',
  boundary: '当前证据支持条件化过程、晚层状态跃迁和读出路径对齐，不支持固定语义向量、统一充分层、跨模型通用机制或完整智能理论。',
  stopReason: '深度 0.7 的 donor 仍有约三成样本不能翻转答案，两个尺寸也没有共同通过的深度；继续把整层 residual 称为统一语义状态会越过证据边界。',
  nextTask: '把干预位置推进到候选首次分叉的决策边界，完成必要性、充分性、特异性、独立预测和跨材料重复的同路径验证。',
};

export const RESEARCH_EVIDENCE_GATES = [
  {
    id: 'material',
    shortLabel: '反事实对象',
    label: '四状态反事实对象',
    status: 'passed',
    value: '四状态',
    phase: 'Phase 1135',
    summary: '原始/交换档案与更新前/更新后查询组成四状态设计，用于分离实体偏好、时间偏好和真实绑定效应。',
    detail: '反事实对象提供可识别研究目标，但行为正确仍不能直接证明内部机制。',
  },
  {
    id: 'machine-review',
    shortLabel: '数值资格',
    label: '跨模型数值资格',
    status: 'passed',
    value: '已排障',
    phase: 'Phase 1126-1134',
    summary: '完成 GLM4、DS7B 的 FP16 数值形成排障，并建立材料、模型哈希与运行路径的资格门。',
    detail: '数值健康只说明仪器可以工作，不说明观测结果构成语义机制。',
  },
  {
    id: 'human-review',
    shortLabel: '仪器等价',
    label: '同路径仪器等价',
    status: 'passed',
    value: 'α=0 漂移 0',
    phase: 'Phase 1139',
    summary: 'live-state 同路径插值使 α=0 自写回漂移精确为零，定位了早期自补丁漂移的执行路径来源。',
    detail: '该结果校准了干预仪器；它本身不是语义机制证据。',
  },
  {
    id: 'behavior',
    shortLabel: '时效行为',
    label: '时效绑定行为门',
    status: 'passed',
    value: 'Qwen3 同族重复',
    phase: 'Phase 1135-1137',
    summary: 'Qwen3-4B 的四状态时效绑定行为在 Qwen3-14B 前瞻复验中得到同族重复。',
    detail: '这是特定任务和同架构家族的行为证据，不能外推为所有语言能力或所有模型。',
  },
  {
    id: 'hidden',
    shortLabel: '状态跃迁',
    label: '晚层可搬运性跃迁',
    status: 'passed',
    value: '相对深度 0.6-0.7',
    phase: 'Phase 1138',
    summary: 'Qwen3 两个尺寸都在后半程出现整残差状态可搬运性增强。',
    detail: '重复的是相对深度区间和功能现象，不是相同绝对层，也不是已经闭合的统一机制。',
  },
  {
    id: 'causal',
    shortLabel: '因果干预',
    label: '统一充分状态检验',
    status: 'blocked',
    value: '未通过',
    phase: 'Phase 1138-1140',
    summary: '深度 0.7 是强调制状态，但约三成样本不足以翻转答案；两个模型没有共同通过的充分深度。',
    detail: '共享首 token 的候选还要求干预覆盖候选预测路径。下一步应对齐候选首次分叉边界，而不是继续事后挑层。',
  },
  {
    id: 'closure',
    shortLabel: '生成闭合',
    label: '预测与生成闭合',
    status: 'pending',
    value: '0',
    phase: '最终证据门',
    summary: '在全新材料上预注册预测干预结果，并稳定改变真实生成输出。',
    detail: '只有完成留出预测、因果选择性和真实生成闭合，候选链才可以升级为机制结论。',
  },
];

export const RESEARCH_COMPUTATION_CHAIN = [
  { id: 'input', label: '事实 / 日期', gateId: 'material', status: 'passed' },
  { id: 'context', label: '语境状态', gateId: 'behavior', status: 'passed' },
  { id: 'competition', label: '候选竞争', gateId: 'behavior', status: 'passed' },
  { id: 'hidden-event', label: '晚层跃迁', gateId: 'hidden', status: 'passed' },
  { id: 'component', label: '分叉边界', gateId: 'causal', status: 'blocked' },
  { id: 'margin', label: '充分性检验', gateId: 'causal', status: 'blocked' },
  { id: 'output', label: '输出闭合', gateId: 'closure', status: 'pending' },
];

export function getResearchEvidenceGate(gateId) {
  return RESEARCH_EVIDENCE_GATES.find((gate) => gate.id === gateId) || null;
}
