/* 语言图谱种子（2026-11-07 用户审定，Phase 320）。
   位置是展示用的人工布局：横向三个区域对应语言目录三大类，纵向是抽象程度。距离不代表模型几何。
   节点 st：real = 项目已有材料（模板或词表里有），cat = 语言目录已有、未测，new = 目录和材料都没有，需新增。
   节点 cat 是 shared/dist_protocol/lang_catalog.json 的条目 ID；also 是交叉归属的另一类（主类只存一次）。
   边的 basis：material = 有模板或已测词表材料，label = 只有词表标签或目录归属，none = 人工定义、无材料。
   审定内容：边加依据；删除没有依据的「否定→但是」；词形「苹果」改挂 SEM.SENSE；加英文词形 apple；
   但是 / 然后 / 因为以推理功能为主（与语言目录一致），也属于语法规则。 */
export const ZONES = {
  KN: { name: '知识网络', color: '#2a78d6', x: -1.3 },
  RS: { name: '推理功能', color: '#1baf7a', x: 0 },
  GR: { name: '语法规则', color: '#c98a00', x: 1.3 },
};

export const NODES = [
  ['obj', '物体', 'KN', '概念', -1.3, 0.92, 0.1, 'new', '', '现有材料没有这一层。'],
  ['food', '食物', 'KN', '概念', -1.55, 0.5, -0.2, 'real', 'SEM.CATEGORY', 'ENT-XF 词表里“食物”和“水果”是同级类别，不是上下位。要检验层级，需要新材料。'],
  ['plant', '植物', 'KN', '概念', -0.92, 0.5, 0.34, 'real', 'SEM.CATEGORY', 'ENT-XF 词表把水果记在植物大类下。'],
  ['fruit', '水果', 'KN', '概念', -1.3, 0.12, 0.02, 'real', 'SEM.ISA', 'KN-ISA 的类别之一（水果、动物、金属、工具、交通工具、乐器，彼此同级）。'],
  ['word', '词形「苹果」', 'KN', '词形', -0.72, -0.72, 0.42, 'cat', 'SEM.SENSE', 'token、词形和词义分开记录。一个词形可以连到多个词义。'],
  ['apple', '苹果（水果义）', 'KN', '词义', -1.42, -0.34, 0.3, 'real', 'SEM.ISA', 'KN-ISA 与 ENT-XF 已有此词，EA 系列测过实体编码与属性回答。'],
  ['company', '苹果（公司义）', 'KN', '词义', -0.62, -0.3, 0.62, 'cat', 'SEM.SENSE', '目录已有多义条目，还没有模式族。'],
  ['apple_en', '词形「apple」', 'KN', '词形', -0.4, -0.66, -0.12, 'real', 'VAR.ALIGNMENT', 'ENT-XF 有中英两种语言，EA-v2 测过中英编码的相互迁移。'],
  ['banana', '香蕉', 'KN', '词义', -1.72, -0.3, -0.3, 'real', 'SEM.ISA', 'KN-ISA 与 ENT-XF 已有此词。'],
  ['pear', '梨', 'KN', '词义', -1.05, -0.36, -0.42, 'real', 'SEM.ISA', 'EA-v2 确认集的新实体，测过按类别中心归类。'],
  ['red', '红色', 'KN', '属性', -1.86, -0.8, 0.3, 'real', 'SEM.ATTRIBUTE', 'ENT-XF 词表记录了颜色，并有颜色问答；KN-ATTR 测过属性绑定。'],
  ['trans', '翻译', 'RS', '操作', 0, 0.62, 0.05, 'cat', 'TASK.TRANSLATION', '目录已有。现有模板没有翻译材料（ENT-XF 的任务因子是颜色和大小提问）。'],
  ['tfr', '翻译成法语', 'RS', '操作', -0.3, 0.16, 0.32, 'new', 'TASK.TRANSLATION', '现有材料没有法语。'],
  ['ten', '翻译成英语', 'RS', '操作', 0.26, 0.16, -0.24, 'new', 'TASK.TRANSLATION', '同一操作换目标语言。要检验模型内部是否共用这一操作。'],
  ['neg', '否定', 'RS', '功能', -0.34, -0.46, -0.42, 'real', 'LOG.NEG', 'GR-NEG、ENT-XF 已有。'],
  ['cond', '条件', 'RS', '功能', 0.06, -0.86, 0.1, 'cat', 'LOG.COND', '目录已有，未研究。'],
  ['cause', '因果', 'RS', '功能', 0.36, -0.46, 0.36, 'cat', 'LOG.CAUSE', '目录已有，未研究。'],
  ['rel', '篇章关系', 'RS', '功能', 0.5, 0.56, 0.44, 'cat', 'DIS.RELATION', '目录把转折、解释放在篇章，属于推理与功能。'],
  ['but', '但是', 'RS', '连词', 0.5, 0.06, 0.56, 'cat', 'DIS.RELATION', '主类推理功能（转折），也属于语法规则（连词）。', 'GR'],
  ['then', '然后', 'RS', '连词', 0.72, 0, 0.18, 'cat', 'DIS.RELATION', '主类推理功能（时序），也属于语法规则（连词）。', 'GR'],
  ['because', '因为', 'RS', '连词', 0.62, -0.5, 0.44, 'cat', 'DIS.RELATION', '主类推理功能（因果），也属于语法规则（连词）。', 'GR'],
  ['conj', '连词', 'GR', '词类', 1.16, 0.5, 0.32, 'new', '', '目录按功能分类，这个词类节点要新增。'],
  ['q', '疑问', 'GR', '规则', 1.42, 0.5, -0.3, 'real', 'SYN.MOOD', 'GR-Q 已有。'],
  ['s', '陈述', 'GR', '规则', 1.72, 0.12, -0.3, 'real', 'SYN.MOOD', 'GR-Q 已有。'],
  ['active', '主动句', 'GR', '规则', 1.22, -0.2, -0.44, 'real', 'SYN.CONSTRUCTION', 'GR-ROLE 已有。'],
  ['bei', '被字句', 'GR', '规则', 1.62, -0.46, -0.26, 'real', 'SYN.CONSTRUCTION', 'GR-ROLE 已有。'],
  ['agent', '施事', 'GR', '规则', 1.3, -0.84, 0.2, 'real', 'SYN.ROLE', 'GR-ROLE 已有。'],
  ['patient', '受事', 'GR', '规则', 1.68, -0.82, 0, 'real', 'SYN.ROLE', 'GR-ROLE 已有。'],
].map(([id, name, zone, kind, x, y, z, st, cat, note, also]) => ({ id, name, zone, kind, x, y, z, st, cat, note, also: also || '' }));

export const EDGE_TYPES = {
  isa: ['上下位', '', '#566373'], sense: ['词形到词义', '2 3', '#566373'], xlang: ['跨语言对应', '6 2 1 2', '#2a78d6'],
  attr: ['属性', '7 3', '#8a5010'], op: ['同一操作，换参数', '', '#087b70'], slot: ['同槽位替换', '5 4', '#a33e43'],
  mem: ['属于', '1 4', '#566373'], cross: ['跨类表达', '9 3 2 3', '#7050a0'], alt: ['句式变换', '5 4', '#245ca8'],
};

export const BASIS = {
  material: ['有材料', 1, 2],
  label: ['只有标签或目录', 0.6, 1.4],
  none: ['人工定义，无材料', 0.32, 1.1],
};

export const EDGES = [
  ['apple', 'fruit', 'isa', 'material'], ['banana', 'fruit', 'isa', 'material'], ['pear', 'fruit', 'isa', 'material'],
  ['fruit', 'food', 'isa', 'none'], ['fruit', 'plant', 'isa', 'label'], ['food', 'obj', 'isa', 'none'], ['plant', 'obj', 'isa', 'none'],
  ['word', 'apple', 'sense', 'none'], ['word', 'company', 'sense', 'none'], ['apple_en', 'apple', 'xlang', 'material'],
  ['apple', 'red', 'attr', 'material'],
  ['tfr', 'trans', 'op', 'none'], ['ten', 'trans', 'op', 'none'], ['tfr', 'ten', 'slot', 'none'], ['ten', 'apple_en', 'cross', 'none'],
  ['but', 'then', 'slot', 'none'], ['but', 'rel', 'mem', 'label'], ['then', 'rel', 'mem', 'label'], ['because', 'rel', 'mem', 'label'],
  ['but', 'conj', 'mem', 'none'], ['then', 'conj', 'mem', 'none'], ['because', 'conj', 'mem', 'none'], ['because', 'cause', 'cross', 'none'],
  ['cond', 'cause', 'slot', 'none'], ['q', 's', 'alt', 'material'], ['active', 'bei', 'alt', 'material'], ['agent', 'patient', 'slot', 'material'],
];

export const STATE_LABEL = { real: '已有项目材料', cat: '目录已有，未测', new: '目录或材料没有' };

export function neighbors(id) {
  const out = new Set([id]);
  for (const [a, b] of EDGES) { if (a === id) out.add(b); if (b === id) out.add(a); }
  return out;
}
