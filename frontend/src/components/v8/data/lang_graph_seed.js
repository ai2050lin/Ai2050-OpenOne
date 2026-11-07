/* 语言图谱种子（草案，2026-11-07，待研究审定）。
   位置是展示用的人工布局：横向三个区域对应语言目录三大类，纵向是抽象程度。距离不代表模型几何。
   st：real = 项目已有材料（模板或词表里有），cat = 语言目录已有、未测，new = 目录和材料都没有，需新增。
   cat 字段是 shared/dist_protocol/lang_catalog.json 的条目 ID。 */
export const ZONES = {
  KN: { name: '知识网络', color: '#2a78d6', x: -1.3 },
  RS: { name: '推理功能', color: '#1baf7a', x: 0 },
  GR: { name: '语法规则', color: '#c98a00', x: 1.3 },
};

export const NODES = [
  ['obj', '物体', 'KN', '概念', -1.3, 0.92, 0.1, 'new', '', '现有材料没有这一层。'],
  ['food', '食物', 'KN', '概念', -1.55, 0.5, -0.2, 'real', 'SEM.CATEGORY', 'ENT-XF 词表里“食物”和“水果”是同级类别，不是上下位。要检验层级，需要新材料。'],
  ['plant', '植物', 'KN', '概念', -0.92, 0.5, 0.34, 'real', 'SEM.CATEGORY', 'ENT-XF 词表把水果记在植物大类下。'],
  ['fruit', '水果', 'KN', '概念', -1.3, 0.12, 0.02, 'real', 'SEM.ISA', 'ENT-XF 的类别之一。'],
  ['word', '词形「苹果」', 'KN', '词形', -0.72, -0.72, 0.42, 'cat', 'FORM.BOUNDARY', 'token、词形和词义分开记录。一个词形可以连到多个词义。'],
  ['apple', '苹果（水果义）', 'KN', '词义', -1.42, -0.34, 0.3, 'real', 'SEM.ISA', 'ENT-XF 词表已有此词，EA 系列测过实体编码与属性回答。'],
  ['company', '苹果（公司义）', 'KN', '词义', -0.62, -0.3, 0.62, 'cat', 'SEM.SENSE', '目录已有多义条目，还没有模式族。'],
  ['banana', '香蕉', 'KN', '词义', -1.72, -0.3, -0.3, 'real', 'SEM.ISA', 'ENT-XF 词表已有此词。'],
  ['pear', '梨', 'KN', '词义', -1.05, -0.36, -0.42, 'real', 'SEM.ISA', 'EA-v2 确认集的新实体。'],
  ['red', '红色', 'KN', '属性', -1.86, -0.8, 0.3, 'real', 'SEM.ATTRIBUTE', '词表记录了颜色属性，KN-ATTR 测过属性绑定。'],
  ['trans', '翻译', 'RS', '操作', 0, 0.62, 0.05, 'cat', 'TASK.TRANSLATION', '目录已有，现有模板没有翻译材料。'],
  ['tfr', '翻译成法语', 'RS', '操作', -0.3, 0.16, 0.32, 'new', 'TASK.TRANSLATION', '现有材料没有法语。'],
  ['ten', '翻译成英语', 'RS', '操作', 0.26, 0.16, -0.24, 'new', 'TASK.TRANSLATION', '同一操作换目标语言。要检验模型内部是否共用这一操作。'],
  ['neg', '否定', 'RS', '功能', -0.34, -0.46, -0.42, 'real', 'LOG.NEG', 'GR-NEG、ENT-XF 已有。'],
  ['cond', '条件', 'RS', '功能', 0.06, -0.86, 0.1, 'cat', 'LOG.COND', '目录已有，未研究。'],
  ['cause', '因果', 'RS', '功能', 0.36, -0.46, 0.36, 'cat', 'LOG.CAUSE', '目录已有，未研究。'],
  ['rel', '篇章关系', 'RS', '功能', 0.62, 0.56, 0.44, 'cat', 'DIS.RELATION', '目录把转折、解释放在篇章，属于推理与功能。'],
  ['but', '但是', 'GR', '连词', 0.76, 0.04, 0.56, 'cat', 'DIS.RELATION', '跨两类：词类属语法，转折属篇章功能。'],
  ['then', '然后', 'GR', '连词', 1.02, 0, 0.18, 'cat', 'DIS.RELATION', '跨两类：词类属语法，时序属篇章功能。'],
  ['because', '因为', 'GR', '连词', 0.84, -0.5, 0.44, 'cat', 'DIS.RELATION', '同时连到因果功能。'],
  ['conj', '连词', 'GR', '词类', 1.16, 0.5, 0.32, 'new', '', '目录按功能分类，这个词类节点要新增。'],
  ['q', '疑问', 'GR', '规则', 1.42, 0.5, -0.3, 'real', 'SYN.MOOD', 'GR-Q 已有。'],
  ['s', '陈述', 'GR', '规则', 1.72, 0.12, -0.3, 'real', 'SYN.MOOD', 'GR-Q 已有。'],
  ['active', '主动句', 'GR', '规则', 1.22, -0.2, -0.44, 'real', 'SYN.CONSTRUCTION', 'GR-ROLE 已有。'],
  ['bei', '被字句', 'GR', '规则', 1.62, -0.46, -0.26, 'real', 'SYN.CONSTRUCTION', 'GR-ROLE 已有。'],
  ['agent', '施事', 'GR', '规则', 1.3, -0.84, 0.2, 'real', 'SYN.ROLE', 'GR-ROLE 已有。'],
  ['patient', '受事', 'GR', '规则', 1.68, -0.82, 0, 'real', 'SYN.ROLE', 'GR-ROLE 已有。'],
].map(([id, name, zone, kind, x, y, z, st, cat, note]) => ({ id, name, zone, kind, x, y, z, st, cat, note }));

export const EDGE_TYPES = {
  isa: ['上下位', '', '#566373'], sense: ['词形到词义', '2 3', '#566373'], attr: ['属性', '7 3', '#8a5010'],
  op: ['同一操作，换参数', '', '#087b70'], slot: ['同槽位替换', '5 4', '#a33e43'], mem: ['属于', '1 4', '#566373'],
  cross: ['跨类表达', '9 3 2 3', '#7050a0'], alt: ['句式变换', '5 4', '#245ca8'],
};

export const EDGES = [
  ['apple', 'fruit', 'isa'], ['banana', 'fruit', 'isa'], ['pear', 'fruit', 'isa'], ['fruit', 'food', 'isa'], ['fruit', 'plant', 'isa'],
  ['food', 'obj', 'isa'], ['plant', 'obj', 'isa'], ['word', 'apple', 'sense'], ['word', 'company', 'sense'], ['apple', 'red', 'attr'],
  ['tfr', 'trans', 'op'], ['ten', 'trans', 'op'], ['tfr', 'ten', 'slot'], ['but', 'then', 'slot'], ['but', 'rel', 'mem'], ['then', 'rel', 'mem'],
  ['but', 'conj', 'mem'], ['then', 'conj', 'mem'], ['because', 'conj', 'mem'], ['because', 'cause', 'cross'], ['cond', 'cause', 'slot'],
  ['q', 's', 'alt'], ['active', 'bei', 'alt'], ['agent', 'patient', 'slot'], ['neg', 'but', 'cross'],
];

export const STATE_LABEL = { real: '已有项目材料', cat: '目录已有，未测', new: '目录或材料没有' };

export function neighbors(id) {
  const out = new Set([id]);
  for (const [a, b] of EDGES) { if (a === id) out.add(b); if (b === id) out.add(a); }
  return out;
}
