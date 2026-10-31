/* 数据集进度（/api/progress/datasets，dataset_progress.v1）的非组件函数：状态外观、分组、阶段计数、逐层条形几何。
   状态同时给符号与文字；「完成」只表示产物存在且判据有结论，不是证据等级。 */

export const TRACKS = [
  { id: 'neuron', label: '神经元逆向', sub: 'Goodfire 式：采集 → 响应画像 → 四轴检验 → 因果使用 → 入图谱' },
  { id: 'attention', label: '注意力语言图谱', sub: '行为门 → 可读出 → 注意力路由 → 优先级与层级 → 保真度' },
];

/* 状态色：完成深绿、部分浅绿、未支持浅红（与完成同等醒目）、未开始浅灰、不适用留白 */
export const STATUS_META = {
  done: { sym: '✓', label: '完成', bg: '#1f7a4d', ink: '#ffffff' },
  partial: { sym: '◐', label: '部分', bg: '#b9e4cc', ink: '#0f5132' },
  negative: { sym: '✗', label: '未支持', bg: '#f6cfc6', ink: '#8a2a12' },
  todo: { sym: '○', label: '未开始', bg: '#eef1f5', ink: '#64748b' },
  na: { sym: '—', label: '不适用', bg: 'transparent', ink: '#b6bfca' },
  unavailable: { sym: '?', label: '数据源不可读', bg: '#fff3cd', ink: '#7a5b00' },
};
export const STATUS_ORDER = ['done', 'partial', 'negative', 'todo', 'na', 'unavailable'];

export function statusMeta(status) {
  return STATUS_META[status] || STATUS_META.unavailable;
}

export const KIND_LABEL = { conflict: '冲突材料', template: '最小对模板', corpus: '自然语料', exam: '考试题库' };
const KIND_ORDER = ['conflict', 'template', 'corpus', 'exam'];

/* 按材料类型分组（固定顺序），组内保持服务器给出的顺序 */
export function groupDatasets(datasets) {
  const groups = [];
  for (const kind of [...KIND_ORDER, ...new Set((datasets || []).map(d => d.kind).filter(k => !KIND_ORDER.includes(k)))]) {
    const rows = (datasets || []).filter(d => d.kind === kind);
    if (rows.length) groups.push({ kind, label: KIND_LABEL[kind] || kind, rows });
  }
  return groups;
}

/* 某条线某阶段在全部数据集上的状态计数（不适用不计入分母） */
export function stageCounts(datasets, track, stageId) {
  const c = Object.fromEntries(STATUS_ORDER.map(s => [s, 0]));
  for (const d of datasets || []) {
    const s = d?.[track]?.[stageId]?.status || 'unavailable';
    c[s] = (c[s] || 0) + 1;
  }
  c.applicable = (datasets || []).length - c.na;
  return c;
}

/* 一个数据集在一条线上走到哪一步：最后一个「完成 / 部分 / 未支持」的阶段序号（1 起），没有则 0 */
export function reached(dataset, track, stages) {
  let last = 0;
  (stages || []).forEach((s, i) => {
    const st = dataset?.[track]?.[s.id]?.status;
    if (st === 'done' || st === 'partial' || st === 'negative') last = i + 1;
  });
  return last;
}

/* 逐层条形：顺序量按最大值归一（0–1），正负量按最大绝对值归一（−1–1）；缺值保持 null */
export function layerBars(values, kind) {
  const vs = (values || []).map(v => (typeof v === 'number' && Number.isFinite(v) ? v : null));
  const nums = vs.filter(v => v !== null);
  if (!nums.length) return { bars: vs.map(() => null), max: 0 };
  const max = kind === 'diverging' ? Math.max(...nums.map(Math.abs)) : Math.max(...nums);
  return { bars: vs.map(v => (v === null || !max ? null : v / max)), max };
}

/* 贡献状态的一句话（只描述服务器上的分片与任务，不表示证据晋级） */
export function contributeLine(c) {
  if (!c) return { open: false, text: '没有贡献信息' };
  if (c.kind === 'census') {
    const n = c.counts || {};
    const parts = [['queued', '待领取'], ['running', '执行中'], ['replica_pending', '等待复算'], ['done', '完成'], ['disputed', '复算不一致']]
      .filter(([k]) => n[k]).map(([k, l]) => `${l} ${n[k]}`);
    return { open: !!c.open, text: `普查分片：${parts.join('、') || '没有分片'}` };
  }
  if (c.kind === 'template') {
    const active = c.tasks_active ? `，进行中任务 ${c.tasks_active}` : '';
    return { open: !!c.open, text: `模板分片：${c.status_label || c.status || '未知'}${active}` };
  }
  return { open: false, text: c.note || '还没有拆成分布式任务' };
}
