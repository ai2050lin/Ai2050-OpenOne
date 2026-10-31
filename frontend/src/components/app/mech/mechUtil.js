/* 机制案例与方法校准报告（/api/mechanism/index、/api/mechanism/case/{id}，server/mechanism_service.py）的读取函数（非组件）。
   报告由研究脚本写出（conflict_case.v1：scripts/conflict_case.py；calib_ioi.v1 / calib_qk.v1：GPT-2 校准），
   这里只取字段、换算显示，不重算任何统计量；字段缺失时返回 null，由页面显示「未接入」。 */

/* 冲突线索的中文名（CF-v1 的 cues；未知线索原样显示） */
export const CUE_LABEL = {
  plain: '直接陈述', hearsay: '「有人说」', assert: '「事实上」', fiction: '「虚构故事里」',
  repeat2: '重复两次', negated: '否定陈述（对照）',
};
export const cueLabel = c => CUE_LABEL[c] || c;

/* 报告判据的中文名（mechanism_service._checks 归一后的键 + 校准报告自带的键） */
export const CHECK_LABEL = {
  head_necessity: '所选头必要（去掉后上下文优势下降超过阈值且大于随机头）',
  head_sufficiency_last_position: '所选头充分（只保留所选头时保留足够比例）',
  negated_control_not_copying: '否定对照：上下文没有只靠复制答案词取胜',
  ioi_query_s_inhibition: 'IOI：S 抑制头经查询侧决定名称搬运头的注意力',
  induction_key_prev_token: '归纳头：键侧依赖前一 token 头 4.11',
  clean_logit_diff: 'IOI：干净输入的对数几率差为正',
  top3_name_movers: 'IOI：名称搬运头排进前三',
  negative_mover_sign: 'IOI：负名称搬运头符号正确',
  atp_spearman: 'IOI：归因修补与真实修补秩相关',
};
/* conflict_case.v2 的三组头（只用发现划分选出，留出划分检验） */
export const GROUP_LABEL = {
  C_context_writers: 'C 组 · 上下文写入头（直接作用最正）',
  M_memory_writers: 'M 组 · 记忆写入头（直接作用最负）',
  U_upstream: 'U 组 · 上游头（总效应最负）',
};
export const GROUP_EXPECT = { C_context_writers: -1, M_memory_writers: 1, U_upstream: -1 };

export function checkLabel(k) {
  const key = String(k);
  const i = key.indexOf(' · ');                      // 进展页把多个报告的判据合在一起：「案例 · 判据」
  if (i >= 0) return `${key.slice(0, i)}：${checkLabel(key.slice(i + 3))}`;
  if (key === 'v2_hierarchy_U_to_C') return 'v2 层级：消融 U 组后 C 组直接作用下降够多且低于全部随机组（U → C 上下游成立）';
  const m = /^v2_(.+)_(C_context_writers|M_memory_writers|U_upstream)$/.exec(key);
  if (m) return `v2 ${cueLabel(m[1])}：${GROUP_LABEL[m[2]].split('（')[0]}消融后方向符合预期且超过全部随机组`;
  return CHECK_LABEL[key] || key;
}

/* conflict_case.v1 的必要性 / 充分性（含头电路保真度）按留出平均 d 归一化；平均 d < 0 时符号失效
   （研究脚本 v2 段的注释，server/mechanism_service._v1_void 同一规则）。返回失效原因或 null。 */
export function v1Void(report) {
  const d = report?.heads?.heldout_mean_d;
  return Number.isFinite(d) && d < 0 ? `留出平均 d = ${d.toFixed(3)} < 0：v1 按平均 d 归一化，符号失效（见 v2）` : null;
}

export const SCHEMA_LABEL = {
  'conflict_case.v1': '上下文与记忆冲突',
  'calib_ioi.v1': 'GPT-2 IOI 校准',
  'calib_qk.v1': 'GPT-2 QK 路径校准',
};
export const schemaLabel = s => SCHEMA_LABEL[s] || s || '—';

export const isConflict = c => c?.schema === 'conflict_case.v1';

export function fmt(v, digits = 2) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return '—';
  return Number(v).toFixed(digits);
}
export function pct(v, digits = 0) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return '—';
  return `${(Number(v) * 100).toFixed(digits)}%`;
}

/* 'L.H' → [L, H] */
export function parseHead(name) {
  const m = /^(\d+)\.(\d+)$/.exec(String(name || ''));
  return m ? [Number(m[1]), Number(m[2])] : null;
}

/* 二维矩阵的最大绝对值（色标对称归一用）；不是数组时返回 0 */
export function maxAbs(matrix) {
  let m = 0;
  for (const row of matrix || []) for (const v of row || []) if (Number.isFinite(v) && Math.abs(v) > m) m = Math.abs(v);
  return m;
}

/* 矩阵逐元素取负（总效应 → 「去掉后上下文优势的下降量」，与直接作用同号可比）；保持 null */
export function negate(matrix) {
  return (matrix || []).map(row => (row || []).map(v => (Number.isFinite(v) ? -v : null)));
}

/* 报告的判据列表 [{key, label, ok}]：先列未通过的，再列通过的（页面把两类同样醒目地并排显示）。 */
export function checkList(checks) {
  const rows = Object.entries(checks || {}).map(([key, ok]) => ({ key, label: checkLabel(key), ok: Boolean(ok) }));
  return [...rows.filter(r => !r.ok), ...rows.filter(r => r.ok)];
}
/* 失效的判据（既不算通过也不算未通过）：[{key, label, why}] */
export function voidList(voids) {
  return Object.entries(voids || {}).map(([key, why]) => ({ key, label: checkLabel(key), why }));
}

/* 冲突案例的优先级表行：每个线索一行。决出层、上下文胜率与负责的头都来自报告原值。 */
export function priorityRows(report) {
  const table = report?.priority_table || {};
  const base = table.plain?.context_win_rate;
  return Object.entries(table).map(([cue, r]) => {
    const win = r?.context_win_rate;
    const flips = Number.isFinite(base) && Number.isFinite(win) && (base - 0.5) * (win - 0.5) < 0;
    return {
      cue, label: cueLabel(cue), n: r?.n ?? null, win: win ?? null, meanD: r?.mean_d ?? null,
      decisive: r?.decisive_layer_median ?? null, heads: r?.top5_heads_dla || [],
      delta: Number.isFinite(base) && Number.isFinite(win) && cue !== 'plain' ? win - base : null,
      flips,                                         // 胜负（以 0.5 为界）相对直接陈述翻转
      race: Array.isArray(r?.race_mean) ? r.race_mean : null,
    };
  });
}

/* 一个冲突案例是否「完整」：索引摘要的 has 五项（优先级表、逐头直接作用、逐头总效应、神经元、头电路保真度）都在报告里。
   只描述报告字段齐全，不表示检验通过。 */
export const CASE_PARTS = [
  ['priority_table', '优先级表'], ['head_dla', '逐头直接作用'], ['head_total_effect', '逐头总效应'],
  ['neurons', '神经元'], ['head_circuit_fidelity', '头电路保真度'],
];
export function caseComplete(summary) {
  return Boolean(summary?.has) && CASE_PARTS.every(([k]) => summary.has[k]);
}

/* 例句：报告的 examples 按线索取一句；fact 选中时附该事实的行为门记录 */
export function exampleFor(report, cue) {
  const ex = report?.examples || [];
  return ex.find(e => e.cue === cue) || ex[0] || null;
}
export function gateFor(report, subject) {
  return (report?.gate || []).find(g => g.subject === subject) || null;
}

/* 句子里标出两个答案（上下文答案 / 记忆答案）出现的位置：返回 [{text, mark}] 片段 */
export function markAnswers(text, contextAnswer, memoryAnswer) {
  const s = String(text || '');
  const keys = [[contextAnswer, 'ctx'], [memoryAnswer, 'mem']].filter(([k]) => k);
  const out = [];
  let i = 0;
  while (i < s.length) {
    let hit = null;
    for (const [k, mark] of keys) {
      const j = s.indexOf(k, i);
      if (j >= 0 && (!hit || j < hit.j)) hit = { j, k, mark };
    }
    if (!hit) { out.push({ text: s.slice(i), mark: '' }); break; }
    if (hit.j > i) out.push({ text: s.slice(i, hit.j), mark: '' });
    out.push({ text: hit.k, mark: hit.mark });
    i = hit.j + hit.k.length;
  }
  return out;
}
