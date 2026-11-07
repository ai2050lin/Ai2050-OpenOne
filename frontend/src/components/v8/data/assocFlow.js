/* 关联机制 3D 流的数据（MA-H2，Phase 332）：逐层读出、各头注意力、各因素差异。只读研究产物，不含演示数字。 */
import report from '../../../../../research/mainline/H2/report.json';
import heads from '../../../../../research/mainline/H2/heads_tcol.json';

/* 因素彩线取“问题中的目标名”位置：该位置 hs[0] 各条相同，没有词身份混杂（答案位置的范围因素被答案前缀词混杂） */
const FACTORS = [['scope', '范围（资料 / 现实）', '#7050a0'], ['pos', '目标记录位置（首 / 末）', '#087b70'], ['k', '记录条数（2 / 3）', '#b07d00'],
  ['draft', '错误草稿', '#c0392b'], ['form', '记录形式（单 / 双实体）', '#4a7fd0']];

export const FLOW_DATA = {
  n: heads.n,
  lensDoc: report.lens.mean_K_minus_P.doc, lensReal: report.lens.mean_K_minus_P.real,
  heads: { doc: heads.doc, real: heads.real },
  factors: FACTORS.map(([key, label, color]) => [key, label, color, report.effects[key].qname.std_rms]),
};

/* 资料范围减现实范围，差值最大的 12 个头 */
export function rankHeads(data = FLOW_DATA, n = 12) {
  const rows = [];
  data.heads.doc.forEach((row, l) => row.forEach((d, h) => rows.push({ l, h, d, r: data.heads.real[l][h] })));
  return rows.sort((a, b) => (b.d - b.r) - (a.d - a.r)).slice(0, n);
}

