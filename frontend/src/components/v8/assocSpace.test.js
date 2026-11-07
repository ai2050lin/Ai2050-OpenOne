import { describe, expect, it } from 'vitest';

import { FLOW_DATA, rankHeads } from './data/assocFlow.js';

describe('关联机制 3D 流数据（MA-H2）', () => {
  it('词嵌入 + 36 层，36 块 × 32 头', () => {
    expect(FLOW_DATA.lensDoc).toHaveLength(37);
    expect(FLOW_DATA.lensReal).toHaveLength(37);
    expect(FLOW_DATA.heads.doc).toHaveLength(36);
    expect(FLOW_DATA.heads.doc.every(row => row.length === 32)).toBe(true);
    expect(FLOW_DATA.factors.every(f => f[3].length === 37)).toBe(true);
  });
  it('因素取问题中的目标名位置：hs[0] 上没有差异（无词身份混杂）', () => {
    FLOW_DATA.factors.forEach(f => expect(f[3][0]).toBeLessThan(0.01));
  });
  it('头排名按资料减现实，与 H2 报告的前几名一致', () => {
    const top = rankHeads().map(x => `L${x.l}H${x.h}`);
    expect(top.slice(0, 5)).toEqual(expect.arrayContaining(['L29H11', 'L23H4', 'L28H0']));
    const d = rankHeads().map(x => x.d - x.r);
    expect([...d].sort((a, b) => b - a)).toEqual(d);
  });
});
