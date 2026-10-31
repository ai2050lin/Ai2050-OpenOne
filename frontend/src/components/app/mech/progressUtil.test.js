import { describe, expect, it } from 'vitest';

import { contributeLine, groupDatasets, layerBars, reached, stageCounts, statusMeta } from './progressUtil.js';

const stages = [{ id: 'N1' }, { id: 'N2' }, { id: 'N3' }];
const ds = [
  { id: 'a', kind: 'template', neuron: { N1: { status: 'done' }, N2: { status: 'negative' }, N3: { status: 'todo' } } },
  { id: 'b', kind: 'conflict', neuron: { N1: { status: 'done' }, N2: { status: 'done' }, N3: { status: 'partial' } } },
  { id: 'c', kind: 'exam', neuron: { N1: { status: 'na' }, N2: { status: 'na' }, N3: { status: 'na' } } },
];

describe('数据集进度', () => {
  it('按固定材料顺序分组', () => {
    expect(groupDatasets(ds).map(g => g.kind)).toEqual(['conflict', 'template', 'exam']);
  });
  it('阶段计数：不适用不计入分母', () => {
    const c = stageCounts(ds, 'neuron', 'N2');
    expect(c.done).toBe(1);
    expect(c.negative).toBe(1);
    expect(c.na).toBe(1);
    expect(c.applicable).toBe(2);
  });
  it('走到哪一步：未支持也算走到（检验做过）', () => {
    expect(reached(ds[0], 'neuron', stages)).toBe(2);
    expect(reached(ds[1], 'neuron', stages)).toBe(3);
    expect(reached(ds[2], 'neuron', stages)).toBe(0);
  });
  it('逐层条：顺序量按最大值、正负量按最大绝对值归一，缺值保持 null', () => {
    expect(layerBars([1, 2, null, 4], 'sequential').bars).toEqual([0.25, 0.5, null, 1]);
    expect(layerBars([-2, 1], 'diverging').bars).toEqual([-1, 0.5]);
    expect(layerBars([], 'sequential').max).toBe(0);
  });
  it('贡献状态只描述分片', () => {
    expect(contributeLine({ kind: 'census', open: true, counts: { queued: 18 } })).toEqual({ open: true, text: '普查分片：待领取 18' });
    expect(contributeLine({ kind: 'none', note: '本机脚本' }).open).toBe(false);
    expect(statusMeta('weird').label).toBe('数据源不可读');
  });
});
