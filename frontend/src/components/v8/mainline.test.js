import { describe, expect, it } from 'vitest';

import catalog from '../../../../shared/dist_protocol/lang_catalog.json';
import { normalizeState, spaceMode, spaceTab } from '../app/workbenchUtil.js';
import { BASIS, EDGES, NODES } from './data/lang_graph_seed.js';
import { STAGES, achievedPower, nRequired, pct, stageProgress } from './data/mainline.js';

describe('空间三张图的路由', () => {
  const mode = input => { const st = normalizeState(input); return [spaceMode(st.view, st.gview), spaceTab(st.view, st.gview)]; };
  it('新入口落到对应的图和视图', () => {
    expect(mode({ lens: 'spatial', view: 'graph', gview: 'language' })).toEqual(['lang', 'language']);
    expect(mode({ lens: 'spatial', view: 'graph', gview: 'structure' })).toEqual(['enc', 'structure']);
    expect(mode({ lens: 'spatial', view: 'live' })).toEqual(['assoc', 'live']);
    expect(mode({ lens: 'spatial', view: 'response' })).toEqual(['assoc', 'response']);
  });
  it('旧链接仍能归到三张图之一', () => {
    // 旧别名 view=lang 原本是语言项实测页，现归关联机制；语言图谱只剩外部框架
    expect(mode({ lens: 'spatial', view: 'lang' })).toEqual(['assoc', 'langmeasure']);
    expect(mode({ lens: 'spatial', view: 'graph', gview: 'langmeasure' })).toEqual(['assoc', 'langmeasure']);
    expect(mode({ lens: 'spatial', view: 'atlas' })).toEqual(['assoc', 'relations']);
    expect(mode({ lens: 'spatial', view: 'neuron' })).toEqual(['enc', 'structure']);
    expect(mode({ lens: 'spatial', view: 'research' })).toEqual(['enc', 'internal']);
  });
});

describe('主线进度', () => {
  it('没有登记完成的阶段时总进度为 0，下一步是 H0', () => {
    const progress = stageProgress();
    expect(progress.total).toBe(8);
    expect(progress.done).toBe(STAGES.filter(s => s.done.length === s.gates.length).length);
    expect(stageProgress(STAGES.map(s => ({ ...s, done: [] }))).next.id).toBe('H0');
  });
  it('分母不明时不给百分比，小比例保留两位', () => {
    expect(pct(3, null)).toBeNull();
    expect(pct(0, 8)).toBe('0');
    expect(pct(265, 350208)).toBe('0.08');
    expect(pct(9, 69)).toBe('13');
  });
});

describe('功效三栏与 tools/power_analysis.py 一致', () => {
  it('所需 n', () => {
    expect(nRequired(0.03)).toBe(873);
    expect(nRequired(0.05)).toBe(314);
    expect(nRequired(0.02, 0.2)).toBe(3925);
    expect(nRequired(0)).toBeNull();
  });
  it('检验力', () => {
    expect(achievedPower(1000, 0.03)).toBeCloseTo(0.8508, 3);
    expect(achievedPower(300, 0.05)).toBeCloseTo(0.7819, 3);
  });
});

describe('语言图谱种子', () => {
  const ids = new Set(NODES.map(n => n.id));
  const catalogIds = new Set(catalog.domains.flatMap(d => d.groups.flatMap(g => g.items.map(i => i.id))));
  it('每条边的两端都是已有节点', () => {
    for (const [a, b] of EDGES) { expect(ids.has(a)).toBe(true); expect(ids.has(b)).toBe(true); }
  });
  it('每条边都有依据，审定删除的边不再出现', () => {
    for (const edge of EDGES) expect(Object.keys(BASIS)).toContain(edge[3]);
    expect(EDGES.some(([a, b]) => (a === 'neg' && b === 'but') || (a === 'but' && b === 'neg'))).toBe(false);
  });
  it('连词以推理功能为主，也属于语法规则', () => {
    for (const id of ['but', 'then', 'because']) {
      const node = NODES.find(n => n.id === id);
      expect([node.zone, node.also]).toEqual(['RS', 'GR']);
    }
  });
  it('节点引用的目录条目都存在', () => {
    for (const node of NODES) if (node.cat) expect(catalogIds.has(node.cat)).toBe(true);
  });
});
