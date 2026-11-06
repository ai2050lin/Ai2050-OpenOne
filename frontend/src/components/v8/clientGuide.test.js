import { describe, expect, it } from 'vitest';

import { languageSections, relationCounts } from './ClientGuide.jsx';

describe('三部分研究摘要', () => {
  it('按语言目录三类入口统计已测条目，总数不受测量集影响', () => {
    const sections = languageSections(new Set(['KN.ATTR', 'GR.Q', 'RS.NEG']));
    expect(sections.map(section => section.total).reduce((sum, value) => sum + value, 0)).toBe(69);
    expect(sections.map(section => section.done)).toEqual([1, 1, 1]);
  });

  it('只统计语言节点到特征卡的连接，并按证据类型分开', () => {
    const objects = [
      { type: 'feature', id: 'f1', model_snapshot: 'a' },
      { type: 'feature', id: 'f2', model_snapshot: 'b' },
      { type: 'edge', model_snapshot: 'a', body: { source: 'language:x', target: 'f1', evidence: { level: 'heldout' } } },
      { type: 'edge', model_snapshot: 'a', body: { source: 'f1', target: 'language:y', evidence: { level: 'negative' } } },
      { type: 'edge', model_snapshot: 'a', body: { source: 'f1', target: 'f1', evidence: { level: 'intervention' } } },
      { type: 'edge', model_snapshot: 'b', body: { source: 'language:x', target: 'f2', evidence: { level: 'heldout' } } },
    ];
    expect(relationCounts(objects, 'a')).toEqual({ heldout: 1, negative: 1 });
  });
});
