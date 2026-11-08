import { describe, expect, it } from 'vitest';

import { HELP_PAGES, helpKey } from './helpPages.js';
import { ROUTE_VIEWS, SPATIAL_MODES, STATIONS } from './workbenchUtil.js';

const CFG = ['conn', 'account', 'node', 'model'];
const INNER = { enc: ['structure', 'internal'], assoc: ['live', 'response', 'langmeasure', 'relations'], lang: ['ring', 'seed', 'catalog'] };

describe('help pages', () => {
  it('every tab has its own help entry', () => {
    const keys = [...ROUTE_VIEWS.map(v => `route/${v.id}`), ...STATIONS.map(s => `process/${s.id}`), ...CFG.map(c => `config/${c}`),
      ...SPATIAL_MODES.flatMap(m => INNER[m.id].map(t => `spatial/${m.id}/${t}`))];
    for (const k of keys) {
      const h = HELP_PAGES[k];
      expect(h, k).toBeTruthy();
      expect(h.title && h.ask, k).toBeTruthy();
      expect(h.principle.length + h.modules.length, k).toBeGreaterThan(1);
    }
  });
  it('resolves inner tab, falls back to base', () => {
    expect(helpKey('spatial/assoc', 'response')).toBe('spatial/assoc/response');
    expect(helpKey('process/plan', 'ring')).toBe('process/plan');
    expect(helpKey('spatial/lang', null)).toBe('spatial/lang');
  });
});
