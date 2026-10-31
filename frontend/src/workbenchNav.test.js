import { describe, expect, it } from 'vitest';

import { parseState, resolveGo, stateQuery, visibleLenses, visibleSubs } from './components/app/workbenchUtil.js';
import { caseComplete, checkLabel, checkList, markAnswers, negate, parseHead, priorityRows, v1Void, voidList } from './components/app/mech/mechUtil.js';

const at = search => {
  const s = parseState(search);
  return [s.lens, s.sub, s.lens === 'admin' ? s.step : undefined].filter(v => v !== undefined).join('/');
};

describe('旧链接换算到 v9 页面', () => {
  it.each([
    ['', 'sentence/case'],
    ['?lens=route', 'progress/'],
    ['?lens=route&rtab=lang', 'progress/'],
    ['?lens=home', 'progress/'],
    ['?lens=spatial', 'neurons/atlas'],
    ['?lens=spatial&view=neuron', 'neurons/atlas'],
    ['?lens=spatial&view=units', 'neurons/atlas'],
    ['?lens=spatial&view=lang', 'datasets/lang'],
    ['?lens=spatial&view=response&tm=TG-GR-Q', 'datasets/response'],
    ['?lens=spatial&view=atlas', 'datasets/relations'],
    ['?lens=spatial&view=relations', 'datasets/relations'],
    ['?lens=spatial&view=live', 'sentence/live'],
    ['?lens=spatial&view=local', 'datasets/local'],
    ['?lens=process&station=catalog', 'datasets/catalog'],
    ['?lens=process&station=probe', 'neurons/probe'],
    ['?lens=process&station=pipeline', 'admin//design'],
    ['?lens=process&station=pipeline&pstep=audit', 'admin//audit'],
    ['?lens=process&station=freeze', 'admin//freeze'],
    ['?station=audit', 'admin//audit'],
    ['?lens=process&station=census', 'admin//census'],
    ['?lens=process&station=autotest', 'admin//autotest'],
    ['?lens=process&station=contribute', 'contribute/'],
    ['?lens=process&station=ai', 'config/'],
    ['?lens=config&csec=model', 'config/'],
    ['?lens=datasets&sub=nope', 'datasets/overview'],
  ])('%s → %s', (search, want) => {
    expect(at(search)).toBe(want);
  });

  it('保留 tm 与例句状态', () => {
    expect(parseState('?lens=spatial&view=response&tm=TG-GR-Q').tm).toBe('TG-GR-Q');
    const s = parseState('?lens=sentence&case=CF-v1&cue=hearsay&fact=法国');
    expect([s.case, s.cue, s.fact]).toEqual(['CF-v1', 'hearsay', '法国']);
    expect(parseState('?lens=process&station=ai').csec).toBe('model');
  });

  it('写回时去掉旧参数、只写非缺省值、保留空间组件的键', () => {
    const q = new URLSearchParams(stateQuery({ lens: 'neurons', sub: 'atlas', step: 'design', csec: 'conn' }, 'lens=spatial&view=neuron&block=3&unit=7'));
    expect(q.get('lens')).toBe('neurons');
    expect(q.has('sub')).toBe(false);
    expect(q.has('view')).toBe(false);
    expect([q.get('block'), q.get('unit')]).toEqual(['3', '7']);
  });

  it('面板的旧跳转约定', () => {
    const prev = parseState('');
    expect(resolveGo(prev, 'process', 'pipeline').lens).toBe('admin');
    expect(resolveGo(prev, 'audit').step).toBe('audit');
    expect(resolveGo(prev, 'spatial', null, null, { view: 'lang' }).sub).toBe('lang');
    expect(resolveGo(prev, 'spatial', null, null, { tm: 'TG-X' })).toMatchObject({ lens: 'datasets', sub: 'response', tm: 'TG-X' });
    expect(resolveGo(prev, 'admin', null, 'census')).toMatchObject({ lens: 'admin', step: 'census' });
    expect(resolveGo(prev, 'route').lens).toBe('progress');
  });

  it('形态决定可见页面', () => {
    const ids = o => visibleLenses(o).map(l => l.id);
    expect(ids({ live: false, admin: true, localTools: false })).not.toContain('admin');
    expect(ids({ live: true, admin: false, localTools: false })).not.toContain('admin');
    expect(ids({ live: true, admin: true, localTools: true })).toEqual(expect.arrayContaining(['admin', 'local']));
    expect(visibleSubs('neurons', { live: false }).map(s => s.id)).not.toContain('probe');
  });
});

describe('机制报告读取', () => {
  const report = {
    priority_table: {
      plain: { n: 10, context_win_rate: 0.8, mean_d: 1.2, decisive_layer_median: 20, race_mean: [0, 1], top5_heads_dla: [{ head: '20.3', dla: 0.4 }] },
      negated: { n: 10, context_win_rate: 0.3 },
      hearsay: { n: 10, context_win_rate: 0.7 },
    },
  };

  it('优先级表：相对直接陈述的变化与翻转只按报告值算', () => {
    const rows = Object.fromEntries(priorityRows(report).map(r => [r.cue, r]));
    expect(rows.plain.delta).toBeNull();
    expect(rows.negated.flips).toBe(true);
    expect(rows.hearsay.flips).toBe(false);
    expect(rows.hearsay.delta).toBeCloseTo(-0.1);
    expect(rows.plain.heads[0].head).toBe('20.3');
    expect(rows.negated.race).toBeNull();
  });

  it('完整案例只看五项是否齐全', () => {
    const has = { priority_table: true, head_dla: true, head_total_effect: true, neurons: true, head_circuit_fidelity: true };
    expect(caseComplete({ has })).toBe(true);
    expect(caseComplete({ has: { ...has, neurons: false } })).toBe(false);
    expect(caseComplete({})).toBe(false);
  });

  it('未通过的判据排在前面', () => {
    expect(checkList({ a: true, b: false }).map(c => c.key)).toEqual(['b', 'a']);
    expect(checkList({ 'C2 · head_necessity': false })[0].label).toMatch(/^C2：所选头必要/);
  });

  it('v1 判据在平均 d < 0 时失效；v2 判据有中文名', () => {
    expect(v1Void({ heads: { heldout_mean_d: -4.17 } })).toMatch(/符号失效/);
    expect(v1Void({ heads: { heldout_mean_d: 0.5 } })).toBeNull();
    expect(v1Void({})).toBeNull();
    expect(checkLabel('v2_repeat2_U_upstream')).toMatch(/^v2 重复两次：U 组/);
    expect(checkLabel('v2_hierarchy_U_to_C')).toMatch(/U → C/);
    expect(voidList({ head_necessity: 'x' })).toEqual([{ key: 'head_necessity', label: expect.stringMatching(/必要/), why: 'x' }]);
  });

  it('小工具', () => {
    expect(parseHead('8.10')).toEqual([8, 10]);
    expect(parseHead('x')).toBeNull();
    expect(negate([[1, null], [-2]])).toEqual([[-1, null], [2]]);
    expect(markAnswers('法国的首都是罗马。法国的首都是', '罗马', '巴黎')).toEqual([
      { text: '法国的首都是', mark: '' }, { text: '罗马', mark: 'ctx' }, { text: '。法国的首都是', mark: '' },
    ]);
  });
});
