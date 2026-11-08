import { describe, expect, it } from 'vitest';

import { f32, liveSampler } from './liveRun.js';

const enc = (arr, shape) => { const b = new Uint8Array(new Float32Array(arr).buffer); let s = ''; for (const x of b) s += String.fromCharCode(x); return { shape, dtype: 'float32', b64: btoa(s) }; };

describe('实时运行 bundle 的取值器（Phase 348）', () => {
  const NL = 3, DM = 4, NB = 2, NH = 2, T = 3, NF = 5;
  const hs = Array.from({ length: NL * DM }, (_, i) => i / 10), at = Array.from({ length: NB * NH * T }, (_, i) => i), mlp = Array.from({ length: NB * NF }, (_, i) => -i);
  const b = { hs: enc(hs, [NL, DM]), attn: enc(at, [NB, NH, T]), mlp: enc(mlp, [NB, NF]), attn_out: enc(Array(NB * DM).fill(1), [NB, DM]), mlp_out: enc(Array(NB * DM).fill(2), [NB, DM]), model_snapshot: 'x', readout_positions: [0, 2] };
  it('解码 float32 并按层 / 块 / 头取值', () => {
    expect(Array.from(f32(b.hs).a).map(v => +v.toFixed(5))).toEqual(hs.map(v => +v.toFixed(5)));
    const s = liveSampler(b, { tokens: ['a', 'b', 'c'] });
    expect([s.NL, s.DM, s.NB, s.NH, s.T, s.NF]).toEqual([NL, DM, NB, NH, T, NF]);
    expect(Array.from(s.hs(99, 2)).map(v => +v.toFixed(5))).toEqual([0.8, 0.9, 1, 1.1]);   // 位置参数被忽略：bundle 只含一个 token
    expect(Array.from(s.att(1, 1))).toEqual([9, 10, 11]);
    expect(Array.from(s.mlp(1))).toEqual([-5, -6, -7, -8, -9]);
    expect(s.hasMlp && s.hasSplit).toBe(true);
    expect(s.d.tokpos).toEqual([0, 1, 2]);
  });
  it('不是模块读出位置时没有 MLP 与写入拆分', () => {
    const s = liveSampler({ ...b, mlp: undefined, attn_out: undefined, mlp_out: undefined }, { tokens: ['a', 'b', 'c'] });
    expect(s.hasMlp).toBe(false);
    expect(s.hasSplit).toBe(false);
    expect(s.mlp(0)).toBeNull();
  });
});
