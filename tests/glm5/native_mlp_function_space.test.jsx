import {it,expect} from 'vitest';
import {nativePoint} from '../../frontend/src/components/v8/ResearchSpace.jsx';
const m={n_layers:36,d_ff:9728,n_heads:32,d_model:2560};
it('locates native unit9727 without truncating the array',()=>{const p=nativePoint({body:{layer:35,index:{space:'mlp',unit:9727}}},m);expect(p).toHaveLength(3);expect(p[1]).toBeLessThan(1.5)});
it('keeps missing or invalid coordinates on an unlocated artifact rack',()=>{expect(nativePoint({body:{layer:4,index:{space:'mlp',unit:9728}}},m)).toBeNull();expect(nativePoint({body:{layer:4,index:{space:'dictionary',feature:2}}},m)).toBeNull()});
it('does not invent native geometry without a real model dimension',()=>{expect(nativePoint({body:{layer:4,index:{space:'mlp',unit:2}}},{})).toBeNull()});
it('uses separate attention andMLP native planes',()=>{const a=nativePoint({body:{layer:4,index:{space:'attention',head:31}}},m);const b=nativePoint({body:{layer:4,index:{space:'mlp',unit:31}}},m);expect(a[2]).toBeLessThan(0);expect(b[2]).toBeGreaterThan(0)});
it('locates explicit layer-level writes and cache outside unit grids',()=>{
  const write=nativePoint({body:{layer:35,index:{space:'mlp',field:'residual_write'}}},m);
  const cache=nativePoint({body:{layer:35,index:{space:'attention',field:'kv_cache'}}},m);
  expect(write[0]).toBeGreaterThan(1);expect(cache[0]).toBeLessThan(-1);expect(cache[1]).toBe(write[1]);
});
