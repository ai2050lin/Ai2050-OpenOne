import {afterEach,expect,it,vi} from 'vitest';
vi.mock('../../frontend/src/config.js',()=>({ADMIN_LIVE:false,API_BASE:'https://center.example',DESKTOP:true,OFFLINE_CLIENT:true,REMOTE_CENTER:true,WEB_URL:''}));
vi.mock('../../frontend/src/components/app/rdc_fusion/dataStore.js',()=>({getLocal:()=>undefined,getExtraEntry:async()=>undefined,getOnDemand:()=>undefined,isOnDemand:()=>false,packHint:()=>null,packInfo:()=>({}),putOnDemand:()=>{},storeTick:()=>0,subscribe:()=>()=>{}}));
import {getJson} from '../../frontend/src/components/app/rdc_fusion/api.js';
afterEach(()=>vi.unstubAllGlobals());
it('reads published 14B arrays from the center in desktop read-only mode',async()=>{
  const fetch=vi.fn().mockResolvedValue({ok:true,json:async()=>({width:17408})});vi.stubGlobal('fetch',fetch);
  expect(await getJson('/api/research/qwen14-mlp/layers/35')).toEqual({width:17408});
  expect(fetch.mock.calls[0][0]).toBe('https://center.example/api/research/qwen14-mlp/layers/35');
});
it('does not silently replace missing 14B data with a local empty package',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:false,status:404,json:async()=>({detail:'14B archive missing'})}));
  await expect(getJson('/api/research/qwen14-mlp')).rejects.toThrow('14B archive missing');
});
