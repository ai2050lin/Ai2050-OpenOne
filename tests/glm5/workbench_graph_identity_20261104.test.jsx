import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import AtlasView from '../../frontend/src/components/app/rdc_fusion/AtlasView.jsx';
import {nativePoint} from '../../frontend/src/components/v8/ResearchSpace.jsx';
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({useApi:()=>({data:null}),API_BASE:'',ADMIN_LIVE:false,fmtNum:String,fmtAgoTs:String,postJson:vi.fn(),useAdminToken:()=>['',vi.fn()]}));
let host,root;
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;host=document.createElement('div');document.body.append(host);root=createRoot(host)});
afterEach(async()=>{await act(async()=>root.unmount());host.remove()});
it('cross-family graphs never substitute the first model when the requested bucket is absent',async()=>{
  await act(async()=>root.render(<AtlasView localAtlas={{graphs:[{model:'another-model',layers:[],nodes:[],axes:[]}],sources:[]}} modelKey="requested-model"/>));
  expect(host.textContent).toContain('未替换成其他模型');
});
it('embedding observations remain on the embedding plane and incomplete locations stay unlocated',()=>{
  const model={n_layers:36,d_model:2560,d_ff:9728,n_heads:32};
  expect(nativePoint({body:{layer:0,boundary:'hidden[0]',index:{space:'residual',coordinate:2559}}},model)[1]).toBe(-1.5);
  expect(nativePoint({body:{layer:36,boundary:'mlp.down_proj.input',index:{space:'mlp',unit:10}}},model)).toBe(null);
  expect(nativePoint({body:{layer:35,boundary:'mlp.down_proj.input',index:{space:'mlp',unit:9728}}},model)).toBe(null);
});
