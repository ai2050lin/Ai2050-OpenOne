import { describe,it,expect,vi,beforeEach,afterEach } from 'vitest';
import React,{ act } from '../../frontend/node_modules/react/index.js';
import { createRoot } from '../../frontend/node_modules/react-dom/client.js';
import ParameterExplorer from '../../frontend/src/components/v8/ParameterExplorer.jsx';
import { parameterRegion } from '../../frontend/src/components/v8/parameterUtil.js';
import { readState,resolveGo,visibleStations,ROUTE_VIEWS,SPATIAL_VIEWS } from '../../frontend/src/components/app/workbenchUtil.js';
const mocks=vi.hoisted(()=>({paths:[],local:false,request:vi.fn()}));
const cfg={num_attention_heads:32,num_key_value_heads:8,hidden_size:2560,head_dim:128,intermediate_size:9728,num_hidden_layers:36,vocab_size:151936};
function response(path){
  if(path?.endsWith('/models'))return {models:[{id:'real-model-id',name:'Qwen3-4B',config:cfg}]};
  if(path?.includes('/tensors'))return {tensors:[{name:'model.layers.0.self_attn.q_proj.weight',shape:[4096,2560],dtype:'BF16'},{name:'model.layers.0.mlp.gate_proj.weight',shape:[9728,2560],dtype:'BF16'}]};
  if(path?.includes('/slice'))return {values:[[0.125,-0.5]],start:[0,0],source:'safetensors_exact_slice',loaded_numel:2,header_sha256:'a'.repeat(64)};
  return null;
}
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({useApi:path=>{mocks.paths.push(path);return {data:response(path),error:'',reload:vi.fn()}}}));
vi.mock('../../frontend/src/components/app/rdc_fusion/ui.jsx',()=>({Empty:({children})=><div>{children}</div>}));
vi.mock('../../frontend/src/components/v8/Tower.jsx',()=>({default:({nLayers})=><div data-testid="tower">{nLayers}层</div>}));
vi.mock('../../frontend/src/components/app/rdc_fusion/contributeUtil.js',()=>({nodeBridge:()=>mocks.local?{}:null,nodeCall:(...args)=>mocks.request(...args)}));
let container,root;
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;window.history.replaceState(null,'','/');mocks.paths=[];mocks.local=false;container=document.createElement('div');document.body.append(container);root=createRoot(container);vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue({clearRect(){},fillRect(){}})});
afterEach(()=>{act(()=>root.unmount());container.remove();vi.restoreAllMocks()});

describe('native parameter regions',()=>{
 it('maps every Q head and shared GQA K/V group without using hidden/head ratio',()=>{
  for(let h=0;h<32;h++){
   expect(parameterRegion(cfg,'head',h,'q_proj.weight')).toMatchObject({row:h*128,rows:128,cols:2560});
   expect(parameterRegion(cfg,'head',h,'k_proj.weight')).toMatchObject({row:Math.floor(h/4)*128,rows:128,shared:true});
   expect(parameterRegion(cfg,'head',h,'o_proj.weight')).toMatchObject({col:h*128,rows:2560,cols:128});
  }
 });
 it('supports the last head and last MLP unit of Qwen3-14B',()=>{
  const c={...cfg,hidden_size:5120,num_attention_heads:40,intermediate_size:17408};
  expect(parameterRegion(c,'head',39,'v_proj.weight').row).toBe(896);
  expect(parameterRegion(c,'mlp',17407,'down_proj.weight')).toMatchObject({col:17407,rows:5120,cols:1});
 });
 it('splits GLM fused gate/up rows and preserves biases/norm shapes',()=>{
  const c={...cfg,hidden_size:4096,num_key_value_heads:2,intermediate_size:13696};
  expect(parameterRegion(c,'mlp',13695,'gate_up_proj.weight:up').row).toBe(27391);
  expect(parameterRegion(c,'mlp',13695,'gate_up_proj.weight:gate').row).toBe(13695);
  expect(parameterRegion(c,'head',31,'k_proj.bias')).toMatchObject({row:128,cols:1,shared:true});
  expect(parameterRegion(c,'head',31,'q_norm.weight')).toMatchObject({row:0,rows:128,cols:1});
 });
});

describe('formal client navigation and rendering',()=>{
 it('does not substitute another model when a bookmarked local directory is missing',async()=>{
  window.history.replaceState(null,'','?lens=spatial&view=parameters&pmodel=missing-directory');
  await act(async()=>root.render(<ParameterExplorer/>));
  expect(container.textContent).toContain('未替换成其他模型');expect(container.querySelectorAll('.v8-head-grid button')).toHaveLength(0);
 });
 it('clamps invalid bookmarked indices to the actual model dimensions',async()=>{
  window.history.replaceState(null,'','?lens=spatial&view=parameters&pblock=999&phead=999&punit=999999');
  await act(async()=>root.render(<ParameterExplorer/>));
  expect(container.querySelector('[aria-label="参数块"]').value).toBe('35');
  expect(new URLSearchParams(window.location.search).get('phead')).toBe('31');
  expect(new URLSearchParams(window.location.search).get('punit')).toBe('9727');
 });
 it('opens a concise progress overview and preserves legacy routes',()=>{
  window.history.replaceState(null,'','/');const state=readState();
  expect(state).toMatchObject({lens:'route',rtab:'overview',station:'plan',view:'graph'});
  expect(ROUTE_VIEWS.map(v=>v.id)).toEqual(['overview','rules']);expect(SPATIAL_VIEWS).toHaveLength(4);
  window.history.replaceState(null,'','?lens=route&rtab=lang');expect(readState()).toMatchObject({lens:'route',rtab:'overview',opane:'language'});
  window.history.replaceState(null,'','?station=audit');expect(readState()).toMatchObject({lens:'process',station:'pipeline',pstep:'audit'});
  expect(resolveGo(state,'config',null,null,{csec:'platform'}).csec).toBe('platform');
  expect(visibleStations({live:false}).map(v=>v.id)).toEqual(['plan','catalog','execution']);
 });
 it('renders all heads, reads actual slices, and opens the complete MLP array',async()=>{
  await act(async()=>root.render(<ParameterExplorer researchModel="Qwen3-4B"/>));
  expect(container.querySelectorAll('.v8-head-grid button')).toHaveLength(32);
  expect(container.textContent).toContain('0.12500');expect(container.textContent).toContain('36层');
  await act(async()=>container.querySelectorAll('.v8-head-grid button')[31].click());
  expect(mocks.paths.some(p=>p?.includes('row=3968'))).toBe(true);
  await act(async()=>[...container.querySelectorAll('button')].find(b=>b.textContent==='全部MLP单元').click());
  expect(container.querySelector('canvas').getAttribute('aria-label')).toContain('9728');
  expect(container.querySelector('input[aria-label="参数单元"]').max).toBe('9727');
 });
 it('desktop reads local IPC parameters instead of remote center weights',async()=>{
  mocks.local=true;mocks.request.mockImplementation(async(_method,path)=>response(path));
  await act(async()=>root.render(<ParameterExplorer/>));
  expect(mocks.request.mock.calls.some(c=>c[1]==='/parameters/models')).toBe(true);
  expect(mocks.paths.filter(Boolean)).toHaveLength(0);
  expect(container.querySelectorAll('.v8-head-grid button')).toHaveLength(32);
 });
});
