import { it,expect,vi,beforeEach,afterEach } from 'vitest';
import React,{ act } from '../../frontend/node_modules/react/index.js';
import { createRoot } from '../../frontend/node_modules/react-dom/client.js';
import ResearchLibrary from '../../frontend/src/components/v8/ResearchLibrary.jsx';
import ResearchOverview from '../../frontend/src/components/v8/ResearchOverview.jsx';
import ParameterExplorer from '../../frontend/src/components/v8/ParameterExplorer.jsx';
import { preferredParameter } from '../../frontend/src/components/v8/parameterUtil.js';
import ManagedNodes from '../../frontend/src/components/app/rdc_fusion/ManagedNodes.jsx';
import { authJson } from '../../frontend/src/components/app/rdc_fusion/api.js';
const state=vi.hoisted(()=>({loading:false,guestUser:false}));
const sha='a'.repeat(64),cfg={num_hidden_layers:36,num_attention_heads:32,num_key_value_heads:8,head_dim:128,hidden_size:2560,intermediate_size:9728,vocab_size:151936};
function api(path){
 const reload=vi.fn();
 if(path==='/api/parameters/models')return {loading:state.loading,data:state.loading?null:{models:[{id:'1111111',name:'Qwen3-4B',config:cfg},{id:'2222222',name:'Qwen3-4B',config:cfg}]},reload};
 if(path?.includes('/parameters/tensors'))return {data:{tensors:[{name:'model.layers.0.self_attn.k_norm.weight',shape:[128],dtype:'BF16'},{name:'model.layers.0.self_attn.q_proj.weight',shape:[4096,2560],dtype:'BF16'}]},reload};
 if(path?.includes('/parameters/slice'))return {data:{values:[[.125]],start:[0,0],source:'safetensors_exact_slice',loaded_numel:1,header_sha256:sha},reload};
 if(path==='/api/neurons/explained?model_key=fixture-census')return {data:{model_key:'fixture-census',results:[
   {block:0,unit:1,grade:'文本可复现'},{block:0,unit:1,grade:'文本可复现'},{block:1,unit:2,grade:'未解释'},{block:2,unit:0,grade:'未解释'},
 ]},reload};
 const item={sha,source:'fixture',model:'fixture-model',kind:'local_report',created:1};
 if(path==='/api/research-library/progress')return {data:{buckets:[{id:'fixture',model:'fixture',snapshot:'cfg-fixture',precision:'bf16',snapshot_verified:false,templates:[{tm_id:'TG-KN-FIXTURE',name:'冻结模板用例',planned:10,done:0}]}]},reload};
 if(path==='/api/research-library/workpackages')return {data:{workpackages:[{body:{run_id:'run-A',title:'A 工作包',model_snapshot:sha,stages:{registration_validated:{report_sha256:sha}}}},{body:{run_id:'run-B',title:'B 工作包',model_snapshot:'b'.repeat(64),stages:{}}}]},reload};
 if(path==='/api/research-library/overview')return {data:{artifacts:1,graph_revisions:{feature:0,edge:0},recent:[item],note:'fixture'},reload};
 if(path?.startsWith('/api/research-library/artifacts?'))return {data:{artifacts:[item],total:1},reload};
 if(path?.endsWith('/artifacts/'+sha))return {data:{kind:'software_fixture',hypothesis:'unverified'},reload};
 if(path?.includes('/research-library/graph'))return {data:{objects:[]},reload};
 return {data:null,reload};
}
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({useApi:path=>api(path),authJson:vi.fn(),ADMIN_LIVE:false}));
vi.mock('../../frontend/src/components/app/rdc_fusion/ui.jsx',()=>({Empty:({children})=><div>{children}</div>}));
vi.mock('../../frontend/src/components/app/rdc_fusion/authContext.js',()=>({useAuth:()=>state.guestUser?{user:{username:'guest'},guest:true}:{user:null,guest:false}}));
vi.mock('../../frontend/src/components/open/DataBar.jsx',()=>({ExtraPackPrompt:()=>null}));
vi.mock('../../frontend/src/components/app/rdc_fusion/contributeUtil.js',()=>({nodeBridge:()=>null,nodeCall:vi.fn()}));
vi.mock('../../frontend/src/components/v8/Tower.jsx',()=>({default:({nLayers,height})=><div data-height={height} data-testid="tower">{nLayers}层</div>}));
let container,root;
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;window.innerWidth=1280;state.loading=false;state.guestUser=false;vi.clearAllMocks();container=document.createElement('div');document.body.append(container);root=createRoot(container)});
afterEach(()=>{act(()=>root.unmount());container.remove();vi.restoreAllMocks()});

it('prefers the actual Q and gate matrices instead of shared norms or bias',()=>{
 const q={name:'layer.self_attn.q_proj.weight'},norm={name:'layer.self_attn.k_norm.weight'};
 expect(preferredParameter([norm,q],'head')).toBe(q);
 const fused={name:'layer.mlp.gate_up_proj.weight',id:'layer.mlp.gate_up_proj.weight:gate'};
 expect(preferredParameter([{name:'layer.mlp.down_proj.weight'},fused],'mlp')).toBe(fused);
});
it('fits the full layer tower in the narrow-screen panel',async()=>{
 await act(async()=>root.render(<ParameterExplorer/>));expect(container.querySelector('[data-testid=tower]').dataset.height).toBe('570');
 await act(async()=>{window.innerWidth=480;window.dispatchEvent(new Event('resize'))});
 expect(container.querySelector('[data-testid=tower]').dataset.height).toBe('280');expect(container.textContent).toContain('36层');
});
it('shows loading instead of claiming the model directory is empty',async()=>{
 state.loading=true;await act(async()=>root.render(<ParameterExplorer/>));
 expect(container.textContent).toContain('正在读取本机模型目录');expect(container.textContent).not.toContain('尚未找到');
});
it('distinguishes identical model names by local directory identity',async()=>{
 await act(async()=>root.render(<ParameterExplorer/>));
 const labels=[...container.querySelectorAll('select')][0].textContent;
 expect(labels).toContain('目录 111111');expect(labels).toContain('目录 222222');
});
it('opens an artifact in a visible modal and Escape restores the source list',async()=>{
 await act(async()=>root.render(<ResearchLibrary/>));
 const sources=container.querySelector('details');
 expect(sources.open).toBe(false);
 expect(container.querySelector('table')).toBeNull();
 await act(async()=>{sources.open=true;sources.dispatchEvent(new Event('toggle'))});
 await act(async()=>[...container.querySelectorAll('button')].find(b=>b.textContent===sha.slice(0,12)).click());
 const dialog=container.querySelector('dialog');expect(dialog.open).toBe(true);expect(dialog.textContent).toContain('unverified');
 await act(async()=>dialog.dispatchEvent(new Event('cancel',{bubbles:false,cancelable:true})));
 expect(container.querySelector('dialog')).toBe(null);expect(sources.open).toBe(true);expect(sources.textContent).toContain('fixture');
});
it('never polls private nodes for a guest placeholder user',async()=>{
 state.guestUser=true;await act(async()=>root.render(<ManagedNodes/>));
 expect(container.textContent).toContain('登录后查看本人绑定节点');expect(authJson).not.toHaveBeenCalled();
});
it('shows three summaries for the URL snapshot and keeps historical denominators in the disclosure',async()=>{
 const previous=window.location.href;
 try{
  window.history.replaceState(null,'','?snapshot='+sha);
  function OverviewFixture(){
    const [open,setOpen]=React.useState('');
    return <ResearchOverview open={open} onGo={(_lens,_view,_context,patch)=>setOpen(patch.opane)} model="fixture"
      skeletons={{fixture:{model:'fixture',model_snapshot:sha,n_layers:2,d_ff:4,census_run:'fixture-run',census_key:'fixture-census'}}}
      tree={{models:['fixture'],items:[{id:'KN.ATTR',status:{by_model:{fixture:1}}},{id:'UNMAPPED',status:{by_model:{fixture:4}}}]}}/>;
  }
  await act(async()=>root.render(<OverviewFixture/>));
  const cards=[...container.querySelectorAll('.v8-direction')];
  expect(cards.map(card=>card.getAttribute('aria-label'))).toEqual(['外部语言图谱','内部模型图谱','两图谱的关联与验证']);
  expect(cards[0].textContent).toContain('1 / 69 项 · 1.45%');
  expect(cards[1].textContent).toContain('2 / 8 个 · 25.00%');
  expect(container.textContent).not.toContain('待核验');
  expect(container.querySelector('#overview-detail')).toBeNull();
  await act(async()=>cards[2].click());
  expect(container.querySelector('#overview-detail').textContent).toContain('A 工作包');
  expect(container.querySelector('#overview-detail').textContent).not.toContain('B 工作包');
  const history=container.querySelector('.v8-overview-fold');
  await act(async()=>{history.open=true;history.dispatchEvent(new Event('toggle'))});
  expect(history.textContent).toContain('当前范围暂无完整快照的冻结分片记录');
  const select=history.querySelector('select');
  await act(async()=>{select.value='fixture';select.dispatchEvent(new Event('change',{bubbles:true}))});
  expect(history.textContent).toContain('待核验 · 计划 10');
 }finally{window.history.replaceState(null,'',previous)}
});
