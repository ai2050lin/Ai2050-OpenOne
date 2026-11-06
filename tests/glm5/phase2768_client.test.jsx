import { beforeEach,afterEach,it,expect,vi } from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import ResearchFocus from '../../frontend/src/components/v8/ResearchFocus.jsx';
import ResearchFragments from '../../frontend/src/components/v8/ResearchFragments.jsx';
import ResearchExecution from '../../frontend/src/components/v8/ResearchExecution.jsx';
import {fragmentLocation,snapshotMatches,responseCompatible,fragmentHref} from '../../frontend/src/components/v8/researchFragmentUtil.js';
import {getJson} from '../../frontend/src/components/app/rdc_fusion/api.js';
const state=vi.hoisted(()=>({}));
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({ADMIN_LIVE:false,getJson:vi.fn(),useApi:path=>({data:state[path],reload:vi.fn()})}));
vi.mock('../../frontend/src/components/app/rdc_fusion/contributeUtil.js',()=>({nodeBridge:()=>null,nodeCall:vi.fn(),useNode:()=>({})}));
vi.mock('../../frontend/src/components/open/DataBar.jsx',()=>({ExtraPackPrompt:()=>null}));
let host,root;
const sha='a'.repeat(64),graph='b'.repeat(64),evidence='c'.repeat(64);
const card={id:'fixture:feature',label:'软件夹具 · 跨句式未复现',layer:18,boundary:'mlp.down_proj.input',index:{space:'mlp',unit:9031},
  evidence:{level:'negative',artifact_sha256:evidence},run_id:'fixture',counterexamples:[{id:'error'}],causal_tests:'not_measured',
  summary:{metrics:{lockbox:{auc:.92,n_entities:16},unseen_frame_and_entities:{auc:.68,n_entities:16}}}};
const model={id:'local-fixture',config:{num_hidden_layers:36,intermediate_size:9728}};
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;host=document.createElement('div');document.body.append(host);root=createRoot(host);vi.clearAllMocks();for(const k of Object.keys(state))delete state[k];
  window.history.replaceState(null,'','/workbench?artifact='+graph+'&feature='+encodeURIComponent(card.id));
  state['/api/research-library/artifacts/'+graph]={model_snapshot:sha,feature_cards:[card]};
});
afterEach(async()=>{await act(async()=>root.unmount());host.remove();window.history.replaceState(null,'','/workbench')});
async function render(component){await act(async()=>root.render(component))}
async function click(text){const b=[...host.querySelectorAll('button')].find(b=>b.textContent===text);await act(async()=>b.click())}
it('requires an exact content hash and a compatible coordinate boundary',()=>{
  expect(snapshotMatches(sha,{sha256:sha,source:'safetensors_header'})).toBe(false);
  expect(snapshotMatches('cfg-short',{sha256:'cfg-short',source:'full_file_sha256'})).toBe(false);
  expect(fragmentLocation({...card,boundary:'residual'},model.config)).toBe(null);
  expect(fragmentLocation({...card,index:{space:'mlp',unit:9728}},model.config)).toBe(null);
});
it('keeps desktop file links inside the same packaged page and preserves the workbench route',()=>{
  const original=globalThis.window;
  vi.stubGlobal('window',{location:{protocol:'file:',pathname:'/D:/client/resources/app.asar/dist/index.html'}});
  try{const href=fragmentHref({sha:graph,id:card.id,body:card});expect(href.startsWith('?')).toBe(true);expect(new URLSearchParams(href).get('wb')).toBe('1')}
  finally{vi.stubGlobal('window',original)}
});
it('locates the complete native unit index after successful full snapshot verification',async()=>{
  getJson.mockResolvedValue({sha256:sha,source:'full_file_sha256'});const locate=vi.fn();
  await render(<ResearchFocus model={model} onLocate={locate}/>);await click('核验本机权重并定位');
  expect(locate).toHaveBeenCalledWith({block:18,kind:'mlp',unit:9031});expect(host.textContent).toContain('完整快照一致');
});
it('does not bind a feature to another weight snapshot',async()=>{
  getJson.mockResolvedValue({sha256:'d'.repeat(64),source:'full_file_sha256'});const locate=vi.fn();
  await render(<ResearchFocus model={model} onLocate={locate}/>);await click('核验本机权重并定位');
  expect(locate).not.toHaveBeenCalled();expect(host.textContent).toContain('完整权重快照不一致');
});
it('ignores an in-flight verification if the selected model changes',async()=>{
  let finish;getJson.mockImplementation(()=>new Promise(resolve=>{finish=resolve}));const locate=vi.fn();
  await render(<ResearchFocus model={model} onLocate={locate}/>);await click('核验本机权重并定位');
  await render(<ResearchFocus model={{...model,id:'other'}} onLocate={locate}/>);
  await act(async()=>finish({sha256:sha,source:'full_file_sha256'}));expect(locate).not.toHaveBeenCalled();
});
it('shows original input and measured activation with the evidence report',async()=>{
  state['/api/research-library/artifacts/'+evidence]={sample_preview:[{id:'fixture',entity:'apple',split:'lockbox',tokens:304,activation:.12,text:'software fixture original input'}]};
  await render(<ResearchFocus model={model} onLocate={vi.fn()}/>);await click('查看证据与输入');
  expect(host.textContent).toContain('software fixture original input');expect(host.textContent).toContain('304 tokens');
});
it('retains the negative label and provides a versioned parameter link',async()=>{
  state['/api/research-library/fragments?offset=0&limit=50']={features:[{sha:graph,id:card.id,model_snapshot:sha,body:card,history:[{sha:graph,level:'negative'},{sha:evidence,level:'observation'}]}],total:1};
  await render(<ResearchFragments onArtifact={vi.fn()}/>);
  expect(host.textContent).toContain('未支持 / 负结果');expect(host.textContent).toContain('2 次修订');
  const url=new URL(host.querySelector('a').href);expect(url.searchParams.get('feature')).toBe(card.id);expect(url.searchParams.get('artifact')).toBe(graph);
});
it('keeps unsupported scientific validation distinct from completed execution stages',async()=>{
  state['/api/research-library/workpackages']={workpackages:[{sha:graph,body:{run_id:'fixture',title:'软件工作包',updated_at:1,stages:{collection:{state:'done'},validation:{state:'not_supported'},intervention:{state:'ready'}}}}]};
  state['/api/tasks']={stats:{queued:57},queued:[],by_template:[{tm_id:'fixture-tasks',template_name:'中心测试',total:65,done:5,active:2,queued:57,failed:1,expired:0}]};
  await render(<ResearchExecution plan onGo={vi.fn()}/>);
  const progress=host.querySelector('progress[aria-label="中心测试已完成任务"]');
  expect(progress.value).toBe(5);expect(progress.max).toBe(65);
  expect(host.querySelector('.v8-workpackage-stages')).toBe(null);
  await expandPlanStages();
  expect(host.textContent).toContain('原场采集');expect(host.textContent).toContain('△ 未支持');
});
async function expandPlanStages(){
  const open=async details=>act(async()=>{details.open=true;details.dispatchEvent(new Event('toggle'))});
  await open([...host.querySelectorAll('summary')].find(el=>el.textContent.startsWith('研究计划')).parentElement);
  await open([...host.querySelectorAll('summary')].find(el=>el.textContent.includes('软件工作包')).parentElement);
  await open([...host.querySelectorAll('summary')].find(el=>el.textContent==='阶段与验收').parentElement);
}
it('renders producer stages chronologically after canonical JSON has sorted their keys',async()=>{
  const stages={client_acceptance:{time:30,state:'done'},collection:{time:20,state:'done'},freeze:{time:10,state:'done'}};
  state['/api/research-library/workpackages']={workpackages:[{sha:graph,body:{run_id:'fixture',title:'软件工作包',updated_at:30,stages}}]};
  await render(<ResearchExecution plan onGo={vi.fn()}/>);
  await expandPlanStages();
  const labels=[...host.querySelectorAll('.v8-workpackage-stages b')].map(n=>n.textContent);
  expect(labels).toEqual(['冻结材料','原场采集','客户端回查']);
});
it('rejects response arrays with another snapshot, boundary, missing units or nonfinite values',()=>{
  const c={...card,scope:{readout:'final'}};
  const r={schema:'research_mlp_response.v1',kind:'real',model_snapshot:sha,block:18,boundary:card.boundary,readout:'final',run_id:'fixture',selected_unit:9031,values:Array(9728).fill(.01)};
  expect(responseCompatible(r,c,sha,model.config)).toBe(true);
  for(const bad of [{...r,model_snapshot:graph},{...r,boundary:'residual'},{...r,run_id:'another-run'},{...r,selected_unit:2},{...r,values:[.01]},{...r,values:r.values.map((v,i)=>i===9727?NaN:v)}])expect(responseCompatible(bad,c,sha,model.config)).toBe(false);
});
it('passes the full measured array only after the local snapshot has been verified',async()=>{
  const rs='d'.repeat(64),c={...card,scope:{readout:'final'}};
  state['/api/research-library/artifacts/'+graph]={model_snapshot:sha,feature_cards:[c]};
  state['/api/research-library/artifacts/'+evidence]={sample_preview:[{id:'sample',entity:'apple',split:'lockbox',tokens:303,activation:.01,text:'fixture',response_sha256:rs}]};
  state['/api/research-library/artifacts/'+rs]={schema:'research_mlp_response.v1',kind:'real',model_snapshot:sha,block:18,boundary:card.boundary,readout:'final',run_id:'fixture',selected_unit:9031,values:Array.from({length:9728},(_,i)=>i===9727?-.001:.01)};
  const response=vi.fn();getJson.mockResolvedValue({sha256:sha,source:'full_file_sha256'});
  await render(<ResearchFocus model={model} onLocate={vi.fn()} onResponse={response}/>);await click('查看证据与输入');
  const first=[...host.querySelectorAll('button')].find(b=>b.textContent==='显示该输入的完整MLP响应');expect(first.disabled).toBe(true);
  await click('核验本机权重并定位');await click('显示该输入的完整MLP响应');
  expect(response).toHaveBeenCalled();expect(response.mock.lastCall[0].tokens).toBe(303);expect(response.mock.lastCall[0].values).toHaveLength(9728);expect(response.mock.lastCall[0].values[9727]).toBe(-.001);
  const calls=response.mock.calls.length;await click('查看证据与输入');await click('显示该输入的完整MLP响应');expect(response.mock.calls.length).toBeGreaterThan(calls);
});
