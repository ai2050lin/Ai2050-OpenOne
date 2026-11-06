import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import ResearchInterventionReport from '../../frontend/src/components/v8/ResearchInterventionReport.jsx';
import DatasetAtlasPlan from '../../frontend/src/components/v8/DatasetAtlasPlan.jsx';
import ResearchExecution from '../../frontend/src/components/v8/ResearchExecution.jsx';
const state=vi.hoisted(()=>({}));
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({ADMIN_LIVE:false,useApi:path=>({data:state[path],reload:vi.fn()})}));
vi.mock('../../frontend/src/components/open/DataBar.jsx',()=>({ExtraPackPrompt:()=>null}));
const sha='7'.repeat(64);let host,root;
function fixture(){return {schema:'research_natural_register_report.v1',run_id:'software-fixture',
  model_snapshot:'a'.repeat(64),matching_audit:{pairs:256,mismatched_pairs:0,lengths:[384,512]},
  pilots:[{run_id:'calibration-failed',style_accuracy:{acad:.9,colloq:.3},content_accuracy:.9,passed:false}],
  result:{baseline_gate:true,baseline_cells:{'F2/L384/acad':.9},hypotheses:{legacy512_a6:{
    qualified_flip:{mean:0,ci:[0,0]},colloquial_loss:{ci:[0,.01]},colloquial_source_preserved:.2,
    content_loss:{ci:[-.01,.01]},capability:{status:'noninferior',n:2200,net_loss:0,loss_interval:[-.01,.01]},
    comparisons:{random:{mean:0,ci:[0,0]}},supported:false}},
    counterexamples:Array.from({length:25},(_,i)=>({sample_id:'failure-'+i,condition:'target',reason:'software fixture'}))},
  limitations:['软件测试夹具，不是研究结果']}}
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;host=document.createElement('div');document.body.append(host);root=createRoot(host);
  state['/api/research-library/artifacts/'+sha]=fixture()});
afterEach(async()=>{await act(async()=>root.unmount());host.remove()});
async function render(){await act(async()=>root.render(<ResearchInterventionReport sha={sha}/>))}
async function click(t){await act(async()=>[...host.querySelectorAll('button')].find(b=>b.textContent===t).click())}
it('routes natural reports and separates NI success from failed generation hypothesis',async()=>{
  await render();expect(host.querySelector('[aria-label="自然生成语域研究报告"]')).not.toBe(null);
  expect(host.textContent).toContain('未通过预注册主判');expect(host.textContent).toContain('通过非劣效门');
  expect(host.textContent).toContain('未提取等长度新方向');expect(host.textContent).toContain('不能说明真实效应恰为零');
  expect(host.textContent).not.toContain('范围内支持');expect(host.textContent).not.toContain('failure-0');
});
it('keeps exact active-token matching and cell gates inspectable',async()=>{
  await render();await click('长度与句式');expect(host.textContent).toContain('256');
  expect(host.textContent).toContain('384 / 512');expect(host.textContent).toContain('F2/L384/acad');
});
it('preserves failed pilot histories and bounded full-output counterexamples',async()=>{
  await render();expect(host.textContent).toContain('calibration-failed');await click('输出与反例');
  expect(host.textContent).toContain('failure-0');expect(host.textContent).not.toContain('failure-24');
  await click('下一组');expect(host.textContent).toContain('failure-24');expect(host.textContent).not.toContain('failure-0');
});
it('does not present calibration alone as a formal mechanism result',async()=>{
  state['/api/research-library/artifacts/'+sha].result=null;await render();
  expect(host.textContent).toContain('正式机制结论未建立');expect(host.textContent).not.toContain('通过非劣效门');
});
it('leads with the failed measurement gate and does not interpret zero as no gates',async()=>{
  const d=state['/api/research-library/artifacts/'+sha];d.result.baseline_gate=false;
  d.result.baseline_by_register={acad:.3,colloq:.02};await render();
  expect(host.textContent).toContain('留出行为门未通过，尚不能验收');
  expect(host.textContent).toContain('不能说明模型没有语域门控');
});
it('loads the dataset plan only when expanded and marks it as proposed',async()=>{
  state['/api/research-library/artifacts/'+sha]={schema:'research_dataset_atlas_plan.v1',
    principles:[{title:'共同图谱',description:'语域的正负结果按主张范围保存'}],
    client_pages:[],dataset_candidates:[],stages:[]};
  await act(async()=>root.render(<DatasetAtlasPlan sha={sha}/>));
  expect(host.textContent).toContain('待实施');expect(host.textContent).not.toContain('共同图谱');
  await act(async()=>host.querySelector('details').open=true);
  await act(async()=>host.querySelector('details').dispatchEvent(new Event('toggle')));
  expect(host.textContent).toContain('共同图谱');expect(host.textContent).toContain('不表示新数据已经接入');
  expect(host.textContent).toContain('数据集选型与使用状态');
});
it('keeps calibration runs under their parent without duplicate top-level rows',async()=>{
  window.history.replaceState(null,'','/workbench?run=parent-fixture');
  state['/api/research-library/workpackages']={workpackages:[
    {sha:'1',body:{run_id:'child-fixture',parent_run_id:'parent-fixture',title:'校准子运行',updated_at:1,stages:{}}},
    {sha:'2',body:{run_id:'parent-fixture',title:'正式研究',updated_at:2,stages:{}}}]};
  await act(async()=>root.render(<ResearchExecution plan onGo={()=>{}}/>));
  expect([...host.querySelectorAll('summary strong')].filter(x=>x.textContent==='校准子运行')).toHaveLength(0);
  const source=[...host.querySelectorAll('summary')].find(el=>el.textContent.startsWith('研究计划')).parentElement;
  await act(async()=>{source.open=true;source.dispatchEvent(new Event('toggle'))});
  expect(host.textContent).toContain('前置校准与试跑（1）');
  const details=[...host.querySelectorAll('summary')].find(el=>el.textContent==='前置校准与试跑（1）').parentElement;
  await act(async()=>{details.open=true;details.dispatchEvent(new Event('toggle'))});
  expect([...host.querySelectorAll('summary strong')].filter(x=>x.textContent==='校准子运行')).toHaveLength(1);
});
