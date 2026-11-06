import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import ResearchInterventionReport from '../../frontend/src/components/v8/ResearchInterventionReport.jsx';
import ResearchExecution from '../../frontend/src/components/v8/ResearchExecution.jsx';
const state=vi.hoisted(()=>({}));
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({useApi:path=>({data:state[path],reload:vi.fn()})}));
vi.mock('../../frontend/src/components/open/DataBar.jsx',()=>({ExtraPackPrompt:()=>null}));
let host,root;
const sha='f'.repeat(64);
function fixture(){return {schema:'research_intervention_report.v1',result:{run_id:'software-fixture',env:{precision:'bf16'},
  model_snapshot:'a'.repeat(64),target_supported_in_scope:false,capability:{status:'inconclusive',n:2200,net_loss:.01,
  loss_interval:[-.005,.03],margin:.02},lockbox:{target:{flip_fraction:.01}},selected:{k:32,alpha:1},
  compactness:{status:'not_found'},capability_sources:{'M3Exam reference':{n:140,base_accuracy:.5,edited_accuracy:.5,net_loss:0}},
  control_comparisons:{random:{mean:0,ci95:[-.1,.1]}},counterexamples:Array.from({length:25},(_,i)=>({sample_id:'error-'+i,reason:'software fixture'})),
  design_sha256:'c'.repeat(64)},generations:{base:[],edited:[]},limitations:['软件测试夹具，非科研数据'],
  dose_curve:{rows:[{id:'a',k:32,alpha:1,shift:.1,flip_fraction:0,addressed_params:100,max_relative_frobenius:.01},
  {id:'b',k:32,alpha:6,shift:.2,flip_fraction:.01,addressed_params:100,max_relative_frobenius:.06}]}}}
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;host=document.createElement('div');document.body.append(host);root=createRoot(host);
  for(const k of Object.keys(state))delete state[k];state['/api/research-library/artifacts/'+sha]=fixture();window.history.replaceState(null,'','/workbench')});
afterEach(async()=>{await act(async()=>root.unmount());host.remove()});
async function render(c){await act(async()=>root.render(c))}
async function click(t){await act(async()=>[...host.querySelectorAll('button')].find(b=>b.textContent===t).click())}
it('reports uncertainty and failed target without promoting a mechanism',async()=>{
  await render(<ResearchInterventionReport sha={sha}/>);
  expect(host.textContent).toContain('能力保持尚未证实');expect(host.textContent).toContain('未通过预注册主判');
  expect(host.textContent).toContain('预算内未定位');expect(host.querySelector('svg')).toBe(null);
});
it('renders only actual dose rows after explicit navigation',async()=>{
  await render(<ResearchInterventionReport sha={sha}/>);await click('剂量曲线');
  expect(host.querySelector('svg').getAttribute('aria-label')).toContain('真实剂量');
  expect(host.querySelectorAll('circle')).toHaveLength(3);expect(host.textContent).toContain('不是能力提升曲线');
});
it('keeps counterexamples inspectable with bounded rendering',async()=>{
  await render(<ResearchInterventionReport sha={sha}/>);await click('反例（25）');
  expect(host.textContent).toContain('error-0');expect(host.textContent).not.toContain('error-24');
  await click('下一组反例');expect(host.textContent).toContain('error-24');expect(host.textContent).not.toContain('error-0');
});
it('does not assume an unrelated old report is an intervention report',async()=>{
  state['/api/research-library/artifacts/'+sha]={schema:'old_report'};
  await render(<ResearchInterventionReport sha={sha}/>);expect(host.textContent).toBe('');
});
it('opens the linked run summary and loads its archived report on demand',async()=>{
  window.history.replaceState(null,'','/workbench?run=software-fixture');
  state['/api/research-library/workpackages']={workpackages:[{sha:'b'.repeat(64),body:{run_id:'software-fixture',title:'W6 软件夹具',updated_at:1,
    stages:{registration_validated:{state:'done',time:1,report_sha256:sha}}}}]};
  await render(<ResearchExecution onGo={vi.fn()}/>);
  expect(host.querySelector('[aria-label="链接指定的实验"]').textContent).toContain('W6 软件夹具');
  expect(host.querySelector('[aria-label="真实权重编辑研究报告"]')).toBe(null);
  const details=[...host.querySelectorAll('summary')].find(el=>el.textContent==='研究报告与反例').parentElement;
  await act(async()=>{details.open=true;details.dispatchEvent(new Event('toggle'))});
  expect(host.querySelector('[aria-label="真实权重编辑研究报告"]')).not.toBe(null);
});
