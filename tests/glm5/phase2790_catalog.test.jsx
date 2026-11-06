import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import ResearchCatalog from '../../frontend/src/components/v8/ResearchCatalog.jsx';
import ExternalAiRegisterReport from '../../frontend/src/components/v8/ExternalAiRegisterReport.jsx';
import DatasetAtlasPlan from '../../frontend/src/components/v8/DatasetAtlasPlan.jsx';
import {responseCompatible} from '../../frontend/src/components/v8/researchFragmentUtil.js';
const state=vi.hoisted(()=>({}));
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({useApi:path=>({data:state[path],reload:vi.fn()})}));
vi.mock('../../frontend/src/components/open/DataBar.jsx',()=>({ExtraPackPrompt:()=>null}));
let host,root;
function fixture(){return {note:'软件夹具；来源和数量不等于机制证据',datasets:[
  {id:'d-control',body:{dataset_id:'受控材料',role:'controlled_mechanism',sample_count:4,source_groups:['f1'],label_status:'条件标签'}},
  {id:'d-monitor',body:{dataset_id:'参考题库',role:'capability_monitor',sample_count:2,source_groups:['q1','q2'],label_status:'unverified',formal_gate_eligible:false}}],
  claims:[{id:'c1',body:{canonical_feature_id:'op1',external_pattern_id:'language:style',outcome_metric:'candidate preference',status:'supported_in_scope',scope:'固定候选',dataset_versions:['d-control'],model_snapshot:'a'.repeat(64),artifact_sha256:'b'.repeat(64),run_id:'r1'}},
    {id:'c2',body:{canonical_feature_id:'op1',external_pattern_id:'language:style',outcome_metric:'free generation',status:'inconclusive',scope:'自然输出',dataset_versions:['d-control'],model_snapshot:'a'.repeat(64),artifact_sha256:'c'.repeat(64),run_id:'r2'}}],
  aliases:[{body:{canonical_feature_id:'op1',run_id:'r1',model_snapshot:'a'.repeat(64),artifact_sha256:'d'.repeat(64),legacy_feature_id:'old:B18'}}],
  reviews:[{id:'ai-fixture',source_type:'external_ai',items:10,rater_models:['model-one','model-two'],completed_raters:1,disagreements:null,status:'awaiting_ai_reviews'}]}}
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;host=document.createElement('div');document.body.append(host);root=createRoot(host);state['/api/research-library/catalog']=fixture()});
afterEach(async()=>{await act(async()=>root.unmount());host.remove()});
async function render(props={}){await act(async()=>root.render(<ResearchCatalog {...props}/>))}
async function click(text){await act(async()=>[...host.querySelectorAll('button')].find(b=>b.textContent===text).click())}
async function expand(text){
  const details=[...host.querySelectorAll('summary')].find(el=>el.textContent.startsWith(text)).parentElement;
  await act(async()=>{details.open=true;details.dispatchEvent(new Event('toggle'))});
}
it('lists materials while keeping claims and source details collapsed',async()=>{
  await render();
  expect(host.textContent).toContain('受控材料');
  expect(host.textContent).not.toContain('条件标签');
  expect(host.textContent).not.toContain('candidate preference');
});
it('keeps scoped positive and inconclusive outcomes side by side',async()=>{
  await render();
  await expand('受控材料');
  await expand('关联主张与实验');
  expect(host.textContent).toContain('范围内支持');expect(host.textContent).toContain('尚未确定');
  expect(host.textContent).toContain('candidate preference');expect(host.textContent).toContain('free generation');
  await expand('candidate preference');
  const link=host.querySelector('a');expect(link.href).toContain('artifact='+'d'.repeat(64));expect(link.href).toContain('feature=old%3AB18');
});
it('keeps the evidence absence explicit for a reference-only dataset',async()=>{
  await render();
  await expand('参考题库');
  expect(host.textContent).toContain('仅供参考，未纳入正式能力门');
  expect(host.textContent).toContain('此材料尚无已登记的机制主张');
  expect(host.textContent).not.toContain('candidate preference');
});
it('AI opinions cannot look like completed human or mechanism validation',async()=>{
  await render();
  await expand('独立评审批次');
  await expand('ai-fixture');
  expect(host.textContent).toContain('等待 AI 评分');
  expect(host.textContent).toContain('1/2 份评分');
  expect(host.textContent).toContain('AI 评分与人工标注分别记录');
});

function report(){return {run_id:'software-fixture',baseline_gate:false,n_inputs:96,n_outputs:384,n_families:8,stats:[],effects:[],
  capability:{result:{status:'noninferior',n:2200,net_loss:0,loss_interval:[-.01,.01],margin:.02}},counterexamples:[{
    sample_id:'fixture-input',condition:'base',text:'这是软件测试文本。',rater_disagreement:true,ratings:{one:{style:'formal',meaning:'changed',naturalness:4,reason:'负责人发生改变。'}}}],
  rater_models:['model-one','model-two'],unique_blind_records:1,unique_style_agreement:1,unique_meaning_agreement:0,semantic_controls:[],limitations:['软件夹具不作为研究。'],
  native_summary:{verified_files:1,bytes:100,valid_generation_steps:1},traces:[{sample_id:'fixture-input',condition:'base',hidden_rms:[Array(37).fill(.2)],generated_token_ids:[7]}]}}
it('external AI report keeps failed gates and counterexample reasons visible',async()=>{
  await act(async()=>root.render(<ExternalAiRegisterReport doc={report()}/>));expect(host.textContent).toContain('探索行为门未通过');
  await click('反例与分歧（1）');expect(host.textContent).toContain('负责人发生改变');
});
it('generation field renders every hidden layer with actual token metadata',async()=>{
  await act(async()=>root.render(<ExternalAiRegisterReport doc={report()}/>));await click('真实生成轨迹');
  expect(host.querySelectorAll('svg rect').length).toBe(37);expect(host.querySelector('title').textContent).toContain('预测token ID 7');
});
it('native step arrays must match the card generation boundary and full model identity',()=>{
  const card={run_id:'fixture',layer:18,boundary:'mlp.down_proj.input',index:{unit:2},scope:{readout:'generation'}};
  const response={schema:'research_mlp_response.v1',kind:'real',model_snapshot:'a'.repeat(64),run_id:'fixture',selected_unit:2,
    block:18,boundary:'mlp.down_proj.input',readout:'generation',values:[1,2,3]};
  expect(responseCompatible(response,card,'a'.repeat(64),{intermediate_size:3})).toBe(true);
  expect(responseCompatible({...response,readout:'final'},card,'a'.repeat(64),{intermediate_size:3})).toBe(false);
  expect(responseCompatible({...response,values:[1,2]},card,'a'.repeat(64),{intermediate_size:3})).toBe(false);
});
it('partial plan distinguishes implemented collection from pending mechanism discovery',async()=>{
  const sha='f'.repeat(64);state['/api/research-library/artifacts/'+sha]={schema:'research_dataset_atlas_plan.v1',implementation_status:'partial'};
  await act(async()=>root.render(<DatasetAtlasPlan sha={sha} implementation/>));expect(host.textContent).toContain('部分已实施');
  await act(async()=>{host.querySelector('details').open=true;host.querySelector('details').dispatchEvent(new Event('toggle'))});
  expect(host.textContent).toContain('字典发现、第二物理节点复算和图谱预测仍待完成');
});
