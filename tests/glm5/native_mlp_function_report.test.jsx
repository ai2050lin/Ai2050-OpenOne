import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import Report,{caseReadout} from '../../frontend/src/components/v8/NativeFunctionReport.jsx';
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({useApi:()=>({data:{cases:Array.from({length:31},(_,i)=>({trace_id:'fixture-'+i,sample_id:'polarity:'+i,run_id:'fixture',layer:24,unit:8443}))}})}));
let host,root;
const fixture={run_id:'software-fixture',model_snapshot:'a'.repeat(64),analysis:{supported_families:1,n_inputs:3264,findings:[],negative_control_early_pass:true},coordinate_candidates:[],calibrations:[],dose:{supported:false,effects:{},specificity:{}},capability:{status:'noninferior'},capability_groups:{native:{status:'noninferior'},coordinate:{status:'noninferior'}},polarity_controls:{name_balanced_place:{behavior_gate:true,findings:[{layer:24,unit:8443,stage:1,interpretation:'not_disambiguated'}]}},case_index_sha256:'fixture',counterexamples:[],examples:[],limitations:[]};
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;host=document.createElement('div');document.body.append(host);root=createRoot(host)});
afterEach(async()=>{await act(async()=>root.unmount());host.remove()});
async function open(){await act(async()=>root.render(<Report doc={fixture}/>))}
async function click(name){await act(async()=>[...host.querySelectorAll('button')].find(x=>x.textContent===name).click())}
it('revises fixed-property claims in the visible conclusion',async()=>{await open();expect(host.textContent).toContain('主要跟随Yes/No答案方向');expect(host.textContent).toContain('解释被后续反例收窄')});
it('keeps unresolved assistant readout separate from the question-end result',async()=>{await open();await click('问法与名称反例');expect(host.textContent).toContain('assistant边界');expect(host.textContent).toContain('尚不能区分')});
it('makes every archived case reachable across pagination',async()=>{await open();await click('全部原生案例');expect(host.querySelectorAll('a')).toHaveLength(30);await click('下一页');expect(host.querySelector('a').getAttribute('href')).toContain('trace=fixture-30');expect(host.textContent).toContain('2/2')});
it('shows both capability operations without turning a negative dose into a successful switch',async()=>{await open();await click('剂量与能力');expect(host.textContent).toContain('未达到效应门');expect(host.textContent).toContain('通过非劣效门');expect(host.textContent).toContain('coordinate');expect(host.textContent).toContain('native')});
it('links measured question and assistant readouts separately',()=>{expect(caseReadout({readout_annotations:[{role:'question_end',position:1023},{role:'assistant_boundary',position:1033}]})).toBe(1023);expect(caseReadout({readout_annotations:[{role:'assistant_boundary',position:136}]})).toBe(136)});
