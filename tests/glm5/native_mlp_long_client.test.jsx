import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import Report,{NativeNeuronCard} from '../../frontend/src/components/v8/NativeMlpReport.jsx';
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({useApi:()=>({data:{card:{layer:3,unit:9,selection:'stratified_random',replicated:false,confirm:{accuracy:.55,delta:.01,p:.4},lockbox:{accuracy:.53,delta:0,p:1},shuffle_confirm:{accuracy:.5},shuffle_lockbox:{accuracy:.49},poles:{'2:Gpos':{coherence:.2,examples:[]}},counterexamples:[],all_error_count:0}}})}));
let host,root;
const fixture={run_id:'software-fixture',model_snapshot:'a'.repeat(64),analysis:{random_replicated:2,random_shuffle_double_pass:0},cards:[{layer:3,unit:9,sha256:'a'}],artifact_refs:{documents:[]},dose:{supported:false,effects:{}},selection:{units:[[3,9]]},capability:{n:2200,status:'noninferior',net_loss:0,loss_interval:[-.01,.01]},examples:[{trace_id:'actual-native',n_tokens:900,pair_index:0,side:'a',url:'?trace=actual-native&tunit=9'}],limitations:['边界记录']};
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;host=document.createElement('div');document.body.append(host);root=createRoot(host)});
afterEach(async()=>{await act(async()=>root.unmount());host.remove()});
async function open(){await act(async()=>root.render(<Report doc={fixture}/>))}
async function click(t){await act(async()=>[...host.querySelectorAll('button')].find(x=>x.textContent===t).click())}
it('states operational holdout prediction rather than a solved-neuron percentage',async()=>{await open();expect(host.textContent).toContain('2/432');expect(host.textContent).toContain('尚非单元完整语义');expect(host.textContent).toContain('未全体解释')});
it('keeps zero passed units a negative result rather than describing a discovered rule',async()=>{await act(async()=>root.render(<Report doc={{...fixture,analysis:{random_replicated:0,random_shuffle_double_pass:0}}}/>));expect(host.textContent).toContain('当前前缀字符类别预测器未通过双留出门');expect(host.textContent).not.toContain('这表示一种有限的上下文规律')});
it('keeps capability success separate from a negative intervention',async()=>{await open();await click('剂量与能力');expect(host.textContent).toContain('未通过冻结效应门');expect(host.textContent).toContain('通过非劣效门');expect(host.textContent).toContain('140 道参考题不入门')});
it('opens the actual trace URL and native unit locator',async()=>{await open();await click('3D计算路径');expect(host.querySelector('a').getAttribute('href')).toContain('trace=actual-native&tunit=9')});
it('renders a failed card and unlabeled semantics without inventing a feature name',async()=>{await act(async()=>root.render(<NativeNeuronCard sha="a" documentRefs={[]}/>));expect(host.textContent).toContain('未通过两次留出响应预测');expect(host.textContent).toContain('尚未得到独立语义标签');expect(host.textContent).toContain('55.0%')});
