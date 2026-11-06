import {beforeEach,afterEach,it,expect} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import Report from '../../frontend/src/components/v8/AtomicArbitrationReport.jsx';
let host,root;
const fixture={run_id:'fixture',n_inputs:704,native:{files:100,bytes:1},analysis:{behavior_gate:true,H2_supported:false,cell_statistics:[],dose_curve:[],output_prediction:{supported:true,graph_KL:.2,scope_blind_KL:1,relative_reduction:.8,top1_agreement:1},internal_prediction:[]},capability:{n:2200,net_loss:0,loss_interval:[-.01,.01],status:'noninferior'},calibration_versions:[],counterexamples:[{sample_id:'E:test',reason:'regression',trace_id:'native-trace'}],examples:[{entity:'Oxygen',scope:'现实',length:1024,trace_id:'native-trace',url:'?trace=native-trace'}],mlp_plan:[{title:'原生单元',detail:'两极与近似反例'}],limitations:[],preregistration:{},freeze:{}};
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;host=document.createElement('div');document.body.append(host);root=createRoot(host)});
afterEach(async()=>{await act(async()=>root.unmount());host.remove()});
async function open(){await act(async()=>root.render(<Report doc={fixture}/>))}
async function click(t){await act(async()=>[...host.querySelectorAll('button')].find(x=>x.textContent===t).click())}
it('does not call reference-assisted source arbitration closed-book memory',async()=>{await open();expect(host.textContent).toContain('无参考现实问答的校准失败另保留');expect(host.textContent).toContain('MLP组效应未达预注册标准')});
it('distinguishes next-token prediction from complete answer prediction',async()=>{await open();await click('未见组合预测');expect(host.textContent).toContain('不预测完整整数');expect(host.textContent).toContain('不宣称图谱对新实体知识泛化')});
it('retains links from counterexamples to the actual native trace',async()=>{await open();await click('反例（1）');expect(host.querySelector('a').getAttribute('href')).toContain('trace=native-trace')});
it('keeps MLP-first research conditional rather than claiming all neurons solved',async()=>{await open();await click('MLP主线方案');expect(host.textContent).toContain('不能保证先把全部MLP破解');expect(host.textContent).toContain('尚不是全部350,208单元')});
