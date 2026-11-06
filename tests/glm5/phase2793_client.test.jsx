import {beforeEach,afterEach,it,expect} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import Report from '../../frontend/src/components/v8/EntityArbitrationReport.jsx';
let host,root;
const fixture={run_id:'fixture',behavior_gate:false,H2_supported:false,n_inputs:576,heldout_entities:32,native:{files:21616,bytes:100,generation_steps:8},scope_stats:[{scope:'文档',n:192,accuracy:1},{scope:'现实',n:192,accuracy:.3177}],source_free_prior_accuracy:.625,H1:{mean:.4167,ci95:[.28,.55]},head_tests:[],cells:[],counterexamples_count:21,counterexamples:Array.from({length:21},(_,i)=>({entity:'entity'+i,reason:'failure',trace_id:'trace'+i})),trace_preview:[{trace_id:'actual-trace',entity:'糖',scope:'文档',n_tokens:512}],generations:[],limitations:['受控材料，不等于自然语料。'],references:{},preregistration:{}};
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;host=document.createElement('div');document.body.append(host);root=createRoot(host)});
afterEach(async()=>{await act(async()=>root.unmount());host.remove()});
async function click(text){await act(async()=>[...host.querySelectorAll('button')].find(b=>b.textContent===text).click())}
it('keeps failed hypotheses and untested capability explicit',async()=>{await act(async()=>root.render(<Report doc={fixture}/>));expect(host.textContent).toContain('范围行为门未通过');expect(host.textContent).toContain('Head组效应未支持');expect(host.textContent).toContain('不能声称已通过');expect(host.textContent).toContain('没有把它包装成剂量曲线')});
it('links real trace IDs and distinguishes registered positions from all tokens',async()=>{await act(async()=>root.render(<Report doc={fixture}/>));await click('三模块真实路径');expect(host.querySelector('a').getAttribute('href')).toContain('trace=actual-trace');expect(host.textContent).toContain('三个读出位置')});
it('retains counterexamples beyond the first page',async()=>{await act(async()=>root.render(<Report doc={fixture}/>));await click('反例与失败（21）');expect(host.textContent).toContain('entity0');await click('下一页');expect(host.textContent).toContain('entity20');expect(host.querySelector('a').getAttribute('href')).toContain('trace20')});
