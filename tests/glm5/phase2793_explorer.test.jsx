import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import Explorer from '../../frontend/src/components/v8/MechanismTraceExplorer.jsx';
import Report from '../../frontend/src/components/v8/EntityArbitrationReport.jsx';
vi.mock('../../frontend/src/components/v8/MechanismScene.jsx',()=>({default:({head,unit})=><div>scene H{head} U{unit}</div>,NativeArray:({values,label})=><div>{label}: {values?.length||0}</div>}));
let host,root;
const manifest={trace_id:'prefix',run_id:'fixture',entity:'实体',scope:'现实',split:'heldout',n_tokens:512,n_layers:36,n_heads:32,n_kv_heads:8,d_ff:9728,d_model:2560,model_snapshot:'a'.repeat(64),readout_positions:[5,511],tokens:Array(513).fill('词'),token_ids:Array(513).fill(2),output_top:[],files:[]};
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;window.history.replaceState({},'','/?trace=prefix&tblock=32&thead=11&tunit=190');host=document.createElement('div');document.body.append(host);root=createRoot(host);
  vi.stubGlobal('fetch',vi.fn(async url=>{const p=new URL(url,'http://localhost').pathname;let data;
    if(p==='/api/traces/catalog')data=url.includes('parent=')?{traces:[{trace_id:'step',generation_step:1,n_tokens:513}]}:{traces:[{trace_id:'prefix',entity:'实体',scope:'现实',deep:true,n_tokens:512}]};
    else if(p.endsWith('/manifest'))data={...manifest,...(p.includes('/step/')?{trace_id:'step',readout_positions:[512],n_tokens:513}:{})};
    else if(p.endsWith('/field'))data={values:Array(url.includes('mlp_')?9728:2560).fill(0.25)};
    else if(p.endsWith('/attention-row'))data={weights:Array(512).fill(1/512),kv_group:2,validation:0.002};
    else if(p==='/api/live/status')data={status:'idle'};
    else if(p==='/api/traces/context')return {ok:false,status:404,json:async()=>({detail:'未注册研究'})};
    return {ok:true,json:async()=>data};}));});
afterEach(async()=>{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals()});
const flush=()=>act(async()=>{await new Promise(r=>setTimeout(r,8))});
async function open(){await act(async()=>root.render(<Explorer/>));await flush();await flush()}
async function button(text){await act(async()=>[...host.querySelectorAll('button')].find(b=>b.textContent===text).click());await flush()}
it('restores exact block/head/unit locators from a research deep link',async()=>{await open();expect(host.querySelector('[aria-label="轨迹块"]').value).toBe('32');expect(host.textContent).toContain('scene H11 U190')});
it('shows all MLP units, keeping native gate/value/activation queries separate',async()=>{await open();await button('完整MLP阵列');expect(host.textContent).toContain('完整MLP中间单元阵列: 9728');const urls=fetch.mock.calls.map(c=>c[0]);expect(urls.some(u=>u.includes('boundary=mlp_gate'))).toBe(true);expect(urls.some(u=>u.includes('boundary=mlp_value'))).toBe(true)});
it('switches to actual cached generation positions while preserving the selected components',async()=>{await open();await act(async()=>{const el=host.querySelector('[aria-label="实际生成步"]');el.value='step';el.dispatchEvent(new Event('change',{bubbles:true}))});await flush();await flush();expect(host.textContent).toContain('scene H11 U190');expect(host.querySelector('[aria-label="真实query位置"]').value).toBe('512');expect(fetch.mock.calls.some(c=>c[0].includes('/step/attention-row?block=32&head=11&query=512'))).toBe(true)});
it('does not invent research labels for unregistered live traces',async()=>{await open();await button('特性与证据');expect(host.textContent).toContain('实时未注册观察不自动产生研究结论')});
it('renders the supplementary capability monitor without upgrading failed arbitration',async()=>{const doc={...manifest,n_inputs:576,n_heldout_entities:32,scope_accuracy:{文档:1,现实:.3177},scope_effect:{mean:.4167,ci95:[.28,.55]},head_ablation:{},native_audit:{native_files_verified:21320,native_bytes:1},examples:[],cell_statistics:[],counterexamples:[],limitations:[],reference_quality:{both_typical:10,disagreements:3},capability:{n:2200,status:'noninferior',net_loss:.00227,loss_interval:[-.005,.00952]}};await act(async()=>root.render(<Report doc={doc}/>));expect(host.textContent).toContain('范围行为门未通过');expect(host.textContent).toContain('通过非劣效门');expect(host.textContent).toContain('不能挽救未支持的H1/H2')});
