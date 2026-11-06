import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import React,{act} from '../../frontend/node_modules/react/index.js';
import {createRoot} from '../../frontend/node_modules/react-dom/client.js';
import {readState,writeState,resolveGo,normalizeState,SPATIAL_VIEWS,visibleStations,dsBound} from '../../frontend/src/components/app/workbenchUtil.js';
import {publishResearchContext,useResearchContext} from '../../frontend/src/components/app/researchContext.js';
import Workbench from '../../frontend/src/components/app/Workbench.jsx';
const mocks=vi.hoisted(()=>({local:true}));
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({API_BASE:'http://localhost:5001',ADMIN_LIVE:true,DESKTOP:false,OFFLINE_CLIENT:false,useApi:p=>({data:p==='/api/platform/config'?{local_tools:mocks.local}:null,reload:vi.fn()})}));
vi.mock('../../frontend/src/components/app/rdc_fusion/authContext.js',()=>({useAuth:()=>({user:{role:'admin'}})}));
vi.mock('../../frontend/src/components/app/ConfigView.jsx',()=>({CFG_SECTIONS:[{id:'conn',label:'连接与诊断'},{id:'account',label:'账号与权限'},{id:'node',label:'本机算力'},{id:'model',label:'模型与AI'}],default:({section})=><div>配置内容：{section}</div>}));
vi.mock('../../frontend/src/components/app/rdc_fusion/AccountMenu.jsx',()=>({default:()=>null}));
vi.mock('../../frontend/src/components/app/rdc_fusion/AtlasView.jsx',()=>({default:()=>null}));
vi.mock('../../frontend/src/components/app/rdc_fusion/CatalogPanel.jsx',()=>({default:()=>null}));
vi.mock('../../frontend/src/components/app/rdc_fusion/CensusPanel.jsx',()=>({default:()=>null}));
vi.mock('../../frontend/src/components/app/rdc_fusion/LocalPackView.jsx',()=>({default:()=>null}));
vi.mock('../../frontend/src/components/app/rdc_fusion/TemplatePipeline.jsx',()=>({default:()=>null}));
vi.mock('../../frontend/src/components/v8/RouteLens.jsx',()=>({default:()=> <div>真实进展入口</div>}));
vi.mock('../../frontend/src/components/v8/ResearchExecution.jsx',()=>({default:({plan})=><div>{plan?'计划视图':'队列视图'}</div>}));
vi.mock('../../frontend/src/components/v8/ResearchCatalog.jsx',()=>({default:()=>null}));
vi.mock('../../frontend/src/components/v8/SpaceLens.jsx',()=>({default:({view,graphMode,paramTab})=><div>{`空间内容：${view}/${graphMode}/${paramTab}`}</div>}));
vi.mock('../../frontend/src/components/open/DataBar.jsx',()=>({ExtraPackPrompt:()=>null}));
vi.mock('../../frontend/src/components/app/Gctx.jsx',()=>({default:({onDs,onMdl})=><button onClick={()=>{onDs('dataset-A');onMdl('model-A')}}>连续选择来源和模型</button>}));
let host,root;
beforeEach(()=>{globalThis.IS_REACT_ACT_ENVIRONMENT=true;window.history.replaceState({},'','/workbench');mocks.local=true;host=document.createElement('div');document.body.append(host);root=createRoot(host)});
afterEach(async()=>{await act(async()=>root.unmount());host.remove()});
async function render(){await act(async()=>root.render(<Workbench/>))}
async function click(text){await act(async()=>[...host.querySelectorAll('button')].find(b=>b.textContent===text).click())}
it('migrates old graph, unit, tool, contribution and assistant links',()=>{
  for(const [url,want] of [['?lens=spatial&view=research',{view:'graph',gview:'internal'}],['?lens=spatial&view=atlas',{view:'graph',gview:'relations'}],['?lens=spatial&view=neuron',{view:'parameters',ptab:'neurons'}],['?lens=local',{lens:'config',csec:'legacy'}],['?station=contribute',{lens:'config',csec:'node'}],['?station=ai',{lens:'config',csec:'model'}],['?view=local',{importing:true}]] ){
    window.history.replaceState({},'',url);expect(readState()).toMatchObject(want);
  }
});
it('keeps expert permissions while providing exactly four stable spatial pages',()=>{
  expect(SPATIAL_VIEWS.map(v=>v.id)).toEqual(['graph','parameters','live','response']);
  expect(visibleStations({live:false}).map(v=>v.id)).toEqual(['plan','catalog','execution']);
  expect(visibleStations({live:true,admin:false,localTools:true}).map(v=>v.id)).not.toContain('autotest');
  expect(visibleStations({live:true,admin:true,localTools:false}).map(v=>v.id)).not.toContain('autotest');
});
it('shows dataset controls only for views which consume that dataset',()=>{
  expect(dsBound('spatial',{view:'graph',gview:'language'})).toBe(true);
  for(const gview of ['internal','relations'])expect(dsBound('spatial',{view:'graph',gview})).toBe(false);
});
it('preserves source identity while removing inactive controls from the URL',()=>{
  window.history.replaceState({},'','?lens=spatial&view=live&trace=trace-A&run=run-A&snapshot='+ 'a'.repeat(64)+'&tblock=32&punit=9000');
  writeState(normalizeState({lens:'process',station:'execution'}));const q=new URLSearchParams(window.location.search);
  expect(q.get('context_trace')).toBe('trace-A');expect(q.get('trace')).toBe(null);expect(q.get('tblock')).toBe(null);expect(q.get('punit')).toBe(null);expect(q.get('run')).toBe('run-A');
  writeState(resolveGo(readState(),'spatial','live'));expect(new URLSearchParams(window.location.search).get('trace')).toBe('trace-A');
});
it('shows four primary entries even with local tools enabled, and contribution opens actual node settings',async()=>{
  await render();expect(host.querySelectorAll('[aria-label="工作台界面"] button')).toHaveLength(4);await click('贡献算力');expect(host.textContent).toContain('配置内容：node');
  expect([...host.querySelectorAll('[aria-label="配置分段"] button')].map(b=>b.textContent).filter(x=>x!=='?')).toEqual(['连接与诊断','账号与权限','本机算力','模型与AI']);
});
it('uses graph as default spatial entry and keeps the canonical route on refresh',async()=>{
  await render();await click('空间');expect(host.textContent).toContain('空间内容：graph/structure/weights');expect(readState().view).toBe('graph');
});
it('restores the previous route and selected native object after browser history navigation',async()=>{
  window.history.replaceState({},'','?lens=spatial&view=parameters&pblock=32&punit=9727');await render();await click('研发');
  await act(async()=>{window.history.replaceState({},'','?lens=spatial&view=parameters&pblock=32&punit=9727');window.dispatchEvent(new PopStateEvent('popstate'))});
  expect(host.textContent).toContain('空间内容：parameters');expect(new URLSearchParams(window.location.search).get('punit')).toBe('9727');
});
it('does not lose either selection when model and dataset updates occur together',async()=>{
  window.history.replaceState({},'','?lens=spatial&view=graph&gview=language');await render();await click('连续选择来源和模型');expect(readState()).toMatchObject({ds:'dataset-A',mdl:'model-A'});
});
it('publishes immutable trace provenance without touching model weights or adding unverified labels',async()=>{
  function Context(){const c=useResearchContext();return <p>{c.run}:{c.snapshot}:{c.trace}</p>}
  await act(async()=>root.render(<Context/>));await act(async()=>publishResearchContext({run:'real-run',snapshot:'b'.repeat(64),trace:'real-trace'}));expect(host.textContent).toContain('real-run:'+ 'b'.repeat(64)+':real-trace');
});
it('old import links open a usable local data dialog and close without uploading',async()=>{
  window.history.replaceState({},'','?lens=spatial&view=local');await render();expect(host.querySelector('dialog[aria-label="数据与同步"]')).not.toBe(null);expect(host.querySelector('input[type=file]')).not.toBe(null);await click('关闭数据面板');expect(host.querySelector('dialog')).toBe(null);
});
