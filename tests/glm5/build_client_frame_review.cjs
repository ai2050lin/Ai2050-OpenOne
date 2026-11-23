/* Build a file:// design review from the real workbench, without writing any frontend source or dist. */
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const {createHash}=require('node:crypto');
const root = path.resolve(__dirname,'../..');
const front = path.join(root,'frontend');
const requireFront = createRequire(path.join(front,'package.json'));
const esbuild = requireFront('esbuild');
const review = path.join(root,'research/glm5/docs/previews');
const distributed = process.argv.includes('--distributed');
const currentExperiments = distributed || process.argv.includes('--current-experiments');
const reviewPython=process.env.REVIEW_PYTHON||path.join(root,'.venv/Scripts/python.exe');
const database = currentExperiments || process.argv.includes('--database');
const fusion = database || process.argv.includes('--fusion');
const read = file => fs.readFileSync(file,'utf8');
const currentGraph=read(path.join(front,'src/components/v8/AssocWorkbench.jsx')).includes("from './GraphWorkspaceEvidence.jsx'");
const json = file => JSON.parse(read(file));
const html = read(path.join(review,'research-evidence-workbench.html'));
const draft = JSON.parse(html.match(/<script type="application\/json" id="evidence-data">([\s\S]+?)<\/script>/)[1]);
const script = html.match(/<script>\s*('use strict';[\s\S]+?)<\/script>/)[1];
const snapshot4 = json(path.join(root,'research/mainline/H0/report_v5.json')).model_snapshot;
const skeleton = (model, snapshot) => {
  const cfg=json(path.join(root,'models/hf',model,'config.json'));
  return {model,model_snapshot:snapshot,n_layers:cfg.num_hidden_layers,n_heads:cfg.num_attention_heads,
    d_model:cfg.hidden_size,d_ff:cfg.intermediate_size,config:cfg};
};
const models=[skeleton('Qwen3-14B',draft.cues[0].snapshot),skeleton('Qwen3-4B',snapshot4)];
const entries = {
  '/api/platform/config': {local_tools:false},
  '/api/model/skeleton': {models}
};
function archivedFiles() {
  const dir=path.join(root,'tests/glm5/result/color_reuse_v1/greedy');
  const {receipt}=json(path.join(dir,'study-receipt.json'));
  const sub=json(path.join(dir,'study-package/submission.json'));
  if(receipt.run_id!==draft.color.run_id || sub.manifest_sha256!==receipt.manifest_sha256)throw Error('Color archive identity mismatch');
  for(const item of receipt.artifacts){
    if(path.basename(item.name)!==item.name)throw Error('Expected package filename');
    const bytes=fs.readFileSync(path.join(dir,'study-package',item.name));
    if(bytes.length!==item.size || createHash('sha256').update(bytes).digest('hex')!==item.sha256)throw Error('Archive hash mismatch: '+item.name);
  }
  return {run:receipt.run_id,preflight:sub.preflight_id,manifest:receipt.manifest_sha256,files:receipt.artifacts,
    review:receipt.evidence_status,readback:receipt.readback_verified};
}
function managementRuns(){
  const files=[
    [3008,'颜色能命名，能用于判断同异吗？','color_reuse_v1/greedy/study-receipt.json',draft.color.run_id],
    [2997,'首行头写入与 128 个 FFN 单元','routine/run-t7dfn8_7/server-receipt.json',draft.path.runId]
  ];
  return files.map(([phase,title,name,run])=>{
    const saved=json(path.join(root,'tests/glm5/result',name)),receipt=saved.receipt||saved;
    if(receipt.run_id!==run||receipt.archive_state!=='server_verified'||receipt.readback_verified!==true||!Number.isFinite(receipt.verified_at))
      throw Error('Missing verified archive receipt: '+name);
    return {phase,title,model:draft.cues[0].model,run,verifiedAt:receipt.verified_at};
  }).sort((a,b)=>b.verifiedAt-a.verifiedAt);
}
function experimentSources(){
  const results='tests/glm5/result/';
  const studies=[
    [2983,'五条记录，三种取值规则','五条记录确认材料',results+'clt_b26_three_cues_v3/confirm-materials.json',['B26/H36','姓名','名字','首行','印刷标签','五条记录','查表']],
    [2990,'在新姓名家族上复核三种规则','新姓名家族确认材料',results+'clt_b26_competition_v1/confirm-materials.json',['B26/H36','新姓名','姓名','名字','首行','物理首行','查表']],
    [2997,'首行头写入与 128 个 FFN 单元','首行阻断与恢复题目',results+'routine/run-t7dfn8_7/questions.json',['B28/H21','U17216','FFN','首行','阻断','恢复']],
    [3008,'颜色能命名，能用于判断同异吗？','颜色同异题目',results+'color_reuse_v1/greedy/study-package/questions.json',['颜色','同色','同异','草莓','番茄','命名']],
    [322,'资料色与常识色的范围切换','资料色与常识色预注册方案','research/mainline/H0/MA-H0-v5.prereg.json',['颜色','资料','常识','中文','英文','冲突']]
  ];
  return studies.map(([phase,title,materialName,file,terms])=>{
    const target=path.join(root,file);
    if(!fs.statSync(target).isFile())throw Error('Missing study material '+file);
    return {phase,key:String(phase),title,materialName,terms,model:phase===322?'Qwen3-4B':'Qwen3-14B',file:path.relative(review,target).split(path.sep).join('/')};
  });
}
function studyDownloads(evidence){
  const results=path.join(root,'tests/glm5/result');
  return evidence.managementRuns.map(run=>{
    const color=run.phase===3008;
    const receiptFile=path.join(results,color?'color_reuse_v1/greedy/study-receipt.json':'routine/run-t7dfn8_7/server-receipt.json');
    const doc=json(receiptFile),receipt=doc.receipt||doc;
    const file=color?path.join(path.dirname(receiptFile),'study-package/case_results.jsonl'):require('node:url').fileURLToPath(evidence.database.causalFile);
    const bytes=fs.readFileSync(file),item=receipt.artifacts.find(a=>a.name==='case_results.jsonl');
    if(bytes.length!==item.size||createHash('sha256').update(bytes).digest('hex')!==item.sha256)throw Error('Results differ from '+receiptFile);
    const metaName=color?'study-package/reproducibility.json':'review.json';
    const snapshot=json(path.join(path.dirname(receiptFile),metaName)).model_snapshot;
    if(snapshot!==models[0].model_snapshot)throw Error('Unexpected model snapshot for phase '+run.phase);
    return {...run,snapshot,sha:item.sha256,bytes:bytes.length,manifest:receipt.manifest_sha256,review:receipt.evidence_status,lines:bytes.toString('utf8')};
  });
}
function editExperimentManagement(source){
  source="import ExperimentManager from 'preview-experiment-manager';\n"+source;
  source=replace(source,"const station = stations.some","const station = ['queue','plans'].includes(st.station) || stations.some");
  source=replace(source,"  const dsOn = ",`  const managementHome=lens==='process' && ['plan','execution'].includes(station) && !context.run && !st.mrun;
  const dsOn = `);
  const nav=source.match(/\{lens === 'process' && <SubNav items=\{stations\.filter[\s\S]*?label="研发工位".*?\/>\}/)
    || source.match(/\{lens==='process' && station!=='plan' && <nav[^>]*>[\s\S]*?<\/nav>\}/);
  if(!nav)throw Error('Cannot locate native experiment subnav');
  source=replace(source,nav[0],'');
  source=replace(source,"{lens!=='route'&&<MainlineStatus/>}","{lens!=='route'&&!managementHome&&<MainlineStatus/>}");
  source=replace(source,"{lens!=='route' && !(lens==='process'&&station==='plan')","{!managementHome && lens!=='route' && !(lens==='process'&&station==='plan')");
  const legacyClick="onClick={() => set(l.id === 'process' && lens !== 'process' ? { lens: 'process', station: 'plan' } : { lens: l.id })}";
  const currentClick=/onClick=\{\(\) => \{\s*if\(l\.id==='process'\)go\('plan',null,null,\{run:''\}\);[\s\S]*?\}\}/.exec(source)?.[0];
  source=replace(source,currentClick||legacyClick,
    "onClick={() => l.id==='process'?commit({...latestState.current,lens:'process',station:'execution',mrun:'',tm:''},{run:''}):set({lens:l.id})}");
  source=replace(source,`{lens==='process'&&stationGroup==='execution'&&station!=='execution'&&<button className="fw-tbtn" type="button" onClick={()=>go('execution')}>返回实验与队列</button>}`,
    `{lens==='process'&&!managementHome&&<button className="em-back" onClick={()=>commit({...latestState.current,station:'execution',mrun:'',tm:''},{run:''})}>← 返回实验管理</button>}`);
  source=replace(source,`{lens==='process'&&['plan','execution'].includes(station)&&<ResearchExecution plan={station==='plan'}`,
    `{managementHome&&<ExperimentManager onStation={id=>set({station:id,mrun:'',tm:''})}/>}
              {lens==='process'&&!managementHome&&['plan','execution','plans','queue'].includes(station)&&<ResearchExecution plan={['plan','plans'].includes(station)}`);
  return source;
}
function installedTheme() {
  const archive=process.env.AI2050_REVIEW_ASAR || 'D:/Program Files/AI2050-client/resources/app.asar';
  const asar=requireFront('@electron/asar');
  const names=asar.listPackage(archive).map(n=>n.replaceAll('\\','/').replace(/^\//,''));
  const theme=['main','Workbench','TestRecords'].map(name=>{
    const matches=names.filter(n=>n.startsWith('dist/assets/'+name+'-')&&n.endsWith('.css'));
    if(matches.length!==1)throw Error('Expected one installed stylesheet for '+name);
    return asar.extractFile(archive,path.normalize(matches[0])).toString('utf8');
  });
  const version=JSON.parse(asar.extractFile(archive,'package.json').toString('utf8')).version;
  return {css:theme.join('\n'),version};
}
function replace(source, from, to) {
  if(!source.includes(from)) throw Error('Review mount no longer matches client: '+String(from).slice(0,110));
  return source.replace(from,to);
}
function editRecord(source) {
  source="import ClientEvidenceReview from 'client-evidence-review';\n"+source;
  source=replace(source,'function TestRecords({model})','function TestRecords({model,onReviewLocate})');
  source=replace(source,'model={model}/>','model={model} onReviewLocate={onReviewLocate}/>');
  source=replace(source,'function Ledger({admin,model})','function Ledger({admin,model,onReviewLocate})');
  source=replace(source,"[scope,setScope]=useState('directions')","[scope,setScope]=useState('evidence')");
  source=replace(source,"  if(draft!==undefined)return",`  function openReview(detail){
    chooseScope('evidence');
    requestAnimationFrame(()=>window.dispatchEvent(new CustomEvent('review-evidence-target',{detail})));
  }
  useEffect(()=>{const reveal=e=>openReview(e.detail);window.addEventListener('review-object-evidence',reveal);
    return ()=>window.removeEventListener('review-object-evidence',reveal);},[]);
  if(draft!==undefined)return`);
  source=replace(source,'<ResearchDirections onRecord={follow}', '<ResearchDirections onReviewStudy={phase=>openReview({phase})} onRecord={follow}');
  source=replace(source,'<button onClick={()=>setDraft(null)}>新建测试</button>',
    '<div><button onClick={()=>document.querySelector(\'.aw-space\')?.scrollIntoView({block:\'start\'})}>查看上方 3D 空间</button> <button onClick={()=>setDraft(null)}>新建测试</button></div>');
  source=replace(source,'<option value="directions">研究方向树','<option value="evidence">研究证据联查</option><option value="directions">研究方向树');
  source=replace(source,"{scope!=='directions' && <label>搜索测试","{scope!=='directions' && scope!=='evidence' && <label>搜索测试");
  source=replace(source,"{scope==='directions' ? null","<div hidden={scope!=='evidence'}><ClientEvidenceReview onTree={()=>chooseScope('directions')} onLocate={onReviewLocate}/></div>\n    {scope==='directions' || scope==='evidence' ? null");
  return source;
}
function buildModule(){
  const css=read(path.join(review,'client-review/evidence.css'))+(fusion?read(path.join(review,'client-review/fusion.css')):'')+(database?read(path.join(review,'client-review/database.css')):'');
  const markup='<style>'+css+'</style><div class="views" role="group" aria-label="证据联查视图">'+
    [['compare','跨测试综合'],['catalog','研究检索'],['parts','部件与路径'],['storage','保存与复核']].map(([id,title])=>
      '<button data-view="'+id+'">'+title+'</button>').join('')+(fusion?'<button class="wide-tools" data-act="ledger">规则账本</button>':'')+'</div><span hidden id="crumb"></span><div id="main"></div>'+
    '<dialog id="detail" aria-labelledby="dialog-title"><div class="dhead split"><h2 id="dialog-title"></h2><button id="close">关闭</button></div><div id="dialog-body" class="dbody"></div></dialog>';
  let logic=replace(script, "const data=JSON.parse(document.querySelector('#evidence-data').textContent);", 'const data=payload;');
  logic=replace(logic,'function storage(){return',`function archiveCard(){
    const a=data.archive,dir=base+'color_reuse_v1/greedy/';
    return '<section class="card"><h2>Phase 3008 · 续写核查的复核包</h2><p>历史回执载明 '+a.files.length+' 份产物，'+(a.readback?'中心读回已核验':'待核验')+'。科学审阅 '+esc(a.review)+'。</p><div class="links">'+button('study','回到颜色九格与逐题输出 →',3008,'textbtn')+link(dir+'study-receipt.json','查看原回执')+'</div><details><summary>展开查重、运行与八份文件</summary>'+facts([['查重',esc(a.preflight)],['运行',esc(a.run)],['清单 SHA-256',esc(a.manifest)]])+'<div class="scroller"><table class="gridtable"><tr><th>角色</th><th>本地核验副本</th><th>字节数</th></tr>'+a.files.map(f=>'<tr><td>'+esc(f.role)+'</td><td>'+link(dir+'study-package/'+encodeURIComponent(f.name),f.name)+'</td><td>'+nf(f.size)+'</td></tr>').join('')+'</table></div></details></section>';
  }
  function storage(){return`);
  logic=replace(logic,"'阅读完整保存与客户端方案')+'</div></section>';}", "'阅读完整保存与客户端方案')+'</div></section>'+archiveCard();}");
  logic=replace(logic,"case 'tree':directionTree();break;","case 'tree':if($('#detail').open)$('#detail').close();actions.current.onTree();break;");
  logic=replace(logic,"+partContent()+'<details>", "+partContent()+(locations[state.part]?button('spatial','在上方 3D 场景定位 '+state.part,state.part,'primary'):'')+'<p role=\"status\" id=\"locate-status\"></p><details>");
  logic=replace(logic,"case 'part':state.part", "case 'spatial':{const msg=actions.current.onLocate?.({snapshot:data.cues[0].snapshot,location:locations[v]});if(msg)$('#locate-status').textContent=msg;break;}case 'part':state.part");
  logic=replace(logic,"window.scrollTo(0,0);","root.host.closest('.aw-right')?.scrollTo(0,0);");
  logic=replace(logic,"window.addEventListener('hashchange',()=>{loadState();render();});loadState();render();",`
    function openEvidence(event){const {phase,part}=event.detail||{};
      if(locations[part]){state.part=part;navigate('parts');}
      else if(rec(Number(phase))){state.lang=state.model=state.area='全部';state.query=String(phase);navigate('catalog');study(Number(phase));}
    }
    window.addEventListener('review-evidence-target',openEvidence);
    loadState();render();
    return ()=>window.removeEventListener('review-evidence-target',openEvidence);`);
  logic=logic.replaceAll('document.querySelector','root.querySelector').replaceAll('document.addEventListener','root.addEventListener');
  logic=logic.replace('AI2050 &nbsp;/&nbsp; RESEARCH ARCHIVE','');
  if(fusion){
    logic=read(path.join(review,'client-review/fusion-comparison.js'))+'\n'+logic;
    logic=replace(logic,"(list.kind!=='div'||(x.c.correct&&x.head!==x.c.gold))","(list.kind==='all'||caseOutcome(x)===list.kind)");
    logic=replace(logic,'total:classify(selectedRows())};const json','total:classify(selectedRows()),outcomes:countOutcomes(selectedRows()),outcome_labels:outcomeLabels};const json');
    logic=replace(logic,`<option value="all">全部题</option><option value="div" '+(list.kind==='div'?'selected':'')+'>头错、模型对</option>`,`'+outcomeOptions(list.kind)+'`);
    logic=replace(logic,'({compare,catalog,parts,storage})[state.view]()','({compare:wide?fusionComparison:compare,catalog,parts,storage})[state.view]()');
    logic=replace(logic,'actions.current.onTree();break;', 'if(actions.current.onTree)actions.current.onTree();else directionTree();break;');
    logic=replace(logic,"case 'spatial':","case 'ledger':window.openEvidenceWorkspace('ledger');break;case 'spatial':");
    logic=replace(logic,"'在上方 3D 场景定位 '+state.part","'在 3D 空间定位 '+state.part");
    logic=replace(logic,"root.host.closest('.aw-right')?.scrollTo(0,0);","(root.host.closest(wide?'.wb-main':'.aw-right'))?.scrollTo(0,0);");
    logic=replace(logic,"return ()=>window.removeEventListener('review-evidence-target',openEvidence);",`const restore=()=>{if(!root.host.isConnected)return;loadState();if($('#detail').open)$('#detail').close();render();};
      window.addEventListener('popstate',restore);
      return ()=>{window.removeEventListener('review-evidence-target',openEvidence);window.removeEventListener('popstate',restore);};`);
  }
  if(database){
    logic=read(path.join(review,'client-review/database.js'))+'\n'+logic;
    logic=replace(logic,"window.addEventListener('review-evidence-target',openEvidence);",
      "const dbController=wide?installDatabase():null; if(wide)root.querySelector('.views').remove();\nwindow.addEventListener('review-evidence-target',openEvidence);");
    logic=replace(logic,'function render(){Object.entries(pages)', 'function render(){if(wide){dbController.draw();return;}Object.entries(pages)');
    logic=replace(logic,"if(!root.host.isConnected)return;loadState();","if(!root.host.isConnected)return;dbController?.restore();loadState();");
  }
  // Shadow DOM isolates the established review's selectors, not the native shell or event targets.
  return `import {useRef,useEffect} from 'react';
    import nativeCues from './threeCueEvidence.json';
    import nativePath from './firstRecordPathStatus.json';
    const payload={...window.__evidenceData,cues:nativeCues.runs,path:nativePath};
    const locations={'B26/H36':{layer:27,module:'attn',index:36},'B28/H21':{layer:29,module:'attn',index:21},U17216:{layer:29,module:'mlp',index:17216}};
    export default function ClientEvidenceReview({onTree,onLocate,wide=false}){
      const host=useRef(null),actions=useRef({});actions.current={onTree,onLocate};
      useEffect(()=>{const root=host.current.attachShadow({mode:'open'});root.innerHTML=${JSON.stringify(markup)};
        const location={get hash(){const q=new URLSearchParams(globalThis.reviewRoute.search);return '#'+(q.get('evidence')||'');}};
        const history={replaceState(_s,_t,hash){const u=new URL(globalThis.reviewRoute.href);u.searchParams.set('evidence',hash.slice(1));globalThis.reviewHistory.replaceState(null,'',u);}};
        ${logic}
      },[]);return <div ref={host} className={wide?'client-evidence-host wide-evidence':'client-evidence-host'} aria-label="跨测试研究证据"/>;
    }`;
}

const overlays={
  'TestRecords.jsx':editRecord,
  'ResearchDirections.jsx':source=>{
    source=replace(source,'{ onRecord, onFamily, admin }','{ onRecord, onFamily, admin, onReviewStudy }');
    return replace(source,'{plan?.centerFamily && admin',`{({'RS.RECORD':[2983,2990,2997],'KN.ENTITY_REF':[2983,2990],'SEM.ATTRIBUTE':[3008,322]}[selected.item.id]||[]).map(phase=>
      <button key={phase} type="button" onClick={()=>onReviewStudy(phase)}>查看{phase===322?' Claude':''} Phase {phase} 同主题实测</button>)}
      {plan?.centerFamily && admin`);
  },
  'AssocWorkbench.jsx':source=>{
    source=replace(source,"[recordTab,setRecordTab]=useState('library')","[recordTab,setRecordTab]=useState('records')");
    source=replace(source,'[recordVisited,setRecordVisited]=useState(()=>new Set())',"[recordVisited,setRecordVisited]=useState(()=>new Set(['records']))");
    source=replace(source,'<TestRecords model=','<TestRecords onReviewLocate={locateArchive} model=');
    source=replace(source,'  const objectEvidence = ',`  const archivePart = activeModel?.model_snapshot===window.__reviewEntries['/api/model/skeleton'].models[0].model_snapshot ?
      ({'27/attn/36':'B26/H36','29/attn/21':'B28/H21','29/mlp/17216':'U17216'})[[object.layer,object.module,object.index].join('/')] : '';
    const objectEvidence = `);
    return replace(source,"hidden={consoleTab!=='object'}><ModelStructure",`hidden={consoleTab!=='object'}>
      {archivePart&&<div className="v6-toolbar"><button className="fw-tbtn" onClick={()=>{
        selectConsoleTab('research'); selectRecordTab('records');
        requestAnimationFrame(()=>window.dispatchEvent(new CustomEvent('review-object-evidence',{detail:{part:archivePart}}))); focusConsole();
      }}>查看 {archivePart} 的研究证据</button></div>}
      <ModelStructure`);
  },
  'Workbench.jsx':source=>replace(source,'{API_BASE || window.location.origin}','静态评审 · 离线快照'),
  'api.js':source=>{
    if(!source.includes('export function useApi(path,'))return source;
    source=replace(source,"export function getSession() { return store('get', SESSION_KEY) || ''; }","export function getSession() { return ''; }");
    source=replace(source,'export function useApi(path, { pollWhilePending',`export function useApi(path) {
      const data=window.__reviewEntries[path]??null;
      return {data,error:path && !data?'离线评审稿未收录此数据':'',loading:false,pending:false,local:true,missingPack:null,reload:()=>{}};
    }
    function unusedOnlineApi(path, { pollWhilePending`);
    return source;
  }
};
async function build(){
  if(fusion){
    const output=require('node:child_process').execFileSync(reviewPython,['-B','-X','utf8','-c',
      'import json; from server.rule_ledger_service import overview; print(json.dumps(overview(),ensure_ascii=True))'],
      {cwd:root,encoding:'utf8',maxBuffer:6*1024*1024});
    entries['/api/rules/overview']=JSON.parse(output);
    if(!entries['/api/rules/overview'].rules.length)throw Error('Missing local rule ledger');
    const applyWorkbench=overlays['Workbench.jsx'];
    overlays['Workbench.jsx']=source=>{
      source="import ClientEvidenceReview from 'client-evidence-review';\n"+applyWorkbench(source);
      const native='<RouteLens tab={rtab} openPane={st.opane} model={st.mdl || \'\'} ds={st.ds || \'\'} onGoSpace={goSpace} onGo={go} />';
      const databaseLens="{lens === 'route' && <ClientEvidenceReview wide onLocate={entry=>window.openEvidenceWorkspace('object',entry)}/>}";
      source=replace(source,'<MainlineStatus/>',database?"{lens!=='route'&&<MainlineStatus/>}":"{!(lens==='route'&&rtab==='rules')&&<MainlineStatus/>}");
      const routePanels=source.match(/\{lens === 'route' && ROUTE_VIEWS\.map[\s\S]*?^\s*\}\)\}/m)?.[0] || "{lens === 'route' && "+native+"}";
      source=replace(source,routePanels,database?databaseLens:`{lens === 'route' && (rtab==='rules' && new URLSearchParams(window.location.search).get('reviewPage')!=='ledger'
        ? <ClientEvidenceReview wide onLocate={entry=>window.openEvidenceWorkspace('object',entry)}/>
        : <>{rtab==='rules'&&<div className="v6-toolbar"><button className="fw-tbtn" onClick={()=>window.openEvidenceWorkspace('analysis')}>← 返回证据分析</button></div>}${native}</>)}
        {lens==='route'&&rtab==='rules'&&<details className="fusion-source"><summary>本页数据来源与同步状态</summary>
          <p>研究比较收录五项已开放的历史测试。规则账本取自本地登记文件。</p><MainlineStatus/></details>}`);
      source=replace(source,'writeText(window.location.href)','writeText(globalThis.location.href)');
      if(database){
        source=replace(source,'const lenses = baseLenses;',`const labels={route:'${distributed?'总览':'测试数据库'}',spatial:'语言与模型',process:'实验管理',config:'系统设置'};
          const lenses=baseLenses.map(l=>({...l,label:labels[l.id]||l.label}));`);
        const routeBar=source.split(/\r?\n/).find(line=>line.includes("{lens === 'route' && <SubNav items={ROUTE_VIEWS}"));
        if(!routeBar)throw Error('Missing native progress subnav');
        source=replace(source,routeBar,distributed?`{lens==='route'&&<OverviewTabs value={rtab} onChange={v=>{if(v!==rtab)set({rtab:v,opane:''})}}/>}`:'');
        source=replace(source,"{!(lens==='process'&&station==='plan') && (context.run","{lens!=='route' && !(lens==='process'&&station==='plan') && (context.run");
        source=editExperimentManagement(source);
        if(distributed){
          source="import {OverviewTabs,ResearchOverview} from 'preview-overview';\nimport {Participation} from 'preview-distributed-work';\n"+source;
          source=replace(source,"const rtab = ROUTE_VIEWS.some(v => v.id === st.rtab) || st.rtab === 'neuron' ? st.rtab : 'overview';","const rtab = st.rtab === 'database' ? 'database' : 'overview';");
          source=replace(source,":set({lens:l.id})}",":set(l.id==='route'?{lens:'route',rtab:'overview',opane:''}:{lens:l.id})}");
          source=replace(source,databaseLens,`{lens==='route'&&<>
            <div role="tabpanel" tabIndex={0} id="overview-panel-overview" aria-labelledby="overview-tab-overview" hidden={rtab!=='overview'}>{rtab==='overview'&&<ResearchOverview/>}</div>
            <div role="tabpanel" tabIndex={0} id="overview-panel-database" aria-labelledby="overview-tab-database" hidden={rtab!=='database'}>{rtab==='database'&&<ClientEvidenceReview wide onLocate={entry=>window.openEvidenceWorkspace('object',entry)}/>}</div>
          </>}`);
          source=replace(source,"['queue','plans'].includes(st.station)","['queue','plans','designs'].includes(st.station)");
          source=replace(source,"['plan','execution'].includes(station) && !context.run","['plan','execution','designs'].includes(station) && !context.run");
          source=replace(source,'{managementHome&&<ExperimentManager',"{managementHome&&station!=='designs'&&<Participation onStation={id=>set({station:id,mrun:'',tm:''})}/>}\n{managementHome&&station==='designs'&&<button className='em-back' onClick={()=>go('execution')}>← 返回参与首页</button>}\n{managementHome&&station==='designs'&&<ExperimentManager");
          source=replace(source,/onClick=\{\(\)=>(?:set\(\{lens:'config',csec:'node'\}\)|go\('plan',null,null,\{run:''\}\))\}>贡献算力/.exec(source)?.[0]||'Missing contribution button',
            "onClick={()=>commit({...latestState.current,lens:'process',station:'execution',mrun:'',tm:''},{run:''})}>贡献算力");
          source=replace(source,"{lens!=='route'&&!managementHome&&<MainlineStatus/>}","{!['route','spatial'].includes(lens)&&!managementHome&&<MainlineStatus/>}");
          source=replace(source,"(context.snapshot && (lens !== 'route'","(context.snapshot && lens!=='spatial' && (lens !== 'route'");
        }
        source=replace(source,'route: <><path d="M2 12h5M9 12h5M17 8l5 4-5 4V8z" /><circle cx="7" cy="12" r="2.5" /><circle cx="15" cy="12" r="2.5" /></>,','route: <><ellipse cx="12" cy="5.5" rx="8" ry="3.5"/><path d="M4 5.5v13c0 4.7 16 4.7 16 0v-13M4 12c0 4.7 16 4.7 16 0"/></>,');
      }
      return source;
    };
    const applySpatial=overlays['AssocWorkbench.jsx'];
    overlays['AssocWorkbench.jsx']=source=>{
      if(source.includes("from './GraphWorkspaceEvidence.jsx'")){
        if(!distributed)return source;
        source="import {GraphStudies} from 'preview-distributed-work';\n"+source;
        source=replace(source,'  const context = useResearchContext(), object = useSpatialObject();',`  const context = useResearchContext(), object = useSpatialObject();
  const params=new URLSearchParams(window.location.search);
  const reviewFocus=params.get('reviewFocus'),studyPack=params.get('studyPack');`);
        source=replace(source,"    const preferred = new URLSearchParams(window.location.search).get('observePlugin');",
          "    const preferred = reviewFocus==='head'?'heads':params.get('observePlugin');");
        source=replace(source,"const requested = new URLSearchParams(window.location.search).get('observeMode');",
          "const requested=reviewFocus==='object'?'features':reviewFocus==='head'?'archive':params.get('observeMode');");
        source=replace(source,"    const preferred = new URLSearchParams(window.location.search).get('observeTab');",
          "    const preferred=reviewFocus==='head'?'plugin':reviewFocus==='object'?'object':params.get('observeTab')||(studyPack?'research':'');");
        source=replace(source,"useState(Boolean(archiveView))","useState(Boolean(archiveView||studyPack))");
        source=require(path.join(review,'client-review/feature-console.cjs')).workspace(source,replace);
        source=require(path.join(review,'client-review/feature-colors.cjs'))(source,replace);
        source=require(path.join(review,'client-review/neuron-study.cjs'))(source,replace);
        return replace(source,"renderArchive?.({onLocate:locateLibraryEntry,onOverlay:setArchiveLayer})","<GraphStudies model={activeModel?.model}/>");
      }
      source=applySpatial(source);
      source=replace(source,"useState(()=>archiveView?'research':","useState(()=>new URLSearchParams(window.location.search).get('reviewFocus')==='object'?'object':archiveView?'research':");
      if(database){
        const from=source.indexOf("{archivePart&&<div");
        const end=source.indexOf('<ModelStructure',from);
        if(from<0||end<0)throw Error('Missing object evidence link');
        const withLinks=source.slice(0,from)+`{archivePart&&<div className="v6-toolbar">
          <button className="fw-tbtn" onClick={()=>window.openTestDatabase({part:archivePart})}>检索 {archivePart} 的测试</button>
          <button className="fw-tbtn" onClick={()=>window.openEvidenceWorkspace('analysis')}>返回测试数据库</button></div>}
          `+source.slice(end);
        return distributed?require(path.join(review,'client-review/graph-layout.cjs'))(withLinks,replace):withLinks;
      }
      return replace(source,'}}>查看 {archivePart} 的研究证据</button></div>}',`}}>查看 {archivePart} 的研究证据</button> <button className="fw-tbtn" onClick={()=>window.openEvidenceWorkspace('analysis')}>${database?'返回测试数据库':'返回成果与证据'}</button></div>}`);
    };
    if(database){
      const records=overlays['TestRecords.jsx'];
      overlays['TestRecords.jsx']=source=>replace(records(source),"<ClientEvidenceReview onTree={()=>chooseScope('directions')} onLocate={onReviewLocate}/>","<button className='fw-tbtn' onClick={()=>window.openTestDatabase({})}>打开测试数据库</button>");
      const directions=overlays['ResearchDirections.jsx'];
      overlays['ResearchDirections.jsx']=source=>replace(directions(source),"onClick={()=>onReviewStudy(phase)}","onClick={()=>window.openTestDatabase({q:String(phase)})}");
    }
  }
  const theme=installedTheme();
  const bundle=await esbuild.build({absWorkingDir:front,stdin:{contents:`
    import {createRoot} from 'react-dom/client';
    import Workbench from './src/components/app/Workbench.jsx';
    import './src/index.css';
    import './src/components/open/open.css';
    createRoot(document.getElementById('root')).render(<Workbench/>);
  `,resolveDir:front,loader:'jsx'},bundle:true,format:'iife',minify:true,write:false,outfile:'client-review.js',legalComments:'none',
    jsx:'automatic',target:'es2022',nodePaths:[path.join(front,'node_modules')],loader:{'.tsv':'text','.svg':'dataurl','.woff2':'dataurl','.png':'dataurl','.jpg':'dataurl'},
    define:{'import.meta.env':'{}','process.env.NODE_ENV':'"production"','window.location.href':'globalThis.reviewRoute.href',
      'window.location.search':'globalThis.reviewRoute.search','window.history':'globalThis.reviewHistory'},
    plugins:[{name:'design-review-only',setup(b){
      if(distributed){
        // Review components live outside frontend. Use its React dispatcher for every component.
        b.onResolve({filter:/^react(\/jsx(-dev)?-runtime)?$/},args=>({
          path:require.resolve(args.path,{paths:[front]})
        }));
        b.onLoad({filter:/[\\/]workbenchUtil\.js$/},a=>{
          let src=replace(read(a.path),"input.importing=q.get('data')==='import';",`if(input.lens==='route'&&(input.rtab==='rules'||(!q.has('rtab')&&['db','dbPick','dbStudy'].some(k=>q.has(k)))))input.rtab='database';
            input.importing=q.get('data')==='import';`);
          const implicitOverview="if(st.lens==='route'&&st.rtab!=='overview')q.set('rtab',st.rtab);";
          if(src.includes(implicitOverview))src=replace(src,implicitOverview,"if(st.lens==='route')q.set('rtab',st.rtab);");
          return {contents:src,loader:'js',resolveDir:path.dirname(a.path)};
        });
        b.onResolve({filter:/^preview-overview$/},()=>({path:'Overview.jsx',namespace:'distributed'}));
        b.onResolve({filter:/^preview-distributed-work$/},()=>({path:'DistributedWork.jsx',namespace:'distributed'}));
        b.onResolve({filter:/^preview-graph-head$/},()=>({path:'GraphHeadEvidence.jsx',namespace:'distributed'}));
        b.onResolve({filter:/^preview-graph-modes$/},()=>({path:'GraphModes.jsx',namespace:'distributed'}));
        b.onResolve({filter:/^preview-neuron-profiles$/},()=>({path:path.join(review,'client-review/NeuronProfiles.jsx')}));
        b.onResolve({filter:/^preview-neuron-study$/},()=>({path:path.join(review,'client-review/NeuronStudy.jsx')}));
        b.onLoad({filter:/[\\/]FullNativeField\.jsx$/},a=>({
          contents:require(path.join(review,'client-review/feature-colors.cjs')).fullNative(read(a.path),replace),
          loader:'jsx',resolveDir:path.dirname(a.path)}));
        b.onLoad({filter:/.*/,namespace:'distributed'},a=>({contents:read(path.join(review,'client-review',a.path)),loader:'jsx',resolveDir:front}));
        b.onLoad({filter:/[\\/]GraphWorkspaceEvidence\.jsx$/},a=>({
          contents:require(path.join(review,'client-review/feature-console.cjs')).evidence(read(a.path),replace),
          loader:'jsx',resolveDir:path.dirname(a.path)}));
      }
      if(database){
        b.onResolve({filter:/^preview-experiment-manager$/},()=>({path:'ExperimentManager.jsx',namespace:'experiment'}));
        b.onLoad({filter:/.*/,namespace:'experiment'},()=>({contents:read(path.join(review,'client-review',currentExperiments?'CurrentExperiments.jsx':'ExperimentManager.jsx')),loader:'jsx',resolveDir:front}));
      }
      b.onResolve({filter:/^client-evidence-review$/},()=>({path:'review',namespace:'review'}));
      b.onLoad({filter:/.*/,namespace:'review'},()=>({contents:buildModule(),loader:'jsx',resolveDir:path.join(front,'src/components/v8')}));
      b.onResolve({filter:/\?raw$/},a=>({path:path.resolve(a.resolveDir,a.path.slice(0,-4)),namespace:'text'}));
      b.onLoad({filter:/.*/,namespace:'text'},a=>({contents:read(a.path),loader:'text'}));
      b.onLoad({filter:/[\\/](TestRecords\.jsx|ResearchDirections\.jsx|AssocWorkbench\.jsx|Workbench\.jsx|api\.js)$/},a=>{
        const name=path.basename(a.path);
        if(distributed&&['TestRecords.jsx','ResearchDirections.jsx'].includes(name))return;
        return {contents:overlays[name](read(a.path)),loader:a.path.endsWith('jsx')?'jsx':'js',resolveDir:path.dirname(a.path)};
      });
    }}]});
  const asset=ext=>bundle.outputFiles.find(f=>f.path.endsWith(ext)).text;
  const {cues, path: unusedPath, ...evidence}=draft;
  evidence.archive=archivedFiles();
  if(database){
    evidence.database=require(path.join(review,'client-review/database-data.cjs'))(root,draft);
    evidence.managementRuns=managementRuns();
    if(currentExperiments)evidence.experimentSources=experimentSources();
    if(distributed)evidence.studyDownloads=studyDownloads(evidence);
  }
  let bootstrap=read(path.join(review,'client-review/offline.js'));
  if(fusion){
    bootstrap=replace(bootstrap,'wb=1&lens=spatial&view=graph&gview=internal&mdl=Qwen3-14B',currentExperiments?'wb=1&lens=process&station=execution&mdl=Qwen3-14B':'wb=1&lens=route&rtab=rules&mdl=Qwen3-14B');
    bootstrap=replace(bootstrap,"    page.hash = '';","    if(page.hash.startsWith('#view=')) page.searchParams.set('evidence',page.hash.slice(1));\n    page.hash = '';");
    bootstrap+=read(path.join(review,'client-review/fusion-navigation.js'));
    if(database)bootstrap+=`\nwindow.openTestDatabase=function(filters,phase){const url=new URL(window.reviewRoute.href);
      if(phase)url.searchParams.set('dbStudy',phase);else url.searchParams.delete('dbStudy');
      url.searchParams.set('db',new URLSearchParams(filters).toString());url.searchParams.set('lens','route');url.searchParams.set('rtab','${distributed?'database':'rules'}');
      url.searchParams.delete('reviewFocus');window.reviewHistory.pushState(null,'',url);window.dispatchEvent(new PopStateEvent('popstate'));};`;
    if(currentExperiments)bootstrap+=`\nwindow.openExperimentPlans=function(){
      const url=new URL(window.reviewRoute.href);
      for(const key of ['run','mrun','tm','reviewFocus','dbStudy'])url.searchParams.delete(key);
      url.searchParams.set('lens','process');url.searchParams.set('station','plans');
      window.reviewHistory.pushState(null,'',url);window.dispatchEvent(new PopStateEvent('popstate'));};`;
  }
  const safe=v=>JSON.stringify(v).replaceAll('<','\\u003c');
  let graphData='';
  if(distributed&&!currentGraph){
    const preview=require('node:child_process').execFileSync(reviewPython,
      ['-B','-X','utf8',path.join(__dirname,'export_graph_observation_preview.py')],{cwd:root,encoding:'utf8',maxBuffer:4*1024*1024});
    const graphObservation=JSON.parse(preview);
    if(graphObservation.snapshot!==models[0].model_snapshot)throw Error('Observation snapshot differs from the preview model');
    graphData='window.__graphObservation='+safe(graphObservation)+';';
  }
  const html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'unsafe-inline\'; style-src \'unsafe-inline\'; img-src data: blob:; font-src data:; connect-src \'none\'; form-action \'none\'; base-uri \'none\'">'+
    '<meta name="ai2050-client-theme" content="'+theme.version+'"><title>AI2050 研究工作台 · 客户端集成评审</title><style>'+
    asset('.css')+theme.css+read(path.join(front,'src/components/v8/test_records.css'))+(database?read(path.join(review,'client-review/experiment-manager.css')):'')+
    (currentExperiments?read(path.join(review,'client-review/current-experiments.css')):'')+
    (distributed?(currentGraph?['distributed-work.css','feature-console.css']:['distributed-work.css','graph-layout.css']).map(name=>read(path.join(review,'client-review',name))).join(''):'')+
    (fusion?'.fusion-source{max-width:1380px;margin:12px auto;color:#546980;font-size:12px}.fusion-source summary{cursor:pointer}.fusion-source p{margin:6px 0}':'')+
    (database?'.wb-root .fw-rail{width:84px;flex:0 0 84px}.wb-root .fw-rbtn{width:72px;min-height:52px;white-space:nowrap}.wb-main:has(>.wide-evidence){padding:0 20px 48px}@media(max-width:620px){.wb-root .fw-rail{width:64px;flex-basis:64px}.wb-root .fw-rbtn{width:60px;padding-inline:1px;line-height:1.18;font-size:11px}.wb-main:has(>.wide-evidence){padding:0 8px 36px}}':'')+'</style></head><body><div id="root"></div><script>'+
    'window.__reviewEntries='+safe(entries)+';window.__evidenceData='+safe(evidence)+';'+graphData+bootstrap+'\n</script><script>'+
    asset('.js').replace(/<\/script/gi,'<\\/script')+'</script></body></html>';
  const output=process.env.AI2050_REVIEW_OUTPUT || path.join(review,distributed?'client-distributed-testing.html':currentExperiments?'client-current-experiments.html':database?'client-test-database.html':fusion?'client-evidence-fusion.html':'client-evidence-workbench.html');fs.writeFileSync(output,html);
  console.log(output+' · '+fs.statSync(output).size+' bytes');
}
build().catch(err=>{console.error(err);process.exitCode=1;});
