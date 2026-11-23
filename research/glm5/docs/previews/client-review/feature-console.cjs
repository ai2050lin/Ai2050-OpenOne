// Apply the review layout to the same React components used by the desktop workbench.
function workspace(source, replace) {
  let result = source.replaceAll('\r\n', '\n');
  const edit = (before, after) => { result = replace(result, before, after); };

  // Keep the input, replay and its readout mounted while a lower instrument changes.
  edit('  const panel = !cur ?', '  const renderPanel = cur => !cur ?');
  edit('  const popBody = visiblePop', '  const panel = renderPanel(cur);\n  const popBody = visiblePop');
  edit('  const [inputMode,setInputMode]', "  const [liveResultView,setLiveResultView]=useState('result');\n  const [inputMode,setInputMode]");
  edit("const showRun = (r, input, condition) => {", "const showRun = (r, input, condition) => {\n    setLiveResultView('result');");
  edit('className="ow-console-layout"', 'className={observeMode===\'live\'?\'ow-console-layout ow-live-stack\':\'ow-console-layout\'}');
  edit('<div className="ow-console-nav">', "{observeMode!=='live'&&<div className=\"ow-console-nav\">");
  edit('        </div>\n        <div className={observeMode', '        </div>}\n        <div className={observeMode');
  edit("{consoleTab==='plugin'&&ops}", "{observeMode!=='live'&&consoleTab==='plugin'&&ops}");
  for(const id of ['run','plugin']){
    edit('id="console-panel-'+id+'" role="tabpanel" aria-labelledby="console-tab-'+id+'"',
      'id="console-panel-'+id+'" role={observeMode===\'live\'?undefined:\'tabpanel\'} aria-labelledby={observeMode===\'live\'?undefined:\'console-tab-'+id+'\'}');
  }
  edit('      <button className="fw-tbtn ow-plugin-manager"', `      {observeMode==='live'&&consoleTab==='plugin'&&<button type="button" className="fw-tbtn ow-collapse-tool"
        aria-controls="console-panel-plugin" onClick={()=>selectConsoleTab('run')}>收起结果</button>}
      <button className="fw-tbtn ow-plugin-manager"`);
  edit("hidden={consoleTab!=='run'}>", "hidden={observeMode!=='live'&&consoleTab!=='run'}>");
  edit("{consoleTab==='run'&&observeMode==='live'&&<section", "{observeMode==='live'&&<section");
  edit("aria-pressed={runTab==='result'} onClick={()=>setTab('result')}", "aria-pressed={liveResultView==='result'} onClick={()=>setLiveResultView('result')}");
  edit("aria-pressed={runTab!=='result'} onClick={()=>setTab('live')}", "aria-pressed={liveResultView!=='result'} onClick={()=>setLiveResultView('live')}");
  edit('</span></div>{panel}\n            </>}', '</span></div>{renderPanel(liveResultView)}\n            </>}');
  edit('        <div id="console-panel-plugin"', '        {observeMode===\'live\'&&ops}\n        <div id="console-panel-plugin"');
  edit('提交文本进行一次本机前向，并在 3D 中查看采集结果。', '选择归档样例查看实测响应，也可以编辑输入草稿。此预览离线运行。');

  // Use the same zero-based block and unit that the spatial picker has selected.
  edit("const [featureKind, setFeatureKind] = useState('unit');",
    "const [featureKind, setFeatureKind] = useState(()=>new URLSearchParams(window.location.search).get('featureView')==='goodfire'?'goodfire':'unit');");
  edit("  const chooseDisplay = id =>", `  const selectFeatureView = value => {
    setFeatureKind(value);
    saveView({featureView:value==='goodfire'?'goodfire':null});
  };
  const openProfile = entry => {
    const target=models.find(m=>m.model_snapshot===entry.snapshot && shortModel(m.model)===shortModel(entry.model));
    if(!target)return;
    if(activeModel?.model_snapshot!==target.model_snapshot)selectModel(target);
    locate({layer:entry.block+1,module:'mlp',index:entry.index});
    selectFeatureView('goodfire');
    saveView({observeMode:'features',observeTab:'object',observePlugin:'object'});
  };
  const chooseDisplay = id =>`);
  for (const before of ["if(h.o==='mlp')setFeatureKind('unit')", "if (location.module === 'mlp') setFeatureKind('unit')",
                        "if(observeMode==='features'&&findKind==='mlp')setFeatureKind('unit')"]) {
    edit(before, before.replace("setFeatureKind('unit')", "setFeatureKind(k=>k==='goodfire'?k:'unit')"));
  }
  edit('            <NeuronExample sample={matchingExample} kind={featureKind} onKind={setFeatureKind}',
    `            <NeuronExample sample={matchingExample} kind={featureKind} onKind={selectFeatureView}
              model={activeModel ? shortModel(activeModel.model) : ''} snapshot={snapshot}
              block={layer-1} selectedUnit={sel.unit} onProfile={openProfile}`);
  edit('            <ModelStructure key={snapshot', '            <div hidden={featureKind===\'goodfire\'}><ModelStructure key={snapshot');
  edit('responseContent={<p className="v8-note">{reason}</p>}/>', 'responseContent={<p className="v8-note">{reason}</p>}/></div>');
  return result;
}

function evidence(source, replace) {
  let result = "import NeuronProfile from 'preview-neuron-profiles';\n" + source.replaceAll('\r\n', '\n');
  const edit = (before, after) => { result = replace(result, before, after); };
  edit('export function ArchiveHead({', `function reviewHeadLink(caseId){
    const filters={model:headRun.model,part:'B26/H36',q:caseId};
    const q=new URLSearchParams({wb:'1',lens:'route',rtab:'database',dbStudy:'2983',db:new URLSearchParams(filters).toString()});
    return {href:'#?'+q.toString(),onClick:event=>{
      if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey||event.button!==0)return;
      event.preventDefault();window.openTestDatabase(filters,2983);
    }};
  }
  export function ArchiveHead({`);
  for(const target of ['caseId:sample.id,fromHead:true,model,fullscreen','caseId:sample.id']){
    edit('href={databaseHref(headRun.id,{'+target+'})}', '{...reviewHeadLink(sample.id)}');
  }
  edit('function NeuronExample({sample,kind,onKind,unit,onLocate,onGroup})', 'function NeuronExample({sample,kind,onKind,unit,onLocate,onGroup,model,snapshot,block,selectedUnit,onProfile})');
  edit("['unit','原生神经元'],['group'", "['unit','原生神经元'],['goodfire','Goodfire 分析'],['group'");
  edit("  const stableUnit=unit ?? sample?.units[0];", "  const stableUnit=unit;");
  edit("{kind==='sae'?<p>", "{kind==='goodfire'?<NeuronProfile model={model} snapshot={snapshot} block={block} index={selectedUnit} onLocate={onProfile}/>:kind==='sae'?<p>");
  edit("      kind==='unit'?<>", `      kind==='unit'&&(block!==sample.block||stableUnit==null)?<div className="ow-feature-entry">
        <p>{selectedUnit!=null?'B'+block+'/U'+selectedUnit+' 尚无随稿的四位置读数。':'选择一个 MLP 单元查看该对象的实测读数。'}</p>
        <button className="fw-tbtn" type="button" onClick={()=>onLocate(sample.units[0])}>定位归档样例 B{sample.block}/U{sample.units[0]}</button>
      </div>:kind==='unit'?<>`);
  return result;
}
module.exports = { workspace, evidence };
