// These layout overrides apply only to the client review bundle.
module.exports = function organizeGraph(source, replace) {
  source = source.replaceAll('\r\n','\n');
  const edit = (from, to) => { source = replace(source, from, to); };
  function replaceBetween(start, end, replacement) {
    const first = source.indexOf(start);
    const last = source.indexOf(end, first + start.length);
    if (first < 0 || last < 0) throw Error('Graph layout no longer matches client: ' + start);
    source = source.slice(0, first) + replacement + source.slice(last);
  }

  source = "import {GraphStudies} from 'preview-distributed-work';\nimport GraphHeadEvidence from 'preview-graph-head';\n"
    + "import {graphModes, graphObservation, observationField, GraphObservation, ObservationAnalysis, GraphFeatures} from 'preview-graph-modes';\n"
    + source;
  edit("[recordTab,setRecordTab]=useState('records')","[recordTab,setRecordTab]=useState('study')");
  edit("setConsoleTab('research');setRecordTab('library');setLibraryVisited(true);}",
    "setConsoleTab('research');setRecordTab('study');setLibraryVisited(true);}");
  edit("new URLSearchParams(window.location.search).get('reviewFocus')==='object'?'object':archiveView?'research':",
    "observeMode==='live'?'run':observeMode==='features'?'object':reviewFocus==='head'?'plugin':reviewFocus==='object'?'object':studyPack||archiveView?'research':");
  edit("  const [tab, setTab] = useState(() => OPTIONAL_MODE_IDS.has(mode)",
    `  const previewRoute = new URLSearchParams(window.location.search);
    const reviewFocus = previewRoute.get('reviewFocus'), studyPack = previewRoute.get('studyPack');
    const [headCase,setHeadCase]=useState(previewRoute.get('headCase') || 'confirm-000-02');
    const requestedMode=previewRoute.get('observeMode');
    const [observeMode,setObserveMode]=useState(graphModes.some(([id])=>id===requestedMode)?requestedMode:'archive');
    const requestedSite=previewRoute.get('observeSite');
    const [observeSite,setObserveSite]=useState(requestedSite!==null && /^[0-3]$/.test(requestedSite)?Number(requestedSite):3);
    const [featureKind,setFeatureKind]=useState('unit');
    const [activationDisplay,setActivationDisplay]=useState(true);
    const [featureDisplay,setFeatureDisplay]=useState(true);
    const modePanels=useRef({});
    const [displayOpen,setDisplayOpen]=useState(false);
    const [tab, setTab] = useState(() => reviewFocus==='head'?'heads':OPTIONAL_MODE_IDS.has(mode)`);
  edit("models.find(m => /qwen3-4b/i.test(m.model)) || models[0]", "models[0]");
  edit("  const [openOpt, setOpenOpt] = useState(() => new Set());", "");
  edit("  const [float, setFloat] = useState({ x: 16, y: 96 });",
    "  const [float, setFloat] = useState({ x: 16, y: 132 });");
  edit("const shown = id => on.has(id) && !hide.has(id);",
    `const shown = id => !hide.has(id);
  const matchesObservation=activeModel?.model_snapshot===graphObservation.snapshot && snapshot===graphObservation.snapshot;
  const observedLayer=graphObservation.block+1;
  const hasReplay=observeMode==='live'&&matchesObservation;`);
  edit(" : sa ? '' : '（结构）'}\`, grid: [hw, hh]",
    " : hasReplay ? '（实测）' : sa ? '' : '（结构）'}\`, grid: [hw, hh]");
  edit(" : sa && mlpOk ? '' : browsing ?", " : hasReplay ? (layer===observedLayer?'（实测）':'（结构）') : sa && mlpOk ? '' : browsing ?");
  edit("  const describe = h => {", `  const describe = h => {
    if(hasReplay&&(h.o==='hs'||h.o==='mlp')){
      const field=observationField(h.o,h.l,observeSite);
      const name=h.o==='mlp'?'B'+(h.l-1)+'/U'+h.idx:'L'+h.l+' · 坐标 '+h.idx;
      return name+(field?' · 激活 '+field[h.idx].toFixed(4):' · 该层未采集 MLP');
    }`);
  edit("setListed(next); setOn(new Set(next)); setHide(new Set());",
    "setListed(next); setOn(new Set(next));");
  edit("    setOn(s => new Set([...s, o])); setHide(s =>",
    "    setHide(s =>");
  edit("    setTab('object'); setConsoleTab('object'); setPop({ h, x, y, scope });",
    `    if(observeMode==='archive'){setTab('object');setConsoleTab('object');setPop({h,x,y,scope});}
    else if(observeMode==='features'&&h.o==='mlp'){setFeatureKind('unit');setConsoleTab('object');}`);
  edit("  useEffect(() => { const k = e => { if (e.key === 'Escape') setMax(false); }; window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k); }, []);",
    `  useEffect(()=>{
    const onEscape=event=>{
      if(event.key!=='Escape'||document.querySelector('dialog[open]'))return;
      if(locator)setLocator(false);
      else if(displayOpen)setDisplayOpen(false);
      else if(max)setMax(false);
    };
    window.addEventListener('keydown',onEscape);
    return ()=>window.removeEventListener('keydown',onEscape);
  },[locator,displayOpen,max]);`);

  replaceBetween("  const ops = ", "  /* ---------- 插件面板", `  const toggleGrid=id=>setHide(current=>{
    const next=new Set(current);
    if(next.has(id))next.delete(id);
    else next.add(id);
    return next;
  });
  const dragConsole = event => {
    if(!max||event.button!==0)return;
    const handle=event.currentTarget, rect=consoleRef.current.getBoundingClientRect();
    const x=event.clientX-rect.left,y=event.clientY-rect.top;
    handle.setPointerCapture(event.pointerId);
    const move=e=>{
      const width=consoleRef.current?.offsetWidth||0;
      const height=handle.offsetHeight;
      setFloat({
        x:Math.max(8,Math.min(window.innerWidth-width-8,e.clientX-x)),
        y:Math.max(52,Math.min(window.innerHeight-height-12,e.clientY-y)),
      });
    };
    const stop=()=>handle.removeEventListener('pointermove',move);
    handle.addEventListener('pointermove',move);
    handle.addEventListener('lostpointercapture',stop,{once:true});
  };
  const ops = <div ref={pluginsRef} className="ow-plugins">
    <div className="ow-tool-row" role="group" aria-label="分析插件">
      <span>分析插件</span>
      {AVAILABLE.filter(p=>listed.has(p.id) && on.has(p.id)).map(p=>
        <button className="fw-tbtn" key={p.id} type="button" aria-pressed={consoleTab==='plugin'&&tab===p.id}
          onClick={()=>openPlugin(p.id)}>{p.id==='heads'?'Head':p.id==='mlp'?'MLP':p.name}</button>)}
      <button className="fw-tbtn ow-plugin-manager" type="button" aria-expanded={cfg} onClick={()=>setCfg(value=>!value)}>管理插件</button>
    </div>
    {cfg&&<div className="ow-plugin-options" role="group" aria-label="插件管理">
      {AVAILABLE.map(p=><label key={p.id}><input type="checkbox" checked={listed.has(p.id)} onChange={e=>listPlugin(p.id,e.target.checked)}/>{p.name}</label>)}
      <button className="fw-tbtn" onClick={()=>resetPlugins(DEFAULT_PLUGINS)}>恢复默认</button>
    </div>}
  </div>;

`);
  replaceBetween('  const modelControls = ', '      <div className="aw-load"', '  const modelStatus = ');
  edit('</>; })()}</div></div>;', '</>; })()}</div>;');

  replaceBetween('  return <div className="v8-route aw" aria-label="统一空间工作台">', '    {!dimensionsReady', `
  const archiveHead = archivePart==='B26/H36';
  const selectedMlp=matchesObservation&&layer===observedLayer&&object.module==='mlp'&&object.index!=null
    ? {index:object.index,value:observationField('mlp',layer,observeSite)[object.index]} : null;
  const featureUnit=selectedMlp?.index ?? graphObservation.units[0];
  const writePreviewSelection=(key,value)=>{
    const url=new URL(window.location.href);
    url.searchParams.set(key,value);
    window.history.replaceState(window.history.state,'',url);
  };
  const chooseObservationSite=value=>{
    setObserveSite(value);
    writePreviewSelection('observeSite',value);
  };
  const changeObserveMode=value=>{
    if(value===observeMode)return;
    modePanels.current[observeMode]={tab,consoleTab};
    const last=modePanels.current[value];
    setObserveMode(value);
    setConsoleTab(last?.consoleTab || (value==='live'?'run':value==='features'?'object':'research'));
    if(last)setTab(last.tab);
    else if(value==='archive')setRecordTab('study');
    setPop(null);setLocator(false);setDisplayOpen(false);setCfg(false);
    writePreviewSelection('observeMode',value);
  };
  const selectObservationModel=()=>{
    const target=models.find(row=>row.model_snapshot===graphObservation.snapshot);
    if(target)selectModel(target);
  };
  const locateNeuron=(index=graphObservation.units[0])=>{
    if(!matchesObservation)return;
    setHide(previous=>{const next=new Set(previous);next.delete('mlp');return next;});
    changeObject({layer:observedLayer,module:'mlp',index});
  };
  const observationTexture = useMemo(()=>{
    if(!activationDisplay)return {};
    if(observeMode==='archive')return TX;
    if(observeMode!=='live'||!matchesObservation)return {};
    const output={};
    for(const [kind,width,height] of [['hs',hw,hh],['mlp',mw,mh]]){
      const field=observationField(kind,layer,observeSite);
      if(!shown(kind)||!field)continue;
      const scale=Math.max(...graphObservation.positions.map((_,i)=>absMax(observationField(kind,layer,i))));
      output[kind]=[];
      output[kind][kind==='mlp'?layer-1:layer]=tex(width,height,pixels=>{
        for(let i=0;i<width*height;i++) put(pixels,i,i<field.length?divc(field[i],scale):[246,248,251]);
      });
    }
    return output;
  },[activationDisplay,observeMode,TX,matchesObservation,hide,layer,observeSite,hw,hh,mw,mh]);
  const modeMarks = observeMode==='features'&&featureDisplay&&featureKind==='group'&&matchesObservation&&layer===observedLayer?
    [...marks,...graphObservation.units.map(idx=>({id:'mlp',idx,feature:true}))]:marks;
  const sampleHeadEntry = {snapshot:models[0]?.model_snapshot, location:{layer:27,module:'attn',index:36}};
  const setArchivedCase = value => {
    setHeadCase(value);
    writePreviewSelection('headCase',value);
  };
  const openHeadExample = () => {
    if(archiveHead){openPlugin('heads');return;}
    if(max && activeModel?.model_snapshot===sampleHeadEntry.snapshot){
      locate(sampleHeadEntry.location);openPlugin('heads');return;
    }
    window.openEvidenceWorkspace('head',sampleHeadEntry);
  };
  const viewSelect = <select aria-label="空间视角" value={view} onChange={e=>setView(e.target.value)}>
    {VORDER.map(key=><option key={key} value={key}>{VIEWS[key][2]}</option>)}
  </select>;
  const currentObject = !layer?'词嵌入':!selected?'第 '+layer+' 层':selected.module==='attn'?
    'B'+(layer-1)+'/H'+selected.index:selected.module==='mlp'?'B'+(layer-1)+'/U'+selected.index:
    '第 '+layer+' 层 · 坐标 '+selected.index;
  return <div className="v8-route aw ow-graph" aria-label="模型分析">
`);
  const toolbarStart = source.indexOf('      <div className="aw-ov tr"><button');
  const toolbarEnd = source.indexOf('      {locator && ', toolbarStart);
  if(toolbarStart<0||toolbarEnd<0)throw Error('Missing client scene toolbar');
  const nativeActions = source.slice(toolbarStart, toolbarEnd).trim()
    .replace('className="aw-ov tr"', 'className="ow-scene-actions"')
    .replace('<button type="button" aria-expanded={locator}',
      '<button type="button" aria-expanded={displayOpen} onClick={()=>{setDisplayOpen(value=>!value);setLocator(false);}}>显示插件</button><button type="button" aria-expanded={locator}')
    .replace('setLocator(v => !v); setPop(null);','setLocator(v => !v); setDisplayOpen(false); setPop(null);')
    .replace('setMax(m => !m); setPop(null);','setMax(m => !m); setPop(null); setLocator(false); setDisplayOpen(false);');
  source = source.slice(0, toolbarStart) + source.slice(toolbarEnd);
  edit('      <Scene dims=', `      <div className="ow-stage-nav">
      <div className="ow-mode-bar">
        <div role="group" aria-label="观察模式">
          {graphModes.map(([id,label])=><button key={id} type="button" aria-pressed={observeMode===id}
            onClick={()=>changeObserveMode(id)}>{label}</button>)}
        </div>
        <span className="ow-mode-source">{observeMode==='live'?'静态稿 · 实测回放':observeMode==='features'?'按模型快照查询':'离线归档'}</span>
      </div>
      <div className="ow-scene-bar">
        <label className="ow-model">模型<select aria-label="模型" value={activeModel?.model||''} onChange={e=>{
          const target=models.find(row=>row.model===e.target.value);if(target)selectModel(target);
        }}>
          {!activeModel&&<option value="">选择模型</option>}{models.map(row=><option key={row.model} value={row.model}>{shortModel(row.model)}</option>)}
        </select></label>
        <span className="ow-object" aria-live="polite">{currentObject}</span>
        <span className="ow-view">{viewSelect}</span>
        ${nativeActions}
      </div>
      {displayOpen&&<div className="aw-locator ow-display" role="group" aria-label="显示插件">
        {[[ 'heads','Head'],['hs','HiddenState'],['mlp','MLP']].map(([id,label])=>
          <label key={id}><input type="checkbox" checked={!hide.has(id)} onChange={()=>toggleGrid(id)}/>{label} 网格</label>)}
        {observeMode!=='features'&&<label><input type="checkbox" checked={activationDisplay} onChange={e=>setActivationDisplay(e.target.checked)}/>数值着色</label>}
        {observeMode==='features'&&<label><input type="checkbox" checked={featureDisplay} onChange={e=>setFeatureDisplay(e.target.checked)}/>单元组标记</label>}
        {observeMode==='archive'&&!browsing&&native&&<ResearchDisclosure title="数值色标与读出">{OPTS.hs()}{OPTS.mlp()}{OPTS.heads()}</ResearchDisclosure>}
      </div>}
      </div>
      <Scene dims=`);
  edit('tex={TX} marks={marks}', 'tex={observationTexture} marks={modeMarks}');
  edit('overlays={overlays} live={!browsing', "overlays={observeMode==='archive'&&activationDisplay?overlays:null} live={observeMode==='archive'&&!browsing");
  edit('links={sceneLinks} archive={libraryVisible', "links={observeMode==='archive'?sceneLinks:[]} archive={observeMode==='archive'&&libraryVisible");
  edit('        <label>视角<select aria-label="空间视角" value={view} onChange={e=>setView(e.target.value)}>{VORDER.map(v=><option key={v} value={v}>{VIEWS[v][2]}</option>)}</select></label>', '');
  replaceBetween('      <div className="aw-big">', '      <div className="aw-legend', '');
  edit('· 点格子看数值 · 点层展开 · 灰色 = 未叠加数据 · 金虚线 = 局部放大', '· 点击层或网格 · 灰色为结构');
  edit("{!browsing && (shown('hs') || shown('mlp'))", "{!browsing && native && (shown('hs') || shown('mlp'))");
  edit("{!browsing && shown('heads')", "{!browsing && native && shown('heads')");
  edit('<span>{sceneLinks.length > 0 &&', `{hasReplay&&activationDisplay&&<span><i style={{background:'linear-gradient(90deg,#d4622b,#fff,#1f5fbf)'}}/>激活值 · 蓝正橙负</span>}
      {observeMode==='features'&&featureDisplay&&featureKind==='group'&&matchesObservation&&layer===observedLayer&&<span><i style={{background:'#b87916'}}/>标记 · 冻结单元组</span>}
      <span>{observeMode==='archive' && sceneLinks.length > 0 &&`);
  edit('      {max && ops}', '');
  replaceBetween('    <section className="v8-card aw-cons"', '      <div className="aw-task-switch', `    <section className={'v8-card aw-cons'+(max?' ow-float':'')} aria-label="控制台" ref={consoleRef}
      style={max?{left:'clamp(8px, '+float.x+'px, calc(100% - var(--ow-float-width) - 8px))',top:'clamp(52px, '+float.y+'px, 45vh)'}:undefined}>
      {max&&<div className="ow-float-head">
        <div className="ow-grip" onPointerDown={dragConsole}><strong>控制台</strong><span>拖动标题栏移动</span></div>
        <label className="ow-layer-step">第 {layer} 层<input type="range" min="0" max={NL-1} value={layer} aria-label="全屏层" onChange={e=>setLayer(+e.target.value)}/></label>
      </div>}
      <div className="aw-body"><div className="aw-right" ref={panelRef}>
        <div className="ow-console-nav">
          <div className="aw-tabs" role="tablist" aria-label="控制台功能">{[['object','对象详情'],['run','运行与对照'],['plugin','插件分析'],['research','成果与方法']].map(([id,name])=><button type="button" role="tab" key={id} id={'console-tab-'+id} aria-controls={'console-panel-'+id} aria-selected={consoleTab===id} onClick={()=>selectConsoleTab(id)}>{name}</button>)}</div>
          {ops}
        </div>
        <div role="tabpanel" id="console-panel-run" aria-labelledby="console-tab-run" hidden={consoleTab!=='run'}>
          {consoleTab==='run'&&(observeMode==='live'?
            <GraphObservation matches={matchesObservation} position={observeSite} setPosition={chooseObservationSite}
              sampleSelected={selectedMlp} onLocate={()=>locateNeuron()} onModel={selectObservationModel} draft={text} onDraft={setText}/>:
            observeMode==='features'?<div className="ow-mode-panel"><p>选择「对象详情」浏览神经元档案，或切换到「测试记录」比较已有结果。</p>
            <button className="fw-tbtn" onClick={()=>selectConsoleTab('object')}>查看当前对象</button></div>:<>
          <div className="ow-run-nav">
            <div className="aw-subtabs" role="tablist" aria-label="运行结果视图">
              {[['three','三线索对照'],['pair','冻结样本对'],['live','逐层变化'],...(gen.res?[['result','生成结果']]:[])].map(([id,label])=>{
                const isComparison=id==='three'||id==='pair';
                const selected=isComparison?runTab==='response'&&comparisonView===id:runTab===id;
                return <button key={id} type="button" role="tab" aria-selected={selected} onClick={()=>{
                  if(isComparison)setComparisonView(id);
                  openPlugin(isComparison?'response':id);
                }}>{label}</button>;
              })}
            </div>
            <button className="fw-tbtn" onClick={openHeadExample}>在 Head 中观察 B26/H36 →</button>
          </div>
          <ResearchDisclosure key={runTab} title="运行输入、样本与本机模型" initiallyOpen={runTab==='live'}>
            {modelStatus}
`);
  replaceBetween('      <div className="aw-body">{!max && ops}', '          {consoleTab === \'run\' && (threeCueActive', `
          </ResearchDisclosure>
`);
  edit('<ThreeCueComparison /></Suspense> : panel)',
    `{activeModel?.model_snapshot===models[0]?.model_snapshot?<ThreeCueComparison/>:
      <div className="ow-section-head"><p>这组对照的归档模型是 Qwen3-14B。</p><button className="fw-tbtn" onClick={()=>selectModel(models[0])}>切换到 Qwen3-14B</button></div>}</Suspense> : panel)`);
  edit('          </ResearchDisclosure>}\n        </div>\n        <div role="tabpanel" id="console-panel-plugin"',
    '          </ResearchDisclosure>}\n          </>)}\n        </div>\n        <div role="tabpanel" id="console-panel-plugin"');
  replaceBetween('          <div className="v6-toolbar aw-panel-toolbar"><label>分析插件', '        <div role="tabpanel" id="console-panel-object"', `
          {consoleTab==='plugin'&&(observeMode==='archive'?(pluginTab==='heads'&&archiveHead?
            <GraphHeadEvidence caseId={headCase} onCase={setArchivedCase}/>:panel):
            ['hs','mlp','heads','lens'].includes(pluginTab)?
              <ObservationAnalysis plugin={pluginTab} matches={matchesObservation} layer={layer} object={object}
                position={observeSite} onLocate={()=>locateNeuron()}/>:panel)}
        </div>
`);
  replaceBetween('{archivePart&&<div className="v6-toolbar">', '<ModelStructure', `
          {consoleTab==='object'&&(observeMode==='features'?
            <GraphFeatures matches={matchesObservation} featureKind={featureKind} setFeatureKind={setFeatureKind}
              position={observeSite} unit={featureUnit}
              onLocate={index=>{setFeatureKind('unit');locateNeuron(index);}}
              onGroup={()=>{setFeatureDisplay(true);locateNeuron();}} onModel={selectObservationModel}/>:
            <>
          <div className="ow-section-head">
            <div><h3>{currentObject}</h3><p>{shortModel(activeModel?.model || '')} {snapshot&&'· 快照 '+snapshot.slice(0,12)}</p></div>
            <div className="ow-object-actions">
              {archiveHead?<button className="fw-tbtn primary" onClick={()=>openPlugin('heads')}>查看 Head 分析 →</button>:
                <button className="fw-tbtn" onClick={openHeadExample}>定位 B26/H36 示例</button>}
              {archivePart&&<button className="fw-tbtn" onClick={()=>window.openTestDatabase({part:archivePart})}>检索 {archivePart} 的测试</button>}
            </div>
          </div>
          `);
  edit("        </div> : <p className=\"v8-note\">{reason}</p>}/></div>",
    "        </div> : <p className=\"v8-note\">{reason}</p>}/></>)}</div>");
  edit("[['library','成果库'],['records','测试记录'],['methods','方法与基准'],['history','历史对照']]", "[['study','测试记录'],['library','图谱成果'],['methods','方法与基准'],['history','历史对照']]");
  edit("          <div hidden={recordTab!=='records'}>{recordVisited.has('records')&&<Suspense fallback={<p>正在载入测试记录…</p>}><TestRecords onReviewLocate={locateArchive} model={activeModel && shortModel(activeModel.model)}/></Suspense>}</div>",
    `          {consoleTab==='research'&&recordTab==='study'&&(observeMode==='archive'?<GraphStudies model={activeModel?.model}/>:
            <div className="ow-mode-panel"><p>测试目录和逐题结果在「测试记录」模式。</p><button className="fw-tbtn" onClick={()=>changeObserveMode('archive')}>打开测试记录</button></div>)}
          {consoleTab==='research'&&recordTab==='methods'&&<button className="fw-tbtn ow-database" onClick={()=>window.openTestDatabase({})}>打开测试数据库</button>}`);
  return source;
};
