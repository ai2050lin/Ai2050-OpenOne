module.exports = function studyWorkspace(source, replace) {
  source = "import NeuronStudy from 'preview-neuron-study';\n" + source;
  source = replace(source,
    'const measure = () => setStageHeight(Math.ceil(nav.getBoundingClientRect().bottom - sceneRef.current.getBoundingClientRect().top));',
    `const scene=sceneRef.current;
    const measure=()=>{if(nav.isConnected&&scene?.isConnected)setStageHeight(Math.ceil(nav.getBoundingClientRect().bottom-scene.getBoundingClientRect().top));};`);
  source = replace(source,'  }, [max]);\n  const [pop, setPop]', '  }, [max, studyDesk]);\n  const [pop, setPop]');
  source = replace(source, "  const [stageHeight, setStageHeight]", `  const [studyDesk,setStudyDesk]=useState(()=>{
    const p=new URLSearchParams(window.location.search);
    if(p.get('studyPane'))return p.get('studyPane')==='study';
    return p.get('smodule')==='mlp' && /^(live|features)$/.test(p.get('observeMode')||'') &&
      p.has('sindex') && !p.get('trace') && !p.get('run') && p.get('featureView')!=='goodfire' &&
      p.get('observeTab')!=='plugin' && p.get('observeFullscreen')!=='1';
  });
  const [stageHeight, setStageHeight]`);
  source = replace(source,"observeMode !== 'features' && !archivedUnit)) return;", "observeMode !== 'features' && !archivedUnit && !(studyDesk && layer===37))) return;");
  source = replace(source,"[example, exampleActive, observeMode, archivedUnit]", "[example, exampleActive, observeMode, archivedUnit, studyDesk, layer]");
  source = replace(source,'  return <div className="v8-route aw ow-graph" aria-label="模型分析">',`  const enterStudy=()=>{
    if(object.module!=='mlp'||object.index==null) {
      const isFour=activeModel?.model==='Qwen3-4B';
      locate({layer:isFour?31:24,module:'mlp',index:isFour?9517:7708});
    }
    setStudyDesk(true);saveView({studyPane:'study'});
  };
  if(studyDesk&&activeModel&&object.module==='mlp'&&Number.isInteger(object.index)&&object.index>=0&&
      object.index<activeModel.d_ff&&layer>=1&&layer<=activeModel.n_layers)return <NeuronStudy model={activeModel} models={models} block={layer-1}
    index={object.module==='mlp'?object.index:null} example={matchingExample} exampleError={exampleError}
    readArchive={featureField} saveView={saveView} initialFull={max}
    onSelect={(m,b,u)=>{if(m.model_snapshot!==snapshot)selectModel(m);locate({layer:b+1,module:'mlp',index:u});}}
    onTools={(liveInput,fullscreen)=>{
      changeFullscreen(Boolean(fullscreen));
      if(liveInput){chooseObserveMode('live');setConsoleTab('run');setTab('live');saveView({observeTab:'run',observePlugin:'live'});}
      setStudyDesk(false);saveView({studyPane:'tools'});
    }}/>;
  return <div className="v8-route aw ow-graph" aria-label="模型分析">`);
  return replace(source,'<h2>模型分析</h2>', '<h2>模型分析</h2><button type="button" className="ns-open" onClick={enterStudy}>{object.module==="mlp"&&object.index!=null?"神经元研究":"打开神经元归档实例"}</button>');
};
