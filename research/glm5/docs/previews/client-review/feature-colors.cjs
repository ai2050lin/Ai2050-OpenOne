/* Adapt the offline client review without changing the desktop renderer. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function archivedColorData(){
  // Use the same public, byte-checked trace as the free observation page. The renderer
  // needs colour scales and the corresponding native vectors. Omit weights and attention rows.
  const dir=path.resolve(__dirname,'..');
  const apiContext={window:{},atob,TextDecoder};
  vm.runInNewContext(fs.readFileSync(path.join(dir,'free-model-data.js'),'utf8'),apiContext);
  const api=apiContext.window.FreeModelData,model=api.model('Qwen3-4B');
  const blocks=[];
  let reference;
  function nativeBits(values){
    const bytes=Buffer.alloc(values.length*2),tmp=Buffer.alloc(4);
    for(let i=0;i<values.length;i++){
      const v=values[i];tmp.writeFloatLE(v);
      if(!Number.isFinite(v)||tmp.readUInt16LE(0)!==0)throw Error('Expected a finite archived BF16 field');
      bytes.writeUInt16LE(tmp.readUInt16LE(2),i*2);
    }
    return bytes.toString('base64');
  }
  for(let b=0;b<model.layers;b++){
    const field=api.fields(model.id,'ffn',b,0)[0];
    if(!field||field.values.length!==model.ffn||field.snapshot!==model.snapshot)throw Error('4B FFN shape or snapshot mismatch');
    reference ||= field;
    const identity=f=>f.snapshot===reference.snapshot&&f.traceId===reference.traceId&&f.caseId===reference.caseId&&
      f.position===reference.position&&f.tokenId===reference.tokenId&&f.run===reference.run;
    if(!identity(field))throw Error('FFN trace changed between blocks');
    const channels={head_z:[],head_q:[],head_k:[],head_v:[]};
    const rms=Array.from({length:model.heads},(_,h)=>{
      const measurements=api.fields(model.id,'head',b,h);
      for(const f of measurements){
        if(!channels[f.measure]||!identity(f)||f.sha256!==field.sha256||f.values.length!==model.headDim)throw Error('Head readout mismatch');
        const k=['head_k','head_v'].includes(f.measure)?f.kvGroup:h;
        if(!channels[f.measure][k])channels[f.measure][k]=nativeBits(f.values);
      }
      const z=measurements.find(f=>f.measure==='head_z');
      if(!z||!identity(z)||z.sha256!==field.sha256||z.block!==b||z.head!==h||z.values.length!==model.headDim||
        !z.values.every(Number.isFinite))throw Error('4B head identity or shape mismatch');
      return Math.sqrt(z.values.reduce((s,v)=>s+v*v,0)/z.values.length);
    });
    blocks.push({bits:nativeBits(field.values),channels,rms,source:field.source,sha256:field.sha256});
  }
  const {position,tokenId,caseId,traceId,run,snapshot}=reference;
  const nativeContext={};
  vm.runInNewContext(fs.readFileSync(path.resolve(dir,'../../../../frontend/src/components/v8/data/nativeHeadChannels.js'),'utf8'),nativeContext);
  const native=nativeContext.AI2050HeadArchive;
  if(native.boundary!=='head_z'||native.headWidth!==128)throw Error('Expected native head Z');
  const {cases,...head}=native;
  head.cases=Object.fromEntries(Object.entries(cases).map(([id,{values,position,tokenId}])=>{
    if(values.length!==native.headWidth||!values.every(Number.isFinite))throw Error('Invalid archived head vector');
    return [id,{position,tokenId,values,rms:Math.sqrt(values.reduce((s,v)=>s+v*v,0)/values.length)}];
  }));
  return {four:{model:model.id,heads:model.heads,kvHeads:model.kvHeads,ffn:model.ffn,position,tokenId,caseId,traceId,run,snapshot,blocks},head};
}

module.exports=(source,replace)=>{
  source=source.replaceAll('\r\n','\n');
  const jsx=fs.readFileSync(path.join(__dirname,'FeatureColors.jsx'),'utf8');
  source="import '../../../../research/glm5/docs/previews/client-review/feature-colors.css';\n"+
    'const observationColors='+JSON.stringify(archivedColorData())+';\n'+jsx+'\n'+source;
  source=replace(source,'  /* 纹理 */',[
    "  const featureScene=useMemo(()=>observeMode==='features' && dimensionsReady ? featureField(snapshot,layer-1,NH,NF,matchingExample,exampleSite,headCase,context.run,traceId) : null,",
    "    [observeMode,dimensionsReady,snapshot,layer,NH,NF,matchingExample,exampleSite,headCase,context.run,traceId]);",
    '  /* 纹理 */'
  ].join('\n'));
  source=replace(source,'    if (exampleReplay) {\n      for (const [kind, w, h]',[
    '    if(featureScene){',
    "      for(const [kind,w,h] of [['heads',8,Math.ceil(NH/8)],['mlp',mw,mh]]){",
    '        if(layer===0 || !shown(kind))continue;',
    '        const surface=featureTexture(featureScene,kind,w,h,kind===\'mlp\'?NF:NH);',
    '        out[kind]=[];out[kind][layer-1]=surface;',
    '      }',
    '    }',
    '    if (exampleReplay) {',
    '      for (const [kind, w, h]'
  ].join('\n'));
  source=replace(source,'if (field?.length || colors?.length) { const max = field', "if ((!featureScene || kind==='hs') && (field?.length || colors?.length)) { const max = field");
  source=replace(source,'colorValues, exampleReplay, exampleSite]);','colorValues, exampleReplay, exampleSite, featureScene]);');
  source=replace(source,'<FullNativeField kind={fieldKind}', '<FullNativeField featureScene={featureScene} kind={fieldKind}');
  source=replace(source,'{matchingHeadRun ? <ArchiveHeadCondition',[
    '{featureScene?.options ? <label>读出位置<select aria-label="全屏特征读出位置" value={featureScene.choice}',
    '  onChange={e=>featureScene.choicesAreCases?changeHeadCase(e.target.value):setExampleSite(+e.target.value)}>',
    '  {featureScene.options.map(o=><option key={o.id} value={o.id}>{o.name}</option>)}</select></label> :',
    ' matchingHeadRun ? <ArchiveHeadCondition'
  ].join('\n'));
  source=replace(source,"if (exampleReplay && ['hs','mlp'].includes(h.o))",[
    "if(featureScene && ['heads','mlp'].includes(h.o))return featureReadout(featureScene,h.o,h.idx);",
    "    if (exampleReplay && ['hs','mlp'].includes(h.o))"
  ].join('\n'));
  source=replace(source,"const popBody = visiblePop && (() => { const { h } = visiblePop;","const popBody = visiblePop && (() => { const { h } = visiblePop;\n    if(featureScene && ['mlp','heads'].includes(h.o))return <><h4>{featureReadout(featureScene,h.o,h.idx)}</h4><p>{featureScene.sourceTitle}{featureScene.position!=null?' · token '+featureScene.position:''}</p></>;");
  source=replace(source,"\${sa ? '' : '（结构）'}", "\${featureScene ? (featureScene.heads?.count ? '（Z 均方根）' : '（未采集）') : sa ? '' : '（结构）'}");
  source=replace(source,"\${browsing && object.module === 'mlp' && localObject?.typeColors ? '（普查类型）'", "\${featureScene ? (featureScene.mlp?.count ? '（门控激活）' : '（未采集）') : browsing && object.module === 'mlp' && localObject?.typeColors ? '（普查类型）'");
  source=replace(source,'<div className="aw-legend v8-note"><span>',[
    '{featureScene ? <FeatureColorLegend scene={featureScene} enabled={colorValues} object={object} groupMarked={marks.some(mark=>mark.feature)}',
    '        showHeads={shown("heads")} showMlp={shown("mlp")} onSite={setExampleSite} onCase={value=>{setHeadCase(value);saveView({headCase:value});}}/> :',
    '      <div className="aw-legend v8-note"><span>'
  ].join('\n'));
  return replace(source,'      </div>\n      {visiblePop&&<div className="aw-pop"', '      </div>}\n      {visiblePop&&<div className="aw-pop"');
};

module.exports.fullNative=(source,replace)=>{
  source=replace(source,'kvHeadCount,example,site=0,sample,trace,position,mlpOk,archiveHead,headCase,onPick,onChannel}',
    'kvHeadCount,example,site=0,sample,trace,position,mlpOk,archiveHead,headCase,onPick,onChannel,featureScene}');
  return replace(source,'  }else if(trace){',[
    '  }else if(featureScene){',
    "    const f=featureScene,ok=f.snapshot===snapshot && f.block===block;",
    "    contextKey=[f.run,f.caseId,f.traceId,f.position].join(':');",
    "    source+=' · '+f.sourceTitle+(f.position!=null?' · token '+f.position:'');",
    "    if(ok && !isHead && measure==='mlp_activation')values=f.mlp?.values;",
    "    if(ok && isHead){",
    '      values=f.readHead?.(measure,head);',
    "      if(values && ['head_k','head_v'].includes(measure))source+=' · 共享 KV'+Math.floor(head/(headCount/kvHeadCount));",
    '    }',
    "    if(!values)missing='当前归档在此位置没有保存这项测量。';",
    '  }else if(trace){'
  ].join('\n'));
};
