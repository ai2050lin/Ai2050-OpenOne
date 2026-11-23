// This review module is compiled alongside AssocWorkbench and uses its native tex/grid and diverging colour function.
const fourBColorCache=new Map();
const featureMissing=[198,205,214];
function decodeFeatureBits(bits){
  const bin=atob(bits),raw=new Uint32Array(bin.length/2);
  for(let i=0;i<raw.length;i++)raw[i]=(bin.charCodeAt(2*i)|bin.charCodeAt(2*i+1)<<8)<<16;
  return new Float32Array(raw.buffer);
}
const featureNumber=n=>Number.isFinite(n)?String(Number(n.toPrecision(5))):'未采集';
function featureMagnitude(values){
  let max=0;
  for(const x of values)if(Number.isFinite(x))max=Math.max(max,Math.abs(x));
  return max;
}
function featureField(snapshot,block,nh,nf,example,site,headCase,run,traceId){
  const {four,head}=observationColors;
  const empty={snapshot,block,sourceTitle:'当前块没有与所选材料匹配的激活',heads:null,mlp:null};
  if(block<0)return {...empty,sourceTitle:'词嵌入层没有 Head 或 MLP'};
  if(snapshot===four.snapshot && nh===four.heads && nf===four.ffn && four.blocks[block]){
    if(run&&run!==four.run || traceId&&traceId!==four.traceId)return empty;
    if(!fourBColorCache.has(block)){
      const {bits,rms,channels,source,sha256}=four.blocks[block];
      const values=decodeFeatureBits(bits),cache=new Map();
      fourBColorCache.set(block,{...empty,sourceTitle:'木盒颜色题 · 输入末尾',position:four.position,tokenId:four.tokenId,
        run:four.run,caseId:four.caseId,traceId:four.traceId,source,sha256,
        readHead(measure,h){
          const i=['head_k','head_v'].includes(measure)?Math.floor(h/(nh/four.kvHeads)):h;
          const packed=channels[measure]?.[i],key=measure+'/'+i;
          if(!packed)return null;
          if(!cache.has(key))cache.set(key,decodeFeatureBits(packed));
          return cache.get(key);
        },
        heads:{values:rms,count:nh,max:featureMagnitude(rms)},mlp:{values,count:nf,max:featureMagnitude(values)}});
    }
    return fourBColorCache.get(block);
  }
  if(traceId)return empty;
  if(example && snapshot===example.snapshot && block===example.block && nf===example.mlpShape[1] && (!run||run===example.run)){
    const position=example.positions[site],values=example.field('mlp',block+1,site);
    if(!position||!values)return empty;
    const max=Math.max(...example.positions.map((_,p)=>featureMagnitude(example.field('mlp',block+1,p))));
    return {...empty,sourceTitle:'算术题 · Phase '+example.phase,run:example.run,caseId:example.caseId,
      position:position.index,tokenId:position.tokenId,source:example.source.file,sha256:example.source.sha256,
      choice:String(site),options:example.positions.map((p,i)=>({id:String(i),name:EXAMPLE_SITES[i]+' · token '+p.index})),
      mlp:{values,count:values.length,max}};
  }
  if(snapshot===head.snapshot && block===head.block && head.head<nh && (!run||run===head.run)){
    const id=headCase||'confirm-000-02',item=head.cases[id];
    if(!item)return empty;
    const values=Array(nh).fill(null);values[head.head]=item.rms;
    const names={'confirm-000-00':'按姓名','confirm-000-01':'按印刷标签','confirm-000-02':'按物理首行'};
    return {...empty,sourceTitle:'五行查找题 · Phase 2983',run:head.run,caseId:id,position:item.position,tokenId:item.tokenId,
      source:head.source,sha256:head.sha256,choice:id,choicesAreCases:true,
      readHead:(measure,h)=>measure==='head_z'&&h===head.head?item.values:null,
      options:Object.entries(head.cases).map(([key,c])=>({id:key,name:names[key]+' · token '+c.position})),
      heads:{values,count:1,max:Math.max(...Object.values(head.cases).map(c=>c.rms))}};
  }
  return empty;
}
function featureTexture(scene,kind,width,height,count){
  const field=scene[kind];
  const texture=tex(width,height,pixels=>{
    for(let i=0;i<width*height;i++){
      const v=field?.values[i];
      put(pixels,i,i>=count?[246,248,251]:Number.isFinite(v)?divc(v,field.max):featureMissing);
    }
  });
  if(texture){texture.dataset.featureGrid=kind;texture.dataset.featureBlock=String(scene.block);}
  return texture;
}
function featureReadout(scene,kind,index){
  const number=scene[kind]?.values[index],prefix=kind==='heads'?'H':'U',label=kind==='heads'?'Z 均方根':'门控激活';
  return 'B'+scene.block+'/'+prefix+index+' · '+label+' '+(Number.isFinite(number)?String(number):'未采集');
}
function FeatureColorLegend({scene,enabled,object,showHeads,showMlp,groupMarked,onSite,onCase}){
  const kind=object.module==='attn'?'heads':object.module==='mlp'?'mlp':null;
  const value=kind&&object.index!=null?scene[kind]?.values[object.index]:null;
  function scale(id){
    const f=scene[id],isHead=id==='heads';
    return <span data-feature-scale={id} data-count={f?.count||0} data-max={f?.max??''} className="fc-scale">
      <strong>{isHead?'Head · Z 均方根':'MLP · 门控激活'}</strong>
      {f?<>{isHead?'0':'−'+featureNumber(f.max)}<i aria-hidden="true" className={isHead?'fc-positive':'fc-signed'}/>{featureNumber(f.max)}</>:<small>未采集</small>}
    </span>;
  }
  return <div className="aw-legend v8-note fc-legend" data-feature-colors="" data-snapshot={scene.snapshot} data-block={scene.block}
    data-case={scene.caseId||''} data-position={scene.position??''} aria-label="特征数值图例">
    <div className="fc-row"><strong>{scene.sourceTitle}</strong>
      {scene.options?<label><span className="sr-only">着色读出位置</span><select aria-label="着色读出位置" value={scene.choice}
        onChange={e=>scene.choicesAreCases?onCase(e.target.value):onSite(Number(e.target.value))}>
        {scene.options.map(o=><option key={o.id} value={o.id}>{o.name}</option>)}</select></label>:scene.position!=null&&<span>token {scene.position}</span>}
      {kind&&object.index!=null&&<output data-feature-value={kind} data-index={object.index} data-value={Number.isFinite(value)?value:''}
        title={featureReadout(scene,kind,object.index)}>{featureReadout(scene,kind,object.index)}</output>}
    </div>
    {enabled?<div className="fc-row">
      {showHeads&&scale('heads')}{showMlp&&scale('mlp')}
      <span className="fc-scale"><i className="fc-missing"/>缺测</span>
      {groupMarked&&<span><i style={{background:'#b87916'}}/>冻结单元组</span>}
    </div>:<span>数值着色已关闭</span>}
    {!!scene.run&&<details><summary>色标与来源</summary><div>Head 颜色表示写回投影前头输出 Z 各通道的均方根。MLP 颜色表示 down_proj 输入的有符号激活。白色对应零，蓝色为正，橙色为负。亮度按相对幅度的 0.6 次幂映射，保留原生编号。范围取当前块同类数据的最大绝对值，有多个可选条件时共用范围。</div>
      <div>运行 <code>{scene.run}</code> · 题目 <code>{scene.caseId}</code> · token {scene.position}（ID {scene.tokenId}）</div>
      <div>快照 <code>{scene.snapshot}</code></div>{scene.traceId&&<div>轨迹 <code>{scene.traceId}</code></div>}
      <div>原场 <code>{scene.source}</code> · SHA-256 <code>{scene.sha256}</code></div>
    </details>}
  </div>;
}
