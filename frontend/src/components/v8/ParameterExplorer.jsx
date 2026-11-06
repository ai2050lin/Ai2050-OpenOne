import { useCallback,useEffect, useRef, useState } from 'react';
import { getJson, useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import Tower from './Tower.jsx';
import ResearchFocus from './ResearchFocus.jsx';
import { parameterRegion, preferredParameter } from './parameterUtil.js';
import { nodeBridge, nodeCall } from '../app/rdc_fusion/contributeUtil.js';
import {publishResearchContext,useResearchContext} from '../app/researchContext.js';

function readParameter(path, signal) {
  return nodeBridge() ? nodeCall('GET', path.replace('/api/parameters/', '/parameters/')) : getJson(path, {online:true, signal});
}

export function useParameters(path){
  const local=!!nodeBridge();const api=useApi(local?null:path,{online:true});
  const [state,setState]=useState({}),[version,setVersion]=useState(0);
  useEffect(()=>{
    if(!local||!path)return undefined;let dead=false;
    nodeCall('GET',path.replace('/api/parameters/','/parameters/')).then(data=>{if(!dead)setState({path,data})}).catch(e=>{if(!dead)setState({path,error:e.message})});
    return()=>{dead=true};
  },[path,local,version]);
  return local?{data:state.path===path?state.data:null,error:state.path===path?state.error:'',loading:!!path&&state.path!==path,reload:()=>setVersion(n=>n+1)}:api;
}

export function ArrayCanvas({ count, selected, onPick, color, label,values,columns=128,maxAbs }) {
  const ref=useRef(null);const cols=columns,cell=5,rows=Math.ceil(count/cols);
  useEffect(()=>{
    const c=ref.current;if(!c)return;const ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
    const max=values?(maxAbs||Math.max(1e-12,...values.map(Math.abs))):1;
    for(let i=0;i<count;i++){const a=values?Math.abs(values[i])/max:0;ctx.fillStyle=values?(values[i]>=0?`rgba(42,120,214,${.06+.84*a})`:`rgba(235,104,52,${.06+.84*a})`):i===selected?'#e34948':color;ctx.fillRect(i%cols*cell,Math.floor(i/cols)*cell,4,4);if(values&&i===selected){ctx.strokeStyle='#111827';ctx.strokeRect(i%cols*cell,Math.floor(i/cols)*cell,4,4)}}
  },[count,selected,color,values,cols,rows,maxAbs]);
  return <div className="v8-gridwrap"><canvas ref={ref} width={cols*cell} height={rows*cell} aria-label={`${label}全部${count}个索引；${values?'已载入完整采集响应，蓝正橙负':'灰色表示未加载状态'}`}
    onClick={onPick?e=>{const r=e.currentTarget.getBoundingClientRect();const x=(e.clientX-r.left)*e.currentTarget.width/r.width,y=(e.clientY-r.top)*e.currentTarget.height/r.height;const i=Math.floor(y/cell)*cols+Math.floor(x/cell);if(x>=0&&x<cols*cell&&i>=0&&i<count)onPick(i)}:undefined} /></div>;
}

export default function ParameterExplorer({ researchModel='', selection, snapshot, onLocate, onResponse, onEmbedding, onSelectModel, onCoordinate }) {
  const context=useResearchContext();
  const embedded=Boolean(selection);
  const expectedSnapshot=snapshot??context.snapshot;
  const [narrow,setNarrow]=useState(()=>window.innerWidth<=720);
  const [response,setResponse]=useState(null);
  useEffect(()=>{const resize=()=>setNarrow(window.innerWidth<=720);window.addEventListener('resize',resize);return()=>window.removeEventListener('resize',resize)},[]);
  const modelsApi=useParameters('/api/parameters/models');
  const q=new URLSearchParams(window.location.search),qi=k=>Math.max(0,Number(q.get(k))||0);
  const [id,setId]=useState(()=>q.get('pmodel')||''),[ownBlock,setBlock]=useState(()=>qi('pblock')),[ownKind,setKind]=useState(()=>['embedding','head','mlp'].includes(q.get('pkind'))?q.get('pkind'):'head'),[ownHead,setHead]=useState(()=>qi('phead')),[ownUnit,setUnit]=useState(()=>qi('punit')),[matrix,setMatrix]=useState(''),[row,setRow]=useState(()=>selection?.module==='embedding'?0:qi('prow')),[col,setCol]=useState(()=>qi('pcol'));
  const block=embedded?Math.max(0,selection.layer-1):ownBlock;
  const kind=embedded?(selection.module==='attn'?'head':selection.module):ownKind;
  const head=embedded?selection.index:ownHead,unit=embedded?selection.index:ownUnit;
  const chooseResponse=useCallback(d=>{
    if(embedded){onResponse?.(d);return;}
    setResponse(d);setBlock(d.block);setKind('mlp');setUnit(d.selected_unit);setMatrix('');setRow(0);setCol(0);
  },[embedded,onResponse]);
  const models=modelsApi.data?.models||[];const model=id?models.find(m=>m.id===id):models.find(m=>m.name.toLowerCase()===(researchModel.split('|')[0]||'').toLowerCase())||models[0];
  const cfg=model?.config;
  useEffect(()=>{if(!cfg||embedded)return;setBlock(b=>Math.max(0,Math.min(cfg.num_hidden_layers-1,b)));setHead(h=>Math.max(0,Math.min(cfg.num_attention_heads-1,h)));setUnit(u=>Math.max(0,Math.min(cfg.intermediate_size-1,u)))},[cfg,embedded]);
  const identity=useParameters(model&&expectedSnapshot?`/api/parameters/identity?model=${encodeURIComponent(model.id)}`:null);
  const contextMatches=!expectedSnapshot||identity.data?.sha256===expectedSnapshot;
  useEffect(()=>{if(!model||embedded)return;const url=new URL(window.location.href);for(const [k,v] of Object.entries({pblock:block,pkind:kind,phead:head,punit:unit,prow:row}))url.searchParams.set(k,v);window.history.replaceState(null,'',url);publishResearchContext({pmodel:model.id})},[model?.id,block,kind,head,unit,row,embedded]);
  const indexMatches=!embedded||Boolean(cfg&&block<cfg.num_hidden_layers&&(
    kind==='embedding'?unit<cfg.vocab_size:kind==='head'?head<cfg.num_attention_heads:unit<cfg.intermediate_size));
  const tensors=useParameters(model&&(!embedded||contextMatches)&&indexMatches?`/api/parameters/tensors?model=${encodeURIComponent(model.id)}&block=${block}&embedding=${kind==='embedding'}`:null);
  const options=(tensors.data?.tensors||[]).filter(t=>kind==='embedding'?true:kind==='head'?t.name.includes('self_attn'):t.name.includes('.mlp.')).flatMap(t=>t.name.includes('gate_up_proj')?[{...t,id:t.name+':gate'},{...t,id:t.name+':up'}]:[{...t,id:t.name}]);
  const tensor=options.find(t=>t.id===matrix)||preferredParameter(options,kind);
  const region=cfg&&tensor?(embedded&&kind==='embedding'&&unit<cfg.vocab_size?{row:unit,col:0,rows:1,cols:cfg.hidden_size}:parameterRegion(cfg,kind,kind==='head'?head:unit,tensor.id)):null;
  const embeddingCoordinate=embedded&&kind==='embedding';
  const changeCol=value=>embeddingCoordinate?onCoordinate?.(value):setCol(value);
  const r=region?Math.min(row,region.rows-1):0,c=region?Math.min((embeddingCoordinate?selection.coordinate:null)??col,region.cols-1):0;
  const slice=useParameters(model&&tensor&&region?`/api/parameters/slice?model=${encodeURIComponent(model.id)}&name=${encodeURIComponent(tensor.name)}&row=${region.row+r}&col=${region.col+c}&rows=${Math.min(8,region.rows-r)}&cols=${Math.min(8,region.cols-c)}`:null);
  const [embeddingError,setEmbeddingError]=useState('');
  const embeddingTensor=kind==='embedding'?preferredParameter(options,'embedding')?.name:'';
  useEffect(()=>{
    if(!embedded||!onEmbedding||!embeddingTensor||!model||!contextMatches||unit>=cfg.vocab_size)return;
    const controller=new AbortController();
    let active=true;
    setEmbeddingError('');
    (async()=>{
      const values=[];
      let fingerprint='';
      for(let start=0;start<cfg.hidden_size;start+=64*4){
        const columns=Array.from({length:Math.min(4,Math.ceil((cfg.hidden_size-start)/64))},(_,i)=>start+64*i);
        const chunks=await Promise.all(columns.map(col=>readParameter(`/api/parameters/slice?${new URLSearchParams({
          model:model.id,name:embeddingTensor,row:unit,col,rows:1,cols:Math.min(64,cfg.hidden_size-col)
        })}`,controller.signal)));
        if(!active)return;
        for(const [i,chunk] of chunks.entries()){
          const rowValues=chunk.values?.[0];
          fingerprint ||= chunk.header_sha256;
          if(chunk.start?.[0]!==unit||chunk.start?.[1]!==columns[i]||chunk.header_sha256!==fingerprint||
             rowValues?.length!==Math.min(64,cfg.hidden_size-columns[i])||!rowValues.every(Number.isFinite))throw Error('词嵌入切片无法对齐');
          values.push(...rowValues);
        }
      }
      onEmbedding({index:unit,snapshot:expectedSnapshot,values});
    })().catch(error=>{if(active)setEmbeddingError(error.message)});
    return()=>{active=false;controller.abort()};
  },[embedded,onEmbedding,embeddingTensor,model?.id,cfg?.hidden_size,cfg?.vocab_size,contextMatches,unit,expectedSnapshot]);
  const reset=()=>{setMatrix('');setRow(0);setCol(0)};
  if(!cfg)return <section className="v8-card"><h3>本机真实参数</h3><Empty>{modelsApi.error||(modelsApi.loading?'正在读取本机模型目录…':id?'此链接指定的本机模型目录不可用，未替换成其他模型。':'尚未找到配置目录中的 safetensors 模型。在配置 → 本机算力指定模型目录后刷新。')}</Empty>{id&&<button className="fw-tbtn" onClick={()=>setId('')}>重新选择本机模型</button>}<button className="fw-tbtn" onClick={modelsApi.reload}>重新读取</button></section>;
  const field=contextMatches&&response?.local_model_id===model.id&&response.block===block&&kind==='mlp'?response:null;
  return <><ResearchFocus model={model} onLocate={p=>{if(embedded){onLocate?.(p);return;}setBlock(p.block);setKind(p.kind);setUnit(p.unit);reset()}} onResponse={chooseResponse}/>{field&&<section className="v8-card"><b>已采集样本的完整 MLP 响应</b><p className="v8-note">{field.sample_id} · {Number.isInteger(field.tokens)?`${field.tokens} tokens`:'token 数未提供'} · {field.readout}{Number.isInteger(field.generation_step)?` · 生成步 ${field.generation_step}`:''} · 全部 {field.values.length} 个单元；蓝正、橙负、灰中点，色标为本数组 max|A|，黑框为选中单元。</p><p className="v8-note">A[{unit}] = {field.values[unit]?.toPrecision(6)}；{field.response_state==='unedited baseline'?'原模型响应（未编辑）':field.response_state==='edited alpha6'?'编辑后响应（α=6）；下方权重仍为原模型参数':'已保存的采集响应'}，非本次即时前向，色彩不代表因果重要性。</p><details><summary>采集或重建方式</summary><p className="v8-note">{field.reconstruction||'未提供；请回查原始证据。'}</p></details><details><summary>查看该样本完整输入</summary><pre className="v8-sample-text">{field.text}</pre>{field.prefix_token_ids&&<p className="v8-note">实际前缀 token IDs：{field.prefix_token_ids.join(', ')}；下一 token ID：{field.predicted_token_id}。原场根据实际已生成历史采集。</p>}</details><button className="fw-tbtn" onClick={()=>setResponse(null)}>清除响应叠加</button></section>}<div className="v8-parameter-explorer">
    {expectedSnapshot&&<p className="v8-note" role="status">{identity.error?`来源快照尚未核验：${identity.error}`:!identity.data?'正在核验来源快照。':contextMatches?'当前原权重与研究来源的完整快照一致。':embedded?'请选择与工作台快照一致的权重目录。':'当前权重与来源快照不同：只显示静态参数，不绑定来源样本或语言特征。'}</p>}
    <div className="v6-toolbar"><label>本机权重 <select value={model.id} onChange={e=>{setId(e.target.value);setResponse(null);reset();if(embedded){onSelectModel?.(e.target.value);return;}publishResearchContext({snapshot:'',trace:'',context_trace:'',run:'',pmodel:e.target.value});setBlock(0);setHead(0);setUnit(0)}}>{models.map(m=><option key={m.id} value={m.id}>{m.name}{models.filter(x=>x.name===m.name).length>1?` · 目录 ${m.id.slice(0,6)}`:''}</option>)}</select></label>
      <button className="fw-tbtn" onClick={()=>{modelsApi.reload();tensors.reload();slice.reload()}}>刷新本机参数</button><span className="v8-note">真实CPU切片 · {cfg.num_hidden_layers}层 · {cfg.num_attention_heads}Q/{cfg.num_key_value_heads}KV · {cfg.intermediate_size.toLocaleString()} MLP单元/层</span></div>
    {!embedded&&<p className="v8-note">本机权重目录模型与研究模型「{researchModel||'未选择'}」分别标识；尚未核验完整快照时，不将语言特征自动绑定到此权重。灰色单元只表示结构索引，不表示激活。</p>}
    <div className={embedded?'v8-embedded-weights':'v8-params-layout'}>{!embedded&&<aside className="v8-card"><Tower nLayers={cfg.num_hidden_layers} selected={block+1} onPick={hs=>{setBlock(Math.max(0,hs-1));if(hs===0)setKind('embedding');reset()}} modules={{attn:1,mlp:1}} selModule={{hs:block+1,mod:kind==='head'?'attn':'mlp'}} onPickModule={(hs,mod)=>{setBlock(hs-1);setKind(mod==='attn'?'head':'mlp');reset()}} height={narrow?280:570} caption="全部层 · 蓝Attention / 橙MLP / 灰嵌入" /></aside>}
    <section className="v8-card">{!embedded&&<><div className="v6-toolbar">{[['head','所有注意力Head'],['mlp','全部MLP单元'],['embedding','词嵌入表']].map(([k,t])=><button className={'fw-tbtn'+(kind===k?' primary':'')} key={k} onClick={()=>{setKind(k);reset()}}>{t}</button>)}</div>
      <label>块 <input aria-label="参数块" type="number" min="0" max={cfg.num_hidden_layers-1} value={block} onChange={e=>{setBlock(Math.max(0,Math.min(cfg.num_hidden_layers-1,Number(e.target.value)||0)));reset()}} /></label>
      {kind==='head'&&<><div className="v8-head-grid">{Array.from({length:cfg.num_attention_heads},(_,h)=><button key={h} className={head===h?'on':''} onClick={()=>{setHead(h);reset()}}>H{h}<small>KV{Math.floor(h/(cfg.num_attention_heads/cfg.num_key_value_heads))}</small></button>)}</div><p className="v8-note">每个Q头独立列出；GQA的K/V按组共享。Q/K归一化及偏置按真实张量目录显示。</p></>}
      {kind==='mlp'&&<><p>块{block} · 全部 {cfg.intermediate_size.toLocaleString()} / {cfg.intermediate_size.toLocaleString()} 单元</p><ArrayCanvas count={cfg.intermediate_size} selected={unit} onPick={u=>{setUnit(u);reset()}} color="#cbd5e1" label="MLP" values={field?.values}/><label>单元索引 <input aria-label="参数单元" type="number" min="0" max={cfg.intermediate_size-1} value={unit} onChange={e=>{setUnit(Math.max(0,Math.min(cfg.intermediate_size-1,Number(e.target.value)||0)));reset()}} /></label></>}
      {kind==='embedding'&&<p className="v8-note">词表 {cfg.vocab_size.toLocaleString()} × {cfg.hidden_size}。用矩阵行索引选择token；不把token行直接称为概念。</p>}</>}
      {embeddingError&&<Empty>{embeddingError}</Empty>}
      {embedded&&!contextMatches ? <Empty>权重快照匹配后可查看该对象的参数。</Empty> : !indexMatches ? <Empty>此索引超出当前本机权重目录的范围。</Empty> : <>
      <div className="v6-toolbar"><label>参数 <select value={tensor?.id||''} onChange={e=>{setMatrix(e.target.value);setRow(0);setCol(0)}}>{options.map(t=><option key={t.id} value={t.id}>{t.id.split('.').slice(-3).join('.')}</option>)}</select></label></div>
      {tensor&&region&&<><p className="v8-note">{tensor.name} · {tensor.dtype} · 原张量 {tensor.shape.join('×')} · 当前对象区域 {region.rows}×{region.cols}{region.shared?' · 共享':''}</p><div className="v6-toolbar">区域行 <input aria-label="矩阵行" type="number" min="0" max={region.rows-1} value={r} onChange={e=>setRow(Math.max(0,Math.min(region.rows-1,Math.trunc(Number(e.target.value)||0))))} /> 列 <input aria-label="矩阵列" type="number" min="0" max={region.cols-1} value={c} onChange={e=>changeCol(Math.max(0,Math.min(region.cols-1,Math.trunc(Number(e.target.value)||0))))} />{[['←',()=>changeCol(Math.max(0,c-8))],['→',()=>changeCol(Math.min(region.cols-1,c+8))],['↑',()=>setRow(Math.max(0,r-8))],['↓',()=>setRow(Math.min(region.rows-1,r+8))]].map(([t,fn])=><button className="fw-tbtn" key={t} onClick={fn}>{t}</button>)}</div></>}
      {tensors.error?<Empty>{tensors.error}</Empty>:!tensor?<Empty>{tensors.loading?'正在读取参数张量目录…':'此模型块尚无受支持的参数张量。'}</Empty>:slice.error?<Empty>{slice.error}</Empty>:slice.data?<><div className="v8-weight-scroll"><table className="v6-table"><tbody>{slice.data.values.map((vs,i)=><tr key={i}><th>{slice.data.start[0]+i}</th>{vs.map((v,j)=><td key={j} title={`${tensor.name}[${slice.data.start[0]+i},${slice.data.start[1]+j}]`} style={{background:v>0?'#eff6ff':'#fff3ed'}}>{v.toPrecision(5)}</td>)}</tr>)}</tbody></table></div><p className="v8-note">来源 {slice.data.source} · 实读 {slice.data.loaded_numel} 个标量 · 文件头指纹 {slice.data.header_sha256.slice(0,16)}（不是完整权重快照）</p></>:<Empty>读取真实权重切片…</Empty>}
      {!embedded&&<details><summary>关联语言特征与研究证据</summary><p className="v8-note">原生单元特征与真实响应在「图谱覆盖」「响应对照」及神经元研究卡中查询。只有模型快照、层、读出边界和原生索引一致的记录才可绑定；尚无匹配证据时不生成解释。</p></details>}</>}
    </section></div>
  </div></>;
}
