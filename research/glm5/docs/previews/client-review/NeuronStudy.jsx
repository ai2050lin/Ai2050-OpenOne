import React, {useEffect, useMemo, useRef, useState} from 'react';
import NeuronHeatmap from './NeuronHeatmap.jsx';
import NeuronProfile from './NeuronProfiles.jsx';
import corpus from './b23-corpus.js';
import './neuron-study.css';

const samples=[
  {model:'Qwen3-14B',block:23,index:7708},
  {model:'Qwen3-14B',block:36,index:2429},
  {model:'Qwen3-4B',block:30,index:9517},
];
const address=(block,index)=>'B'+block+'/U'+index;
const number=value=>Number.isFinite(value)?String(value):'未采集';
const siteFromUrl=()=>new URLSearchParams(window.location.search).get('studySite')||'';

let savedCorpus;
function corpusFields(){
  if(!savedCorpus){
    savedCorpus=corpus.rows.map(row=>{
      const binary=atob(row.bits);
      const words=new Uint32Array(binary.length/2);
      for(let i=0;i<words.length;i++)words[i]=(binary.charCodeAt(2*i)|binary.charCodeAt(2*i+1)<<8)<<16;
      const values=new Float32Array(words.buffer);
      let max=0;
      for(const v of values)max=Math.max(max,Math.abs(v));
      const language=row.source.source==='wikipedia-zh'?'中文':'英文';
      return {...row,values,max,name:language+'片段 · 位置 '+row.position,title:row.document,run:corpus.run};
    });
  }
  return savedCorpus;
}

function archivedFields(model,block,example,readArchive){
  if(model.model_snapshot===corpus.snapshot&&block===corpus.block&&model.d_ff===corpus.width)return corpusFields();
  const n=example?.block===block&&example.snapshot===model.model_snapshot?example.positions.length:1;
  const result=[];
  for(let i=0;i<n;i++){
    const f=readArchive(model.model_snapshot,block,model.n_heads,model.d_ff,example,i);
    if(!f?.mlp?.values||f.mlp.values.length!==model.d_ff)continue;
    result.push({key:String(i),values:f.mlp.values,max:f.mlp.max,position:f.position,tokenId:f.tokenId,
      name:f.options?.[i]?.name||f.sourceTitle,title:f.sourceTitle,document:f.caseId,run:f.run,sourceFile:f.source,hash:f.sha256});
  }
  return result;
}

export default function NeuronStudy({model,models,block,index,example,exampleError,readArchive,saveView,onSelect,onTools,initialFull}){
  const fields=useMemo(()=>archivedFields(model,block,example,readArchive),[model.model_snapshot,model.n_heads,model.d_ff,block,example,readArchive]);
  const [site,setSite]=useState(siteFromUrl);
  const current=fields.find(r=>r.key===site)||fields[0];
  const [draft,setDraft]=useState(()=>address(block,index));
  const [error,setError]=useState('');
  const [origin,setOrigin]=useState(null);
  const [full,setFull]=useState(Boolean(initialFull));
  const panel=useRef(null), fullBtn=useRef(null), heatPanel=useRef(null), reportPanel=useRef(null);
  const positions=useRef(new Map());
  const corpusView=model.model_snapshot===corpus.snapshot&&block===corpus.block;
  const waiting=model.model_snapshot===corpus.snapshot&&block===36&&!example&&!exampleError;
  useEffect(()=>{setDraft(address(block,index));setError('');},[model.model_snapshot,block,index]);
  useEffect(()=>{
    const sync=()=>setFull(document.fullscreenElement===panel.current);
    const esc=e=>{if(e.key==='Escape')setFull(false);};
    document.addEventListener('fullscreenchange',sync);
    window.addEventListener('keydown',esc);
    return ()=>{document.removeEventListener('fullscreenchange',sync);window.removeEventListener('keydown',esc);};
  },[]);
  function changeSite(key){setSite(key);saveView({studySite:key||null});}
  function select(m,b,u,s=''){
    if(!m||b<0||b>=m.n_layers||u<0||u>=m.d_ff)return;
    positions.current.set(model.model_snapshot,{block,index,site:current?.key||''});
    changeSite(s);onSelect(m,b,u);
  }
  function locate(e){
    e.preventDefault();
    const match=draft.trim().match(/^B(\d+)\s*\/\s*U(\d+)$/i);
    if(!match||Number(match[1])>=model.n_layers||Number(match[2])>=model.d_ff){
      setError('请输入 B0–B'+(model.n_layers-1)+' / U0–U'+(model.d_ff-1)+' 范围内的编号。');return;
    }
    select(model,Number(match[1]),Number(match[2]),Number(match[1])===block?current?.key:'');
  }
  function sample(s){
    if(!origin)setOrigin({model:model.model,block,index,site:current?.key||''});
    select(models.find(m=>m.model===s.model),s.block,s.index);
  }
  function restore(){
    select(models.find(m=>m.model===origin.model),origin.block,origin.index,origin.site);
    setOrigin(null);
  }
  function switchModel(name){
    const m=models.find(item=>item.model===name);
    if(!m)return;
    const saved=positions.current.get(m.model_snapshot)||samples.find(s=>s.model===name)||{block:0,index:0};
    select(m,saved.block,saved.index,saved.site);
  }
  async function fullscreen(){
    if(full){
      if(document.fullscreenElement)await document.exitFullscreen();
      setFull(false);fullBtn.current?.focus();return;
    }
    setFull(true);
    try{await panel.current?.requestFullscreen?.();}catch{/* App hosts can use the same full-window layout. */}
  }
  function openTools(live){
    if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});
    onTools(live,full);
  }
  const maxAbs=fields.reduce((v,r)=>Math.max(v,r.max),0);
  const empty=waiting?'正在读取本页随附的算术题激活。':exampleError&&block===36?'归档读取失败。'+exampleError:
    model.model+' 的 B'+block+' 暂无随页归档的整层激活向量。可切换到下方有实测数据的归档实例。';
  return <section ref={panel} aria-label="神经元研究" className={'ns-study'+(full?' ns-full':'')}>
    <header className="ns-title">
      <div><p className="ns-path">模型分析 / MLP</p><h2>神经元研究</h2></div>
      <div className="ns-actions">
        <button onClick={()=>openTools(true)}>实时输入</button><button onClick={()=>openTools(false)}>3D 与插件</button>
        <button className="ns-full-btn" ref={fullBtn} aria-pressed={full} onClick={fullscreen}>{full?'退出全屏':'全屏观察'}</button>
      </div>
    </header>
    <form className="ns-object" onSubmit={locate}>
      <label>模型<select aria-label="研究模型" value={model.model} onChange={e=>switchModel(e.target.value)}>
        {models.map(m=><option key={m.model_snapshot} value={m.model}>{m.model}</option>)}</select></label>
      <label>原生神经元<div><input id="study-address" autoComplete="off" spellCheck={false} aria-label="原生神经元编号" value={draft}
        onChange={e=>{setDraft(e.target.value);setError('');}}/><button id="study-locate" type="submit">定位</button></div></label>
      <span className="ns-location">第 <b>{block}</b> 块 · MLP 单元 <b>{index}</b></span>
      {error&&<span className="ns-error" role="alert">{error}</span>}
    </form>
    <nav className="ns-jump" aria-label="查看分析结果"><button onClick={()=>heatPanel.current?.scrollIntoView({block:'start'})}>神经元热力图</button>
      <button onClick={()=>reportPanel.current?.scrollIntoView({block:'start'})}>Goodfire 报告</button></nav>
    <div className="ns-columns">
      <article ref={heatPanel} className="ns-card ns-field">
        <p className="ns-eyebrow">单次响应{current?' · 归档位置 '+current.position:''}</p>
        <NeuronHeatmap values={current?.values} width={model.d_ff} index={index} block={block} maxAbs={maxAbs}
          onPick={u=>onSelect(model,block,u)} missingReason={empty}/>
        {!!fields.length&&<div className="ns-material">
          <label htmlFor="study-site">观察位置</label><select id="study-site" value={current.key} onChange={e=>changeSite(e.target.value)}>
            {fields.map(row=><option key={row.key} value={row.key}>{row.name}</option>)}</select>
          {fields.length>1&&<div className="ns-readings" aria-label="同一神经元在各位置的原值">
            {fields.map(r=><button key={r.key} aria-pressed={r.key===current.key} onClick={()=>changeSite(r.key)}>
              <span>位置 {r.position}</span><strong>{number(r.values[index])}</strong></button>)}</div>}
          {current.text&&<blockquote>{current.text}</blockquote>}
          {corpusView&&index!==7708&&<p className="ns-method">此处沿用 U7708 选出的两个文本位置，便于逐单元比较。</p>}
          <details><summary>归档材料与测量口径</summary>
            <p>{current.title} · 位置 {current.position} · token ID {current.tokenId}。每个色块是这个位置上一个单元的 down_proj 输入。</p>
            {corpusView&&<p>这两个文本位置取自 U7708 在留出集的最高和最低响应。选择同层其他神经元时沿用这两个位置。两篇文本的语言和内容均不同。</p>}
            <p>运行 {current.run}</p><p>{current.sourceFile||corpus.source.directory}</p>
            {(current.hash||corpusView)&&<p>SHA-256 · {current.hash||corpus.hashes['B23-A.npy']}</p>}
          </details>
        </div>}
      </article>
      <article className="ns-card ns-dossier" ref={reportPanel}>
        <p className="ns-eyebrow">跨语料证据 · 项目归档</p>
        <NeuronProfile model={model.model} snapshot={model.model_snapshot} block={block} index={index}/>
        <p className="ns-method">Goodfire 风格的分析由本项目根据归档语料生成。</p>
      </article>
    </div>
    <section className="ns-examples" aria-label="归档实例"><strong>有实测数据的实例</strong>
      {samples.filter(s=>models.some(m=>m.model===s.model)).map(s=><button key={s.model+s.block}
        disabled={s.model===model.model&&s.block===block&&s.index===index} onClick={()=>sample(s)}>
        {s.model.replace('Qwen3-','')} · {address(s.block,s.index)}</button>)}
      {origin&&<button className="ns-return" onClick={restore}>返回原对象</button>}
    </section>
  </section>;
}
