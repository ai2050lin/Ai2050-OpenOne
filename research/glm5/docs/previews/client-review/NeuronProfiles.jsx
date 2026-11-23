import React from 'react';
import atlas from './neuron-profiles.json';
import b23 from './b23-corpus.js';
import probe9517 from '../../../../neurons/goodfire/GF-L30-U9517-local.json';
import probe80 from '../../../../neurons/goodfire/GF-L30-U80-local.json';
import './neuron-profiles.css';

const q4='6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad';
const q14=atlas.snapshot;
const probes=new Map([[9517,probe9517],[80,probe80]]);
const probeHashes={9517:'878f2cb3b60428da80ac3160ef30a39e941bf138b025cae5cc59ef21a3a70ccf',
  80:'1174879499b5cd32915a7a3c3a8f4c0c0ad2cecc6cf1994b15992a527a5efac1'};
const examples=[
  {model:'Qwen3-14B',snapshot:q14,block:36,index:2429},
  {model:'Qwen3-4B',snapshot:q4,block:30,index:9517},
  {model:'Qwen3-4B',snapshot:q4,block:30,index:80},
];
const label=s=>'B'+s.block+'/U'+s.index;
const num=(value,digits=4)=>Number.isFinite(value)?String(Number(value.toFixed(digits))):'缺值';
const cache=new Map();
function decode(Type,b64){
  const bytes=Uint8Array.from(atob(b64),char=>char.charCodeAt(0));
  return new Type(bytes.buffer);
}
function layer(block){
  if(cache.has(block))return cache.get(block);
  const pack=atlas.layers[block];
  const decoded={run:pack.run,hashes:pack.hashes,mean:decode(Float64Array,pack.mean),std:decode(Float64Array,pack.std),
    words:decode(Uint32Array,pack.terms),matches:decode(Uint8Array,pack.repeat),
    poles:pack.poles.map(([id,bits])=>[decode(Uint16Array,id),decode(Uint16Array,bits)])};
  cache.set(block,decoded);return decoded;
}
const bits=new DataView(new ArrayBuffer(4));
function bf16(word){bits.setUint32(0,word*65536,true);return bits.getFloat32(0,true);}
function modelId(model){
  const name=typeof model==='string'?model:model?.model;
  return String(name||'').replaceAll('\\','/').split('/').at(-1).toLowerCase();
}
export function lookup({model,snapshot,block,index}){
  if(!Number.isInteger(block)||!Number.isInteger(index)||index<0)return null;
  if(modelId(model)==='qwen3-14b'&&snapshot===b23.snapshot&&block===b23.block&&index<b23.width){
    return {...b23.lookup(index),counts:b23.counts,sourceRoot:b23.source.directory};
  }
  if(snapshot===q4&&modelId(model)==='qwen3-4b'&&block===30&&probes.has(index)){
    return {state:'probe',report:probes.get(index),hash:probeHashes[index]};
  }
  if(snapshot!==q14||modelId(model)!=='qwen3-14b'||!Object.hasOwn(atlas.layers,block)||index>=atlas.width)return null;
  const a=layer(block),matches=a.matches[index];
  const samples=a.poles.map(([ids,values])=>{
    const [di,position,tokenId,text]=atlas.examples[ids[index]];
    return {document:atlas.docs[di].doc_id,source:atlas.docs[di],di,position,tokenId,text,activation:bf16(values[index])};
  });
  return {state:'atlas',run:a.run,hashes:a.hashes,mean:a.mean[index],std:a.std[index],
    words:Array.from(a.words.subarray(index*3,index*3+3)).map(i=>atlas.words[i]).filter(Boolean),
    confirm:matches>>4,holdout:matches&15,samples};
}

function Examples({model,snapshot,block,index,onLocate}){
  if(!onLocate)return null;
  return <div className="np-examples" aria-label="切换到有档案的神经元">
    <span>查看归档实例</span>
    {examples.filter(x=>!(x.snapshot===snapshot&&x.block===block&&x.index===index)).map(x=>
      <button key={x.model+x.index} type="button" className="fw-tbtn" onClick={()=>onLocate(x)}>
        {modelId(model)===modelId(x.model)?'':x.model+' · '}{label(x)}
      </button>)}
  </div>;
}
function AtlasProfile({block,index,card}){
  const counts=card.counts||atlas.counts;
  return <>
    <h4>哪些上下文伴随这个单元的高响应？</h4>
    <p>{card.words.length?'发现集高响应片段的候选词包括 '+card.words.join('、')+'。':'发现集未提取到满足筛选条件的候选词。'}
      归档用完整候选词表复查高响应片段，确认集命中 <strong>{card.confirm}/4</strong>，留出集命中 <strong>{card.holdout}/4</strong>。
      {card.holdout===0?'这组词没有预测到留出集中的四个高响应片段。':'这些词可以帮助检索相似上下文，单元语义仍待独立检验。'}</p>
    <div className="np-stats">
      <span>平均激活 <strong>{num(card.mean)}</strong></span><span>标准差 <strong>{num(card.std)}</strong></span>
      <span>{counts.documents.toLocaleString()} 篇文本 · 每篇 {counts.readouts} 处读出</span>
    </div>
    <p className="np-boundary">报告统计 down_proj 输入处的原生激活 A = SiLU(gate) × up。色块对应一次读数，均值和复现结果汇总了多个文本位置。</p>
    <h4>留出集中的两端</h4>
    <div className="np-quotes">{card.samples.map((s,i)=><article key={i}>
      <div><strong>{i?'最低激活':'最高激活'}</strong> <output>{num(s.activation,7)}</output></div>
      <p>{s.text}</p><small>{s.document} · 位置 {s.position} · token ID {s.tokenId}</small>
    </article>)}</div>
    <details><summary>测量口径与来源</summary>
      <p>取 MLP 的 down_proj 输入 A = SiLU(gate) × up 的第 {index} 个原生单元。均值和标准差基于全部 {counts.tokens.toLocaleString()} 个固定读出。两端分别为留出集固定读出中的最高和最低，展示截至读出位置最近的 32 个 token。</p>
      <p>这是响应与上下文的观察关联。单元的因果作用需要独立的干预记录。</p>
      <dl><dt>模型快照</dt><dd>{q14}</dd><dt>原始运行</dt><dd>{card.run}</dd>
        <dt>目录</dt><dd>{card.sourceRoot||'tests/glm5/result/qwen14_client_v2/native · '+atlas.run}</dd>
        {Object.entries(card.hashes).map(([suffix,hash])=><React.Fragment key={suffix}><dt>B{block}{suffix} · SHA-256</dt><dd>{hash}</dd></React.Fragment>)}
        <dt>语料与读出</dt><dd>documents.json · tokenizer.json · holdout</dd>
        {card.samples.map((s,i)=><React.Fragment key={i}><dt>{i?'低':'高'}端文档来源</dt>
          <dd>{s.source.repo} · {s.source.license} · {s.source.revision}</dd></React.Fragment>)}</dl>
    </details>
  </>;
}
function ProbeProfile({card,index}){
  const r=card.report, checks=r.checks;
  return <>
    <h4>门控响应与语境</h4>
    <p>正端方向读出的词包括 {r.S6.pos.slice(0,4).join('、')}。三组最小句对中，
      {Math.round(r.S9.summary.sign_acc*r.S9.summary.n)} 组符合预期方向。
      {index===9517?'历史、地区词转换标记等不同语境都能产生高响应。':'这组三个句对的符号检验未过门，仍缺少可复现的语义解释。'}</p>
    <div className="np-stats"><span>门控均值 <strong>{num(r.S1.g_mean)}</strong></span>
      <span>门控标准差 <strong>{num(r.S1.g_std)}</strong></span>
      <span>{r.corpus.n_windows.toLocaleString()} 个窗口 · {r.corpus.n_positions.toLocaleString()} 个位置</span></div>
    <p className="np-boundary">本报告测的是整份语料中的 gate_proj 预激活 g，位置在 SiLU 之前。空间颜色测的是当前题目单个位置的 A = SiLU(g) × up，两种读数的数值口径不同。</p>
    <div className="np-quotes">{[r.poles.pos[0],r.poles.neg[0]].map((p,i)=><article key={i}>
      <div><strong>{i?'负端样本':'正端样本'}</strong> <output>g {num(p.g)} · z {num(p.z,3)}</output></div>
      <p>{p.left}<mark>{p.hit}</mark></p><small>高亮为当前 token，摘录截至读出位置。</small>
    </article>)}</div>
    <details><summary>三组原句与逐句结果</summary>
      <p>取每句话中最大的 gate_proj 预激活，用该单元参照语料的均值和标准差换成 z 分数。</p>
      {r.S9.pairs.map(p=><div className="np-pair" key={p.a}><p>A · {p.a} <strong>{num(p.z_a,3)}</strong></p>
        <p>B · {p.b} <strong>{num(p.z_b,3)}</strong></p><b>A − B　{num(p.dz,3)}</b></div>)}
    </details>
    <details><summary>检验结果与来源</summary>
      <p>劈半方向余弦 {num(checks.S4_split_half.value,3)}。读写对齐分差 {num(checks.S8_read_write.value,3)}，阈值 ≤ {r.criteria.read_write_gap_max}。未通过的项目为 {r.verdict.failed.join('、')}。单一门控单元的因果效应仍待验证。</p>
      <p>这两份早期报告由历史测试索引关联到 Qwen3-4B。原报告记录了运行环境和语料指纹，未单独写入模型权重哈希。</p>
      <dl><dt>测量</dt><dd>gate_proj 第 {index} 维，SiLU 之前</dd><dt>项目索引关联的模型快照</dt><dd>{q4}</dd>
        <dt>运行与语料</dt><dd>{r.corpus.job_id} · {r.corpus.corpus_fp}</dd><dt>原始档案</dt>
        <dd>research/neurons/goodfire/GF-L30-U{index}-local.json</dd><dt>SHA-256</dt><dd>{card.hash}</dd></dl>
    </details>
  </>;
}
export {examples};
export default function NeuronProfile({model,snapshot,block,index,onLocate}){
  const selected=Number.isInteger(block)&&Number.isInteger(index);
  const card=lookup({model,snapshot,block,index});
  return <section className="np-profile" aria-label="Goodfire 神经元分析结果" data-testid="goodfire-profile" data-state={card?.state||'missing'}>
    <header><h3>{selected?label({block,index})+' · ':''}Goodfire 神经元分析</h3>
      <span>原生 MLP 单元</span></header>
    {card?.state==='atlas'?<AtlasProfile block={block} index={index} card={card}/>:
      card?.state==='probe'?<ProbeProfile card={card} index={index}/>:
      <p role="status">{selected?'此模型快照与单元的 Goodfire 报告尚未收入本页。':'在模型空间选择一个 MLP 单元，可查看与该原生编号匹配的报告。'}
        当前离线索引覆盖 Qwen3-14B 的 B23、B28–B39 以及 Qwen3-4B 的 B30/U9517、B30/U80。</p>}
    <Examples {...{model,snapshot,block,index,onLocate}}/>
  </section>;
}
