import {useState} from 'react';
import {useWorkspaceCatalog,useWorkspaceResource} from './nativeObservationArchive.js';

const factors = {location:'地点属性',industry:'行业属性',query:'查询条件'};

function BlockOptions({blocks,block,onLocate}) {
  if (!onLocate) return null;
  return <div className="ow-position-list" aria-label="选择有 SAE 档案的模型块">
    {blocks.map(b=><button type="button" className="fw-tbtn" key={b} disabled={b===block}
      onClick={()=>onLocate(b)}>B{b} 字典</button>)}
  </div>;
}

function FeatureCard({layer}) {
  const [id,setId] = useState(layer.features[0].id);
  const feature = layer.features.find(f=>f.id===id);
  return <>
    <label>已登记的字典特征 <select aria-label="字典特征 F" value={feature.id} onChange={e=>setId(Number(e.target.value))}>
      {layer.features.map(f=><option key={f.id} value={f.id}>F{f.id} · {factors[f.nomination]}</option>)}
    </select></label>
    <p>候选变量为<strong>{factors[feature.nomination]}</strong>。问题前{feature.replicated[0]?'通过':'未通过'}复现，
      回答边界{feature.replicated[1]?'通过':'未通过'}复现。</p>
    <p>后续因果检验准入 · <strong>{feature.eligible?'符合准入条件':'未满足准入条件'}</strong></p>
    <p>锁箱中最弱的匹配对为样本 {feature.counterexampleIndices.join(' / ')}，定向差 {feature.orientedDelta.toFixed(5)}。
      {feature.signFailure?'此对未给出预期的正向差。':'此对仍保留预期的正向差。'}</p>
  </>;
}

export default function SparseDictionaryExplorer({snapshot,block,onLocate}) {
  const {catalog}=useWorkspaceCatalog();
  const entry=catalog?.resources.find(item=>item.kind==='dictionary'&&item.snapshot===snapshot);
  const {data:archive,loading,error}=useWorkspaceResource(entry?.id);
  const matching = archive?.snapshot === snapshot;
  const layer = matching && archive.layers.find(item=>item.block===block);
  return <section className="aw-method-content" aria-label="SAE 特征字典">
    <h3>SAE 特征字典</h3>
    {!matching ? <p role="status">{error||(loading?'正在读取中心字典…':'当前模型快照没有可读的中心字典归档。')}</p>
      : <>
        <BlockOptions blocks={archive.blocks} block={block} onLocate={onLocate}/>
        {!layer ? <p role="status">当前层没有字典档案。请选择已登记的模型块。</p> : <>
          <p><b>B{layer.block}</b> · 在 MLP 输出上训练的稀疏自编码器 · {layer.m} 个字典特征 · Top-{layer.k} · {layer.features.length} 个已登记候选</p>
          <FeatureCard key={layer.dictionarySha} layer={layer}/>
          <details><summary>字典保真度与实验来源</summary>
            <p>未解释方差比例（FVU）越低，重建误差相对越小。确认集 {layer.confirmFVU.toFixed(4)}，锁箱 {layer.lockboxFVU.toFixed(4)}。
              本层保真门{layer.fidelityGate?'通过':'未通过'}，原任务生成行为门{layer.behaviorGate?'通过':'未通过'}。</p>
            <p>F 编号指这版字典学到的特征。模型本身的 FFN 神经元使用 U 编号。语义提名、留出复现和因果准入均保留各自判据。</p>
            <a href="https://arxiv.org/abs/2309.08600" target="_blank" rel="noreferrer">稀疏自编码器特征分析的方法论文</a>
            <p>运行 <code>{archive.run}</code> · 模型快照 <code style={{overflowWrap:'anywhere'}}>{archive.snapshot}</code></p>
            <p>字典 SHA-256 <code style={{overflowWrap:'anywhere'}}>{layer.dictionarySha}</code>。此页回放已登记结果，权重身份来自训练清单、留出报告和历史存储审计。</p>
            <ul>{[archive.plan,...layer.sources,archive.historicalAudit].map(s=><li key={s.path}>
              <code style={{overflowWrap:'anywhere'}}>{s.path}</code><br/><small style={{overflowWrap:'anywhere'}}>SHA-256 {s.sha256}</small>
            </li>)}</ul>
          </details>
        </>}
      </>}
  </section>;
}
