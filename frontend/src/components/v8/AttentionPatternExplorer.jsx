import {useMemo, useState} from 'react';
import './attentionPatternStats.js';

const pct = p => (100 * p).toFixed(2) + '%';
const topChoices = [1,3,5,10,20];

function labelAt(labels,index,grouped) {
  const item = labels?.[index];
  if (typeof item === 'string') return item || '空文本';
  if (typeof item === 'number') return String(item);
  return item?.text ?? item?.token ?? (grouped ? '来源组 '+(index+1) : '位置 '+index);
}

export default function AttentionPatternExplorer({weights,labels,source,reason,grouped=false,onPin}) {
  const [selectedK,setK] = useState(5);
  const result = useMemo(() => globalThis.AI2050AttentionStats.summarize(weights,{topK:selectedK}),[weights,selectedK]);
  const mismatch = labels != null && weights != null && labels.length !== weights.length;
  const problem = reason || (mismatch ? '来源标签数与权重数不一致，无法定位来源。' : (!result.ok && result.reason));
  const unit = grouped ? '来源组' : '源 token';
  const choices = result.ok ? [...new Set([...topChoices.map(n=>Math.min(n,result.count)),result.topK])] : [];
  return <section className="aw-method-content" aria-label="注意力分布">
    <h3>注意力分布</h3>
    {source && <p className="v8-note">{source}</p>}
    {problem ? <p role="status">{problem}</p> : <>
      <p>{grouped?'按归档范围合并来源后，比较各组分到的注意力。组内 token 的分布未知。':'固定一个查询 token，查看此 Head 对各个源 token 分配的注意力。'}</p>
      <div style={{display:'flex',flexWrap:'wrap',gap:'6px 20px',alignItems:'baseline'}}>
        <span>分布熵 <b>{result.entropyBits.toFixed(3)} bit</b></span>
        <span>有效{unit}数 <b>{result.effectiveSources.toFixed(2)}</b></span>
        <label>前 <select aria-label="前 K 个来源" value={result.topK} onChange={e=>setK(Number(e.target.value))}>
          {choices.map(k=><option key={k}>{k}</option>)}
        </select> 项累计 <b>{pct(result.topMass)}</b></label>
      </div>
      <table className="v6-table" aria-label="来源按注意力质量排序">
        <thead><tr><th>排名</th><th>{unit}</th><th className="num">归一化质量</th></tr></thead>
        <tbody>{result.ranked.slice(0,result.topK).map((row,rank)=><tr key={row.index}>
          <td>{rank+1}</td><td>{!grouped && <small>#{row.index} </small>}{labelAt(labels,row.index,grouped)}</td>
          <td className="num">{pct(row.probability)}</td>
        </tr>)}</tbody>
      </table>
      {onPin&&<button type="button" className="fw-tbtn" onClick={()=>onPin({title:'注意力分布',scope:grouped?'按归档来源组汇总':'按源 token 汇总',
        values:[`熵 ${result.entropyBits.toFixed(3)} bit`,`前 ${result.topK} 项 ${pct(result.topMass)}`]})}>将注意力结果标在 3D</button>}
      <details><summary>计算口径</summary>
        <p>原始权重和 {result.total.toFixed(6)}，共 {result.count} 个{unit}。在量化容差 ±0.005 内用权重和归一化。熵为 −Σ p log₂ p，有效来源数为 2 的熵次方。</p>
        <p>这里衡量来源的集中程度。判断来源怎样改变输出还需要值向量、写入路径与干预对照。</p>
        <a href="https://aclanthology.org/P19-3007/" target="_blank" rel="noreferrer">逐头注意力可视化参考 · BertViz</a>
      </details>
    </>}
  </section>;
}
