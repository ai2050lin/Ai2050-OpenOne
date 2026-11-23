import { useEffect, useMemo, useState } from 'react';
import { request } from './MechanismTraceExplorer.jsx';
import './representationGeometry.js';

const { compare: compareRepresentations } = globalThis.AI2050RepresentationGeometry;

function layerLabel(index) {
  return index === 0 ? 'L0 · 嵌入层' : 'L'+index+' · B'+(index-1)+' 后';
}

function comparisonLayer(reference, layer, layerCount) {
  return reference != null && reference < layerCount ? reference : Math.max(0, layer === 0 ? 1 : layer - 1);
}

export default function RepresentationGeometry({positions = [], read, layer, layerCount, source, reason, referenceLayer, onReferenceChange,onPin}) {
  const [localReference, setLocalReference] = useState(null);
  const [matrixKey, setMatrixKey] = useState('rdmA');
  const haveLayers = Number.isInteger(layerCount) && layerCount >= 2 &&
    Number.isInteger(layer) && layer >= 0 && layer < layerCount;
  const other = comparisonLayer(onReferenceChange ? referenceLayer : localReference, layer, layerCount);
  const result = useMemo(() => {
    if (reason) return {error:reason};
    if (!haveLayers || !read) return {error:'需要同一模型快照的两层 HiddenState 和位置索引。'};
    if (new Set(positions.map(p => p.key)).size !== positions.length) return {error:'读出位置有重复，无法确定一一对应关系。'};
    try {
      return compareRepresentations(positions.map(p => read(p.key, layer)), positions.map(p => read(p.key, other)));
    } catch {
      return {error:'未能读取所选层的原生向量，请检查运行产物与读出位置。'};
    }
  }, [reason, haveLayers, read, positions, layer, other]);
  const matrixLayer = matrixKey === 'rdmA' ? layer : other;
  return <section className="ow-mode-panel" aria-label="表征几何">
    <div className="ow-section-head"><div><h3>表征几何 · CKA / RSA</h3>
      <p>比较同一组位置在不同层中的相对关系。{source}</p></div>
      {haveLayers && <label>对照层 <select aria-label="表征几何对照层" value={other}
        onChange={event => (onReferenceChange || setLocalReference)(Number(event.target.value))}>
        {Array.from({length:layerCount}, (_, i) => <option key={i} value={i}>{layerLabel(i)}</option>)}
      </select></label>}
    </div>
    {result.error ? <p className="v8-note" role="status">{result.error}</p> : <>
      <p>{layerLabel(layer)} 与 {layerLabel(other)} · {result.count} 个配对位置</p>
      <dl className="ow-feature-results">
        <div><dt>线性 CKA</dt><dd>{result.cka?.toFixed(4) ?? '中心化后无变化'}</dd></div>
        <div><dt>RSA · Spearman ρ</dt><dd>{result.rsa?.toFixed(4) ?? '距离缺失或无秩变化'}</dd></div>
      </dl>
      {onPin&&(result.cka!=null||result.rsa!=null)&&<button type="button" className="fw-tbtn" onClick={()=>onPin({
        title:'表征几何',scope:`L${layer} / L${other} · ${result.count} 个配对位置`,
        values:[result.cka!=null&&'CKA '+result.cka.toFixed(4),result.rsa!=null&&'RSA '+result.rsa.toFixed(4)].filter(Boolean),
        layers:[...new Set([layer,other])],
      })}>将几何结果标在 3D</button>}
      <div className="ow-step-list" role="group" aria-label="距离矩阵层">
        {[['rdmA',layer,'当前'],['rdmB',other,'对照']].map(([key, l, title]) =>
          <button type="button" key={key} className="fw-tbtn" aria-pressed={matrixKey === key}
            onClick={() => setMatrixKey(key)}>{title}层 L{l} 距离矩阵</button>)}
      </div>
      <div style={{maxWidth:'100%',overflowX:'auto'}}>
        <table className="ow-value-table"><caption>{layerLabel(matrixLayer)} · 余弦距离（0–2）</caption>
          <thead><tr><th scope="col">读出位置</th>{positions.map(p => <th key={p.key} scope="col">{p.label}</th>)}</tr></thead>
          <tbody>{result[matrixKey].map((row,i) => <tr key={positions[i].key}>
            <th scope="row">{positions[i].label}</th>
            {row.map((v,j) => <td key={positions[j].key}
              style={v == null ? undefined : {backgroundColor:'rgba(62,132,183,'+(v*.18)+')'}}>
              {v?.toFixed(3) ?? '零向量'}</td>)}
          </tr>)}</tbody></table>
      </div>
      <p className="v8-note">这些位置来自同一条输入或生成轨迹，仅描述该条记录。小样本下的高相似度需要用独立家族复核。</p>
      <details className="ow-provenance"><summary>计算口径与方法来源</summary>
        <p>线性 CKA 对两层的特征列分别按位置中心化，对所得 Gram 矩阵计算归一化内积。RSA 对两层余弦距离矩阵的全部上三角元素求 Spearman 秩相关，并列距离取平均秩。</p>
        <p>零向量的余弦距离缺失。中心化后没有变化时 CKA 未定义；距离缺失或距离秩恒定时 RSA 未定义。</p>
        <a href="https://proceedings.mlr.press/v97/kornblith19a.html" target="_blank" rel="noreferrer">线性 CKA 原论文</a>
        {' · '}<a href="https://doi.org/10.3389/neuro.06.004.2008" target="_blank" rel="noreferrer">RSA 原论文</a>
      </details>
    </>}
  </section>;
}

// Read only the selected native fields. For a dense trace, expose consecutive
// windows explicitly so a long prompt cannot issue thousands of GETs at once.
const WINDOW_SIZE = 12;

export function TraceRepresentationGeometry({base, traceId, snapshot, dimensions, positions = [], layer, layerCount,onPin}) {
  const [reference, setReference] = useState(null);
  const [page, setPage] = useState(0);
  const [retry, setRetry] = useState(0);
  const [stored, setStored] = useState(null);
  const width = typeof dimensions === 'number' ? dimensions : dimensions?.d_model;
  const other = comparisonLayer(reference, layer, layerCount);
  const pages = Math.max(1, Math.ceil(positions.length / WINDOW_SIZE));
  const currentPage = Math.min(page, pages-1);
  const start = Math.min(currentPage * WINDOW_SIZE, Math.max(0, positions.length-WINDOW_SIZE));
  const aligned = positions.slice(start, start+WINDOW_SIZE);
  const keys = positions.map(p => p.key);
  const invalidPositions = keys.some(p => !Number.isSafeInteger(p) || p<0) || new Set(keys).size!==keys.length;
  let invalid = '';
  if (base == null || !traceId || !snapshot || !Number.isInteger(width) || width < 1)
    invalid = '需要已核对模型快照、轨迹编号和原生维度的归档。';
  else if (!Number.isInteger(layerCount) || !Number.isInteger(layer) || layer<0 || layer>=layerCount || layerCount<2)
    invalid = '所选模型与读出层索引不匹配。';
  else if (invalidPositions || aligned.length < 3)
    invalid = '该轨迹至少需要三个具有不同 token 编号的已采集位置。';
  const positionKey = JSON.stringify(aligned.map(p => p.key));
  const identity = JSON.stringify([base,traceId,snapshot,width,layer,other,positionKey,retry]);
  useEffect(() => {
    if (invalid) return undefined;
    const controller = new AbortController();
    const selectedKeys = JSON.parse(positionKey);
    async function load() {
      const byLayer = new Map();
      for (const l of new Set([layer,other])) {
        const pairs = await Promise.all(selectedKeys.map(async p => {
          const route = '/api/traces/'+encodeURIComponent(traceId)+'/field?block='+l+'&position='+p+'&boundary=hidden_state';
          const d = await request(base, route, {signal:controller.signal});
          if (d.trace_id !== traceId || d.layer !== l || d.position !== p ||
              d.boundary !== (l===0?'embedding':'hidden_state') ||
              (d.model_snapshot != null && d.model_snapshot !== snapshot))
            throw Error('轨迹、模型快照或读出边界与请求不一致。');
          if (d.shape?.length !== 1 || d.shape[0] !== width || !Array.isArray(d.values) ||
              d.values.length !== width || !d.values.every(Number.isFinite))
            throw Error('读出位置的原生向量缺失、非有限或维度不匹配。');
          return [p, d.values];
        }));
        byLayer.set(l, new Map(pairs));
      }
      if (!controller.signal.aborted) setStored({identity, byLayer});
    }
    load().catch(error => {
      if (!controller.signal.aborted) {
        setStored({identity,error:error.message});
        controller.abort();
      }
    });
    return () => controller.abort();
  }, [base,traceId,snapshot,width,layer,other,positionKey,identity,invalid]);
  const current = !invalid && stored?.identity === identity ? stored : null;
  const reason = invalid || current?.error || (!current ? '正在读取所选两层的归档向量…' : '');
  return <>
    {pages > 1 && <label>配对位置范围 <select aria-label="表征几何位置范围" value={currentPage}
      onChange={event => setPage(Number(event.target.value))}>
      {Array.from({length:pages}, (_,i) => {
        const first = Math.min(i*WINDOW_SIZE, positions.length-WINDOW_SIZE);
        return <option key={i} value={i}>第 {first+1}–{first+WINDOW_SIZE} 处 / 共 {positions.length} 处</option>;
      })}
    </select></label>}
    <RepresentationGeometry positions={aligned} read={(p,l) => current?.byLayer?.get(l)?.get(p)}
      layer={layer} layerCount={layerCount} referenceLayer={reference} onReferenceChange={setReference}
      source={<span style={{overflowWrap:'anywhere'}}>轨迹 {traceId} · 模型快照 {snapshot}</span>}
      reason={reason} onPin={onPin}/>
    {current?.error && <button type="button" className="fw-tbtn" onClick={() => setRetry(n=>n+1)}>重新读取归档</button>}
  </>;
}
