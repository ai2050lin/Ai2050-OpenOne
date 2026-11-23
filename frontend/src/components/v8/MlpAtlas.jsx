import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';

const metrics = {
  std:'激活标准差', mean:'平均激活', max:'最大激活', min:'最小激活',
};
const display = value => Number.isFinite(value) ? String(Number(value.toPrecision(7))) : '未采集';

export function useMlpAtlas({active, snapshot, block, width, unit}) {
  const [metric, setMetric] = useState('std');
  const [preferred, setPreferred] = useState('');
  const [pole, setPole] = useState(null);
  const root = active && snapshot ? '/api/research/workspace/models/'+encodeURIComponent(snapshot)+'/mlp-atlas' : null;
  const catalog = useApi(root);
  const known = catalog.data?.schema==='workspace_mlp_atlas.v1' && catalog.data.snapshot===snapshot;
  const runs = known ? catalog.data.runs.filter(run=>run.width===width) : [];
  const inBlock = runs.filter(run=>run.blocks.includes(block));
  const run = inBlock.find(item=>item.run===preferred) || inBlock[0];
  const identity = run ? [snapshot, block, run.run, run.revision].join(':') : '';
  const query = run ? '?run='+encodeURIComponent(run.run)+'&revision='+encodeURIComponent(run.revision) : '';
  const layer = useApi(run ? root+'/'+block+query+'&metric='+metric : null);
  const card = useApi(run && Number.isInteger(unit) ? root+'/'+block+'/units/'+unit+query : null);
  const showPole = pole && pole.identity===identity;
  const field = useApi(showPole ? root+'/'+block+'/field'+query+'&sample='+pole.sample : null);
  const match = (data,schema) => data?.schema===schema && data.snapshot===snapshot && data.block===block &&
    data.run===run?.run && data.revision===run.revision && data.width===width;
  const layerMatches = match(layer.data,'workspace_mlp_layer.v1') && layer.data.metric===metric;
  const fieldMatches = match(field.data,'workspace_mlp_field.v1') && showPole && field.data.sample===pole.sample &&
    field.data.document===pole.document && field.data.position===pole.position && field.data.token_id===pole.token_id;
  const vector = showPole ? (fieldMatches ? field.data.values : null) : (layerMatches ? layer.data.values : null);
  const values = vector?.length===width && vector.every(Number.isFinite) ? vector : null;
  const title = showPole ? '门控后激活 A' : metrics[metric];
  const source = showPole ? pole.document+' · token '+pole.position+'（ID '+pole.token_id+'）' : '自然语料固定读出';
  const vectorError = showPole ? field.error : layer.error;
  const vectorLoading = showPole ? field.loading : layer.loading;
  return {active:Boolean(root),run,inBlock, known, catalog,
    blocks:[...new Set(runs.flatMap(item=>item.blocks))].sort((a,b)=>a-b),
    values,title,source,metric,signed:showPole||metric!=='std', identity,
    isPole:Boolean(showPole), local:catalog.local||layer.local||card.local||field.local,
    missing:catalog.error||(!known ? '正在读取 MLP 普查目录…' : !run ? '当前模型块没有可读取的 MLP 普查。' :
      vectorError||(!values ? vectorLoading ? '正在读取整层数据…' : '读数与当前档案身份或维度不一致。' : '')),
    selectMetric:value=>{setMetric(value);setPole(null);},
    selectRun:value=>{setPreferred(value);setPole(null);},
    selectPole:sample=>setPole({...sample,identity}),
    card:match(card.data,'workspace_mlp_unit.v1')&&card.data.unit===unit ? card.data : null,
    cardError:card.error||(card.data&&(!match(card.data,'workspace_mlp_unit.v1')||card.data.unit!==unit)?'单元报告与当前档案身份不一致。':''),
    cardLoading:card.loading,
    reload:()=>{catalog.reload();layer.reload();card.reload();field.reload();},
  };
}

export function MlpAtlasControls({atlas, block, onBlock, scale, unit, onLock, locked}) {
  return <div className="ow-heat-controls ow-mlp-atlas-controls" role="group" aria-label="MLP 神经元普查">
    <strong>MLP 神经元普查</strong>
    {!!atlas.blocks.length&&<>
      <label>普查块<select aria-label="普查模型块" value={atlas.run?block:''} onChange={e=>onBlock(Number(e.target.value))}>
        {!atlas.run&&<option value="">选择模型块</option>}
        {atlas.blocks.map(item=><option value={item} key={item}>B{item}</option>)}
      </select></label>
      <label>MLP 着色<select aria-label="普查着色量" value={atlas.isPole?'position':atlas.metric} onChange={e=>atlas.selectMetric(e.target.value)}>
        {Object.entries(metrics).map(([id,title])=><option key={id} value={id}>{title}</option>)}
        {atlas.isPole&&<option value="position" disabled>选中语料位置的激活</option>}
      </select></label>
    </>}
    {atlas.inBlock.length>1&&<label>归档运行<select aria-label="普查归档运行" value={atlas.run.run} onChange={e=>atlas.selectRun(e.target.value)}>
      {atlas.inBlock.map(item=><option value={item.run} key={item.run}>B{item.blocks[0]}–B{item.blocks.at(-1)}</option>)}
    </select></label>}
    {atlas.values&&<>
      <span>{atlas.values.length.toLocaleString('zh-CN')} 个单元</span>
      {Number.isFinite(scale)&&<span>{atlas.signed?'±':'0–'}{display(scale)}</span>}
      {onLock&&<button type="button" className="fw-tbtn" aria-pressed={locked} onClick={onLock}>锁定普查色标</button>}
      {Number.isInteger(unit)&&<span>B{block}/U{unit} · {atlas.title} <output aria-label="普查神经元原值">{atlas.values[unit]}</output></span>}
      {atlas.isPole&&<span>{atlas.source}</span>}
    </>}
    {atlas.missing&&<span role="status">{atlas.missing}</span>}
    {atlas.local&&<span>已缓存档案</span>}
    {(atlas.catalog.error||atlas.cardError||atlas.missing&&atlas.run&&!atlas.values)&&
      <button type="button" className="fw-tbtn" onClick={atlas.reload}>重新读取</button>}
  </div>;
}

export function MlpAtlasReport({atlas, block, unit, onShowPole}) {
  const card = atlas.card;
  if(!atlas.run)return null;
  const candidate = card?.candidate;
  return <section className="ow-mlp-atlas-report" aria-label="Goodfire 神经元分析">
    <h3>{unit!=null?`B${block}/U${unit} · `:''}Goodfire 风格神经元分析</h3>
    {!card?<p role="status">{atlas.cardError||(unit==null?'点击 3D 网格中的单元，或输入 B/U 编号。':'正在读取这个单元的语料档案…')}</p>:<>
      <p>{card.document_count.toLocaleString('zh-CN')} 篇自然文本 · 每篇 {card.readouts_per_document} 个读出 · A = SiLU(G) × V</p>
      <p>激活均值 <b>{display(card.stats.A.mean)}</b> · 标准差 <b>{display(card.stats.A.std)}</b>
        {' · '}观察等级 <b>{candidate.semantic_confirmed?'语义已确认':'候选语境关联'}</b></p>
      <p>正端候选词 {candidate.positive_keywords?.join('、')||'暂无'} · 负端候选词 {candidate.negative_keywords?.join('、')||'暂无'}</p>
      <p>确认集命中 <b>{candidate.repeat.confirm.hits}/{candidate.repeat.confirm.n}</b>
        {' · '}留出集命中 <b>{candidate.repeat.holdout.hits}/{candidate.repeat.holdout.n}</b></p>
      <div className="ow-mlp-atlas-contexts">{card.poles.map(pole=><article key={pole.pole}>
        <strong>{pole.pole==='positive'?'留出集高响应':'留出集低响应'}</strong> <output>{pole.value}</output>
        <p>{pole.text}</p><small>{pole.document} · token {pole.position}</small>
        <button type="button" className="fw-tbtn" onClick={()=>onShowPole(pole)}>在 3D 显示此位置</button>
      </article>)}</div>
      <details><summary>档案身份与 G/V 统计</summary>
        <p>归档运行 <code>{card.run}</code> · 原始采集 <code>{card.source_run}</code></p>
        <p>模型快照 <code>{card.snapshot}</code> · 清单 SHA-256 <code>{card.revision}</code></p>
        {['G','V'].map(field=><p key={field}>{field} · 均值 {display(card.stats[field].mean)} · 标准差 {display(card.stats[field].std)}
          {' · '}范围 [{display(card.stats[field].min)}, {display(card.stats[field].max)}]</p>)}
      </details>
    </>}
  </section>;
}
