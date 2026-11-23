import {getJson} from '../app/rdc_fusion/api.js';
import {useWorkspaceResource} from './nativeObservationArchive.js';
import { databaseHref } from './testDatabaseLocation.js';
import { workbenchHref } from './v8Util.js';
import './head_evidence.css';

export const WORKSPACE_MODES = [['live','实时模式'],['archive','测试记录'],['features','特征分析']];
const rules = ['按姓名', '按印刷标签', '按物理首行'];
export const EXAMPLE_SITES = ['输入末尾', '最后一步前', 'Value 前', 'Code 前'];
const percent = value => (value*100).toFixed(1)+'%';

export function selectionHead(data) {
  const run = data?.evidence?.runs?.[0];
  if(!run?.readout)return null;
  return {snapshot:run.snapshot,run:run.id,location:{layer:run.readout.layer+1,module:'attn',index:run.readout.head}};
}

const selectionCases = data => data?.evidence?.runs?.[0]?.families?.[0]?.cases?.filter(row=>!row.labelSwap&&!row.positionSwap) || [];

export function archiveHeadCase(data,caseId) {
  const examples = selectionCases(data);
  return examples.find(row=>row.id===caseId) || (!caseId ? examples.find(row=>row.rule===2) : null);
}

export function archivedHeadPrompt(data,caseId) {
  const run = data?.evidence?.runs?.[0], inputs = data?.prompts;
  return run&&inputs?.run===run.id&&inputs.snapshot===run.snapshot&&
    inputs.materialsSha256===run.provenance['confirm-materials.json'] ? inputs.prompts[caseId] : null;
}

export function ArchiveHeadCondition({data,caseId,onCase}) {
  const sample=archiveHeadCase(data,caseId);
  return <label>测试条件<select aria-label="归档样本" value={sample?.id||''} onChange={e=>onCase(e.target.value)}>
    {!sample&&<option value="">选择原题条件</option>}
    {selectionCases(data).map(row=><option key={row.id} value={row.id}>{rules[row.rule]}</option>)}
  </select></label>;
}

export function ArchiveHead({data,caseId,onCase,model,fullscreen}) {
  const headRun = data?.evidence?.runs?.[0];
  const sample = archiveHeadCase(data,caseId);
  if(!sample)return <p role="status">当前中心未提供该样本的归档。<button className="fw-tbtn" onClick={()=>onCase('')}>查看已归档条件</button></p>;
  const winner = sample.mass.indexOf(Math.max(...sample.mass));
  return <section className="ow-head" aria-label="B26/H36 归档注意力">
    <header><div><h3>B26/H36 关注了哪条记录？</h3><p>Phase 2983 · <code>{sample.id}</code> · 最后一个输入 token</p></div>
      <ArchiveHeadCondition data={data} caseId={sample.id} onCase={onCase}/></header>
    <div className="ow-head-grid">
      <div className="ow-masses">
        {sample.order.map((record,index)=><div key={record} data-gold={record===sample.gold} data-winner={record===winner}><span>第 {index+1} 行</span>
          <progress aria-label={`第 ${index+1} 行注意力质量${record===sample.gold?'，正确行':''}${record===winner?'，最高行':''}`} max={1} value={sample.mass[record]}/><b>{percent(sample.mass[record])}</b></div>)}
        <p className="ow-mass-key"><span className="gold">正确行</span> <span className="winner">最高行</span> <span>其余 token 合计 {percent(1-sample.mass.reduce((sum,value)=>sum+value,0))}</span></p>
      </div>
      <div className="ow-head-source"><h4>五条记录中，第 {sample.order.indexOf(winner)+1} 行权重最高</h4>
        <p>题目要求读第 {sample.order.indexOf(sample.gold)+1} 行。模型输出 <code>{sample.output}</code>，{sample.correct?'与正确答案一致。':'与正确答案不同。'}</p>
        <a className="fw-tbtn primary" href={databaseHref(headRun.id,{caseId:sample.id,fromHead:true,model,fullscreen})}>打开历史测试</a></div>
    </div>
  </section>;
}

export function ArchivedHeadCase({runId,caseId,model,fullscreen}) {
  const {data,error}=useWorkspaceResource('selection-cases');
  const headRun = data?.evidence?.runs?.[0];
  if(!headRun)return <p role="status">{error||'正在读取中心原题…'}</p>;
  const headFamily = headRun.families[0];
  const sample = runId===headRun.id && selectionCases(data).find(row=>row.id===caseId);
  if(!sample) return <p role="status">该运行与样本编号没有对应的 Head 归档。 <a href={databaseHref(runId)}>查看测试记录</a></p>;
  const returnQuery = new URLSearchParams({lens:'spatial',view:'graph',gview:'structure',mdl:model||headRun.model,
    snapshot:headRun.snapshot,sblock:String(headRun.readout.layer+1),smodule:'attn',sindex:String(headRun.readout.head),
    observeMode:'archive',observeTab:'plugin',observePlugin:'heads',headCase:sample.id});
  if(fullscreen)returnQuery.set('observeFullscreen','1');
  const nativeQuery=new URLSearchParams(returnQuery);
  for(const [key,value] of Object.entries({observeFullscreen:'1',observeField:'1',observeArchive:'head',run:headRun.id}))nativeQuery.set(key,value);
  const liveQuery=new URLSearchParams(returnQuery);
  liveQuery.set('observeMode','live'); liveQuery.set('observeTab','run'); liveQuery.set('liveCase',sample.id);
  const prompt = archivedHeadPrompt(data,sample.id);
  return <section className="head-record" aria-label="历史测试原题">
    <header><div><h2>Phase 2983 · 三线索选源</h2><p><code>{sample.id}</code> · {rules[sample.rule]} · {headRun.model} / {headRun.precision} · B26/H36</p></div>
      <div><a className="fw-tbtn" href={workbenchHref(nativeQuery.toString())}>回放本题原生读数</a>
        {prompt&&<a className="fw-tbtn" href={workbenchHref(liveQuery.toString())}>本机原样复跑</a>}
        <a className="fw-tbtn" href={workbenchHref(returnQuery.toString())}>返回分析</a></div></header>
    <p lang="en" className="head-record-question">{sample.question}</p>
    <div className="head-record-body">
      <table><caption>原题中的五条记录</caption><thead><tr>{['物理行','标签','姓名','代码','注意力'].map(name=><th scope="col" key={name}>{name}</th>)}</tr></thead><tbody>
        {sample.order.map((record,index)=><tr key={record} data-gold={record===sample.gold}>
          <th scope="row">{index+1}</th><td>{sample.labels[record]}</td><td>{headFamily.names[record]}</td>
          <td><code>{headFamily.codes[record]}</code></td><td>{percent(sample.mass[record])}</td></tr>)}
      </tbody></table>
      <div><h3>本题输出 <code>{sample.output}</code></h3><p>正确记录为第 {sample.order.indexOf(sample.gold)+1} 行。注意力取自最后一个输入 token，对各行原文范围求和。</p>
        <details><summary>原始模型输入</summary>{prompt?<pre lang="en">{prompt}</pre>:<p>当前归档缺少可核对的原始输入。</p>}</details>
        <details><summary>运行与快照</summary><dl><dt>运行</dt><dd>{headRun.id}</dd><dt>模型快照</dt><dd>{headRun.snapshot}</dd><dt>确认材料 SHA-256</dt><dd>{headRun.provenance['confirm-materials.json']}</dd></dl></details>
        <a href={databaseHref(headRun.id,{caseId:sample.id})}>查看完整测试</a>
      </div>
    </div>
  </section>;
}

let pendingExample;
export function loadGraphExample(resource) {
  if(!resource)return Promise.reject(Error('当前中心没有可用的多位置样例。'));
  if(pendingExample?.id===resource)return pendingExample.request;
  const request=getJson('/api/research/workspace/resources/'+encodeURIComponent(resource)).then(sample=>{
    const decode = encoded => {
      const bytes = Uint8Array.from(atob(encoded), character=>character.charCodeAt(0));
      const reader = new DataView(bytes.buffer);
      const result = new Float32Array(bytes.length/2);
      const words = new Uint32Array(result.buffer);
      for(let index=0;index<words.length;index++)words[index]=reader.getUint16(index*2,true)<<16;
      return result;
    };
    const {hidden,mlp,...metadata}=sample;
    const hs=decode(hidden),activations=decode(mlp);
    const [layers,positions,hiddenSize]=sample.hiddenShape;
    const [,mlpSize]=sample.mlpShape;
    if(hs.length!==layers*positions*hiddenSize || activations.length!==positions*mlpSize)throw Error('样例的归档维度校验失败。');
    return {...metadata,field(kind,layer,position){
      if(!Number.isInteger(position)||position<0||position>=positions||layer<0||layer>=layers)return null;
      if(kind==='hs'){const offset=(layer*positions+position)*hiddenSize;return hs.subarray(offset,offset+hiddenSize);}
      if(kind==='mlp'&&layer===sample.block+1){const offset=position*mlpSize;return activations.subarray(offset,offset+mlpSize);}
      return null;
    }};
  }).catch(error=>{pendingExample=null;throw error;});
  pendingExample={id:resource,request};
  return request;
}

export function ExamplePositions({sample,site,onSite}) {
  return <div className="ow-step-list" role="group" aria-label="采集位置">
    {sample.positions.map((entry,index)=><button key={entry.site} className="fw-tbtn" aria-pressed={site===index}
      title={`Token ${entry.index} · ID ${entry.tokenId}`} onClick={()=>onSite(index)}>{EXAMPLE_SITES[index]}</button>)}
  </div>;
}

export function ExampleSource({sample}) {
  return <details className="ow-provenance"><summary>样例的数据来源</summary>
    <p>Phase {sample.phase} · {sample.run} · {sample.caseId}</p>
    <p>模型快照 {sample.snapshot}</p><p>原始文件 SHA-256 {sample.source.sha256}</p>
    <pre>{sample.prefix}</pre>
  </details>;
}

export function NeuronExample({sample,kind,onKind,unit,onLocate,onGroup}) {
  const types=[['unit','原生神经元'],['group','原生单元组'],['sae','稀疏字典特征']];
  const stableUnit=unit ?? sample?.units[0];
  return <section className="ow-mode-panel" aria-label="特征分析工作区">
    <div className="ow-section-head"><div><h3>特征分析</h3><p>当前模型的原生索引和已有证据</p></div></div>
    <div className="ow-step-list" role="group" aria-label="特征类型">
      {types.map(([value,name])=><button key={value} className="fw-tbtn" aria-pressed={kind===value} onClick={()=>onKind(value)}>{name}</button>)}
    </div>
    {kind==='sae'?<p>当前工作区尚未接入与此快照绑定的 SAE 字典。</p>:
      !sample?<p>当前模型无随包样例。在 3D 点选单元，可查询相应的特征、参数与研究档案。</p>:
      kind==='unit'?<>
        <div className="ow-feature-entry"><div><h4>B{sample.block}/U{stableUnit}</h4><p>Phase {sample.phase} · 原生 MLP 单元</p></div>
          <button className="fw-tbtn" onClick={()=>onLocate(stableUnit)}>在 3D 中定位</button></div>
        <table className="ow-value-table"><thead><tr><th>采集位置</th><th>激活值</th></tr></thead><tbody>
          {sample.positions.map((entry,index)=><tr key={entry.site}><td>{EXAMPLE_SITES[index]}</td>
            <td>{sample.field('mlp',sample.block+1,index)[stableUnit]?.toFixed(4)??'未采集'}</td></tr>)}
        </tbody></table><p>这是单题观测，单元语义与独立因果作用仍待验证。</p><ExampleSource sample={sample}/>
      </>:<>
        <div className="ow-feature-entry"><div><h4>B{sample.block} 固定 {sample.units.length} 个 MLP 单元</h4>
          <p>Phase {sample.phase} · 冻结原生单元组</p></div><button className="fw-tbtn" onClick={onGroup}>在 3D 中标出整组</button></div>
        <dl className="ow-feature-results">{sample.comparisons.map(row=><div key={row.arm}>
          <dt>{row.arm==='mask0'?'原模型':'三段位置联合屏蔽'}</dt><dd>{row.joint_family_correct} / {row.families} 家族正确</dd>
        </div>)}</dl>
        <p>本次 Code 读出正确率未提供该组必要性的证据。</p>
        <details className="ow-provenance"><summary>原生单元编号</summary>
          <div className="ow-step-list">{sample.units.map(index=><button className="fw-tbtn" key={index}
            onClick={()=>{onKind('unit');onLocate(index);}}>U{index}</button>)}</div>
        </details><ExampleSource sample={sample}/>
      </>}
  </section>;
}
