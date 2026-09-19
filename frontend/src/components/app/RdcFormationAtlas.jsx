import {useEffect,useState} from 'react';
import {Field} from './RdcJointAtlas.jsx';
import FormationEvidence from './RdcFormationEvidence.jsx';

const API=import.meta.env.VITE_RDC_CONSTRUCTION_API||'http://127.0.0.1:5004/api/rdc-construction';
const n=v=>v==null?'尚未提交':Number(v).toLocaleString('zh-CN');
const f=v=>v==null?'尚未提交':Number(v).toFixed(6);
const labels={true_token:'真实 token',within_surface_class_permuted_token:'同表面类别内目标置乱',surface_class_mass:'表面类别总概率'};
async function get(path,params={}){const r=await fetch(API+path+'?'+new URLSearchParams(params));if(!r.ok)throw new Error((await r.json()).detail||r.statusText);return r.json();}

export default function FormationAtlas({summary,refresh}){
  const data=summary?.phase2747;
  const [samples,setSamples]=useState([]),[sample,setSample]=useState(''),[run,setRun]=useState('native');
  const [checkpoint,setCheckpoint]=useState('1'),[field,setField]=useState('hidden'),[view,setView]=useState('raw');
  const [loaded,setLoaded]=useState(null),[error,setError]=useState('');
  useEffect(()=>{let alive=true;get('/formation-samples').then(v=>alive&&setSamples(v)).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[refresh]);
  const chosen=samples.some(r=>r.sample_id===sample)?sample:samples[0]?.sample_id||'';
  const id=JSON.stringify([chosen,run,checkpoint,field,view]);
  const runs=data?.runs||[];
  const current=loaded?.id===id?loaded.value:null;
  async function load(){try{setLoaded({id,value:await get('/formation-field',{sample:chosen,run,checkpoint,field,view})});setError('');}catch(e){setLoaded(null);setError(e.message);}}
  return <section id="construction-formation"><h2>Phase 2747：真实参数学习形成</h2>
    <p>直接训练 block16 的全部 74,711,040 个原生 gate/up/down 参数，不使用低秩适配或坐标筛选。其余参数固定；三个监督条件、两个固定种子共享对应输入顺序。每次完整使用 {n(data?.material?.counts?.train)} 条训练记录，检查点完成不代表语言机制已破解。</p>
    <p>已完成训练：{runs.filter(r=>r.complete).length} / {runs.length||6}。新来源确认：{n(data?.material?.counts?.fresh)} 篇中文文档；当前库存无符合排除条件的新英文来源。词频先验来自 {n(data?.material?.prior?.source_groups)} 篇训练侧文档、{n(data?.material?.prior?.token_occurrences)} 次 token 出现，重叠窗口不算独立样本。</p>
    <p className="prefix-warning">“表面类别总概率”只是有限的解码表面规则，含词形和任务线索，不能称为纯格式或语义已移除。FP32 训练桥接、原生 BF16 部署与自由生成另行比较。</p>
    <div className="prefix-table"><table><thead><tr><th>监督</th><th>种子</th><th>已提交步数</th><th>实际消费样本</th><th>最终 BF16 位移</th><th>训练状态</th></tr></thead><tbody>{runs.map(r=><tr key={r.run}><td>{labels[r.condition]||r.condition}</td><td>{r.seed}</td><td>{r.latest_committed_step} / 128</td><td>{n(r.distinct_drawn_examples)}</td><td>{f(r.deployed_BF16_delta_L2)}</td><td>{r.complete?'训练已完成':r.latest_committed_step?'运行中':'待提交'}</td></tr>)}</tbody></table></div>
    <h3>同一大阶段的七项交付</h3>
    <ul>{(data?.tasks||[]).map(task=><li key={task.name}>{task.name}：{task.complete?'证据已核验':'尚未整体完成'}（已提交 {task.committed} / {task.total}）</li>)}</ul>
    <p>计数是各项自己的任务单位，不能相加当作独立样本数；准备、先导或单个训练完成不等于整个 Phase 完成。</p>
    {data?.recovery?.change&&<p>保存核验发现极小参数的 FP32 差值重建余项。已保存全坐标无损余项并继续逐位核对；失败记录保留，未放宽相等判据。</p>}
    <h3>同一表达：训练前后全部坐标与单元</h3>
    <div className="prefix-controls"><label className="prefix-wide">形成检验表达<select aria-label="形成检验表达" value={chosen} onChange={e=>setSample(e.target.value)}>{samples.map(r=><option key={r.sample_id} value={r.sample_id}>{r.family} · {r.language} · {r.sample_id}</option>)}</select></label>
      <label>形成参数条件<select aria-label="形成参数条件" value={run} onChange={e=>setRun(e.target.value)}><option value="native">原生 BF16</option><option value="bridge">FP32 MLP 桥接基线</option>{runs.map(r=><option key={r.run} value={r.run}>{labels[r.condition]} · {r.seed}</option>)}</select></label>
      <label>形成检查点<select aria-label="形成检查点" value={checkpoint} onChange={e=>setCheckpoint(e.target.value)}>{['1','8','32','128','deployed_BF16'].map(s=><option key={s}>{s}</option>)}</select></label>
      <label>形成原生场<select aria-label="形成原生场" value={field} onChange={e=>setField(e.target.value)}>{['hidden','gate','up','product','Q','MLP_input','MLP_write','postnorm_BF16'].map(s=><option key={s}>{s}</option>)}</select></label>
      <label>形成数值视图<select aria-label="形成数值视图" value={view} onChange={e=>setView(e.target.value)}><option value="raw">原始值</option><option value="RMS">完整向量 RMS</option></select></label>
      <button disabled={!chosen} onClick={load}>读取形成全坐标场</button></div>
    {error&&<p role="alert" className="prefix-error">{error}</p>}
    {current&&<><blockquote>{current.material.original_text}</blockquote><p>{current.axes}</p><p>{current.boundary}</p></>}
    <Field data={current}/>
    <FormationEvidence data={data} refresh={refresh}/>
    <details><summary>完整冻结协议、资源先导、恢复记录与已提交分析</summary><pre>{JSON.stringify(data||{},null,2)}</pre></details>
  </section>;
}
