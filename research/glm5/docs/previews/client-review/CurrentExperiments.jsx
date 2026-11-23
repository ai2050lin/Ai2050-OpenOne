import {useEffect,useRef,useState} from 'react';

const models=['Qwen3-14B','GLM-4-9B','Qwen3-4B'];
// Drafts survive navigation inside this file. Explicit export/import carries them across reloads.
const drafts=new Map();
const schema='ai2050.preview.experiment-draft.v1';
const examples=[
  {id:'example-archive',question:'颜色判断的归档复核',status:'failed',model:models[0]},
  {id:'example-running',question:'五条记录的选源复核',status:'running',model:models[0]},
  {id:'example-draft',question:'更换姓名后，B26/H36 还能选中目标行吗？',status:'draft',model:models[0]}
];
const stages={draft:['草稿','继续设计'],preflight:['待中心查重','核对方案'],running:['运行中','看进度'],failed:['归档失败','查看原因']};
const rank={failed:0,running:1,preflight:2,draft:3};
const blank=()=>({id:crypto.randomUUID(),question:'',model:models[0],material:'',materialText:'',plan:'',status:'draft',updated:Date.now()});
const sources=()=>window.__evidenceData.experimentSources;
const norm=s=>s.toLowerCase().replace(/[\s/_-]+/g,'');
function similar(question,model){
  const q=norm(question);
  return sources().map(s=>{
    const words=[String(s.phase),s.title,...s.terms].filter(term=>q.includes(norm(term)));
    const score=words.reduce((n,t)=>n+Math.min(8,t.length),0);
    return {s,words,score};
  }).filter(r=>r.score>0).sort((a,b)=>b.score-a.score||Number(b.s.model===model)-Number(a.s.model===model));
}
function complete(d){return !!(d.question.trim()&&d.material&&(d.material!=='custom'||d.materialText.trim())&&d.plan.trim());}
function download(d){
  const file={schema,draft:d,preflight_receipt:null,material_source:sources().find(s=>s.key===d.material)?.file||null};
  const a=document.createElement('a');a.href='data:application/json;charset=utf-8,'+encodeURIComponent(JSON.stringify(file,null,2));
  a.download='AI2050-实验草稿-'+d.id.slice(0,8)+'.json';a.click();
}
function parseDraft(text){
  const doc=JSON.parse(text),d=doc.draft;
  if(doc.schema!==schema||!d||typeof d.id!=='string'||!/^[\da-f]{8}-(?:[\da-f]{4}-){3}[\da-f]{12}$/i.test(d.id)||!models.includes(d.model)||
    ![d.question,d.material,d.materialText,d.plan].every(s=>typeof s==='string')||!d.question.trim()||d.question.length>300||
    d.materialText.length>2000||d.plan.length>4000||!['','custom',...sources().map(s=>s.key)].includes(d.material))throw Error('format');
  return {...blank(),id:d.id,question:d.question,model:d.model,material:d.material,materialText:d.materialText,plan:d.plan,
    status:d.status==='preflight'&&complete(d)?'preflight':'draft'};
}
function Modal({title,close,children}){
  const ref=useRef(null);
  useEffect(()=>{const el=ref.current;el.showModal();return ()=>el.close();},[]);
  return <dialog ref={ref} className="cx-modal" aria-labelledby="cx-modal-heading" onCancel={e=>{e.preventDefault();close();}}>
    <header><h2 id="cx-modal-heading">{title}</h2><button type="button" onClick={close}>关闭</button></header>{children}</dialog>;
}
function DraftEditor({initial,initialStep=1,onSave,onClose}){
  const [d,update]=useState(initial),[step,setStep]=useState(initialStep),[error,setError]=useState('');
  const patch=value=>update(prev=>({...prev,...value})),changeStep=n=>{setError('');setStep(n);};
  const matches=similar(d.question,d.model), selected=sources().find(s=>s.key===d.material);
  async function readFile(e){
    const input=e.target,file=input.files[0];if(!file)return;
    try{
      if(file.size>100000)throw Error('size');
      const imported=parseDraft(await file.text());
      update(imported);changeStep(1);setError('已载入草稿，可继续编辑。');
    }catch{setError('无法读取该草稿。请选择从此页面导出的实验草稿 JSON。');}
    input.value='';
  }
  return <Modal title={drafts.has(d.id)?'编辑实验':'新建实验'} close={onClose}>
    <div className="cx-step"><span>{step} / 3</span><strong>{['研究问题','查相似测试','材料与方案'][step-1]}</strong></div>
    {step===1&&<>
      <label>想验证什么问题？<textarea rows={3} maxLength={300} value={d.question} onChange={e=>patch({question:e.target.value})} placeholder="例如，更换姓名后，B26/H36 还能选中目标行吗？"/></label>
      <label>拟用模型<select value={d.model} onChange={e=>patch({model:e.target.value})}>{models.map(m=><option key={m}>{m}</option>)}</select></label>
      <details><summary>草稿文件</summary><label>导入草稿 JSON<input type="file" accept=".json,application/json" onChange={readFile}/></label>
        {d.question.trim()&&<button onClick={()=>download({...d,status:'draft'})}>导出草稿</button>}</details>
    </>}
    {step===2&&<>
      <p className="cx-context">{d.question} <small>· {d.model}</small></p>
      <p>已查本地快照的 {sources().length} 项测试，{matches.length? '有 '+matches.length+' 项包含相关词。':'未找到相同的关键词。'}</p>
      {matches.map(({s,words})=><article key={s.phase} className="cx-match"><div><strong>{s.title}</strong><small>Phase {s.phase} · {s.model} · 命中 {words.join('、')}</small></div>
        <button onClick={()=>window.openTestDatabase({q:String(s.phase)},s.phase)}>复用已有结果</button></article>)}
      <p className="cx-note">这里查询已收录的历史记录。正式登记前仍须向中心查重并取得回执。</p>
    </>}
    {step===3&&<>
      <label>选择测试材料<select value={d.material} onChange={e=>patch({material:e.target.value})}><option value="">请选择</option><option value="custom">自行设计新材料</option>
        <optgroup label="已开放材料，可用于复核">{sources().map(s=><option key={s.key} value={s.key}>Phase {s.phase} · {s.materialName}</option>)}</optgroup></select></label>
      {selected&&<div className="cx-note"><a href={selected.file} target="_blank" rel="noreferrer">打开材料或冻结方案 ↗</a> · 已开放材料用于复核；独立确认需要另备未见家族。</div>}
      {d.material==='custom'&&<label>材料来源或设计<textarea rows={2} maxLength={2000} value={d.materialText} onChange={e=>patch({materialText:e.target.value})}/></label>}
      <label>新增价值与检验方法<textarea rows={3} maxLength={4000} value={d.plan} onChange={e=>patch({plan:e.target.value})} placeholder="准备改变哪个条件？与哪组对照比较？"/></label>
      <p className="cx-note">保存后会列入「待中心查重」。当前离线稿仅保存本地方案。</p>
    </>}
    {error&&<p role="status">{error}</p>}
    <footer>{step>1?<button onClick={()=>changeStep(step-1)}>上一步</button>:<span/>}<div><button disabled={!d.question.trim()} onClick={()=>onSave(d,'draft')}>暂存草稿</button>{step<3?
      <button className="cx-primary" disabled={!d.question.trim()} onClick={()=>changeStep(step+1)}>{step===1?'查相似测试':'继续设计新实验'}</button>:
      <button className="cx-primary" disabled={!complete(d)} onClick={()=>onSave(d,'preflight')}>保存方案</button>}</div></footer>
  </Modal>;
}
function ExperimentDetail({row,demo,onEdit,onClose,onStation}){
  const local=!demo,material=sources().find(s=>s.key===row.material);
  const status=stages[row.status][0],running=row.status==='running',failed=row.status==='failed';
  return <Modal title={failed?'归档失败原因':running?'实验进度':'实验方案'} close={onClose}>
    <div className="cx-context"><strong>{row.question}</strong><small>{row.model} · {status}{demo?' · 界面示例':''}</small></div>
    {local?<><dl><dt>测试材料</dt><dd>{material?<a href={material.file} target="_blank" rel="noreferrer">{material.materialName} ↗</a>:row.materialText||'待选择'}</dd>
      <dt>检验方法</dt><dd>{row.plan||'待补充'}</dd></dl><p className="cx-note">下一步需通过协议向中心查重。获得回执并登记后，才能启动正式测试。</p></>:
      <p>{failed?'示例中的文件上传中断，中心尚未返回读回回执。本地材料应保留，恢复传输后再检查回执。':'示例中的材料已冻结，节点正在逐批采集。采集结束后进入结果校验与协议归档。'}</p>}
    <details><summary>队列与运行日志</summary><p>{local?'本地方案尚未登记，没有任务编号或运行日志。':failed?'示例日志：采集完成 → 结果校验完成 → 上传中断。':'示例日志：任务已领取 → 材料校验完成 → 正在采集。'}</p></details>
    <details><summary>审计与归档回执</summary><p>{local?'中心查重回执、正式运行和归档回执尚未取得。':failed?'示例状态：等待读回核验，归档未完成。':'示例状态：执行中，结果审阅和归档尚未开始。'}</p></details>
    <footer><button onClick={()=>onStation('queue')}>查看完整队列</button>{local&&<div><button onClick={()=>download(row)}>导出草稿</button><button className="cx-primary" onClick={onEdit}>修改方案</button></div>}</footer>
  </Modal>;
}
export default function CurrentExperiments({onStation}){
  const [list,setList]=useState(()=>[...drafts.values()]),[mode,setMode]=useState('local'),[editor,setEditor]=useState(null),[detail,setDetail]=useState(null);
  const demo=mode==='example';
  const rows=demo?examples:[...list].sort((a,b)=>rank[a.status]-rank[b.status]||b.updated-a.updated);
  function save(d,status){
    const updated={...d,question:d.question.trim(),status,updated:Date.now()};
    drafts.set(updated.id,updated);setList([...drafts.values()]);setEditor(null);setMode('local');
  }
  function act(r){
    if(r.status==='draft'){setEditor({d:demo?{...blank(),question:r.question}:r,step:1});}
    else setDetail(r);
  }
  return <section className="cx-page" aria-labelledby="cx-title">
    <header className="cx-head"><h1 id="cx-title">实验管理</h1><button className="cx-primary" onClick={()=>setEditor({d:blank(),step:1})}>新建实验</button></header>
    <section className="cx-table-card" aria-labelledby="cx-current"><header><h2 id="cx-current">当前实验</h2>
      <span className={'cx-source '+(demo?'example':'')}>{demo?'状态示例 · 仅供预览':'离线快照'}</span></header>
      <table><thead><tr><th>实验名称</th><th>当前状态</th><th>下一步操作</th></tr></thead>
      <tbody>{rows.map(r=><tr key={r.id}><td><strong>{r.question}</strong><small>{r.model}</small></td>
        <td><span className={'cx-status '+r.status}>{stages[r.status][0]}</span></td><td><button className="cx-action" onClick={()=>act(r)}>{stages[r.status][1]} →</button></td></tr>)}</tbody></table>
      {!rows.length&&<div className="cx-empty"><strong>当前实验状态未知</strong><p>尚未连接中心队列。你可以先新建实验，在本页准备草稿。</p></div>}
      <div className="cx-table-foot"><span>{demo?'需处理的异常优先显示。':rows.length?'仅列本页草稿，中心运行状态未知。':''}</span>
        <button className="cx-action" onClick={()=>setMode(demo?'local':'example')}>{demo?'返回本地清单':'预览状态示例'}</button></div>
    </section>
    <nav className="cx-bottom" aria-label="实验记录与队列"><button onClick={()=>window.openTestDatabase({archive:'本稿附中心回执'})}>查看已归档实验 →</button><button onClick={()=>onStation('queue')}>查看完整队列</button></nav>
    {!demo&&rows.length>0&&<p className="cx-note">本页草稿在切换菜单时保留。刷新或关闭前请导出，下次可在「新建实验」中导入。</p>}
    {editor&&<DraftEditor key={editor.d.id} initial={editor.d} initialStep={editor.step} onClose={()=>setEditor(null)} onSave={save}/>}
    {detail&&<ExperimentDetail row={detail} demo={demo} onEdit={()=>{setEditor({d:detail,step:3});setDetail(null);}} onClose={()=>setDetail(null)} onStation={onStation}/>}
  </section>;
}
