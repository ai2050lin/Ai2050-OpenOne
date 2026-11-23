import {useEffect,useRef,useState,useSyncExternalStore} from 'react';

// This preview has no node bridge. Its timer belongs to the session, so visiting the atlas
// neither starts a job nor resets an explicitly started rehearsal.
const phases=['领取任务','本机测试','上传结果','中心汇总'];
const labels={idle:'尚未参与',claim:'匹配并领取',run:'正在测试',upload:'正在上传',verify:'中心核验',done:'已汇总',empty:'暂无匹配任务',model:'模型待准备',retry:'等待续传',stopped:'已停止'};
const stageIndex={claim:0,run:1,upload:2,retry:2,verify:3,done:4};
const scenarios={normal:'正常完成',empty:'没有匹配任务',model:'缺少模型',network:'上传中断一次'};
const waiting=new Set(['claim','run','upload','verify']);
let session={phase:'idle',scenario:'normal'},timer;
const listeners=new Set();
const subscribe=fn=>{listeners.add(fn);return ()=>listeners.delete(fn);};
function move(phase,scenario=session.scenario,retried=false){
  clearTimeout(timer);session={phase,scenario};listeners.forEach(fn=>fn());
  const next={claim:scenario==='empty'?'empty':scenario==='model'?'model':'run',run:'upload',upload:scenario==='network'&&!retried?'retry':'verify',verify:'done'}[phase];
  if(next)timer=setTimeout(()=>move(next,scenario),{claim:1200,run:3500,upload:1600,verify:1300}[phase]);
}
const studies=()=>window.__evidenceData.studyDownloads;
function atlas(id){
  const url=new URL(window.reviewRoute.href),q=url.searchParams;
  for(const k of ['run','snapshot','trace','context_trace','tm','ds','sblock','smodule','sindex','smode','reviewFocus','scoord','strace','straceb','observeTab','observePlugin'])q.delete(k);
  const study=studies().find(p=>p.phase===id);
  Object.entries({lens:'spatial',view:'graph',gview:'internal',mdl:'Qwen3-14B',observeMode:'archive',studyPack:study?String(id):'browse'}).forEach(([k,v])=>q.set(k,v));
  if(study){
    q.set('run',study.run);q.set('snapshot',study.snapshot);q.set('sblock','29');
    q.set('smodule',study.phase===2997?'attn':'mlp');
    if(study.phase===2997)q.set('sindex','21');
  }
  window.reviewHistory.pushState(null,'',url);window.dispatchEvent(new PopStateEvent('popstate'));
  requestAnimationFrame(()=>document.querySelector('.wb-main')?.scrollTo(0,0));
}
function download(name,type,contents){
  const url=URL.createObjectURL(new Blob([contents],{type})),a=document.createElement('a');
  a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);
}
function Dialog({title,close,children}){
  const ref=useRef(null);
  useEffect(()=>{const el=ref.current;el.showModal();return ()=>el.close();},[]);
  return <dialog ref={ref} className="cx-modal dp-dialog" aria-labelledby="dp-dialog-title" onCancel={e=>{e.preventDefault();close();}}>
    <header><h2 id="dp-dialog-title">{title}</h2><button onClick={close}>关闭</button></header>{children}</dialog>;
}
const topics=[
  ['knowledge','知识网络','更换物体后，还能用颜色判断同异吗？'],
  ['reason','分析推理','更换姓名后，模型还能按条件找到正确的一行吗？'],
  ['syntax','语法系统','增加修饰语后，模型还能识别主语吗？']
];
function Plan({close,onDraft}){
  const [selected,choose]=useState(topics.map(t=>t[0])),[bilingual,setBilingual]=useState(true),[glm,setGlm]=useState(false);
  const [families,setFamilies]=useState('128'),[batch,setBatch]=useState('32');
  const models=glm?['Qwen3-14B','GLM-4-9B']:['Qwen3-14B'],langs=bilingual?['中文','英文']:['中文'];
  const n=Number(families),b=Number(batch),valid=selected.length&&Number.isInteger(n)&&n>=1&&n<=4096&&Number.isInteger(b)&&b>=1&&b<=512;
  const count=valid?selected.length*models.length*langs.length*Math.ceil(n/b):0;
  function exportPlan(){
    if(!valid)return;
    const shards=[];
    for(const [topic] of topics.filter(t=>selected.includes(t[0])))for(const model of models)for(const language of langs)
      for(let start=1;start<=n;start+=b)shards.push({topic,model,language,first_family:start,last_family:Math.min(start+b-1,n)});
    download('系统测试分片草案.json','application/json',JSON.stringify({schema:'ai2050.preview.shard-plan.v1',status:'draft',topics:topics.filter(t=>selected.includes(t[0])).map(([id,name,question])=>({id,name,question})),models,languages:langs,families_per_group:n,shards},null,2));
  }
  return <Dialog title="系统测试方案" close={close}><p>将一个研究问题做成一组题，再按模型、语言和题目家族分给不同机器。</p>
    <fieldset><legend>研究问题</legend>{topics.map(([id,title,q])=><label className="dp-check" key={id}><input type="checkbox" checked={selected.includes(id)} onChange={e=>choose(e.target.checked?[...selected,id]:selected.filter(v=>v!==id))}/><span><b>{title}</b> · {q}</span></label>)}</fieldset>
    <div className="dp-fields"><label>语言范围<select value={bilingual?'both':'zh'} onChange={e=>setBilingual(e.target.value==='both')}><option value="both">中文、英文</option><option value="zh">中文</option></select></label>
    <label>模型范围<select value={glm?'both':'q'} onChange={e=>setGlm(e.target.value==='both')}><option value="q">Qwen3-14B</option><option value="both">Qwen3-14B、GLM-4-9B</option></select></label>
    <label>每组家族数<input type="number" min="1" max="4096" value={families} onChange={e=>setFamilies(e.target.value)}/></label>
    <label>每个分片最多家族数<input type="number" min="1" max="512" value={batch} onChange={e=>setBatch(e.target.value)}/></label></div>
    <p role="status" className="dp-callout">{valid?<><b>{count} 个分片</b> · 每个研究问题在每种语言和模型上各测 {n} 个家族。</>:'请选择研究问题，每组填写 1–4096 个家族，分片容量填写 1–512。'}</p>
    <p className="cx-note">此处导出分片草案。正式分发前须完成查重、材料划分和版本冻结。中心按模型快照与方案版本分别汇总，每个合格分片计一次。</p>
    <footer><button onClick={onDraft}>实验设计与草稿</button><button className="cx-primary" disabled={!valid} onClick={exportPlan}>导出分片草案</button></footer></Dialog>;
}
const notes={
  idle:'点击后自动走完一个示例分片。',claim:'检查模型快照和可用资源，向中心领取匹配的分片。',
  run:'本机正在测试分配到的姓名家族。',upload:'正在提交本分片的统计结果与执行信息。',verify:'结果已送达，中心正在核对分片身份和版本。',
  done:'示例中心已计入本机这份分片。科学审阅待进行。',empty:'当前演示没有可分配的分片，可调整演示条件后重新匹配。',
  model:'此任务需要匹配的 Qwen3-14B 快照。正式客户端可在「系统设置」的节点模型页准备权重。',
  retry:'本地结果已保留。可以从上传步骤继续，无需重新测试。',stopped:'已结束本次参与。未确认完成的分片由中心在租约到期后重新派发。'
};
export function Participation({onStation}){
  const {phase,scenario}=useSyncExternalStore(subscribe,()=>session);
  const [plan,setPlan]=useState(false);
  const active=waiting.has(phase),retry=phase==='retry',idle=phase==='idle',index=stageIndex[phase]??-1;
  const click=()=>retry?move('upload',scenario,true):move('claim',scenario);
  return <section className="dp-home" aria-label="实验管理">
    <header className="dp-heading"><div><h1>实验管理</h1><p>用自己的电脑完成一份测试，和其他参与者一起检验模型。</p></div><span className="dp-offline">离线演示</span></header>
    <section className="dp-join"><div className="dp-lead"><div><h2>参与分布式测试</h2><p>中心把同一套测试分给多台机器。客户端自动领取适合本机的任务，完成后上传结果。</p></div>
      {active?<button onClick={()=>move('stopped')}>停止参与</button>:<button className="dp-primary" onClick={click}>{retry?'继续上传':phase==='done'?'重新演示':'一键参与'}</button>}
    </div>
    <dl className="dp-scope"><div><dt>任务来源</dt><dd>项目受信中心 · 已批准的测试</dd></div><div><dt>本机资源</dt><dd>GPU 与 CPU · 先检查显存和磁盘</dd></div><div><dt>上传范围</dt><dd>分片统计与执行记录 · 原始激活留在本机</dd></div></dl>
    <ol className="dp-steps" aria-label="分布式测试流程">{phases.map((p,i)=><li key={p} className={i<index?'finished':i===index?'now':''} aria-current={i===index?'step':undefined}><span aria-hidden="true">{i<index?'✓':i+1}</span>{p}</li>)}</ol>
    <div className="dp-task"><div role="status"><strong>{labels[phase]}</strong><span>{notes[phase]}</span></div>
      {!idle&&index>=0&&<p>五条记录如何选源 · Qwen3-14B · 示范分片 4 / 4</p>}
      {(phase==='verify'||phase==='done')&&<p className="dp-count">示例中心已核验 <b>{phase==='done'?'4':'3'} / 4</b> 个分片</p>}</div>
    <details className="dp-conditions"><summary>演示条件</summary><label>模拟情况 <select aria-label="模拟情况" disabled={active||retry} value={scenario} onChange={e=>move('idle',e.target.value)}>{Object.entries(scenarios).map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label><p>此页仅演示一次领取流程。实时队列、设备检查与上传回执由正式客户端提供。</p>{retry&&<button onClick={()=>move('stopped')}>停止参与</button>}</details>
    </section>
    <section className="dp-study"><div><h2>用已有结果做研究</h2><p>在图谱中选择测试，查看不同条件下的表现，或下载逐题结果。无需贡献算力。</p></div><button onClick={()=>atlas('browse')}>进入语言与模型 →</button></section>
    <nav className="dp-tools" aria-label="研究组织"><button onClick={()=>setPlan(true)}>系统测试方案</button><button onClick={()=>onStation('queue')}>查看完整队列</button><button onClick={()=>window.openTestDatabase({archive:'本稿附中心回执'})}>查看已归档实验</button></nav>
    {plan&&<Plan close={()=>setPlan(false)} onDraft={()=>onStation('designs')}/>}
  </section>;
}

const arms={base:'原始模型',target:'目标阻断',cut:'目标阻断',ctrl:'匹配对照',random:'随机对照',restore:'恢复',restore_mlp:'恢复 FFN 组',sham:'假操作'};
function StudyResult({study}){
  const parsed=study.lines.trim().split(/\r?\n/).map(s=>JSON.parse(s));
  const grouped=new Map();
  for(const r of parsed){const g=grouped.get(r.arm)||{n:0,correct:0};g.n++;if(r.correct)g.correct++;grouped.set(r.arm,g);}
  return <div className="dp-readout"><div className="dp-lead"><div><small>Phase {study.phase} · {study.model}</small><h2>{study.title}</h2></div>
    <button onClick={()=>download('phase-'+study.phase+'-case_results.jsonl','application/x-ndjson',study.lines)}>下载逐题结果</button></div>
    <p>{study.phase===3008?'按实际生成的颜色同异答案计分。':'按全词表得分最高的下一个 token 是否命中目标答案计分。'}每行显示该实验条件内答对的题数与总题数。</p>
    <ul className="dp-bars">{[...grouped].map(([arm,g])=><li key={arm}><span>{arms[arm]||arm}</span><progress aria-label={arm+' 答对比例'} max={g.n} value={g.correct}/><b>{g.correct} / {g.n}</b></li>)}</ul>
    <div className="dp-tools"><button onClick={()=>window.openTestDatabase({q:String(study.phase)},study.phase)}>查看题目和原始记录</button>
      {study.phase===2997&&<button onClick={()=>window.openEvidenceWorkspace('object',{snapshot:study.snapshot,location:{layer:29,module:'attn',index:21}})}>定位注意力头 B28/H21</button>}</div>
    <details><summary>来源与核验</summary><p>本机保存的中心读回回执 · 科学审阅 {study.review}。下载内容是归档中的逐题 JSONL。</p>
    <dl><dt>运行编号</dt><dd>{study.run}</dd><dt>模型快照</dt><dd>{study.snapshot}</dd><dt>逐题文件 SHA-256</dt><dd>{study.sha}</dd></dl></details></div>;
}
export function GraphStudies({model}){
  const id=new URLSearchParams(window.reviewRoute.search).get('studyPack');
  const study=studies().find(s=>String(s.phase)===id);
  const [text,setText]=useState('');
  const focus=new URLSearchParams(window.reviewRoute.search).get('reviewFocus');
  if(!id||focus==='object')return <div className="dp-atlas dp-strip"><b>测试数据</b><button onClick={()=>atlas('browse')}>选择测试数据</button></div>;
  const available=studies().filter(s=>(s.phase+' '+s.title+' '+s.model).toLowerCase().includes(text.trim().toLowerCase()));
  return <section className="dp-atlas" aria-label="图谱测试数据"><header><h2>测试数据</h2><span>中心归档 · 本地快照</span></header>
    {study&&study.model===model?<><StudyResult study={study}/><button onClick={()=>atlas('browse')}>更换测试</button></>:<>
      <p>正式客户端从中心获取所选测试。此稿内置两份已核验的逐题文件，可载入图谱或下载到本机。</p><input type="search" aria-label="查找可载入测试" placeholder="查找研究问题、模型或 Phase" value={text} onChange={e=>setText(e.target.value)}/>
      {available.map(s=><article key={s.phase}><div><small>Phase {s.phase} · {s.model}</small><h3>{s.title}</h3></div><button onClick={()=>atlas(s.phase)}>载入图谱</button></article>)}
      {!available.length&&<p>没有匹配的研究。请更换关键词。</p>}</>}</section>;
}
