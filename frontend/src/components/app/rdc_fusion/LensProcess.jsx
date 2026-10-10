/* 过程透镜 v3：AI 自动研发工作台（M3-P1 内容层协议化，design/ui_decoupled_plan_v2.md §2）
   ─────────────────────────────────────────────────────────────
   流程骨架不变（与「测什么」无关）：五证据门 + AI 模型配置 + 目标 composer + SSE 实时事件。
   内容层 = 三个数据插槽，LIVE/DEMO 双模，只认数据键（/api/ai-rnd/queue、/workspace、
   /api/templates、/api/results 的 schema）；demo 内容全部在 demoData.js。
   红线（v2 §0）：本文件不出现任何具体实验字面量（Q05/F#3734/collect_ar.py 等）。 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { DEMO_QUEUE, DEMO_WORKSPACE, DEMO_ARTIFACTS, DEMO_TERMINAL } from './demoData.js';
import { TEMPLATES, DEMO_RESULTS, DEMO_SCHED, TECH_CATEGORIES, DEMO_COVERAGE, ANALYSES, LANG_TEMPLATES } from './distributedData.js';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');
const DEFAULT_WS_PATH = 'tests/deepseek';   // 工作区默认根（顶栏特征源选择器未来接管）

const GATES=[
  {id:'gap',label:'证据缺口',phases:['analyze']},
  {id:'contract',label:'冻结契约',phases:['plan']},
  {id:'execute',label:'执行实验',phases:['generate','execute']},
  {id:'review',label:'独立复核',phases:['summarize']},
  {id:'writeback',label:'证据回写',phases:[]},
];
const STATUS_LABEL={idle:'未运行',running:'运行中',paused:'已暂停',stopped:'已停止',waiting_step:'等待确认',waiting_approval:'等待确认',completed:'已完成',blocked:'已阻塞',review_required:'待复核',plan_completed:'计划完成'};
const EVENT_META={
  objective:['研究目标','#0284c7'],project_agent_status:['项目 Agent','#0284c7'],project_agent_progress:['任务推进','#0284c7'],
  phase_change:['研究门','#64748b'],round_change:['新一轮','#64748b'],status_change:['运行状态','#64748b'],mode_change:['执行模式','#64748b'],
  analysis:['独立分析','#6d28d9'],planning:['计划','#0284c7'],generation:['代码生成','#6d28d9'],code_generated:['代码产物','#6d28d9'],
  execution:['执行','#b45309'],execution_result:['执行结果','#b45309'],review:['复核','#6d28d9'],summary:['综合','#0284c7'],
  finding:['研究发现','#059669'],database_writeback:['数据回写','#059669'],error:['错误','#dc2626'],
};
const API_TYPES=[['openai','OpenAI'],['nownextai','NowNextAI'],['zhipu','智谱兼容'],['deepseek','DeepSeek'],['dashscope','DashScope'],['openai-compatible','OpenAI 兼容']];
const EMPTY_MASTER={name:'主研发模型',model_type:'master',api_type:'openai',api_base:'https://api.openai.com/v1',api_key:'',model_id:'gpt-5',analysis_prompt:'',planning_prompt:'',code_gen_prompt:'',summary_prompt:''};
const EMPTY_ANALYST={name:'独立分析模型',model_type:'analyst',api_type:'openai',api_base:'https://api.openai.com/v1',api_key:'',model_id:'gpt-5-mini',analysis_prompt:'',planning_prompt:'',code_gen_prompt:'',summary_prompt:''};
const DEFAULT_FORM={project_goal:'',max_loops:3,execution_mode:'auto',stop_on_accepted:true,stop_on_rejected:true,max_consecutive_inconclusive:3};

/* ── 数据键 → 展示的纯函数（live/demo 共用，零内容分支） ── */
function nextIdOf(items){ const p=(items||[]).find(x=>x.status==='pending'); return p?p.id:null; }
function lastSealed(items){ const q=items||[]; for(let i=q.length-1;i>=0;i--) if(q[i].status==='sealed') return q[i]; return null; }
function queueRows(items,n){ // 队列侧栏排序：待办优先，sealed 殿后（数据驱动，无内容分支）
  const rank=x=>x.status==='pending'?0:(x.status==='device_built'?1:2);
  return [...items].sort((a,b)=>(rank(a)-rank(b))||((a.q||0)-(b.q||0))).slice(0,n);
}
function evidenceChain(items,sealed,count){
  const last=lastSealed(items), nx=nextIdOf(items);
  return [
    `队列 sealed ${sealed}/${count}`,
    last?`最近 seal ${last.id}${last.res_sha8?' · '+last.res_sha8:''}`:'',
    nx?`next ${nx} · ${nx===((items||[]).find(x=>x.id===nx)||{}).title||''}`:'',
  ].filter(Boolean);
}
function demoDistRows(){ // 分布式队列 demo 回退（数据源自 distributedData）
  const cnt={}; DEMO_RESULTS.forEach(r=>{cnt[r.tm_id]=(cnt[r.tm_id]||0)+1;});
  return TEMPLATES.map(t=>({tm_id:t.id,name:t.name,dim:t.dim,results:cnt[t.id]||0}));
}
/* M7-P1 缺口任务（覆盖矩阵空格 → 任务建议）：数据 = DEMO_COVERAGE（P2 真源 /api/coverage），
   技术/模板名反查注册表，零字面量。status!=='done' 即缺口。 */
function gapRows(catFilter){
  const tplName=id=>(LANG_TEMPLATES.find(t=>t.id===id)||{}).name||id;
  const catOf=id=>TECH_CATEGORIES.find(c=>c.id===id)||{};
  return DEMO_COVERAGE
    .filter(c=>c.status!=='done'&&(!catFilter||catFilter==='all'||c.cat===catFilter))
    .map(c=>{ const cat=catOf(c.cat); return {
      tpl:c.tpl, cat:c.cat, status:c.status,
      tplName:tplName(c.tpl), catName:cat.name||c.cat, catColor:cat.color,
      input:cat.input||'—', cost:cat.cost||'—', note:cat.note||'',
      nTech:ANALYSES.filter(a=>a.category===c.cat).length,
    };});
}
function fmtSize(n){ return n>1048576?(n/1048576).toFixed(1)+' MB':(n/1024).toFixed(1)+' KB'; }
function fmtTime(v){
  if(!v) return '—';
  const d=new Date(v);
  if(Number.isNaN(d.getTime())) return String(v);
  return new Intl.DateTimeFormat('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}).format(d);
}
function eventText(e){
  if(e.content) return e.content;
  if(e.message) return e.message;
  if(e.objective) return e.objective;
  if(e.finding&&e.finding.summary) return e.finding.summary;
  if(e.finding&&e.finding.decision) return '裁决：'+e.finding.decision;
  if(e.run_id) return '工件已保存至 '+e.run_id;
  if(e.phase) return '进入 '+(GATES.find(g=>g.phases.includes(e.phase))||{}).label||e.phase;
  if(e.status) return STATUS_LABEL[e.status]||e.status;
  if(e.round) return '开始 Loop '+e.round;
  if(e.type==='code_generated') return '主模型已生成待验证代码';
  if(e.type==='execution_result') return '执行完成，原始结果已进入测试检查器';
  return '运行状态已更新';
}

/* 模型字段表单（紧凑版 ModelFields：主模型含四段提示词，分析模型仅独立分析提示词） */
function ModelFields({model,onChange,onRemove,master}){
  const set=(k,v)=>onChange({...model,[k]:v});
  return (
    <div className="fw-ai-fields">
      {onRemove&&<button type="button" className="fw-ai-del" onClick={onRemove}>删除该模型</button>}
      <div className="fw-ai-grid2">
        <label>名称<input value={model.name||''} onChange={e=>set('name',e.target.value)}/></label>
        <label>模型 ID<input value={model.model_id||''} onChange={e=>set('model_id',e.target.value)}/></label>
      </div>
      <div className="fw-ai-grid2">
        <label>API 类型<select value={model.api_type||'openai'} onChange={e=>set('api_type',e.target.value)}>{API_TYPES.map(([v,t])=><option key={v} value={v}>{t}</option>)}</select></label>
        <label>API Key<input type="password" autoComplete="off" placeholder="仅保存在本地配置" value={model.api_key||''} onChange={e=>set('api_key',e.target.value)}/></label>
      </div>
      <label>API 地址<input value={model.api_base||''} onChange={e=>set('api_base',e.target.value)}/></label>
      {master?(
        <details className="fw-ai-prompts">
          <summary>提示词（分析 / 规划 / 编程 / 裁决）</summary>
          <label>分析提示词<textarea rows={2} value={model.analysis_prompt||''} onChange={e=>set('analysis_prompt',e.target.value)}/></label>
          <label>规划提示词<textarea rows={2} value={model.planning_prompt||''} onChange={e=>set('planning_prompt',e.target.value)}/></label>
          <label>编程提示词<textarea rows={2} value={model.code_gen_prompt||''} onChange={e=>set('code_gen_prompt',e.target.value)}/></label>
          <label>裁决提示词<textarea rows={2} value={model.summary_prompt||''} onChange={e=>set('summary_prompt',e.target.value)}/></label>
        </details>
      ):(
        <details className="fw-ai-prompts">
          <summary>独立分析提示词</summary>
          <textarea rows={2} value={model.analysis_prompt||''} onChange={e=>set('analysis_prompt',e.target.value)}/>
        </details>
      )}
    </div>
  );
}

/* ── 通用 kv 渲染（选中项详情：有什么键渲染什么键） ── */
function KVList({pairs}){
  return (
    <div className="fw-m3-kv">
      {pairs.filter(p=>p[1]!==undefined&&p[1]!==null&&p[1]!=='').map(([k,v,mono])=>(
        <div key={k} className="fw-m3-kvline"><i>{k}</i><b className={mono?'fw-mono':''}>{String(v)}</b></div>
      ))}
    </div>
  );
}

export default function LensProcess({on,onGo}){
  const [config,setConfig]=useState({master_model:EMPTY_MASTER,analyst_models:[{...EMPTY_ANALYST}]});
  const [form,setForm]=useState(DEFAULT_FORM);
  const [status,setStatus]=useState({status:'idle',mode:'auto',round:0});
  const [agent,setAgent]=useState({});
  const [events,setEvents]=useState([]);
  const [tab,setTab]=useState(0);
  const [offline,setOffline]=useState(false);
  const [busy,setBusy]=useState(false);
  const [saving,setSaving]=useState(false);
  const [err,setErr]=useState('');
  const eid=useRef(0);

  /* ── M3-P1 内容层状态 ── */
  const [rnd,setRnd]=useState(null);            // /api/ai-rnd/queue 原样
  const [dist,setDist]=useState(null);          // [{tm_id,name,dim,results}]
  const [sched,setSched]=useState(null);        // M6-P2 /api/tasks 原样（S8 调度队列只读视图）
  const [qTab,setQTab]=useState('rnd');
  const [gapCat,setGapCat]=useState('all');   // M7-P1 缺口任务的技术类别过滤器
  const [qSel,setQSel]=useState(null);          // {src:'rnd'|'dist', item}
  const [ws,setWs]=useState(null);              // /workspace 原样
  const [wsLive,setWsLive]=useState(false);
  const [wsPath,setWsPath]=useState(DEFAULT_WS_PATH);
  const [wsErr,setWsErr]=useState('');
  const [art,setArt]=useState(null);            // {type,label,loading,err,item,rows,file}

  const api=useCallback(async(path,opts)=>{
    let r;
    try{ r=await fetch(`${API_BASE}/api/ai-rnd${path}`,opts); }
    catch{ setOffline(true); throw new Error('后端 :5001 不可达（server.py 未启动？）'); }
    const p=await r.json().catch(()=>({}));
    if(!r.ok) throw new Error(p.detail||('HTTP '+r.status));
    return p;
  },[]);

  /* 内容层：队列 + 分布式模板 + 工作区（失败一律静默降级 DEMO） */
  const loadWs=useCallback(async(p)=>{
    setWsErr('');
    try{
      const r=await fetch(`${API_BASE}/api/ai-rnd/workspace?path=${encodeURIComponent(p)}`);
      if(!r.ok) throw new Error('HTTP '+r.status);
      const payload=await r.json();
      setWs(payload); setWsLive(true); setWsPath(payload.path||p);
    }catch{ setWs(null); setWsLive(false); setWsPath(p); }
  },[]);
  useEffect(()=>{
    (async()=>{
      try{
        const r=await fetch(`${API_BASE}/api/ai-rnd/queue`);
        const p=r.ok?await r.json():null;
        setRnd(p&&Array.isArray(p.queue)&&p.queue.length?p:null);
      }catch{ setRnd(null); }
      try{
        const [tp,rs]=await Promise.all([
          fetch(`${API_BASE}/api/templates`).then(x=>x.ok?x.json():null).catch(()=>null),
          fetch(`${API_BASE}/api/results?limit=200`).then(x=>x.ok?x.json():null).catch(()=>null),
        ]);
        if(tp&&Array.isArray(tp.templates)){
          const cnt={}; ((rs&&rs.results)||[]).forEach(x=>{cnt[x.tm_id]=(cnt[x.tm_id]||0)+1;});
          setDist(tp.templates.map(t=>({tm_id:t.tm_id||t.id,name:t.name,dim:t.dim,version:t.version,results:cnt[t.tm_id||t.id]||0})));
        }else setDist(null);
      }catch{ setDist(null); }
      try{
        const r=await fetch(`${API_BASE}/api/tasks`);
        const p=r.ok?await r.json():null;
        setSched(p&&p.stats?p:null);
      }catch{ setSched(null); }
      loadWs(DEFAULT_WS_PATH);
    })();
  },[]); // eslint-disable-line

  const load=useCallback(async(quiet)=>{
    try{
      const st=await api('/session/status',{cache:'no-store'});
      setStatus(st);
      setAgent(st.project_agent||{});
      if(st.project_agent&&st.project_agent.config){
        setForm(f=>({...f,...st.project_agent.config,project_goal:st.project_agent.project_goal||f.project_goal}));
      }
      setOffline(false);
      if(quiet!==true) setErr('');
    }catch{ /* offline 已标记 */ }
  },[api]);

  useEffect(()=>{
    (async()=>{
      try{
        const cf=await api('/config',{cache:'no-store'});
        setConfig({
          master_model:{...EMPTY_MASTER,...(cf.master_model||{})},
          analyst_models:Array.isArray(cf.analyst_models)&&cf.analyst_models.length?cf.analyst_models.map(m=>({...EMPTY_ANALYST,...m,model_type:'analyst'})):[{...EMPTY_ANALYST}],
        });
        setOffline(false);
      }catch{ /* offline */ }
    })();
    load(true);
    const t=setInterval(()=>load(true),5000);
    return ()=>clearInterval(t);
  },[]); // eslint-disable-line

  useEffect(()=>{
    let es=null;
    try{
      es=new EventSource(`${API_BASE}/api/ai-rnd/session/events`);
      es.onopen=()=>setOffline(false);
      es.onmessage=(m)=>{
        try{
          const ev=JSON.parse(m.data);
          eid.current+=1;
          setEvents(c=>[...c.slice(-199),{...ev,id:(ev.timestamp||Date.now())+'-'+eid.current}]);
        }catch{ /* 忽略畸形事件 */ }
      };
    }catch{ /* offline */ }
    return ()=>{ if(es) es.close(); };
  },[]); // eslint-disable-line

  const projectAction=async(type)=>{
    setBusy(true); setErr('');
    try{
      if(type==='plan'){
        await api('/project-agent/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_goal:form.project_goal.trim(),max_tasks:Number(form.max_loops)||3})});
      }else if(type==='start'){
        await api('/project-agent/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...form,project_goal:form.project_goal.trim(),max_loops:Number(form.max_loops)||3})});
        setTab(1);
      }else if(type==='stop'){
        await api('/project-agent/stop',{method:'POST'});
      }
      await load(true);
    }catch(e){ setErr(e.message); }
    setBusy(false);
  };
  const sessionAction=async(type,value)=>{
    setBusy(true); setErr('');
    try{
      if(type==='pause') await api('/session/pause',{method:'POST'});
      else if(type==='resume') await api('/session/start',{method:'POST'});
      else if(type==='step') await api('/session/step',{method:'POST'});
      else if(type==='stop') await api('/session/stop',{method:'POST'});
      else if(type==='mode') await api('/session/mode',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:value})});
      await load(true);
    }catch(e){ setErr(e.message); }
    setBusy(false);
  };
  const saveConfig=async()=>{
    setSaving(true); setErr('');
    try{
      await api('/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({
        master_model:{...config.master_model,model_type:'master'},
        analyst_models:config.analyst_models.map(m=>({...m,model_type:'analyst'})),
      })});
    }catch(e){ setErr(e.message); }
    setSaving(false);
  };

  /* ── 内容层派生（live → demo 回退，渲染共用） ── */
  const rndLive=Boolean(rnd);
  const distLive=Boolean(dist);
  const schedLive=Boolean(sched);               // M6-P2 调度队列（S8 /api/tasks）
  const rndItems=rndLive?rnd.queue:DEMO_QUEUE.queue;
  const rndSealed=rndLive?rnd.sealed:DEMO_QUEUE.sealed;
  const rndCount=rndLive?rnd.count:DEMO_QUEUE.count;
  const distRows=distLive?dist:demoDistRows();
  const rndNext=nextIdOf(rndItems);
  const rndSorted=queueRows(rndItems,8);
  const distSorted=[...distRows].sort((a,b)=>(b.results||0)-(a.results||0)).slice(0,8);
  const gapItems=gapRows(gapCat);   // M7-P1 缺口任务（覆盖矩阵空格）
  const wsView=wsLive?ws:DEMO_WORKSPACE;
  /* M6-P2 调度队列派生：LIVE 从 /api/tasks 现算剩余租约；DEMO 用快照自带 remain_h */
  const schedSrc=schedLive?sched:DEMO_SCHED;
  const schedStats=schedLive?(sched.stats||{}):DEMO_SCHED.stats;
  const schedLease=schedSrc.lease_hours;
  const schedRows=(schedSrc.active||[]).map(a=>schedLive
    ?{...a,remain_h:Math.max(0,(a.lease_until-sched.server_time)/3600)}
    :a);
  const chain=evidenceChain(rndItems,rndSealed,rndCount);

  /* 默认选中：研发线最高优先待办（无则首条）；渲染与 live/demo 共用 */
  useEffect(()=>{
    if(qSel) return;
    const it=rndItems.find(x=>x.status==='pending')||rndItems[0];
    if(it) setQSel({src:'rnd',item:it});
  },[rndItems]); // eslint-disable-line

  /* 选中动作：队列条目 / 工作区文件 / 分布式结果明细 */
  const selectRnd=(item)=>{ setQSel({src:'rnd',item}); setArt({type:'queue',label:item.id+' · '+item.title,item}); setTab(0); };
  const selectDist=(row)=>{
    setQSel({src:'dist',item:row});
    setArt({type:'dist',label:row.tm_id+' · '+(row.name||''),loading:true,row});
    setTab(0);
    (async()=>{
      try{
        const r=await fetch(`${API_BASE}/api/results?tm_id=${encodeURIComponent(row.tm_id)}&limit=20`);
        const p=r.ok?await r.json():null;
        setArt(a=>(a&&a.type==='dist'&&a.row.tm_id===row.tm_id)?{...a,loading:false,rows:(p&&p.results)||[]}:a);
      }catch{ setArt(a=>(a&&a.type==='dist')?{...a,loading:false,err:'结果列表拉取失败'}:a); }
    })();
  };
  const openFile=async(f)=>{
    if(f.demo){
      const d=DEMO_ARTIFACTS[f.demo];
      setArt(d?{type:'file',label:f.name+'（demo）',file:{name:f.name,content:d.text}}:{type:'file',label:f.name,err:'demo 工件缺失'});
      setTab(0); return;
    }
    setArt({type:'file',label:f.name,loading:true}); setTab(0);
    try{
      const r=await fetch(`${API_BASE}/api/ai-rnd/workspace/file?path=${encodeURIComponent(f.path)}`);
      if(!r.ok) throw new Error('HTTP '+r.status);
      const p=await r.json();
      setArt(a=>(a&&a.type==='file')?{...a,loading:false,file:p}:a);
    }catch(e){ setArt(a=>(a&&a.type==='file')?{...a,loading:false,err:'文件读取失败（'+e.message+'）'}:a); }
  };
  const wsUp=()=>{
    const parts=wsView.path.split('/').filter(Boolean);
    if(parts.length>1){ const up=parts.slice(0,-1).join('/'); if(loadWs) loadWs(up); }
  };

  const active=Boolean(agent&&agent.enabled);
  const running=status.status==='running';
  const masterReady=Boolean(config.master_model.api_key&&config.master_model.api_key.trim());
  const analystReady=config.analyst_models.filter(m=>m.api_key&&m.api_key.trim()).length;
  const ready=masterReady&&analystReady>0;
  const activeGate=GATES.findIndex(g=>g.id===gateId(status.current_phase,status.status));
  const tabs=art?[{k:'art',label:art.label},{k:'events',label:'实时事件'}]:[{k:'events',label:'实时事件'}];
  const tabIdx=Math.min(tab,tabs.length-1);

  function gateId(phase,st){
    if(!phase) return st==='stopped'?'writeback':'gap';
    const g=GATES.find(x=>x.phases.includes(phase));
    return g?g.id:'gap';
  }

  return (
    <section className={'fw-view fw-process'+(on?' on':'')}>
      {/* ===== 左栏：AI 模型配置 + 任务队列 + 工作区 ===== */}
      <div className="fw-pr-left">
        <div className="fw-pr-sec">AI 模型 {offline?<span className="fw-ai-off">后端离线</span>:<span className={'fw-ai-dot'+(ready?' ok':'')} title={ready?'已就绪（主模型+分析模型均配 Key）':'配置主模型与至少一个分析模型的 API Key'}/>}</div>
        <details className="fw-ai-card">
          <summary><span className="fw-ai-role">主</span>{config.master_model.name||'主研发模型'}<small>{masterReady?'Key 已配':config.master_model.model_id}</small></summary>
          <ModelFields master model={config.master_model} onChange={v=>setConfig({...config,master_model:{...v,model_type:'master'}})}/>
        </details>
        {config.analyst_models.map((m,i)=>(
          <details className="fw-ai-card" key={i}>
            <summary><span className="fw-ai-role an">辅</span>{m.name||('分析模型 '+(i+1))}<small>{(m.api_key&&m.api_key.trim())?'Key 已配':m.model_id}</small></summary>
            <ModelFields model={m}
              onChange={v=>setConfig({...config,analyst_models:config.analyst_models.map((x,j)=>j===i?{...v,model_type:'analyst'}:x)})}
              onRemove={()=>setConfig({...config,analyst_models:config.analyst_models.filter((_,j)=>j!==i)})}/>
          </details>
        ))}
        <button type="button" className="fw-ai-add" onClick={()=>setConfig({...config,analyst_models:[...config.analyst_models,{...EMPTY_ANALYST,name:'独立分析模型 '+(config.analyst_models.length+1)}]})}>＋ 添加分析模型</button>
        <button type="button" className="fw-ai-save" disabled={saving||offline} onClick={saveConfig}>{saving?'保存中…':'保存模型与提示词'}</button>

        <div className="fw-pr-sec">
          任务队列
          <span className="fw-m3-qtabs">
            <button type="button" className={'fw-m3-qtab'+(qTab==='rnd'?' on':'')} onClick={()=>setQTab('rnd')}>研究线</button>
            <button type="button" className={'fw-m3-qtab'+(qTab==='dist'?' on':'')} onClick={()=>setQTab('dist')}>分布式</button>
            <button type="button" className={'fw-m3-qtab'+(qTab==='sched'?' on':'')} onClick={()=>setQTab('sched')}>调度</button>
            <button type="button" className={'fw-m3-qtab'+(qTab==='gap'?' on':'')} onClick={()=>setQTab('gap')} title="覆盖矩阵空格 → 任务建议（对象 × 语言模板 × 技术类别）">缺口</button>
            <span className={'fw-src-chip mini '+((qTab==='gap'?false:(qTab==='rnd'?rndLive:qTab==='dist'?distLive:schedLive))?'live':'demo')}>
              {qTab==='rnd'?(rndLive?'LIVE':'DEMO'):qTab==='gap'?(gapItems.length+' 格待做'):qTab==='dist'?(distLive?'LIVE':'DEMO'):(schedLive?'LIVE':'DEMO')}
            </span>
          </span>
        </div>
        {qTab==='rnd'&&rndSorted.map(it=>{
          const [pl,cls]=it.status==='pending'?[(it.id===rndNext?'NEXT':'QUEUED'),(it.id===rndNext?'fw-pill-next':'')]:it.status==='sealed'?['SEALED','fw-pill-done']:['BUILT','fw-pill-run'];
          return (
            <button key={it.id} type="button" className={'fw-q-row'+(it.id===rndNext?' on':'')}
                    onClick={()=>selectRnd(it)} title={it.deliverable||it.note||''}>
              <span className="id">{it.id}</span>{it.title}<span className={'fw-pill '+cls}>{pl}</span>
            </button>
          );
        })}
        {qTab==='rnd'&&<div className="fw-m3-more">共 {rndCount} 条 · sealed {rndSealed}（{rndLive?'phase_queue_v1':'demo'}）</div>}
        {qTab==='dist'&&distSorted.map(row=>(
          <button key={row.tm_id} type="button" className={'fw-q-row'+((row.results||0)>0?' has':'')} onClick={()=>selectDist(row)} title={row.name||''}>
            <span className="id">{row.tm_id}</span>{row.name||row.dim}<span className={'fw-pill '+((row.results||0)>0?'fw-pill-run':'')}>{(row.results||0)>0?(row.results+' 结果'):'待测'}</span>
          </button>
        ))}
        {qTab==='dist'&&<div className="fw-m3-more">共 {distRows.length} 模板（{distLive?'中心节点结果库':'demo'}）</div>}
        {qTab==='sched'&&schedRows.map(a=>{
          const remain=a.remain_h;
          return (
            <div key={a.task_id} className="fw-q-row" style={{cursor:'default'}} title={'task '+a.task_id+' · 模板 '+a.tm_id+' · node '+(a.node_name||'?')}>
              <span className="id">{a.tm_id}</span>{a.node_name||'?'} · {a.model||'?'}<span className={'fw-pill '+(remain<1?'':'fw-pill-run')}>{'租约 '+remain.toFixed(1)+'h'}</span>
            </div>
          );
        })}
        {qTab==='sched'&&<div className="fw-m3-more">
          活跃 {schedStats.claimed||0} · 完成 {schedStats.done||0} · 失败 {schedStats.failed||0} · 过期 {schedStats.expired||0} · 单租约 {schedLease}h
          （{schedLive?'GET /api/tasks 实时':'demo'}）
        </div>}
        {qTab==='sched'&&<div className="fw-m3-more" title="概念消歧：调度队列=节点租约实时状态；研究线队列=phase_queue 依赖序；分布式=模板结果库">
          ↑ 调度队列=节点租约（谁在领什么）· 研究线=phase 依赖序 · 两层不同
        </div>}
        {qTab==='gap'&&(<>
          <div className="fw-gap-filter">
            <button type="button" className={'fw-m3-qtab'+(gapCat==='all'?' on':'')} onClick={()=>setGapCat('all')}>全部</button>
            {TECH_CATEGORIES.map(c=>(
              <button key={c.id} type="button" className={'fw-m3-qtab'+(gapCat===c.id?' on':'')}
                      style={gapCat===c.id?{borderColor:c.color,color:c.color}:undefined}
                      onClick={()=>setGapCat(c.id)}>{c.short||c.name}</button>
            ))}
          </div>
          {gapItems.map(g=>(
            <button key={g.tpl+'__'+g.cat} type="button" className="fw-q-row"
                    onClick={()=>{ setQSel({src:'gap',item:g}); setArt({type:'gap',label:g.tplName+' × '+g.catName,item:g}); }}
                    title={g.note}>
              <span className="id" style={{color:g.catColor}}>{g.tpl}</span>
              {g.tplName} × {g.catName}
              <span className={'fw-pill '+(g.status==='pending'?'fw-pill-run':'')}>{g.status==='pending'?'装置已建':'空格'}</span>
            </button>
          ))}
          {gapItems.length===0&&<div className="fw-m3-more">该类别暂无缺口——全部模板已有该类结果。</div>}
          <div className="fw-m3-more">缺口 = 覆盖矩阵中 status≠done 的格（对象 × 模板 × 技术类别）· 点击看输入契约与预计算力 · P2 由 /api/coverage 转真</div>
        </>)}

        <div className="fw-pr-sec">
          工作区 {wsView.path&&<span className="fw-m3-wspath fw-mono" title={wsView.path}>/{wsView.path.split('/').pop()}</span>}
          <span className={'fw-src-chip mini '+(wsLive?'live':'demo')}>{wsLive?'LIVE':'DEMO'}</span>
        </div>
        <button type="button" className="fw-file up" disabled={!wsLive} onClick={wsUp} title={wsLive?'上级目录':'demo 模式不可导航'}>◂ 上级</button>
        {wsErr&&<div className="fw-ai-err">{wsErr}</div>}
        {(wsView.dirs||[]).map(d=>(
          <button key={d.path} type="button" className="fw-file dir" onClick={()=>{ if(wsLive) loadWs(d.path); }} title={d.path}>▸ {d.name}/</button>
        ))}
        {(wsView.files||[]).map(f=>(
          <button key={f.path} type="button" className="fw-file" onClick={()=>openFile(f)} title={f.path+(f.size?(' · '+fmtSize(f.size)): '')}>{f.name}</button>
        ))}
        {wsView.truncated&&<div className="fw-m3-more">文件过多已截断（API limit）</div>}

        <div className="fw-pr-sec">运行配置</div>
        <div className="fw-q-row" style={{cursor:'default'}}><span className="id" style={{width:'auto'}}>4b bf16 · 14B/9B NF4</span></div>
        <div className="fw-q-row" style={{cursor:'default'}}><span className="id" style={{width:'auto'}}>drift 断言 · 预注册冻结</span></div>
      </div>

      {/* ===== 中栏：目标 composer + 证据门 + 工件/事件 ===== */}
      <div className="fw-pr-mid">
        <div className="fw-ai-composer">
          <textarea rows={2} value={form.project_goal} disabled={active}
            onChange={e=>setForm({...form,project_goal:e.target.value})}
            placeholder="研究目标：描述要自动完成的项目研发；留空则从最高优先级证据缺口开始……"/>
          <div className="fw-ai-ctrl">
            <div className="fw-ai-mode" aria-label="执行模式">
              <button type="button" className={form.execution_mode==='auto'?'on':''} disabled={active} onClick={()=>setForm({...form,execution_mode:'auto'})}>自动</button>
              <button type="button" className={form.execution_mode==='manual'?'on':''} disabled={active} onClick={()=>setForm({...form,execution_mode:'manual'})}>手动</button>
            </div>
            {!active&&(
              <label className="fw-ai-loops">最多<input type="number" min="1" max="12" value={form.max_loops} onChange={e=>setForm({...form,max_loops:Number(e.target.value)})}/>轮</label>
            )}
            <span className="fw-ai-status">
              <span className={'fw-ai-dot'+(running?' run':'')}/>
              {offline?'后端离线':((STATUS_LABEL[status.status]||'未运行')+(status.round?(' · Loop '+status.round):''))}
            </span>
            <div className="fw-ai-actions">
              {!active&&<button type="button" disabled={busy||offline} onClick={()=>projectAction('plan')}>生成计划</button>}
              {!active&&<button type="button" className="pri" disabled={busy||!ready||offline} title={ready?'启动自动研发循环':'先在左侧配置主模型与至少一个分析模型的 API Key'} onClick={()=>projectAction('start')}>▶ 开始研发</button>}
              {active&&running&&<button type="button" disabled={busy} onClick={()=>sessionAction('pause')}>暂停</button>}
              {active&&status.status==='paused'&&<button type="button" className="pri" disabled={busy} onClick={()=>sessionAction('resume')}>继续</button>}
              {active&&status.status==='waiting_step'&&<button type="button" className="pri" disabled={busy} onClick={()=>sessionAction('step')}>确认下一门</button>}
              {active&&<button type="button" className="stop" disabled={busy} onClick={()=>sessionAction('stop')}>停止</button>}
            </div>
          </div>
          {err&&<div className="fw-ai-err">{err}</div>}
        </div>

        <div className="fw-gates" aria-label="研究证据门">
          {GATES.map((g,i)=>(
            <div key={g.id} className={'fw-gate '+(i<activeGate?'done':(i===activeGate?'act':''))}>
              <span>{i<activeGate?'✓':i+1}</span><b>{g.label}</b>
            </div>
          ))}
        </div>

        <div className="fw-ed-tabs">
          {tabs.map((t,i)=>(
            <button key={t.k} type="button" className={'fw-ed-tab'+(tabIdx===i?' on':'')} onClick={()=>setTab(i)}>{t.label}</button>
          ))}
        </div>
        <div className="fw-ed-body">
          {tabIdx===0&&art&&(
            <div className="fw-m3-art">
              {art.loading&&<div className="fw-pc-empty">拉取工件…</div>}
              {art.err&&<div className="fw-ai-err">{art.err}</div>}
              {art.type==='file'&&art.file&&(
                <pre className="fw-m3-pre">{art.file.content}</pre>
              )}
              {art.type==='queue'&&art.item&&(
                <KVList pairs={[
                  ['ID',art.item.id,true],['标题',art.item.title],['状态',art.item.status,true],
                  ['块',art.item.block],['KPI',art.item.kpi,true],['GPU',art.item.gpu],
                  ['交付物',art.item.deliverable],['seal 记录',art.item.seal_record,true],
                  ['结果 SHA8',art.item.res_sha8,true],['预注册 design_sha',art.item.prereg_design_sha,true],
                  ['注记',art.item.note],
                ]}/>
              )}
              {art.type==='dist'&&(
                <div className="fw-m3-dist">
                  <KVList pairs={[['模板',art.row.tm_id,true],['名称',art.row.name],['维度',art.row.dim],['结果数',art.row.results]]}/>
                  {!art.loading&&Array.isArray(art.rows)&&art.rows.length>0&&(
                    <div className="fw-m3-drows">
                      <div className="fw-pl-row head"><span>结果</span><span>模型 / 节点</span><span>状态</span><span className="r">摘要</span></div>
                      {art.rows.map(r=>{
                        const dg=r.summary_digest||{};
                        const brief=dg.eta2_by_factor?('η² '+Object.entries(dg.eta2_by_factor).filter(([,v])=>typeof v==='number').map(([k,v])=>`${k} ${v.toFixed(3)}`).join('/')):(dg.status||'—');
                        return (
                          <div key={r.sha} className="fw-pl-row">
                            <span className="fw-mono" title={r.sha}>{r.sha.slice(0,10)}…</span>
                            <span className="dim2">{r.model_id||'—'} · {r.node_id}</span>
                            <span><span className={'fw-pill '+(r.kind==='real'?'fw-pill-run':'fw-pill-done')}>{r.kind}</span></span>
                            <span className="r dim2">{brief}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {!art.loading&&(!art.rows||!art.rows.length)&&<div className="fw-pc-empty">该模板暂无已上传结果（节点领取执行后自动出现）。</div>}
                </div>
              )}
              {art.type==='gap'&&art.item&&(
                <div className="fw-m3-gap">
                  <KVList pairs={[
                    ['缺口',art.item.tplName+' × '+art.item.catName],
                    ['技术类别',art.item.cat,true],
                    ['输入契约',art.item.input,true],
                    ['预计算力',art.item.cost],
                    ['状态',art.item.status==='pending'?'装置已建 · 等结果':'空格 · 无任何结果'],
                    ['该类技术数',art.item.nTech+' 项（ANALYSES 注册表）'],
                    ['说明',art.item.note],
                  ]}/>
                  <button type="button" className="fw-tbtn" style={{marginTop:8}}
                          onClick={()=>{ try{ navigator.clipboard.writeText('python node_agent.py run'); window.alert('领取命令已复制：python node_agent.py run'); }catch(e){ window.alert('领取命令：python node_agent.py run'); } }}>
                    复制领取命令
                  </button>
                </div>
              )}
            </div>
          )}
          {tabIdx===tabs.length-1&&(
            <div className="fw-ai-events">
              {offline&&<div className="fw-ai-empty">后端 :5001 离线——启动 server.py 后，此处接入 /api/ai-rnd/session/events 实时事件流。</div>}
              {!offline&&!events.length&&<div className="fw-ai-empty">启动研发后，结构化事件（目标 / 计划 / 代码生成 / 执行 / 复核 / 回写）将在此实时滚动。</div>}
              {events.slice(-60).reverse().map(ev=>{
                const meta=EVENT_META[ev.type]||['事件','#64748b'];
                return (
                  <div key={ev.id} className="fw-ai-ev">
                    <span className="dot" style={{background:meta[1]}}/><b>{meta[0]}</b><time>{fmtTime(ev.timestamp||ev.created_at)}</time><p>{eventText(ev)}</p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
        <div className="fw-term">
          {(events.length?events.slice(-14).reverse().map(ev=>`[${ev.type||'event'}] ${eventText(ev)}`):DEMO_TERMINAL).map((line,i)=>(
            <div key={i} className={events.length?'fw-m3-tl':''}>{line}</div>
          ))}
        </div>
      </div>

      {/* ===== 右栏：选中概览 + 证据链 + 跨透镜动作（全部数据驱动） ===== */}
      <div className="fw-pr-right">
        <div className="fw-res-card">
          <h5>选中概览 <span className={'fw-src-chip mini '+(rndLive?'live':'demo')}>{rndLive?'LIVE':'DEMO'}</span></h5>
          {qSel&&qSel.src==='rnd'&&(
            <KVList pairs={[
              ['ID',qSel.item.id,true],['标题',qSel.item.title],['状态',qSel.item.status],
              ['KPI',qSel.item.kpi,true],['交付物',qSel.item.deliverable],
              ['seal 记录',qSel.item.seal_record,true],['结果 SHA8',qSel.item.res_sha8,true],
            ]}/>
          )}
          {qSel&&qSel.src==='dist'&&(
            <KVList pairs={[['模板',qSel.item.tm_id,true],['名称',qSel.item.name],['维度',qSel.item.dim],['结果数',qSel.item.results]]}/>
          )}
          {qSel&&qSel.src==='gap'&&(
            <KVList pairs={[['缺口',qSel.item.tplName+' × '+qSel.item.catName],['状态',qSel.item.status==='pending'?'装置已建':'空格'],['契约',qSel.item.input,true]]}/>
          )}
        </div>
        <div className="fw-res-card">
          <h5>证据链</h5>
          <div className="meta">{chain.map((l,i)=><span key={i}>{l}<br/></span>)}</div>
        </div>
        <div className="fw-res-card">
          <h5>跨透镜动作</h5>
          <div className="fw-xact">
            <button className="fw-tbtn" onClick={()=>onGo('spatial')}>在 3D 中查看</button>
            <button className="fw-tbtn" onClick={()=>onGo('data')}>到分析技术运行</button>
          </div>
        </div>
      </div>
    </section>
  );
}
