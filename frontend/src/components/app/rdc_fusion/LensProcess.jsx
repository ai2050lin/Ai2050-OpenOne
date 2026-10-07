/* 过程透镜 v2：AI 自动研发工作台
   参考旧版 LoopEngineeringWorkspace：主研发模型 + 多个独立分析模型、
   五证据门循环（缺口→契约→执行→复核→回写）、SSE 实时事件流、自动/手动执行。
   后端：:5001 /api/ai-rnd/*（server/ai_rnd_service.py，真实可用）；
   离线时表单仍可编辑，状态条显示「后端离线」，不阻塞页面。
   demo 部分（任务队列/代码节选/结果卡）为真实研究数值，接入点见各注释。 */
import { useCallback, useEffect, useRef, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

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

/* 任务队列（demo 接入点 ← phase_queue.json：Q03–Q06 已 seal，Q07 next） */
const QUEUE=[
  {id:'Q07',name:'KPI 曲线 v0 汇总',pill:'NEXT',cls:'fw-pill-next',on:true},
  {id:'Q06',name:'C_steer 基座',pill:'SEALED',cls:'fw-pill-done'},
  {id:'Q05',name:'E_ar(k) 测量',pill:'SEALED',cls:'fw-pill-done'},
  {id:'Q04',name:'E_ar 装置',pill:'SEALED',cls:'fw-pill-done'},
  {id:'Q03',name:'E_read 基线',pill:'SEALED',cls:'fw-pill-done'},
];
const TERM_LINES=[
  [['$ ','dim'],['python q05_ar_sweep/collect_ar.py --all-arms --K 16','']],
  [['[Q05] ','g'],['D4 bridge max|Δrel| = 0.0489 ',''],['<',''],[' 0.05 ',''],['PASS','g']],
  [['[Q05] ','g'],['shape 4b=',''],['flat','y'],[' · 14B=',''],['saturating','y'],[' · 9B=',''],['flat','y']],
  [['[Q06] ','g'],['C_steer_main = 0.0000 · 95%CI ≤ 1.0% · collateral clean 93.3%','']],
  [['[review] ','g'],['independent TOTAL PASS=47 FAIL=0 → sealed ',''],['1acb1e78','dim']],
  [['$ ','dim'],['_','']],
];

function gateId(phase,status){
  if(!phase) return status==='stopped'?'writeback':'gap';
  const g=GATES.find(x=>x.phases.includes(phase));
  return g?g.id:'gap';
}
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

  const api=useCallback(async(path,opts)=>{
    let r;
    try{ r=await fetch(`${API_BASE}/api/ai-rnd${path}`,opts); }
    catch{ setOffline(true); throw new Error('后端 :5001 不可达（server.py 未启动？）'); }
    const p=await r.json().catch(()=>({}));
    if(!r.ok) throw new Error(p.detail||('HTTP '+r.status));
    return p;
  },[]);

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
        setTab(3);
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

  const active=Boolean(agent&&agent.enabled);
  const running=status.status==='running';
  const masterReady=Boolean(config.master_model.api_key&&config.master_model.api_key.trim());
  const analystReady=config.analyst_models.filter(m=>m.api_key&&m.api_key.trim()).length;
  const ready=masterReady&&analystReady>0;
  const activeGate=GATES.findIndex(g=>g.id===gateId(status.current_phase,status.status));

  return (
    <section className={'fw-view fw-process'+(on?' on':'')}>
      {/* ===== 左栏：AI 模型配置 + 任务队列 ===== */}
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

        <div className="fw-pr-sec">任务队列</div>
        {QUEUE.map(q=>(
          <button key={q.id} type="button" className={'fw-q-row'+(q.on?' on':'')} onClick={()=>window.alert(q.id+' '+q.name+'：'+q.pill+'（详情接入 phase_queue.json 后开放）')}>
            <span className="id">{q.id}</span>{q.name}<span className={'fw-pill '+q.cls}>{q.pill}</span>
          </button>
        ))}
        <div className="fw-pr-sec">工作区 tests/deepseek/</div>
        <span className="fw-file dir">▾ q05_ar_sweep/</span>
        <span className="fw-file">collect_ar.py</span>
        <span className="fw-file">metric_dict_v4.json</span>
        <span className="fw-file">review_report.txt</span>
        <span className="fw-file dir">▸ q06_c_steer/</span>
        <span className="fw-file dir">▸ shared/</span>
        <div className="fw-pr-sec">运行配置</div>
        <div className="fw-q-row" style={{cursor:'default'}}><span className="id" style={{width:'auto'}}>4b bf16 · 14B/9B NF4</span></div>
        <div className="fw-q-row" style={{cursor:'default'}}><span className="id" style={{width:'auto'}}>drift 断言 · 预注册冻结</span></div>
      </div>

      {/* ===== 中栏：目标 composer + 证据门 + 代码/事件 ===== */}
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
          {['collect_ar.py','result.json','review_report.txt','实时事件'].map((t,i)=>(
            <button key={t} type="button" className={'fw-ed-tab'+(tab===i?' on':'')} onClick={()=>setTab(i)}>{t}</button>
          ))}
        </div>
        <div className="fw-ed-body">
          {tab===0&&(
            <pre style={{margin:0,font:'inherit'}}>
<span className="cm"># Q05 · E_ar(k) 正式测量 — 四臂 738×K16（节选）</span>{'\n'}
<span className="kw">def</span> <span className="fn">sweep_ar</span>(model, K=<span className="num">16</span>, arms=(<span className="st">&quot;is_a&quot;</span>, <span className="st">&quot;attr&quot;</span>, <span className="st">&quot;syntax&quot;</span>, <span className="st">&quot;rand&quot;</span>)):{'\n'}
{'    '}share = <span className="fn">exact_additive_budget</span>(model){'          '}<span className="cm"># 铁律 (a) 精确可加向量预算</span>{'\n'}
{'    '}<span className="kw">for</span> arm <span className="kw">in</span> arms:{'\n'}
{'        '}E = [share.<span className="fn">write</span>(arm, k=k) <span className="kw">for</span> k <span className="kw">in</span> <span className="fn">range</span>(<span className="num">1</span>, K+<span className="num">1</span>)]{'\n'}
{'        '}rel = (E[<span className="num">0</span>] - E) / E[<span className="num">0</span>]{'                  '}<span className="cm"># S_rel 归一化</span>{'\n'}
{'        '}<span className="kw">yield</span> arm, rel{'\n'}
{'\n'}
<span className="cm"># D4 桥（4b bf16 ↔ 4-bit NF4 口径桥）</span>{'\n'}
<span className="kw">assert</span> <span className="fn">max</span>(<span className="fn">abs</span>(rel_bf16 - rel_nf4)) &lt; <span className="num">0.05</span>{'   '}<span className="cm"># PASS: 0.0489</span>
            </pre>
          )}
          {tab===1&&(
            <pre style={{margin:0,font:'inherit'}}>
{'{'}{'\n'}
{'  '}&quot;phase&quot;: <span className="st">&quot;Q05&quot;</span>, <span className="st">&quot;status&quot;</span>: <span className="st">&quot;measured&quot;</span>,{'\n'}
{'  '}&quot;d4_bridge_max_delta_rel&quot;: <span className="num">0.0489</span>,{'\n'}
{'  '}&quot;shape&quot;: {'{'}<span className="st">&quot;4b&quot;</span>: <span className="st">&quot;flat&quot;</span>, <span className="st">&quot;14b&quot;</span>: <span className="st">&quot;saturating&quot;</span>, <span className="st">&quot;9b&quot;</span>: <span className="st">&quot;flat&quot;</span>{'}'},{'\n'}
{'  '}&quot;s_rel_min&quot;: [<span className="num">0.6858</span>, <span className="num">0.6939</span>, <span className="num">0.7392</span>],{'\n'}
{'  '}&quot;sealed_sha&quot;: <span className="st">&quot;1acb1e78&quot;</span>,{'\n'}
{'  '}&quot;review&quot;: {'{'}<span className="st">&quot;pass&quot;</span>: <span className="num">47</span>, <span className="st">&quot;fail&quot;</span>: <span className="num">0</span>{'}'}{'\n'}
{'}'}
            </pre>
          )}
          {tab===2&&(
            <pre style={{margin:0,font:'inherit',whiteSpace:'pre-wrap'}}>
<span className="cm"># 独立复核结论（节选）</span>{'\n'}
装置门 4/4 通过；份额全程使用精确可加向量预算；{'\n'}
冻结锚逐位复现（drift 0.00e+00）；D4 桥在预注册门内。{'\n'}
未支持结论：k→∞ 外推、跨层迁移、权重级因果证明。
            </pre>
          )}
          {tab===3&&(
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
          {TERM_LINES.map((line,i)=>(
            <div key={i}>{line.map(([txt,cls],j)=><span key={j} className={cls||''}>{txt}</span>)}</div>
          ))}
        </div>
      </div>

      {/* ===== 右栏：结果卡 + 跨透镜动作 + 证据链 ===== */}
      <div className="fw-pr-right">
        <div className="fw-res-card">
          <h5>E_ar(k) 消融曲线 <span className="fw-pill fw-pill-run" style={{background:'#d1fae5'}}>v4 口径</span></h5>
          <svg viewBox="0 0 260 110" style={{width:'100%',display:'block'}}>
            <line x1="30" y1="95" x2="250" y2="95" stroke="#e2e8f0"/>
            <line x1="30" y1="95" x2="30" y2="8" stroke="#e2e8f0"/>
            <text x="10" y="20" fontSize="8" fill="#94a3b8">S_rel</text>
            <text x="225" y="107" fontSize="8" fill="#94a3b8">k</text>
            <path d="M30 28 L60 29 L90 30 L120 31 L150 32 L180 33 L210 34 L240 35" fill="none" stroke="#0284c7" strokeWidth="2"/>
            <path d="M30 28 L62 36 L94 48 L126 60 L158 71 L190 81 L222 88 L240 90" fill="none" stroke="#6366f1" strokeWidth="2" strokeDasharray="4 3"/>
            <path d="M30 24 L60 25 L90 26 L120 27 L150 28 L180 29 L210 30 L240 31" fill="none" stroke="#10b981" strokeWidth="2"/>
            <text x="150" y="24" fontSize="8.5" fill="#0284c7">4b flat</text>
            <text x="150" y="56" fontSize="8.5" fill="#6366f1">14B saturating</text>
            <text x="150" y="45" fontSize="8.5" fill="#059669">9B flat</text>
          </svg>
          <div className="meta">形状 <b>4b flat / 14B saturating / 9B flat</b><br/>含义：4b 中 is-a 关系沿 k <b>无递减</b> → 关系非碎片化存储</div>
        </div>
        <div className="fw-res-card">
          <h5>跨透镜动作</h5>
          <div className="fw-xact">
            <button className="fw-tbtn" onClick={()=>onGo('spatial')}>在 3D 中查看</button>
            <button className="fw-tbtn" onClick={()=>onGo('progress')}>登记到路线图</button>
          </div>
        </div>
        <div className="fw-res-card">
          <h5>证据链</h5>
          <div className="meta">Q03 E_read <b>0.331615</b>（4b 基线）<br/>Q04 装置 <b>device_built</b> → Q05 <b>measured</b><br/>Q06 C_steer <b>0.0000</b>（无定向杠杆）<br/>队列 sealed <b>1acb1e78</b> · 复核 47/0</div>
        </div>
      </div>
    </section>
  );
}
