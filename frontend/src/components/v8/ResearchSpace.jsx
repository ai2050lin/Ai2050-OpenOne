import {useEffect,useMemo,useRef,useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import ResearchInterventionReport from './ResearchInterventionReport.jsx';
import {publishResearchContext} from '../app/researchContext.js';
const COLORS={hypothesis:'#94a3b8',observation:'#8b99aa',heldout:'#2d83c4',intervention:'#db7943',parameter_source:'#305a86',negative:'#9c6457'};
export function nativePoint(object,model){
  const c=object.body,i=c.index||{},l=c.layer;if(!model?.n_layers||!Number.isInteger(l)||l<0||l>=model.n_layers)return null;
  const y=-1.5+3*(l+1)/(model.n_layers+1);
  if(i.space==='mlp'&&Number.isInteger(i.unit)&&i.unit>=0&&i.unit<model.d_ff)return [-.9+1.8*(i.unit%128)/127,y,.15+.5*Math.floor(i.unit/128)/Math.ceil(model.d_ff/128)];
  if(i.space==='attention'&&Number.isInteger(i.head)&&i.head>=0&&i.head<model.n_heads)return [-.8+1.6*(i.head%8)/7,y,-.6+.35*Math.floor(i.head/8)/Math.ceil(model.n_heads/8)];
  if(i.space==='mlp'&&i.field==='residual_write')return [1.25,y,.55];
  if(i.space==='attention'&&i.field==='kv_cache')return [-1.25,y,-.55];
  if(i.space==='residual'&&i.field==='final_readout')return [1.5,y,0];
  const coord=i.coordinate??i.coord;
  if(i.space==='residual'&&Number.isInteger(coord)&&coord>=0&&coord<model.d_model)return [-.9+1.8*(coord%128)/127,['embedding','hidden[0]'].includes(c.boundary)?-1.5:y,-.6+1.2*Math.floor(coord/128)/Math.ceil(model.d_model/128)];
  return null;
}

function ArchiveScene({objects,artifacts,model,selected,onPick}){
  const ref=useRef(null),hits=useRef([]),drag=useRef(null);const [yaw,setYaw]=useState(-.4);
  useEffect(()=>{const cv=ref.current,ctx=cv?.getContext('2d');if(!ctx)return;ctx.clearRect(0,0,900,600);hits.current=[];
    const project=([x,y,z])=>[370+100*(x*Math.cos(yaw)-z*Math.sin(yaw)),300-125*y+25*(x*Math.sin(yaw)+z*Math.cos(yaw))];
    const line=(a,b,color,dash=false)=>{ctx.beginPath();ctx.moveTo(...project(a));ctx.lineTo(...project(b));ctx.strokeStyle=color;ctx.setLineDash(dash?[3,3]:[]);ctx.stroke();ctx.setLineDash([])};
    for(let l=0;model?.n_layers&&l<=model.n_layers;l++){const y=-1.5+3*l/(model.n_layers+1);line([-1,y,-.7],[1,y,-.7],'#b5c6d8');line([-1,y,.7],[1,y,.7],'#d8bdad');line([-1,y,-.7],[-1,y,.7],'#d8dee6');line([1,y,-.7],[1,y,.7],'#d8dee6')}
    const points=new Map(),unlocated=[];
    for(const o of objects.filter(o=>o.type==='feature')){const p=nativePoint(o,model);if(p)points.set(o.id,p);else unlocated.push(o)}
    for(const o of objects.filter(o=>o.type==='edge')){const b=o.body,t=points.get(b.target);if(!t)continue;const s=points.get(b.source)||(b.source?.startsWith('language:')?[-2.1,t[1],-.2]:null);if(s)line(s,t,COLORS[b.evidence?.level]||COLORS.observation,true)}
    function dot(p,o,color,r){const [x,y]=project(p);ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();if(selected?.id===o.id&&selected?.sha===o.sha){ctx.strokeStyle='#172d48';ctx.lineWidth=2;ctx.stroke();ctx.lineWidth=1}hits.current.push({x,y,o})}
    objects.filter(o=>o.type==='feature').forEach(o=>{const p=points.get(o.id);if(p)dot(p,o,COLORS[o.body.evidence?.level]||COLORS.observation,2.6)});
    const rack=[...unlocated,...artifacts.map(a=>({...a,id:a.sha,type:'artifact',body:a}))];rack.forEach((o,i)=>dot([2.4+(i%100)/55,1.5-3*Math.floor(i/100)/Math.max(1,Math.ceil(rack.length/100)-1),.1],o,o.type==='artifact'?'#c3cbd5':'#8b99aa',1.5));
    ctx.fillStyle='#475569';ctx.font='13px sans-serif';ctx.fillText('外部语言关系',45,55);ctx.fillText('真实原生层 / Head / MLP 索引',250,55);ctx.fillText(`产物架 · ${rack.length}项`,610,55);
    ctx.font='12px sans-serif';ctx.fillText(`全部显示 ${points.size} 个已定位卡；${unlocated.length} 个未定位卡；${artifacts.length} 个产物版本`,35,555);
    ctx.fillText('板内按原生索引定位，层级写出和缓存标在板侧。距离仅用于布局，边按登记证据绘制。',35,580);
  },[objects,artifacts,model,yaw,selected]);
  return <canvas ref={ref} width="900" height="600" style={{width:'100%',touchAction:'none',cursor:'grab'}} aria-label="全部归档成果3D索引：原生参数定位与未定位产物架"
    onPointerDown={e=>{drag.current={x:e.clientX,moved:false};e.currentTarget.setPointerCapture(e.pointerId)}} onPointerMove={e=>{if(drag.current&&e.buttons){const dx=e.clientX-drag.current.x;if(Math.abs(dx)>2)drag.current.moved=true;setYaw(v=>v+dx/160);drag.current.x=e.clientX}}}
    onPointerUp={e=>{if(drag.current&&!drag.current.moved){const b=e.currentTarget.getBoundingClientRect(),x=(e.clientX-b.left)*900/b.width,y=(e.clientY-b.top)*600/b.height;const h=hits.current.filter(h=>Math.hypot(h.x-x,h.y-y)<8).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0];if(h)onPick(h.o)}drag.current=null}}/>;
}

export default function ResearchSpace(){
  const index=useApi('/api/traces/research-space');const [snapshot,setSnapshot]=useState(()=>new URLSearchParams(window.location.search).get('snapshot')||''),[run,setRun]=useState(()=>new URLSearchParams(window.location.search).get('run')||''),[selected,setSelected]=useState(null),[mode,setMode]=useState('study'),[page,setPage]=useState(0),[search,setSearch]=useState(''),[showArchive,setShowArchive]=useState(false);
  const data=index.data;
  const visibleStudies=data?.studies.filter(s=>!snapshot||!s.model_snapshot||s.model_snapshot===snapshot)||[];
  const study=visibleStudies.find(s=>s.run_id===run);
  const activeSnapshot=study?.model_snapshot||snapshot;
  const model=data?.models.find(m=>m.snapshot===activeSnapshot);
  const objects=useMemo(()=>data?.objects.filter(o=>(!activeSnapshot||o.model_snapshot===activeSnapshot)&&(!run||o.body.run_id===run))||[],[data,activeSnapshot,run]);
  const artifacts=data?.artifacts||[],filtered=artifacts.filter(a=>!search||[a.source,a.sha,a.model,a.kind].join(' ').toLowerCase().includes(search.toLowerCase()));
  const selectedSha=selected?.type==='artifact'?selected.sha:selected?.body.evidence?.artifact_sha256;
  const raw=useApi(selectedSha?`/api/research-library/artifacts/${selectedSha}`:null);
  useEffect(()=>{publishResearchContext({snapshot,run})},[snapshot,run]);
  if(index.error)return <Empty>{index.error}；归档索引未连接，不显示模拟图谱。</Empty>;
  if(!data)return <Empty>读取全部归档研究与原生定位…</Empty>;
  const selectStudy=key=>{
    const next=visibleStudies.find(s=>s.study_key===key);
    if(next?.model_snapshot)setSnapshot(next.model_snapshot);
    setRun(next?.run_id||'');setSelected(null);setMode('study');
  };
  return <div className="v8-route"><section className="v8-card"><h2>研究成果 · 3D总图</h2><p className="v8-note">全部已归档成果都有可回查入口。已定位特征落在对应模型的层与单元索引；尚无定位的结果放在产物架，保留失败、反例和历史版本。存有统计不等于已破解。</p>
    <div className="v6-toolbar"><select aria-label="成果图模型快照" value={snapshot} onChange={e=>{setSnapshot(e.target.value);setRun('');setSelected(null);publishResearchContext({trace:'',context_trace:''})}}><option value="">全部模型与未定位成果</option>{snapshot&&!data.models.some(m=>m.snapshot===snapshot)&&<option value={snapshot}>{snapshot.slice(0,16)} · 无原生维度</option>}{data.models.map(m=><option key={m.snapshot} value={m.snapshot}>{m.snapshot.slice(0,16)}{m.n_layers?' · '+m.n_layers+'层':' · 无原生维度'}</option>)}</select><select aria-label="成果图研究工作包" value={study?.study_key||''} onChange={e=>selectStudy(e.target.value)}><option value="">选择研究工作包（{visibleStudies.length}）</option>{visibleStudies.map(s=><option key={s.study_key} value={s.study_key}>{s.title}{s.model_snapshot?` · ${s.model_snapshot.slice(0,8)}`:''}</option>)}</select><button className="fw-tbtn" onClick={index.reload}>刷新全部成果</button></div>
    <p className="v8-note">{data.counts.studies} 个工作包，{data.counts.objects} 个最新图谱对象，{data.counts.artifacts} 个产物版本；原始产物架保留全部模型，按工作包筛选的是原生定位卡。当前模型快照不与其他模型平均。</p>
    <label className="v8-note"><input type="checkbox" checked={showArchive} onChange={e=>setShowArchive(e.target.checked)}/> 叠加历史产物架（档案排列，不是机制节点）</label>
    {!model?.n_layers&&<p className="v8-note">选择具体模型快照可查看原生层与参数定位。全部模型的报告和产物均可从下方打开。</p>}
    {(model?.n_layers||showArchive)&&<ArchiveScene objects={objects} artifacts={showArchive?artifacts:[]} model={model} selected={selected} onPick={o=>{setSelected(o);setMode('evidence')}}/>}
    <div className="v6-toolbar">{Object.entries(COLORS).map(([k,c])=><span key={k} style={{color:c}}>● {({hypothesis:'假设',observation:'观察',heldout:'留出',intervention:'范围干预',parameter_source:'参数来源',negative:'负结果'})[k]}</span>)}</div></section>
    <section className="v8-card"><div className="v6-toolbar">{[['study','研究报告与实验'],['features','定位卡与关联'],['evidence','选中证据'],['archive','全部产物版本']].map(([k,t])=><button className="fw-tbtn" key={k} onClick={()=>setMode(k)} aria-pressed={mode===k}>{t}</button>)}</div>
      {mode==='study'&&<>{!study?<Empty>选择一个工作包，直接在空间内阅读报告、剂量、能力保持和反例。</Empty>:<><p><strong>{study.title}</strong></p><p className="v8-note">{study.localization==='native_binding'?'已绑定真实原生轨迹':study.localization==='native_bank'?'原生归档已接入。采集层、字段和样本范围见研究报告。':study.localization==='native_aggregate'?'全量原生单元统计已接入；组统计与单样本轨迹分别回查。':'当前只有研究元数据；不补画未采集的参数或激活。'}</p>{study.traces.map(t=><p key={t.trace_id}><a href={`?lens=spatial&view=live&trace=${t.trace_id}&tblock=${t.layer||0}&thead=${t.head||0}&tunit=${t.unit||0}`}>{t.sample_id||t.trace_id.slice(0,12)} → 原始输入、完整数组与参数路径</a></p>)}{study.report_sha256?<ResearchInterventionReport rawFallback sha={study.report_sha256} planSha={study.stages.native_mlp_plan?.plan_sha256} capabilityInputsSha={study.stages.client_acceptance?.capability_inputs_sha256}/>:<Empty>尚未登记正式报告；阶段与原始产物仍保留。</Empty>}<details><summary>全部阶段与状态</summary><pre className="v8-artifact-json">{JSON.stringify(study.stages,null,2)}</pre></details></>}</>}
      {mode==='features'&&<><p className="v8-note">当前范围内 {objects.length} 个定位卡或关联均可回查，包括未定位和负结果。</p><select aria-label="成果图定位卡" value={selected?`${selected.sha}:${selected.id}`:''} onChange={e=>{setSelected(objects.find(o=>`${o.sha}:${o.id}`===e.target.value));setMode('evidence')}}><option value="">选择定位卡或关联</option>{objects.map(o=><option key={`${o.sha}:${o.id}`} value={`${o.sha}:${o.id}`}>{o.body.label||o.body.relation||o.id} · {o.body.evidence.level}</option>)}</select></>}
      {mode==='evidence'&&<>{selected?<><p><strong>{selected.body.label||selected.body.source||selected.id}</strong></p>{selected.body.scope?.trace_id&&<a href={`?lens=spatial&view=live&trace=${selected.body.scope.trace_id}&tblock=${selected.body.layer||0}&tunit=${selected.body.index?.unit||0}&thead=${selected.body.index?.head||0}&tpos=${selected.body.scope.position??""}`}>打开该卡对应真实运行</a>}<details><summary>定位记录 JSON</summary><pre className="v8-artifact-json">{JSON.stringify(selected.body,null,2)}</pre></details><details><summary>完整原始证据 · {selectedSha}</summary>{raw.error?<Empty>{raw.error}</Empty>:<pre className="v8-artifact-json">{JSON.stringify(raw.data,null,2)}</pre>}</details></>:<Empty>点击3D中的节点，或从定位卡和全部产物中选择。</Empty>}</>}
      {mode==='archive'&&<><label>检索全部产物 <input aria-label="成果产物检索" value={search} onChange={e=>{setSearch(e.target.value);setPage(0)}} placeholder="来源、快照或SHA"/></label><p className="v8-note">{filtered.length} 个匹配版本，均可逐页回查；未产生定位的结果仍保留。</p>{filtered.slice(page*30,(page+1)*30).map(a=><p key={a.sha}><button className="fw-tbtn" onClick={()=>{setSelected({...a,id:a.sha,type:'artifact',body:a});setMode('evidence')}}>{a.source} · {a.sha.slice(0,10)}</button></p>)}<button className="fw-tbtn" disabled={!page} onClick={()=>setPage(page-1)}>上一页</button><button className="fw-tbtn" disabled={(page+1)*30>=filtered.length} onClick={()=>setPage(page+1)}>下一页</button></>}
    </section></div>;
}
