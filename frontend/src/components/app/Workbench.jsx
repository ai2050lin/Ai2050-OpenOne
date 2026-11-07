/* 正式研究工作台：进展 / 空间 / 研发 / 配置。
   图谱浏览统一语言模式、内部特征与跨族映射；结构参数与原生特征共享入口。
   顶栏提供本机算力与数据同步，旧本机工具仅在配置的高级兼容区访问。
   专业工位按角色与运行形态开放；旧链接归一到新路由，浏览历史保留对象身份。 */
import { Fragment, Suspense, lazy, useEffect, useRef, useState, useSyncExternalStore } from 'react';

import AccountMenu from './rdc_fusion/AccountMenu.jsx';
import AtlasView from './rdc_fusion/AtlasView.jsx';
import CensusPanel from './rdc_fusion/CensusPanel.jsx';
import ConfigView, { CFG_SECTIONS } from './ConfigView.jsx';
import Gctx from './Gctx.jsx';
import LocalPackView from './rdc_fusion/LocalPackView.jsx';
import TemplatePipeline from './rdc_fusion/TemplatePipeline.jsx';
import { ADMIN_LIVE, API_BASE, DESKTOP, OFFLINE_CLIENT, useApi } from './rdc_fusion/api.js';
import { useAuth } from './rdc_fusion/authContext.js';
import { loadLocalPack } from './rdc_fusion/localPack.js';
import { Empty } from './rdc_fusion/ui.jsx';
import RouteLens from '../v8/RouteLens.jsx';
import SpaceLens from '../v8/SpaceLens.jsx';
import WorkbenchHelp from './WorkbenchHelp.jsx';
import { useClientMode } from '../v8/ClientGuide.jsx';
import { readState as readSpaceState, writeState as writeSpaceState } from '../v8/v8Util.js';
import { ENC_TABS, LENSES, PIPE_STEPS, ROUTE_VIEWS, SPACE_TARGET, SPATIAL_MODES, SPATIAL_VIEWS, dsBound, normalizeState, readState, resolveGo, spaceMode, spaceTab, visibleStations, writeState } from './workbenchUtil.js';
import {useResearchContext} from './researchContext.js';
import {downloadDataPack,packInfo,storeTick,subscribe} from './rdc_fusion/dataStore.js';
import {ExtraPackPrompt} from '../open/DataBar.jsx';
import './rdc_fusion/rdc_fusion.css';
import './rdc_fusion/rdc_fusion_v6.css';
import './workbench.css';
import ResearchExecution from '../v8/ResearchExecution.jsx';
import ResearchCatalog from '../v8/ResearchCatalog.jsx';

const AiChatDrawer = lazy(() => import('./rdc_fusion/AiChatDrawer.jsx'));
const AutoTestPanel = lazy(() => import('./rdc_fusion/AutoTestPanel.jsx'));
const GoodfireProbePanel = lazy(() => import('./rdc_fusion/GoodfireProbePanel.jsx'));
const LensSpatial = lazy(() => import('./rdc_fusion/LensSpatial.jsx'));

function LocalPack() {
  const [pack, setPack] = useState(null);
  const [err, setErr] = useState('');
  const open = async e => {
    const f = e.target.files?.[0];
    if (!f) return;
    setErr('');
    try { setPack(await loadLocalPack(f)); } catch (reason) { setErr(reason.message); setPack(null); }
  };
  return (
    <div className="wb-local">
      <div className="v6-toolbar">
        <label className="fw-tbtn">打开本地包（.zip）<input type="file" accept=".zip" onChange={open} hidden /></label>
        {pack && <span className="v6-note">{pack.name} · {pack.kind === 'respack' ? '研究包' : '谱图包'}</span>}
        {err && <span className="v6-note warn">{err}</span>}
      </div>
      {!pack && <Empty>研究包（实验响应页「下载研究包」）与谱图包（语言谱图页「导出谱图包」）都在浏览器内解压查看，不上传到任何服务器。</Empty>}
      {pack?.kind === 'respack' && <LocalPackView pack={pack} />}
      {pack?.kind === 'atlaspack' && <AtlasView localAtlas={pack.atlas} localName={pack.name} />}
    </div>
  );
}

function ImportDataDialog({onClose}){
  const ref=useRef(null);
  useEffect(()=>{const el=ref.current;el.showModal();return()=>el.close()},[]);
  return <dialog ref={ref} className="wb-help-modal wb-data-dialog" aria-label="数据与同步" onCancel={e=>{e.preventDefault();onClose()}} onClick={e=>{if(e.target===e.currentTarget)onClose()}}><header className="wb-help-head"><h2>数据与同步</h2><button className="fw-tbtn" autoFocus onClick={onClose}>关闭数据面板</button></header><DataSyncActions/><p className="v8-note">已有研究包或谱图包只在本机浏览器解压。原始模型张量仍保存在采集节点；导入不会提升证据等级。</p><LocalPack/></dialog>;
}

function DataSyncActions(){
  useSyncExternalStore(subscribe,storeTick);
  const info=packInfo(),remote=useApi('/api/platform/config').data?.datapack;
  const [message,setMessage]=useState(''),[busy,setBusy]=useState(false);
  async function update(){setBusy(true);try{await downloadDataPack(API_BASE,(got,total)=>setMessage(total?`下载 ${Math.round(100*got/total)}%`:`已下载 ${got} bytes`));setMessage('数据包已更新，相关页面将读取新版本。')}catch(e){setMessage(e.message)}finally{setBusy(false)}}
  return <section className="v8-card"><strong>中心研究数据</strong><p className="v8-note">本机版本：{info.version||'未下载'}；中心版本：{remote?.version||'未发布或不可达'}。完整原场和快照按各自产物记录回查。</p><button className="fw-tbtn" disabled={busy||!remote?.version} onClick={update}>{busy?'下载中…':'下载或更新主数据包'}</button><p className="v8-note" role="status">{message}</p><ExtraPackPrompt pack="research-library"/></section>;
}

/* 左轨图标（v5 工作台同款线性图标） */
function LensIcon({ id }) {
  const p = {
    home: <path d="M3 10.5 12 3l9 7.5V21H3z" />,
    spatial: <><circle cx="12" cy="12" r="3" /><ellipse cx="12" cy="12" rx="10" ry="4.2" /><ellipse cx="12" cy="12" rx="10" ry="4.2" transform="rotate(60 12 12)" /></>,
    process: <path d="m8 6-6 6 6 6M16 6l6 6-6 6M13 4l-3 16" />,
    route: <><path d="M2 12h5M9 12h5M17 8l5 4-5 4V8z" /><circle cx="7" cy="12" r="2.5" /><circle cx="15" cy="12" r="2.5" /></>,
    config: <><path d="M3 7.5h3.5M11 7.5h10M3 16.5h10M17.5 16.5h3.5" /><circle cx="9" cy="7.5" r="2.1" /><circle cx="15" cy="16.5" r="2.1" /></>,
    local: <><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>,
  }[id];
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">{p}</svg>;
}

function SubNav({ items, value, onChange, label, onHelp }) {
  return (
    <nav className="wb-subnav" aria-label={label}>
      {items.map(it => (
        <button key={it.id} type="button" className={value === it.id ? 'on' : ''} onClick={() => onChange(it.id)} title={it.desc || it.label}>
          {it.label}
        </button>
      ))}
      {onHelp && (
        <button type="button" className="wb-help-btn" onClick={onHelp} title="本页功能与原理说明" aria-label="本页功能与原理说明">?</button>
      )}
    </nav>
  );
}

export default function Workbench() {
  const cfg = useApi('/api/platform/config');
  const localTools = Boolean(cfg.data?.local_tools);
  const [st, setSt] = useState(readState);
  const latestState=useRef(st);latestState.current=st;
  const [helpOpen, setHelpOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  /* 地址栏已带 lens / view / run / snapshot 等定位参数，复制它就是分享当前对象 */
  const copyLink = () => navigator.clipboard?.writeText(window.location.href)
    .then(() => { setCopied(true); setTimeout(() => setCopied(false), 1600); }).catch(() => setCopied(false));
  const [clientMode, setClientMode] = useClientMode();
  const [revision,setRevision]=useState(0),context=useResearchContext();
  useEffect(()=>{const pop=()=>{setSt(readState());setRevision(n=>n+1)};window.addEventListener('popstate',pop);return()=>window.removeEventListener('popstate',pop)},[]);
  // 网页的实验管理工位保持只读。正式研究包上传由登录账号的管理员权限单独控制。
  const { user } = useAuth();
  const isAdmin = !OFFLINE_CLIENT || user?.role === 'admin';
  const stations = visibleStations({ live: ADMIN_LIVE, admin: isAdmin, localTools });
  const baseLenses = !ADMIN_LIVE
    ? LENSES.map(l => (l.id === 'process' ? { ...l, desc: '浏览研究与任务，管理员可登记研究包' } : l))
    : LENSES;
  const lenses = baseLenses;
  const lens = lenses.some(l => l.id === st.lens) ? st.lens : 'spatial';
  const station = stations.some(s => s.id === st.station) ? st.station : (stations[0]?.id || 'catalog');
  const pstep = PIPE_STEPS.some(s => s.id === st.pstep) ? st.pstep : 'design';
  const csec = CFG_SECTIONS.some(s => s.id === st.csec) || ['platform','legacy'].includes(st.csec) ? st.csec : 'conn';
  // 旧空间入口在 readState / normalizeState 中归一为图谱图层或参数子视图。
  const view = SPATIAL_VIEWS.some(v => v.id === st.view) ? st.view : 'graph';
  const researchGraph = view==='graph'&&st.gview!=='structure';
  const rtab = ROUTE_VIEWS.some(v => v.id === st.rtab) || st.rtab === 'neuron' ? st.rtab : 'overview';
  const stationGroup=['plan','catalog'].includes(station)?station:'execution';
  const dsOn = dsBound(lens, { view, gview:st.gview });
  const smode = spaceMode(view, st.gview);

  useEffect(() => { writeState({ ...st, lens, station, view, rtab, pstep, csec }); }, [st, lens, station, view, rtab, pstep, csec]);
  const lensLabel = lenses.find(l => l.id === lens)?.label || '';
  useEffect(() => { document.title = `研究工作台 · ${lensLabel} · AI2050`; }, [lensLabel]);

  const commit = (next, context) => {
    const normalized=normalizeState(next);
    latestState.current=normalized;
    writeState(normalized,{push:true,...context});
    setSt(normalized);
  };
  const go = (a, b, c, {run,snapshot,...extra}={}) => commit(resolveGo(latestState.current,a,b,c,extra),{run,snapshot});
  const set = patch => commit({...latestState.current,...patch});
  /* 进展 → 空间：先把空间内部状态（语言项、块等）写进地址栏，空间挂载时读取 */
  const goSpace = (v, patch = {}, model = st.mdl) => {
    writeSpaceState({ ...readSpaceState(), ...patch });
    set({ lens: 'spatial', view: v, mdl: model });
  };

  return (
    <div className="fw-root wb-root" data-client-mode={clientMode}>
      <header className="wb-top">
        <a className="wb-brand" href={window.location.protocol === 'file:' ? window.location.pathname : '/'}><span className="op-logo" aria-hidden="true">A2</span><span><b>研究工作台</b><small>返回首页</small></span></a>
        <span className="wb-lens-title">{lensLabel}<small>{lenses.find(l => l.id === lens)?.desc}</small></span>
        <div className="wb-top-right">
          <div className="wb-mode" role="group" aria-label="显示模式">{[['simple','简明'],['pro','专业']].map(([id,label])=><button type="button" key={id} aria-pressed={clientMode===id} onClick={()=>setClientMode(id)}>{label}</button>)}</div>
          <button type="button" className="fw-tbtn" onClick={()=>set({lens:'config',csec:'node'})}>贡献算力</button>
          <button type="button" className="fw-tbtn" onClick={()=>set({importing:true})}>数据与同步</button>
          {cfg.data?.sandbox_synthetic && <span className="op-sandbox">合成·沙盒</span>}
          <span className="wb-center wb-pro" title="当前连接的中心服务">{API_BASE || window.location.origin}</span>
          <AccountMenu />
        </div>
      </header>
      {cfg.error && <div className="op-banner" role="alert">连不上中心服务：{cfg.error}</div>}
      <div className="wb-frame">
        {/* 左轨：一级界面（沿用 v5 工作台的图标轨布局）。分隔线之上是研究对象与实验生产，之下是形态相关的界面 */}
        <nav className="fw-rail" aria-label="工作台界面">
          {lenses.map(l => (
            <Fragment key={l.id}>
              {l.id === 'config' && <div className="fw-rdiv" />}
              {l.id === 'local' && <div className="fw-rdiv" />}
              <button type="button" className={'fw-rbtn' + (lens === l.id ? ' on' : '')} onClick={() => set(l.id === 'process' && lens !== 'process' ? { lens: 'process', station: 'plan' } : { lens: l.id })}
                      title={`${l.label}：${l.desc}`} aria-current={lens === l.id ? 'page' : undefined}>
                <LensIcon id={l.id} />{l.label}
              </button>
            </Fragment>
          ))}
        </nav>
        <div className="wb-body">
          {lens === 'route' && <SubNav items={ROUTE_VIEWS} value={rtab==='neuron'?'overview':rtab} onChange={v => set({ rtab: v, opane: '' })} label="进展视图" onHelp={() => setHelpOpen(true)} />}
          {lens === 'spatial' && <SubNav items={SPATIAL_MODES} value={smode}
            onChange={v => set(SPACE_TARGET[v])} label="空间图谱" onHelp={() => setHelpOpen(true)} />}
          {lens === 'spatial' && <div className="wb-space-tabs" role="group" aria-label="本图的视图">
            {smode === 'enc' && ENC_TABS.map(tab => <button type="button" key={tab.id}
              aria-pressed={spaceTab(view, st.gview) === tab.id} onClick={() => set(tab.patch)}>{tab.label}</button>)}
            <button type="button" className="v6-linkbtn wb-copy-link" onClick={copyLink}>{copied ? '链接已复制' : '复制当前链接'}</button>
          </div>}
          {lens === 'process' && <SubNav items={stations.filter(s=>['plan','catalog','execution'].includes(s.id))} value={stationGroup}
            onChange={s => set({ station: s, mrun: '', ...(s === 'catalog' ? { tm: '' } : {}) })} label="研发工位" onHelp={() => setHelpOpen(true)} />}
          {/* 配置的分段与其它界面用同一条子导航：连接 / 账号与节点 / 模型与 AI / 平台 */}
          {lens === 'config' && <SubNav items={CFG_SECTIONS} value={csec} onChange={v => set({ csec: v })} label="配置分段" onHelp={() => setHelpOpen(true)} />}
          {/* 图谱覆盖在摘要旁选模型，在数据集实测里选题库；空间继续使用全局维度条。 */}
          {!(lens==='process'&&station==='plan') && (context.run || context.trace || context.context_trace || (context.snapshot && (lens !== 'route' || !['overview','lang'].includes(rtab)))) && <div className="wb-research-context wb-pro" aria-label="当前研究上下文"><span>研究：{context.run||'未绑定实验'}</span>{context.snapshot&&<span title={context.snapshot}>来源快照 {context.snapshot.slice(0,16)} · 页面内核验</span>}{(context.trace||context.context_trace)&&view!=='live'&&<button type="button" className="v6-linkbtn" onClick={()=>set({lens:'spatial',view:'live'})}>回到来源样本</button>}</div>}
          {lens === 'spatial' && view==='graph'&&['langmeasure','relations'].includes(st.gview) && (
            <Gctx ds={st.ds || ''} mdl={st.mdl || ''} dsBound={dsOn} onDs={v => set({ ds: v })} onMdl={v => set({ mdl: v })}
                  scope={view === 'graph' ? '当前图层的数据来源与被测模型；不同快照分别查看'
                    : '本页只随模型变：读的是已载入权重，普查读自然语料（1,900 万 token）'} />
          )}
          <main className="wb-main">
            {lens==='config'&&<details className="v8-card"><summary>高级与兼容工具</summary><button className="fw-tbtn" onClick={()=>set({csec:'platform'})}>平台信息</button>{localTools&&<button className="fw-tbtn" onClick={()=>set({csec:'legacy'})}>旧版兼容与教学工具</button>}</details>}
            {lens==='process'&&stationGroup==='execution'&&station!=='execution'&&<button className="fw-tbtn" type="button" onClick={()=>go('execution')}>返回实验与队列</button>}
            <Suspense fallback={<Empty>加载中…</Empty>}>
              {lens==='process'&&['plan','execution'].includes(station)&&<ResearchExecution plan={station==='plan'} onGo={go}
                tools={stations.filter(s=>!['plan','execution','catalog'].includes(s.id))}/>}
              {lens === 'spatial' && <SpaceLens key={`${revision}:${researchGraph?st.gview:'workbench'}`} view={view} graphMode={st.gview}
                                                  onGraphMode={gview=>set({view:'graph',gview})} paramTab={st.ptab} onParamTab={ptab=>set({ptab})} setView={v => set({ view: v })} tm={st.tm} setTm={tm => set({ tm })}
                                                  model={st.mdl || ''} onModel={mdl=>set({mdl})} ds={st.ds || ''} onDs={v => set({ ds: v })}
                                                  localPack={<LocalPack />} />}
              {lens === 'process' && station === 'catalog' && <ResearchCatalog key={`${st.mrun||''}:${st.tm||''}`} runFilter={st.mrun} template={st.tm} onGo={go}
                onAutotest={!(ADMIN_LIVE && isAdmin) ? null : () => go(localTools ? 'autotest' : 'dispatch')}
                onStudio={!(ADMIN_LIVE && isAdmin) ? null : () => go('pipeline')} />}
              {lens === 'process' && station === 'probe' && <GoodfireProbePanel />}
              {lens === 'process' && station === 'pipeline' && <TemplatePipeline step={pstep} onStep={v => set({ pstep: v })} onGo={go} canAudit={isAdmin} />}
              {lens === 'process' && station === 'census' && <CensusPanel />}
              {lens === 'process' && station === 'autotest' && <AutoTestPanel onGo={go} onStation={s => go(s)} />}
              {lens === 'route' && <RouteLens tab={rtab} openPane={st.opane} model={st.mdl || ''} ds={st.ds || ''} onGoSpace={goSpace} onGo={go} />}
              {lens === 'config' && csec!=='legacy' && <ConfigView cfg={cfg} onGo={go} section={csec} nodeStep={st.cstep} />}
              {lens === 'config' && csec==='legacy' && (localTools?<><section className="v8-card"><h2>旧版兼容与教学工具</h2><p className="v8-note">此处保留历史实测与显式教学演示；演示不进入研究统计。正式前向请进入运行与路径。</p><button className="fw-tbtn" onClick={()=>go('spatial','live')}>打开真实运行与路径</button></section><LensSpatial on onGo={go}/></>:<Empty>旧版工具只在本机中心开放；正式研究使用空间中的图谱、参数和运行页面。</Empty>)}
            </Suspense>
          </main>
          {/* 「问 AI」侧栏：只在桌面客户端（Key 在主进程）；网页版不显示 */}
          {DESKTOP && <Suspense fallback={null}><AiChatDrawer onSettings={() => set({ lens: 'config', csec: 'model' })} /></Suspense>}
        </div>
        {helpOpen && <WorkbenchHelp lens={lens} onClose={() => setHelpOpen(false)} />}
        {st.importing&&<ImportDataDialog onClose={()=>set({importing:false})}/>}
      </div>
    </div>
  );
}
