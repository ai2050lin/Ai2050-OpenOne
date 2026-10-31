/* 研究工作台（v8，/workbench 与 /atlas）：进展 / 空间 / 研发 / 配置。
   空间：左侧常驻 LLM 多层结构，右侧是神经元逆向分析结果与语言图谱（v8/SpaceLens.jsx）；进展：神经元与语言图谱的进展（v8/RouteLens.jsx）；
   研发：题库、贡献算力与管理工位；配置：连接与运行形态、账号、被测模型（ConfigView.jsx）。
   左轨分隔线之上是研究对象与实验生产（进展 / 空间 / 研发），之下是形态相关的界面（配置、本机工具）。
   进展与空间只读本机数据包；网页只能浏览（题库只读、贡献算力只给下载入口），
   执行测试与管理工位只在桌面客户端（config.js 的 ADMIN_LIVE；管理工位另需管理员账号）。没有数据时显示空态，不回退演示数字。
   只适合研究者本机的工具（自动测试、读取本机模型权重的旧版空间视图）只在本机中心（config.local_tools）出现。 */
import { Fragment, Suspense, lazy, useEffect, useState } from 'react';

import AccountMenu from './rdc_fusion/AccountMenu.jsx';
import AtlasView from './rdc_fusion/AtlasView.jsx';
import CatalogPanel from './rdc_fusion/CatalogPanel.jsx';
import CensusPanel from './rdc_fusion/CensusPanel.jsx';
import ConfigView, { CFG_SECTIONS } from './ConfigView.jsx';
import LocalPackView from './rdc_fusion/LocalPackView.jsx';
import TemplatePipeline from './rdc_fusion/TemplatePipeline.jsx';
import { ADMIN_LIVE, API_BASE, DESKTOP, OFFLINE_CLIENT, useApi } from './rdc_fusion/api.js';
import { useAuth } from './rdc_fusion/authContext.js';
import { loadLocalPack } from './rdc_fusion/localPack.js';
import { Empty } from './rdc_fusion/ui.jsx';
import RouteLens from '../v8/RouteLens.jsx';
import SpaceLens from '../v8/SpaceLens.jsx';
import WorkbenchHelp from './WorkbenchHelp.jsx';
import { readState as readSpaceState, writeState as writeSpaceState } from '../v8/v8Util.js';
import { LENSES, PIPE_STEPS, ROUTE_VIEWS, SPATIAL_VIEWS, readState, resolveGo, visibleStations, writeState } from './workbenchUtil.js';
import './rdc_fusion/rdc_fusion.css';
import './rdc_fusion/rdc_fusion_v6.css';
import './workbench.css';

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
  const [helpOpen, setHelpOpen] = useState(false);
  // 网页只能浏览：题库只读、贡献算力只给下载入口；测试与管理在桌面客户端，管理工位另需管理员账号（本机开发不限）
  const { user } = useAuth();
  const isAdmin = !OFFLINE_CLIENT || user?.role === 'admin';
  const stations = visibleStations({ live: ADMIN_LIVE, admin: isAdmin, localTools });
  const baseLenses = !ADMIN_LIVE
    ? LENSES.map(l => (l.id === 'process' ? { ...l, desc: '网页只能浏览：题库只读；执行测试与管理请用桌面客户端' } : l))
    : LENSES;
  const lenses = localTools ? [...baseLenses, { id: 'local', label: '本机工具', desc: '读取本机模型权重的空间视图（只在本机中心可用）' }] : baseLenses;
  const lens = lenses.some(l => l.id === st.lens) ? st.lens : 'spatial';
  const station = stations.some(s => s.id === st.station) ? st.station : (stations[0]?.id || 'catalog');
  const pstep = PIPE_STEPS.some(s => s.id === st.pstep) ? st.pstep : 'design';
  const csec = CFG_SECTIONS.some(s => s.id === st.csec) ? st.csec : 'conn';
  // 谱图关系（relations）不在 SPATIAL_VIEWS 导航里（已并入语言图谱入口），但仍是合法视图：入口与旧链接都能进入
  const view = SPATIAL_VIEWS.some(v => v.id === st.view) || st.view === 'relations' ? st.view : 'neuron';
  const rtab = ROUTE_VIEWS.some(v => v.id === st.rtab) ? st.rtab : 'neuron';

  useEffect(() => { writeState({ ...st, lens, station, view, rtab, pstep, csec }); }, [st, lens, station, view, rtab, pstep, csec]);
  const lensLabel = lenses.find(l => l.id === lens)?.label || '';
  useEffect(() => { document.title = `研究工作台 · ${lensLabel} · AI2050`; }, [lensLabel]);

  const go = (a, b, c, extra) => setSt(prev => resolveGo(prev, a, b, c, extra));
  const set = patch => setSt(prev => ({ ...prev, ...patch }));
  /* 进展 → 空间：先把空间内部状态（语言项、块等）写进地址栏，空间挂载时读取 */
  const goSpace = (v, patch = {}) => { writeSpaceState({ ...readSpaceState(), ...patch }); set({ lens: 'spatial', view: v }); };

  return (
    <div className="fw-root wb-root">
      <header className="wb-top">
        <a className="wb-brand" href={window.location.protocol === 'file:' ? window.location.pathname : '/'}><span className="op-logo" aria-hidden="true">A2</span><span><b>研究工作台</b><small>返回首页</small></span></a>
        <span className="wb-lens-title">{lensLabel}<small>{lenses.find(l => l.id === lens)?.desc}</small></span>
        <div className="wb-top-right">
          {cfg.data?.sandbox_synthetic && <span className="op-sandbox">合成·沙盒</span>}
          <span className="wb-center" title="当前连接的中心服务">{API_BASE || window.location.origin}</span>
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
              <button type="button" className={'fw-rbtn' + (lens === l.id ? ' on' : '')} onClick={() => set({ lens: l.id })}
                      title={`${l.label}：${l.desc}`} aria-current={lens === l.id ? 'page' : undefined}>
                <LensIcon id={l.id} />{l.label}
              </button>
            </Fragment>
          ))}
        </nav>
        <div className="wb-body">
          {lens === 'route' && <SubNav items={ROUTE_VIEWS} value={rtab} onChange={v => set({ rtab: v })} label="进展视图" onHelp={() => setHelpOpen(true)} />}
          {lens === 'spatial' && <SubNav items={SPATIAL_VIEWS} value={view} onChange={v => set({ view: v })} label="空间视图" onHelp={() => setHelpOpen(true)} />}
          {lens === 'process' && <SubNav items={stations} value={station} onChange={s => set({ station: s })} label="研发工位" onHelp={() => setHelpOpen(true)} />}
          {/* 配置的分段与其它界面用同一条子导航：连接 / 账号与节点 / 模型与 AI / 平台 */}
          {lens === 'config' && <SubNav items={CFG_SECTIONS} value={csec} onChange={v => set({ csec: v })} label="配置分段" onHelp={() => setHelpOpen(true)} />}
          <main className="wb-main">
            <Suspense fallback={<Empty>加载中…</Empty>}>
              {lens === 'spatial' && <SpaceLens view={view} setView={v => set({ view: v })} tm={st.tm} setTm={tm => set({ tm })} localPack={<LocalPack />} />}
              {lens === 'process' && station === 'catalog' && <CatalogPanel onGo={go} onAutotest={!(ADMIN_LIVE && isAdmin) ? null : () => go(localTools ? 'autotest' : 'dispatch')} onStudio={!(ADMIN_LIVE && isAdmin) ? null : () => go('pipeline')} />}
              {lens === 'process' && station === 'probe' && <GoodfireProbePanel />}
              {lens === 'process' && station === 'pipeline' && <TemplatePipeline step={pstep} onStep={v => set({ pstep: v })} onGo={go} canAudit={isAdmin} />}
              {lens === 'process' && station === 'census' && <CensusPanel />}
              {lens === 'process' && station === 'autotest' && <AutoTestPanel onGo={go} onStation={s => go(s)} />}
              {lens === 'route' && <RouteLens tab={rtab} onGoSpace={goSpace} onGo={go} />}
              {lens === 'config' && <ConfigView cfg={cfg} onGo={go} section={csec} />}
              {lens === 'local' && <LensSpatial on onGo={() => {}} />}
            </Suspense>
          </main>
          {/* 「问 AI」侧栏：只在桌面客户端（Key 在主进程）；网页版不显示 */}
          {DESKTOP && <Suspense fallback={null}><AiChatDrawer onSettings={() => set({ lens: 'config', csec: 'model' })} /></Suspense>}
        </div>
        {helpOpen && <WorkbenchHelp lens={lens} onClose={() => setHelpOpen(false)} />}
      </div>
    </div>
  );
}
