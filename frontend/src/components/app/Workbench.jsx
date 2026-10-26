/* 研究工作台（v6，/workbench）：按研究闭环组织的完整工作区 —— 总览 / 空间 / 研发 / 路线。
   全部面板只读写中心服务的真源端点（/api/platform/snapshot/*、/api/catalog、/api/agg、/api/analysis、/api/atlas、
   /api/census、/api/audit、/api/grammar 等），连接本机或公开中心都可用；没有数据时显示空态，不回退演示数字。
   只适合研究者本机的工具（自动测试、读取本机模型权重的旧版空间视图）只在本机中心（config.local_tools）出现。 */
import { Fragment, Suspense, lazy, useEffect, useState } from 'react';

import AccountMenu from './rdc_fusion/AccountMenu.jsx';
import AtlasView from './rdc_fusion/AtlasView.jsx';
import AuditQueue from './rdc_fusion/AuditQueue.jsx';
import CatalogPanel from './rdc_fusion/CatalogPanel.jsx';
import CensusPanel from './rdc_fusion/CensusPanel.jsx';
import DispatchPanel from './rdc_fusion/DispatchPanel.jsx';
import FreezePanel from './rdc_fusion/FreezePanel.jsx';
import HomeOverview from './rdc_fusion/HomeOverview.jsx';
import LocalPackView from './rdc_fusion/LocalPackView.jsx';
import NeuronUnitPanel from './rdc_fusion/NeuronUnitPanel.jsx';
import ResponseExplorer from './rdc_fusion/ResponseExplorer.jsx';
import RouteBoard from './rdc_fusion/RouteBoard.jsx';
import TemplateStudio from './rdc_fusion/TemplateStudio.jsx';
import { API_BASE, useApi } from './rdc_fusion/api.js';
import { loadLocalPack } from './rdc_fusion/localPack.js';
import { Empty } from './rdc_fusion/ui.jsx';
import { LENSES, SPATIAL_VIEWS, STATIONS, readState, resolveGo, writeState } from './workbenchUtil.js';
import './rdc_fusion/rdc_fusion.css';
import './rdc_fusion/rdc_fusion_v6.css';
import './workbench.css';

const AutoTestPanel = lazy(() => import('./rdc_fusion/AutoTestPanel.jsx'));
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
    local: <><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>,
  }[id];
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">{p}</svg>;
}

function SubNav({ items, value, onChange, label }) {
  return (
    <nav className="wb-subnav" aria-label={label}>
      {items.map(it => (
        <button key={it.id} type="button" className={value === it.id ? 'on' : ''} onClick={() => onChange(it.id)} title={it.desc || it.label}>
          {it.label}
        </button>
      ))}
    </nav>
  );
}

export default function Workbench() {
  const cfg = useApi('/api/platform/config');
  const localTools = Boolean(cfg.data?.local_tools);
  const [st, setSt] = useState(readState);
  const [draft, setDraft] = useState(null);
  const stations = STATIONS.filter(s => !s.localOnly || localTools);
  const lenses = localTools ? [...LENSES, { id: 'local', label: '本机工具', desc: '读取本机模型权重的空间视图（只在本机中心可用）' }] : LENSES;
  const lens = lenses.some(l => l.id === st.lens) ? st.lens : 'home';
  const station = stations.some(s => s.id === st.station) ? st.station : 'catalog';
  const view = SPATIAL_VIEWS.some(v => v.id === st.view) ? st.view : 'response';

  useEffect(() => { writeState({ ...st, lens, station, view }); }, [st, lens, station, view]);
  const lensLabel = lenses.find(l => l.id === lens)?.label || '';
  useEffect(() => { document.title = `研究工作台 · ${lensLabel} · AI2050`; }, [lensLabel]);

  const go = (a, b, c, extra) => setSt(prev => resolveGo(prev, a, b, c, extra));
  const set = patch => setSt(prev => ({ ...prev, ...patch }));

  return (
    <div className="fw-root wb-root">
      <header className="wb-top">
        <a className="wb-brand" href="/"><span className="op-logo" aria-hidden="true">A2</span><span><b>研究工作台</b><small>返回首页</small></span></a>
        <span className="wb-lens-title">{lensLabel}<small>{lenses.find(l => l.id === lens)?.desc}</small></span>
        <div className="wb-top-right">
          {cfg.data?.sandbox_synthetic && <span className="op-sandbox">合成·沙盒</span>}
          <span className="wb-center" title="当前连接的中心服务">{API_BASE || window.location.origin}</span>
          <AccountMenu />
        </div>
      </header>
      {cfg.error && <div className="op-banner" role="alert">连不上中心服务：{cfg.error}</div>}
      <div className="wb-frame">
        {/* 左轨：一级界面（沿用 v5 工作台的图标轨布局） */}
        <nav className="fw-rail" aria-label="工作台界面">
          {lenses.map((l, i) => (
            <Fragment key={l.id}>
              {i === 1 && <div className="fw-rdiv" />}
              {l.id === 'local' && <div className="fw-rdiv" />}
              <button type="button" className={'fw-rbtn' + (lens === l.id ? ' on' : '')} onClick={() => set({ lens: l.id })}
                      title={`${l.label}：${l.desc}`} aria-current={lens === l.id ? 'page' : undefined}>
                <LensIcon id={l.id} />{l.label}
              </button>
            </Fragment>
          ))}
        </nav>
        <div className="wb-body">
          {lens === 'spatial' && <SubNav items={SPATIAL_VIEWS} value={view} onChange={v => set({ view: v })} label="空间视图" />}
          {lens === 'process' && <SubNav items={stations} value={station} onChange={s => set({ station: s })} label="研发工位" />}
          <main className="wb-main">
            <Suspense fallback={<Empty>加载中…</Empty>}>
              {lens === 'home' && <HomeOverview onGo={go} />}
              {lens === 'spatial' && view === 'response' && <ResponseExplorer key={st.tm} initialTm={st.tm || undefined} />}
              {lens === 'spatial' && view === 'atlas' && <AtlasView snapshot onOpenFamily={({ tm }) => go('spatial', null, null, { tm })} />}
              {lens === 'spatial' && view === 'units' && <NeuronUnitPanel />}
              {lens === 'spatial' && view === 'local' && <LocalPack />}
              {lens === 'process' && station === 'catalog' && <CatalogPanel onGo={go} onAutotest={() => go(localTools ? 'autotest' : 'dispatch')} onStudio={() => go('studio')} />}
              {lens === 'process' && station === 'studio' && <TemplateStudio draft={draft} onDraft={setDraft} onNext={() => go('freeze')} />}
              {lens === 'process' && station === 'freeze' && <FreezePanel draft={draft} onDraft={setDraft} onGo={go} />}
              {lens === 'process' && station === 'dispatch' && <DispatchPanel onGo={go} />}
              {lens === 'process' && station === 'audit' && <AuditQueue />}
              {lens === 'process' && station === 'census' && <CensusPanel />}
              {lens === 'process' && station === 'autotest' && <AutoTestPanel onGo={go} onStation={s => go(s)} />}
              {lens === 'route' && <RouteBoard onGo={go} />}
              {lens === 'local' && <LensSpatial on onGo={() => {}} />}
            </Suspense>
          </main>
        </div>
      </div>
    </div>
  );
}
