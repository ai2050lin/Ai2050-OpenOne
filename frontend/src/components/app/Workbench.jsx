/* 研究工作台（v9，/workbench 与 /atlas）：以「一句话」为中心的五个研究页面（设计见 design/attention_neuron_plan_v1.md §4）。
     数据集        各数据集与类别、模型正确率、行为门；点题目或句子进入「例句」（子视图：语言项、响应点云、题库、本地包）
     例句          一句话 → 行为 → 注意力路由（层 × 头：直接作用与总效应并列）→ 激活的神经元 → 保真度（子视图：输入一句话）
     模式与优先级  每种冲突的逐层赛跑曲线与优先级表
     神经元图谱    按层浏览 MLP 单元（3D 层塔降为侧边小导航）、归因优先级、本机测试
     进展          各模式保真度、完整案例数、未通过的检验
   左轨分隔线之下：贡献（贡献算力 + 我的上传）、管理（管理员的分步骤向导，只在桌面客户端）、配置、本机工具。
   全部读取经 rdc_fusion/api.js（只读数据包模式）；没有数据时显示「未接入」，不回退演示数字。每个面板包 PanelBoundary，
   一个面板出错不让整页白屏。旧链接（?lens=route|spatial|process 等）由 workbenchUtil.parseState 换算到新页面。 */
import { Fragment, Suspense, lazy, useEffect, useState } from 'react';

import AccountMenu from './rdc_fusion/AccountMenu.jsx';
import AtlasView from './rdc_fusion/AtlasView.jsx';
import CatalogPanel from './rdc_fusion/CatalogPanel.jsx';
import ConfigView, { CFG_SECTIONS } from './ConfigView.jsx';
import LocalPackView from './rdc_fusion/LocalPackView.jsx';
import PanelBoundary from './PanelBoundary.jsx';
import TemplatePipeline from './rdc_fusion/TemplatePipeline.jsx';
import { ADMIN_LIVE, API_BASE, DESKTOP, OFFLINE_CLIENT, useApi } from './rdc_fusion/api.js';
import { useAuth } from './rdc_fusion/authContext.js';
import { loadLocalPack } from './rdc_fusion/localPack.js';
import { Empty } from './rdc_fusion/ui.jsx';
import WorkbenchHelp from './WorkbenchHelp.jsx';
import { readState as readSpaceState, writeState as writeSpaceState } from '../v8/v8Util.js';
import { DIVIDER_BEFORE, readState, resolveGo, visibleLenses, visibleSteps, visibleSubs, writeState } from './workbenchUtil.js';
import './rdc_fusion/rdc_fusion.css';
import './rdc_fusion/rdc_fusion_v6.css';
import './workbench.css';
import './mech/mech.css';

const AiChatDrawer = lazy(() => import('./rdc_fusion/AiChatDrawer.jsx'));
const AutoTestPanel = lazy(() => import('./rdc_fusion/AutoTestPanel.jsx'));

const CensusPanel = lazy(() => import('./rdc_fusion/CensusPanel.jsx'));
const ContributePage = lazy(() => import('./mech/ContributePage.jsx'));
const DatasetsPage = lazy(() => import('./mech/DatasetsPage.jsx'));
const GoodfireProbePanel = lazy(() => import('./rdc_fusion/GoodfireProbePanel.jsx'));
const LensSpatial = lazy(() => import('./rdc_fusion/LensSpatial.jsx'));
const ModesPage = lazy(() => import('./mech/ModesPage.jsx'));
const NeuronPriority = lazy(() => import('./mech/NeuronPriority.jsx'));
const ProgressPage = lazy(() => import('./mech/ProgressPage.jsx'));
const SentencePage = lazy(() => import('./mech/SentencePage.jsx'));
const SpaceLens = lazy(() => import('../v8/SpaceLens.jsx'));

/* 子视图 → 空间组件（v8/SpaceLens）的 view：层塔是侧边小导航，主区是二维视图 */
const SPACE_VIEW = {
  'neurons/atlas': 'neuron', 'datasets/lang': 'lang', 'datasets/response': 'response',
  'datasets/relations': 'relations', 'datasets/local': 'local', 'sentence/live': 'live',
};

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
      {!pack && <Empty>研究包（响应点云页「下载研究包」）与谱图包（谱图关系页「导出谱图包」）都在浏览器内解压查看，不上传到任何服务器。</Empty>}
      {pack?.kind === 'respack' && <LocalPackView pack={pack} />}
      {pack?.kind === 'atlaspack' && <AtlasView localAtlas={pack.atlas} localName={pack.name} />}
    </div>
  );
}

/* 左轨线性图标 */
function LensIcon({ id }) {
  const p = {
    datasets: <><ellipse cx="12" cy="5.5" rx="8" ry="3" /><path d="M4 5.5v13c0 1.7 3.6 3 8 3s8-1.3 8-3v-13M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>,
    sentence: <path d="M4 6h16M4 12h10M4 18h13" />,
    modes: <path d="M3 20 9 12l4 4 8-11M15 5h6v6" />,
    neurons: <><circle cx="6" cy="6" r="2.2" /><circle cx="18" cy="6" r="2.2" /><circle cx="12" cy="18" r="2.2" /><path d="M7.6 7.6 10.7 16M16.4 7.6 13.3 16M8.2 6h7.6" /></>,
    progress: <><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></>,
    contribute: <><rect x="4" y="4" width="16" height="12" rx="2" /><path d="M8 20h8M12 16v4M9 10l2 2 4-4" /></>,
    admin: <><path d="M12 3 4 6v6c0 4.4 3.4 8.2 8 9 4.6-.8 8-4.6 8-9V6z" /><path d="m9 12 2 2 4-4" /></>,
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
  // 网页只能浏览；测试与管理在桌面客户端，管理向导另需管理员账号（连接本机后端的开发不限）
  const { user } = useAuth();
  const isAdmin = !OFFLINE_CLIENT || user?.role === 'admin';
  const lenses = visibleLenses({ live: ADMIN_LIVE, admin: isAdmin, localTools });
  const lens = lenses.some(l => l.id === st.lens) ? st.lens : 'sentence';
  const subs = visibleSubs(lens, { live: ADMIN_LIVE });
  const sub = st.sub && (subs.some(s => s.id === st.sub) || SPACE_VIEW[`${lens}/${st.sub}`]) ? st.sub : (subs[0]?.id || '');
  const csec = CFG_SECTIONS.some(s => s.id === st.csec) ? st.csec : 'conn';
  const steps = visibleSteps({ localTools });

  useEffect(() => { writeState({ ...st, lens, sub, csec }); }, [st, lens, sub, csec]);
  const lensMeta = lenses.find(l => l.id === lens);
  useEffect(() => { document.title = `研究工作台 · ${lensMeta?.label || ''} · AI2050`; }, [lensMeta]);

  const go = (a, b, c, extra) => setSt(prev => resolveGo(prev, a, b, c, extra));
  const set = patch => setSt(prev => ({ ...prev, ...patch }));
  /* 跨页跳转：先把空间组件的内部状态（块、单元、语言项）写进地址栏，空间组件挂载时读取 */
  const openUnit = (layer, unit) => { writeSpaceState({ ...readSpaceState(), block: layer, unit, mod: 'mlp' }); set({ lens: 'neurons', sub: 'atlas' }); };
  const openItem = (item, model) => { writeSpaceState({ ...readSpaceState(), item, model, ltab: 'items' }); set({ lens: 'datasets', sub: 'lang' }); };
  const openSentence = (caseId, cue, fact = '') => set({ lens: 'sentence', sub: 'case', case: caseId, cue: cue || '', fact });

  const spaceView = SPACE_VIEW[`${lens}/${sub}`];
  const page = (() => {
    if (spaceView) {
      return (
        <div className="wb9-space">
          <SpaceLens view={spaceView} setView={v => go('spatial', null, null, { view: v })} tm={st.tm} setTm={tm => set({ tm })} localPack={<LocalPack />} />
        </div>
      );
    }
    if (lens === 'datasets' && sub === 'catalog') {
      return <CatalogPanel onGo={go} onAutotest={!(ADMIN_LIVE && isAdmin) ? null : () => go('admin', null, localTools ? 'autotest' : 'dispatch')}
                           onStudio={!(ADMIN_LIVE && isAdmin) ? null : () => go('admin', null, 'design')} />;
    }
    if (lens === 'datasets') return <DatasetsPage onFact={(id, fact) => openSentence(id, 'plain', fact)} onItem={openItem} />;
    if (lens === 'sentence') return <SentencePage st={st} set={set} onUnit={openUnit} />;
    if (lens === 'modes') return <ModesPage onSentence={(id, cue) => openSentence(id, cue)} />;
    if (lens === 'neurons' && sub === 'priority') return <NeuronPriority st={st} set={set} onUnit={openUnit} />;
    if (lens === 'neurons' && sub === 'probe') return <GoodfireProbePanel />;
    if (lens === 'progress') return <ProgressPage />;
    if (lens === 'contribute') return <ContributePage />;
    if (lens === 'admin') {
      return (
        <TemplatePipeline step={st.step} onStep={v => set({ step: v })} onGo={go} canAudit={isAdmin} steps={steps}
                          panels={{ census: <CensusPanel />, autotest: <AutoTestPanel onGo={go} onStation={s => go(s)} /> }} />
      );
    }
    if (lens === 'config') return <ConfigView cfg={cfg} onGo={go} section={csec} />;
    if (lens === 'local') return <LensSpatial on onGo={() => {}} />;
    return <Empty>没有这个页面。</Empty>;
  })();

  return (
    <div className="fw-root wb-root wb9">
      <header className="wb-top">
        <a className="wb-brand" href={window.location.protocol === 'file:' ? window.location.pathname : '/'}><span className="op-logo" aria-hidden="true">A2</span><span><b>研究工作台</b><small>返回首页</small></span></a>
        <span className="wb-lens-title">{lensMeta?.label}<small>{lensMeta?.desc}</small></span>
        <div className="wb-top-right">
          {cfg.data?.sandbox_synthetic && <span className="op-sandbox">合成·沙盒</span>}
          <span className="wb-center" title="当前连接的中心服务">{API_BASE || window.location.origin}</span>
          <AccountMenu />
        </div>
      </header>
      {cfg.error && <div className="op-banner" role="alert">连不上中心服务：{cfg.error}</div>}
      <div className="wb-frame">
        <nav className="fw-rail" aria-label="工作台界面">
          {lenses.map(l => (
            <Fragment key={l.id}>
              {(DIVIDER_BEFORE.has(l.id) || l.id === 'local') && <div className="fw-rdiv" />}
              <button type="button" className={'fw-rbtn' + (lens === l.id ? ' on' : '')} onClick={() => set({ lens: l.id, sub: '' })}
                      title={`${l.label}：${l.desc}`} aria-current={lens === l.id ? 'page' : undefined}>
                <LensIcon id={l.id} />{l.label}
              </button>
            </Fragment>
          ))}
        </nav>
        <div className="wb-body">
          {subs.length > 0 && <SubNav items={subs} value={sub} onChange={v => set({ sub: v })} label={`${lensMeta?.label}视图`} onHelp={() => setHelpOpen(true)} />}
          {lens === 'config' && <SubNav items={CFG_SECTIONS} value={csec} onChange={v => set({ csec: v })} label="配置分段" />}
          <main className="wb-main">
            <PanelBoundary name={lensMeta?.label} resetKey={`${lens}|${sub}|${csec}`}>
              <Suspense fallback={<Empty>加载中…</Empty>}>{page}</Suspense>
            </PanelBoundary>
          </main>
          {/* 「问 AI」侧栏：只在桌面客户端（Key 在主进程）；网页版不显示 */}
          {DESKTOP && <Suspense fallback={null}><AiChatDrawer onSettings={() => set({ lens: 'config', csec: 'model' })} /></Suspense>}
        </div>
        {helpOpen && <WorkbenchHelp lens={lens} onClose={() => setHelpOpen(false)} />}
      </div>
    </div>
  );
}
