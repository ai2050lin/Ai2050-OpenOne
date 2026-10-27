/* 网站外壳（v8，/ 与 /atlas 共用）：首页 · ① 语言图谱 · ② 神经元 · 参与；学习方法、进展、响应点云、谱图关系是分支（「更多」），
   研究工作台是独立页面（网页版）。模型是全局选择，两条主线共用；URL 即状态（旧开放平台的 ?tab= 链接自动换算）。
   没有数据时显示空态，不回退演示数字。 */
import { Suspense, lazy, useEffect, useRef, useState } from 'react';

import AccountMenu from '../app/rdc_fusion/AccountMenu.jsx';
import { API_BASE, REMOTE_CENTER, WEB_URL, useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import '../app/rdc_fusion/rdc_fusion.css';
import '../app/rdc_fusion/rdc_fusion_v6.css';
import '../open/open.css';
import LangPage from './LangPage.jsx';
import SiteHome from './SiteHome.jsx';
import { BRANCHES, PAGES, defaultModel, readState, shortModel, writeState } from './v8Util.js';
import './v8.css';

const NeuronPage = lazy(() => import('./NeuronPage.jsx'));
const ProgressPage = lazy(() => import('./ProgressPage.jsx'));
const ResponseExplorer = lazy(() => import('../app/rdc_fusion/ResponseExplorer.jsx'));
const AtlasView = lazy(() => import('../app/rdc_fusion/AtlasView.jsx'));
const OpenJoin = lazy(() => import('../open/OpenJoin.jsx'));
const OpenLearn = lazy(() => import('../open/OpenLearn.jsx'));

function MoreMenu({ st, set }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const close = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, []);
  const on = BRANCHES.some(b => b.id === st.page);
  return (
    <div className="v8-more" ref={ref}>
      <button type="button" className={on ? 'on' : ''} aria-expanded={open} onClick={() => setOpen(v => !v)}>更多 ▾</button>
      {open && (
        <ul role="menu">
          {BRANCHES.map(b => (
            <li key={b.id}><button type="button" role="menuitem" onClick={() => { set({ page: b.id }); setOpen(false); }}>{b.label}<small>{b.desc}</small></button></li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function AtlasApp() {
  const cfg = useApi('/api/platform/config');
  const config = cfg.data;
  const treeApi = useApi('/api/lang/tree', { pollWhilePending: true });
  const skApi = useApi('/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s]));
  const [st, setSt] = useState(readState);
  const model = st.model && tree?.models?.includes(st.model) ? st.model : defaultModel(tree, skeletons);
  const set = patch => setSt(prev => ({ ...prev, ...patch }));
  const pages = PAGES.filter(p => p.id !== 'join' || !config?.ui?.tabs || config.ui.tabs.includes('join'));

  useEffect(() => { writeState(st); }, [st]);
  const label = [...PAGES, ...BRANCHES].find(p => p.id === st.page)?.label || '';
  useEffect(() => { document.title = `${st.page === 'home' ? '' : `${label} · `}${config?.title || 'AI2050'}`; }, [label, st.page, config]);
  useEffect(() => { window.scrollTo?.(0, 0); }, [st.page]);
  const needsTree = st.page === 'lang' || st.page === 'neuron' || st.page === 'progress';

  return (
    <div className="fw-root v8-root">
      <header className="v8-top">
        <a className="v8-brand" href="/" onClick={e => { e.preventDefault(); set({ page: 'home' }); }}>
          <span className="v8-logo" aria-hidden="true">A2</span><span><b>AI2050</b><small>开源 LLM 逆向研究平台</small></span>
        </a>
        <nav className="v8-nav" aria-label="主导航">
          {pages.map(p => (
            <button key={p.id} type="button" className={st.page === p.id ? 'on' : ''} title={p.desc} aria-current={st.page === p.id ? 'page' : undefined}
                    onClick={() => set({ page: p.id })}>{p.n && <span className="n">{p.n}</span>}{p.label}</button>
          ))}
          {/* 研究工作台：独立页面（管理员的实时操作，网页版），用描边表示「跳出本站」；远端客户端打开中心网址上的工作台 */}
          <a className="v8-nav-wb" href={REMOTE_CENTER ? `${WEB_URL}/workbench` : '/workbench'} target={REMOTE_CENTER ? '_blank' : undefined} rel="noreferrer"
             title="研究工作台：题库、模板工作室、试跑与冻结、分发、审计、普查作业（实时连接中心，管理操作用）">研究工作台 ↗</a>
        </nav>
        <div className="v8-right">
          {config?.sandbox_synthetic && <span className="v8-sandbox" title="该中心允许合成数据，只用于流程验证">合成·沙盒</span>}
          {(st.page === 'lang' || st.page === 'neuron') && (
            <label>模型
              <select value={model} onChange={e => set({ model: e.target.value, block: null, unit: null })} disabled={!tree?.models?.length}>
                {!tree?.models?.length && <option value="">—</option>}
                {(tree?.models || []).map(m => <option key={m} value={m} title={m}>{shortModel(m)}{skeletons[m]?.census_run ? '' : '（未普查）'}</option>)}
              </select>
            </label>
          )}
          <MoreMenu st={st} set={set} />
          <AccountMenu />
        </div>
      </header>
      {cfg.error && <div className="v8-banner" role="alert">连不上中心服务 {API_BASE || window.location.origin}：{cfg.error}</div>}
      <main className="v8-main">
        {needsTree && treeApi.error && <Empty>语言谱图未接入：{treeApi.error}</Empty>}
        {needsTree && !treeApi.error && !tree && <Empty>{treeApi.data?.note || '正在读取语言谱图…'}</Empty>}
        <Suspense fallback={<Empty>加载中…</Empty>}>
          {st.page === 'home' && <SiteHome config={config} tree={tree} model={model} set={set} />}
          {st.page === 'lang' && tree && <LangPage tree={tree} model={model} skeleton={skeletons[model]} st={st} set={set} />}
          {st.page === 'neuron' && tree && <NeuronPage model={model} skeleton={skeletons[model]} skeletons={skeletons} st={st} set={set} />}
          {st.page === 'join' && <OpenJoin config={config} />}
          {st.page === 'learn' && <OpenLearn onGo={t => set({ page: t === 'atlas' ? 'lang' : t })} />}
          {st.page === 'progress' && tree && <ProgressPage tree={tree} model={model} set={set} />}
          {st.page === 'response' && <ResponseExplorer key={`${st.tm}|${model}`} initialTm={st.tm || undefined} initialBucket={model || undefined} />}
          {st.page === 'relations' && <AtlasView snapshot onOpenFamily={({ tm }) => set({ page: 'response', tm })} />}
        </Suspense>
      </main>
      <footer className="v8-foot">
        <span>{config?.title || 'AI2050'} · v{config?.version || '—'}</span>
        <span>代码 {config?.license?.code || 'Apache-2.0'} · 数据 {config?.license?.data || 'CC BY 4.0'}</span>
        {config?.repo_url && <a href={config.repo_url} target="_blank" rel="noreferrer">源代码</a>}
        {config?.contact && <span>联系：{config.contact}</span>}
      </footer>
    </div>
  );
}
