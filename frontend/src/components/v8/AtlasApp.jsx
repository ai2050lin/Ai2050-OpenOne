/* 网站外壳（/）：首页 · 介绍 · 参与，另有「研究工作台 ↗」。语言图谱与神经元在研究工作台的「空间」里，
   进展在「路线」里；旧链接（?tab=atlas、?page=lang / neuron / progress …）转到研究工作台对应的页面。
   只读数据包模式：除登录、读取配置、下载数据包外不连接服务器。没有数据时显示空态，不回退演示数字。 */
import { Suspense, lazy, useEffect, useState } from 'react';

import AccountMenu from '../app/rdc_fusion/AccountMenu.jsx';
import { API_BASE, useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import '../app/rdc_fusion/rdc_fusion.css';
import '../app/rdc_fusion/rdc_fusion_v6.css';
import '../open/open.css';
import SiteHome from './SiteHome.jsx';
import { PAGES, WORKBENCH_REDIRECT, readState, workbenchHref, writeState } from './v8Util.js';
import './v8.css';

const OpenJoin = lazy(() => import('../open/OpenJoin.jsx'));
const OpenLearn = lazy(() => import('../open/OpenLearn.jsx'));

export default function AtlasApp() {
  const cfg = useApi('/api/platform/config');
  const config = cfg.data;
  const [st, setSt] = useState(readState);
  const set = patch => setSt(prev => ({ ...prev, ...patch }));
  const redirect = WORKBENCH_REDIRECT[st.page];
  const pages = PAGES.filter(p => p.id !== 'join' || !config?.ui?.tabs || config.ui.tabs.includes('join'));

  useEffect(() => { if (redirect) window.location.replace(redirect); else writeState(st); }, [st, redirect]);
  const label = PAGES.find(p => p.id === st.page)?.label || '';
  useEffect(() => { document.title = `${st.page === 'home' ? '' : `${label} · `}${config?.title || 'AI2050'}`; }, [label, st.page, config]);
  useEffect(() => { window.scrollTo?.(0, 0); }, [st.page]);
  if (redirect) return <Empty>正在打开研究工作台…</Empty>;

  return (
    <div className="fw-root v8-root">
      <header className="v8-top">
        <a className="v8-brand" href="/" onClick={e => { e.preventDefault(); set({ page: 'home' }); }}>
          <span className="v8-logo" aria-hidden="true">A2</span><span><b>AI2050</b><small>开源 LLM 逆向研究平台</small></span>
        </a>
        <nav className="v8-nav" aria-label="主导航">
          {pages.map(p => (
            <button key={p.id} type="button" className={st.page === p.id ? 'on' : ''} title={p.desc} aria-current={st.page === p.id ? 'page' : undefined}
                    onClick={() => set({ page: p.id })}>{p.label}</button>
          ))}
          {/* 研究工作台：空间（LLM 多层结构、神经元与语言图谱）、路线（进展）、研发；用描边表示「进入工作台」 */}
          <a className="v8-nav-wb" href={workbenchHref()} title="研究工作台：空间（LLM 多层结构、神经元逆向分析、语言图谱）、路线（进展）、研发">研究工作台 ↗</a>
        </nav>
        <div className="v8-right">
          {config?.sandbox_synthetic && <span className="v8-sandbox" title="该中心允许合成数据，只用于流程验证">合成·沙盒</span>}
          <AccountMenu />
        </div>
      </header>
      {cfg.error && <div className="v8-banner" role="alert">连不上中心服务 {API_BASE || window.location.origin}：{cfg.error}</div>}
      <main className="v8-main">
        <Suspense fallback={<Empty>加载中…</Empty>}>
          {st.page === 'home' && <SiteHome config={config} set={set} />}
          {st.page === 'learn' && <OpenLearn onGo={t => (WORKBENCH_REDIRECT[t] ? window.location.assign(WORKBENCH_REDIRECT[t]) : set({ page: t }))} />}
          {st.page === 'join' && <OpenJoin config={config} />}
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
