/* v7 客户端外壳：图谱 / 3D 空间 / 进展三个界面，模型与语种是全局选择，三页共用；URL 即状态。
   数据只来自 lang_atlas.v1 端点（离线客户端从主数据包读取）；没有数据时显示空态，不回退演示数字。 */
import { Suspense, lazy, useEffect, useState } from 'react';

import AccountMenu from '../app/rdc_fusion/AccountMenu.jsx';
import { API_BASE, OFFLINE_CLIENT, useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import '../app/rdc_fusion/rdc_fusion.css';
import '../app/rdc_fusion/rdc_fusion_v6.css';
import AtlasPage from './AtlasPage.jsx';
import { LANG_LABEL, PAGES, allLanguages, defaultModel, readState, shortModel, writeState } from './v7Util.js';
import './v7.css';

const SpacePage = lazy(() => import('./SpacePage.jsx'));
const ProgressPage = lazy(() => import('./ProgressPage.jsx'));

export default function AtlasApp() {
  const cfg = useApi('/api/platform/config');
  const treeApi = useApi('/api/lang/tree', { pollWhilePending: true });
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const [st, setSt] = useState(readState);
  const model = st.model && tree?.models?.includes(st.model) ? st.model : defaultModel(tree);
  const set = patch => setSt(prev => ({ ...prev, ...patch }));

  useEffect(() => { writeState({ ...st, model: st.model }); }, [st]);
  const page = PAGES.find(p => p.id === st.page) || PAGES[0];
  useEffect(() => { document.title = `${page.label} · AI2050 语言谱图`; }, [page]);

  /* 图谱 → 3D：把该语言项设为 3D 对照的第一项 */
  const openInSpace = itemId => set({ page: 'space', view: 'layers', item: itemId,
                                      cmp: [itemId, ...st.cmp.filter(x => x !== itemId)].slice(0, 4) });
  const langs = allLanguages(tree);

  return (
    <div className="fw-root v7-root">
      <header className="v7-top">
        <a className="v7-brand" href="/" title="返回开放平台首页">
          <span className="v7-logo" aria-hidden="true">A2</span>
          <span><b>AI2050</b><small>语言谱图</small></span>
        </a>
        <nav className="v7-nav" aria-label="主导航">
          {PAGES.map(p => (
            <button key={p.id} type="button" className={st.page === p.id ? 'on' : ''} title={p.desc}
                    aria-current={st.page === p.id ? 'page' : undefined} onClick={() => set({ page: p.id })}>{p.label}</button>
          ))}
        </nav>
        <div className="v7-global">
          <label>模型
            <select value={model} onChange={e => set({ model: e.target.value })} disabled={!tree?.models?.length}>
              {!tree?.models?.length && <option value="">—</option>}
              {(tree?.models || []).map(m => <option key={m} value={m} title={m}>{shortModel(m)}</option>)}
            </select>
          </label>
          <label>语种
            <select value={st.lang} onChange={e => set({ lang: e.target.value })}>
              <option value="">全部</option>
              {langs.map(l => <option key={l} value={l}>{LANG_LABEL[l] || l}</option>)}
            </select>
          </label>
          {cfg.data?.sandbox_synthetic && <span className="v7-sandbox">合成·沙盒</span>}
          {!OFFLINE_CLIENT && <a className="v7-wb" href="/workbench" title="研究工作台：模板、冻结、分发、审计、普查（管理操作）">工作台 ↗</a>}
          <AccountMenu />
        </div>
      </header>
      {cfg.error && <div className="v7-banner" role="alert">连不上中心服务 {API_BASE || window.location.origin}：{cfg.error}</div>}
      <main className="v7-main">
        {treeApi.error && <Empty>语言谱图未接入：{treeApi.error}</Empty>}
        {!treeApi.error && !tree && <Empty>{treeApi.data?.note || '正在读取语言谱图…'}</Empty>}
        {tree && (
          <Suspense fallback={<Empty>加载中…</Empty>}>
            {st.page === 'atlas' && <AtlasPage tree={tree} model={model} lang={st.lang} st={st} set={set} onOpenSpace={openInSpace} />}
            {st.page === 'space' && <SpacePage tree={tree} model={model} lang={st.lang} st={st} set={set} />}
            {st.page === 'progress' && <ProgressPage tree={tree} model={model} st={st} set={set} />}
          </Suspense>
        )}
      </main>
    </div>
  );
}
