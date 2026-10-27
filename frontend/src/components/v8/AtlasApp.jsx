/* v8 客户端外壳（/atlas）：两条主线 —— ① 语言图谱、② 神经元；其余功能是分支，从主线内的按钮或「更多」进入。
   模型是全局选择，两条主线共用（神经元主线用该模型的普查）；URL 即状态。没有数据时显示空态，不回退演示数字。 */
import { Suspense, lazy, useEffect, useRef, useState } from 'react';

import AccountMenu from '../app/rdc_fusion/AccountMenu.jsx';
import { API_BASE, OFFLINE_CLIENT, useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import '../app/rdc_fusion/rdc_fusion.css';
import '../app/rdc_fusion/rdc_fusion_v6.css';
import LangPage from './LangPage.jsx';
import { PAGES, defaultModel, readState, shortModel, writeState } from './v8Util.js';
import './v8.css';

const NeuronPage = lazy(() => import('./NeuronPage.jsx'));
const ProgressPage = lazy(() => import('./ProgressPage.jsx'));
const ResponseExplorer = lazy(() => import('../app/rdc_fusion/ResponseExplorer.jsx'));

function MoreMenu({ set }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const close = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, []);
  const go = patch => { set(patch); setOpen(false); };
  return (
    <div className="v8-more" ref={ref}>
      <button type="button" aria-expanded={open} onClick={() => setOpen(v => !v)}>更多 ▾</button>
      {open && (
        <ul role="menu">
          <li><button type="button" role="menuitem" onClick={() => go({ page: 'progress' })}>进展与参与<small>覆盖与研究缺口、证据门链、题库与领题、本地包</small></button></li>
          <li><button type="button" role="menuitem" onClick={() => go({ page: 'response' })}>响应点云<small>单个模式族的样本点云与规律分析</small></button></li>
          {!OFFLINE_CLIENT && <li><a role="menuitem" href="/workbench">工作台 ↗<small>模板、冻结、分发、审计、普查（管理操作）</small></a></li>}
          <li><a role="menuitem" href="/">开放平台首页 ↗<small>学习、参与与贡献榜</small></a></li>
        </ul>
      )}
    </div>
  );
}

export default function AtlasApp() {
  const cfg = useApi('/api/platform/config');
  const treeApi = useApi('/api/lang/tree', { pollWhilePending: true });
  const skApi = useApi('/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s]));
  const [st, setSt] = useState(readState);
  const model = st.model && tree?.models?.includes(st.model) ? st.model : defaultModel(tree, skeletons);
  const set = patch => setSt(prev => ({ ...prev, ...patch }));

  useEffect(() => { writeState(st); }, [st]);
  const label = { lang: '语言图谱', neuron: '神经元', progress: '进展与参与', response: '响应点云' }[st.page] || '';
  useEffect(() => { document.title = `${label} · AI2050`; }, [label]);
  useEffect(() => { window.scrollTo?.(0, 0); }, [st.page]);

  return (
    <div className="fw-root v8-root">
      <header className="v8-top">
        <a className="v8-brand" href="/" title="返回开放平台首页"><span className="v8-logo" aria-hidden="true">A2</span><span><b>AI2050</b><small>LLM 语言编码研究</small></span></a>
        <nav className="v8-nav" aria-label="主线">
          {PAGES.map(p => (
            <button key={p.id} type="button" className={st.page === p.id ? 'on' : ''} title={p.desc} aria-current={st.page === p.id ? 'page' : undefined}
                    onClick={() => set({ page: p.id })}><span className="n">{p.n}</span>{p.label}</button>
          ))}
          {(st.page === 'progress' || st.page === 'response') && <span className="v8-branch">分支 · {label}</span>}
        </nav>
        <div className="v8-right">
          {cfg.data?.sandbox_synthetic && <span className="v8-sandbox">合成·沙盒</span>}
          <label>模型
            <select value={model} onChange={e => set({ model: e.target.value, block: null, unit: null })} disabled={!tree?.models?.length}>
              {!tree?.models?.length && <option value="">—</option>}
              {(tree?.models || []).map(m => <option key={m} value={m} title={m}>{shortModel(m)}{skeletons[m]?.census_run ? '' : '（未普查）'}</option>)}
            </select>
          </label>
          <MoreMenu set={set} />
          <AccountMenu />
        </div>
      </header>
      {cfg.error && <div className="v8-banner" role="alert">连不上中心服务 {API_BASE || window.location.origin}：{cfg.error}</div>}
      <main className="v8-main">
        {treeApi.error && <Empty>语言谱图未接入：{treeApi.error}</Empty>}
        {!treeApi.error && !tree && <Empty>{treeApi.data?.note || '正在读取语言谱图…'}</Empty>}
        {tree && (
          <Suspense fallback={<Empty>加载中…</Empty>}>
            {st.page === 'lang' && <LangPage tree={tree} model={model} skeleton={skeletons[model]} st={st} set={set} />}
            {st.page === 'neuron' && <NeuronPage model={model} skeleton={skeletons[model]} skeletons={skeletons} st={st} set={set} />}
            {st.page === 'progress' && <ProgressPage tree={tree} model={model} set={set} />}
            {st.page === 'response' && <ResponseExplorer key={`${st.tm}|${model}`} initialTm={st.tm || undefined} initialBucket={model || undefined} />}
          </Suspense>
        )}
      </main>
    </div>
  );
}
