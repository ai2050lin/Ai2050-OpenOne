import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './css/DNNAnalysisControlPanel.css'
import ErrorBoundary from './ErrorBoundary.jsx'
// 网站外壳（默认入口 /；Electron 的 file:// 页面同样进入）：首页 · 介绍 · 参与；语言图谱与神经元在研究工作台（/workbench，/atlas 同）
import AtlasApp from './components/v8/AtlasApp.jsx'
import AuthGate from './components/app/rdc_fusion/AuthGate.jsx'
import { DataBarHost } from './components/open/DataBar.jsx'
import { initDataStore } from './components/app/rdc_fusion/dataStore.js'

initDataStore()          // 启动时载入本机保存的数据包（IndexedDB）；页面优先读本地，不定时访问服务器

/* 其余入口按需加载：新人打开首页不下载工作台、旧版 3D 界面与专项页的代码。 */
// 旧版 3D 主界面 —— 归档为 /legacy（原样保留、无入口链接）
const App = lazy(() => import('./App.jsx'))
// 研究工作台 v6（/workbench，旧入口 /rdc-fusion 等价；URL 即状态 ?lens ?view ?station ?tm）
const Workbench = lazy(() => import('./components/app/Workbench.jsx'))
// rdc-* 专项页统一壳（P2 收敛：页头含返回工作台；目录见图谱透镜「专项视图」tab）
const RdcPageShell = lazy(() => import('./components/app/RdcPageShell.jsx'))

/* 路由表：[路径, 页面组件, 标题]。逐页协议化并入图谱透镜后，从此表删除对应行即可。 */
const RDC_ROUTES = [
  ['/rdc', () => import('./components/app/RdcFeatureAtlas.jsx'), '语言结构 · 原生坐标 · 提取算法'],
  ['/rdc-prefix', () => import('./components/app/RdcPrefixAtlas.jsx'), '共同前缀库 · 全坐标图谱'],
  ['/rdc-relation', () => import('./components/app/RdcRelationStudy.jsx'), '自然前缀 · 关系与状态更新'],
  ['/rdc-joint', () => import('./components/app/RdcJointAtlas.jsx'), '自然语言 · 条件坐标联合图谱'],
  ['/rdc-operator', () => import('./components/app/RdcOperatorAtlas.jsx'), '普通语言位置 · 完整坐标 · 真实输出'],
  ['/rdc-law', () => import('./components/app/RdcLawAtlas.jsx'), '条件规律与训练形成图谱'],
  ['/rdc-binding', () => import('./components/app/RdcBindingAtlas.jsx'), '来源绑定与训练形成图谱'],
  ['/rdc-update', () => import('./components/app/RdcUpdateAtlas.jsx'), '关系、学习与自回归接续图谱'],
  ['/rdc-query', () => import('./components/app/RdcQueryAtlas.jsx'), '条件查询与有序来源图谱'],
  ['/rdc-construction', () => import('./components/app/RdcConstructionAtlas.jsx'), '语境怎样构造查询'],
];

const path = typeof window !== 'undefined' ? window.location.pathname : '/';
/* 旧版 3D 界面与 rdc-* 专项页依赖研究者本机的分片研究服务，只在开发模式（vite dev）可达；公开构建回到首页 */
const devOnly = import.meta.env.DEV;
const hit = devOnly ? RDC_ROUTES.find(([p]) => p === path) : null;
const LegacyPage = hit ? lazy(hit[1]) : null;
const legacyTitle = hit ? hit[2] : '';
const loading = <div style={{ padding: 24, color: '#64748b', fontSize: 13 }}>加载中…</div>;

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <Suspense fallback={loading}>
        {devOnly && path === '/legacy'
          ? <><App /><a href="/" style={{position:'fixed',right:18,bottom:18,zIndex:10000,background:'#173347',color:'#b8e5ee',border:'1px solid #4b718a',borderRadius:7,padding:'10px 16px',fontSize:12}}>返回首页 ↗</a></>
          : hit
            ? <RdcPageShell title={legacyTitle}><LegacyPage /></RdcPageShell>
            : (path === '/workbench' || path === '/rdc-fusion' || path === '/atlas' || new URLSearchParams(window.location.search).get('wb') === '1')
              ? <AuthGate><Workbench /></AuthGate>
              : <AuthGate><AtlasApp /></AuthGate>}
        {!hit && !(devOnly && path === '/legacy') && <DataBarHost />}
      </Suspense>
    </ErrorBoundary>
  </StrictMode>,
)
