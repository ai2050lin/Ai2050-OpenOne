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
import { installGlobalErrorCapture } from './diagnostics.js'

initDataStore()          // 启动时载入本机保存的数据包（IndexedDB）；页面优先读本地，不定时访问服务器
installGlobalErrorCapture()   // 未捕获异常/Promise 拒绝 → localStorage 环形缓冲（连接页可导出诊断包）

/* 研究工作台按需加载：新人打开首页不下载工作台的代码。
   旧版 3D 界面（/legacy）与 rdc-* 专项页已删除（2026-10-31，v9 工作台以「一句话」为中心重组；见 AGENTS.md §10）。 */
const Workbench = lazy(() => import('./components/app/Workbench.jsx'))

const path = typeof window !== 'undefined' ? window.location.pathname : '/';
const loading = <div style={{ padding: 24, color: '#64748b', fontSize: 13 }}>加载中…</div>;
const isWorkbench = path === '/workbench' || path === '/rdc-fusion' || path === '/atlas'
  || new URLSearchParams(window.location.search).get('wb') === '1';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <Suspense fallback={loading}>
        <AuthGate>{isWorkbench ? <Workbench /> : <AtlasApp />}</AuthGate>
        <DataBarHost />
      </Suspense>
    </ErrorBoundary>
  </StrictMode>,
)
