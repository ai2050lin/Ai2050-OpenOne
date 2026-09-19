import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// 3D可视化主界面（炫酷风格）
import App from './App.jsx'
// 新工作台架构备份: import App from './AppNew.jsx'
import './index.css'
import './css/DNNAnalysisControlPanel.css'
import ErrorBoundary from './ErrorBoundary.jsx'
import RdcFeatureAtlas from './components/app/RdcFeatureAtlas.jsx'
import RdcPrefixAtlas from './components/app/RdcPrefixAtlas.jsx'
import RdcRelationStudy from './components/app/RdcRelationStudy.jsx'
import RdcJointAtlas from './components/app/RdcJointAtlas.jsx'
import RdcOperatorAtlas from './components/app/RdcOperatorAtlas.jsx'
import RdcLawAtlas from './components/app/RdcLawAtlas.jsx'
import RdcBindingAtlas from './components/app/RdcBindingAtlas.jsx'
import RdcUpdateAtlas from './components/app/RdcUpdateAtlas.jsx'
import RdcQueryAtlas from './components/app/RdcQueryAtlas.jsx'
import RdcConstructionAtlas from './components/app/RdcConstructionAtlas.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      {window.location.pathname === '/rdc-construction' ? <RdcConstructionAtlas /> : window.location.pathname === '/rdc-query' ? <RdcQueryAtlas /> : window.location.pathname === '/rdc-update' ? <RdcUpdateAtlas /> : window.location.pathname === '/rdc-binding' ? <RdcBindingAtlas /> : window.location.pathname === '/rdc-law' ? <RdcLawAtlas /> : window.location.pathname === '/rdc-operator' ? <RdcOperatorAtlas /> : window.location.pathname === '/rdc-joint' ? <RdcJointAtlas /> : window.location.pathname === '/rdc-relation' ? <RdcRelationStudy /> : window.location.pathname === '/rdc-prefix' ? <RdcPrefixAtlas /> : window.location.pathname === '/rdc' ? <RdcFeatureAtlas /> : <><App /><a href="/rdc-query" style={{position:'fixed',right:18,bottom:18,zIndex:10000,background:'#173347',color:'#b8e5ee',border:'1px solid #4b718a',borderRadius:7,padding:'10px 16px',fontSize:12}}>RDC 条件查询与有序来源图谱 ↗</a></>}
    </ErrorBoundary>
  </StrictMode>,
)
