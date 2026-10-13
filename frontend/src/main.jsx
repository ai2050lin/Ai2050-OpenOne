import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// 旧版 3D 主界面 —— 归档为 /legacy（原样保留、无入口链接）
import App from './App.jsx'
import './index.css'
import './css/DNNAnalysisControlPanel.css'
import ErrorBoundary from './ErrorBoundary.jsx'
// v5 工作台（唯一主入口，/ 与 /rdc-fusion 等价；URL 即状态 ?lens ?obj ?atlasTab）
import RdcFusionWorkspace from './components/app/RdcFusionWorkspace.jsx'
// rdc-* 专项页统一壳（P2 收敛：页头含返回工作台；目录见图谱透镜「专项视图」tab）
import RdcPageShell from './components/app/RdcPageShell.jsx'
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

/* 路由表：[路径, 页面组件, 标题]。逐页协议化并入图谱透镜后，从此表删除对应行即可。 */
const RDC_ROUTES = [
  ['/rdc', RdcFeatureAtlas, '语言结构 · 原生坐标 · 提取算法'],
  ['/rdc-prefix', RdcPrefixAtlas, '共同前缀库 · 全坐标图谱'],
  ['/rdc-relation', RdcRelationStudy, '自然前缀 · 关系与状态更新'],
  ['/rdc-joint', RdcJointAtlas, '自然语言 · 条件坐标联合图谱'],
  ['/rdc-operator', RdcOperatorAtlas, '普通语言位置 · 完整坐标 · 真实输出'],
  ['/rdc-law', RdcLawAtlas, '条件规律与训练形成图谱'],
  ['/rdc-binding', RdcBindingAtlas, '来源绑定与训练形成图谱'],
  ['/rdc-update', RdcUpdateAtlas, '关系、学习与自回归接续图谱'],
  ['/rdc-query', RdcQueryAtlas, '条件查询与有序来源图谱'],
  ['/rdc-construction', RdcConstructionAtlas, '语境怎样构造查询'],
];

const path = typeof window !== 'undefined' ? window.location.pathname : '/';
const hit = RDC_ROUTES.find(([p]) => p === path);
const LegacyPage = hit ? hit[1] : null;
const legacyTitle = hit ? hit[2] : '';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      {path === '/legacy'
        ? <><App /><a href="/" style={{position:'fixed',right:18,bottom:18,zIndex:10000,background:'#173347',color:'#b8e5ee',border:'1px solid #4b718a',borderRadius:7,padding:'10px 16px',fontSize:12}}>返回工作台 ↗</a></>
        : hit
          ? <RdcPageShell title={legacyTitle}><LegacyPage /></RdcPageShell>
          : <RdcFusionWorkspace />}
    </ErrorBoundary>
  </StrictMode>,
)
