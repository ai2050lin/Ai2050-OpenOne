/* RdcPageShell：rdc-* 专项页统一壳（P2 收敛）。
   只做页头（返回工作台 + 页面标题 + 收敛注记）与主容器，不改写页面自身内容与样式。
   路由挂载点 = src/main.jsx 路由表；目录入口 = 工作台·图谱透镜「专项视图」tab。
   逐页协议化并入图谱透镜后，独立路由随之删除。 */
export default function RdcPageShell({ title, children }) {
  return (
    <div style={{ minHeight: '100vh' }}>
      <header style={{
        position: 'sticky', top: 0, zIndex: 50,
        display: 'flex', alignItems: 'baseline', gap: 12,
        padding: '10px 20px',
        background: 'rgba(255,255,255,.94)',
        borderBottom: '1px solid rgba(0,0,0,.08)',
      }}>
        <a href="/" style={{ fontSize: 12.5, color: '#185fa5', textDecoration: 'none', whiteSpace: 'nowrap' }}>← 返回工作台</a>
        <b style={{ fontSize: 14, fontWeight: 500 }}>{title}</b>
        <span style={{ fontSize: 11.5, color: '#8a8f98' }}>RDC 专项视图 · 目录见工作台·图谱透镜「专项视图」· 逐步并入后收敛</span>
      </header>
      <main>{children}</main>
    </div>
  );
}
