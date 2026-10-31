/* 贡献：普通用户只看到两件事——贡献算力（本机节点；网页版只给下载入口）与我的上传（我名下节点与各自上传的结果数）。
   上传数只描述服务器上收到了什么，不表示证据晋级。 */
import { Suspense, lazy } from 'react';

import { MyNodes } from '../rdc_fusion/AccountMenu.jsx';
import { useAuth } from '../rdc_fusion/authContext.js';
import { Empty } from '../rdc_fusion/ui.jsx';
import PanelBoundary from '../PanelBoundary.jsx';

const ContributePanel = lazy(() => import('../rdc_fusion/ContributePanel.jsx'));

export default function ContributePage() {
  const { user, guest } = useAuth();
  return (
    <div className="wb9-page">
      <section className="wb9-sec">
        <header><h3>贡献算力</h3><small>本机节点服务：设备、模型、资源限制、运行与日志；只有节点服务访问中心</small></header>
        <PanelBoundary name="贡献算力"><Suspense fallback={<Empty>加载中…</Empty>}><ContributePanel /></Suspense></PanelBoundary>
      </section>
      <section className="wb9-sec">
        <header><h3>我的上传</h3><small>我名下的节点与各自上传的结果份数（点击打开时连接中心读取）</small></header>
        <PanelBoundary name="我的上传">
          {user && !guest ? <MyNodes /> : <Empty>登录后显示你名下的节点与上传；访客没有上传记录。</Empty>}
        </PanelBoundary>
      </section>
    </div>
  );
}
