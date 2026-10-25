/* 开放平台外壳（platform.v1）：面向新人的默认入口 —— 首页 / 学习 / 图谱 / 参与，另有「研究工作台」进入完整工作区。
   配置来自中心 /api/platform/config（站点标题、注册方式、任务类型、所需模型）；URL 即状态（?tab=）。
   没有数据时各页显示空态或「未接入」，不回退演示数字。 */
import { useEffect, useState } from 'react';

import AccountMenu from '../app/rdc_fusion/AccountMenu.jsx';
import AuthGate from '../app/rdc_fusion/AuthGate.jsx';
import { API_BASE, useApi } from '../app/rdc_fusion/api.js';
import '../app/rdc_fusion/rdc_fusion.css';
import '../app/rdc_fusion/rdc_fusion_v6.css';
import OpenAtlas from './OpenAtlas.jsx';
import OpenHome from './OpenHome.jsx';
import OpenJoin from './OpenJoin.jsx';
import OpenLearn from './OpenLearn.jsx';
import { TABS, readTab, writeTab } from './openUtil.js';
import './open.css';

function Shell() {
  const cfg = useApi('/api/platform/config');
  const [tab, setTab] = useState(readTab);
  const config = cfg.data;
  const tabs = TABS.filter(t => !config?.ui?.tabs || config.ui.tabs.includes(t.id));
  useEffect(() => {
    const t = TABS.find(x => x.id === tab);
    if (config?.title) document.title = `${t && t.id !== 'home' ? t.label + ' · ' : ''}${config.title}`;
  }, [config, tab]);
  const go = t => { setTab(t); writeTab(t); window.scrollTo?.(0, 0); };
  return (
    <div className="fw-root op-root">
      <header className="op-top">
        <a className="op-brand" href="/" onClick={e => { e.preventDefault(); go('home'); }}>
          <span className="op-logo" aria-hidden="true">A2</span>
          <span><b>AI2050</b><small>开源 LLM 逆向研究平台</small></span>
        </a>
        <nav className="op-nav" aria-label="主导航">
          {tabs.map(t => (
            <button key={t.id} type="button" className={tab === t.id ? 'on' : ''} aria-current={tab === t.id ? 'page' : undefined}
                    onClick={() => go(t.id)} title={t.desc}>{t.label}</button>
          ))}
        </nav>
        <div className="op-top-right">
          {config?.sandbox_synthetic && <span className="op-sandbox" title="该中心允许合成数据，只用于流程验证">合成·沙盒</span>}
          <a className="op-link" href="/workbench" title="研究工作台：题库、模板工作室、试跑与冻结、实验响应、审计、普查作业、证据门链">研究工作台 ↗</a>
          <AccountMenu />
        </div>
      </header>
      {cfg.error && <div className="op-banner" role="alert">连不上中心服务 {API_BASE || location.origin}：{cfg.error}</div>}
      <main className="op-main">
        {tab === 'home' && <OpenHome config={config} onGo={go} />}
        {tab === 'learn' && <OpenLearn onGo={go} />}
        {tab === 'atlas' && <OpenAtlas />}
        {tab === 'join' && <OpenJoin config={config} />}
      </main>
      <footer className="op-foot">
        <span>{config?.title || 'AI2050'} · v{config?.version || '—'}</span>
        <span>代码 {config?.license?.code || 'Apache-2.0'} · 数据 {config?.license?.data || 'CC BY 4.0'}</span>
        {config?.repo_url && <a href={config.repo_url} target="_blank" rel="noreferrer">源代码</a>}
        {config?.contact && <span>联系：{config.contact}</span>}
      </footer>
    </div>
  );
}

export default function OpenPlatform() {
  return <AuthGate><Shell /></AuthGate>;
}
