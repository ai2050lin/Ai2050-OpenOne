/* 开放平台 · 图谱：语言谱图（atlas.v1 快照）与原生神经元单元卡。复用工作台的同名视图，数据只读。 */
import { useState } from 'react';

import AtlasView from '../app/rdc_fusion/AtlasView.jsx';
import NeuronUnitPanel from '../app/rdc_fusion/NeuronUnitPanel.jsx';

const VIEWS = [
  { id: 'atlas', label: '语言谱图', desc: '跨模式族的特征、共享编码轴与概念假设' },
  { id: 'units', label: '神经元单元卡', desc: '自然语料普查、两极片段、读出 / 控制 / 解释三条证据轴' },
];

export default function OpenAtlas() {
  const [view, setView] = useState(() => {
    try { return new URLSearchParams(window.location.search).get('unit') ? 'units' : 'atlas'; } catch { return 'atlas'; }
  });
  return (
    <div className="op-page op-atlas">
      <div className="op-subtabs" role="tablist" aria-label="图谱视图">
        {VIEWS.map(v => (
          <button key={v.id} type="button" role="tab" aria-selected={view === v.id} className={view === v.id ? 'on' : ''} onClick={() => setView(v.id)}>
            <b>{v.label}</b><small>{v.desc}</small>
          </button>
        ))}
      </div>
      <div className="op-atlas-body">
        {view === 'atlas' ? <AtlasView snapshot /> : <NeuronUnitPanel />}
      </div>
    </div>
  );
}
