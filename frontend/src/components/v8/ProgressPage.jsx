/* 分支：进展与参与。进度（按大类的覆盖与研究缺口、证据门链）、参与（题库与领题）、本地包（研究包 / 谱图包在浏览器内打开）。
   各来源分开计数，不拼成转化率；任务完成、上传、审计通过都不表示证据晋级。 */
import { useState } from 'react';

import AtlasView from '../app/rdc_fusion/AtlasView.jsx';
import CatalogPanel from '../app/rdc_fusion/CatalogPanel.jsx';
import LocalPackView from '../app/rdc_fusion/LocalPackView.jsx';
import RouteBoard from '../app/rdc_fusion/RouteBoard.jsx';
import { loadLocalPack } from '../app/rdc_fusion/localPack.js';
import { Card, Empty, Tabs } from '../app/rdc_fusion/ui.jsx';
import '../app/workbench.css';
import { LevelChip } from './LangPage.jsx';
import { LEVELS, itemLevel, shortModel } from './v8Util.js';

const TABS = [{ id: 'coverage', label: '进度' }, { id: 'join', label: '参与' }, { id: 'local', label: '本地包' }];

function Coverage({ tree, model, set }) {
  const gaps = tree.items.filter(it => itemLevel(it, model) === 0);
  const piecesT = tree.items.reduce((s, it) => s + (it.status.pieces_T || 0), 0);
  return (
    <div className="v8-progress">
      <Card title="语言项覆盖" sub={`读出轴等级 · 当前模型 ${shortModel(model)}；每格是语言项个数，各等级分开计数`}>
        <table className="v6-mini-table v8-cov">
          <thead><tr><th>大类</th>{LEVELS.map(l => <th key={l.level}><LevelChip level={l.level} /></th>)}<th className="num">合计</th></tr></thead>
          <tbody>
            {tree.types.map(t => {
              const its = tree.items.filter(it => it.type === t.id);
              return (
                <tr key={t.id}>
                  <th><button type="button" className="v6-linkbtn" onClick={() => set({ page: 'lang', type: t.id, item: '' })}>{t.name}</button></th>
                  {LEVELS.map(l => <td key={l.level} className="num">{its.filter(it => itemLevel(it, model) === l.level).length || ''}</td>)}
                  <td className="num">{its.length}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="v8-note">路线 T（模板对照）共 {piecesT} 张拼图，覆盖 {tree.models.length} 个模型；路线 N（神经元经模板确认后入树）尚未接入。</p>
      </Card>
      <Card title="研究缺口" sub="当前模型上还没有数据的语言项：可以设计新的模式族，或在「参与」中领取已有题目">
        {gaps.length ? (
          <ul className="v8-gaps">
            {gaps.map(it => (
              <li key={it.id}>
                <button type="button" className="v6-linkbtn" onClick={() => set({ page: 'lang', item: it.id, type: it.type })}>{it.name}</button>
                <small>{tree.types.find(t => t.id === it.type)?.name} · {it.status.planned ? '还没有模式族' : '其他模型上有数据'}</small>
              </li>
            ))}
          </ul>
        ) : <Empty>当前模型上每个语言项都已有数据。</Empty>}
      </Card>
      <RouteBoard onGo={(_lens, _a, _b, extra) => set({ page: 'response', tm: extra?.tm || '' })} />
    </div>
  );
}

function LocalPacks() {
  const [pack, setPack] = useState(null);
  const [err, setErr] = useState('');
  const open = async e => {
    const f = e.target.files?.[0];
    if (!f) return;
    setErr('');
    try { setPack(await loadLocalPack(f)); } catch (reason) { setErr(reason.message); setPack(null); }
  };
  return (
    <div className="v8-progress">
      <div className="v6-toolbar">
        <label className="fw-tbtn">打开本地包（.zip）<input type="file" accept=".zip" onChange={open} hidden /></label>
        {pack && <span className="v8-note">{pack.name} · {pack.kind === 'respack' ? '研究包' : '谱图包'}</span>}
        {err && <span className="v8-note">{err}</span>}
      </div>
      {!pack && <Empty>研究包（响应点云「下载研究包」）与谱图包都在浏览器内解压查看，不上传到任何服务器。</Empty>}
      {pack?.kind === 'respack' && <LocalPackView pack={pack} />}
      {pack?.kind === 'atlaspack' && <AtlasView localAtlas={pack.atlas} localName={pack.name} />}
    </div>
  );
}

export default function ProgressPage({ tree, model, set }) {
  const [tab, setTab] = useState('coverage');
  return (
    <div className="v8-progress-page">
      <Tabs tabs={TABS} value={tab} onChange={setTab} label="进展与参与" />
      {tab === 'coverage' && <Coverage tree={tree} model={model} set={set} />}
      {tab === 'join' && <div className="v8-progress"><CatalogPanel onGo={(_lens, _a, _b, extra) => set({ page: 'response', tm: extra?.tm || '' })} /></div>}
      {tab === 'local' && <LocalPacks />}
    </div>
  );
}
