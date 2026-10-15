/* 本地研究包（respack.v1）：在浏览器内载入 analysis.json，离线查看 3D 响应空间与规律分析，不连接服务器。 */
import { useMemo, useState } from 'react';
import { Empty, Tabs } from './ui.jsx';
import Space3D from './Space3D.jsx';
import FindingsView from './FindingsView.jsx';

const SPLITS = ['discovery', 'confirm', 'lockbox', 'lockbox_frame'];

export default function LocalPackView({ pack }) {
  const an = pack.analysis;
  const factors = an.factors || [];
  const tags = (an.layers || []).filter(l => !l.insufficient).map(l => l.tag);
  const [view, setView] = useState('space3d');
  const [layerPick, setLayer] = useState('');
  const [method, setMethod] = useState(factors[0] ? `lda:${factors[0]}` : 'pca');
  const [colorBy, setColorBy] = useState(factors[0] ? `factor:${factors[0]}` : 'split');
  const visible = useMemo(() => new Set(SPLITS), []);
  const layer = tags.includes(layerPick) ? layerPick : tags[tags.length - 1];
  const m = pack.manifest;
  if (!an.available) return <Empty>研究包里的规律分析不可用：{an.reason || '未知原因'}</Empty>;
  return (
    <div className="v6-response">
      <div className="v6-toolbar">
        <label>层
          <select value={layer || ''} onChange={e => setLayer(e.target.value)} disabled={view !== 'space3d' || method === 'stack'}>
            {tags.map(t => <option key={t}>{t}</option>)}
          </select>
        </label>
        {view === 'space3d' && (
          <>
            <label>坐标
              <select value={method} onChange={e => setMethod(e.target.value)}>
                {factors.slice(0, 2).map(f => <option key={f} value={`lda:${f}`}>LDA · {f}</option>)}
                <option value="pca">PCA（原始）</option>
                {an.stack && <option value="stack">层堆叠 · {an.stack.factor} 轨迹</option>}
              </select>
            </label>
            <label>着色
              <select value={colorBy} onChange={e => setColorBy(e.target.value)}>
                {factors.map(f => <option key={f} value={`factor:${f}`}>{f}</option>)}
                <option value="split">划分</option>
                <option value="correct">模型作答对错（行为门）</option>
              </select>
            </label>
          </>
        )}
        <span className="v6-toolbar-src"><span className="v6-src loading" title={pack.name}>本地研究包</span></span>
      </div>
      <div className="v6-ctx">
        <b className="mono">{m.tm_id} v{m.version}</b>
        <span className="mono">{m.bucket}</span>
        <span className="mono">design {String(m.design_sha || '').slice(0, 12)}</span>
        <span>{an.n_items} 条样本{an.n_excluded ? ` · 行为门排除 ${an.n_excluded}` : ''} · 向量空间 {an.space}</span>
        <span>{m.complete ? '分片齐全' : '分片未齐'} · 生成于 {new Date(m.created_at * 1000).toLocaleString()}</span>
      </div>
      {m.synthetic && <div className="v6-synthetic"><b>合成数据 · 方法与流程验证</b><span>该研究包来自沙盒合成模式，不代表任何真实模型。</span></div>}
      <Tabs tabs={[{ id: 'space3d', label: '3D 响应空间', sub: '每点 = 一条样本' }, { id: 'findings', label: '规律分析', sub: '留出 · 置换 · 不变性' }]} value={view} onChange={setView} label="本地研究包视图" />
      <div className="v6-card">
        {view === 'space3d' && <Space3D analysis={an} layerTag={layer} method={method} colorBy={colorBy} visibleSplits={visible} />}
        {view === 'findings' && <FindingsView analysis={an} />}
      </div>
      <p className="v6-note">本地文件 {pack.name}：全部数据来自包内 analysis.json（{(m.files || []).length} 个文件，manifest 含每个文件的 sha256），不连接服务器。</p>
    </div>
  );
}
