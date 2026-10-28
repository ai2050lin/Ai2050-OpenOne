/* 空间 · 语言谱图（atlas.v1）：跨模式族的稳定特征 → 共享编码轴 → 预注册概念假设 → 跨模型一致性。
   左：特征谱图空间（AtlasSpace3D）；右：共享编码轴 / 概念假设 / 规律陈述 / AI 解读。
   数据：GET /api/atlas（或本地谱图包 atlaspack.v1）；AI：POST /api/atlas/interpret（管理令牌），历史 GET /api/atlas/hypotheses。
   纪律：共享方向不等于共享机制；AI 假设一律「未检验」。 */
import { useMemo, useState } from 'react';
import { ADMIN_LIVE, API_BASE, fmtAgoTs, fmtNum, postJson, useAdminToken, useApi } from './api.js';
import { AdminTokenField, Empty, SourceTag, StatusPill, Tabs } from './ui.jsx';
import AtlasSpace3D from './AtlasSpace3D.jsx';
import { HYP_STATUS } from './atlasUtil.js';
import { safeName } from './catalogUtil.js';
import { CopyButton } from './CatalogPanel.jsx';

const FIND_KIND = { supported: ['pass', '支持'], negative: ['fail', '未支持'], warning: ['warn', '注意'], info: ['na', '描述'] };

function AxisCard({ ax, on, onClick }) {
  return (
    <button type="button" className={'v6-ax' + (on ? ' on' : '') + (ax.supported ? '' : ' weak')} onClick={onClick} aria-pressed={on}>
      <span className="v6-ax-hd">
        {ax.concepts.length ? ax.concepts.map(c => <em key={c} className="v6-ax-concept">{c}</em>) : <em className="v6-ax-concept novel">新发现候选</em>}
        <small className="mono">{ax.layer}</small>
        <StatusPill status={ax.supported ? 'pass' : 'fail'} label={ax.supported ? '迁移通过' : '迁移未过'} />
      </span>
      <span className="v6-ax-label">{ax.label}</span>
      <span className="v6-ax-meta">
        {ax.families.length} 族 · 平均 cos {fmtNum(ax.mean_cos)} · 留一族迁移 {fmtNum(ax.transfer)}
        {ax.replicated_in && ` · 其他模型复现 ${ax.replicated_in.length}`}
      </span>
    </button>
  );
}

function ConceptTable({ hyps }) {
  if (!hyps.length) return <Empty>没有出现在 ≥ 2 个模式族的预注册概念。在模板工作室给因子水平标注 concepts 后重新冻结与测试。</Empty>;
  return hyps.map(h => {
    const [icon, label] = HYP_STATUS[h.status] || ['na', h.status];
    return (
      <div key={h.concept} className="v6-hyp">
        <div className="v6-hyp-hd"><StatusPill status={icon} label={label} /><b>「{h.concept}」</b><small>{h.families.join('、')}</small></div>
        <table className="v6-mini-table">
          <thead><tr><th>层</th><th>声明对成边</th><th>平均 cos</th><th>留出句式预测</th></tr></thead>
          <tbody>
            {h.layers.map(p => (
              <tr key={p.layer} className={p.layer === h.best_layer ? 'best' : ''}>
                <th scope="row" className="mono">{p.layer}{p.layer === h.best_layer && <em className="v6-tag">最佳</em>}</th>
                <td className="num">{p.pairs ? `${p.linked}/${p.pairs}` : '—'}</td>
                <td className="num">{fmtNum(p.mean_cos)}</td>
                <td>{p.heldout.length ? p.heldout.map(x => (
                  <span key={x.id} className="v6-hyp-held" title={x.id}>
                    <StatusPill status={x.pass ? 'pass' : 'fail'} label={`${x.id.split(':')[0]} ${fmtNum(x.pred_cos)}`} />
                  </span>
                )) : <span className="muted">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <small className="v6-note">成员：{h.members.join('、')}</small>
      </div>
    );
  });
}

function HypothesisCards({ items }) {
  return items.map((h, i) => (
    <article key={i} className="v6-aihyp">
      <header><StatusPill status="pending" label="未检验" /><b>{h.statement}</b></header>
      {h.prediction && <p><i>可推翻的预测</i>{h.prediction}</p>}
      {h.alternative && <p><i>替代解释</i>{h.alternative}</p>}
      {h.evidence?.length > 0 && <div className="v6-aihyp-ev">{h.evidence.map(e => <code key={e}>{e}</code>)}</div>}
      {h.new_family && (
        <details>
          <summary>建议的新模式族 {h.new_family.family_id || ''}</summary>
          <pre>{JSON.stringify(h.new_family, null, 1)}</pre>
        </details>
      )}
    </article>
  ));
}

function AiPanel({ model, local }) {
  const [token, setToken] = useAdminToken();
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState('');
  const history = useApi(local ? null : '/api/atlas/hypotheses?limit=20');
  const ask = async () => {
    setBusy(true); setErr(''); setRes(null);
    try { setRes(await postJson('/api/atlas/interpret', { model, question }, { adminToken: token })); history.reload(); }
    catch (e) { setErr(e.message); }
    setBusy(false);
  };
  if (local) return <Empty>本地谱图包不连接服务器；包里的 prompt.md 是现成的 AI 提示词，可复制到任何 AI 使用。</Empty>;
  if (!ADMIN_LIVE) return <Empty>网页只能浏览，不调用 AI。AI 解读需要管理权限，请在桌面客户端操作；产生的假设一律记为「未检验」。</Empty>;
  return (
    <div className="v6-ai">
      <p className="v6-note">主模型（研发 · Agent 工作流里配置）读取当前模型的谱图摘要，提出 ≤ 5 个可检验假设。假设一律记为「未检验」：只有写成新模式族（concepts 预注册）并运行通过后才有结论。</p>
      <AdminTokenField token={token} onChange={setToken} />
      <textarea className="v6-ai-q" rows={3} placeholder="研究问题（可空）：例如「疑问方向在各层是否同一编码？哪些新测试能区分句末标点与语气？」"
        value={question} onChange={e => setQuestion(e.target.value)} />
      <button type="button" className="fw-tbtn primary" disabled={!token || busy} onClick={ask}>{busy ? 'AI 分析中…' : 'AI 解读当前谱图'}</button>
      {!token && <small className="muted">需要管理令牌（调用会消耗 API 额度）</small>}
      {err && <div className="fw-ai-err">{err}</div>}
      {res && !res.available && (
        <div className="v6-ai-fallback">
          <div className="fw-ai-err">AI 不可用：{res.reason}</div>
          <p className="v6-note">可以把下面的提示词复制到任何 AI 中使用（只含谱图统计，不含原始向量）：</p>
          <div className="v6-ai-prompt"><pre>{res.prompt}</pre><CopyButton text={`${res.system}\n\n${res.prompt}`} label="复制提示词" /></div>
        </div>
      )}
      {res?.available && (
        <>
          {res.status !== 'ok' && <div className="fw-ai-err">回复无法解析为假设（{res.status}）。原文：{String(res.raw || '').slice(0, 400)}</div>}
          <HypothesisCards items={res.hypotheses || []} />
        </>
      )}
      <h5 className="v6-h5">历史假设</h5>
      {!history.data?.items?.length && <Empty>还没有记录。</Empty>}
      {(history.data?.items || []).map(r => (
        <details key={r.id} className="v6-ai-hist">
          <summary><span className="mono">{r.id}</span> · {fmtAgoTs(r.created_at)} · {r.question || '（无问题）'} · {r.hypotheses?.length || 0} 条</summary>
          <HypothesisCards items={r.hypotheses || []} />
        </details>
      ))}
    </div>
  );
}

export default function AtlasView({ localAtlas, localName, onOpenFamily, snapshot = false }) {
  // snapshot：读中心后台定时计算的快照（开放平台公开页；谱图全量计算要十几秒）
  const remote = useApi(localAtlas ? null : (snapshot ? '/api/platform/snapshot/atlas' : '/api/atlas'));
  const atlas = localAtlas || (snapshot ? remote.data?.data : remote.data);
  const [modelPick, setModel] = useState('');
  const [layerPick, setLayer] = useState('');
  const [axisId, setAxisId] = useState('');
  const [nodeId, setNodeId] = useState('');
  const [showUnstable, setShowUnstable] = useState(true);
  const [allLayers, setAllLayers] = useState(false);
  const [tab, setTab] = useState('axes');
  const graphs = useMemo(() => atlas?.graphs || [], [atlas]);
  const g = graphs.find(x => x.model === modelPick) || graphs[0];
  const layer = g?.layers.includes(layerPick) ? layerPick : g?.layers[Math.floor((g?.layers.length || 1) / 2)];
  const node = g?.nodes.find(n => n.id === nodeId && n.layer === layer);
  const axesShown = (g?.axes || []).filter(a => allLayers || a.layer === layer);

  if (!atlas) {
    if (snapshot && remote.data?.pending) return <Empty>{remote.data.note}</Empty>;
    return <Empty>{remote.error ? `中心服务未接入（${remote.error}）。可以打开本地谱图包。` : '读取语言谱图…'}</Empty>;
  }
  if (!g) {
    const unused = (atlas.sources || []).filter(s => !s.used);
    return (
      <Empty>
        语言谱图需要同一模型下至少两个模式族的分片都已齐全。去「研发 · 自动测试」或「研发 · 题库」运行更多模式族。
        {unused.length > 0 && ` 未纳入：${unused.map(s => `${s.family_id}·${s.bucket}（${s.reason}）`).join('；')}`}
      </Empty>
    );
  }
  const synthetic = g.model.startsWith('合成·');
  const xm = atlas.cross_model;
  const pickAxis = id => { setAxisId(prev => (prev === id ? '' : id)); };
  const tabs = [
    { id: 'axes', label: '共享编码轴', badge: axesShown.length || undefined },
    { id: 'concepts', label: '概念假设', badge: g.concept_hypotheses.length || undefined },
    { id: 'findings', label: '规律陈述' },
    { id: 'ai', label: 'AI 解读' },
  ];
  return (
    <div className="v6-atlas">
      <div className="v6-toolbar">
        <label>模型
          <select value={g.model} onChange={e => { setModel(e.target.value); setAxisId(''); setNodeId(''); }}>
            {graphs.map(x => <option key={x.model} value={x.model}>{x.model}</option>)}
          </select>
        </label>
        <div className="v6-seg" role="group" aria-label="采样层">
          {g.layers.map(t => <button key={t} type="button" className={t === layer ? 'on' : ''} onClick={() => { setLayer(t); setNodeId(''); }}>{t}</button>)}
        </div>
        <label className="fw-chk"><input type="checkbox" checked={showUnstable} onChange={e => setShowUnstable(e.target.checked)} /> 显示未复现特征</label>
        {!localAtlas && <a className="fw-tbtn" href={`${API_BASE}/api/atlas/export?model=${encodeURIComponent(g.model)}`} download={`atlas_${safeName(g.model)}.zip`}>导出谱图包 ↓</a>}
        <span className="v6-toolbar-src">{localAtlas ? <span className="v6-src loading" title={localName}>本地谱图包</span> : <SourceTag state={remote.data ? 'live' : 'loading'} label="atlas.v1" />}</span>
      </div>
      <div className="v6-ctx">
        <b>{g.families.length} 个模式族</b>
        <span>{g.families.join(' · ')}</span>
        <span>{g.nodes.length} 个特征（{g.nodes.filter(n => n.stable).length} 个稳定）· {g.edges.length} 条跨族边 · {g.axes.length} 条共享轴</span>
        {xm && <span>跨模型：可检验边 {xm.edges_testable}，复现 {xm.edges_replicated}（{fmtNum(xm.replication_rate, 2)}）</span>}
        {g.skipped.length > 0 && <span className="warn">未纳入：{g.skipped.map(s => s.family).join('、')}</span>}
      </div>
      {synthetic && (
        <div className="v6-synthetic"><b>合成数据 · 方法与流程验证</b><span>该模型桶来自沙盒合成模式：共享方向是按文法 concepts 植入的阳性对照，不代表任何真实模型。</span></div>
      )}
      <div className="v6-atlas-body">
        <div className="v6-atlas-main">
          <AtlasSpace3D graph={g} layer={layer} axisId={axisId} nodeId={node?.id} onSelectNode={id => setNodeId(id || '')} showUnstable={showUnstable} />
          <p className="v6-note">坐标 = {layer} 层全部特征单位方向的经典 MDS（距离 √(2−2cos)）：方向相近的特征靠在一起；只在同一模型内有意义。实线 = 预注册概念之间的边，虚线 = 未预注册（新发现候选）。共享方向不等于共享机制，机制需要沿轴干预。</p>
        </div>
        <aside className="v6-atlas-side">
          {node && (
            <div className="v6-node">
              <div className="v6-node-hd"><b>{node.family} · {node.factor}={node.level}</b><button type="button" className="fw-tbtn ghost" onClick={() => setNodeId('')} aria-label="关闭">×</button></div>
              <div className="v6-cat-meta">
                <span>层 <b className="mono">{node.layer}</b></span>
                {node.concept && <span>概念 <b>{node.concept}</b></span>}
                <span>{node.heldout_only ? '只出现在留出句式' : <>复现 cos <b>{fmtNum(node.rep)}</b> · 效应/噪声 <b>{fmtNum(node.snr, 1)}</b></>}</span>
                <span>{node.stable ? '稳定' : node.heldout_only ? '未见句式预测对象' : '未复现'}</span>
              </div>
              {onOpenFamily && node.tm_id && (
                <button type="button" className="fw-tbtn primary" onClick={() => onOpenFamily({ tm: node.tm_id, bucket: g.model, layer: node.layer })}>在实验响应中查看 {node.family} →</button>
              )}
            </div>
          )}
          <Tabs tabs={tabs} value={tab} onChange={setTab} label="谱图面板" />
          <div className="v6-atlas-panel">
            {tab === 'axes' && (
              <>
                <label className="fw-chk"><input type="checkbox" checked={allLayers} onChange={e => setAllLayers(e.target.checked)} /> 显示全部层</label>
                {!axesShown.length && <Empty>{allLayers ? '该模型没有共享编码轴。' : `${layer} 层没有共享编码轴。`}</Empty>}
                {axesShown.map(ax => <AxisCard key={ax.id} ax={ax} on={axisId === ax.id} onClick={() => { if (ax.layer !== layer) setLayer(ax.layer); pickAxis(ax.id); }} />)}
                <p className="v6-note">判据：两次估计 cos ≥ {atlas.criteria.min_cos}、稳健 z ≥ {atlas.criteria.min_z}；共享轴需留一族迁移 ≥ {atlas.criteria.min_transfer}。点击轴高亮其成员。</p>
              </>
            )}
            {tab === 'concepts' && <ConceptTable hyps={g.concept_hypotheses} />}
            {tab === 'findings' && g.findings.map(f => (
              <article key={f.id} className={'v6-find ' + f.kind}>
                <header><StatusPill status={FIND_KIND[f.kind]?.[0] || 'na'} label={FIND_KIND[f.kind]?.[1]} /><b>{f.title}</b>{f.level && <em>{f.level}</em>}</header>
                <p>{f.text}</p>
              </article>
            ))}
            {tab === 'ai' && <AiPanel model={g.model} local={Boolean(localAtlas)} />}
          </div>
        </aside>
      </div>
      {atlas.policy && <p className="v6-note">{atlas.policy}</p>}
    </div>
  );
}
