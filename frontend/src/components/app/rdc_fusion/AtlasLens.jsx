/* AtlasLens 图谱透镜 v1.11（tab 收敛 4 + 全景四模；design/atlas_module_space3d_scheme_v1.md）：
   - 图谱全景（默认 · 特征唯一家）：指标条 + E 级分布条 + 四模式切换（模块=2D 双栏对照 AtlasModuleView /
     模块空间=三舱一塔三桥 3D AtlasModuleSpace / 模板空间=模板 3D AtlasTemplateSpace / 卡片=特征卡网格），
     四模共享同一选中态与 FeatureDetail 详情抽屉；registry 版本链 select 移入全景头；底部「专项视图」折叠区
   - v1.10 去重（用户确认执行）：退役 map/gallery/list 三模式；AtlasGallery/AtlasTree 组件已删（留档 git 历史），
     AtlasPanorama 保留（总览 home 首屏仍引用）但不再进入图谱全景；
     gallery3d 派生数据段保留不删
   - 跨线账本：GET /api/ai-rnd/atlas/ledger 分页时间线（append-only 只读；唯一 Phase 时间线，
     全景指标条两卡可点击直达）
   - 结果库：ResultsExplorer mode="results"（原技术透镜「按结果」面，知识轴归图谱）
   - 模板与技术（原「注册表」）：RegistryPanel——数据产线资产（语言模板 × 分析技术 × 研究包 × 成熟度），
     与特征 registry 撞名已消除
   - v1.6 变更：撤销「特征登记表」「专项视图」独立 tab（特征并入全景第三模式；专项降为折叠区）；
     删「最近闭环」列表（≡账本，唯一家原则）；旧 ?atlasTab=tree|map|features|special 兼容归一到全景
     （features 归一时默认切到卡片模式）。
   - 证据链钻取：特征详情抽屉（复现协议 / 关键读数 / anchors 逐条 res/seal sha8，协议键驱动，零实验字面量）
   - 队列/闭环不再在此重复：特征详情「→ 研发透镜」跳转（onGo('process')）。
   LIVE/DEMO 范式：服务未启动时显示空态提示，不伪造数据。 */
import { useEffect, useState, useCallback } from 'react';
import ResultsExplorer from './ResultsExplorer.jsx';
import RegistryPanel from './RegistryPanel.jsx';
import AtlasModuleView from './AtlasModuleView.jsx';
import AtlasModuleSpace from './AtlasModuleSpace.jsx';
import AtlasTemplateSpace from './AtlasTemplateSpace.jsx';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const FAM_NAME = {
  read: '知识读出', syntax: '语法轴', mech: '机制', context: '条件化',
  limit: '边界', spatial: '空间', logic: '逻辑',
};
const ECLASS = {
  E2_predictive: 'at-e2', E1_repeatable: 'at-e1', E3_causal_scoped: 'at-e3',
};

/* 专项视图目录（P2 收敛）：10 个 /rdc-* 独立页面按研究主题分三组。
   页面名称/路径为 UI 导航词表（非实验字面量）；每页已挂 RdcPageShell 统一壳。 */
const SPECIAL_VIEWS = [
  ['特征与坐标', [
    ['/rdc', 'RdcFeatureAtlas', '语言结构 · 原生坐标 · 提取算法'],
    ['/rdc-prefix', 'RdcPrefixAtlas', '共同前缀库 · 全坐标图谱'],
    ['/rdc-operator', 'RdcOperatorAtlas', '普通语言位置 · 完整坐标 · 真实输出'],
  ]],
  ['关系与更新', [
    ['/rdc-relation', 'RdcRelationStudy', '自然前缀 · 关系与状态更新'],
    ['/rdc-update', 'RdcUpdateAtlas', '关系、学习与自回归接续图谱'],
    ['/rdc-binding', 'RdcBindingAtlas', '来源绑定与训练形成图谱'],
  ]],
  ['条件与规律', [
    ['/rdc-joint', 'RdcJointAtlas', '自然语言 · 条件坐标联合图谱'],
    ['/rdc-law', 'RdcLawAtlas', '条件规律与训练形成图谱'],
    ['/rdc-query', 'RdcQueryAtlas', '条件查询与有序来源图谱'],
    ['/rdc-construction', 'RdcConstructionAtlas', '语境怎样构造查询'],
  ]],
];

async function jget(path, opts) {
  const r = await fetch(`${API_BASE}/api/ai-rnd${path}`, opts);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}

/* 特征详情抽屉（全景与特征登记表共用）：复现协议 / 关键读数 / anchors 证据链（res/seal sha8） */
function FeatureDetail({ f, onGo, onClose }) {
  return (
    <div className="at-detail">
      <div className="at-detail-head">
        <span className="mono at-fid">{f.id}</span>
        <span className={'at-badge ' + (ECLASS[f.evidence_level] || 'at-e1')}>{f.evidence_level}</span>
        <span className="at-fam">{f.model_scope.join(' · ')}</span>
        {onGo && <button className="at-link" onClick={() => onGo('process')}
          title="队列与五段式闭环在研发透镜（唯一归属）">→ 研发透镜</button>}
        <button className="at-close" onClick={onClose}>×</button>
      </div>
      <div className="at-stmt">{f.statement}</div>
      <div className="at-sub">复现协议</div>
      <div className="at-mono">{f.replication}</div>
      {f.scope_limits && (<><div className="at-sub">适用范围</div><div className="at-mono">{f.scope_limits}</div></>)}
      {Object.keys(f.values || {}).length > 0 && (<>
        <div className="at-sub">关键读数</div>
        <div className="at-mono">{Object.entries(f.values).map(([k, v]) => `${k}=${v}`).join('  ')}</div>
      </>)}
      <div className="at-sub">证据链（anchors，逐条 res/seal sha8 可回查）</div>
      {f.anchors.map((a, i) => (
        <div key={i} className="at-anchor">
          <span className="mono">[{a.src}·P{a.phase}]</span> {a.role}
          {(a.tags || []).map(t => <span key={t} className="at-tag">{t}</span>)}
          {a.res_sha8 && <span className="mono at-sha">res={a.res_sha8} seal={a.seal_sha8}</span>}
        </div>
      ))}
      {f.counter_evidence.length > 0 && (<>
        <div className="at-sub">反证 / 失败关联</div>
        {f.counter_evidence.map((c, i) => <div key={i} className="at-counter">{c}</div>)}
      </>)}
    </div>
  );
}

function ValueChips({ values }) {
  const ks = Object.keys(values || {});
  if (!ks.length) return null;
  return (
    <div className="at-values">
      {ks.slice(0, 4).map(k => (
        <span key={k} className="at-chip mono">{k}={String(values[k]).slice(0, 8)}</span>
      ))}
      {ks.length > 4 && <span className="at-chip">+{ks.length - 4}</span>}
    </div>
  );
}

export default function AtlasLens({ on, onGo, initialTab, initialSel }) {
  /* v1.6：tab 收敛为 4；旧 ?atlasTab=tree|map|features|special 兼容归一到全景（features 归一到卡片模式） */
  const normTab = t => ({ tree: 'overview', map: 'overview', features: 'overview', special: 'overview' }[t] || t || 'overview');
  const [tab, setTab] = useState(normTab(initialTab));        // overview | ledger | results | registry
  /* v1.10 去重（design/atlas_lens_dedup_audit_v1.md）：退役 map/gallery/list 三模式
     （AtlasGallery/AtlasTree 已删，AtlasPanorama 保留供总览首屏）。
     v1.11（scheme design/atlas_module_space3d_scheme_v1.md，用户确认落地 module_space.v1）：
     全景默认 = 模块视图（2D 双栏查询工具）；新增「模块空间」模式（module_space.v1，三舱一塔三桥
     3D 关系发现工具，数据 = atlas_tree_v2.module_space 派生段）。「模板空间」保留（模板粒度），
     「卡片」保留（特征扫读）。特征选中态共用 FeatureDetail 抽屉协议，不引入第二事实源。 */
  const [panMode, setPanMode] = useState(initialTab === 'features' ? 'cards' : 'modules');  // modules/modspace/tplspace/cards
  const [data, setData] = useState(null);                // registry payload
  const [versions, setVersions] = useState([]);
  const [version, setVersion] = useState(null);
  const [live, setLive] = useState(null);                // true LIVE / false 无法连接
  const [sel, setSel] = useState(null);                  // 选中特征 id
  const [ledger, setLedger] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { if (initialTab) setTab(normTab(initialTab)); }, [initialTab]);
  /* v1.7：?atlasSel=FTR-xx（顶栏特征源选择器 / URL 直达）→ 打开全景详情抽屉（协议键，不引入第二事实源） */
  useEffect(() => { if (initialSel) setSel(initialSel); }, [initialSel]);

  const loadRegistry = useCallback(async (v) => {
    setErr(null);
    try {
      const p = jget('/atlas/registry' + (v ? `?version=${encodeURIComponent(v)}` : ''));
      const d = await p;
      setData(d); setVersions(d.versions || []); setVersion(d.version); setLive(true);
    } catch (e) { setErr(String(e.message || e)); setLive(false); }
  }, []);

  const loadLedger = useCallback(async () => {
    try {
      const d = await jget('/atlas/ledger?limit=30');
      setLedger(d); setLive(true);
    } catch (e) { setErr(String(e.message || e)); setLive(false); }
  }, []);

  useEffect(() => { if (on) loadRegistry(); }, [on, loadRegistry]);
  useEffect(() => { if (on && !ledger) loadLedger(); }, [on, ledger, loadLedger]);   // 总览与账本共用

  if (!on) return null;
  const counts = data ? data.features.reduce((m, f) => {
    m[f.evidence_level] = (m[f.evidence_level] || 0) + 1; return m;
  }, {}) : {};
  /* 总览聚合（协议键驱动，零实验字面量：一切数字来自 registry / ledger 端点） */
  /* 账本跨线交错（非按 phase 排序）：最新闭环按 phase 最大值取，保证「最新」语义 */
  const maxPhaseM = ledger && ledger.measurements.length
    ? ledger.measurements.reduce((a, m) => (m.phase > a.phase ? m : a))
    : null;
  const e2plus = (counts.E2_predictive || 0) + (counts.E3_causal_scoped || 0);

  return (
    <div className="atlas-lens">
      <div className="at-toolbar">
        <div className="at-tabs">
          {[['overview', '图谱全景'], ['ledger', '跨线账本'], ['results', '结果库'], ['registry', '模板与技术']].map(([k, lb]) => (
            <button key={k} className={'at-tab' + (tab === k ? ' on' : '')}
              onClick={() => setTab(k)}>{lb}</button>
          ))}
        </div>
        {live !== null && <span className={'at-badge ' + (live ? 'at-live' : 'at-dead')}>{live ? 'LIVE' : '未连接'}</span>}
      </div>

      {err && tab !== 'results' && tab !== 'registry' && <div className="at-err">图谱服务不可达（{err}）——启动 :5001 后自动重试（切换 tab 触发）。</div>}

      {/* ===== 总览（聚合首屏）：整体进度 → 点击钻取明细 ===== */}
      {tab === 'overview' && data && (
        <div className="at-overview">
          <div className="at-ov-stats">
            <div className="at-ov-stat link" title="唯一 Phase 时间线在「跨线账本」——点击直达"
              onClick={() => setTab('ledger')}><div className="n mono">{maxPhaseM ? 'P' + maxPhaseM.phase : (ledger ? '—' : '…')}</div><div className="l">最新闭环 Phase →</div></div>
            <div className="at-ov-stat link" title="唯一时间线家 =「跨线账本」——点击直达"
              onClick={() => setTab('ledger')}><div className="n">{ledger ? ledger.total : '…'}</div><div className="l">账本 measurements →</div></div>
            <div className="at-ov-stat"><div className="n">{data.features.length}</div><div className="l">在册特征 · registry v{data.version}</div></div>
            <div className="at-ov-stat hl"><div className="n">{e2plus}<small>/{data.features.length}</small></div><div className="l">E2+ 确证特征</div></div>
            <div className="at-ov-stat"><div className="n">{data.upgrade_log.length}</div><div className="l">E 级升级</div></div>
            <div className="at-ov-stat"><div className="n">{data.failures.length}</div><div className="l">失败账本 F</div></div>
          </div>

          <div className="at-ov-bar" title="特征证据级分布（E2 可预测 / E3 协议内因果 / E1 可重复）">
            {['E2_predictive', 'E3_causal_scoped', 'E1_repeatable'].map(k => counts[k] ? (
              <span key={k} className={'at-ov-seg ' + (ECLASS[k] || 'at-e1')}
                style={{ width: (100 * counts[k] / data.features.length) + '%' }}>
                {k.split('_')[0]} · {counts[k]}
              </span>
            ) : null)}
          </div>

          {/* ===== 图谱全景：三模一态（图形/列表/卡片共享选中态 + 详情抽屉）；版本链 select 归此 ===== */}
          <div className="at-pan-head">
            <b>三域图谱全景</b>
            <span className="dim2">知识网络 / 分析推理 / 语法系统 → 语言族 → 特征（E 级着色 · 机制横切 · 交叉边）</span>
            {versions.length > 1 && (
              <select className="at-ver" value={version || ''} onChange={e => loadRegistry(e.target.value)}
                title="特征 registry 版本链（客户端已登记版本；研究侧最新为 v1.28，跨线施加挂账中）">
                {versions.map(v => <option key={v.version} value={v.version}>registry v{v.version}</option>)}
              </select>
            )}
            <span className="at-pan-toggle">
              <button type="button" className={'at-tab' + (panMode === 'modules' ? ' on' : '')} onClick={() => setPanMode('modules')}
                title="三模块视图（atlas_tree_v2）：知识网络/推理功能/语法机制 × 语言模板↔编码机制双栏对照 + 关联边三态 + 缺口清单">模块</button>
              <button type="button" className={'at-tab' + (panMode === 'modspace' ? ' on' : '')} onClick={() => setPanMode('modspace')}
                title="三模块 3D 关系空间（module_space.v1）：三舱一塔三桥——KN 晶格舱/AR 链舱/GR 环形舱 + 机制竖井塔 + 汇合核；桥线宽编码分离度（registry 角度派生）">模块空间</button>
              <button type="button" className={'at-tab' + (panMode === 'tplspace' ? ' on' : '')} onClick={() => setPanMode('tplspace')}
                title="语言模板 3D 关系空间：模块聚簇 + 共享机制/同族/跨模块桥边；点击节点看测试详情">模板空间</button>
              <button type="button" className={'at-tab' + (panMode === 'cards' ? ' on' : '')} onClick={() => setPanMode('cards')}>卡片</button>
            </span>
          </div>
          {panMode === 'modules' && <AtlasModuleView sel={sel} onPickFeature={fid => setSel(cur => (cur === fid ? null : fid))}/>}
          {panMode === 'modspace' && <AtlasModuleSpace sel={sel} onPickFeature={fid => setSel(cur => (cur === fid ? null : fid))}/>}
          {panMode === 'tplspace' && <AtlasTemplateSpace onPickFeature={fid => setSel(cur => (cur === fid ? null : fid))}/>}
          {panMode === 'cards' && (
            <div className="at-grid">
              {data.features.map(f => (
                <div key={f.id} className={'at-card' + (sel === f.id ? ' sel' : '')}
                  onClick={() => setSel(sel === f.id ? null : f.id)}>
                  <div className="at-card-head">
                    <span className="mono at-fid">{f.id}</span>
                    <span className="at-fam">{FAM_NAME[f.family] || f.family}</span>
                    <span className={'at-badge ' + (ECLASS[f.evidence_level] || 'at-e1')}>{f.evidence_level}</span>
                  </div>
                  <div className="at-stmt">{f.statement}</div>
                  <ValueChips values={f.values} />
                  <div className="at-anchors-line">{f.n_anchors} 锚{f.counter_evidence.length ? ' · 有反证' : ''}</div>
                </div>
              ))}
            </div>
          )}
          {data && sel && data.features.find(f => f.id === sel) && (
            <FeatureDetail f={data.features.find(f => f.id === sel)} onGo={onGo} onClose={() => setSel(null)}/>
          )}

          {/* ===== 专项视图折叠区（v1.6 由独立 tab 降级）：10 个 /rdc-* 页三组链接，逐页协议化并入后删除独立路由 ===== */}
          <details className="at-ov-special">
            <summary>专项视图 · {SPECIAL_VIEWS.reduce((n, [, rows]) => n + rows.length, 0)} 个 /rdc-* 独立页（新窗口 · 统一壳 RdcPageShell）</summary>
            <div className="at-ov-special-grid">
              {SPECIAL_VIEWS.map(([group, rows]) => (
                <div key={group} className="at-ov-special-group">
                  <b>{group}</b>
                  {rows.map(([path, name, desc]) => (
                    <a key={path} href={path} target="_blank" rel="noopener noreferrer"
                      title={path + ' · ' + desc}>
                      <span className="mono">{name} ↗</span><span>{desc}</span>
                    </a>
                  ))}
                </div>
              ))}
            </div>
          </details>
        </div>
      )}

      {tab === 'ledger' && ledger && (
        <div className="at-ledger">
          <div className="at-summary">
            <span className="at-kv">measurements {ledger.total} 条</span>
            <span className="at-kv mono">ledger_sha256_8={ledger.ledger_sha256_8}（stale 自 Phase 8，见账本注记）</span>
          </div>
          <table className="at-table">
            <thead><tr><th>phase</th><th>线</th><th>name</th><th>证据级</th><th>prereg</th><th>日期</th><th>verdict</th></tr></thead>
            <tbody>
              {ledger.measurements.map(m => (
                <tr key={m.phase}>
                  <td className="mono">{m.phase}</td>
                  <td>{m.line}</td>
                  <td className="mono">{m.name}</td>
                  <td>{m.evidence_level || '—'}</td>
                  <td className="mono">{m.prereg_id || '—'}</td>
                  <td>{m.date}</td>
                  <td className="at-verdict">{(m.verdict || '').slice(0, 120)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'results' && <ResultsExplorer mode="results" />}

      {tab === 'registry' && <RegistryPanel/>}
    </div>
  );
}
