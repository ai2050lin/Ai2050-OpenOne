/* 空间透镜：两大顶层模式
   ── 数据分析：下载的数据（研究包/中心节点结果）驱动，四种视图
      ① 层平铺（默认）——36 层玻璃盒阵列，点击层盒看内部结构与全部参数
      ② 神经元级——单层内部 3D 全量展开（16,384 个可独立选中的点：residual/MLP/attn）
      ③ 特征点云——L6 特征族点云
      ④ 参数热图——单层权重分块统计（demo）
      语言模板选择器（M5-P0）：神经元级 / 特征点云两模式叙事由 LANG_TEMPLATES 驱动。
   ── 实时分析：连接本机 HF 模型目录（服务端 /api/live/* 读 config + safetensors 权重头，
      不加载权重、秒级完成）。3D 舞台与数据分析共用（middle 插槽：连接栏/统计 → 3D → 参数浏览器）；
      激活级实时 forward 流属 P2。离线回退 LIVE_MODEL_DEMO（协议结构 demo，显式标注）。
   零依赖 canvas 伪 3D；接入点在各模式组件内注释 */
import { useEffect, useState } from 'react';
import StackMode from './StackMode.jsx';
import NeuronMode from './NeuronMode.jsx';
import CloudMode from './CloudMode.jsx';
import ParamMode from './ParamMode.jsx';
import { LANG_TEMPLATES, LIVE_MODEL_DEMO, PROJECTIONS, TECH_CATEGORIES } from './distributedData.js';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

/* M7-P1 投影层（PROJECTIONS 注册表驱动）：特征点云按技术产物切换投影。
   available 且 render=cloud-cos → 当前视图；render=panel → 显示说明面板（含契约缺口与跳转）；
   planned → 置灰，点击显示缺口说明（与 TechPanel 同一套 input 契约谓词逻辑）。 */

const TOP_MODES = [
  { k: 'data', t: '数据分析', d: '下载的数据 · 研究包/结果驱动' },
  { k: 'live', t: '实时分析', d: '连接本地模型 · 真实权重数值' },
];

const MODES = [
  { k: 'stack', t: '层平铺', d: '36 层平放' },
  { k: 'neuron', t: '神经元级', d: '单神经元' },
  { k: 'cloud', t: '特征点云', d: 'L6 族' },
  { k: 'param', t: '参数热图', d: '单层 W' },
];

/* 参数量/字节格式化：1.04M / 388.96M / 4.02B */
const fmtP = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'K' : String(n);
const fmtB = (n) => n >= (1 << 30) ? (n / (1 << 30)).toFixed(2) + ' GB' : n >= (1 << 20) ? (n / (1 << 20)).toFixed(1) + ' MB' : (n / 1024).toFixed(1) + ' KB';
/* 权重标量数值显示（「全部参数」隐藏后暂无引用；恢复参数表数值列时取消注释）
const fmtW = (v) => v == null ? '—' : (v !== 0 && Math.abs(v) < 0.001 ? v.toExponential(2) : String(+v.toFixed(4)));
*/

/* 实时分析 · 连接本机模型（3D 舞台 + 真实权重数值；「全部参数」浏览器已隐藏）
   LIVE：POST /api/live/connect {model_path} → 服务端 _scan_live_model（权重头扫描）；
   离线/未连接：LIVE_MODEL_DEMO 协议结构快照（显式 DEMO 徽标与注记），渲染器同一套。
   middle 插槽 = 共享 3D 舞台（置顶）。 */
function LiveAnalysis({ conn, onConn, middle }) {    // conn 由 LensSpatial 持有（连接 UI 仅未连接时显示精简条；状态上提供右侧「模型总览」共用）
  /* 「全部参数」已隐藏（2026-10-09）：过滤/展开/权重懒加载状态随模块停用（恢复时取消注释）
  const [filter, setFilter] = useState('');
  const [open, setOpen] = useState({});        // prefix → bool；未登记时默认展开第一组
  const [wt, setWt] = useState({});            // name → 真实权重标量统计（展开组时懒加载 summary）
  */
  const [path, setPath] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

  const connect = async () => {
    if (!path.trim() || busy) return;
    setBusy(true); setErr('');
    try {
      const r = await fetch(`${API_BASE}/api/live/connect`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: path.trim() }),
      });
      if (!r.ok) {
        const t = await r.json().catch(() => ({}));
        throw new Error(t.detail || ('HTTP ' + r.status));
      }
      onConn && onConn(await r.json());
    } catch (e) {
      setErr(String(e.message || e).includes('Failed to fetch')
        ? '中心节点不可达（:5001）' : String(e.message || e));
    } finally { setBusy(false); }
  };

  /* 「全部参数」逻辑随模块隐藏（2026-10-09）：派生分组/展开状态/懒加载 fetch 一并停用（恢复时取消注释）
  const data = conn || LIVE_MODEL_DEMO;
  const f = filter.trim().toLowerCase();
  const groups = f
    ? (data.groups || [])
        .map(g => ({ ...g, tensors: g.tensors.filter(t => t.name.toLowerCase().includes(f)) }))
        .filter(g => g.tensors.length || g.prefix.toLowerCase().includes(f))
        .map(g => (g.tensors.length ? g : { ...g, tensors: g.tensors }))
    : (data.groups || []);
  const isOpen = (g, i) => (f ? true : open[g.prefix] !== undefined ? open[g.prefix] : i === 0);
  const toggle = (p) => setOpen(o => ({ ...o, [p]: !(o[p] !== undefined ? o[p] : false) }));

  // 展开组时懒加载真实权重标量（POST /live/weights/summary：32×64 行块切片，超限均匀抽样）
  useEffect(() => {
    if (!conn) return;
    const names = [];
    (conn.groups || []).forEach((g, i) => {
      if (isOpen(g, i)) (g.tensors || []).forEach(t => {
        if (t.params >= 4096 && !t.name.endsWith('bias') && !wt[t.name] && names.length < 16) names.push(t.name);
      });
    });
    if (!names.length) return;
    let stop = false;
    fetch(`${API_BASE}/api/live/weights/summary`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ names }),
    }).then(r => r.ok ? r.json() : null).then(d => {
      if (d && d.results && !stop) setWt(w => ({ ...w, ...d.results }));
    }).catch(() => {});
    return () => { stop = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conn, open, filter]);
  */

  /* ===== 「全部参数」浏览器（已隐藏 2026-10-09）：分组（层）→ 逐 tensor 行 + 真实权重数值列
     恢复方式：取消本段注释得 const tbl，并在下方 return 中 {middle} 之后引用 {tbl}
  const tbl = (
    <div className="fw-pl-card">
      <h6>
        全部参数
        <span>{(data.groups || []).reduce((a, g) => a + g.tensor_count, 0)} tensors · 按层分组 · 悬停看完整形状</span>
      </h6>
      <input className="fw-live-filter" placeholder="过滤 tensor 名（如 q_proj / layers.5 / norm）"
             value={filter} onChange={e => setFilter(e.target.value)}/>
      <div className="fw-live-groups">
        {groups.map((g, i) => (
          <div key={g.prefix} className="fw-live-group">
            <button type="button" className="fw-live-ghd" onClick={() => toggle(g.prefix)}>
              <span className="fw-tr">{isOpen(g, i) ? '▾' : '▸'}</span>
              <b className="fw-mono">{g.prefix}</b>
              <span className="dim2">{g.tensor_count} tensors</span>
              <span className="fw-mono dim2">{fmtP(g.params)}</span>
            </button>
            {isOpen(g, i) && (
              <div className="fw-live-table">
                {g.tensors.map(t => {
                  const w = wt[t.name];
                  return (
                    <div key={t.name} className="fw-live-trow" title={t.name + ' · shape [' + t.shape.join(', ') + '] · ' + t.dtype + ' · ' + t.params + ' params'}>
                      <span className="fw-mono fw-live-tname">{t.name}</span>
                      <span className="fw-mono dim2">[{t.shape.join('×')}]</span>
                      <span className="fw-live-tdtype">{t.dtype}</span>
                      <span className="fw-mono r">{fmtP(t.params)}</span>
                      {conn && (
                        <span className={'fw-mono fw-live-tval' + (w ? '' : ' pend')}
                              title={w ? ('rms ' + w.rms + ' · mean|W| ' + w.mean_abs + ' · max|W| ' + w.max_abs + ' · ' + (w.sampled ? '32×64 行块抽样' : '全量遍历'))
                                      : (t.params >= 4096 && !t.name.endsWith('bias') ? '真实权重统计加载中…' : '')}>
                          {w ? fmtW(w.rms) : (t.params >= 4096 && !t.name.endsWith('bias') ? '…' : '')}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
        {!groups.length && <div className="dim2" style={{ padding: '6px 2px', fontSize: 10 }}>无匹配 tensor</div>}
      </div>
      <p className="fw-tut-note">实时范围：参数结构实时读取（config + 权重头扫描）+ 数值列/热图/3D 着色的<b>真实权重懒加载</b>
        （safetensors 按行块切片读，rms/mean|W| 为 32×64 行块统计，超限均匀抽样；模型文件不变即零重复成本）；
        激活级实时流（真实 forward、逐层 norm 曲线）需 runner 侧加载权重，属下一期（P2）。</p>
    </div>
  );
  ===== */

  return (
    <div className="fw-live">
      {/* 未连接时的精简连接条（连接成功后整条隐藏，界面回到 3D 舞台 + 右侧模型总览） */}
      {!conn && (
        <div className="fw-pl-card fw-live-conn">
          <h6>连接本地模型 <span>HF 模型目录（config.json + *.safetensors）· 权重头 + 真实数值懒加载，不整体加载权重、秒级</span></h6>
          <div className="fw-live-bar">
            <input className="fw-live-path" placeholder="例：D:\models\Qwen3-4B 或 /home/user/models/qwen3-4b"
                   value={path} onChange={e => setPath(e.target.value)}
                   onKeyDown={e => { if (e.key === 'Enter') connect(); }}/>
            <button type="button" className="fw-tbtn" disabled={busy || !path.trim()} onClick={connect}>{busy ? '连接中…' : '连接'}</button>
          </div>
          {err && <div className="fw-ai-err" style={{ marginTop: 6 }}>{err}</div>}
        </div>
      )}

      {/* 3D 舞台（与数据分析共用）：置顶展示；模型名/参数/数据类型已整合进右侧「模型总览」面板 */}
      {middle}

      {/* 「全部参数」浏览器已隐藏（2026-10-09）：恢复=取消上方 tbl 注释并在此处引用 {tbl} */}
    </div>
  );
}

export default function LensSpatial({ on, onGo }) {
  const [top, setTop] = useState('data');
  const [mode, setMode] = useState('stack');
  const [neuronLayer, setNeuronLayer] = useState(6);
  const [paramLayer, setParamLayer] = useState(6);
  const [tplId, setTplId] = useState(LANG_TEMPLATES[0].id);
  const [projId, setProjId] = useState(PROJECTIONS[0].id);   // M7-P1 投影层选择
  const [conn, setConn] = useState(null);                    // live 连接（原 LiveAnalysis 内部状态上提，供右侧模型总览共用）
  const [scan, setScan] = useState(null);                    // 各层真实权重统计（S9b /live/weights/scan，3D 层盒数值着色）
  const [scanState, setScanState] = useState('idle');        // idle | loading | ready | unavailable
  const tpl = LANG_TEMPLATES.find(t => t.id === tplId) || LANG_TEMPLATES[0];
  const catOf = id => TECH_CATEGORIES.find(c => c.id === id) || {};
  const proj = PROJECTIONS.find(p => p.id === projId) || PROJECTIONS[0];

  useEffect(() => {                            // 挂载时探测服务端已有连接（上次连接可恢复）
    fetch(`${API_BASE}/api/live/status`).then(r => r.ok ? r.json() : null).catch(() => null)
      .then(d => { if (d && d.connected) setConn(d); });
  }, []);

  useEffect(() => {                            // 连接建立后拉各层真实权重统计（首次慢：逐层懒加载，服务端缓存）
    if (!conn) { setScan(null); setScanState('idle'); return; }
    setScanState('loading');
    let stop = false;
    fetch(`${API_BASE}/api/live/weights/scan`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (!stop) { setScan(d); setScanState(d && d.layers && d.layers.length ? 'ready' : 'unavailable'); } })
      .catch(() => { if (!stop) setScanState('unavailable'); });
    return () => { stop = true; };
  }, [conn]);

  /* 实时模型信息包 → StackMode 右侧「模型总览」整合（模型名/参数量/config/数据类型/权重文件） */
  const liveData = top === 'live' ? (conn || LIVE_MODEL_DEMO) : null;
  const liveCfg = liveData ? (liveData.config || {}) : {};
  const liveBytes = liveData ? (liveData.files || []).reduce((a, f) => a + (f.bytes || 0), 0) : 0;
  const liveInfo = liveData && {
    head: (liveData.arch || '—') + ' · ' + fmtP(liveData.total_params || 0),
    cfg: [
      ['layers', liveCfg.layers ?? '—'],
      ['d_model', liveCfg.hidden ?? '—'],
      ['attention', (liveCfg.heads ?? '—') + ' Q / ' + (liveCfg.kv_heads ?? '—') + ' KV'],
      ['vocab', liveCfg.vocab != null ? liveCfg.vocab.toLocaleString() : '—'],
      ['tie_emb', liveCfg.tie_word_embeddings ? 'true' : 'false'],
      ['safetensors', (liveData.files || []).length + ' 个 · ' + fmtB(liveBytes)],
    ],
    dtypes: Object.entries(liveData.dtypes || {}).map(([k, v]) => [k, fmtP(v) + ' tensors']),
    meta: liveData.model_path || '',
  };

  /* 3D 舞台（数据分析/实时分析共用）：模式工具条 + 投影层 + 视口。live 模式以 middle 插入连接栏与参数表之间 */
  const stage = (
    <>
      <div className="fw-sp-toolbar">
        <div className="fw-mode-seg">
          {MODES.map(m => (
            <button key={m.k} className={mode === m.k ? 'on' : ''} onClick={() => setMode(m.k)} title={m.d}>
              {m.t}<small>{m.d}</small>
            </button>
          ))}
        </div>
        {(mode === 'neuron' || mode === 'cloud') && (
          <div className="fw-tpl-sel" title="语言模板（LANG_TEMPLATES 注册表）：切换后神经元级/点云两模式叙事整体跟随，渲染器不变">
            {LANG_TEMPLATES.map(t => (
              <button key={t.id} type="button"
                      className={'fw-tech-chip' + (tplId === t.id ? ' on' : '') + (t.status === 'demo' ? ' demo' : '')}
                      title={t.desc + ' · 句式 ' + t.pattern}
                      onClick={() => setTplId(t.id)}>
                {t.name}{t.status === 'demo' && <small>·DEMO</small>}
              </button>
            ))}
            {onGo && (
              <button type="button" className="fw-tech-chip" style={{ borderStyle: 'dashed' }}
                      title="下载该语言模板 × 分析技术的研究包（corpus+runner 口径+结果清单+README）——在平台进度·研究包组合选择，或终端 node_agent.py download"
                      onClick={() => onGo('progress')}>
                研究包 ↓<small>下载</small>
              </button>
            )}
          </div>
        )}
        {mode === 'cloud' && (
          <>
            <label className="fw-chk"><input type="checkbox" defaultChecked/> 特征点云</label>
            <label className="fw-chk"><input type="checkbox" defaultChecked/> top-k 邻居连线</label>
            <label className="fw-chk"><input type="checkbox"/> 残差流骨架</label>
            <label className="fw-chk"><input type="checkbox"/> 14B/9B 叠加</label>
          </>
        )}
        {mode === 'stack' && (
          <span style={{ color: 'var(--fw-text-3)' }}>全部 36 层平放（L0 → L35）· 玻璃盒=层结构 · <b>左键=旋转（上下左右） · 右键=平移（上下左右）</b> · 滚轮缩放 · 点击层盒 → 右侧全部参数 → 可下钻神经元级</span>
        )}
        {mode === 'neuron' && (
          <span style={{ color: 'var(--fw-text-3)' }}><b>左键=旋转（上下左右） · 右键=平移（上下左右）</b> · 滚轮缩放 · L{neuronLayer} 层内部 16,384 点独立选中 · 底部时间轴播放 token 步（激活/变化/写入）· 选中单元看 t 轴曲线 + ℓ 轴跨层基座</span>
        )}
        {mode === 'cloud' && (
          <span style={{ color: 'var(--fw-text-3)' }}><b>左键=旋转（上下左右） · 右键=平移（上下左右）</b> · 滚轮缩放</span>
        )}
        {mode === 'param' && (
          <span style={{ color: 'var(--fw-text-3)' }}>当前层 L{paramLayer} · 在层栈/神经元模式选中后可切换</span>
        )}
        <span style={{ marginLeft: 'auto', fontFamily: 'var(--fw-mono)', color: 'var(--fw-text-3)' }}>
          {mode === 'stack' ? '36L · 公开架构数字' : 'd_model 2560 · d_ff 9728 · 32h×128'} · 数据源={top === 'live' ? (scan ? '本地模型权重数值' : '本地模型权重头') : '研究包/结果'}
        </span>
      </div>
      {mode === 'cloud' && (
        <div className="fw-proj-bar">
          <span className="fw-proj-lb">投影层 · 技术产物驱动</span>
          {PROJECTIONS.map(p => {
            const c = catOf(p.cat);
            return (
              <button key={p.id} type="button"
                      className={'fw-proj-chip' + (projId === p.id ? ' on' : '') + (p.status !== 'available' ? ' na' : '')}
                      style={{ '--cat': c.color }}
                      title={(c.name || p.cat) + ' · 契约 ' + p.input + ' · ' + (p.status === 'available' ? '可用' : '装置待建') + ' — ' + p.note}
                      onClick={() => setProjId(p.id)}>
                <i style={{ background: c.color }}/><span>{p.name}</span>{p.status !== 'available' && <small>待建</small>}
              </button>
            );
          })}
        </div>
      )}
      {mode === 'cloud' && proj.render === 'panel' && (
        <div className="fw-proj-panel">
          <b style={{ color: catOf(proj.cat).color }}>{proj.name}</b>
          <span>{proj.note}</span>
          <em>输入契约：<code>{proj.input}</code> · {proj.status === 'available' ? '当前结果库可支撑（3D 布局待 P2）' : '结果库暂无满足契约的数据——在研发透镜领取对应缺口任务'}</em>
          {onGo && <button type="button" className="fw-tbtn" onClick={() => onGo(proj.status === 'available' ? 'data' : 'process')}>
            {proj.status === 'available' ? '到数据透镜看 RDM 热图 →' : '去研发透镜领缺口任务 →'}
          </button>}
        </div>
      )}
      <div className="fw-sp-wrap">
        {mode === 'stack' && <StackMode live={top === 'live' ? liveInfo : null} scan={top === 'live' && conn ? scan : null} scanState={top === 'live' && conn ? scanState : null} onOpenParam={l => { setParamLayer(l); setMode('param'); }} onOpenNeuron={l => { setNeuronLayer(l); setMode('neuron'); }}/>}
        {mode === 'neuron' && <NeuronMode layer={neuronLayer} onLayer={setNeuronLayer} onOpenParam={l => { setParamLayer(l); setMode('param'); }} onBack={() => setMode('stack')} tpl={tpl}/>}
        {mode === 'cloud' && <CloudMode tpl={tpl}/>}
        {mode === 'param' && <ParamMode layer={paramLayer} live={top === 'live' && conn ? conn : null} scan={scan}/>}
      </div>
    </>
  );

  return (
    <section className={'fw-view fw-spatial' + (on ? ' on' : '')}>
      {/* 顶层模式：数据分析 / 实时分析 */}
      <div className="fw-tech-seg" style={{ margin: '14px 0 10px 8px' }}>
        {TOP_MODES.map(m => (
          <button key={m.k} type="button" className={top === m.k ? 'on' : ''} title={m.d} onClick={() => setTop(m.k)}>
            {m.t}<small>{m.d}</small>
          </button>
        ))}
      </div>

      {top === 'live' && <LiveAnalysis conn={conn} onConn={setConn} middle={stage}/>}

      {top === 'data' && stage}
    </section>
  );
}
