/* 关联机制 · 深入部分：HiddenState 流（3D）、算法分析页签、反常句生成文法草稿。
   数据只来自主线已有产物（MA-H2 报告、MA-H0–H3 记录）；没有测过的写“未测”，不放示例数值。 */
import { useEffect, useState } from 'react';
import { panScene, useScenePointer } from '../scenePointer.js';

import report from '../../../../research/mainline/H2/report.json';
import { LineChart } from './H2Results.jsx';
import { BENCH } from './data/mainline.js';
import './client_guide.css';

const FACTORS = [['scope', '范围', '#2a78d6'], ['draft', '草稿', '#eb6834'], ['form', '记录形式', '#1baf7a'], ['pos', '目标位置', '#c98a00'], ['k', '记录数', '#7050a0']];
const POS = [['qname', '问题中的目标名'], ['ans', '答案位置'], ['tcol', '记录中的颜色'], ['tname', '记录中的目标名'], ['lastcol', '最后一个颜色']];
const L = 37;

/* 第 0 层就能分开（AUC ≥ 0.9）的位置 × 因素是词身份混淆，不画 */
const confounded = (f, p) => (report.effects[f][p].auc[0] ?? 0) >= 0.9;

function project(v, x, y, z, W, H, s) {
  const cy = Math.cos(v.yaw), sy = Math.sin(v.yaw), cp = Math.cos(v.pitch), sp = Math.sin(v.pitch);
  const X = x * cy + z * sy, Z = -x * sy + z * cy, Y = y * cp - Z * sp, D = y * sp + Z * cp, f = 3.4 / (3.4 + D);
  return { x: W / 2 + (v.panX ?? 0) + X * s * f, y: H / 2 + (v.panY ?? 0) - Y * s * f, d: D, f };
}

export function FlowRibbons() {
  const [pos, setPos] = useState('qname');
  const [layer, setLayer] = useState(36);
  const [view, setView] = useState({ yaw: -0.5, pitch: 0.18 });
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return undefined;
    const t = setInterval(() => setLayer(l => { if (l >= L - 1) { setPlaying(false); return l; } return l + 1; }), 200);
    return () => clearInterval(t);
  }, [playing]);
  const W = 640, H = 470, S = 172;
  const shown = FACTORS.filter(([f]) => !confounded(f, pos));
  const max = Math.max(1e-6, ...shown.flatMap(([f]) => report.effects[f][pos].std_rms));
  const lanes = shown.map((_, i) => -0.8 + (1.6 * i) / Math.max(1, shown.length - 1));
  const Y = l => -1 + (2 * l) / (L - 1);
  const segs = [];
  shown.forEach(([f, name, color], i) => {
    const vals = report.effects[f][pos].std_rms;
    const pts = Array.from({ length: layer + 1 }, (_, l) => project(view, lanes[i], Y(l), 0, W, H, S));
    for (let l = 1; l < pts.length; l++) segs.push({ d: pts[l].d, el: <line key={`${f}${l}`} x1={pts[l - 1].x} y1={pts[l - 1].y} x2={pts[l].x} y2={pts[l].y}
      stroke={color} strokeWidth={1.2 + 9 * vals[l] / max} strokeLinecap="round" strokeOpacity="0.85"><title>{`${name} · 第 ${l} 层 · 标准化效应 ${vals[l].toFixed(3)}`}</title></line> });
    const p = pts[pts.length - 1];
    segs.push({ d: p.d - 9, el: <text key={`${f}t`} x={p.x} y={p.y - 10 - (i % 2) * 13} fontSize="12" fontWeight="650" fill="#16202c" textAnchor="middle" paintOrder="stroke" stroke="#fff" strokeWidth="3">{name}</text> });
  });
  const plates = [0, 6, 12, 18, 24, 30, 36].map(l => {
    const c = [[-1.05, -0.32], [1.05, -0.32], [1.05, 0.32], [-1.05, 0.32]].map(([x, z]) => project(view, x, Y(l), z, W, H, S));
    return { d: project(view, 0, Y(l), 0, W, H, S).d + 0.5, el: <polygon key={`p${l}`} points={c.map(q => `${q.x},${q.y}`).join(' ')} fill={l <= layer ? '#eef3fa' : '#fff'} fillOpacity="0.6" stroke="#cbd5e2" strokeWidth="0.7"/> };
  });
  const tag = l => { const p = project(view, -1.12, Y(l), 0.32, W, H, S); return <text key={`l${l}`} x={p.x} y={p.y} fontSize="10" fill="#566373" textAnchor="end">{l === 0 ? '词嵌入' : `第 ${l} 层`}</text>; };
  const rotate = (a, b) => setView(v => ({ ...v, yaw: v.yaw + a, pitch: Math.max(-0.2, Math.min(1, v.pitch + b)) }));
  const pointer = useScenePointer({
    onPan: (dx, dy, svg) => {
      const scale = svg.getScreenCTM()?.a || 1;
      setView(v => panScene(v, dx / scale, dy / scale));
    },
    onRotate: (dx, dy) => rotate(dx * 0.008, dy * 0.006),
  });
  return <section className="v8-card cg-section" aria-label="HiddenState 流">
    <h3>HiddenState 流：各因素在每一层能区分多少 <span className="mp-tag ok">真实数据 · MA-H2</span></h3>
    <p className="v8-note">线宽 = 该因素两个水平在这一层的标准化差异（各坐标差值除以组内标准差的均方根），读出位置可切换。宽说明读得出，不说明被使用；线的位置和走向是布局，不代表模型几何。</p>
    <div className="v6-toolbar" role="group" aria-label="读出位置">{POS.map(([id, label]) =>
      <button type="button" className="fw-tbtn" key={id} aria-pressed={pos === id} onClick={() => setPos(id)}>{label}</button>)}</div>
    <div className="v6-toolbar">
      <button type="button" className="fw-tbtn" onClick={() => { if (layer >= L - 1) setLayer(0); setPlaying(!playing); }}>{playing ? '暂停' : '逐层播放'}</button>
      <label className="v8-note">层 {layer} <input type="range" min="0" max={L - 1} value={layer} onChange={e => { setPlaying(false); setLayer(Number(e.target.value)); }} aria-label="选择层"/></label>
    </div>
    <div className="lg3-grid">
      <svg className="lg3-svg" viewBox={`0 0 ${W} ${H}`} tabIndex={0} role="img" aria-label="各因素逐层标准化差异的三维视图，左键拖动平移，右键拖动或方向键旋转"
        {...pointer}
          onKeyDown={e => { const m = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] }[e.key]; if (m) { e.preventDefault(); rotate(...m); } }}>
        {[...plates, ...segs].sort((a, b) => b.d - a.d).map(s => s.el)}
        {[0, 12, 24, 36].map(tag)}
      </svg>
      <aside className="lg3-side">
        <h4>第 {layer} 层 · {POS.find(p => p[0] === pos)[1]}</h4>
        {FACTORS.map(([f, name, color]) => {
          const off = confounded(f, pos), v = report.effects[f][pos].std_rms[layer], auc = report.effects[f][pos].auc[layer];
          return <div className="h2-bar" key={f}><span>{name}</span>
            {off ? <small className="v8-note">词身份混淆，不画</small> : <span className="mp-bar"><i style={{ width: `${Math.min(100, 100 * v / max)}%`, background: color }}/></span>}
            <small className="v8-note">{off ? '' : `${v.toFixed(3)}${auc != null ? ` · AUC ${auc.toFixed(2)}` : ''}`}</small></div>;
        })}
        <p className="v8-note">AUC 是实体留出的读出准确度，只在部分层计算。资料记录位置上范围、草稿都在 0.5 附近：它们出现在记录之后，按因果读不到，属预期的负对照。</p>
      </aside>
    </div>
  </section>;
}

const ALG = [
  ['逐层读出', '哪层已经偏向哪个词', '读出近似', '已用：MA-H2 逐层 K − P'],
  ['因素读出', '风格、范围等是否读得出', '可读', '已用：MA-H2 实体留出 AUC'],
  ['直接作用（dla）', '哪个头、MLP 直接写入候选差', '归因近似', 'MA-H3 进行中（同题比较）'],
  ['路径归因（atp、atp_ig）', '哪些组件对候选差贡献大', '归因近似', 'MA-H3 进行中；MB-v1 中 atp_ig 最稳'],
  ['精确修补（exact）', '换掉一个组件后输出是否改变', '因果（单组件）', 'MA-H3 进行中'],
  ['两因素对照', '两个因素是否互相改变作用', '交互', '答案位置交互比来自 MA-H2；正式交互检验在 H5'],
  ['字典特征图', '哪些特征连到输出', '替代模型', '只有第 30 块有字典，未纳入'],
  ['文字解释', '这一状态能否用文字描述', '未检验假设', '未接入'],
];
const TABS = [['pair', '两因素交互'], ['curve', '强度曲线'], ['card', '因素卡'], ['life', '生命周期'], ['gen', '生成步'], ['bench', '方法基准']];
const PAIR_COLORS = { 'scope×draft': '#2a78d6', 'scope×form': '#eb6834', 'scope×pos': '#1baf7a', 'form×pos': '#c98a00' };
const CARDS = {
  scope: { name: '范围（资料 / 现实）', obs: '中文 discovery：第 24–33 层两种范围同时分化；第 21–29 层若干头在资料范围读资料色约 0.6，现实范围 ≤ 0.08（MA-H2）。',
    behavior: '加“通常”时现实范围准确率 0.89；只在问题里写“现实世界”时 0.14，答案前缀里再写一次时 0.30（MA-H3 问法检查）。', use: '未测（H4 干预）', bad: '预注册的 R1 / R2 判据对噪声敏感，不可用（MA-H2）。' },
  draft: { name: '错误草稿', obs: '从第 2 层起在答案位置和问题位置读得出（MA-H2）。', behavior: '草稿被照抄的增量在各版中英文为 0.005–0.054，未达 0.1；英文把现实范围拉回资料色 0.04–0.09（MA-H0 v3–v5）。', use: '未测', bad: 'D1 与 E1 两项预注册特征都未支持。' },
  form: { name: '记录形式（单 / 双实体）', obs: '在记录位置第 2–24 层可读（MA-H2）。', behavior: '双实体代价约 0.02，未达 0.1；资料范围的少量绑定错误全部是选了同一记录另一个实体的颜色（MA-H0-v5）。', use: '未测', bad: '“现实范围 + 双实体 + 末条 + k=3”稳定低于 0.8，已排除。' },
  pos: { name: '目标记录位置（首 / 末）', obs: '在记录位置第 2 层起可读（MA-H2）。', behavior: '近因特征约 0.02，未达 0.1：首条与末条的绑定错误差不多（MA-H0-v5）。', use: '未测', bad: '资料范围接近满分，行为层面无法区分“按名检索”与“近因启发”。' },
};

export function AlgorithmPanel() {
  const [tab, setTab] = useState('pair');
  const [card, setCard] = useState('scope');
  const c = CARDS[card];
  return <section className="v8-card cg-section" aria-label="算法分析">
    <h3>算法分析</h3>
    <p className="v8-note">同一个决策点，不同算法回答不同的问题，结果性质也不同。下表是主线现在实际用到的方法。</p>
    <div className="v8-weight-scroll"><table className="v6-table"><thead><tr><th>算法</th><th>回答的问题</th><th>结果性质</th><th>主线现状</th></tr></thead>
      <tbody>{ALG.map(r => <tr key={r[0]}>{r.map((x, i) => <td key={i}>{x}</td>)}</tr>)}</tbody></table></div>
    <div className="wb-space-tabs h2-tabs" role="tablist">{TABS.map(([id, label]) =>
      <button type="button" role="tab" key={id} aria-pressed={tab === id} aria-selected={tab === id} onClick={() => setTab(id)}>{label}</button>)}</div>
    {tab === 'pair' && <div>
      <LineChart label="答案位置两因素交互比" yLabel="纵轴：交互向量范数 ÷ 两个主效应范数的平均（MA-H2，答案位置）。"
        series={Object.entries(report.interaction_ratio_ans).map(([k, v]) => ({ name: k.replace('scope', '范围').replace('draft', '草稿').replace('form', '形式').replace('pos', '位置'), color: PAIR_COLORS[k], values: v }))}/>
      <p className="v8-note">接近 0 表示两个因素的影响近似可以相加。偏离 0 只说明非加性，要靠干预区分路由、门控和补偿。
        注意：MA-H2 的答案位置两种范围末 token 不同，范围的主效应含词身份，这里的“范围 ×”比值偏小；“形式 × 位置”≥ 1 是构造关系（两种形式下“位置”的含义不同）。</p>
    </div>}
    {tab === 'curve' && <p className="v8-note"><span className="mp-tag">未测</span> 强度曲线要逐步增强一条线索（例如重复资料陈述、改变“通常”的强度），看选资料色的比例怎样变化。主线还没有做，放在 H5 之后。MA-H3 的问法检查给出了一个三点粗看：现实范围准确率 0.14 → 0.30 → 0.89，但三版改动不是单一线索的剂量，不能当曲线。</p>}
    {tab === 'card' && <div>
      <div className="v6-toolbar" role="group" aria-label="因素卡">{Object.entries(CARDS).map(([id, x]) =>
        <button type="button" className="fw-tbtn" key={id} aria-pressed={card === id} onClick={() => setCard(id)}>{x.name}</button>)}</div>
      <div className="cg-grid2">
        <div><h4>内部观察</h4><p className="v8-note">{c.obs}</p></div>
        <div><h4>行为结果</h4><p className="v8-note">{c.behavior}</p></div>
        <div><h4>使用证据</h4><p className="v8-note">{c.use}</p></div>
        <div><h4>反例与限制</h4><p className="v8-note">{c.bad}</p></div>
      </div>
      <p className="v8-note">卡片内容取自 Phase 321–324 的研究记录，都是候选观察或行为校准，没有一项登记为正式证据。</p>
    </div>}
    {tab === 'life' && <div className="v8-weight-scroll"><table className="v6-table">
      <thead><tr><th>因素</th><th>种子（第 0 层）</th><th>汇集</th><th>载体</th><th>接续</th><th>读出（首个可读出层）</th></tr></thead>
      <tbody>{FACTORS.map(([f, name]) => {
        const best = POS.map(([p, label]) => [label, report.effects[f][p].first_readable, confounded(f, p)]).filter(x => x[1] != null && !x[2]).sort((a, b) => a[1] - b[1])[0];
        const seed = POS.some(([p]) => confounded(f, p));
        return <tr key={f}><td>{name}</td><td>{seed ? '部分位置第 0 层已可分（词身份）' : '第 0 层不可分'}</td>
          <td>未测</td><td>H3 进行中</td><td>未测</td><td>{best ? `${best[0]}，第 ${best[1]} 层` : '—'}</td></tr>;
      })}</tbody></table>
      <p className="v8-note">只有“读出”一栏有数据（MA-H2）；汇集、接续要靠 H4 的路径干预，载体由 H3 冻结后填入。</p></div>}
    {tab === 'gen' && <p className="v8-note"><span className="mp-tag">未测</span> 生成步分三种口径：原始前缀下的下一步、公共续写路径、各自生成历史。对应主线 H6，尚未开始。</p>}
    {tab === 'bench' && <div>
      <p className="v8-note">MB-v1（Phase 315）lockbox 上 cf 节点的 auc_f，越高说明越少节点就能恢复行为。完整表格在“响应对照”。</p>
      <div className="v8-weight-scroll"><table className="v6-table"><thead><tr>{BENCH.mb.head.map(h => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>{BENCH.mb.rows.map(r => <tr key={r[0]}>{r.map((x, i) => <td key={i}>{x}</td>)}</tr>)}</tbody></table></div>
    </div>}
  </section>;
}

/* 反常句 → 文法草稿：只在本机生成 template_grammar.v1 的草稿文件，不提交、不冻结。 */
export function AnomalyDraft() {
  const [text, setText] = useState('');
  const [note, setNote] = useState('');
  const [done, setDone] = useState('');
  const make = () => {
    const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    const draft = {
      schema_version: 'template_grammar.v1', family_id: `DRAFT-${stamp}`, grammar_version: 0, status: 'draft',
      revision_note: '由客户端“这句反常”生成的草稿；需补齐因子、填充词、读出位置、行为门与预注册后，经模板流水线预览和冻结才能运行',
      name: '待命名', description: note || '（写下你觉得哪里反常、预期模型应该怎样）', source_sentence: text,
      factors: {}, frames: [{ id: 'f1', text, levels: {}, target: '', probe: { prompt: '', answers: [], contrast: [] } }],
      fillers: [], readouts: ['final'], layers: 'all', splits: { discovery: 0.4, confirm: 0.3, lockbox: 0.3 },
      prereg: { min_behavior_acc: 0.8, min_cell_n: 20 },
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(draft, null, 1)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url; a.download = `${draft.family_id}.json`; a.click();
    URL.revokeObjectURL(url);
    setDone(`已生成 ${draft.family_id}.json`);
  };
  return <section className="v8-card cg-section" aria-label="这句反常">
    <h3>这句反常？生成文法草稿</h3>
    <p className="v8-note">把你觉得模型表现反常的句子写下来，生成一份模板草稿（template_grammar.v1）。草稿只保存在你的电脑上，要补齐因子、填充词和行为门，经模板流水线预览、冻结后才能运行。</p>
    <label className="v8-note">句子<textarea className="h2-text" rows={2} value={text} onChange={e => setText(e.target.value)} placeholder="例如：资料里说香蕉是蓝色的。问：在现实世界里，香蕉是什么颜色的？答：在现实世界里，香蕉是"/></label>
    <label className="v8-note">哪里反常<textarea className="h2-text" rows={2} value={note} onChange={e => setNote(e.target.value)} placeholder="例如：问的是现实世界，模型却答了资料里的颜色"/></label>
    <div className="v6-toolbar"><button type="button" className="fw-tbtn primary" disabled={!text.trim()} onClick={make}>生成并下载草稿</button>
      <span className="v8-note" role="status">{done}</span></div>
  </section>;
}
