import { useMainline } from './data/mainlineState.js';
/* 研究工作台 · 三部分研究框架的共用内容。
   语言图谱、LLM 内部图谱、两图谱关联贯穿各页；这里放页面共用的白话摘要、归因图和保真看板。
   约束：研究数字来自接口或登记过的记录；归因图是未预注册的界面演示，不计入研究证据；没有跑过的实验写“未测”。 */
import { useEffect, useMemo, useState } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import catalog from '../../../../shared/dist_protocol/lang_catalog.json';
import { shortModel } from './v8Util.js';

import './client_guide.css';

export const DEMO_SNAPSHOT = '6be1034f0b55a6df';
export const EVIDENCE = {
  observation: ['obs', '看到过'], heldout: ['held', '新材料也成立'], intervention: ['int', '改动后输出变了'],
  negative: ['neg', '未成立'], parameter_source: ['read', '权重可算'],
};
export function Ev({ kind, children }) {
  const [tone, label] = EVIDENCE[kind] || [kind || 'gap', ''];
  return <span className={`cg-ev ${tone}`}>{children || label || '已登记'}</span>;
}

/* 语言目录的三类入口和已测数量。分类服务浏览，不代表模型内部结构。 */
export function languageSections(measured) {
  const domains = new Map(catalog.domains.map(domain => [domain.id, domain]));
  return catalog.sections.map(section => {
    const items = section.domains.flatMap(id => domains.get(id)?.groups.flatMap(group => group.items) ?? []);
    return { id: section.id, name: section.name, total: items.length, done: items.filter(item => measured.has(item.measurement)).length };
  });
}

export function relationCounts(objects, snapshot) {
  const scoped = (objects || []).filter(item => !snapshot || item.model_snapshot === snapshot);
  const features = new Set(scoped.filter(item => item.type === 'feature').map(item => item.id));
  const counts = {};
  for (const item of scoped) {
    if (item.type !== 'edge') continue;
    const { source, target, evidence } = item.body;
    const linked = (source?.startsWith('language:') && features.has(target)) || (target?.startsWith('language:') && features.has(source));
    if (linked) counts[evidence?.level || 'other'] = (counts[evidence?.level || 'other'] || 0) + 1;
  }
  return counts;
}

/* 最近的发现读研究登记表（data/mainline.js 的 FINDINGS，每个 Phase 追加），按 Phase 倒序，首屏 4 条，其余折叠。 */
export function LatestFindings({ onGo, bare = false }) {
  const { FINDINGS } = useMainline();
  const list = [...FINDINGS].sort((a, b) => b.phase - a.phase);
  const card = f => <div key={f.title}><Ev kind={f.kind}/> <small className="v8-note">Phase {f.phase}</small><h4>{f.title}</h4><p className="v8-note">{f.text}</p></div>;
  return <section className={bare ? "cg-section" : "v8-card cg-section"} aria-label="最近的发现">{!bare && <h3>最近的发现</h3>}
    <div className="cg-grid2">{list.slice(0, 4).map(card)}</div>
    {list.length > 4 && <details className="mp-fold"><summary>更早的发现（{list.length - 4}）</summary><div className="mp-fold-body cg-grid2">{list.slice(4).map(card)}</div></details>}
    <div className="v6-toolbar"><button type="button" className="fw-tbtn" onClick={() => onGo('route', null, null, { rtab: 'rules' })}>查看全部成果与证据</button></div>
  </section>;
}

export function NextStep({ onGo }) {
  return <section className="v8-card cg-next"><div><h3>下一步可以做什么</h3>
    <p className="v8-note">不用安装也能先看懂一个发现；有 NVIDIA 显卡可以贡献算力。</p></div>
    <button type="button" className="fw-tbtn primary" onClick={() => onGo('process', 'plan')}>参与研究</button></section>;
}

export function ParticipationWays({ onGo }) {
  return <div className="cg-grid3 cg-ways">
    <section className="cg-way"><h3>看懂一个发现</h3><small>无需安装 · 约 3 分钟</small>
      <p>从一条已确认规律出发，看到问题、结果和原始来源。</p>
      <button type="button" className="fw-tbtn" onClick={() => onGo('route', null, null, { rtab: 'rules' })}>开始浏览</button></section>
    <section className="cg-way"><h3>帮神经元写说明</h3><small>无需显卡 · 提交功能待接入</small>
      <p>先看神经元最强和最弱时的句子。提交入口接入后，你写的说明只算待检验假设。</p>
      <button type="button" className="fw-tbtn" onClick={() => onGo('spatial', 'graph', null, { gview: 'structure' })}>打开 3D 模型空间</button></section>
    <section className="cg-way"><h3>贡献算力</h3><small>桌面客户端 · NVIDIA 显卡</small>
      <p>先检查本机是否满足条件。任务、资源和上传范围都会提前说明，可以随时暂停。</p>
      <button type="button" className="fw-tbtn primary" onClick={() => onGo('config', null, null, { csec: 'node' })}>检查我的电脑</button></section>
  </div>;
}

export function ResearchGaps() {
  const rows = [['知识网络', '部分与整体、来源与产地、尺寸与用途'], ['推理与功能', '条件、因果、比较、翻译、上下文指代'], ['语法体系', '标点符号、一致关系、长距离依存']];
  return <section className="cg-section" aria-label="研究缺口"><h3>接下来缺什么</h3><p className="v8-note">这些语言现象还没有完整实验；“尺寸与用途”还要先加入语言目录。</p>
    <div className="cg-grid3">{rows.map(([name, text]) => <div key={name}><strong>{name}</strong><p>{text}</p></div>)}</div></section>;
}

function Steps({ items }) {
  return <ol className="cg-steps">{items.map(([title, note]) => <li key={title}><strong>{title}</strong>{note && <small>{note}</small>}</li>)}</ol>;
}

export function AutoExplainCard() {
  return <section className="cg-section" aria-label="自动解释"><h3>自动解释全部神经元 <span className="cg-ev new">流程设计</span></h3>
    <Steps items={[['取两端例句', '普查数据已有'], ['AI 写说明', '只算假设'], ['写新句子打分', '含相近反例'], ['分级入库', '未过的也保留']]}/>
    <p className="v8-note">当前 265 / 350,208 个神经元做过解释检验，批量任务尚未运行。外部参考：Transluce 公开约 0.046 美元 / 神经元，按此量级全部约 1.6 万美元，实际费用未测算。</p></section>;
}

export function AiLoopCard() {
  return <section className="cg-section" aria-label="AI 研究循环"><h3>AI 研究循环 <span className="cg-ev new">设计中</span></h3>
    <p className="v8-note">AI 可以提出下一个实验，但不能自行认定结论。每轮都要冻结材料和判据，再由人确认。</p>
    <Steps items={[['提出假设'], ['设计句子'], ['冻结并运行'], ['留出验证'], ['修改说明'], ['人工确认']]}/></section>;
}

export function CatalogSummary({ templates, models, tokens }) {
  return <div className="cg-grid3 cg-stats">
    <div><span>语言材料</span><b>{templates ?? '—'}</b><small>套测试模板</small></div>
    <div><span>模型</span><b>{models ?? '—'}</b><small>个模型有分析记录</small></div>
    <div><span>自然语料</span><b>{tokens ?? '—'}</b><small>token，用于神经元普查</small></div>
  </div>;
}

/* ---- 运行与路径：信息流 ---- */
const FLOW_ROWS = [['语言（中/英）', 'ENT-XF', 'lang'], ['语法', 'GR-Q', 'mood'], ['知识', 'KN-ISA', 'category'], ['上下文逻辑', 'ENT-XF', 'polarity'], ['任务线索', 'ENT-XF', 'task']];

export function InfoFlowMap({ model }) {
  const overlay = useApi(model ? `/api/space/overlay?model_key=${encodeURIComponent(model)}` : null);
  const [layer, setLayer] = useState(13);
  const rows = useMemo(() => FLOW_ROWS.map(([name, family, factor]) => {
    const row = (overlay.data?.route_T || []).find(item => item.family_id === family && item.factor === factor);
    if (!row?.sep?.some(value => value != null)) return { name, values: null };
    const values = Array.from({ length: 37 }, () => null);
    row.layers.forEach((hs, index) => { if (hs <= 36) values[hs] = row.sep[index]; });
    const max = Math.max(...values.slice(1).filter(value => value != null));
    return { name, values, max, peak: values.indexOf(max) };
  }), [overlay.data]);
  const strong = rows.filter(row => row.values?.[layer] != null && row.values[layer] / row.max >= .7).map(row => row.name);
  return <section className="v8-card cg-section" aria-label="HiddenState 流中的信息">
    <div className="cg-flow"><div><b>词嵌入</b><small>词 → 数字</small></div><span aria-hidden="true">→</span><div><b>HiddenState 流</b><small>多种信息叠在同一条流里</small></div><span aria-hidden="true">→</span><div><b>输出</b><small>下一个词</small></div></div>
    <h3>流中的信息</h3><p className="v8-note">{shortModel(model)} · 颜色越深越容易读出 · 每行按自身最高值换算</p>
    {overlay.error && <Empty>{overlay.error}</Empty>}
    {!overlay.data && !overlay.error && <p className="v8-note">读取逐层数据…</p>}
    {overlay.data && <div className="cg-heat">{rows.map(row => <div className="cg-heat-row" key={row.name}><strong>{row.name}</strong>
      <div className="cg-cells">{row.values ? row.values.map((value, index) => <i key={index} title={value == null ? `第 ${index} 层：无数据` : `第 ${index} 层：${value}`}
        className={index === layer ? 'on' : ''} style={{ background: value == null ? '#eef1f5' : `rgba(24,79,149,${(.06 + .94 * Math.min(1, value / row.max)).toFixed(2)})` }}/>) : <span className="v8-note">当前模型暂无这一行数据</span>}</div>
      <small>{row.values ? `最清楚：第 ${row.peak} 层` : ''}</small></div>)}
      <div className="cg-heat-row"><strong>风格</strong><span className="v8-note">第 3 层起全程可读，中后层只能“打开”不能“关闭”（Phase 275）。逐层曲线待接入。</span></div></div>}
    <label>移动到某一层：{layer === 0 ? '词嵌入（第 0 层）' : `第 ${layer} 层`}<input type="range" min="0" max="36" value={layer} onChange={event => setLayer(Number(event.target.value))}/></label>
    <p>{strong.length ? `这一层较清楚：${strong.join('、')}` : '这一层没有信息接近各自的最高值'}</p>
    <p className="v8-note">第 0 层就能分开语言，多半来自词本身，不算模型计算出来的。能读出也不等于模型实际使用了它。</p>
  </section>;
}

/* ---- 运行与路径：单句归因图 ---- */
const W = 760, H = 470;
const nodeName = node => `第 ${node.l} 块 · ${node.k === 'm' ? '神经元' : 'Head'} ${node.i}`;

export function AttributionGraph({ snapshot, onOpen }) {
  const [data, setData] = useState(null), [error, setError] = useState('');
  const [graphIndex, setGraphIndex] = useState(0), [selected, setSelected] = useState(0);
  const matches = (snapshot || '').startsWith(DEMO_SNAPSHOT);
  useEffect(() => {
    if (!matches) return undefined;
    let live = true;
    import('./data/qwen3_4b_attr_demo.json').then(file => { if (live) setData(file.default); }).catch(reason => { if (live) setError(reason.message); });
    return () => { live = false; };
  }, [matches]);
  const graph = data?.graphs[graphIndex];
  const layout = useMemo(() => {
    if (!graph) return null;
    const last = Math.max(1, graph.tokens.length - 1), x = position => 70 + position * (W - 150) / last, y = layer => 415 - layer / 35 * 365;
    const seen = {}, maxNode = Math.max(...graph.nodes.map(node => Math.abs(node.s)));
    const points = graph.nodes.map(node => {
      const key = `${node.p}:${Math.floor(node.l / 3)}`, order = seen[key] = (seen[key] || 0) + 1;
      return [x(node.p) + (order % 2 ? 1 : -1) * Math.floor(order / 2) * 18, y(node.l)];
    });
    return { x, points, maxNode, maxEdge: Math.max(.0001, ...graph.edges.map(edge => Math.abs(edge[2]))) };
  }, [graph]);
  if (!matches) return <section className="v8-card cg-section"><h3>这句话的归因图</h3><p className="v8-note">演示数据只对应 Qwen3-4B 快照 {DEMO_SNAPSHOT}。切换到这个模型后可以查看。</p></section>;
  const node = graph?.nodes[Math.min(selected, graph.nodes.length - 1)];
  const percent = value => `${(100 * value).toFixed(1)}%`;
  return <section className="v8-card cg-section" aria-label="单句归因图"><h3>这句话的归因图</h3>
    <p className="v8-note">模型为什么说出这个词？下面三句用本机 Qwen3-4B 实算，是未预注册的界面演示，不计入研究证据。</p>
    <div className="v6-toolbar">{['知识网络', '推理与功能', '语法体系'].map((label, index) => <button type="button" className="fw-tbtn" key={label} aria-pressed={graphIndex === index} onClick={() => { setGraphIndex(index); setSelected(0); }}>{label}</button>)}</div>
    {error && <Empty>{error}</Empty>}
    {!graph && !error && <p className="v8-note">读取归因数据…</p>}
    {graph && <>
      <p>原文“{graph.text}”，模型最可能接“{graph.top[0][0]}”（{percent(graph.top[0][1])}），第二是“{graph.top[1][0]}”（{percent(graph.top[1][1])}）。下面解释两者的差距 {graph.base.toFixed(2)}。{graphIndex === 1 && ' 注意：“奥”和“奥斯”都可能通向“奥斯汀”，这句的差距只作示意。'}</p>
      <div className="cg-attr"><div className="cg-attr-scroll"><svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`${graph.text}的归因图`}>
        {graph.edges.map((edge, index) => {
          const from = edge[0] >= 0 ? layout.points[edge[0]] : [layout.x(-1 - edge[0]), 430], to = layout.points[edge[1]], active = edge[0] === selected || edge[1] === selected;
          return <line key={index} x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} stroke={edge[2] >= 0 ? '#184f95' : '#b4531c'} strokeDasharray={edge[2] >= 0 ? undefined : '7 4'} strokeWidth={1 + 3 * Math.abs(edge[2]) / layout.maxEdge} opacity={active ? .95 : .14}/>;
        })}
        {graph.nodes.slice(0, 10).map((item, index) => <line key={`out${index}`} x1={layout.points[index][0]} y1={layout.points[index][1]} x2={layout.x(graph.tokens.length - 1)} y2={24} stroke={item.s >= 0 ? '#184f95' : '#b4531c'} strokeDasharray={item.s >= 0 ? undefined : '7 4'} strokeWidth={1 + 3 * Math.abs(item.s) / layout.maxNode} opacity={index === selected ? .95 : .14}/>)}
        {graph.tokens.map((token, index) => <text key={index} x={layout.x(index)} y={450} textAnchor="middle" className="cg-attr-token">{token}</text>)}
        <text x={layout.x(graph.tokens.length - 1)} y={16} textAnchor="end" className="cg-attr-out">输出：{graph.top[0][0]}</text>
        <text x={6} y={46} className="cg-attr-axis">深层</text><text x={6} y={416} className="cg-attr-axis">浅层</text>
        {graph.nodes.map((item, index) => {
          const size = 6 + 8 * Math.abs(item.s) / layout.maxNode, [cx, cy] = layout.points[index], label = `${nodeName(item)}，读到“${graph.tokens[item.p]}”`;
          const common = { fill: item.k === 'm' ? '#d6541e' : '#245ca8', stroke: index === selected ? '#16202c' : '#fff', strokeWidth: index === selected ? 3 : 1.5, opacity: item.s < 0 ? .72 : 1 };
          return <g key={index} role="button" tabIndex={0} aria-label={label} className="cg-attr-node" onClick={() => setSelected(index)}
            onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSelected(index); } }}><title>{label}</title>
            {item.k === 'm' ? <circle cx={cx} cy={cy} r={size} {...common}/> : <rect x={cx - size} y={cy - size} width={size * 2} height={size * 2} rx="3" {...common}/>}</g>;
        })}
      </svg></div>
      <div className="cg-attr-side"><h4>{nodeName(node)}</h4><p className="v8-note">读到“{graph.tokens[node.p]}”这个位置</p>
        <dl className="cg-kv"><dt>归因分数</dt><dd>{node.s.toFixed(3)}</dd><dt>关闭后差距</dt><dd>{graph.base.toFixed(2)} → {node.off.toFixed(2)}</dd><dt>增强 2 倍后差距</dt><dd>{graph.base.toFixed(2)} → {node.x2.toFixed(2)}</dd><dt>关闭后首词</dt><dd>{node.flip ? `变成“${node.top}”` : '不变'}</dd></dl>
        {onOpen && <button type="button" className="fw-tbtn" onClick={() => onOpen(node)}>在 3D 中查看</button>}
        <h4>关闭一组节点</h4><dl className="cg-kv"><dt>归因最强的 {graph.group.k} 个支持节点</dt><dd>{graph.base.toFixed(2)} → {graph.group.off.toFixed(2)}，首词“{graph.group.top}”</dd><dt>同位置随机 {graph.group.k} 个（20 次中位）</dt><dd>{graph.base.toFixed(2)} → {graph.group.rnd_med.toFixed(2)}</dd><dt>随机范围</dt><dd>{graph.group.rnd_min.toFixed(2)} 到 {graph.group.rnd_max.toFixed(2)}</dd></dl></div></div>
      <p className="v8-note">节点越大影响越强；蓝色实线推动，橙色虚线抑制。连线是梯度近似，包含间接路径，不等于已证实的计算连接；“关闭”和“增强 2 倍”是实际重跑模型的结果。</p>
    </>}
  </section>;
}

/* ---- 响应对照：第三部分看板 ---- */
const PAIRS = [
  ['疑问与陈述', '我想说，火是热的。', '我想问，火是热的吗？', 'GR-NEG', 'mood', '真实模板例句 · 读目标词“热”'],
  ['否定', '灯亮着。', '灯没有亮。', 'ENT-XF', 'polarity', '语言目录例句；曲线来自 TG-ENT-XF'],
  ['中文与英文', '早上好', 'Good morning', 'ENT-XF', 'lang', '语言目录例句；曲线来自 TG-ENT-XF'],
  ['施事与受事', '警察抓了小偷。', '小偷抓了警察。', 'GR-ROLE', 'role', '真实模板例句 · TG-GR-ROLE'],
];

export function MinimalPairs({ model }) {
  const overlay = useApi(model ? `/api/space/overlay?model_key=${encodeURIComponent(model)}` : null);
  const [index, setIndex] = useState(0);
  const pair = PAIRS[index];
  const row = (overlay.data?.route_T || []).find(item => item.family_id === pair[3] && item.factor === pair[4]);
  const values = row?.sep?.some(value => value != null) ? row.sep : null;
  const max = values ? Math.max(...values.filter(value => value != null)) : 1;
  const peak = values ? row.layers[values.indexOf(max)] : null;
  return <section className="v8-card cg-section" aria-label="最小对照"><h3>只改一处，看内部哪里变了</h3>
    <div className="v6-toolbar">{PAIRS.map((item, position) => <button type="button" className="fw-tbtn" key={item[0]} aria-pressed={index === position} onClick={() => setIndex(position)}>{item[0]}</button>)}</div>
    <div className="cg-pair"><div><small>句子 A</small><p>{pair[1]}</p></div><div><small>句子 B</small><p>{pair[2]}</p></div></div>
    <p className="v8-note">{pair[5]}</p>
    {values ? <><div className="cg-bars" role="img" aria-label={`${pair[0]}在各层的差别`}>{values.map((value, position) => <i key={position} title={`第 ${row.layers[position]} 层：${value ?? '无数据'}`} style={{ height: `${value == null ? 0 : 100 * value / max}%` }}/>)}</div><p>第 {peak} 层差别最大。这是留出样本上的投影间隔，只说明内部能读出差别。</p></>
      : <p>这组有数据，但没有读出结果。空白也是结果，不用估计值填充。</p>}
  </section>;
}

const LANGUAGE_MAP = [['document_*_answer / selection', '属性绑定 KN.ATTR', '56', '有留出响应'], ['register*', '正式与口语 VAR.REGISTER', '40', '部分干预，部分未成立'],
  ['source-scope / reference / entity-*', '知识来源与冲突 SEM.SOURCE', '84', '观察和未成立'], ['prefix-token-category-context', '名词类别 KN.CAT', '466', '未成立'],
  ['lookup / dispatch_selected', '属性绑定 KN.ATTR', '2', '已做干预'], ['conditioned_yes_no_answer', '条件推理 RS.COND（待审）', '7', '有留出响应']];
const FIDELITY = [['图谱独立预测下一词（KL）', '未测', '词身份、位置、n-gram：未测', '关键缺口'], ['字典激活重建 FVU（第 35 块）', '0.1138', '旧字典 0.4275', '通过保真门'],
  ['自然下一词损失恢复（第 35 块）', '字典 0.589', '同预算原生 1.055', '未胜过原生'], ['演示句关闭节点', '见运行与路径', '随机节点对照', '工程演示'], ['未解释部分', '349,943 个神经元未做解释检验', '—', '保留']];

function DataTable({ head, rows, label }) {
  return <div className="cg-table" role="table" aria-label={label}><div role="row" className="head">{head.map(cell => <span role="columnheader" key={cell}>{cell}</span>)}</div>
    {rows.map(row => <div role="row" key={row[0]}>{row.map((cell, index) => <span role="cell" key={index}>{cell}</span>)}</div>)}</div>;
}

export function RelationBoards({ only }) {
  const { BENCH } = useMainline();
  const show = key => !only || only === key;
  return <>
    {show('map') && <section className="v8-card cg-section"><h3>语言目录映射</h3><p className="v8-note">已登记连接使用实验内部名称。这里是建议归属，审定后才能按语言项筛选。</p><DataTable label="语言目录映射" head={['实验内部名称', '建议语言项', '连接数', '结果']} rows={LANGUAGE_MAP}/></section>}
    {show('fidelity') && <section className="v8-card cg-section"><h3>保真看板</h3><p className="v8-note">这里回答“解释到了多少”。覆盖数量不代表破解进度。</p><DataTable label="保真看板" head={['指标', '当前结果', '对照', '状态']} rows={FIDELITY}/></section>}
    {show('bench') && <section className="v8-card cg-section"><h3>方法基准（自建，Phase 315、316）</h3>
      <p className="v8-note">同一批冻结任务、同一预算比较定位方法。数字是 lockbox 上 cf 节点的 auc_f，越高说明越少节点就能恢复行为。这是候选观测，不是官方 MIB。</p>
      <DataTable label="MB-v1 方法基准" head={BENCH.mb.head} rows={BENCH.mb.rows}/>
      <p className="v8-note">1% 预算下的特异性（SP-v1）：排除通用主干后，排序才开始显示任务特异。</p>
      <DataTable label="SP-v1 特异性" head={BENCH.sp.head} rows={BENCH.sp.rows}/>
      <p className="v8-note">来源：{BENCH.source}</p></section>}
    {show('public') && <section className="v8-card cg-section"><h3>公开基准</h3><p className="v8-note">官方 MIB 还没有提交过，所以不给排名。</p><DataTable label="公开基准" head={['MIB 赛道', '原生单元', '字典特征', '路径归因']} rows={[['电路定位', '未测', '未测', '未测'], ['因果变量定位', '未测', '未测', '未测']]}/></section>}
  </>;
}

/* ---- 结构：基底、二维列表、Head 内部 ---- */
export function BasisCompare() {
  return <section className="v8-card cg-section"><h3>特征基底对比</h3><p className="v8-note">同一模型可以用不同单位来解释。三种基底分开记录，不互相替代。</p>
    <DataTable label="特征基底对比" head={['基底', '规模', '保真结果', '状态']} rows={[['原生神经元', '350,208 个', '同预算损失恢复 1.055（第 35 块）', '解释检验 265 个'], ['字典特征（SAE）', '已登记 50 张卡：第 8、24、35 块', '第 35 块 FVU 0.1138，损失恢复 0.589', '36 层旧字典 0 层优于原生'], ['参数组件（VPD 类）', '—', '未测', '未实现']]}/></section>;
}

export function LayerList({ tower, onPick }) {
  if (!tower) return <p className="v8-note">当前模型暂无逐层普查数据。</p>;
  const names = tower.type_names || [];
  return <section className="v8-card cg-section" aria-label="全部层列表"><h3>全部层</h3><p className="v8-note">不用 3D 也能定位。点一层后，页面会选中同一个对象。</p>
    <div className="cg-layers">{tower.blocks.map(block => {
      const types = block.types || [], main = types.indexOf(Math.max(...types));
      const explained = Object.values(block.explained || {}).reduce((sum, value) => sum + value, 0);
      return <button type="button" key={block.block} onClick={() => onPick(block.block + 1)}><strong>第 {block.block} 块</strong><span>已解释检验 {explained} 个</span><span>{(names[main] || '').replace(/^T\d\s*/, '')}</span></button>;
    })}</div></section>;
}

function useDataFile(enabled, load) {
  const [data, setData] = useState(null);
  useEffect(() => {
    if (!enabled) return undefined;
    let live = true;
    load().then(file => { if (live) setData(file.default); }).catch(() => {});
    return () => { live = false; };
  }, [enabled, load]);
  return data;
}
const loadHeadInfo = () => import('./data/qwen3_4b_head_info.json');
const loadHeadWeights = () => import('./data/qwen3_4b_head_weights.json');

export function HeadFeature({ snapshot, block, head }) {
  const matches = (snapshot || '').startsWith(DEMO_SNAPSHOT), info = useDataFile(matches, loadHeadInfo);
  if (!matches) return <p className="v8-note">Head 读写数据目前只有 Qwen3-4B。</p>;
  if (!info) return <p className="v8-note">读取 Head 数据…</p>;
  const out = info.ov[`${block}:${head}`];
  return <div className="cg-grid2"><div><h4>它主要写出什么</h4><p className="v8-note">O 矩阵最强方向投到词表的两端，只是直接通路近似。</p>
    <strong>一端</strong><div className="cg-tokens">{out[1].map((token, index) => <code key={index}>{token}</code>)}</div>
    <strong>另一端</strong><div className="cg-tokens">{out[2].map((token, index) => <code key={index}>{token}</code>)}</div></div>
    <div><h4>在演示句里看哪里</h4><p className="v8-note">最后一个词的注意力，颜色越深看得越多。</p>
      {info.att.map((sample, index) => <div className="cg-attn" key={index}>{sample.tokens.map((token, position) => {
        const value = sample.a[block][head][position];
        return <span key={position} title={value.toFixed(2)} style={{ background: `rgba(24,79,149,${value})`, color: value > .5 ? '#fff' : undefined }}>{token}</span>;
      })}</div>)}</div></div>;
}

function decode(text) {
  const raw = atob(text), values = new Uint16Array(raw.length / 2);
  for (let index = 0; index < values.length; index += 1) values[index] = raw.charCodeAt(2 * index) | (raw.charCodeAt(2 * index + 1) << 8);
  return values;
}

export function HeadInternals({ snapshot, block, head }) {
  const matches = (snapshot || '').startsWith(DEMO_SNAPSHOT), weights = useDataFile(matches, loadHeadWeights);
  const [cell, setCell] = useState(null);
  if (!matches) return null;
  if (!weights) return <p className="v8-note">读取 Head 权重数据…</p>;
  const group = Math.floor(head / 4), layer = weights.L[block];
  const parts = [['q', 'Q 查询：它想找什么', '这个 Head 独有', head], ['k', 'K 键：内容怎样被找到', `第 ${group} 组，4 个 Head 共用`, group], ['v', 'V 值：找到后带走什么', `第 ${group} 组，4 个 Head 共用`, group], ['o', 'O 输出：写回信息流', '这个 Head 独有', head]];
  return <section className="cg-section" aria-label="Head 内部全部维度"><h4>Head 内部的全部维度</h4>
    <p className="v8-note">Q、K、V、O 各 128 维，共 512 个。颜色越深，这一维的权重向量越长。这是固定参数强度，不是运行时激活；单个维度不等于一个概念。</p>
    <div className="cg-head-grid">{parts.map(([key, title, note, offset]) => {
      const values = decode(layer[key]), [low, high] = layer.s[key];
      return <div key={key}><strong>{title}</strong><small>{note}</small><div className="cg-dims">{Array.from({ length: 128 }, (_, dim) => {
        const value = values[offset * 128 + dim], depth = Math.max(0, Math.min(1, (value - low) / Math.max(1, high - low))), label = `${key.toUpperCase()} 第 ${dim} 维：${(value / 1000).toFixed(3)}`;
        return <button type="button" key={dim} title={label} aria-label={label} className={cell?.key === key && cell.dim === dim ? 'on' : ''}
          style={{ background: `rgba(24,79,149,${(.04 + .96 * depth).toFixed(3)})` }} onClick={() => setCell({ key, dim, value: value / 1000 })}/>;
      })}</div><div className="cg-scale"><small>浅 {(low / 1000).toFixed(2)}</small><small>深 {(high / 1000).toFixed(2)}</small></div></div>;
    })}</div>
    <p>{cell ? `${cell.key.toUpperCase()} 第 ${cell.dim} 维：权重向量长度 ${cell.value.toFixed(3)}` : '点任意一格，查看这一维的数值。'}</p>
    <p className="v8-note">颜色只在同一层、同一类矩阵内比较，两端 1% 截断。</p></section>;
}

export function PermissionTable() {
  return <DataTable label="账号权限" head={['能做什么', '访客', '成员', '管理员']} rows={[['浏览公开研究', '可以', '可以', '可以'], ['贡献算力并署名', '不可以', '可以', '可以'], ['登记研究包', '不可以', '不可以', '可以'], ['管理账号和数据', '不可以', '不可以', '可以']]}/>;
}
