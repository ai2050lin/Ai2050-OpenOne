/* 研发 · 神经元测试（goodfire_probe.v1）：Goodfire《Neural Networks Are Already Clean》找特征流程的 S1–S8 自动 + S9 最小对。
   只在桌面客户端：本机节点服务下载中心普查作业的语料切片（按 sha256 核对），用本机 GPU 对一个 MLP 中间单元跑两极采集、
   集中度、Δ 与劈半、对偶与肩部留出、词表投影、读 / 写方向、双向 steering 与最小对；结果只存在本机，不上传。
   判据固定在 shared/dist_protocol/goodfire_probe.py 的 CRITERIA 并随结果返回；通过只描述本次检验，不是证据等级。
   方法拆解见 research/gpt5/docs/GOODFIRE_FIND_FEATURE_STEPS_v2.md。 */
import { useEffect, useRef, useState } from 'react';

import { fmtTime, nodeBridge, nodeCall } from './contributeUtil.js';
import { Card, Empty } from './ui.jsx';
import './contribute.css';

const DEFAULT_PROMPTS = ['我想聊聊关于', '这篇新闻报道的是', 'The article is about', 'Let me tell you about'];
const VERDICT = {
  candidate: ['good', '✓', '可入册候选（S1–S9 全部通过）'],
  auto_passed_needs_S9: ['warn', '◐', '自动步全部通过，S9 待补最小对'],
  not_passed: ['bad', '✗', '未全部通过'],
};
const READING = { axis: '两极都集中：像一条轴', mixture: '正极集中、负极散：像混合物', insufficient_data: '两极都散：语料覆盖可能不够', unreadable: '读不出结构' };
const STATUS = { preparing: ['warn', '…', '准备语料'], running: ['info', '▶', '运行中'], done: ['good', '✓', '完成'], failed: ['bad', '✗', '失败'], cancelled: ['muted', '○', '已取消'] };

function Pill({ tone, icon, children }) {
  return <span className={'v6-pill ' + tone}><i aria-hidden="true">{icon}</i>{children}</span>;
}

const fx = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d));

function parsePairs(text) {
  return text.split('\n').map(l => l.split('||').map(x => x.trim())).filter(p => p.length === 2 && p[0] && p[1]).map(([a, b]) => ({ a, b }));
}

function Ledger({ r }) {
  const c = r.checks;
  const C = r.criteria;
  const rows = [
    ['S1', '语料位置 / 窗口', `${r.corpus.n_positions.toLocaleString()} / ${r.corpus.n_windows.toLocaleString()}`, '—', null],
    ['S2', '正极集中度 coh_P（h − μ）', fx(c.S2_coh.value), `≥ ${C.coh_P_min} 且 ≥ ${C.coh_ratio_to_random}×随机 ${fx(c.S2_coh.random, 2)}`, c.S2_coh.pass],
    ['S3', 'k', c.S3_k.value, `≥ ${C.k_min}`, c.S3_k.pass],
    ['S3', '有效文本段（正 / 负）', `${c.S3_effective_texts.pos} / ${c.S3_effective_texts.neg}`, `≥ ${c.S3_effective_texts.need}`, c.S3_effective_texts.pass],
    ['S4', '劈半 cos → 信度', `${fx(c.S4_split_half.value)} → ${fx(c.S4_split_half.reliability)}`, `≥ ${C.split_half_min}`, c.S4_split_half.pass],
    ['S5', '对偶 R = coh_N / coh_P', `${fx(c.S5_duality.value)}（coh_N ${fx(c.S5_duality.coh_N)}）`, `≥ ${C.R_min}`, c.S5_duality.pass],
    ['S5', '肩部留出落在正确一侧', c.S5_shoulder.acc == null ? '—' : `${(100 * c.S5_shoulder.acc).toFixed(0)}%（${c.S5_shoulder.n}）`, `≥ ${100 * C.shoulder_acc_min}%`, c.S5_shoulder.pass],
    ['S6', '‖Δ‖', fx(r.delta_norm, 2), '方向才重要', null],
    ['S7', 'steering 单调 / 双向', r.S7.summary ? `${(100 * r.S7.summary.monotone_frac).toFixed(0)}% / ${(100 * r.S7.summary.both_sides_frac).toFixed(0)}%` : '未做', `≥ ${100 * C.steer_monotone_min}%`, r.S7.summary ? r.S7.summary.pass : false],
    ['S8', 'cos(读, Δ) − cos(写, Δ)', `${fx(c.S8_read_write.cos_read)} − ${fx(c.S8_read_write.cos_write)} = ${fx(c.S8_read_write.value)}`, `≤ ${C.read_write_gap_max}`, c.S8_read_write.pass],
    ['S9', '最小对符号正确', r.S9.summary ? `${(100 * r.S9.summary.sign_acc).toFixed(0)}%（${r.S9.summary.n} 对，平均 Δz ${fx(r.S9.summary.mean_dz, 2)}）` : '未做', `≥ ${100 * C.pairs_sign_min}%`, r.S9.summary ? r.S9.summary.pass : null],
  ];
  return (
    <table className="v6-mini-table">
      <thead><tr><th>步</th><th>量</th><th>本次值</th><th>判据</th><th>结论</th></tr></thead>
      <tbody>{rows.map(([s, q, v, crit, ok], i) => (
        <tr key={i}><td>{s}</td><td>{q}</td><td className="ct-mono">{v}</td><td>{crit}</td>
          <td>{ok == null ? '—' : ok ? <Pill tone="good" icon="✓">通过</Pill> : <Pill tone="bad" icon="✗">不通过</Pill>}</td></tr>
      ))}</tbody>
    </table>
  );
}

function Poles({ items, label }) {
  return (
    <div>
      <b className="gf-sub">{label}</b>
      <ol className="gf-poles">{items.map((p, i) => (
        <li key={i}><span className="ct-mono gf-z">z {fx(p.z, 1)}</span><span className="gf-ctx">{p.left}<mark>{p.hit}</mark>{p.right}</span></li>
      ))}</ol>
    </div>
  );
}

function Report({ r }) {
  const [tone, icon, label] = VERDICT[r.verdict?.status] || ['muted', '·', r.verdict?.status];
  return (
    <div className="gf-report">
      <div className="gf-head">
        <b>第 {r.params.layer} 层 · 单元 {r.params.unit}</b>
        <Pill tone={tone} icon={icon}>{label}</Pill>
        <span>{READING[r.reading] || r.reading}</span>
        {r.verdict?.failed?.length > 0 && <small>未通过：{r.verdict.failed.join('、')}</small>}
      </div>
      <p className="v6-note">语料：普查作业 {r.corpus.job_id}（{r.corpus.slices.join('、')}）· {r.env.gpu} · torch {r.env.torch} · 用时 {r.seconds.total} 秒。
        本机检验结论，不是证据等级，不上传。h 取 post_attention_layernorm 的输入，两极按 gate 的 pre-activation g 选取。</p>
      <Ledger r={r} />
      <div className="gf-two">
        <Poles items={r.poles.pos} label="正极（g 最大）" />
        <Poles items={r.poles.neg} label="负极（g 最小）" />
      </div>
      <div className="gf-two">
        <div><b className="gf-sub">S6 Δ 正端推高的词</b><p className="gf-words">{r.S6.pos.join('  ')}</p>
          <small>对照：单元写方向 W_down 的直接读出 {r.S6.write_pos.slice(0, 8).join(' ')}</small></div>
        <div><b className="gf-sub">S6 Δ 负端推高的词</b><p className="gf-words">{r.S6.neg.join('  ')}</p>
          <small>对照：{r.S6.write_neg.slice(0, 8).join(' ')}</small></div>
      </div>
      {r.S7.prompts.length > 0 && (
        <>
          <b className="gf-sub">S7 steering（α × ‖h‖ = {fx(r.S7.h_scale, 1)}，加在第 {r.params.layer} 层输入；shift = Δ 正端词 − 负端词的平均 logit）</b>
          <table className="v6-mini-table gf-steer">
            <thead><tr><th>提示</th><th className="num">α</th><th className="num">shift</th><th className="num">单元 g</th><th>贪心续写</th></tr></thead>
            <tbody>{r.S7.prompts.map(p => p.rows.map((row, i) => (
              <tr key={p.prompt + row.alpha} className={i === 0 ? 'first' : ''}>
                {i === 0 && <td rowSpan={p.rows.length}>{p.prompt}</td>}
                <td className="num">{row.alpha > 0 ? '+' : ''}{row.alpha}</td><td className="num">{fx(row.shift, 2)}</td><td className="num">{fx(row.g_last, 2)}</td>
                <td className="gf-gen">{row.text}</td>
              </tr>
            )))}</tbody>
          </table>
          <p className="v6-note">只有一侧有反应、或续写变成重复与乱码，说明在破坏分布而不是移动轴；文本是否通顺需要人读。</p>
        </>
      )}
      {r.S9.pairs.length > 0 && (
        <>
          <b className="gf-sub">S9 最小对（A 预期在正端；z 为该句 g 最大值的语料 z 分数）</b>
          <table className="v6-mini-table">
            <thead><tr><th>A</th><th>B</th><th className="num">z_A</th><th className="num">z_B</th><th className="num">Δz</th></tr></thead>
            <tbody>{r.S9.pairs.map((p, i) => (
              <tr key={i}><td>{p.a}</td><td>{p.b}</td><td className="num">{fx(p.z_a, 2)}</td><td className="num">{fx(p.z_b, 2)}</td>
                <td className="num">{p.dz > 0 ? '✓ ' : '✗ '}{fx(p.dz, 2)}</td></tr>
            ))}</tbody>
          </table>
        </>
      )}
      {r.S8.upstream && (
        <p className="v6-note">S8 上游候选（第 {r.S8.upstream.layer} 层，|cos| 的 99 分位 {fx(r.S8.upstream.abs_p99)}）：
          {r.S8.upstream.top.map(u => ` U${u.unit}（${fx(u.cos)}）`).join('、')}。{r.S8.upstream.note}</p>
      )}
    </div>
  );
}

export default function GoodfireProbePanel() {
  const bridge = nodeBridge();
  const [models, setModels] = useState(null);
  const [form, setForm] = useState({ snapshot: '', layer: 30, unit: 9517, k: 100, n_slices: 2, prompts: DEFAULT_PROMPTS.join('\n'), pairs: '' });
  const [st, setSt] = useState(null);
  const [items, setItems] = useState([]);
  const [report, setReport] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    if (!bridge) return undefined;
    nodeCall('GET', '/models').then(r => {
      const ok = r.models.filter(m => m.status === 'ok');
      setModels(ok);
      if (ok[0]) setForm(f => ({ ...f, snapshot: f.snapshot || ok[0].snapshot }));
    }).catch(e => setErr(e.message));
    nodeCall('GET', '/probes').then(r => setItems(r.items)).catch(() => {});
    return undefined;
  }, [bridge]);

  const busy = st && (st.status === 'preparing' || st.status === 'running');
  const wasBusy = useRef(false);
  useEffect(() => {
    if (!bridge) return undefined;
    const t = setInterval(async () => {
      try {
        const s = await nodeCall('GET', '/probe');
        const running = s.status === 'preparing' || s.status === 'running';
        if (wasBusy.current && !running) {                       // 刚结束：刷新记录，完成时打开报告
          if (s.status === 'done') nodeCall('GET', `/probes/${s.id}`).then(setReport).catch(e => setErr(e.message));
          nodeCall('GET', '/probes').then(r => setItems(r.items)).catch(() => {});
        }
        wasBusy.current = running;
        setSt(s);
      } catch { /* 服务重启中 */ }
    }, busy ? 1500 : 4000);
    return () => clearInterval(t);
  }, [bridge, busy]);

  if (!bridge) return <Card title="神经元测试"><Empty>神经元测试要在本机 GPU 上运行模型，只能在桌面客户端里进行。</Empty></Card>;
  const set = patch => setForm(f => ({ ...f, ...patch }));
  const start = async () => {
    setErr(''); setReport(null); wasBusy.current = true;
    try {
      setSt(await nodeCall('POST', '/probe/start', {
        snapshot: form.snapshot, layer: Number(form.layer), unit: Number(form.unit), k: Number(form.k), n_slices: Number(form.n_slices),
        prompts: form.prompts.split('\n').map(x => x.trim()).filter(Boolean), pairs: parsePairs(form.pairs),
      }));
    } catch (e) { setErr(e.message); }
  };
  const open = id => nodeCall('GET', `/probes/${id}`).then(setReport).catch(e => setErr(e.message));
  const [stTone, stIcon, stLabel] = STATUS[st?.status] || [];
  const prog = st?.progress;
  return (
    <div className="ct-root">
      <Card title="Goodfire 神经元测试" sub="选一个 MLP 中间单元：本机 GPU 跑 S1 两极采集 → S2–S5 集中度 / Δ / 对偶 / 肩部 → S6 词表 → S7 双向 steering → S8 读写方向；S9 填最小对"
            extra={busy ? <button type="button" className="fw-tbtn ghost" onClick={() => nodeCall('POST', '/probe/cancel').then(setSt).catch(e => setErr(e.message))}>取消</button>
              : <button type="button" className="fw-tbtn" disabled={!form.snapshot} onClick={start}>开始测试</button>}>
        <div className="ct-form">
          <label>模型<select value={form.snapshot} onChange={e => set({ snapshot: e.target.value })}>
            {(models || []).map(m => <option key={m.snapshot} value={m.snapshot}>{m.name}（{m.snapshot.slice(0, 10)}）</option>)}
          </select>{models && !models.length && <small className="ct-err">还没有快照一致的模型：先在「贡献算力 → 模型」下载或核对</small>}</label>
          <label>层（块序号）<input className="ct-input num" type="number" min="0" value={form.layer} onChange={e => set({ layer: e.target.value })} /></label>
          <label>单元编号<input className="ct-input num" type="number" min="0" value={form.unit} onChange={e => set({ unit: e.target.value })} /></label>
          <label>两极 k<input className="ct-input num" type="number" min="20" max="500" value={form.k} onChange={e => set({ k: e.target.value })} /></label>
          <label>语料切片数<span><input className="ct-input num" type="number" min="1" max="8" value={form.n_slices} onChange={e => set({ n_slices: e.target.value })} /> 个（每个约 50 万 token，来自中心普查作业）</span></label>
        </div>
        <div className="gf-two">
          <label className="gf-area">S7 steering 提示（每行一条）<textarea rows={4} value={form.prompts} onChange={e => set({ prompts: e.target.value })} /></label>
          <label className="gf-area">S9 最小对（每行「A 句 || B 句」，A 预期在正端；可留空）
            <textarea rows={4} value={form.pairs} placeholder="香港与内地的往来越来越频繁。 || 巴黎与里昂的往来越来越频繁。" onChange={e => set({ pairs: e.target.value })} /></label>
        </div>
        {st && st.status !== 'idle' && (
          <p className="gf-status"><Pill tone={stTone} icon={stIcon}>{stLabel}</Pill> <span className="ct-mono">{st.id}</span>
            {prog && <small>{prog.stage}{prog.total ? ` ${prog.done}/${prog.total}` : ''}{prog.tokens ? ` · ${prog.tokens.toLocaleString()} 个位置 · ${prog.tok_per_s} 位置/s` : ''}</small>}
            {st.error && <small className="ct-err">{st.error}</small>}</p>
        )}
        {err && <p className="v6-note warn">{err}</p>}
        <p className="v6-note">测试期间贡献算力不领取新任务；正在运行贡献任务时不能开始。参数与判据的出处见方法拆解文档；判据随结果一并保存。</p>
      </Card>
      {report && <Card title="测试报告"><Report r={report} /></Card>}
      <Card title="本机测试记录" sub="结果只存在本机（节点服务目录 probes/），不上传">
        {!items.length ? <Empty>还没有测试记录。</Empty> : (
          <table className="v6-mini-table">
            <thead><tr><th>时间</th><th>层 · 单元</th><th>状态</th><th>结论</th><th /></tr></thead>
            <tbody>{items.map(it => {
              const [t1, i1, l1] = STATUS[it.status] || ['muted', '·', it.status];
              const [t2, i2, l2] = VERDICT[it.verdict?.status] || [];
              return (
                <tr key={it.id}><td>{fmtTime(it.started)}</td><td>L{it.params?.layer} · U{it.params?.unit}</td><td><Pill tone={t1} icon={i1}>{l1}</Pill></td>
                  <td>{t2 ? <Pill tone={t2} icon={i2}>{l2}</Pill> : <small className="ct-err">{it.error?.slice(0, 80)}</small>}</td>
                  <td>{it.status === 'done' && <button type="button" className="fw-tbtn ghost" onClick={() => open(it.id)}>查看</button>}</td></tr>
              );
            })}</tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
