/* 研究工作台 · 进展：内容按 tab 渲染（?rtab=neuron|lang；页签本身在 Workbench 的顶部子导航，
   与空间/研发共用同一个 SubNav 组件与 .wb-subnav 样式）。
   神经元逆向分析进展：数据派生的一句话进展 + 全景剖面图（四条轨道共用同一横轴：分型构成 / 强激活率 /
   已解释 / 模板选择性，一列 = 一个块，点击进入该块）。
   语言图谱进展：图谱测试题（C-Eval 数据集对本机模型的实测正确率）+ 大类 × 等级的覆盖 +
   语言项 × 模型的等级矩阵 + 研究缺口。
   各来源分开计数，不拼成转化率；全部读取来自本机数据包。 */
import { useEffect, useMemo, useRef, useState } from 'react';

import RouteBoard from '../app/rdc_fusion/RouteBoard.jsx';
import { API_BASE, postJson, useApi } from '../app/rdc_fusion/api.js';
import { Card, Empty } from '../app/rdc_fusion/ui.jsx';
import '../app/workbench.css';
import { LevelChip } from './LangPage.jsx';
import RuleLedgerPage from './RuleLedgerPage.jsx';
import { LEVELS, GRADE_COLOR, READOUT_LABEL, TYPE_COLOR, defaultModel, itemLevel, levelMeta, shortModel } from './v8Util.js';
import './v8.css';

function fmtTok(v) {
  if (!v) return '—';
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  return `${Math.round(v / 1e3)}K`;
}

const EXPL_LABEL = { 文本可复现: '文本可复现', 只在语料中可识别: '语料可识别', 未解释: '未解释' };
const EXPL_GRADES = ['文本可复现', '只在语料中可识别', '未解释'];

/* 数据集 / 模型的取名与比对。题库的 source 是 'C-Eval val' / 'MMLU test' / 'M3Exam zh test'，
   归类键用第一段（'C-Eval'）；模型 id 是 'qwen3-4b|cfg-…|bf16'，实载目录名是 'Qwen3-4B' / 'glm4-9b-chat-hf'，
   只比发布名（不含精度与配置），因为 bench 跑的就是当前载入的那份权重。 */
const DS_SHORT = { 'C-Eval val': 'C-Eval', 'MMLU test': 'MMLU', 'M3Exam zh test': 'M3Exam',
  'CMMLU test': 'CMMLU', 'CMExam train': 'CMExam' };
const dsLabel = id => DS_SHORT[id] || String(id || '').split(' ')[0] || id;
const dsPrefix = id => String(id || '').trim().split(' ')[0];
const modelBase = m => String(m || '').split('|')[0].trim().toLowerCase().replace(/_/g, '-');
const sameModel = (a, b) => {
  const x = modelBase(a), y = modelBase(b);
  return Boolean(x) && Boolean(y) && (x === y || y.startsWith(x));
};
const BENCH_COLOR = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#7a3fbf', '#008300'];

/* 全景剖面图：x 轴 = 块 0…n-1，四条轨道共用同一横轴（自上而下：分型构成 / 强激活率中位数 /
   已解释单元 / 模板选择性入选）。一列 = 一个块：跨轨对齐能直接看出「哪一段深度的块检测器密集、
   又恰好被解释得多」。每根条悬停有原始数值，点击任意一列进入该块的神经元阵列。 */
function ProfileChart({ tower, selVals, selRuns, ro, onPick }) {
  const n = tower.n_layers;
  const blocks = tower.blocks;
  const colW = 15; const gap = 3; const left = 160; const pitch = colW + gap;
  const trackH = 32; const trackGap = 28;
  const W = left + n * pitch + 6;
  const axisY = 3 * (trackH + trackGap) + trackH + 18;
  const H = axisY + 22;
  const x = b => left + b * pitch;
  const densMax = Math.max(1e-9, ...blocks.map(b => b.dens_median || 0));
  const explMax = Math.max(1e-9, ...blocks.map(b => EXPL_GRADES.reduce((a, g) => a + ((b.explained || {})[g] || 0), 0)));
  const selMax = Math.max(1e-9, ...(selVals || [0]).map(v => v || 0));
  const detMax = Math.max(1e-9, ...blocks.map(b => (b.types || []).slice(0, 3).reduce((a, c) => a + c, 0)));
  const segs = [];
  blocks.forEach((b, bi) => {
    let acc = 0;
    (b.types || []).slice(0, 3).forEach((cnt, t) => {
      const h = Math.max(1, (cnt / detMax) * trackH);
      segs.push({ track: 0, block: bi, yOff: acc, h: Math.min(h, trackH - acc), fill: TYPE_COLOR[t],
                  title: `${tower.type_names[t]}：${cnt.toLocaleString()} 个（占本块 ${((cnt / tower.d_ff) * 100).toFixed(1)}%）` });
      acc += h;
    });
  });
  blocks.forEach((b, bi) => {
    const v = b.dens_median || 0;
    segs.push({ track: 1, block: bi, yOff: 0, h: Math.max(1.5, (v / densMax) * trackH), fill: '#256abf',
                title: `强激活率中位数：${v ? v.toPrecision(3) : '0'}` });
  });
  blocks.forEach((b, bi) => {
    let acc = 0;
    EXPL_GRADES.forEach(g => {
      const cnt = (b.explained || {})[g] || 0;
      if (!cnt) return;
      const h = (cnt / explMax) * trackH;
      segs.push({ track: 2, block: bi, yOff: acc, h, fill: GRADE_COLOR[g], title: `${EXPL_LABEL[g] || g}：${cnt}（占本块 ${((cnt / tower.d_ff) * 100).toFixed(2)}%）` });
      acc += h;
    });
  });
  if (selVals) {
    blocks.forEach((b, bi) => {
      const v = selVals[bi] || 0;
      if (!v) return;
      const parts = selRuns.map(r => `${r.item || r.tm_id}：${((r.by_readout?.[ro] || [])[bi] || 0).toLocaleString()}`).join('，');
      segs.push({ track: 3, block: bi, yOff: 0, h: Math.max(1.5, (v / selMax) * trackH), fill: '#e34948', title: `入选单元合计：${v.toLocaleString()}（${parts}）` });
    });
  }
  const trackTop = k => k * (trackH + trackGap);
  const labels = [
    ['① 检测器构成', '每块 T1 / T2 / T3 检测器型单元的数量（分色堆叠）'],
    ['② 强激活率中位数', '该块单元整体的激活频繁程度'],
    ['③ 已解释单元', '解释检验覆盖数（深 = 文本可复现）'],
    [selVals ? '④ 模板选择性入选' : '④ 模板选择性', selVals ? '对照句式能区分两个水平的单元（各运行合计）' : '还没有对照实验数据'],
  ];
  const ticks = [0, 6, 12, 18, 24, 30, n - 1].filter(b => b < n);
  return (
    <svg className="v8-prof" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="逐块全景剖面图">
      {labels.map(([name, sub], k) => (
        <text key={k} x={left - 10} y={trackTop(k) + trackH / 2 - 3} textAnchor="end" className="v8-prof-lab">
          <tspan x={left - 10} dy="0">{name}</tspan>
          <tspan x={left - 10} dy="13" className="v8-prof-labsub">{sub.length > 18 ? `${sub.slice(0, 17)}…` : sub}</tspan>
        </text>
      ))}
      {!selVals && <text x={left} y={trackTop(3) + trackH / 2} className="v8-prof-labsub">该模型上还没有模板对照运行的入选数据</text>}
      {segs.map((s, i) => (
        <rect key={i} x={x(s.block)} width={colW}
              y={trackTop(s.track) + trackH - s.yOff - s.h} height={s.h} fill={s.fill} rx={1}>
          <title>{`块 ${s.block}：${s.title}`}</title>
        </rect>
      ))}
      {blocks.map((_, b) => (
        <rect key={b} className="v8-prof-col" x={x(b)} y={0} width={pitch} height={axisY - 16} onClick={() => onPick?.(b)}>
          <title>{`块 ${b}：点开这一块的神经元阵列`}</title>
        </rect>
      ))}
      {ticks.map(b => (
        <text key={b} x={x(b) + colW / 2} y={axisY} textAnchor="middle" className="v8-prof-tick">{b}</text>
      ))}
      <text x={x(n - 1) + colW + 4} y={axisY} textAnchor="start" className="v8-prof-tick">← 块号（浅层 → 深层）</text>
    </svg>
  );
}

function NeuronProgress({ census, skeleton, model, available = [], onOpenBlock }) {
  const tw = useApi(census ? `/api/neurons/tower?model_key=${encodeURIComponent(census)}` : null);
  const idx = useApi('/api/neurons/index');
  const tower = tw.data && !tw.data.pending ? tw.data : null;
  const [ro, setRo] = useState('final');
  if (!census) {
    return <Empty>当前模型（{shortModel(model)}）还没有自然语料普查，没有神经元逆向分析数据。
      {available.length ? ` 已有普查的模型：${available.map(shortModel).join('、')}——把页顶「模型」换成它就能看到这一页。` : ' 本机还没有任何模型做过普查。'}</Empty>;
  }
  if (tw.error) return <Empty>层塔数据不在本机数据包里：{tw.error}</Empty>;
  if (!tower) return <Empty>读取层塔…</Empty>;
  const run = (idx.data?.census || []).find(r => r.model_key === census && r.run === tower.run);
  const blocks = tower.blocks;
  const grades = {};
  blocks.forEach(b => Object.entries(b.explained || {}).forEach(([g, n]) => { grades[g] = (grades[g] || 0) + n; }));
  const nExpl = Object.values(grades).reduce((a, b) => a + b, 0);
  const typeSum = k => blocks.reduce((s, b) => s + (b.types ? b.types[k] : 0), 0);
  const selRuns = tower.selectivity || [];
  const selVals = selRuns.length ? blocks.map((_, b) => selRuns.reduce((s, r) => s + ((r.by_readout?.[ro] || [])[b] || 0), 0)) : null;
  const nUnits = tower.d_ff * tower.n_layers;
  const typeTotals = tower.type_names.map((_, k) => typeSum(k));
  const domType = typeTotals.indexOf(Math.max(...typeTotals));
  const domPct = Math.round((typeTotals[domType] / nUnits) * 100);
  const detPct = ((typeSum(0) + typeSum(1)) / nUnits) * 100;
  const explPct = (nExpl / nUnits) * 100;
  const fmtPct = v => (v >= 1 ? `${v.toFixed(1)}%` : `${v.toFixed(2)}%`);
  const peakSel = selVals ? selVals.indexOf(Math.max(...selVals)) : -1;
  const exParts = Object.entries(grades).filter(([, n]) => n > 0).map(([g, n]) => `${EXPL_LABEL[g] || g} ${n}`).join('、');
  return (
    <div className="v8-nprog">
      <p className="v8-nprog-say">
        全部 <b>{tower.n_layers}</b> 块 × <b>{tower.d_ff.toLocaleString()}</b> 个 MLP 中间单元已完成普查与分型（100%），检测器型单元占 <b>{fmtPct(detPct)}</b>，最常见的类型是「{tower.type_names[domType]}」（{domPct}%）。
        {nExpl > 0
          ? <>解释检验已覆盖 <b>{nExpl}</b> 个单元（{exParts}），覆盖率 <b>{fmtPct(explPct)}</b>。</>
          : '解释检验尚未开始。'}
        {selVals
          ? <>模板对照 <b>{selRuns.length}</b> 个运行（如「{selRuns[0].item || selRuns[0].tm_id}」），入选信号最多出现在<b>块 {peakSel}</b>。</>
          : '模板选择性还没有对照运行。'}
      </p>
      <div className="v8-nprog-tiles">
        <div><b>{fmtTok(run?.n_tokens)}</b><span>普查语料规模</span><small>{run ? `${run.sources?.length || 0} 个语料来源 · 每窗口 ${run.seq_len} token` : '—'}</small></div>
        <div><b>{nUnits.toLocaleString()}</b><span>MLP 单元全部有画像</span><small>{tower.n_layers} 块 × {tower.d_ff.toLocaleString()} · 摘要 {tower.summaries_ready}/{tower.n_layers} 已生成</small></div>
        <div><b>{fmtPct(detPct)}</b><span>检测器型单元（T1 + T2）</span><small>T1 单 token {typeSum(0).toLocaleString()} · T2 上下文 {typeSum(1).toLocaleString()} · 按 L30-v1 规则分型</small></div>
        <div><b>{fmtPct(explPct)}</b><span>解释检验覆盖率</span><small>{nExpl.toLocaleString()} 个单元：{exParts || '尚未开始'} · 分母 = 全部 {nUnits.toLocaleString()} 个单元</small></div>
      </div>
      <div>
        <h4 className="v8-h4">全景剖面 <small>一列 = 一个块 · 悬停看数值 · 点击进入该块</small></h4>
        {selRuns.length > 0 && (
          <div className="v8-tools">
            <span className="v8-seg">{['final', 'probe_answer'].map(x => (
              <button key={x} type="button" className={ro === x ? 'on' : ''} onClick={() => setRo(x)}>在「{READOUT_LABEL[x] || x}」位置读出</button>
            ))}</span>
          </div>
        )}
        <ProfileChart tower={tower} selVals={selVals} selRuns={selRuns} ro={ro} onPick={onOpenBlock} />
        <div className="v8-prof-legend">
          {tower.type_names.map((t, k) => <span key={t}><i style={{ background: TYPE_COLOR[k] }} />{t}</span>)}
          <span className="v8-prof-lg-sep" />
          {EXPL_GRADES.map(g => <span key={g}><i style={{ background: GRADE_COLOR[g] }} />{EXPL_LABEL[g] || g}</span>)}
          {selVals && <span><i style={{ background: '#e34948' }} />选择性入选</span>}
        </div>
      </div>
      <p className="v8-note">边界：分型是描述性分类，不是证据等级；注意力头逐头分析、跨层计算边（归因图）与多数单元的因果检验尚未覆盖——控制轴结论见各单元卡的「干预记录」。{skeleton?.model ? ` 模型：${shortModel(skeleton.model)}。` : ''}</p>
    </div>
  );
}

const BENCH_FIRST = ['知识网络', '功能推理', '语法规则'];

/* 图谱测试题（公开数据集实测）：数据集决定考哪批题，模型决定谁去考。
   overview 一次返回数据集清单与按 (数据集, 模型) 分键的结果表，这里只做筛选展示——
   旧版不管上面选哪个模型、哪个数据集，都显示同一份混跑结果，换一次模型看到的还是上一个模型的百分比。
   运行前置条件是「页面选的模型 = 本机已加载的权重」：图谱测试题只能测已加载的那个模型，不装第二份权重。 */
function LangBench({ dsId, dsRow, model }) {
  const ov = useApi('/api/lang/bench/overview');
  const [st, setSt] = useState(null);          // 运行状态（/status 轮询结果）
  const [busy, setBusy] = useState(false);
  const [runErr, setRunErr] = useState('');
  const timer = useRef(null);
  const active = st && (st.state === 'loading' || st.state === 'running');

  /* 轮询运行状态：active 时每 1.5 秒读一次 /status，直到 done / error */
  useEffect(() => {
    if (!active) return undefined;
    let dead = false;
    const tick = () => {
      fetch(`${API_BASE}/api/lang/bench/status`, { cache: 'no-store' })
        .then(r => r.json())
        .then(d => { if (!dead) setSt(d); })
        .catch(() => {});
    };
    tick();
    timer.current = window.setInterval(tick, 1500);
    return () => { dead = true; window.clearInterval(timer.current); };
  }, [active]);

  if (ov.loading) return null;
  if (ov.error) return <Empty>图谱测试题数据读取失败：{ov.error}</Empty>;
  const data = ov.data;
  if (!data || !data.available) {
    return <p className="v8-note">本机数据包里还没有图谱测试题（data/lang_bench/）：在仓库根目录运行 scripts/download_lang_bench.py 下载题库后显示。</p>;
  }

  const tpl = dsId === 'TPL';
  const prefix = dsPrefix(dsId);
  const row = (data.datasets || []).find(d => d.prefix === prefix) || null;
  const nQ = row?.n_questions || 0;
  /* 可映射三类的题数：在线端点给 graph_n；离线数据包没有题库正文（null），退回数据集清单里的 testable */
  const graphN = row?.graph_n == null ? (dsRow?.testableN ?? null) : row.graph_n;
  const byCat = row?.by_category || {};
  const results = data.results || {};
  const hitKey = Object.keys(results).find(k => k.startsWith(`${prefix}|`) && sameModel(model, results[k]?.model_name));
  const running = active && dsPrefix(st?.dataset) === prefix;
  const result = running ? null
    : st?.state === 'done' && st.result && dsPrefix(st.result.dataset) === prefix && sameModel(model, st.result.model_name) ? st.result
      : hitKey ? results[hitKey] : null;
  const modelOk = sameModel(model, data.loaded_model);
  const canRun = data.model_status !== 'unavailable' && !tpl && nQ > 0 && modelOk;
  const pct = v => `${(v * 100).toFixed(1)}%`;
  const fmtPct = v => (v * 100 >= 1 ? `${(v * 100).toFixed(1)}%` : `${(v * 100).toFixed(2)}%`);
  const etaText = nQ >= 1000 ? `预计 ${Math.max(1, Math.round(nQ * 0.0024))} 分钟` : '预计几分钟';

  const run = async () => {
    setBusy(true); setRunErr('');
    try {
      const r = await postJson('/api/lang/bench/run', { dataset: dsId, model });
      if (r.accepted) setSt({ state: 'loading', done: 0, total: r.total || nQ, dataset: dsId });
      else setRunErr(r.reason || '未能启动实测');
    } catch (e) { setRunErr(e.message || String(e)); } finally { setBusy(false); }
  };

  const cats = (result?.by_category) || (running ? st.by_category : null) || {};
  const benchCats = [
    ...BENCH_FIRST,
    ...Object.keys(byCat || {}).filter(c => !BENCH_FIRST.includes(c)),
  ].filter(c => (byCat || {})[c] || cats[c]);
  const overall = result ? { correct: result.correct, n: result.n_questions, acc: result.acc } : null;
  const srcLabel = row?.by_source ? Object.keys(row.by_source).join(' + ') : dsLabel(dsId);

  if (tpl) {
    return (
      <div className="v8-bench">
        <h4 className="v8-h4" style={{ margin: 0 }}>图谱测试题 <small>生成式模板族不参与这一栏</small></h4>
        <p className="v8-note">模板族（ENT-XF / GR-NEG / GR-Q / GR-ROLE / KN-ATTR / KN-ISA）是生成式对照材料，不是选择题，
          没有「正确率」这个口径。它的进展看下方每条语言规律的读出等级。要跑题库实测，把上方的数据集换成 C-Eval / MMLU / M3Exam。</p>
      </div>
    );
  }
  if (!nQ) {
    return (
      <div className="v8-bench">
        <h4 className="v8-h4" style={{ margin: 0 }}>图谱测试题 <small>{dsLabel(dsId) || '未选数据集'}</small></h4>
        <p className="v8-note">题库里没有这个数据集的题。当前题库（questions.v3）只含 C-Eval val / MMLU test / M3Exam zh test；
          CMMLU 与 CMExam 在旧题库里，已退出主口径。</p>
      </div>
    );
  }

  return (
    <div className="v8-bench">
      <div className="v8-bench-head">
        <h4 className="v8-h4" style={{ margin: 0 }}>图谱测试题</h4>
        {canRun ? (
          <button type="button" className="v8-bench-btn" disabled={busy || active} onClick={run}>
            {running ? `实测中 ${st.done}/${st.total}` : result ? '重新实测' : '▶ 运行实测'}
          </button>
        ) : (
          <span className="v8-note">
            {data.model_status === 'unavailable' ? '中心不做 GPU 计算：只保存数据与转发命令，实测在本机执行'
              : !modelOk ? `本机当前加载的是 ${data.loaded_model || '未知模型'}，不是 ${shortModel(model)}：图谱测试题只测已加载权重`
                : '该数据集没有可跑实测的题'}
          </span>
        )}
      </div>
      <p className="v8-bench-sub">
        {srcLabel} · {nQ.toLocaleString()} 道选择题
        {graphN != null ? ` · 按学科可映射三类 ${graphN.toLocaleString()} 道` : ''}
        {benchCats.length ? ` · 三类：${benchCats.map(c => `${c} ${((byCat || {})[c] || 0).toLocaleString()}`).join(' / ')}` : ''}
        {row?.n_unmapped ? ` · 其余 ${row.n_unmapped.toLocaleString()} 道不能映射三类（只进总正确率）` : ''}
      </p>
      {/* 题目从哪来：正文不随包分发（CC BY-NC-SA / MIT 原样引用），这里给原始数据地址与许可；
          中心只有题量、科目、三类分布这些元信息，正文按这个地址在本机取。 */}
      {row?.source_url && (
        <p className="v8-bench-src">
          原始数据 <a href={row.source_url} target="_blank" rel="noreferrer">{row.source_repo} ↗</a>
          {row.source_split ? ` · ${row.source_split}` : ''}
          {row.license ? ` · ${row.license}` : ''}
          {row.stats_source === 'meta' ? ' · 题量为建库时统计（本处不含题目正文）' : ''}
        </p>
      )}
      {graphN === 0 ? (
        <p className="v8-note">这个数据集没有能映射到知识网络 / 功能推理 / 语法规则的题
          （题库给它们标的类别是「{Object.keys(byCat)[0] || '题材'}」），只能给总正确率，回答不了三类图谱的进展。</p>
      ) : benchCats.length > 0 && !benchCats.some(c => BENCH_FIRST.includes(c)) ? (
        <p className="v8-note">题库把这一批题标为「{benchCats.join(' / ')}」——那是数据集自带的题材分类，不是三类图谱。
          按学科能映射到三类的有 <b>{graphN.toLocaleString()}</b> 道，但实测结果的分类口径仍是题材，
          要拿到「知识 / 推理 / 语法」三条正确率，还需要按学科映射重算（尚未做）。</p>
      ) : null}
      {running && (
        <div className="v8-bench-runbar">
          <div className="v8-bench-runtrack"><i style={{ width: `${st.total ? Math.round((st.done / st.total) * 100) : 0}%` }} /></div>
          <span>逐题前向中：{st.current_subject || '启动…'} · 已用 {Math.round(st.elapsed_s || 0)} 秒 · {etaText}</span>
        </div>
      )}
      {runErr && <p className="v8-note" style={{ color: '#c0392b' }}>{runErr}</p>}
      {st?.state === 'error' && <p className="v8-note" style={{ color: '#c0392b' }}>上一轮实测失败：{st.error}</p>}
      {overall ? (
        <>
          <p className="v8-nprog-say">
            在 <b>{result.model_name || shortModel(model)}</b> 上实测 <b>{overall.n}</b> 题（{dsLabel(dsId)}），答对 <b>{overall.correct}</b> 题，总正确率 <b>{pct(overall.acc)}</b>
            （{result.finished_at} · 用时 {Math.round(result.elapsed_s)} 秒）。
          </p>
          <div className="v8-nprog-cats">
            {benchCats.map((c, i) => {
              const v = cats[c];
              if (!v) return <div key={c}><b>{c}</b><span>—</span></div>;
              return (
                <div key={c} style={{ borderLeftColor: BENCH_COLOR[i % BENCH_COLOR.length] }}>
                  <b>{c}</b>
                  <span className="v8-bench-acc">{fmtPct(v.acc)}</span>
                  <small>答对 {v.correct} / {v.n} 题</small>
                </div>
              );
            })}
          </div>
          <div className="v8-prog-list" style={{ marginTop: 10 }}>
            <div>
              <div className="v8-prog-catname">分科目正确率</div>
              {Object.entries(result.by_subject || {}).map(([name, v]) => (
                <div key={name} className="v8-prog-row" style={{ cursor: 'default' }}>
                  <span className="v8-prog-name">{name}</span>
                  <span className="v8-prog-track"><i style={{ width: `${Math.round(v.acc * 100)}%`, background: '#2a78d6' }} /></span>
                  <b className="v8-prog-pct">{pct(v.acc)}</b>
                  <span className="v8-prog-lab">答对 {v.correct}/{v.n}</span>
                </div>
              ))}
            </div>
          </div>
          <p className="v8-note">评测方式：零样本，每题一次前向，在「答案：」位置比较各选项字母的 next-token 概率
            （与官方 loglikelihood 评分同前缀等价）。结果按（数据集, 模型）分开存，换数据集或换模型不会互相覆盖。</p>
        </>
      ) : (
        <p className="v8-note">{running ? '各大类正确率将随实测进度实时出现。'
          : data.model_status === 'unavailable'
            /* 中心没有 GPU：这里不能指着一个不存在的按钮说「点它」。给出真正能做的事：
               在本机跑（本机后端有 /run），跑完的结果按 (数据集, 模型) 分键上传，这里就会显示。 */
            ? `还没有「${dsLabel(dsId)} × ${shortModel(model)}」的实测结果：中心不做 GPU 计算，实测要在本机后端执行`
              + `（${etaText}），跑完按「数据集 | 模型」分键上传后这里显示真实百分比。`
              + (data.legacy ? `（库里另有一份旧口径结果：${data.legacy.n_questions.toLocaleString()} 题混跑、${data.legacy.model_name}、${data.legacy.finished_at}，
                当时没按数据集与模型分键，所以不放到这里冒充本选择的进展。）` : '')
            : `还没有「${dsLabel(dsId)} × ${shortModel(model)}」的实测结果：点「运行实测」逐题打分，${etaText}出真实百分比。`
              + (data.legacy ? `（本机另有一份旧口径结果：${data.legacy.n_questions.toLocaleString()} 题混跑、${data.legacy.model_name}、${data.legacy.finished_at}，
                当时没按数据集与模型分键，所以不放到这里冒充本选择的进展。）` : '')}</p>
      )}
    </div>
  );
}

function LangProgress({ tree, model, dsId, dsRow, onOpenItem }) {
  const models = tree.models || [];
  const lv = it => itemLevel(it, model);
  const gaps = tree.items.filter(it => lv(it) === 0);
  const nHas = tree.items.filter(it => lv(it) >= 1).length;
  const nHeld = tree.items.filter(it => lv(it) >= 3).length;
  const nItems = tree.items.length || 1;
  const pct = n => `${Math.round((n / nItems) * 100)}%`;
  /* 大类置顶排序：知识网络、功能推理、语法规则在最前，其余按原顺序跟随 */
  const CAT_FIRST = ['知识网络', '功能推理', '语法规则'];
  const orderedTypes = [
    ...CAT_FIRST.map(id => tree.types.find(t => t.id === id || t.name === id)).filter(Boolean),
    ...tree.types.filter(t => !CAT_FIRST.includes(t.id) && !CAT_FIRST.includes(t.name)),
  ];
  const CAT_COLOR = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#7a3fbf', '#008300'];
  const catItems = t => tree.items.filter(it => it.type === t.id);
  return (
    <div className="v8-nprog">
      <LangBench dsId={dsId} dsRow={dsRow} model={model} />
      <p className="v8-nprog-say">
        共 <b>{tree.items.length}</b> 条语言规律，分属 <b>{tree.types.length}</b> 个大类；在 {shortModel(model)} 上有数据 <b>{nHas}</b> 条（{pct(nHas)}），其中留出复现及以上 <b>{nHeld}</b> 条（{pct(nHeld)}）。
        {gaps.length ? <>还没有数据 <b>{gaps.length}</b> 条（{pct(gaps.length)}），见下方缺口。</> : '每个语言项都已有数据。'}
      </p>
      <div className="v8-nprog-cats">
        {orderedTypes.map((t, i) => {
          const its = catItems(t);
          const has = its.filter(it => lv(it) >= 1).length;
          return (
            <div key={t.id} style={{ borderLeftColor: CAT_COLOR[i % CAT_COLOR.length] }}>
              <b>{t.name}</b>
              <span>{its.length} 条 · {has} 条有数据</span>
              <small>{its.length - has ? `${its.length - has} 条还没有数据` : '全部有数据'}</small>
            </div>
          );
        })}
      </div>
      <div className="v8-nprog-tiles">
        <div><b>{tree.items.length}</b><span>条语言规律</span><small>分属 {tree.types.length} 个大类 · 全部来自模板对照实验</small></div>
        <div><b>{nHas}</b><span>有数据 · 占 {pct(nHas)}</span><small>至少完成过一次读出测量</small></div>
        <div><b>{nHeld}</b><span>留出复现及以上 · 占 {pct(nHeld)}</span><small>证据最强：新数据上也读得出来</small></div>
        <div><b>{gaps.length}</b><span>还没有数据 · 占 {pct(gaps.length)}</span><small>{gaps.length ? '见下方研究缺口，可点进条目看怎么补' : '当前模型上每个语言项都已有数据'}</small></div>
      </div>
      <div>
        <h4 className="v8-h4">完整列表：逐条进度 <small>进度 = 等级 ÷ 4（每级 25%）；平均 {Math.round((tree.items.reduce((s, it) => s + lv(it), 0) / nItems) / 4 * 100)}%；点行进入该条目的拼图</small></h4>
        <div className="v8-prog-list">
          {orderedTypes.map((t, ci) => {
            const its = catItems(t);
            return (
              <div key={t.id}>
                <div className="v8-prog-catname" style={{ borderLeftColor: CAT_COLOR[ci % CAT_COLOR.length] }}>{t.name}</div>
                {its.map(it => {
                  const l = lv(it);
                  const p = (l / 4) * 100;
                  const meta = levelMeta(l);
                  return (
                    <button key={it.id} type="button" className="v8-prog-row" onClick={() => onOpenItem(it)}>
                      <span className="v8-prog-name">{it.name}</span>
                      <span className="v8-prog-track"><i style={{ width: `${p}%`, background: meta.color }} /></span>
                      <b className="v8-prog-pct">{p}%</b>
                      <span className="v8-prog-lab">{meta.sym} {meta.label}</span>
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
        <div className="v8-prof-legend" style={{ marginTop: 10 }}>
          {LEVELS.map(l => <span key={l.level}><i style={{ background: l.color }} />{l.level * 25}% {l.label}</span>)}
        </div>
      </div>
      <div>
        <h4 className="v8-h4">语言项 × 模型明细 <small>每格一个等级（悬停看全称）；各模型分开记录，方向不跨模型比较</small></h4>
        <div className="v8-levels-legend">
          {LEVELS.slice(1).map(l => <LevelChip key={l.level} level={l.level} />)}
          <LevelChip none="○ 未研究" />
        </div>
        <div className="v8-matrix-wrap">
          <table className="v6-mini-table v8-matrix">
            <thead><tr><th>语言项</th>{models.map(m => <th key={m} title={m}>{shortModel(m)}</th>)}</tr></thead>
            <tbody>
              {orderedTypes.map(t => tree.items.filter(it => it.type === t.id).map((it, i) => (
                <tr key={it.id} className={i === 0 ? 'first' : ''}>
                  <th><button type="button" className="v6-linkbtn" onClick={() => onOpenItem(it)}>{it.name}</button><small>{t.name}</small></th>
                  {models.map(m => {
                    const l = it.status?.by_model?.[m] ?? 0;
                    const meta = levelMeta(l);
                    return <td key={m} title={`${it.name} · ${shortModel(m)}：${meta.label}`}><span className="v8-chip" style={{ background: meta.color, color: meta.ink }}>{meta.sym}</span></td>;
                  })}
                </tr>
              )))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="v8-note">{tree.policy}</p>
    </div>
  );
}

/* 当前口径卡：顶部全局条负责「选」，这一条负责「解释」——指标各自被哪一维驱动。
   不写笼统的「全部进展」：神经元页只有模型维度（普查读自然语料，不是题库），规则账本两个维度都不参与
   （按规则记录，不按题库/模型分列），语言图谱页只有「图谱测试题」跟着数据集变，等级矩阵只跟模型变。
   页顶的数据集方块同样按这条口径显隐（workbenchUtil.dsBound），这里只说「这一页读什么」，
   所以神经元页与规则账本不再列「数据集：不参与」那一格——没有可选项，就没有要解释的空维度。 */
function ScopeCard({ tab, model, census, dsId, dsRow, dsTestable, dsError }) {
  const tpl = dsId === 'TPL';
  const dsText = !dsId ? '未选（自动）'
    : tpl ? '生成式模板族'
      : `${dsLabel(dsId)} · ${(dsRow?.n_questions || 0).toLocaleString()} 题${dsTestable ? ` · 可映射三类 ${dsTestable.toLocaleString()}` : ' · 无可映射三类科目'}`;
  const isNeuron = tab === 'neuron';
  const isRules = tab === 'rules';
  const modelText = isNeuron
    ? (census ? '有自然语料普查：逐单元画像可用' : '还没有自然语料普查：本页只有结构，没有单元数据')
    : isRules ? '决定账本里哪些结论已有本模型的证据'
      : '决定等级矩阵的列与图谱测试题的实测对象';
  const dsText2 = tpl
    ? '不参与：模板族是生成式对照材料，不是选择题，进不了正确率口径'
    : '只影响「图谱测试题」：题目从这个数据集取；等级矩阵不受影响';
  return (
    <div className="v8-scope">
      <div className="v8-scope-item">
        <span className="v8-scope-k">模型</span>
        <b>{shortModel(model) || '—'}</b>
        <span className="v8-scope-v">{modelText}</span>
      </div>
      {!isNeuron && !isRules && (
        <div className="v8-scope-item">
          <span className="v8-scope-k">数据集</span>
          <b>{dsText}</b>
          <span className="v8-scope-v">{dsText2}</span>
        </div>
      )}
      <p className="v8-scope-note">
        {isNeuron
          ? '换模型 → 整页数字全变。本页读自然语料普查（1,900 万 token），不读题库，所以页顶没有数据集可选；神经元进展与语言图谱进展读的不是同一批数据，不能相加、不能拼成转化率。'
          : isRules
            ? '账本记的是「这条规则被试过、结论是什么」，不按题库分组，所以页顶没有数据集可选；换模型时变的是证据列。'
            : '换模型 → 等级矩阵与实测都变；换数据集 → 只有正确率变。两类进展分开记录。'}
        {dsError ? ` 数据集清单读取失败：${dsError}` : ''}
      </p>
    </div>
  );
}

/* 页签由工作台顶部的子导航（workbenchUtil.ROUTE_VIEWS）提供，这里只按 tab 渲染内容：
   页签与空间/研发同一层级、同一组件，避免同一页面出现两种页签样式。
   模型与数据集由页顶的全局条驱动（props model / ds，来自地址栏 mdl / ds）：
   这一页此前用 defaultModel() 自己挑一个模型，于是页顶换模型时进展数字不动——那是个真缺陷。 */
export default function RouteLens({ tab = 'neuron', model: gModel, ds = '', onGoSpace, onGo }) {
  const treeApi = useApi('/api/lang/tree');
  const skApi = useApi('/api/model/skeleton');
  const dsApi = useApi('/api/lang/atlas/datasets');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = useMemo(() => Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s])), [skApi.data]);
  const model = (gModel && tree?.models?.includes(gModel) ? gModel : '') || defaultModel(tree, skeletons);
  const skeleton = skeletons[model];
  const census = skeleton?.census_run ? skeleton.census_key : null;
  const censused = Object.values(skeletons).filter(s => s.census_run).map(s => s.model);

  /* 数据集：URL 里没选就取第一个可测的（与空间页同一口径），模板族（TPL）单独对待 */
  const dsList = (dsApi.data?.datasets || []).filter(d => d.n_questions > 0);
  const dsPicked = dsList.find(d => d.id === ds) || dsList.find(d => d.testable) || (ds === 'TPL' ? null : dsList[0]) || null;
  const dsId = ds === 'TPL' ? 'TPL' : (dsPicked?.id || '');
  const dsTestable = dsPicked ? Object.values(dsPicked.by_category || {}).reduce((s, c) => s + (c.n || 0), 0) : 0;
  const dsRow = dsPicked ? { n_questions: dsPicked.n_questions, testableN: dsTestable } : null;

  return (
    <div className="v8-route">
      {/* 口径卡只在真读这两维的两页出现：规则账本按「规则」记录，不按模型/数据集分列，
          给它加一条维度说明反而会暗示账本随选择变 */}
      {tab !== 'rules' && (
        <ScopeCard tab={tab} model={model} census={census} dsId={dsId} dsRow={dsRow} dsTestable={dsTestable} dsError={dsApi.error} />
      )}
      {tab === 'neuron' && (
        <Card title="神经元逆向分析进展" sub={`${shortModel(model)} · 普查 → 分型 → 解释检验 → 模板选择性（各来源分开计数，不拼成转化率）`}>
          <NeuronProgress census={census} skeleton={skeleton} model={model} available={censused}
                          onOpenBlock={b => onGoSpace('neuron', { block: b, unit: null })} />
        </Card>
      )}
      {tab === 'lang' && (
        <>
          <Card title="语言图谱进展" sub={`${shortModel(model)} · ${dsId === 'TPL' ? '生成式模板族' : dsLabel(dsId)} 实测正确率 + 每条语言规律的读出等级（模板对照实验）`}>
            {tree ? <LangProgress tree={tree} model={model} dsId={dsId} dsRow={dsRow} onOpenItem={it => onGoSpace('lang', { item: it.id, type: it.type })} />
              : <Empty>{treeApi.error ? `语言谱图不在本机数据包里：${treeApi.error}` : '读取语言谱图…'}</Empty>}
          </Card>
          <RouteBoard onGo={onGo} />
        </>
      )}
      {tab === 'rules' && (
        <Card title="规则账本" sub={`${shortModel(model)} · 已测过的规则与结论：先查重，再决定要不要做新一轮实验`}>
          <RuleLedgerPage />
        </Card>
      )}
    </div>
  );
}
