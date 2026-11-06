/* 研究工作台 · 进展：内容按 tab 渲染（页签本身在 Workbench 的顶部子导航，
   与空间/研发共用同一个 SubNav 组件与 .wb-subnav 样式）。
   神经元逆向分析进展：数据派生的一句话进展 + 全景剖面图（四条轨道共用同一横轴：分型构成 / 强激活率 /
   已解释 / 模板选择性，一列 = 一个块，点击进入该块）。
   语言图谱覆盖：按模型显示已有数据、留出复现和待补条目，题库实测与矩阵按需展开。
   各来源分开计数，不拼成转化率；全部读取来自本机数据包。 */
import { Fragment, useEffect, useMemo, useRef, useState } from 'react';

import { publishResearchContext } from '../app/researchContext.js';
import RouteBoard from '../app/rdc_fusion/RouteBoard.jsx';
import { API_BASE, postJson, useApi } from '../app/rdc_fusion/api.js';
import { Card, Empty } from '../app/rdc_fusion/ui.jsx';
import '../app/workbench.css';
import { LevelChip } from './LangPage.jsx';
import LanguageCatalog from './LanguageCatalog.jsx';
import RuleLedgerPage from './RuleLedgerPage.jsx';
import ResearchOverview from './ResearchOverview.jsx';
import ResearchLibrary from './ResearchLibrary.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
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

  if (ov.loading) return <p className="v8-note">读取题库与实测记录…</p>;
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
          可从上方覆盖卡片查看逐项读出等级。题库实测使用选择题数据集。</p>
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
          <details>
            <summary>分科目正确率与评测方法</summary>
            <div className="v8-prog-list" style={{ marginTop: 10 }}>
              {Object.entries(result.by_subject || {}).map(([name, v]) => (
                <div key={name} className="v8-prog-row" style={{ cursor: 'default' }}>
                  <span className="v8-prog-name">{name}</span>
                  <span className="v8-prog-track"><i style={{ width: `${Math.round(v.acc * 100)}%`, background: '#2a78d6' }} /></span>
                  <b className="v8-prog-pct">{pct(v.acc)}</b>
                  <span className="v8-prog-lab">答对 {v.correct}/{v.n}</span>
                </div>
              ))}
            </div>
            <p className="v8-note">零样本评测，每题一次前向，在「答案：」位置比较各选项字母的 next-token 概率。结果按数据集与模型分别保存。</p>
          </details>
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

function DatasetBench({ dataset, model, onDataset }) {
  const catalog = useApi('/api/lang/atlas/datasets');
  const options = (catalog.data?.datasets || []).filter(entry => entry.n_questions > 0);
  const picked = dataset ? options.find(entry => entry.id === dataset) : options.find(entry => entry.testable) || options[0];
  const dsId = dataset || picked?.id || '';
  const testableN = picked ? Object.values(picked.by_category || {}).reduce((count, entry) => count + (entry.n || 0), 0) : null;
  return <>
    <label className="v8-coverage-model">评测数据集
      <select aria-label="评测数据集" value={dsId} onChange={event => onDataset(event.target.value)}>
        {!dsId && <option value="">选择数据集</option>}
        {dsId && dsId !== 'TPL' && !picked && <option value={dsId}>{dsId}</option>}
        {options.map(entry => <option key={entry.id} value={entry.id} disabled={!entry.testable && entry.id !== dsId}>
          {entry.id} · {entry.n_questions.toLocaleString()} 题{entry.testable ? '' : ' · 无三类对照'}
        </option>)}
        <option value="TPL">本地生成式模板族</option>
      </select>
    </label>
    <p className="v8-note">这里比较模型在所选题库上的作答表现。上方语言项覆盖来自模板对照实验。</p>
    {catalog.error && <p className="v8-note">数据集清单读取失败。{catalog.error}</p>}
    {dsId && <LangBench dsId={dsId} dsRow={testableN == null ? null : { testableN }} model={model} />}
  </>;
}

function CoverageMatrix({ tree, onOpenItem }) {
  return <>
    <div className="v8-levels-legend">
      {LEVELS.slice(1).map(level => <LevelChip key={level.level} level={level.level} />)}
      <LevelChip none="○ 未研究" />
    </div>
    <div className="v8-matrix-wrap">
      <table className="v6-mini-table v8-matrix">
        <thead><tr><th>语言项</th>{tree.models.map(model => <th key={model} title={model}>{shortModel(model)}</th>)}</tr></thead>
        <tbody>
          {tree.types.map(type => tree.items.filter(item => item.type === type.id).map((item, index) => (
            <tr key={item.id} className={index === 0 ? 'first' : ''}>
              <th><button type="button" className="v6-linkbtn" onClick={() => onOpenItem(item)}>{item.name}</button><small>{type.name}</small></th>
              {tree.models.map(model => {
                const meta = levelMeta(itemLevel(item, model));
                return <td key={model} title={model}><span className="v8-chip" style={{ background: meta.color, color: meta.ink }} aria-label={meta.label}>{meta.sym}</span></td>;
              })}
            </tr>
          )))}
        </tbody>
      </table>
    </div>
    <p className="v8-note">{tree.policy}</p>
  </>;
}

function LangCoverage({ tree, treeError, model, dataset, onModel, onDataset, onOpenItem, neuron, onGo }) {
  const [active, setActive] = useState('');
  const models = tree?.models || [];
  const items = tree?.items || [];
  const knownModel = models.includes(model);
  const level = item => itemLevel(item, model);
  const groups = [
    { id: 'measured', label: '已有数据', note: '至少有一次读出测量', items: items.filter(item => level(item) >= 1) },
    { id: 'heldout', label: '留出复现', note: '已有数据中在留出材料上复现的项', items: items.filter(item => level(item) >= 3) },
    { id: 'missing', label: '待补数据', note: '当前模型暂无测量记录的项', items: items.filter(item => level(item) === 0) },
  ];
  return <div className="v8-route v8-overview v8-coverage">
    <header className="v8-overview-heading">
      <div>
        <h3>实验覆盖摘要</h3>
        {tree && <p className="v8-note">按已登记的 {items.length} 个实验条目统计。</p>}
      </div>
      {models.length > 0 && <label className="v8-coverage-model">模型
        <select aria-label="覆盖模型" value={model} onChange={event => onModel(event.target.value)}>
          {!knownModel && <option value={model}>{shortModel(model)} · 暂无覆盖记录</option>}
          {models.map(key => <option key={key} value={key}>{shortModel(key)}{key.split('|')[2] ? ` · ${key.split('|')[2]}` : ''}</option>)}
        </select>
      </label>}
    </header>
    {!tree ? <Empty>{treeError || '读取语言模式目录…'}</Empty> : !knownModel ? <Empty>当前模型暂无图谱覆盖记录。</Empty> :
      <div className="v8-direction-cards v8-overview-cards">
        {groups.map(group => <Fragment key={group.id}>
          <button type="button" className="v8-card v8-direction v8-overview-card" aria-label={group.label}
            aria-expanded={active === group.id} aria-controls={active === group.id ? 'coverage-items' : undefined}
            onClick={() => setActive(active === group.id ? '' : group.id)}>
            <strong>{group.label}</strong>
            <span className="v8-overview-metric v8-coverage-number">{group.items.length} / {items.length} 项</span>
            <span className="v8-note">{group.note}</span>
            <span className="v8-overview-action">{active === group.id ? '收起 ↑' : '查看条目 →'}</span>
          </button>
          {active === group.id && <section id="coverage-items" aria-label={`${group.label}条目`} className="v8-card v8-overview-detail">
            <h3>{group.label} · {group.items.length} 项</h3>
            {group.items.length ? <ul className="v8-coverage-items">
              {group.items.map(item => <li key={item.id}>
                <button type="button" className="v6-linkbtn" onClick={() => onOpenItem(item)}>{item.name}</button>
                <LevelChip level={level(item)} />
              </li>)}
            </ul> : <p className="v8-note">当前模型在这一组没有条目。</p>}
          </section>}
        </Fragment>)}
      </div>}
    <ResearchDisclosure title={`数据集实测${dataset ? ` · ${dataset === 'TPL' ? '本地模板' : dsLabel(dataset)}` : ''}`}>
      <DatasetBench dataset={dataset} model={model} onDataset={onDataset} />
    </ResearchDisclosure>
    {models.length > 0 && <ResearchDisclosure title="跨模型对照">
      <CoverageMatrix tree={tree} onOpenItem={onOpenItem} />
    </ResearchDisclosure>}
    <ResearchDisclosure title="更多研究细节">
      <div className="v8-route">
        <ResearchDisclosure title="神经元普查与解释覆盖">{neuron}</ResearchDisclosure>
        <ResearchDisclosure title="证据门链与预测记分板"><RouteBoard onGo={onGo} /></ResearchDisclosure>
      </div>
    </ResearchDisclosure>
  </div>;
}

function LanguageGraphView({ onOpenMeasurement, coverage }) {
  const [view, setView] = useState('catalog');
  const [coverageVisited, setCoverageVisited] = useState(false);

  return <div className="v8-route">
    <div className="v8-catalog-switch" role="group" aria-label="语言图谱视图">
      <button type="button" className="fw-tbtn" aria-pressed={view === 'catalog'} onClick={() => setView('catalog')}>语言目录</button>
      <button type="button" className="fw-tbtn" aria-pressed={view === 'coverage'} onClick={() => {
        setCoverageVisited(true);
        setView('coverage');
      }}>实验覆盖</button>
    </div>
    <div hidden={view !== 'catalog'}><LanguageCatalog onOpenMeasurement={onOpenMeasurement} /></div>
    {coverageVisited && <div hidden={view !== 'coverage'}>{coverage}</div>}
  </div>;
}

/* 覆盖模型与题库选择仍由工作台地址栏 mdl / ds 持久化。 */
export default function RouteLens({ tab = 'overview', openPane = '', model: gModel, ds = '', onGoSpace, onGo }) {
  const treeApi = useApi(tab === 'rules' ? null : '/api/lang/tree');
  const skApi = useApi(tab === 'rules' ? null : '/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = useMemo(() => Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s])), [skApi.data]);
  const model = gModel || defaultModel(tree, skeletons);
  const censused = Object.values(skeletons).filter(s => s.census_run).map(s => s.model);
  const neuron = key => <NeuronProgress census={skeletons[key]?.census_run ? skeletons[key].census_key : null}
    skeleton={skeletons[key]} model={key} available={censused}
    onOpenBlock={block => onGoSpace('neuron', { block, unit: null }, key)} />;

  if (tab === 'overview') return <ResearchOverview onGo={onGo} tree={tree} model={model} dataset={ds}
    skeletons={skeletons} open={openPane} onOpenUnit={(row, key) => onGoSpace('neuron', { block: row.block, unit: row.unit }, key)}
    renderLanguage={(key, snapshot) => {
      const openItem = item => onGoSpace('lang', { item: item.id, type: item.type }, key);
      return <LanguageGraphView onOpenMeasurement={id => openItem({ id, type: id.split('.')[0] })}
        coverage={<LangCoverage tree={tree} treeError={treeApi.error} model={key} dataset={ds} onGo={onGo}
          onModel={mdl => {
            onGo('route', null, null, { mdl });
            if (snapshot && skeletons[mdl]?.model_snapshot !== snapshot) {
              publishResearchContext({ snapshot: skeletons[mdl]?.model_snapshot || '', run: '', trace: '', context_trace: '', pmodel: '' });
            }
          }}
          onDataset={dataset => onGo('route', null, null, { ds: dataset })}
          onOpenItem={openItem} neuron={neuron(key)} />} />;
    }} />;
  return (
    <div className="v8-route">
      {tab === 'neuron' && <Card title="神经元逆向分析进展" sub={`${shortModel(model)} · 自然语料普查与解释检验`}>{neuron(model)}</Card>}
      {tab === 'rules' && (
        <RuleLedgerPage>
          <ResearchDisclosure title="原始报告与版本记录"><ResearchLibrary /></ResearchDisclosure>
        </RuleLedgerPage>
      )}
    </div>
  );
}
