/* 行为广度题库（C-Eval val + CMMLU test + CMExam 抽样，本机 data/lang_bench/）：各大类与科目的正确率。
   对本机模型逐题一次前向，在「答案：」位置比较各选项字母的 next-token 概率；正确率是真实百分比。
   运行与轮询只在能写的形态（ADMIN_LIVE：桌面客户端或本机开发）；其他形态只显示已有结果。
   考试题只做行为广度与消融后的分科退化画像，不作机制材料（选项字母格式会污染读出）。原 v8/RouteLens.jsx 的 LangBench。 */
import { useEffect, useState } from 'react';

import { ADMIN_LIVE, getJson, postJson, useApi } from '../rdc_fusion/api.js';
import { Empty, seqColor } from '../rdc_fusion/ui.jsx';

const FIRST = ['知识网络', '功能推理', '语法规则'];
const pct = v => `${(v * 100).toFixed(1)}%`;

export default function BenchCard() {
  const ov = useApi('/api/lang/bench/overview');
  const [st, setSt] = useState(null);
  const [busy, setBusy] = useState(false);
  const [runErr, setRunErr] = useState('');
  const active = st && (st.state === 'loading' || st.state === 'running');

  useEffect(() => {
    if (!active) return undefined;
    let dead = false;
    const tick = () => getJson('/api/lang/bench/status', { online: true }).then(d => { if (!dead) setSt(d); }).catch(() => {});
    tick();
    const timer = window.setInterval(tick, 1500);
    return () => { dead = true; window.clearInterval(timer); };
  }, [active]);

  if (ov.loading) return <Empty>读取…</Empty>;
  if (ov.error) return <Empty>未接入：题库数据不在本机数据包里（{ov.error}）。</Empty>;
  const data = ov.data;
  if (!data?.available) return <Empty>未接入：本机还没有题库（data/lang_bench/）。在仓库根目录运行 scripts/download_lang_bench_large.py 后显示。</Empty>;

  const result = active ? null : (st?.state === 'done' && st.result ? st.result : data.last_result);
  const nQ = data.meta?.n_questions || 0;
  const cats = result?.by_category || (active ? st.by_category : null) || {};
  const order = [...FIRST, ...Object.keys(data.meta?.by_category || {}).filter(c => !FIRST.includes(c))]
    .filter(c => (data.meta?.by_category || {})[c] || cats[c]);
  const src = data.meta?.by_source ? Object.keys(data.meta.by_source).join(' + ') : 'C-Eval';
  const run = async () => {
    setBusy(true); setRunErr('');
    try {
      const r = await postJson('/api/lang/bench/run', {});
      if (r.accepted) setSt({ state: 'loading', done: 0, total: nQ });
      else setRunErr(r.reason || '未能启动实测');
    } catch (e) { setRunErr(e.message || String(e)); } finally { setBusy(false); }
  };

  return (
    <>
      <div className="wb9-toolbar">
        <span>{src} · {nQ.toLocaleString()} 道选择题</span>
        {ADMIN_LIVE && <button type="button" className="fw-tbtn" disabled={busy || active} onClick={run}>
          {active ? `实测中 ${st.done}/${st.total}` : result ? '重新实测' : '▶ 运行实测'}</button>}
      </div>
      {runErr && <p className="v6-note warn">{runErr}</p>}
      {st?.state === 'error' && <p className="v6-note warn">上一轮实测失败：{st.error}</p>}
      {result ? (
        <>
          <p className="v6-note">在 {result.model_name || '本机模型'} 上实测 {result.n_questions} 题，答对 {result.correct} 题，总正确率 <b>{pct(result.acc)}</b>（{result.finished_at}）。</p>
          <div className="wb9-tiles">
            {order.map(c => {
              const v = cats[c];
              return <div key={c}><b>{v ? pct(v.acc) : '—'}</b><span>{c}</span><small>{v ? `答对 ${v.correct} / ${v.n}` : '没有题目'}</small></div>;
            })}
          </div>
          <details className="wb9-details">
            <summary>分科目正确率（{Object.keys(result.by_subject || {}).length} 个科目）</summary>
            <table className="wb9-table">
              <thead><tr><th>科目</th><th>正确率</th><th>答对 / 题数</th></tr></thead>
              <tbody>{Object.entries(result.by_subject || {}).map(([name, v]) => (
                <tr key={name}><td>{name}</td>
                  <td><span className="wb9-minibar"><i style={{ width: `${Math.round(v.acc * 100)}%`, background: seqColor(0.75) }} /></span>{pct(v.acc)}</td>
                  <td>{v.correct} / {v.n}</td></tr>
              ))}</tbody>
            </table>
          </details>
          <p className="v6-note">零样本，每题一次前向，比较 A–E 各选项字母的 next-token 概率。考试题只看行为广度，不作机制材料。</p>
        </>
      ) : <Empty>{active ? '各大类正确率将随实测进度出现。' : '还没有实测结果。'}</Empty>}
    </>
  );
}
