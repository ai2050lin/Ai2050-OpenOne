/* 空间 · 关联机制 · 运行与路径（Phase 332，按 design/mockups/association_redesign_v2_20261107.html 精简版）。
   一页只放一个核心模块：全部层的 3D 空间 + HiddenState 流动画。
   其余功能收在下面一排按钮里，默认收起，点一个只展开一项，再点同一个收起。 */
import { useMemo, useState } from 'react';

import report from '../../../../research/mainline/H2/report.json';
import AssocFlow3D from './AssocFlow3D.jsx';
import { AlgorithmPanel, AnomalyDraft } from './AssocDeep.jsx';
import { AttributionGraph, DEMO_SNAPSHOT, InfoFlowMap } from './ClientGuide.jsx';
import { LineChart } from './H2Results.jsx';
import { AssocIntro } from './SpaceGuides.jsx';
import { FLOW_DATA, rankHeads } from './data/assocFlow.js';
import { PageHead, PillTools } from './CoreKit.jsx';
import './assoc_space.css';

const TOOLS = [['run', '输入文字'], ['lens', '逐层读出'], ['factor', '因素'], ['heads', '注意力头'], ['method', '方法比较'], ['more', '更多工具']];

function Spark({ values, color }) {
  const w = 300, h = 28, mx = Math.max(...values) || 1;
  const d = values.map((v, i) => `${i ? 'L' : 'M'}${((i / (values.length - 1)) * w).toFixed(1)},${(h - 2 - (v / mx) * (h - 4)).toFixed(1)}`).join('');
  return <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true"><path d={d} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" /></svg>;
}

export default function AssocSpace({ model, children }) {
  const [cur, setCur] = useState(0);
  const [facOn, setFacOn] = useState(() => Object.fromEntries(FLOW_DATA.factors.map(([k]) => [k, true])));
  const [selHead, setSelHead] = useState(null);
  const [allHeads, setAllHeads] = useState(false);
  const top = useMemo(() => rankHeads(), []);
  const ci = Math.round(cur);
  const pct = v => `${Math.round(v * 100)}%`;

  const flow = <AssocFlow3D data={FLOW_DATA} cur={cur} onCur={setCur} facOn={facOn} selHead={selHead} />;
  return <div className="v8-route as-page" aria-label="关联机制">
    <PageHead title="关联机制 · 运行与路径" sub="一段文字在各层怎样被读取、改写并决定下一个词。上面是全部层的 HiddenState 流，下面每一行点开看细节。" />
    {flow}
    <PillTools label="关联机制的更多内容" items={TOOLS.map(([id, label]) => ({ id, label, render: () => <>
      {id === 'run' && <>
        {children}
        <details><summary>三种运行有什么区别</summary><AssocIntro /></details>
      </>}

      {id === 'lens' && <>
        <LineChart label="逐层读出：资料色减常识色" yLabel="资料色 − 常识色（logit），阴影为分歧区间 24–33" band={[24, 33]}
          series={[{ name: '按资料', color: '#2a78d6', values: FLOW_DATA.lensDoc }, { name: '按现实', color: '#eb6834', values: FLOW_DATA.lensReal }]} />
        <p className="v8-note">当前第 {ci} 层：按资料 {FLOW_DATA.lensDoc[ci].toFixed(1)}，按现实 {FLOW_DATA.lensReal[ci].toFixed(1)}。
          最终读对：按资料 {pct(report.lens.final_correct_share.doc)}，按现实 {pct(report.lens.final_correct_share.real)}。</p>
        <details><summary>怎么算的</summary><p className="v8-note">每层先做最终归一化再接 unembedding，比较资料色与常识色的 logit。这是读出近似，早层不可靠；预注册的 R1 / R2 判据对噪声敏感，不可用。</p></details>
      </>}

      {id === 'factor' && <>
        {FLOW_DATA.factors.map(([key, label, color, values]) => { const pk = values.indexOf(Math.max(...values));
          return <div className="as-frow" key={key}>
            <label><input type="checkbox" checked={facOn[key]} onChange={e => setFacOn(f => ({ ...f, [key]: e.target.checked }))} /> {label}</label>
            <Spark values={values} color={color} />
            <span className="v8-note">峰值第 {pk} 层 {values[pk].toFixed(2)}</span></div>; })}
        <p className="v8-note">勾选决定 3D 里显示哪几条线。曲线是“问题中的目标名”位置的差异幅度（std_rms），不是因果作用。</p>
      </>}

      {id === 'heads' && <>
        <div className="v8-weight-scroll"><table className="v6-table">
          <thead><tr><th>头</th><th>按资料</th><th>按现实</th><th>状态</th></tr></thead>
          <tbody>{top.slice(0, allHeads ? top.length : 5).map(x => <tr key={`${x.l}-${x.h}`} className={selHead === x ? 'as-sel' : ''}
            tabIndex={0} onClick={() => { setSelHead(x); setCur(x.l + 1); }} onKeyDown={e => { if (e.key === 'Enter') { setSelHead(x); setCur(x.l + 1); } }}>
            <td>L{x.l}H{x.h}</td><td>{x.d.toFixed(2)}</td><td>{x.r.toFixed(2)}</td><td><span className="mp-tag">候选 · 未干预</span></td></tr>)}</tbody>
        </table></div>
        <div className="as-row">{!allHeads && <button type="button" className="fw-tbtn" onClick={() => setAllHeads(true)}>显示全部 {top.length} 个</button>}
          <span className="v8-note">点一行跳到该层并高亮。注意力去向不等于计算使用，因果要到 H4 的路径干预。</span></div>
      </>}

      {id === 'method' && <AlgorithmPanel />}

      {id === 'more' && <>
        <AnomalyDraft />
        <details><summary>各模式族的信息在流中何时可读</summary><InfoFlowMap model={model} /></details>
        <details><summary>演示：三句话的归因（教学）</summary><AttributionGraph snapshot={DEMO_SNAPSHOT} /></details>
      </>}
    </> }))} />
  </div>;
}
