/* 空间 · 关联机制 · 运行与路径（Phase 332 → 335）。
   上面：全部 37 层的 3D 空间与 HiddenState 信息流（案例 MA-H2 汇总；生成后叠加本次输入的逐层改写幅度）。
   下面：控制台——输入文字生成，选择模式（信息流、逐层读出、注意力头、因素、对照、方法比较）进行观察和分析。 */
import { useMemo, useState } from 'react';

import report from '../../../../research/mainline/H2/report.json';
import AssocConsole from './AssocConsole.jsx';
import AssocFlow3D from './AssocFlow3D.jsx';
import { AnomalyDraft } from './AssocDeep.jsx';
import { AttributionGraph, DEMO_SNAPSHOT, InfoFlowMap } from './ClientGuide.jsx';
import { LineChart } from './H2Results.jsx';
import { AssocIntro } from './SpaceGuides.jsx';
import { FLOW_DATA, rankHeads } from './data/assocFlow.js';
import { CLAIMS, CLAIM_LIFE, LIFECYCLE } from './data/mainline.js';
import { PageHead, PillTools } from './CoreKit.jsx';
import './assoc_space.css';

function Spark({ values, color }) {
  const w = 300, h = 28, mx = Math.max(...values) || 1;
  const d = values.map((v, i) => `${i ? 'L' : 'M'}${((i / (values.length - 1)) * w).toFixed(1)},${(h - 2 - (v / mx) * (h - 4)).toFixed(1)}`).join('');
  return <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true"><path d={d} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" /></svg>;
}

export default function AssocSpace({ model, children, bare = false }) {
  const [cur, setCur] = useState(36);   // 默认显示完整的流；点播放从词嵌入开始
  const [mode, setMode] = useState('flow');
  const [live, setLive] = useState(null);
  const [facOn, setFacOn] = useState(() => Object.fromEntries(FLOW_DATA.factors.map(([k]) => [k, true])));
  const [selHead, setSelHead] = useState(null);
  const [allHeads, setAllHeads] = useState(false);
  const [stage, setStage] = useState('');
  const claim = CLAIMS[0], life = CLAIM_LIFE[claim.id] || {};
  const lifeNow = stage ? life[stage] : null;
  const top = useMemo(() => rankHeads(), []);
  const ci = Math.round(cur);
  const pct = v => `${Math.round(v * 100)}%`;
  const caseTag = <span className="mp-tag">案例 MA-H2</span>;

  /* 控制台里三种分析模式用的案例数据面板 */
  const panels = {
    lens: <>
      <p className="ac-k">逐层读出 {caseTag}</p>
      <LineChart label="逐层读出：资料色减常识色" yLabel="资料色 − 常识色（logit），阴影为分歧区间 24–33" band={[24, 33]} height={190}
        series={[{ name: '按资料', color: '#2a78d6', values: FLOW_DATA.lensDoc }, { name: '按现实', color: '#eb6834', values: FLOW_DATA.lensReal }]} />
      <p className="v8-note">3D 当前第 {ci} 层：按资料 {FLOW_DATA.lensDoc[ci].toFixed(1)}，按现实 {FLOW_DATA.lensReal[ci].toFixed(1)}。最终读对：按资料 {pct(report.lens.final_correct_share.doc)}，按现实 {pct(report.lens.final_correct_share.real)}。每层先做最终归一化再接 unembedding，是读出近似，早层不可靠。</p>
    </>,
    heads: <>
      <p className="ac-k">最看资料颜色的注意力头 {caseTag}</p>
      <div className="v8-weight-scroll"><table className="v6-table">
        <thead><tr><th>头</th><th>按资料</th><th>按现实</th><th>状态</th></tr></thead>
        <tbody>{top.slice(0, allHeads ? top.length : 6).map(x => <tr key={`${x.l}-${x.h}`} className={selHead === x ? 'as-sel' : ''}
          tabIndex={0} onClick={() => { setSelHead(x); setCur(x.l + 1); }} onKeyDown={e => { if (e.key === 'Enter') { setSelHead(x); setCur(x.l + 1); } }}>
          <td>L{x.l}H{x.h}</td><td>{x.d.toFixed(2)}</td><td>{x.r.toFixed(2)}</td><td><span className="mp-tag">候选 · 未干预</span></td></tr>)}</tbody>
      </table></div>
      <div className="as-row">{!allHeads && <button type="button" className="fw-tbtn" onClick={() => setAllHeads(true)}>显示全部 {top.length} 个</button>}
        <span className="v8-note">点一行，3D 跳到该层并高亮这个头。注意力去向不等于计算使用。</span></div>
    </>,
    factor: <>
      <p className="ac-k">五个因素的逐层差异 {caseTag}</p>
      {FLOW_DATA.factors.map(([key, label, color, values]) => { const pk = values.indexOf(Math.max(...values));
        return <div className="as-frow" key={key}>
          <label><input type="checkbox" checked={facOn[key]} onChange={e => setFacOn(f => ({ ...f, [key]: e.target.checked }))} /> {label}</label>
          <Spark values={values} color={color} />
          <span className="v8-note">峰值第 {pk} 层 {values[pk].toFixed(2)}</span></div>; })}
      <p className="v8-note">勾选决定 3D 里显示哪几条彩线；“问题中的目标名”位置的差异幅度，不是因果作用。</p>
    </>,
  };

  return <div className="v8-route as-page" aria-label="关联机制">
    {!bare && <PageHead title="关联机制 · 运行与路径" sub="上面是全部 37 层的 3D 空间和 HiddenState 信息流；下面的控制台可以输入文字生成，也可以切换模式观察和分析。" />}
    <AssocFlow3D data={FLOW_DATA} cur={cur} onCur={setCur} facOn={facOn} selHead={selHead} live={live} claim={claim.id}
      band={lifeNow?.band || null} toolbar={<div className="as-life">
        <div className="as-life-chips" role="group" aria-label="生命周期环节"><span className="v8-note">生命周期</span>
          {[['', '全部', ''], ...LIFECYCLE].map(([id, label, desc]) => <button type="button" key={id || 'all'} aria-pressed={stage === id} title={desc}
            onClick={() => { setStage(id); const band = id && life[id]?.band; if (band) setCur(band[1]); }}>{label}</button>)}</div>
        {lifeNow && <p className="v8-note"><b>{LIFECYCLE.find(([id]) => id === stage)[1]}</b>：{lifeNow.text}{lifeNow.band ? ` 高亮第 ${lifeNow.band[0]}–${lifeNow.band[1]} 层板。` : ' 没有观察依据，不标层段。'}层段来自本主张已有的观察，不是固定的模型分区。</p>}
      </div>} />
    <AssocConsole mode={mode} onMode={setMode} onLive={setLive} panels={panels} />
    <PillTools label="关联机制的更多内容" items={[
      { id: 'trace', label: '原生轨迹查看器（逐层、逐头、逐个 MLP 单元）', render: () => children },
      { id: 'modes', label: '三种运行方式的区别', render: () => <AssocIntro /> },
      { id: 'more', label: '更多工具：反常句草稿、跨族读出、归因演示', render: () => <>
        <AnomalyDraft />
        <details><summary>各模式族的信息在流中何时可读</summary><InfoFlowMap model={model} /></details>
        <details><summary>演示：三句话的归因（教学）</summary><AttributionGraph snapshot={DEMO_SNAPSHOT} /></details>
      </> },
    ]} />
  </div>;
}
