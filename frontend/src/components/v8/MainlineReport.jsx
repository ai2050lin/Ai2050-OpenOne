import { useState } from 'react';

const num = v => Number.isFinite(v) ? v.toFixed(4) : '未测';

export default function MainlineReport({ doc }) {
  const [method, setMethod] = useState('exact');
  const report = doc.report, curve = report.curves?.[method];
  const budgets = Object.entries(report.ks || {}).sort(([a], [b]) => Number(a) - Number(b));
  const values = curve ? [...curve.f_keep, ...(curve.f_remove || []), .6, 0] : [];
  const lo = Math.min(...values), hi = Math.max(...values);
  const y = value => 180 - 145 * (value - lo) / (hi - lo || 1);
  const x = i => 50 + i * 100;
  return <section aria-label={`${doc.run_id} 范围内研究报告`}>
    <h3>{doc.run_id} · {curve ? '头级充分性未通过当前判据' : '组件集合的范围内留出确认'}</h3>
    <p className="v8-note">{report.n_items} 对实测样本 · {doc.model_snapshot.slice(0, 16)} · 真实归档。修补保真与图谱独立预测分别记账。</p>
    {curve ? <>
      <p>保留已测排序前 0.5%–20% 的头，最高保真 {num(Math.max(...curve.f_keep))}；未达 0.6 门槛，未冻结充分载体。不能排除其他组合。</p>
      <label className="v8-note">排序方法 <select aria-label="H3b剂量曲线排序方法" value={method} onChange={e => setMethod(e.target.value)}>{Object.keys(report.curves).map(key => <option key={key}>{key}</option>)}</select></label>
      <svg viewBox="0 0 610 225" role="img" aria-label="真实头预算曲线：蓝线保留，橙线移除，虚线为0.6门槛" style={{width:'100%',maxWidth:730}}>
        <line x1="45" x2="565" y1={y(.6)} y2={y(.6)} stroke="#94a3b8" strokeDasharray="4 4"/>
        <text x="565" y={y(.6)-4} fontSize="11">0.6</text>
        {[['f_keep','#2a78d6'],['f_remove','#eb6834']].filter(([key]) => Array.isArray(curve[key])).map(([key,color]) => <g key={key}>
          <polyline fill="none" stroke={color} strokeWidth="2" points={curve[key].map((v,i)=>`${x(i)},${y(v)}`).join(' ')}/>
          {curve[key].map((v,i)=><circle key={i} cx={x(i)} cy={y(v)} r="3" fill={color}><title>{key} · {budgets[i]?.[1]} heads · {v}</title></circle>)}
        </g>)}
        {budgets.map(([fraction],i)=><text key={fraction} x={x(i)} y="202" textAnchor="middle" fontSize="11">{Number(fraction)*100}%</text>)}
        <text x="50" y="220" fontSize="11" fill="#2a78d6">保留排序组 f_keep</text><text x="260" y="220" fontSize="11" fill="#eb6834">移除排序组 f_remove</text>
      </svg>
      <div className="v8-weight-scroll"><table className="v6-table"><thead><tr><th>头预算</th><th>头数</th><th>保留保真</th><th>移除后保真</th><th>保留 CI（报告口径）</th></tr></thead>
        <tbody>{budgets.map(([fraction,k],i)=><tr key={fraction}><td>{Number(fraction)*100}%</td><td>{k}</td><td>{num(curve.f_keep[i])}</td><td>{num(curve.f_remove?.[i])}</td><td>{curve.keep_ci?.[i]?.map(num).join(' – ') || '未记录'}</td></tr>)}</tbody></table></div>
      <p className="v8-note">负值保留原值。少量头移除影响集中，不说明同一集合足以实现功能；入口∪接力是待检验假设。组排序不等于每个头各自的因果证明。</p>
    </> : <>
      <p>594 / 1188 组件，留出干预保真 f(C)={num(report.P1_sufficiency?.f_C)}；最佳匹配对照 {num(Math.max(...Object.values(report.P1_sufficiency?.controls || {})))}。</p>
      <p className="v8-note">原报告 P1–P3 在其口径下通过；C2 校准与 H3b 负结果已经收紧解释。此结果不等于独立预测、单个 Head/神经元机制或整体破解比例。</p>
    </>}
    <p className="v8-note">本报告未提供独立图谱预测、能力保持和第二节点复算的完整验收；这些项目不自动通过。</p>
    <details><summary>原始数值、反例与来源哈希</summary><p className="v8-note">{doc.source_path} · {doc.source_sha256}</p><pre className="v8-artifact-json">{JSON.stringify(report,null,2)}</pre></details>
  </section>;
}
