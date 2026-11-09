import { useState } from 'react';

const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const number = value => Number.isFinite(value) ? value.toPrecision(4) : '—';
const pass = value => value ? '通过' : '未通过';

function DosePlot({ rows, source }) {
  const points = rows.filter(row => Number.isFinite(row.edit_margin) && Number.isFinite(row.control_margin));
  if (!points.length) return <p>没有可用于比较的等范数样本。</p>;
  const extent = Math.max(.01, ...points.flatMap(row => [Math.abs(row.edit_margin), Math.abs(row.control_margin)]));
  const x = dose => 70 + dose / Math.max(...points.map(row => row.dose)) * 570;
  const y = value => 95 - value / extent * 68;
  return <svg className="v8-dose-chart" viewBox="0 0 680 195" role="img" aria-label={`${source} 编辑与等范数对照剂量曲线`}>
    <line x1="64" x2="645" y1="95" y2="95" stroke="#94a3b8" />
    <text x="8" y="22">nats</text>
    <text x="8" y="32">{number(extent)}</text><text x="8" y="100">0</text><text x="8" y="165">{number(-extent)}</text>
    {points.map(row => <text key={row.dose} x={x(row.dose)} y="188" textAnchor="middle">{row.dose}×</text>)}
    {[['edit_margin', '#2d83c4', '原生编辑'], ['control_margin', '#db7943', '等范数对照']].map(([key, color, label]) => <g key={key}>
      <polyline fill="none" stroke={color} strokeWidth="2" points={points.map(row => `${x(row.dose)},${y(row[key])}`).join(' ')} />
      {points.map(row => <circle key={row.dose} cx={x(row.dose)} cy={y(row[key])} r="4" fill={color}>
        <title>{label} {row.dose}× · {number(row[key])} nats · n={row.switch.n}</title>
      </circle>)}
    </g>)}
  </svg>;
}

function KeywordAudit({ audit }) {
  const count = key => audit.layers.reduce((total, row) => total + row[key], 0);
  const total = count('units'), zero = count('zero'), matched = count('matched_n');
  return <>
    <p>原图谱 {total.toLocaleString()} 个候选单元中，{zero.toLocaleString()} 个在留出集最高的 8 个门控位置零命中。其中 {count('absent_everywhere').toLocaleString()} 个的关键词在全部留出窗口均未出现。</p>
    <p>把范围放到 64 个门控位置后，{count('zero_rescued_top64').toLocaleString()} 个原零命中单元出现至少一次命中。匹配同来源、同当前 token 后，极值命中率 {percent(count('matched_true_hits') / matched)}，随机窗口 {percent(count('matched_control_hits') / matched)}。</p>
    <p>每篇文档只取一个极值后，补回 {count('zero_rescued_distinct8').toLocaleString()} 个单元。原门控极值样本中，{percent(count('g_positive_a_nonpositive') / (total * 8))} 的位置门控为正而写入系数不为正。</p>
    <table className="v6-table"><thead><tr><th>层</th><th>零命中</th><th>留出窗口均缺词</th><th>前 64 位置补回</th><th>改用 8 个写入极值补回</th></tr></thead>
      <tbody>{audit.layers.map(row => <tr key={row.block}><td>B{row.block}</td><td>{row.zero}</td><td>{row.absent_everywhere}</td>
        <td>{row.zero_rescued_top64}</td><td>{row.zero_rescued_a8}</td></tr>)}</tbody></table>
    <p className="v8-note">{audit.scope}</p>
    <details><summary>各层计数和原生单元回查样例</summary><pre className="v8-artifact-json">{JSON.stringify(audit, null, 2)}</pre></details>
  </>;
}

export default function MlpB34ConfirmReport({ doc }) {
  const [tab, setTab] = useState('results');
  const [page, setPage] = useState(0);
  const authority = doc.schema === 'mlp_b31_authority_report.v1';
  const source = authority ? 'B31' : 'B34', target = authority ? 'B32' : 'B35';
  const primary = doc.dose_curve.find(row => row.dose === 2);
  const errors = doc.counterexamples;
  const worst = doc.diagnosis?.metrics.B35.recursive_composed.worst || [];
  const tabs = [['results', '留出与能力'], ['dose', '剂量与对照'], authority ? ['keywords', '关键词复现'] : ['recursion', '递推诊断'], ['examples', '反例'], ['design', '预注册与溯源']];
  const reasons = { norm_invalid: '范数匹配未通过', control_also_switches: '对照也翻转', no_desired_switch: '没有目标翻转' };
  return <section className="v8-intervention-report" aria-label={`${source} 到 ${target} 长文本留出研究`}>
    <p className="v8-note">Qwen3-4B BF16 · {doc.discovery_pairs} 对发现样本 · {doc.holdout_pairs} 对新长文本留出样本</p>
    <div className="v6-toolbar" role="group" aria-label={`${source} 到 ${target} 研究视图`}>
      {tabs.map(([id, title]) => <button key={id} className="fw-tbtn" aria-pressed={tab === id} onClick={() => setTab(id)}>{title}</button>)}
    </div>
    {tab === 'results' && <>
      <p>原模型成对正确率 {percent(doc.holdout_both_correct)}，行为测量门{pass(doc.baseline_gate)}。{source} 编辑翻转门{pass(doc.controlled_behavior_passed)}，{target} 路径门{pass(doc.mediation_passed)}。</p>
      {authority && <p className="v8-note">范围为英文长文档中的授权角色。取消编辑后，{doc.restored_pairs} 对样本均逐位恢复全词表 logits。</p>}
      {authority && doc.preregistration.prior_negative && <details><summary>首版留出行为门未通过</summary>
        <p>成对正确率 {percent(doc.preregistration.prior_negative.accuracy)}。首版保留基线结果，第二版用新材料重新预注册。</p>
        <p>{Object.entries(doc.preregistration.prior_negative.by_frame).map(([frame, value]) => `${frame} ${percent(value)}`).join(' · ')}</p>
        {doc.prior_behavior_examples?.map(row => <details key={row.id}><summary>{row.frame} · {row.id} 原模型错误</summary>
          <pre className="v8-sample-text">{JSON.stringify(row, null, 2)}</pre></details>)}
        <p className="v8-note">首版行为记录 {doc.prior_behavior_failure_sha256}</p>
      </details>}
      <table className="v6-table"><thead><tr><th>层</th><th>组方向正确率</th><th>随机组</th><th>组判定</th><th>单元方向门</th></tr></thead>
        <tbody>{Object.entries(doc.semantic).map(([block, row]) => <tr key={block}>
          <td>{block}</td><td>{percent(row.group.edited_rate)}</td><td>{percent(row.group.control_rate)}</td><td>{pass(row.group.passed)}</td>
          <td>{row.units.filter(unit => unit.passed).length}/{row.units.length}</td>
        </tr>)}</tbody></table>
      <details><summary>逐单元、反转答案编码与模板分组</summary><pre className="v8-artifact-json">{JSON.stringify(doc.semantic, null, 2)}</pre></details>
      {authority && <details><summary>在原生尾层图谱打开已测单元</summary>
        {Object.entries(doc.semantic).map(([block, row]) => <div key={block} className="v6-toolbar">{row.units.map(unit => <a key={unit.unit}
          href={`?lens=spatial&view=graph&gview=internal&run=MLP-TAIL-B28-B35-ATLAS-v2&snapshot=${doc.model_snapshot}&tail_block=${block.slice(1)}&tail_unit=${unit.unit}`}>
          {block}/U{unit.unit}</a>)}</div>)}
      </details>}
      {doc.selectivity_diagnosis && <details><summary>事后诊断 · 模板行为与整层范数对照</summary>
        <pre className="v8-artifact-json">{JSON.stringify(doc.selectivity_diagnosis, null, 2)}</pre>
      </details>}
      <p>能力监测覆盖 {doc.capability.positive.n} 道历史选择题。允许单向损失上限 {percent(doc.capability.positive.margin)}。</p>
      {Object.entries(doc.capability).map(([direction, value]) => <p key={direction}>
        {direction === 'positive' ? '正向' : '反向'}编辑：原对改错 {value.control_only} 题，损失置信上界 {percent(value.loss_upper)}，{pass(value.passed)}。
      </p>)}
    </>}
    {tab === 'dose' && <>
      <p className="v8-note">蓝色为原生编辑，橙色为等范数随机方向。纵轴是相对原模型的目标答案对数优势变化。主剂量为 2×。</p>
      <DosePlot rows={doc.dose_curve} source={source} />
      <table className="v6-table"><thead><tr><th>剂量</th><th>有效配对</th><th>编辑目标翻转</th><th>对照目标翻转</th><th>{target} 方向变化 / 对照</th>{authority && <th>全词表首词翻转 / 对照</th>}</tr></thead>
        <tbody>{doc.dose_curve.map(row => <tr key={row.dose}><td>{row.dose}×</td><td>{row.switch.n}</td>
          <td>{percent(row.switch.edited_rate)}</td><td>{percent(row.switch.control_rate)}</td>
          <td>{number(row.target_shift)} / {number(row.control_target_shift)}</td>{authority && <td>{row.full_vocabulary.edit_target} / {row.full_vocabulary.control_target}（原模型首词正确 {row.full_vocabulary.base_gold}）</td>}</tr>)}</tbody></table>
      <p>{target} 恢复比较使用 {primary.rescue.n} 对等范数样本。一次成功须先出现目标翻转，再回到原答案。恢复组 {percent(primary.rescue.edited_rate)}，随机对照 {percent(primary.rescue.control_rate)}。</p>
      {authority && primary.switch.edited_rate === 0 && <p className="v8-note">主剂量没有目标翻转，因此没有满足答案恢复的起始条件。取消全部编辑后的原模型逐位恢复已独立核验。</p>}
      {authority && doc.margin_diagnostic && <p className="v8-note">事后量级对照。原模型 A/B 优势中位数 {number(doc.margin_diagnostic.baseline_median_ab_margin)} nats。自然交换授权者时，B32 组方向变化 {number(doc.margin_diagnostic.b32_natural_normalized_shift)}，2× 编辑时为 {number(primary.target_shift)}。</p>}
    </>}
    {tab === 'keywords' && <KeywordAudit audit={doc.keyword_audit} />}
    {tab === 'recursion' && <>
      <p>FVU 以历史发现均值预测的平方误差为分母，越小越好。真实残差与注意力复算用于诊断 B35 所需的输入状态。</p>
      <table className="v6-table"><thead><tr><th>层</th><th>单步 A</th><th>递推 A</th><th>单步 G/V</th><th>递推 G/V</th></tr></thead>
        <tbody>{Object.entries(doc.diagnosis.metrics).map(([block, rows]) => <tr key={block}><td>{block}</td>
          {['one_step_direct', 'recursive_direct', 'one_step_composed', 'recursive_composed'].map(key => <td key={key}>{number(rows[key].fvu)}</td>)}</tr>)}</tbody></table>
      <table className="v6-table"><thead><tr><th>B35 计算条件</th><th>FVU</th></tr></thead><tbody>
        {Object.entries({ native_replay: '完整原生边界', without_attention: '移除 B35 注意力写入', without_carried_residual: '移除继承残差',
          write_only: '仅保留 B34 MLP 写入', true_gate_recursive: '递推中恢复真实门控', true_value_recursive: '递推中恢复真实 value' }).map(([key, label]) =>
          <tr key={key}><td>{label}</td><td>{number(doc.diagnosis.metrics.B35[key].fvu)}</td></tr>)}
      </tbody></table>
      {doc.historical_diagnosis && <details><summary>原图谱留出集的误差放大复算</summary>
        <p>原留出集复算 {doc.historical_diagnosis.n_rows} 个位置。G/V 递推 FVU 为 {number(doc.historical_diagnosis.fvu.rec_composed)}，
          门控误差与 value 误差的乘积项平方和除以相同分母为 {number(doc.historical_diagnosis.fvu.interaction_term)}。</p>
        <p>使用真实 B34 输入做单步预测时 FVU 为 {number(doc.historical_diagnosis.fvu.teacher_composed)}。
          将递推输入的标准化值限制在 ±10 后，诊断 FVU 为 {number(doc.historical_diagnosis.fvu.rec_clipped)}。</p>
        <pre className="v8-artifact-json">{JSON.stringify(doc.historical_diagnosis.state, null, 2)}</pre>
      </details>}
    </>}
    {tab === 'examples' && <>
      <p>主剂量保留 {errors.length} 个反例或匹配失败记录。{!authority && '另列出 B35 递推误差最大的文本。'}</p>
      {worst.map(row => <details key={row.id}><summary>{row.id} · 递推平方误差 {number(row.sse)}</summary>
        <pre className="v8-sample-text">{JSON.stringify(doc.counterexample_materials[row.id], null, 2)}</pre></details>)}
      {errors.slice(page * 12, (page + 1) * 12).map(row => <details key={row.sample_id}>
        <summary>{row.sample_id} · {reasons[row.reason] || row.reason}</summary>
        <pre className="v8-sample-text">{JSON.stringify({ ...row, material: doc.counterexample_materials[row.sample_id] }, null, 2)}</pre>
      </details>)}
      <button className="fw-tbtn" disabled={!page} onClick={() => setPage(page - 1)}>上一组</button>
      <button className="fw-tbtn" disabled={(page + 1) * 12 >= errors.length} onClick={() => setPage(page + 1)}>下一组</button>
    </>}
    {tab === 'design' && <>
      <p className="v8-note">{doc.scope}</p>
      <details open><summary>实验设计、功效和五字段身份</summary><pre className="v8-artifact-json">{JSON.stringify({
        provenance: doc.provenance, plan: doc.plan_sha256, selection: doc.selection_sha256, preregistration: doc.preregistration,
      }, null, 2)}</pre></details>
      <details><summary>逐文件原始证据哈希</summary><pre className="v8-artifact-json">{JSON.stringify(doc.raw_manifest, null, 2)}</pre></details>
    </>}
  </section>;
}
