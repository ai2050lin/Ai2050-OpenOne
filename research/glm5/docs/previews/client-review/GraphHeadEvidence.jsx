import evidence from './src/components/v8/threeCueEvidence.json';

const run = evidence.runs[0];
const family = run.families[0];
const cases = family.cases.filter(sample => !sample.labelSwap && !sample.positionSwap);
const rules = ['按姓名', '按印刷标签', '按物理首行'];

export default function GraphHeadEvidence({caseId, onCase}) {
  const sample = cases.find(row => row.id === caseId) || cases.find(row => row.rule === 2);
  const highest = sample.mass.indexOf(Math.max(...sample.mass));
  const outside = 1 - sample.mass.reduce((sum, value) => sum + value, 0);
  return <section className="ow-head" aria-label="B26/H36 归档注意力">
    <header>
      <div><h3>B26/H36 的注意力分布</h3><p>Phase 2983 · Qwen3-14B · 最后输入 Token</p></div>
      <label>归档样本<select value={sample.id} onChange={event => onCase(event.target.value)}>
        {cases.map(row => <option key={row.id} value={row.id}>{rules[row.rule]} · {row.id}</option>)}
      </select></label>
    </header>
    <div className="ow-head-grid">
      <div className="ow-masses">
        {sample.order.map((record, index) => <div key={record}>
          <span>第 {index + 1} 行</span>
          <progress aria-label={`第 ${index + 1} 行注意力质量`} max={1} value={sample.mass[record]}/>
          <b>{(100 * sample.mass[record]).toFixed(1)}%</b>
        </div>)}
        <p>其余 Token 合计 {(outside * 100).toFixed(1)}%</p>
      </div>
      <div className="ow-head-source">
        <dl>
          <div><dt>该头最关注</dt><dd>第 {sample.order.indexOf(highest) + 1} 行</dd></div>
          <div><dt>正确记录</dt><dd>第 {sample.order.indexOf(sample.gold) + 1} 行</dd></div>
          <div><dt>模型输出</dt><dd><code>{sample.output}</code></dd></div>
        </dl>
        <button className="fw-tbtn primary" onClick={() => window.openTestDatabase({
          model: run.model, part: 'B26/H36', q: sample.id,
        }, 2983)}>打开这道历史题 →</button>
      </div>
    </div>
  </section>;
}
