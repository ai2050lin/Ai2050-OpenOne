export default function KvReceiptReport({ doc }) {
  const { panels, limitations } = doc;
  const inserted = panels.reduce((n, p) => n + p.inserted, 0);
  const wins = panels.reduce((n, p) => n + p.receipt_wins, 0);
  return <section aria-label="Phase 2955 固定 State 的真实改码">
    <p><strong>{inserted} 个合格插入视图中，Receipt 的注意力总量超过 State 的视图有 {wins} 个。</strong></p>
    <p className="v8-note">Phase {doc.phase} · Qwen3-14B 原始 BF16 · L25H18 · {doc.run_id}</p>
    <table className="v6-table"><thead><tr><th>留出面板</th><th>准入家族</th><th>P/Q 改码</th><th>Receipt / State 质量比最大值</th></tr></thead>
      <tbody>{panels.map(p => <tr key={p.split}><td>{p.label}</td><td>{p.families}/{p.planned}</td><td>{p.real_flips}/{p.inserted}</td>
        <td>{Number.isFinite(p.peak_ratio) ? (100 * p.peak_ratio).toFixed(1) + '%' : '—'}</td></tr>)}</tbody></table>
    {panels.filter(p => p.eligible_flips > 0).map(p => <p key={p.split}>在 {p.eligible_flips} 个干预对齐的改码视图中，清零 State Value 后 {p.state_reverted} 个回到原码。联合清零后，
      单独恢复 State 使 {p.state_restore}/{p.joint_reverted} 个再次发出插入后的码。</p>)}
    <details><summary>阻断、恢复与反例</summary>
      {panels.map(p => <div key={p.split}>
        <p>{p.label} · Receipt 寻址阻断回原码 {p.receipt_address_reverted} 次，Receipt Value 清零回原码 {p.receipt_value_reverted} 次。
          寻址／Value 的等幅剂量门 {p.dose_pass.address ? '通过' : '未过'}／{p.dose_pass.value ? '通过' : '未过'}。</p>
        {p.cases.map((c, i) => <details key={c.id + i}><summary>{c.id} · State {c.state} · 原事实 {c.fact}</summary>
          <p>全词表首选 {c.letter.join(' → ')}。插入文本 {c.text}</p>
          <table className="v6-table"><thead><tr><th>复算分支</th><th>实发码</th></tr></thead><tbody>
            {[['sham', '插入原样'], ['address', '屏蔽 Receipt'], ['state-value', '清 State Value'],
              ['joint', '联合清零'], ['value', '仅恢复 State'], ['restore-all', '全部恢复']].map(([key, label]) =>
              <tr key={key}><td>{label}</td><td>{c.arm_letters[key]}</td></tr>)}</tbody></table>
        </details>)}
        {p.excluded_sham.map(id => <p className="v8-note" key={id}>{id} · sham 与干净分支对齐失败</p>)}
      </div>)}
    </details>
    <details><summary>准入、预测与证据来源</summary>
      {panels.map(p => <p key={p.split}>{p.label} · 寻址预测 {p.read_pass ? '过门' : '未过门'} · 改码预测 {p.flip_pass ? '过门' : '未过门'}
        {p.failed_cells.length > 0 && ' · 反角色 State 行为门存在未过格'}</p>)}
      {limitations.map(line => <p className="v8-note" key={line}>{line}</p>)}
      <pre className="v8-artifact-json">{JSON.stringify(doc.source_file_sha256, null, 2)}</pre>
    </details>
  </section>;
}
