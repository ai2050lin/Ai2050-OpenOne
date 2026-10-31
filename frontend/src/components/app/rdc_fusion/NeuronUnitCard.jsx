/* 原生 MLP 神经元单元卡（C1）：GET /api/neurons/unit 的只读展示。
   纪律：文本与标签全部来自服务端（普查片段、读出 token、选择性与干预记录），前端不生成任何描述；
   可读 / 读出 / 控制 / 连接四轴与解释轴分开显示（证据等级另行登记，不自动晋级）；没有数据时显示空态，不用其他单元代替。 */
import { OFFLINE_CLIENT, useApi } from './api.js';
import { ExtraPackPrompt } from '../../open/DataBar.jsx';
import { AXIS_META, GRADE_META, REGIMES, REGIME_META, axisTone, fmtNum, fmtRate, histColumns, unitAxes } from './neuronUtil.js';
import './neuron_units.css';

function Histogram({ hist }) {
  const cols = histColumns(hist);
  const W = 520, H = 120, pad = 18;
  const totals = cols.map(c => REGIMES.reduce((s, r) => s + hist.by_regime[r][c.i], 0));
  const maxLog = Math.max(1, ...totals.map(t => Math.log10(t + 1)));
  const zeroX = pad + (hist.n_side / cols.length) * (W - 2 * pad);
  return (
    <svg className="nu-hist" viewBox={`0 0 ${W} ${H + 26}`} role="img" aria-label="激活值分布（symlog 横轴，对数计数，按符号区堆叠）">
      {cols.map(c => {
        let y = H;
        const x = pad + c.x * (W - 2 * pad);
        const w = Math.max(1, c.w * (W - 2 * pad) - 1);
        const tot = totals[c.i];
        const hTot = tot > 0 ? (Math.log10(tot + 1) / maxLog) * (H - 4) : 0;
        return REGIMES.map(r => {
          const v = hist.by_regime[r][c.i];
          if (!v) return null;
          const h = hTot * (v / tot);
          y -= h;
          return <rect key={c.i + r} x={x} y={y} width={w} height={h} fill={REGIME_META[r].color}>
            <title>{`${c.lo >= 0 ? '≥' : '≤'} ${fmtNum(c.lo, 3)} · ${REGIME_META[r].label} · ${v} 次`}</title>
          </rect>;
        });
      })}
      <line x1={zeroX} x2={zeroX} y1={0} y2={H} className="nu-hist-zero"/>
      <text x={pad} y={H + 16} className="nu-hist-ax">−100</text>
      <text x={zeroX} y={H + 16} textAnchor="middle" className="nu-hist-ax">0</text>
      <text x={W - pad} y={H + 16} textAnchor="end" className="nu-hist-ax">+100</text>
      <text x={W / 2} y={H + 26} textAnchor="middle" className="nu-hist-ax">激活值 A（symlog；|A|&lt;0.001 为零箱）· 纵轴为对数计数</text>
    </svg>
  );
}

function Pole({ p }) {
  return (
    <li className="nu-pole">
      <span className="nu-pole-a">{fmtNum(p.a)}</span>
      <span className="nu-reg" style={{ background: REGIME_META[p.regime].color }} title={REGIME_META[p.regime].label}>{p.regime}</span>
      <span className="nu-ctx">{p.left}<mark>{p.hit}</mark>{p.right}</span>
      <span className="nu-src">{p.source}</span>
    </li>
  );
}

export default function NeuronUnitCard({ modelKey, block, unit, run }) {
  const qs = modelKey != null && block != null && unit != null
    ? `/api/neurons/unit?model_key=${encodeURIComponent(modelKey)}&block=${block}&unit=${unit}${run ? `&run=${encodeURIComponent(run)}` : ''}`
    : null;
  const { data, error, loading, missingPack } = useApi(qs);
  if (!qs) return <div className="nu-empty">选择一个单元（左侧检索或按层浏览）。</div>;
  if (error) {
    return (
      <div className="nu-empty nu-err">单元卡不可用：{error}
        {OFFLINE_CLIENT && !missingPack && <div className="nu-note">离线单元卡包只含已解释、模板选择性前列与经过检验的单元。</div>}
        <ExtraPackPrompt pack={missingPack} />
      </div>
    );
  }
  if (loading || !data) return <div className="nu-empty">读取中…</div>;
  const c = data.census;
  return (
    <article className="nu-card" aria-label={`单元 ${data.id}`}>
      <header className="nu-head">
        <div>
          <h3>L{data.layer}.U{data.unit}</h3>
          <div className="nu-sub">{data.unit_label}</div>
          <div className="nu-sub nu-mono">{data.id}</div>
        </div>
        <div className="nu-axes">
          {unitAxes(data).map(([k, v]) => (
            <span key={k} className={`nu-axis nu-axis-${axisTone(v)}`} title={`${AXIS_META[k]}：${v}`}>
              <b>{AXIS_META[k]}</b> {v}
            </span>
          ))}
          <span className="nu-axis nu-axis-none" title="证据等级在 Research OS 登记，不由单元卡的检验记录自动晋级">
            <b>证据等级</b> {data.evidence_level || '未登记'}
          </span>
        </div>
      </header>

      <section className="nu-sec">
        <h4>自然语料激活（{c.n_tokens.toLocaleString()} token · {c.sources.join(' / ')}）</h4>
        <div className="nu-stats">
          <span>最大 <b>{fmtNum(c.max)}</b></span>
          <span>最小 <b>{fmtNum(c.min)}</b></span>
          <span>均值 {fmtNum(c.mean, 3)} · 标准差 {fmtNum(c.std, 3)}</span>
          {Object.entries(c.fire_rate || {}).map(([k, f]) => (
            <span key={k}>{k === 'gt_10pct_max' ? '>10% 最大值' : '>50% 最大值'}：{fmtRate(f.rate)}（{f.per_page}）</span>
          ))}
        </div>
        <Histogram hist={c.hist}/>
        <div className="nu-legend">
          {REGIMES.map(r => (
            <span key={r}><i style={{ background: REGIME_META[r].color }}/>{REGIME_META[r].label} {fmtRate(c.regime_share[r])}</span>
          ))}
        </div>
        <div className="nu-note">{c.excluded}</div>
      </section>

      <section className="nu-sec nu-poles">
        <div>
          <h4>正极（最高激活）</h4>
          <ol>{c.poles.pos.map((p, i) => <Pole key={i} p={p}/>)}</ol>
        </div>
        <div>
          <h4>负极（最低激活）</h4>
          <ol>{c.poles.neg.map((p, i) => <Pole key={i} p={p}/>)}</ol>
        </div>
      </section>

      <section className="nu-sec">
        <h4>直接输出读出（读出轴）</h4>
        <div className="nu-tokens"><span className="nu-tl">推高</span>{c.readout.promotes.map((t, i) => <code key={i}>{t}</code>)}</div>
        <div className="nu-tokens"><span className="nu-tl">压低</span>{c.readout.suppresses.map((t, i) => <code key={i}>{t}</code>)}</div>
        <div className="nu-note">{c.readout.note}</div>
      </section>

      <section className="nu-sec">
        <h4>解释检验（解释轴）</h4>
        {(data.explanations || []).length === 0 ? <div className="nu-note">未检验：该单元不在 EXPL-v1 / v2 的抽样中。</div> : (
          <div className="nu-expl-list">
            {data.explanations.slice().reverse().map((r, i) => (
              <div key={i} className={'nu-expl' + (i > 0 ? ' nu-expl-old' : '')}>
                <div className="nu-expl-head">
                  <span className="nu-grade" style={{ color: GRADE_META[r.grade].color, background: GRADE_META[r.grade].bg }}>
                    {GRADE_META[r.grade].mark} {r.grade}
                  </span>
                  <span className="nu-mono" title={r.prereg_sha256 || ''}>{r.run}{i > 0 ? '（较早一次）' : ''}</span>
                  <span className="nu-note">置信度 {r.confidence}</span>
                </div>
                <div className="nu-expl-text">{r.explanation}</div>
                <div className="nu-note">
                  文本测试：{r.text_outcome}
                  {r.strong_rate && <> · 强激活比例 new {fmtRate(r.strong_rate.new)} / 近似反例 {fmtRate(r.strong_rate.near_miss)} / 远距 {fmtRate(r.strong_rate.distant)} / 对照 {fmtRate(r.strong_rate.control)}</>}
                </div>
                <div className="nu-note">
                  语料检索：{r.retrieval}{r.recall != null && <> · 正例召回 {fmtRate(r.recall)} · 负例误报 {fmtRate(r.fpr)}</>}
                </div>
                {(r.cause || r.reason === '无法解释') && (
                  <div className="nu-note">失败原因：{r.cause || '无法解释'}{r.cause_source && `（${r.cause_source}）`}{r.cause_reason && ` · ${r.cause_reason}`}</div>
                )}
              </div>
            ))}
            <div className="nu-note">解释、测试文本与原因分类均由 Claude 子代理在隔离上下文中生成，判据见对应预注册；等级以最新一次检验为准。</div>
          </div>
        )}
      </section>

      <section className="nu-sec">
        <h4>模板选择性（可读轴）</h4>
        {data.template_selectivity.length === 0 ? <div className="nu-note">该模型没有模板选择性运行。</div> : (
          <table className="nu-table">
            <thead><tr><th>运行</th><th>对照</th><th>读出</th><th>d（发现）</th><th>d（留出）</th><th>阈值</th><th>入选</th><th>来源</th></tr></thead>
            <tbody>{data.template_selectivity.map((r, i) => (
              <tr key={i}>
                <td className="nu-mono">{r.run}</td><td>{r.factor}：{r.hi} 对 {r.lo}</td><td>{r.readout}</td>
                <td>{fmtNum(r.d_disc)}</td><td>{fmtNum(r.d_held)}</td><td>{fmtNum(r.cut)}</td>
                <td>{r.selected == null ? '—' : r.selected ? '✓ 是' : '✗ 否'}</td>
                <td className="nu-note">{r.source || (r.in_top_lists ? '运行的前 20 名记录' : r.recompute_error || '—')}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </section>

      <section className="nu-sec">
        <h4>干预记录（控制轴）</h4>
        {data.interventions.length === 0 ? <div className="nu-note">未检验：该单元不在任何预注册干预的单元清单中。</div> : (
          <table className="nu-table">
            <thead><tr><th>预注册</th><th>段</th><th>方式</th><th>必要性</th><th>充分性</th><th>范围内因果使用</th></tr></thead>
            <tbody>{data.interventions.map((r, i) => (
              <tr key={i}>
                <td className="nu-mono" title={r.prereg_sha256}>{r.prereg}</td><td>{r.segment}</td><td>{r.role}</td>
                <td>{r.necessity} {r.necessity_mean != null && `(${fmtNum(r.necessity_mean, 3)})`}</td>
                <td>{r.sufficiency} {r.sufficiency_mean != null && `(${fmtNum(r.sufficiency_mean, 3)})`}</td>
                <td>{r.causal_use_in_scope}{r.self_check_note && <div className="nu-note">⚠ {r.self_check_note}</div>}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </section>

      {(data.attribution || []).length > 0 && (
        <section className="nu-sec">
          <h4>因果归因（控制轴 · 自然语料）</h4>
          {data.attribution.map((r, i) => (
            <div key={i} className="nu-expl">
              <div className="nu-expl-head">
                <span className="nu-mono" title={r.prereg_sha256}>{r.prereg}</span>
                <span>本层排序 <b>#{r.rank}</b> / {r.n_units}（按 |ΔL̂|）</span>
                <span className="nu-note">{r.validation_verdict}</span>
              </div>
              <div className="nu-stats">
                <span>ΔL̂ 总 <b>{fmtNum(r.total * 1e4)}</b>×10⁻⁴</span>
                <span>直接 {fmtNum(r.direct * 1e4)}×10⁻⁴</span>
                <span>经后续层 {fmtNum(r.indirect * 1e4)}×10⁻⁴</span>
                <span>逐位置幅度 {fmtNum(r.magnitude * 1e4)}×10⁻⁴</span>
              </div>
              {r.true_ablation && (
                <div className="nu-note">真实均值消融：ΔL = {fmtNum(r.true_ablation.delta * 1e4)}×10⁻⁴ nats（噪声底 z = {fmtNum(r.true_ablation.z, 1)}）</div>
              )}
              {r.conditional && (
                <div className="nu-note">
                  触发位置消融（{r.conditional.n_fire} 个位置）：局部损失变化均值 {fmtNum(r.conditional.local_mean * 1e3)}×10⁻³，
                  读出 token 对数概率变化 {fmtNum(r.conditional.dlogp_promote, 3)}，局部检验{r.conditional.sig_local ? '显著' : '不显著'}（整体判定：{r.conditional.run_verdict}）
                </div>
              )}
              <div className="nu-note">{r.note}</div>
            </div>
          ))}
          {(data.trigger || []).map((r, i) => (
            <div key={`t${i}`} className="nu-expl">
              <div className="nu-expl-head">
                <span className="nu-mono" title={r.prereg_sha256}>{r.prereg}</span>
                <span>触发点消融：{r.n_trigger} 个触发位置，其中 {r.n_R} 个的下一 token 属于读出</span>
                <span className="nu-note">主检验（整体）{r.primary_verdict} · 第二检验 {r.secondary_verdict}</span>
              </div>
              <div className="nu-stats">
                {r.d_R != null && <span>读出为答案的位置：损失 <b>{r.d_R > 0 ? '+' : ''}{fmtNum(r.d_R, 3)}</b> nats（p = {fmtNum(r.p_R, 4)}{r.sig_p_R ? '，BH 显著' : ''}；替换单元对照 {fmtNum(r.ctrl_d_R, 4)}）</span>}
                {r.d_T != null && <span>全部触发位置：{fmtNum(r.d_T, 4)}（p = {fmtNum(r.p_T, 3)}）</span>}
                {r.dlogp_promote != null && <span>读出 token 对数概率 {fmtNum(r.dlogp_promote, 3)}</span>}
                {r.d_R == null && r.d_T == null && <span className="nu-note">样本不足，未检验</span>}
              </div>
              <div className="nu-note">{r.note}</div>
            </div>
          ))}
        </section>
      )}

      <footer className="nu-sec nu-caveats">
        {data.caveats.map((t, i) => <div key={i}>· {t}</div>)}
      </footer>
    </article>
  );
}
