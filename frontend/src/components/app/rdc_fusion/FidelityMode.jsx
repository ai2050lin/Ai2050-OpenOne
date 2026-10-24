/* 谱图预测保真度曲线（fidelity.v2 · Phase 3239/3240/3246-3250 封存产物只读视图）
   数据 = GET /api/ai-rnd/atlas/fidelity?model=...（census 六族族方向保真 + 逐行 KL + R² + 现役配方标注）
   纪律：前端零口径计算——所有数字来自封存 result.json/npz（sha8 注记在卡片上），
   本组件只做可视化与封存数字查表；负保真 = 琥珀、正保真 = emerald（与 NeuronSpace 发散色带同语义）。 */
import { useEffect, useMemo, useState } from 'react';
import { API_BASE } from '../../../config.js';

const MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b'];
const GREEN = '#047857', AMBER = '#b45309', BLUE = '#0369a1', GRAY = '#94a3b8', PURPLE = '#7c3aed';

function famColor(v) { return v >= 0 ? GREEN : AMBER; }

/* 族方向保真分组条形图：每族 N 序列，0 轴居中自适应 */
function FamBarChart({ fams, series }) {
  const W = 640, ROW_H = 74, LBL = 78, GAP = 10;
  const H = fams.length * (ROW_H + GAP) + 30;
  const all = series.flatMap(s => fams.map(f => s.data[f] ?? 0));
  const vmax = Math.max(1, ...all.map(Math.abs));
  const zeroX = LBL + (W - LBL) * (vmax / (2 * vmax)) * 0.92;   // 负值区占少量宽度
  const span = W - LBL - 14;
  const px = v => zeroX + (v / (2 * vmax)) * span;
  const rows = fams.map((f, i) => ({ f, y: i * (ROW_H + GAP) + 6 }));
  const slot = ROW_H / series.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxHeight: 480 }}>
      <line x1={zeroX} y1={0} x2={zeroX} y2={H - 24} stroke="#cbd5e1" strokeWidth="1.5"/>
      <text x={zeroX} y={H - 8} fontSize="11" fill="#64748b" textAnchor="middle">0（=B0 常量基线）</text>
      {rows.map(r => (
        <g key={r.f}>
          <text x={LBL - 10} y={r.y + ROW_H / 2} fontSize="12" fontWeight="700" fill="#334155" textAnchor="end">{r.f}</text>
          {series.map((s, si) => {
            const v = s.data[r.f] ?? 0;
            const h = Math.min(14, slot - 4), y = r.y + si * slot + (slot - h) / 2;
            const x = v >= 0 ? px(0) : px(v), w = Math.max(1.5, Math.abs(px(v) - px(0)));
            return (
              <g key={s.name}>
                <rect x={x} y={y} width={w} height={h} rx="2" fill={s.color} opacity="0.88"/>
                <text x={v >= 0 ? x + w + 5 : x - 5} y={y + 11} fontSize="10.5"
                      fill="#475569" textAnchor={v >= 0 ? 'start' : 'end'}>
                  {v >= 0 ? '+' : ''}{v.toFixed(2)}
                </text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

/* 缺口收线曲线：SCLS 保真沿 phase 里程碑（3240 G0 → 3250 现役 → oracle 上界），
   每点 = 封存 aggregate 查表（零口径计算）；点越接近 0 = 缺口越小 */
function GapLineChart({ fams, milestones }) {
  const W = 640, H = 190, PADL = 52, PADB = 46, PADT = 18;
  const vals = milestones.filter(m => m.v != null);
  if (vals.length < 2) return null;
  const ymin = Math.min(...vals.map(m => m.v)), ymax = Math.max(...vals.map(m => m.v));
  const y0 = Math.min(0, ymin), y1 = Math.max(0.15, ymax);
  const px = i => PADL + (i / (vals.length - 1)) * (W - PADL - 16);
  const py = v => H - PADB - ((v - y0) / (y1 - y0)) * (H - PADB - PADT);
  const zeroY = py(0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      <line x1={PADL} y1={zeroY} x2={W - 16} y2={zeroY} stroke="#059669" strokeWidth="1.2" strokeDasharray="4 3"/>
      <text x={W - 18} y={zeroY - 5} fontSize="10" fill="#047857" textAnchor="end">0 = 转正线</text>
      <polyline points={vals.map((m, i) => `${px(i)},${py(m.v)}`).join(' ')}
                fill="none" stroke={BLUE} strokeWidth="2" opacity="0.85"/>
      {vals.map((m, i) => (
        <g key={m.label}>
          <circle cx={px(i)} cy={py(m.v)} r={m.hot ? 5 : 3.5} fill={m.hot ? PURPLE : (m.v >= 0 ? GREEN : AMBER)}/>
          <text x={px(i)} y={py(m.v) - 9} fontSize="10" fontWeight={m.hot ? 800 : 600}
                fill={m.hot ? PURPLE : '#475569'} textAnchor="middle">{m.v.toFixed(3)}</text>
          <text x={px(i)} y={H - PADB + 14} fontSize="9.5" fill="#64748b" textAnchor="middle"
                transform={`rotate(-18 ${px(i)} ${H - PADB + 14})`}>{m.label}</text>
        </g>
      ))}
    </svg>
  );
}

/* 逐行 KL 折线（3239：876 OOV 行 × {model, B0, B1}，3 seed 均值，服务端预聚合） */
function KlLineChart({ kl, n }) {
  const W = 640, H = 220, PADL = 46, PADB = 22;
  const keys = [['B0', AMBER], ['B1', BLUE], ['model', GREEN]];
  const all = keys.flatMap(k => kl[k[0]] || []);
  const ymin = 0, ymax = Math.max(0.5, ...all) * 1.05;
  const px = i => PADL + (i / Math.max(1, n - 1)) * (W - PADL - 12);
  const py = v => H - PADB - (v / ymax) * (H - PADB - 8);
  // 下采样到 ≤300 点防 polyline 过密
  const step = Math.max(1, Math.ceil(n / 300));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      {[0, 0.25, 0.5, 0.75, 1].map(t => (
        <g key={t}>
          <line x1={PADL} y1={py(ymax * t)} x2={W - 12} y2={py(ymax * t)} stroke="#e2e8f0"/>
          <text x={PADL - 6} y={py(ymax * t) + 4} fontSize="10" fill="#64748b" textAnchor="end">{(ymax * t).toFixed(2)}</text>
        </g>
      ))}
      {keys.map(([k, c]) => {
        const arr = kl[k] || [];
        const pts = arr.filter((_, i) => i % step === 0).map((v, j) => `${px(j * step)},${py(v)}`).join(' ');
        return <polyline key={k} points={pts} fill="none" stroke={c} strokeWidth={k === 'model' ? 2 : 1.2} opacity={k === 'model' ? 1 : 0.75}/>;
      })}
      <g>
        {keys.map(([k, c], i) => (
          <g key={k} transform={`translate(${PADL + i * 110}, 12)`}>
            <rect width="18" height="3" y="5" fill={c}/>
            <text x="24" y="10" fontSize="11" fill="#475569">{k}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

export default function FidelityMode() {
  const [model, setModel] = useState('qwen3-4b');
  const [doc, setDoc] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setDoc(null); setErr(null);
    fetch(`${API_BASE}/api/ai-rnd/atlas/fidelity?model=${model}`).then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(setDoc).catch(e => setErr(String(e)));
  }, [model]);

  const p39 = useMemo(() => (doc?.phases || []).find(p => p.phase === '3239'), [doc]);
  const p40 = useMemo(() => (doc?.phases || []).find(p => p.phase === '3240'), [doc]);
  const p50 = useMemo(() => (doc?.phases || []).find(p => p.phase === '3250'), [doc]);
  const fams = doc?.fam_order || [];
  const incChamp = useMemo(() => (doc?.incumbent || []).find(x => x.tag === '现役综合最优') || null, [doc]);

  const recFid = useMemo(() => (phase, rn) => {
    const p = (doc?.phases || []).find(x => x.phase === phase);
    if (!p || !p.recipes) return null;
    const i = p.recipes.indexOf(rn);
    if (i < 0) return null;
    const si = fams.indexOf('SCLS');
    return { scls: p.agg_fid_matrix?.[i]?.[si] ?? null, r2: p.agg_r2?.[i] ?? null };
  }, [doc, fams]);

  const series = useMemo(() => {
    if (!p39 || !p40) return [];
    const m39 = p39.aggregate?.fam_fid_oov_mean || {};
    const rec = p40.recipes || [];
    const fid = p40.agg_fid_matrix || [];
    const out = [{ name: '3239 model(F0)', color: GRAY, data: m39 }];
    rec.forEach((rn, i) => {
      if (rn === 'F0' || rn === 'F1') return;
      out.push({ name: '3240 ' + rn, color: rn === 'F2' ? GREEN : '#d97706',
                 data: Object.fromEntries(fams.map((f, j) => [f, fid[i]?.[j] ?? 0])) });
    });
    // 里程碑对照：3246 XC_b4（声明体制首破 −0.77）+ 3250 现役 XC_p5_b8
    const p46 = (doc?.phases || []).find(x => x.phase === '3246');
    if (p46?.recipes) {
      const i = p46.recipes.indexOf('XC_b4');
      if (i >= 0) out.push({ name: '3246 XC_b4', color: BLUE,
        data: Object.fromEntries(fams.map((f, j) => [f, p46.agg_fid_matrix?.[i]?.[j] ?? 0])) });
    }
    if (p50?.recipes) {
      const i = p50.recipes.indexOf('XC_p5_b8');
      if (i >= 0) out.push({ name: '3250 现役 XC_p5_b8', color: PURPLE,
        data: Object.fromEntries(fams.map((f, j) => [f, p50.agg_fid_matrix?.[i]?.[j] ?? 0])) });
    }
    return out;
  }, [doc, p39, p40, p50, fams]);

  // 缺口收线里程碑（SCLS，全部封存查表）
  const milestones = useMemo(() => ([
    { label: 'G0 基线', phase: '3246', rn: 'G0' },
    { label: 'T3 迁移', phase: '3246', rn: 'T3_rep' },
    { label: 'XC_b4', phase: '3246', rn: 'XC_b4' },
    { label: 'XC_poly_b4', phase: '3247', rn: 'XC_poly_b4' },
    { label: 'XC_p3_b4', phase: '3248', rn: 'XC_p3_b4' },
    { label: 'XC_p5_b8 现役', phase: '3250', rn: 'XC_p5_b8', hot: true },
    { label: 'oracle 上界', phase: '3250', rn: 'ORC_ls_cls' },
  ].map(m => { const f = recFid(m.phase, m.rn); return { ...m, v: f?.scls ?? null }; })), [recFid]);

  if (err) {
    return <div className="fw-panel" style={{ padding: 18, color: 'var(--fw-text-3)' }}>
      保真度曲线不可用：{err}（需先在服务端完成 Phase 3239/3240/3246-3250 封存产物落盘）
    </div>;
  }
  if (!doc) return <div className="fw-panel" style={{ padding: 18 }}>加载保真度曲线…</div>;

  const agg39 = p39?.aggregate || {};
  const gapPct = agg39.kl_B0_mean_mean
    ? 100 * (agg39.kl_B0_mean_mean - agg39.kl_model_mean_mean) / agg39.kl_B0_mean_mean : null;
  const r2 = p40?.agg_r2 || [];
  const recs = p40?.recipes || [];
  const f2i = recs.indexOf('F2'), f0i = recs.indexOf('F0');

  return (
    <div className="fw-panel" style={{ padding: '14px 18px', overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <b style={{ fontSize: 15 }}>谱图预测保真度曲线</b>
        <span style={{ fontSize: 12, color: 'var(--fw-text-3)' }}>「破解程度」= census 六族族方向保真 + 下一 token KL + R²（876 谱外行）</span>
        <span style={{ flex: 1 }}/>
        {MODELS.map(m => (
          <button key={m} className={'fw-tech-chip' + (model === m ? ' on' : '')}
                  onClick={() => setModel(m)}>{m}</button>
        ))}
      </div>

      {/* 摘要卡 */}
      <div style={{ display: 'flex', gap: 12, margin: '12px 0', flexWrap: 'wrap' }}>
        <div className="fw-stat-card" style={{ minWidth: 200 }}>
          <div style={{ fontSize: 11, color: 'var(--fw-text-3)' }}>3239 · KL 缺口恢复（vs B0）</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: GREEN }}>{gapPct != null ? gapPct.toFixed(1) + '%' : '—'}</div>
          <div style={{ fontSize: 11.5, color: 'var(--fw-text-3)' }}>
            model {agg39.kl_model_mean_mean?.toFixed(4)} &lt; B1 {agg39.kl_B1_mean_mean?.toFixed(4)} &lt; B0 {agg39.kl_B0_mean_mean?.toFixed(4)}
          </div>
        </div>
        <div className="fw-stat-card" style={{ minWidth: 200 }}>
          <div style={{ fontSize: 11, color: 'var(--fw-text-3)' }}>3240 · R² OOV（F2 子空间补全）</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: BLUE }}>
            {f2i >= 0 ? r2[f2i]?.toFixed(3) : '—'}
            {f0i >= 0 && <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--fw-text-3)' }}> （F0 {r2[f0i]?.toFixed(3)}）</span>}
          </div>
          <div style={{ fontSize: 11.5, color: 'var(--fw-text-3)' }}>R² 分布内 seen-test {agg39.r2_seen_model_mean?.toFixed(3)}</div>
        </div>
        <div className="fw-stat-card" style={{ minWidth: 200 }}>
          <div style={{ fontSize: 11, color: 'var(--fw-text-3)' }}>3240 · SCLS / ENT 保真（F2）</div>
          {(() => {
            const f2 = series.find(s => s.name === '3240 F2');
            return <div style={{ fontSize: 22, fontWeight: 800, color: AMBER }}>
              {f2 ? `${f2.data.SCLS?.toFixed(2)} / ${f2.data.ENT?.toFixed(2)}` : '—'}
            </div>;
          })()}
          <div style={{ fontSize: 11.5, color: 'var(--fw-text-3)' }}>F1 均匀先验塌缩（机制性零增益）</div>
        </div>
        <div className="fw-stat-card" style={{ minWidth: 220, borderColor: PURPLE }}>
          <div style={{ fontSize: 11, color: PURPLE, fontWeight: 700 }}>3250 · 现役配方 XC_p5_b8</div>
          {incChamp ? (
            <>
              <div style={{ fontSize: 22, fontWeight: 800, color: AMBER }}>
                {incChamp.fam_fid.SCLS?.toFixed(2)} / {incChamp.fam_fid.ENT?.toFixed(2)}
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--fw-text-3)' }}>
                R² {incChamp.r2_oov?.toFixed(3)} · KL {incChamp.kl_mean?.toFixed(3)}（poly5×β8 seen-only 迁移）
              </div>
            </>
          ) : <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--fw-text-3)' }}>—</div>}
        </div>
      </div>

      {/* 缺口收线（3240→3250 SCLS 里程碑，封存查表） */}
      {milestones.some(m => m.v != null) && (
        <div style={{ border: '1px solid #e2e8f0', borderRadius: 10, padding: '10px 14px', marginBottom: 12 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 2 }}>SCLS 缺口收线（Phase 3240→3250 里程碑；每点 = 封存 aggregate 查表，3 seed 均值）</div>
          <div style={{ fontSize: 11.5, color: 'var(--fw-text-3)', marginBottom: 6 }}>
            紫 = 3250 现役 XC_p5_b8；绿虚线 = 转正线（0）；最后一点 = 完美类信息 oracle 上界（EVALUATION-ONLY）
          </div>
          <GapLineChart fams={fams} milestones={milestones}/>
        </div>
      )}

      {/* 族方向保真 */}
      <div style={{ border: '1px solid #e2e8f0', borderRadius: 10, padding: '10px 14px', marginBottom: 12 }}>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 2 }}>族方向保真度（1 − 残差族基能量比 vs B0；正值=图谱预测优于常量基线）</div>
        <div style={{ fontSize: 11.5, color: 'var(--fw-text-3)', marginBottom: 6 }}>
          序列：灰=3239 原版 ridge（=F0 复现锚 d39=0.0）· 绿=3240 F2 类子空间补全 · 蓝=3246 XC_b4（声明体制首破）· 紫=3250 现役 XC_p5_b8（poly5×β8 迁移）
        </div>
        {series.length > 0 && <FamBarChart fams={fams} series={series}/>}
      </div>

      {/* 逐行 KL */}
      {p39?.kl_per_row_mean3seed && (
        <div style={{ border: '1px solid #e2e8f0', borderRadius: 10, padding: '10px 14px' }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 2 }}>下一 token KL（876 谱外行逐行，3 seed 均值；lens = tied unembedding）</div>
          <div style={{ fontSize: 11.5, color: 'var(--fw-text-3)', marginBottom: 6 }}>
            绿=model（图谱预测）· 蓝=B1 identity · 琥珀=B0 全局均值常量；曲线整体越低=输出分布越接近真实
          </div>
          <KlLineChart kl={p39.kl_per_row_mean3seed} n={(p39.kl_per_row_mean3seed.model || []).length}/>
        </div>
      )}

      <div style={{ fontSize: 11, color: 'var(--fw-text-3)', marginTop: 10 }}>
        数据源：{doc.phases.map(p => `phase${p.phase} ${p.name}（verdict ${p.verdict} · freeze ${p.design_sha8} · res ${p.res_sha8}）`).join('；')}
        {doc.incumbent?.length > 0 && <>；现役标注：{doc.incumbent.map(x => `${x.recipe}（${x.tag}）`).join('、')}</>}
        。前端零口径计算，全部数字来自封存产物。
      </div>
    </div>
  );
}
