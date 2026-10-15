/* 模型层位条：把「结构」（模型有多少层、采样的是哪几层）和「响应」（该层的读出准确率）放在同一条轴上。
   数据：/api/analysis 的 model_layers（runner 记录的 hidden_states 下标；0 = 嵌入输出，k = 第 k 个层块输出）
   与 layers[].factors[主因子].splits.lockbox.probe_acc。颜色 = 顺序蓝色阶，按「高出机会水平的比例」归一。
   旧结果没有层位记录时，只列出采样标签。点击采样层 = 切换当前层。 */
import { fmtNum } from './api.js';
import { inkOn, seqColor } from './ui.jsx';

function readout(layer, factor) {
  const f = layer?.factors?.[factor];
  if (!f) return null;
  const acc = f.splits?.lockbox?.probe_acc ?? f.splits?.confirm?.probe_acc;
  if (acc == null) return null;
  const chance = f.chance ?? 0;
  return { acc, chance, norm: chance < 1 ? Math.max(0, Math.min(1, (acc - chance) / (1 - chance))) : 0 };
}

export default function LayerRail({ modelLayers, layers, factor, selected, onSelect }) {
  const byTag = Object.fromEntries((layers || []).map(l => [l.tag, l]));
  const tags = (layers || []).map(l => l.tag);
  if (!tags.length) return null;
  const chance = tags.map(t => readout(byTag[t], factor)?.chance).find(v => v != null);
  const legend = factor ? `颜色 = 「${factor}」在未见词表上的读出准确率（机会 ${fmtNum(chance ?? 0, 2)}）` : '';

  if (!modelLayers) {
    return (
      <div className="v6-rail">
        <div className="v6-rail-hd"><b>模型层位</b><small>层位未记录（旧结果；用新版 runner 重新采集后显示采样层在模型中的真实位置）· {legend}</small></div>
        <div className="v6-rail-chips">
          {tags.map(t => {
            const r = readout(byTag[t], factor);
            return (
              <button key={t} type="button" className={'v6-rail-chip' + (t === selected ? ' on' : '')} onClick={() => onSelect(t)}
                style={r ? { background: seqColor(r.norm), color: inkOn(r.norm) } : undefined}>
                {t}{r ? ` · ${(r.acc * 100).toFixed(0)}%` : ''}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  const n = modelLayers.n_hidden_states;
  const idx = modelLayers.layer_index || {};
  const at = {};
  for (const t of tags) if (idx[t] != null) at[idx[t]] = t;
  const blocks = n - 1;
  const step = blocks > 48 ? 12 : blocks > 24 ? 6 : 4;
  const ticks = [0];
  for (let k = step; k < n - 1; k += step) ticks.push(k);
  ticks.push(n - 1);
  const pct = k => `${((k + 0.5) / n) * 100}%`;
  const anchor = k => (k / n < 0.08 ? '-15%' : k / n > 0.92 ? '-85%' : '-50%');   // 两端标签不越界
  return (
    <div className="v6-rail">
      <div className="v6-rail-hd">
        <b>模型层位</b>
        <small>{blocks} 个层块 · 采样 {tags.length} 层 · {legend} · 点击采样层切换</small>
      </div>
      <div className="v6-rail-body">
        <div className="v6-rail-labels" aria-hidden="true">
          {Object.entries(at).map(([k, t]) => {
            const r = readout(byTag[t], factor);
            return (
              <span key={t} className={t === selected ? 'on' : ''} style={{ left: pct(Number(k)), transform: `translateX(${anchor(Number(k))})` }}>
                <b>{t}</b>{r ? ` ${(r.acc * 100).toFixed(0)}%` : ''}
              </span>
            );
          })}
        </div>
        <div className="v6-rail-track" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
          {Array.from({ length: n }, (_, k) => {
            const t = at[k];
            if (!t) return <span key={k} className="v6-rail-cell" title={k === 0 ? '嵌入输出（未采样）' : `第 ${k} 层块输出（未采样）`} />;
            const r = readout(byTag[t], factor);
            return (
              <button key={k} type="button" className={'v6-rail-cell s' + (t === selected ? ' on' : '')}
                style={{ background: r ? seqColor(0.15 + 0.85 * r.norm) : '#94a3b8' }}
                aria-label={`${t}：hidden_states[${k}]${r ? `，读出准确率 ${(r.acc * 100).toFixed(0)}%` : ''}`}
                title={`${t} · hidden_states[${k}]（${k === 0 ? '嵌入输出' : `第 ${k} 层块输出`}）${r ? `\n读出准确率 ${fmtNum(r.acc)}（机会 ${fmtNum(r.chance, 2)}）` : ''}`}
                onClick={() => onSelect(t)} />
            );
          })}
        </div>
        <div className="v6-rail-axis" aria-hidden="true">
          {ticks.map(k => <span key={k} style={{ left: pct(k) }}>{k === 0 ? '嵌入' : k}</span>)}
        </div>
      </div>
    </div>
  );
}
