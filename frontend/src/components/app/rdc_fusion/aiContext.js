/* AI 助手的页面上下文（ai_assist.v1）：页面用 useAiContext 登记「当前看的是什么」，问答侧栏读取最近登记的一项。
   上下文只含页面已显示的数据；发送前侧栏会显示标题、字符数与目标主机。组件文件只导出组件，所以构造函数也放在这里。 */
import { useEffect, useSyncExternalStore } from 'react';

let current = null;
const listeners = new Set();
const emit = () => listeners.forEach(fn => fn());

export function aiBridge() {
  return (typeof window !== 'undefined' && window.ai2050Desktop && window.ai2050Desktop.ai) || null;
}

/* ctx = { kind, id, title, data, local? }；调用方用 useMemo 保持引用稳定。卸载时只撤下自己登记的那一项。 */
export function useAiContext(ctx) {
  useEffect(() => {
    if (!ctx) return undefined;
    current = ctx;
    emit();
    return () => { if (current === ctx) { current = null; emit(); } };
  }, [ctx]);
}

export function useCurrentAiContext() {
  return useSyncExternalStore(cb => { listeners.add(cb); return () => listeners.delete(cb); }, () => current, () => null);
}

export function contextChars(ctx) {
  if (!ctx) return 0;
  return (typeof ctx.data === 'string' ? ctx.data : JSON.stringify(ctx.data ?? null)).length;
}

/* ---- 三类页面的上下文构造（只取页面上显示的字段） ---- */

export function langItemContext(d, piece, model) {
  if (!d || !d.item) return null;
  const it = d.item;
  // 逐层曲线只给当前拼图的最佳读出位置，其他读出位置与拼图只留汇总，避免上下文被曲线占满
  const pick = (p, withCurve) => p && {
    id: p.id, tm_id: p.tm_id, factor: p.factor, model: p.model, level: p.level, best_readout: p.readout, languages: p.languages,
    n_items: p.n_items, gate_excluded_cells: p.gate_excluded_cells, behavior: withCurve ? p.behavior : undefined,
    readouts: Object.fromEntries(Object.entries(p.readouts || {}).map(([r, v]) => {
      const { curve, ...rest } = v || {};
      return [r, withCurve && r === p.readout ? { ...rest, curve } : rest];
    })),
  };
  return {
    kind: 'lang_item', id: it.id, title: `语言项「${it.name}」`,
    data: {
      item: { id: it.id, name: it.name, desc: it.desc, type: it.type, category: it.category, also_in: it.also_in },
      model, status: d.status, level_labels: d.level_labels, criteria: d.criteria,
      current_piece: pick(piece, true),
      other_pieces: (d.pieces || []).filter(p => p !== piece).slice(0, 6).map(p => pick(p, false)),
      rules: (d.rules || []).slice(0, 8),
      examples: (d.examples || []).slice(0, 3),
      selectivity: (d.selectivity || []).slice(0, 6),
    },
  };
}

export function neuronUnitContext(layer, unit, { census, block, links } = {}) {
  if (!layer || unit == null || !layer.columns) return null;
  const col = layer.columns;
  const at = k => (Array.isArray(col[k]) ? col[k][unit] : undefined);
  return {
    kind: 'neuron_unit', id: `${census}:L${block}.N${unit}`, title: `神经元 块 ${block} · 单元 ${unit}`,
    data: {
      model_key: census, block, unit, n_tokens: layer.n_tokens,
      type: layer.type_names?.[at('type')], regime: layer.regime_names?.[at('regime')],
      strong_rate: at('dens'), neg_strong_rate: at('neg'), max: at('amax'), std: at('std'),
      pos_top_token: at('hit'), pos_concentration: at('pconc'), direct_readout_top: at('ro'),
      explained: layer.explained?.[String(unit)] || null,
      template_selectivity: links || [],
      rules: layer.rules?.source, note: layer.layout,
    },
  };
}

/* r = 本机节点服务 /probes/<id> 返回的 result.json（goodfire_probe.v1） */
export function probeContext(r) {
  if (!r || r.schema !== 'goodfire_probe.v1' || !r.params) return null;
  const { layer, unit } = r.params;
  return {
    kind: 'goodfire_probe', id: `L${layer}.U${unit}`, title: `神经元测试 L${layer}.U${unit}（本机结果）`, local: true,
    data: {
      params: r.params, corpus: r.corpus && { job_id: r.corpus.job_id, n_windows: r.corpus.n_windows, n_positions: r.corpus.n_positions },
      verdict: r.verdict, reading: r.reading, checks: r.checks, g: r.g, criteria: r.criteria,
      poles: { pos: (r.poles?.pos || []).slice(0, 8), neg: (r.poles?.neg || []).slice(0, 8) },
      S6_vocab: r.S6, S7_steering: r.S7 && { summary: r.S7.summary, note: r.S7.note },
      S8_read_write: r.S8 && { cos_read: r.S8.cos_read, cos_write: r.S8.cos_write, upstream: Array.isArray(r.S8.upstream) ? r.S8.upstream.slice(0, 8) : r.S8.upstream },
      S9_pairs: r.S9,
    },
  };
}
