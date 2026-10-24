/* 证据等级唯一派生（单一事实源）
   ────────────────────────────────────────────────────────────
   事实源 = registry 的 evidence_level_taxonomy（协议字段，随端点返回）。
   前端任何界面都不再自抄一份等级表——这是 v2 审计 A1 的根因：
   AtlasLens 只硬编码 3 个键 ⇒ E2_held_out / E2_behavioral 回落成 E1 颜色，
   且不在首屏分布条上（26 条特征里有 3 条因此看不见）。

   派生规则：
   1. 精确命中分类法 → 原样；
   2. 未在分类法内但带 E<n> 前缀（如 E2_held_out）→ 视为 **E2 的子档**：
      色调继承 E2，标签写成「E2 · held-out」，绝不回落到别的等级；
   3. 连 E<n> 都解析不出 → 灰色虚线「未知等级」，并在控制台告警一次。

   为什么子档不另立一级：分类法是协议定义的（E0–E3 四段），
   界面若自行加级就等于制造第二事实源；子档只做「同一级内的细分」。

   本模块纯 JS 无依赖，可直接被 node 脚本断言（tests/gpt5_temp/_evidence_smoke.mjs）。
*/

export const LEVEL_COLOR = {
  E0: '#64748b',
  E1: '#24558a',
  E2: '#1e5c22',
  E3: '#8a4b16',
  X: '#94a3b8',
};

export const LEVEL_SHORT = {
  E0: 'E0 候选',
  E1: 'E1 可重复',
  E2: 'E2 可预测',
  E3: 'E3 因果',
  X: '未知等级',
};

/* 子档短名（仅用于显示，不参与判定；未知子档原样显示） */
export const SUB_LABEL = {
  held_out: 'held-out',
  behavioral: '行为',
  intervention: '干预',
  candidate: '候选',
};

export const FALLBACK_TAXONOMY = {
  E0_candidate: '单次观察/单模型未复现',
  E1_repeatable: '同协议跨 run/跨 cell 复现，或单模型内稳健',
  E2_predictive: '通过 held-out 预测（新实体/新主题/新折），或跨模型指纹一致',
  E3_causal_scoped: '协议内干预完成且排除主要替代解释；scope 内有效，不可外推',
};

const _warned = new Set();

/** 等级 → 色调键（E0/E1/E2/E3/X）。E2_held_out 这类子档返回父级色调。 */
export function toneOf(level) {
  const s = String(level == null ? '' : level).trim();
  if (!s) return 'X';
  const m = /^(E[0-9])/.exec(s);
  return m && LEVEL_COLOR[m[1]] ? m[1] : 'X';
}

/** 等级 → 颜色（子档继承父级色调） */
export function colorOf(level) {
  return LEVEL_COLOR[toneOf(level)];
}

/** 等级 → CSS 类名（at-e0/at-e1/at-e2/at-e3/at-ex），供徽标与节点复用 */
export function classOf(level) {
  return 'at-' + toneOf(level).toLowerCase();
}

/** 等级 → 短标签；子档显示为「E2 · held-out」 */
export function labelOf(level) {
  const tone = toneOf(level);
  const s = String(level == null ? '' : level).trim();
  if (tone === 'X') return LEVEL_SHORT.X;
  if (s === tone || !s) return LEVEL_SHORT[tone];
  const rest = s.replace(new RegExp('^' + tone + '[_-]?'), '');
  return LEVEL_SHORT[tone] + (rest ? ' · ' + (SUB_LABEL[rest] || rest) : '');
}

/** 分类法 → 父级键（E2 → 分类法里第一个 E2_* 键） */
function parentIndex(taxonomy) {
  const keys = Object.keys(taxonomy || {});
  const byNum = {};
  keys.forEach(k => {
    const m = /^(E[0-9])/.exec(k);
    if (m) (byNum[m[1]] = byNum[m[1]] || []).push(k);
  });
  return byNum;
}

/**
 * 建一个以 registry 分类法为准的判定器。
 * @param {Object} taxonomy registry.evidence_level_taxonomy（缺失时用 FALLBACK_TAXONOMY，并在返回里标 fallback=true）
 */
export function createEvidenceModel(taxonomy) {
  const hasTax = !!(taxonomy && Object.keys(taxonomy).length);
  const tax = hasTax ? taxonomy : FALLBACK_TAXONOMY;
  const byNum = parentIndex(tax);
  const order = Object.keys(byNum).sort();

  function resolve(level) {
    const s = String(level == null ? '' : level).trim();
    const tone = toneOf(s);
    if (tone === 'X') {
      if (s && !_warned.has(s)) {
        _warned.add(s);
        // eslint-disable-next-line no-console
        console.warn('[evidence] 未知证据等级，已按「未知」渲染（不回落任何已知等级）:', s);
      }
      return { key: s, tone: 'X', parent: null, sub: null,
        label: LEVEL_SHORT.X, color: LEVEL_COLOR.X, cls: 'at-ex',
        desc: '未在分类法中登记', known: false };
    }
    const parentKey = (byNum[tone] || [])[0] || null;
    const exact = Object.prototype.hasOwnProperty.call(tax, s);
    const sub = exact ? null : (s === tone ? null : s.replace(new RegExp('^' + tone + '[_-]?'), '') || null);
    return {
      key: s, tone, parent: parentKey, sub,
      label: labelOf(s), color: LEVEL_COLOR[tone], cls: 'at-' + tone.toLowerCase(),
      desc: (parentKey && tax[parentKey]) || '', known: exact,
    };
  }

  /** 分布聚合：父级为一档、子档在该档内计数；width 之和 = 100%（含未知档） */
  function summarize(features) {
    const list = Array.isArray(features) ? features : [];
    const buckets = new Map();
    list.forEach(f => {
      const r = resolve(f && f.evidence_level);
      if (!buckets.has(r.tone)) {
        buckets.set(r.tone, { tone: r.tone, label: LEVEL_SHORT[r.tone],
          color: LEVEL_COLOR[r.tone], cls: 'at-' + r.tone.toLowerCase(),
          count: 0, subs: {}, levels: [] });
      }
      const b = buckets.get(r.tone);
      b.count += 1;
      if (r.sub) b.subs[r.sub] = (b.subs[r.sub] || 0) + 1;
      if (!b.levels.includes(r.key)) b.levels.push(r.key);
    });
    const tones = [...buckets.values()].sort((a, b) => {
      const ia = order.indexOf(a.tone); const ib = order.indexOf(b.tone);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    const total = list.length;
    tones.forEach(t => { t.share = total ? t.count / total : 0; t.width = total ? (100 * t.count) / total : 0; });
    return {
      total,
      tones,
      /* 父级 E2+ 计数（含 E2 的全部子档）——旧实现只数 E2_predictive + E3，会漏子档 */
      atLeastE2: tones.filter(t => t.tone === 'E2' || t.tone === 'E3').reduce((n, t) => n + t.count, 0),
      unknown: list.filter(f => toneOf(f && f.evidence_level) === 'X').length,
      /* 对账断言：总数必须等于特征数，否则说明有等级被漏算 */
      ok: tones.reduce((n, t) => n + t.count, 0) === total,
      fallback: !hasTax,
      taxonomyKeys: Object.keys(tax),
    };
  }

  return { taxonomy: tax, hasTaxonomy: hasTax, order, resolve, summarize };
}
