# -*- coding: utf-8 -*-
"""语言谱图拼图（pieces.v1）：把两条路线的结果挂到同一棵语言树（lang_tree.json）上。

路线 T（自顶向下，语言 → 内部）：每个已冻结正式模板 × 因子 × 模型桶一张 piece，取自 analysis.v1 规律分析
（行为门开），逐读出位置记录逐层留出投影间隔、探针准确率与可读出判定（与 analysis.findings 同一判据）。
路线 N（自底向上，神经元 → 语言）尚未接入（设计见 design/client_v7_two_routes_atlas.md §2.2）。

读出轴等级由本文件 CRITERIA 固定判定，随结果返回：
  0 未研究；1 有数据（行为门过滤后的合并结果已有规律分析）；
  2 confirm 划分上可读出（hs[0] 之后的层）；
  3 lockbox（未见词表）上可读出，且同一读出位置的嵌入层 hs[0] 各样本相同或不可分（排除 token 身份；
    基线从宽：hs[0] 探针高于机会 + margin 即视为可分，不要求置换检验；没有 hs[0] 记录时也不升到 3）；
  4 语言项层面：等级 3 的 piece 来自 ≥2 个模型或 ≥2 个模式族。
控制轴（干预）与解释轴本模块不判定：路线 T 没有干预结果，一律记为未检验。
等级只描述「可读出」证据，不表示该方向被模型使用，也不表示是计算单元。
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = 'lang_atlas.v1'
TREE_PATH = Path(__file__).resolve().parent / 'lang_tree.json'

CRITERIA = {
    'level2_split': 'confirm',          # discovery 拟合，confirm 上可读出
    'level3_split': 'lockbox',          # 未见词表上可读出（含置换检验）
    'baseline_layer': 0,                # 嵌入层 hs[0]：在这里已可分 → 视为 token 身份，不升到 3
    'baseline_rule': 'hs[0] lockbox 探针准确率 ≥ 机会 + probe_margin 即为可分（不要求置换检验）；所有样本向量相同为 identical',
    'min_cross': 2,                     # 等级 4：等级 3 的 piece 来自 ≥2 个模型或 ≥2 个模式族
    'gate': True,                       # 只用行为门过滤后的分析
    'readable_rule': 'pieces.layer_readable = analysis.findings 的 readable（probe_acc ≥ 机会 + probe_margin，且该划分有置换检验时 sep_p ≤ perm_alpha）',
}
LEVEL_LABEL = {0: '未研究', 1: '有数据', 2: '可读出（confirm）', 3: '留出复现（未见词表）', 4: '跨模型 / 跨族复现'}
READOUT_ORDER = ('target', 'cue', 'final', 'probe_answer')
MAX_SCATTER = 300
_TAG = re.compile(r'^(?P<ro>[A-Za-z_]+)@L(?P<i>\d+)$')


# ---------------- 语言树 ----------------

def load_tree(path: Optional[Path] = None) -> Dict[str, Any]:
    tree = json.loads(Path(path or TREE_PATH).read_text(encoding='utf-8'))
    errs = validate_tree(tree)
    if errs:
        raise ValueError('lang_tree 无效：' + '；'.join(errs))
    return tree


def validate_tree(tree: Dict[str, Any]) -> List[str]:
    errs = []
    types = {t['id'] for t in tree.get('types', [])}
    cats = {c['id']: c for c in tree.get('categories', [])}
    for c in cats.values():
        if c.get('type') not in types:
            errs.append(f"类别 {c['id']} 的类型 {c.get('type')} 不存在")
    ids = [i['id'] for i in tree.get('items', [])]
    if len(ids) != len(set(ids)):
        errs.append('语言项 id 重复')
    for it in tree.get('items', []):
        if it.get('category') not in cats:
            errs.append(f"语言项 {it['id']} 的类别 {it.get('category')} 不存在")
        for t in it.get('also_in') or []:
            if t not in types:
                errs.append(f"语言项 {it['id']} 的 also_in {t} 不存在")
    for fam, m in (tree.get('family_map') or {}).items():
        for f, item in m.items():
            if item not in ids:
                errs.append(f'family_map {fam}.{f} 指向不存在的语言项 {item}')
    return errs


def factor_item(tree: Dict[str, Any], family_id: str, factor: str, lang_items: Optional[Dict[str, str]] = None) -> Optional[str]:
    """因子 → 语言项：文法自带的 lang_items 优先，其次 lang_tree.family_map；都没有返回 None（未归类）。"""
    if lang_items and lang_items.get(factor):
        return lang_items[factor]
    return ((tree.get('family_map') or {}).get(family_id) or {}).get(factor)


def _type_of(tree: Dict[str, Any]) -> Dict[str, str]:
    cats = {c['id']: c['type'] for c in tree['categories']}
    return {it['id']: cats[it['category']] for it in tree['items']}


# ---------------- 路线 T：从 analysis.v1 抽取 ----------------

def _parse_tag(tag: str) -> Optional[Tuple[str, int]]:
    m = _TAG.match(tag or '')
    return (m['ro'], int(m['i'])) if m else None


def extract_analysis(doc: Dict[str, Any]) -> Dict[str, Any]:
    """一份 analysis.v1 结果 → 逐因子、逐读出位置的层曲线与等级（不涉及语言树映射，可按文件缓存）。"""
    C = doc.get('criteria') or {}
    out: Dict[str, Any] = {'factors': {}, 'n_items': doc.get('n_items'), 'n_excluded': doc.get('n_excluded'),
                           'gate_excluded_cells': doc.get('gate_excluded_cells') or [], 'available': doc.get('available', True) is not False}
    if not out['available'] or not C:          # analysis.analyze 直接输出没有 available 字段
        return out
    by_ro: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    degenerate: Dict[str, set] = {}                    # 该读出位置 × 层所有样本向量相同（如嵌入层上读出 token 相同）
    for layer in doc.get('layers') or []:
        pt = _parse_tag(layer.get('tag'))
        if pt and not layer.get('insufficient'):
            by_ro.setdefault(pt[0], []).append((pt[1], layer))
        elif pt and layer.get('degenerate'):
            degenerate.setdefault(pt[0], set()).add(pt[1])
    for f in doc.get('factors') or []:
        ros = {}
        for ro, rows in by_ro.items():
            rows.sort(key=lambda x: x[0])
            curve = []
            for i, layer in rows:
                r = (layer.get('factors') or {}).get(f)
                if not r:
                    continue
                lb = r['splits'].get('lockbox') or {}
                lf = r['splits'].get('lockbox_frame') or {}
                curve.append({'layer': i, 'sep': _r(lb.get('separation')), 'acc': _r(lb.get('probe_acc')),
                              'rep': _r(lb.get('rep_cos_white')), 'chance': r.get('chance'),
                              'eta2': _r((layer.get('trace_eta2') or {}).get(f)),
                              'readable': layer_readable(layer, f, CRITERIA['level3_split'], C),
                              'readable_confirm': layer_readable(layer, f, CRITERIA['level2_split'], C),
                              'frame_acc': _r(lf.get('probe_acc')),
                              'lexical_excess': _r(layer.get('lexical_excess')),
                              'length_corr': _r(r.get('length_corr_max'))})
            if curve:
                ros[ro] = _readout_summary(curve, C, CRITERIA['baseline_layer'] in degenerate.get(ro, ()))
        out['factors'][f] = ros
        if ros:
            sc = _scatter(doc, f, best_readout(ros), ros[best_readout(ros)])
            if sc:
                out.setdefault('scatter', {})[f] = sc
    return out


def best_readout(ros: Dict[str, Any]) -> str:
    """一张 piece 的代表读出位置：等级最高，其次峰值间隔最大，再按 target → cue → final → probe_answer 的顺序。"""
    order = {r: i for i, r in enumerate(READOUT_ORDER)}
    return max(ros, key=lambda ro: (ros[ro]['level'], (ros[ro]['peak'] or {}).get('sep') or -1, -order.get(ro, 9)))


def _scatter(doc: Dict[str, Any], f: str, ro: str, summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """代表读出位置峰值层上的二维点云（该因子的 LDA 前两轴，discovery 拟合），最多 MAX_SCATTER 个点（按序等距抽样）。"""
    if not summary.get('peak'):
        return None
    tag = f"{ro}@L{summary['peak']['layer']}"
    co = ((doc.get('coords') or {}).get(tag) or {}).get(f'lda:{f}')
    pts = doc.get('points') or []
    if not co or len(co) != len(pts):
        return None
    step = max(1, -(-len(pts) // MAX_SCATTER))
    rows = [[round(float(c[0]), 3), round(float(c[1]) if len(c) > 1 else 0.0, 3), str((p.get('cond') or {}).get(f)),
             p.get('split'), p.get('text') or ''] for c, p in list(zip(co, pts))[::step]]
    return {'readout': ro, 'layer': summary['peak']['layer'], 'tag': tag, 'axes': f'LDA（{f}，discovery 拟合，白化空间）',
            'levels': [str(x) for x in (doc.get('levels') or {}).get(f, [])], 'pts': rows,
            'n_total': len(pts), 'sampled': step > 1}


def layer_readable(layer: Dict[str, Any], f: str, sp: str, C: Dict[str, Any]) -> bool:
    # 与 analysis.findings 内的 readable 同一判据（不改 analysis.py：它的内容哈希是规律分析缓存键的一部分）；
    # test_pieces 核对两者在同一份分析上给出相同的可读出层
    r = (layer.get('factors') or {}).get(f)
    if not r:
        return False
    s = r['splits'].get(sp)
    perm = r.get('perm') or {}
    if not s or s.get('probe_acc') is None:
        return False
    ok = s['probe_acc'] >= r['chance'] + C['probe_margin']
    if perm.get('split') == sp:
        ok = ok and perm.get('sep_p', perm['acc_p']) <= C['perm_alpha']
    return ok


def _r(v, d: int = 4):
    return None if v is None else round(float(v), d)


def _segments(idx: List[int]) -> List[List[int]]:
    segs: List[List[int]] = []
    for i in sorted(idx):
        if segs and i == segs[-1][1] + 1:
            segs[-1][1] = i
        else:
            segs.append([i, i])
    return segs


def _baseline_state(b: Optional[Dict[str, Any]], identical: bool, C: Dict[str, Any]) -> str:
    """嵌入层基线：identical（所有样本向量相同）/ separable（lockbox 探针高于机会 + margin，不要求置换检验——基线从宽）/
    not_separable / missing（没有可用的嵌入层记录，无法排除 token 身份）。"""
    if identical:
        return 'identical'
    if b is None or b.get('acc') is None or b.get('chance') is None:
        return 'missing'
    return 'separable' if b['acc'] >= b['chance'] + C.get('probe_margin', 0.25) else 'not_separable'


def _readout_summary(curve: List[Dict[str, Any]], C: Dict[str, Any], identical: bool = False) -> Dict[str, Any]:
    base = CRITERIA['baseline_layer']
    b = next((c for c in curve if c['layer'] == base), None)
    baseline = _baseline_state(b, identical, C)
    token_identity = baseline == 'separable'
    readable = [c['layer'] for c in curve if c['readable'] and c['layer'] != base]
    confirm = [c['layer'] for c in curve if c['readable_confirm'] and c['layer'] != base]
    level = 1
    if confirm or readable:
        level = 2
    if readable and baseline in ('identical', 'not_separable'):
        level = 3
    peak = max((c for c in curve if c['layer'] in readable), key=lambda c: c['sep'] or -1, default=None)
    frame_general = None
    if peak and peak.get('frame_acc') is not None and peak.get('chance') is not None:
        frame_general = peak['frame_acc'] >= peak['chance'] + C.get('probe_margin', 0.25)
    lexical = [c['layer'] for c in curve if (c.get('lexical_excess') or 0) >= C.get('lexical_excess', 0.15)]
    length = [c['layer'] for c in curve if (c.get('length_corr') or 0) >= C.get('length_corr', 0.3)]
    return {'level': level, 'readable_layers': readable, 'segments': _segments(readable),
            'peak': {k: peak[k] for k in ('layer', 'sep', 'acc', 'rep')} if peak else None,
            'baseline': {'layer': base, 'state': baseline, 'acc': b['acc'] if b else None, 'sep': b['sep'] if b else None},
            'token_identity': token_identity, 'frame_general': frame_general,
            'lexical_layers': lexical, 'length_layers': length,
            'curve': [{k: c[k] for k in ('layer', 'sep', 'acc', 'eta2', 'readable')} for c in curve]}


def make_piece(ext: Dict[str, Any], *, family_id: str, tm_id: str, factor: str, item: Optional[str], model: str,
               synthetic: bool = False, languages: Iterable[str] = (), source: Optional[Dict[str, Any]] = None,
               behavior: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    ros = (ext.get('factors') or {}).get(factor)
    if not ros:
        return None
    best = best_readout(ros)
    sc = (ext.get('scatter') or {}).get(factor)
    return {'id': f'T:{tm_id}:{factor}:{model}', 'route': 'T', 'item': item, 'family_id': family_id, 'tm_id': tm_id,
            'factor': factor, 'model': model, 'synthetic': bool(synthetic), 'languages': sorted(set(languages)),
            'readout': best, 'level': ros[best]['level'], 'readouts': ros,
            'axes': {'read': ros[best]['level'], 'control': None, 'explain': None},
            'gate_excluded_cells': ext.get('gate_excluded_cells') or [], 'n_items': ext.get('n_items'),
            'scatter': sc if sc and sc.get('readout') == best else None, 'behavior': behavior or {},
            'source': source or {}}


def examples_of(items: List[Dict[str, Any]], factor: str, n_fill: int = 2, n_frame: int = 2) -> Dict[str, Any]:
    """对照例句：discovery 划分里同一填充在该因子各水平上都出现的句子（最小对），每个填充每水平最多 n_frame 句；
    另给每个水平一条探针（问题、可接受答案、对照答案）。只取 discovery，不展示留出材料。"""
    disc = [it for it in items if not it.get('canary') and it.get('split') == 'discovery' and factor in (it.get('cond') or {})]
    levels = sorted({str(it['cond'][factor]) for it in disc})
    by_fill: Dict[str, List[Dict[str, Any]]] = {}
    for it in disc:
        by_fill.setdefault(str(it.get('filler_id')), []).append(it)
    full = [f for f in sorted(by_fill) if {str(it['cond'][factor]) for it in by_fill[f]} >= set(levels)][:n_fill]
    rows = []
    for f in full:
        for lv in levels:
            for it in sorted((x for x in by_fill[f] if str(x['cond'][factor]) == lv), key=lambda x: str(x.get('frame_id')))[:n_frame]:
                rows.append({'filler': f, 'level': lv, 'frame': it.get('frame_id'), 'text': it.get('text'),
                             'target': it.get('target'), 'target_char': it.get('target_char')})
    probes = {}
    for lv in levels:
        it = next((x for x in disc if str(x['cond'][factor]) == lv and x.get('probe')), None)
        if it:
            pr = it['probe']
            probes[lv] = {'prompt': pr.get('prompt'), 'answers': pr.get('answers') or [pr.get('answer')], 'contrast': pr.get('contrast')}
    return {'levels': levels, 'rows': rows, 'probes': probes}


def rule_text(p: Dict[str, Any]) -> str:
    """一张 piece 的一句话规则（只转述数字，不加解释）。"""
    ro = p['readouts'][p['readout']]
    head = f"{p['family_id']}.{p['factor']}（{p['readout']} 读出位置）"
    if p['level'] <= 1:
        return f'{head}：各层都未达到可读出判据。'
    segs = '、'.join(f'L{a}' if a == b else f'L{a}–L{b}' for a, b in ro['segments']) or '—'
    pk = ro['peak'] or {}
    text = f"{head}：{segs} 在未见词表上可读出"
    if pk:
        text += f"，峰值 L{pk['layer']}（留出投影间隔 {pk['sep']}，探针准确率 {pk['acc']}）"
    text += {'identical': '；嵌入层 hs[0] 上各样本读出 token 相同（差异来自上下文计算）',
             'not_separable': '；嵌入层 hs[0] 不可分',
             'separable': '；嵌入层 hs[0] 已可分，差异可能来自读出位置的 token 身份（不计为留出复现）',
             'missing': '；没有嵌入层 hs[0] 记录，无法排除 token 身份（不计为留出复现）'}[ro['baseline']['state']]
    if ro['frame_general'] is not None:
        text += '；可推广到未见句式' if ro['frame_general'] else '；未能推广到未见句式'
    return text + '。'


# ---------------- 汇总：语言树状态、语言项详情、3D 叠加 ----------------

def assemble(tree: Dict[str, Any], pieces: List[Dict[str, Any]], skeletons: Dict[str, Dict[str, Any]],
             examples: Optional[Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]]] = None,
             selectivity: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    types = _type_of(tree)
    models = sorted({p['model'] for p in pieces} | set(skeletons))
    by_item: Dict[str, List[Dict[str, Any]]] = {}
    unmapped = []
    for p in pieces:
        if p['item'] in types:
            by_item.setdefault(p['item'], []).append(p)
        else:
            unmapped.append({k: p[k] for k in ('id', 'family_id', 'factor', 'model', 'level')})
    items_out, status = {}, {}
    for it in tree['items']:
        ps = sorted(by_item.get(it['id'], []), key=lambda p: (p['model'], p['family_id'], p['factor']))
        by_model: Dict[str, Dict[str, Any]] = {}
        for p in ps:
            m = by_model.setdefault(p['model'], {'read': 0, 'pieces': 0, 'families': set()})
            m['read'] = max(m['read'], p['level'])
            m['pieces'] += 1
            if p['level'] >= 3:
                m['families'].add(p['family_id'])
        l3_models = {p['model'] for p in ps if p['level'] >= 3}
        l3_fams = {p['family_id'] for p in ps if p['level'] >= 3}
        cross_model = len(l3_models) >= CRITERIA['min_cross']
        cross_family = any(len(m['families']) >= CRITERIA['min_cross'] for m in by_model.values())
        for m in by_model.values():
            if m['read'] >= 3 and (cross_model or len(m['families']) >= CRITERIA['min_cross']):
                m['read'] = 4
            m['families'] = sorted(m['families'])
        read = max([m['read'] for m in by_model.values()], default=0)
        st = {'read': read, 'read_label': LEVEL_LABEL[read], 'control': None, 'explain': None,
              'routes': {'T': {'pieces': len(ps), 'read': read}, 'N': {'pieces': 0, 'read': 0, 'note': '路线 N 尚未接入'}},
              'by_model': by_model, 'cross_model': cross_model, 'cross_family': cross_family,
              'families': sorted({p['family_id'] for p in ps}), 'level3_families': sorted(l3_fams),
              'languages_tested': sorted({lang for p in ps for lang in p['languages']}),
              'planned': bool(it.get('planned')) and not ps}
        status[it['id']] = st
        items_out[it['id']] = {'schema': 'lang_item.v1', 'item': {**it, 'type': types[it['id']]}, 'status': st,
                               'rules': [{'piece': p['id'], 'model': p['model'], 'level': p['level'], 'text': rule_text(p)} for p in ps],
                               'pieces': ps, 'criteria': CRITERIA, 'level_labels': LEVEL_LABEL,
                               'examples': [{'tm_id': tm, 'factor': f, 'family_id': fam, **ex}
                                            for (tm, f), (fam, ex) in sorted((examples or {}).items())
                                            if factor_item(tree, fam, f) == it['id']],
                               'selectivity': [s for s in (selectivity or []) if s.get('item') == it['id']]}
    tree_out = {'schema': 'lang_tree_status.v1', 'version': tree.get('version'), 'note': tree.get('note'),
                'types': tree['types'], 'categories': tree['categories'],
                'items': [{**it, 'type': types[it['id']], 'status': _status_brief(status[it['id']])} for it in tree['items']],
                'models': models, 'unmapped': unmapped, 'criteria': CRITERIA, 'level_labels': LEVEL_LABEL,
                'policy': '等级只描述路线 T 的可读出证据（未见词表、嵌入层基线、跨模型 / 跨族）；不表示被模型使用，也不是计算单元。'}
    overlays = {m: overlay(pieces, m, skeletons.get(m)) for m in models}
    return {'schema': SCHEMA, 'tree': tree_out, 'items': items_out, 'skeletons': skeletons, 'overlays': overlays}


def _status_brief(st: Dict[str, Any]) -> Dict[str, Any]:
    return {'read': st['read'], 'read_label': st['read_label'], 'pieces_T': st['routes']['T']['pieces'],
            'pieces_N': st['routes']['N']['pieces'], 'models': sorted(st['by_model']),
            'by_model': {m: v['read'] for m, v in st['by_model'].items()}, 'planned': st['planned'],
            'languages_tested': st['languages_tested'], 'cross_model': st['cross_model'], 'cross_family': st['cross_family']}


def overlay(pieces: List[Dict[str, Any]], model: str, skeleton: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """3D 默认场景的叠加数据：该模型上每个语言项的逐层曲线（只含标量与索引，不含方向向量）。"""
    rows = []
    for p in pieces:
        if p['model'] != model or not p['item']:
            continue
        ro = p['readouts'][p['readout']]
        rows.append({'item': p['item'], 'piece': p['id'], 'family_id': p['family_id'], 'factor': p['factor'],
                     'readout': p['readout'], 'level': p['level'], 'token_identity': ro['token_identity'],
                     'peak': ro['peak'], 'segments': ro['segments'],
                     'layers': [c['layer'] for c in ro['curve']], 'sep': [c['sep'] for c in ro['curve']],
                     'readable': [c['readable'] for c in ro['curve']]})
    return {'schema': 'space_overlay.v1', 'model': model, 'n_layers': (skeleton or {}).get('n_layers'),
            'route_T': rows, 'route_N': [], 'meet': [],
            'note': '路线 T：逐层留出投影间隔（lockbox）与可读出判定；路线 N 与会合边尚未接入。'}


CONFIG_KEYS = ('model_type', 'num_hidden_layers', 'hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
               'head_dim', 'hidden_act', 'rope_theta', 'rms_norm_eps', 'vocab_size', 'tie_word_embeddings', 'max_position_embeddings')


def skeleton_from_summary(model: str, summary: Dict[str, Any], census_runs: Iterable[Dict[str, Any]] = (),
                          census_config: Optional[Any] = None) -> Dict[str, Any]:
    """模型骨架：只取结果摘要、普查清单与普查保存的 config.json 里记录的真实结构参数，缺的字段留空。
    census_config(census_key, run) → config.json 内容（没有普查时为 None，注意力头数等字段就留空）。"""
    snap = (summary.get('model_snapshot') or {}).get('sha256') or ''
    prec = summary.get('precision') or ''
    census_key = f'{snap[:16]}-{prec}' if snap and prec else None
    run = max((r for r in census_runs if r.get('model_key') == census_key), key=lambda r: r.get('n_tokens') or 0, default=None)
    n_hs = summary.get('n_hidden_states')
    cfg = (census_config(census_key, run['run']) if census_config and run else None) or {}
    return {'schema': 'model_skeleton.v1', 'model': model, 'model_id': summary.get('model_id') or model.split('|')[0],
            'n_layers': (n_hs - 1) if n_hs else (run or {}).get('n_layers'),
            'd_model': summary.get('dim'), 'd_ff': (summary.get('mlp_units') or {}).get('d_ff') or (run or {}).get('d_ff'),
            'final_layer_norm': summary.get('final_layer_norm'), 'precision': prec, 'model_snapshot': snap or None,
            'census_key': census_key, 'census_run': (run or {}).get('run'),
            'hidden_state_boundary': summary.get('hidden_state_boundary'),
            'config': {k: cfg[k] for k in CONFIG_KEYS if k in cfg} or None,
            'note': '层数、宽度来自结果摘要与普查清单；3D 中层板内的位置是单元索引排布，不代表模型几何。'}


# ---------------- 中心端：从数据库与缓存收集 ----------------

def code_hash() -> str:
    src = ''.join(inspect.getsource(f) for f in (extract_analysis, _readout_summary, layer_readable, best_readout, _scatter))
    return hashlib.sha256((src + json.dumps(CRITERIA, sort_keys=True)).encode('utf-8')).hexdigest()[:12]


def _cached_extract(data_dir: Path, path: Path) -> Dict[str, Any]:
    """规律分析文件可达几十 MB：每个文件只解析一次，抽取结果按（代码哈希, 文件名）落盘。"""
    d = Path(data_dir) / 'lang_cache'
    d.mkdir(parents=True, exist_ok=True)
    out = d / f'{code_hash()}-{path.name}'
    if out.is_file():
        try:
            return json.loads(out.read_text(encoding='utf-8'))
        except ValueError:
            pass
    with path.open(encoding='utf-8') as fh:
        ext = extract_analysis(json.load(fh))
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(ext, ensure_ascii=False), encoding='utf-8')
    tmp.replace(out)
    return ext


def collect(db, data_dir: Path, tree: Optional[Dict[str, Any]] = None, census_runs: Iterable[Dict[str, Any]] = (),
            log=print, census_config=None, selectivity: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """已冻结正式模板（不含试跑）× 模型桶 × 因子 → piece；只读已缓存的规律分析，不在这里计算。
    另附：对照例句（冻结语料 discovery 划分）、行为门逐格结果、普查模型的 config 与模板选择性（selectivity，由调用方提供）。"""
    from . import pipeline as P
    tree = tree or load_tree()
    census_runs = list(census_runs)
    pieces, skeletons, missing, examples = [], {}, [], {}
    for t, c in P.templates_v3(db):
        if not P.is_v3(c) or c.get('mode') == 'pilot':
            continue
        tm, fam = t['tm_id'], c.get('family_id') or t['tm_id']
        agg = P.aggregate(db, data_dir, tm, gate=CRITERIA['gate'])
        langs = (tree.get('family_languages') or {}).get(fam) or []
        for f in c.get('factors') or []:
            examples[(tm, f)] = (fam, examples_of(c.get('items') or [], f))
        for b in agg['buckets']:
            if not b.get('agg'):
                continue
            model = b['bucket']
            if model not in skeletons and b.get('primary_shas'):
                row = db.execute('SELECT summary_json FROM results WHERE sha=?', (b['primary_shas'][0],)).fetchone()
                if row:
                    skeletons[model] = skeleton_from_summary(model, P.summary_of(row), census_runs, census_config)
            path = P.analysis_cached_path(db, data_dir, tm, model, CRITERIA['gate'])
            if path is None:
                missing.append(f'{tm} · {model}')
                continue
            try:
                ext = _cached_extract(data_dir, Path(path))
            except Exception as exc:
                log(f'[pieces] {tm} {model}: {type(exc).__name__}: {exc}')
                continue
            for f in c.get('factors') or []:
                p = make_piece(ext, family_id=fam, tm_id=tm, factor=f, item=factor_item(tree, fam, f, c.get('lang_items')),
                               model=model, synthetic=b.get('synthetic', False), languages=langs,
                               source={'tm_id': tm, 'bucket': model, 'design_sha': t['design_sha'],
                                       'analysis_cache': Path(path).name, 'gate': CRITERIA['gate'],
                                       'shards': b.get('primary_shas') or []},
                               behavior={'cells': b.get('behavior') or {}, 'mode': b.get('gate_mode'),
                                         'threshold': b.get('behavior_threshold'), 'excluded': b.get('gate_excluded_cells') or []})
                if p:
                    pieces.append(p)
    out = assemble(tree, pieces, skeletons, examples, selectivity)
    out['missing_analyses'] = missing
    out['code_hash'] = code_hash()
    return out
