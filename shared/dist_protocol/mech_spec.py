# -*- coding: utf-8 -*-
"""机制基准特异性实验 mech_spec.v1（SP-v1）：MB-v1 的高保真节点是任务专属，还是通用主干。

MB-v1 的 lockbox 已开封，不能再用于确认，这里另备全新填充词与句式（sealed），只评测一次。

四个问题，判据固定在 CRITERIA，随结果返回：
  H1 细预算    只移除前 1% 节点时，原始排序是否已经特异。
  H2 去主干    对比排序（本任务份额 - 其他任务最大份额）或排除自然文本主干后，是否变得特异。
  H3 主干之上  始终保留主干时，任务节点能否用很小的集合补全行为：
               f_bb(C) = mean(m(B∪C) - m(B)) / mean(m(全) - m(B))。
  H4 成组保留  exact 落后 atp_ig 是否来自单节点移除漏掉成组节点：迭代移除能否补回差距。

drop(C) = 1 - f(去掉 C)；特异 = 目标下降够大、比其他任务大、区间下界 > 0，且自然文本 KL 不高于随机太多。
本模块只放材料、判据和统计（纯 numpy）；GPU 部分在 scripts/mech_spec_run.py。
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from shared.dist_protocol import mech_bench as B

SCHEMA = 'mech_spec.v1'

LADDER = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
BB_K = [1, 2, 4, 8, 16, 32, 64, 128, 256]
REMOVE_METHODS = ['atp_ig', 'exact', 'ig_contrast', 'exact_contrast', 'ig_nobb', 'exact_iter']
BB_METHODS = ['ig_nobb', 'exact_nobb', 'ig_contrast']
GROUP_METHODS = ['atp_ig', 'exact', 'exact_iter', 'exact_denoise']
ABLATIONS = ['cf', 'mean']

CRITERIA = {
    'schema': SCHEMA,
    'primary_budget': 0.01,        # H1 / H2 在这个预算上判
    'target_drop_min': 0.2, 'ratio_min': 2.0,
    'natural_ratio_max': 2.0,      # 自然文本 KL 不超过同规模随机集合的倍数
    'tasks_needed': 2,             # 三项任务中至少几项特异，才算这个模型上支持
    'backbone_frac': 0.05,         # 主干 = 自然文本单节点 KL 前 5%（只用定义窗口）
    'bb_gate_min': 0.2,            # 只留主干时至少缺这么多行为，否则 H3 无检验力
    'bb_f': 0.8,
    'gap_min': 0.05, 'gap_closed_min': 0.5,    # H4：atp_ig 领先 exact 至少这么多才判；补回一半算支持
    'iter_keep': 0.6, 'iter_stop': 8, 'n_sub': 64,
    'n_sealed': 320, 'n_draws': 3, 'n_boot': 2000, 'boot_seed': 0, 'ci': 0.95,
    'expected_delta': 0.05, 'p_d': 0.10,
    'natural_windows': 16, 'natural_seq': 128,
}

# 全新材料：与 MB-v1 的填充词、句式都不重叠
SEALED_NAMES = ['Nora', 'Owen', 'Ella', 'Ivan', 'Leah', 'Hugo', 'Maya', 'Noah', 'Tara', 'Dean', 'Gina', 'Neil', 'Vera', 'Seth', 'Iris',
                'Kyle', 'Beth', 'Evan', 'Dora', 'Glen', 'Lucy', 'Ross', 'Jill', 'Todd', 'Fiona', 'Colin', 'Wendy', 'Bruce', 'Holly', 'Gavin',
                'Paula', 'Keith', 'Molly', 'Harry', 'Sally', 'Roger', 'Nina', 'Philip', 'Claire', 'Dennis']
SEALED_PLACES = ['library', 'market', 'museum', 'theater', 'park', 'airport']
SEALED_OBJECTS = ['pen', 'watch', 'camera', 'guitar', 'wallet', 'blanket']
SEALED_CAPITALS = {'Uruguay': 'Montevideo', 'Ecuador': 'Quito', 'Jamaica': 'Kingston', 'Mongolia': 'Ulaanbaatar', 'Cambodia': 'Phnom',
                   'Laos': 'Vientiane', 'Malaysia': 'Kuala', 'Bangladesh': 'Dhaka', 'Iceland': 'Reykjavik', 'Croatia': 'Zagreb',
                   'Slovakia': 'Bratislava', 'Slovenia': 'Ljubljana', 'Lithuania': 'Vilnius', 'Latvia': 'Riga', 'Estonia': 'Tallinn',
                   'Albania': 'Tirana', 'Georgia': 'Tbilisi', 'Armenia': 'Yerevan', 'Azerbaijan': 'Baku', 'Kazakhstan': 'Astana',
                   'Uzbekistan': 'Tashkent', 'Israel': 'Jerusalem', 'Yemen': 'Sanaa', 'Oman': 'Muscat', 'Qatar': 'Doha', 'Libya': 'Tripoli',
                   'Sudan': 'Khartoum', 'Uganda': 'Kampala', 'Zambia': 'Lusaka', 'Zimbabwe': 'Harare', 'Senegal': 'Dakar', 'Mali': 'Bamako',
                   'Angola': 'Luanda', 'Mozambique': 'Maputo', 'Netherlands': 'Amsterdam', 'Switzerland': 'Bern', 'Taiwan': 'Taipei',
                   'Brazil': 'Brasilia', 'Bolivia': 'Sucre', 'Paraguay': 'Asuncion'}
SEALED_NOUNS = [('baker', 'bakers'), ('pilot', 'pilots'), ('actor', 'actors'), ('judge', 'judges'), ('chef', 'chefs'), ('clerk', 'clerks'),
                ('guard', 'guards'), ('tiger', 'tigers'), ('snake', 'snakes'), ('whale', 'whales'), ('spider', 'spiders'),
                ('rabbit', 'rabbits'), ('monkey', 'monkeys'), ('clock', 'clocks'), ('mirror', 'mirrors'), ('pillow', 'pillows'), ('basket', 'baskets'),
                ('candle', 'candles'), ('hammer', 'hammers'), ('ladder', 'ladders'), ('jacket', 'jackets'), ('glove', 'gloves'), ('sandwich', 'sandwiches'),
                ('tractor', 'tractors'), ('rocket', 'rockets'), ('island', 'islands'), ('valley', 'valleys'), ('forest', 'forests'), ('castle', 'castles'),
                ('temple', 'temples'), ('village', 'villages'), ('poet', 'poets'), ('dancer', 'dancers'), ('sailor', 'sailors'), ('hunter', 'hunters'),
                ('tailor', 'tailors'), ('barber', 'barbers'), ('wizard', 'wizards'), ('pirate', 'pirates')]
SEALED_FRAMES = {
    'kn_capital': ['Everyone in {C} knows that the capital is', 'If you travel to {C}, the capital you will reach is'],
    'rs_ioi': ['Later, {B} and {A} walked to the {P}, and {B} threw a {O} to',
               'During lunch at the {P}, {B} and {A} chatted, and {B} showed a {O} to'],
    'gr_agree': ['The {N} across from the {X}', 'The {N} under the {X}'],
}


def build_sealed(task: str, n: int, enc: Callable[[str], List[int]], *, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(f'{SCHEMA}|{task}|sealed|{seed}')
    frames = list(enumerate(SEALED_FRAMES[task]))
    plural = dict(SEALED_NOUNS)
    items, seen, tried, dropped = [], set(), 0, 0
    while len(items) < n and tried < n * 200:
        tried += 1
        fi, frame = frames[tried % len(frames)]
        if task == 'rs_ioi':
            a, b, c, d, e = rng.sample(SEALED_NAMES, 5)
            p, o = rng.choice(SEALED_PLACES), rng.choice(SEALED_OBJECTS)
            clean = frame.format(A=a, B=b, P=p, O=o)
            first, _, tail = frame.partition('{B}')
            corrupt = first.format(P=p, O=o) + d + tail.format(A=c, B=e, P=p, O=o)
            key, family, ans, con = clean, a, a, b
        elif task == 'kn_capital':
            x, y = rng.sample(list(SEALED_CAPITALS), 2)
            clean, corrupt = frame.format(C=x), frame.format(C=y)
            key, family, ans, con = clean + y, x, SEALED_CAPITALS[x], SEALED_CAPITALS[y]
        else:
            s, t = rng.sample(list(plural), 2)
            attractor = rng.choice([t, plural[t]])
            sing = rng.random() < 0.5
            subj, other = (s, plural[s]) if sing else (plural[s], s)
            clean, corrupt = frame.format(N=subj, X=attractor), frame.format(N=other, X=attractor)
            key, family, ans, con = clean, s, 'is' if sing else 'are', 'are' if sing else 'is'
        if key in seen:
            continue
        seen.add(key)
        item = B._item(task, 'sealed', fi, family, clean, corrupt, ans, con, enc)
        if item is None:
            dropped += 1
            continue
        items.append(item)
    return {'items': items, 'dropped': dropped, 'tried': tried, 'n_families': len({i['family'] for i in items})}


def design_sha(code_hash: str = '') -> str:
    body = json.dumps({'criteria': CRITERIA, 'ladder': LADDER, 'bb': BB_K, 'rm': REMOVE_METHODS, 'bbm': BB_METHODS, 'gm': GROUP_METHODS,
                       'names': SEALED_NAMES, 'capitals': SEALED_CAPITALS, 'nouns': SEALED_NOUNS, 'frames': SEALED_FRAMES, 'code': code_hash},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


# ---------------- 排序 ----------------

def ladder_counts(n: int) -> List[int]:
    return sorted(set(B.budget_counts(n, LADDER)))


def primary_k(n: int) -> int:
    return B.budget_counts(n, [CRITERIA['primary_budget']])[0]


def shares(score: np.ndarray) -> np.ndarray:
    s = np.nan_to_num(np.asarray(score, float))
    total = s.sum()
    return s / total if total > 0 else s


def contrast_score(scores: Dict[str, np.ndarray], task: str) -> np.ndarray:
    """本任务份额减去其他任务的最大份额：多任务都高的节点被压下去。"""
    others = np.max([shares(v) for t, v in scores.items() if t != task], axis=0)
    return shares(scores[task]) - others


def order_of(score: np.ndarray) -> np.ndarray:
    return np.argsort(-np.nan_to_num(score, nan=-np.inf), kind='stable')


def backbone_of(natural_kl: np.ndarray) -> np.ndarray:
    n = len(natural_kl)
    k = int(np.ceil(n * CRITERIA['backbone_frac']))
    return order_of(natural_kl)[:k]


def excluding(order: np.ndarray, banned: np.ndarray) -> np.ndarray:
    bad = np.zeros(len(order), dtype=bool)
    bad[banned] = True
    return np.concatenate([order[~bad[order]], order[bad[order]]])


def iterative_order(n: int, effect_of: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """从全部节点起，每轮在当前保留集合里量单节点效应，去掉最弱的一批；后被去掉的排在前面。

    effect_of(alive) 返回 alive 中每个节点移除后的 |效应|，顺序与 alive 相同。
    """
    alive = np.arange(n)
    dropped: List[np.ndarray] = []
    while len(alive) > CRITERIA['iter_stop']:
        eff = np.asarray(effect_of(alive), float)
        by = np.argsort(-eff, kind='stable')
        keep = max(CRITERIA['iter_stop'], int(len(alive) * CRITERIA['iter_keep']))
        dropped.append(alive[by[keep:]])
        alive = alive[by[:keep]]
    eff = np.asarray(effect_of(alive), float)
    return np.concatenate([alive[np.argsort(-eff, kind='stable')]] + dropped[::-1])


# ---------------- 键 ----------------

def rkey(abl, method, k, src='self', draw=0) -> str:
    return f'R|{abl}|{method}|{k}|{src}|{draw}'


def bkey(method, k, draw=0) -> str:
    return f'B|{method}|{k}|{draw}'


def gkey(method, budget) -> str:
    return f'G|{method}|{budget:g}'


# ---------------- 统计 ----------------

def drop_of(m, full, empty) -> float:
    return 1 - B.faithfulness(m, full, empty)


def _ci(values) -> Dict[str, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    a = (1 - CRITERIA['ci']) / 2
    return {'lo': float(np.quantile(v, a)), 'hi': float(np.quantile(v, 1 - a))}


def spec_cell(raws: Dict[str, Dict[str, np.ndarray]], clusters: Dict[str, Sequence[str]], task: str, method: str, k: int, *,
              abl: str = 'cf', natural: Optional[float] = None, natural_random: Optional[float] = None, n_boot=None) -> Dict[str, Any]:
    """task 的集合 method@k：目标任务下降、其他任务下降、差的聚类区间、自然文本倍数。"""
    n_boot = n_boot or CRITERIA['n_boot']
    others = [t for t in raws if t != task]
    if task not in raws or len(others) != 2:
        return {'status': 'pending', 'note': '三项任务不全'}
    tk = rkey(abl, method, k)
    ok = rkey(abl, method, k, src=task)
    empty = f'empty|{abl}'
    if tk not in raws[task] or any(ok not in raws[o] for o in others):
        return {'status': 'pending', 'note': '缺移除记录'}
    d = lambda t, key, i=slice(None): drop_of(raws[t][key][i], raws[t]['clean'][i], raws[t][empty][i])
    target = d(task, tk)
    offs = {o: d(o, ok) for o in others}
    worst = max(0.0, max(offs.values()))
    row: Dict[str, Any] = {'target_drop': target, 'off_drops': offs, 'worst_off': worst, 'ratio': target / worst if worst > 0 else float('inf')}
    boots = {t: B.cluster_boot_indices(clusters[t], n_boot, CRITERIA['boot_seed'] + j) for j, t in enumerate(sorted(raws))}
    diffs = [d(task, tk, boots[task][b]) - max(d(o, ok, boots[o][b]) for o in others) for b in range(n_boot)]
    row['diff_ci'] = _ci(diffs)
    if natural is not None and natural_random:
        row['natural_kl'], row['natural_ratio'] = natural, natural / natural_random
    if target < CRITERIA['target_drop_min']:
        row['status'] = 'no_effect'
    elif row['ratio'] < CRITERIA['ratio_min'] or row['diff_ci']['lo'] <= 0:
        row['status'] = 'not_specific'
    elif 'natural_ratio' not in row:
        row['status'] = 'pending'
    elif row['natural_ratio'] > CRITERIA['natural_ratio_max']:
        row['status'] = 'broad_natural'
    else:
        row['status'] = 'specific'
    return row


def bb_score(raw: Dict[str, np.ndarray], clusters: Sequence[str], ks: Sequence[int], method: str, *, n_boot=None) -> Dict[str, Any]:
    """始终保留主干时的补全曲线。"""
    full, base, empty = raw['clean'], raw['bb_only'], raw['empty|cf']
    missing = 1 - B.faithfulness(base, full, empty)
    out: Dict[str, Any] = {'missing': missing}
    if missing < CRITERIA['bb_gate_min']:
        return {**out, 'status': 'no_power', 'note': '只留主干时行为几乎完整，这项没有检验力'}
    curve = {k: raw.get(bkey(method, k)) for k in ks}
    rand = [{k: raw.get(bkey('random', k, d)) for k in ks} for d in range(CRITERIA['n_draws'])]
    if any(v is None for v in curve.values()) or any(v is None for r in rand for v in r.values()):
        return {**out, 'status': 'pending', 'note': '缺记录'}
    f = lambda c, k, i=slice(None): B.faithfulness(c[k][i], full[i], base[i])
    area = lambda c, i=slice(None): B.curve_area(list(ks), [f(c, k, i) for k in ks])
    diff = lambda i=slice(None): area(curve, i) - float(np.mean([area(r, i) for r in rand]))
    boots = B.cluster_boot_indices(clusters, n_boot or CRITERIA['n_boot'], CRITERIA['boot_seed'])
    need = [k for k in ks if f(curve, k) >= CRITERIA['bb_f']]
    ci = _ci([diff(i) for i in boots])
    return {**out, 'status': 'done', 'curve': {str(k): f(curve, k) for k in ks}, 'random_curve': {str(k): float(np.mean([f(r, k) for r in rand])) for k in ks},
            'auc': area(curve), 'random_auc': float(np.mean([area(r) for r in rand])), 'diff': diff(), 'diff_ci': ci,
            'k_f80': need[0] if need else None, 'beats_random': ci['lo'] > 0}


def group_score(raw: Dict[str, np.ndarray], clusters: Sequence[str], *, n_boot=None) -> Dict[str, Any]:
    """H4：只有 atp_ig 明显领先 exact 时才判。"""
    full, empty = raw['clean'], raw['empty|cf']
    curves = {m: {b: raw.get(gkey(m, b)) for b in B.BUDGETS} for m in GROUP_METHODS}
    if any(v is None for c in curves.values() for v in c.values()):
        return {'status': 'pending', 'note': '缺记录'}
    auc = {m: B.auc_f(c, full, empty) for m, c in curves.items()}
    gap = auc['atp_ig'] - auc['exact']
    out: Dict[str, Any] = {'auc': auc, 'gap': gap}
    if gap < CRITERIA['gap_min']:
        return {**out, 'status': 'no_gap'}
    for name in ('exact_iter', 'exact_denoise'):
        ci = B.paired_auc_diff(curves[name], [curves['exact']], full, empty, clusters, n_boot=n_boot or CRITERIA['n_boot'])
        closed = (auc[name] - auc['exact']) / gap
        out[name] = {'vs_exact': ci, 'closed': closed, 'verdict': '支持' if ci['lo'] > 0 and closed >= CRITERIA['gap_closed_min'] else '未支持'}
    out['status'] = 'judged'
    return out


def summarize(cells: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """cells[method][task] → 单元。三项都有结果才下结论。"""
    out = {}
    for method, row in cells.items():
        states = [c.get('status') for c in row.values()]
        count = sum(s == 'specific' for s in states)
        if len(states) < len(B.TASKS) or 'pending' in states:
            out[method] = {'verdict': '待检', 'specific_tasks': count}
        else:
            out[method] = {'verdict': '支持' if count >= CRITERIA['tasks_needed'] else '未支持', 'specific_tasks': count}
    return out
