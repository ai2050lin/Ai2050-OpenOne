# -*- coding: utf-8 -*-
"""统一机制评测基准 mech_bench.v1：同一冻结任务、同一预算下比较定位方法。

本模块只放材料、判据和统计（纯 numpy，可单测）；GPU 部分在 scripts/mech_bench_run.py。

问题：给定一个语言行为，方法 M 选出的前 k 个组件，能否在没见过的词和句式上保留这个行为，
去掉后是否只伤害这个行为。

  faithfulness  f(C) = mean(m(C) - m(空)) / mean(m(全) - m(空))
                C 之外的组件全部换成消融值，m 是 logit(答案) - logit(对照答案)。
  曲线面积      auc_f：f 对 log10(预算) 的梯形面积，除以横轴长度；曲线距离 cmd：|1 - f| 的同样面积。
                形式借鉴 MIB 的 CPR / CMD，但这里是节点级、本项目自建材料，不是 MIB 官方分数。
  必要性        drop(C) = 1 - f(去掉 C，其余保留)。
  脱靶          同一集合在其他任务上的 drop，以及自然文本上的 KL。

判据固定在 CRITERIA，随结果返回；缺条件时返回「待检」，不用有值的子集代替。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = 'mech_bench.v1'

BUDGETS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
UNIT_BUDGETS = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0]
DICT_K = [1, 4, 16, 64, 256, 1024]
OFF_TARGET_BUDGETS = [0.01, 0.05, 0.2]

METHODS = ['act_diff', 'dla', 'atp', 'atp_ig', 'exact']     # 进入比较的方法；random / matched 是负控制
ABLATIONS = ['cf', 'mean', 'zero']
PRIMARY_ABLATION = 'cf'

CRITERIA = {
    'schema': SCHEMA,
    'min_clean_acc': 0.8,          # 行为门：干净句 m > 0 的比例（discovery）
    'min_gap': 1.0,                # 行为门：mean m(干净) - mean m(反事实)，单位 nat
    'identity_tol': 0.02,          # 仪器门：|f(全部保留) - 1| 与 |f(全部替换)| 的上限
    'ci': 0.95, 'n_boot': 2000, 'boot_seed': 0,
    'ablation_spearman_min': 0.7,  # 方法排序跨消融方式的一致性下限
    'specific_budget': 0.05,       # 特异性在这个预算上判
    'specific_ratio_min': 2.0,     # 目标下降 / 最大脱靶下降 的下限
    'specific_target_drop_min': 0.2,   # 目标下降太小时不判「特异」
    'dict_sensitivity_min': 0.02,  # 字典对比：整层替换至少改变缺口的这个比例，否则记「无检验力」
    'n_discovery': 256, 'n_confirm': 320, 'n_lockbox': 320,
    'expected_delta': 0.05, 'p_d': 0.10,   # AGENTS §13：预期效应 → 所需 n → 检验力
    'ig_steps': 5, 'n_random': 3, 'n_matched': 3,
}

# Wang et al. 2022 列出的 GPT-2 small IOI 电路 26 个头
KNOWN_IOI_HEADS = ['9.9', '9.6', '10.0', '10.10', '10.6', '10.2', '10.1', '11.2', '9.7', '9.0', '11.9',
                   '10.7', '11.10', '7.3', '7.9', '8.6', '8.10', '5.5', '5.8', '5.9', '6.9',
                   '0.1', '0.10', '3.0', '2.2', '4.11']

NAMES = ['Mary', 'John', 'Tom', 'James', 'Dan', 'Sid', 'Martin', 'Amy', 'Kate', 'Paul', 'Mark', 'Lisa', 'Anna', 'Jack',
         'Sam', 'Ben', 'Alex', 'David', 'Laura', 'Emma', 'Mike', 'Sarah', 'Peter', 'Kevin', 'Robert', 'Susan', 'Linda',
         'Karen', 'Brian', 'Eric', 'Steve', 'Adam', 'Ryan', 'Jason', 'Scott', 'Jeff', 'Frank', 'Henry', 'Rachel', 'Julia',
         'Grace', 'Helen', 'Alice', 'Oliver', 'Victor', 'Simon', 'Nancy', 'Diana', 'George', 'Edward', 'Bill', 'Luke',
         'Carl', 'Ruth', 'Rose', 'Jane', 'Ann', 'Tim', 'Joe', 'Fred']
PLACES = ['store', 'garden', 'restaurant', 'school', 'hospital', 'office', 'house', 'station']
OBJECTS = ['ring', 'book', 'bone', 'basketball', 'computer', 'necklace', 'drink', 'snack']
IOI_FRAMES = [
    'Then, {B} and {A} went to the {P}. {B} gave a {O} to',
    'When {B} and {A} got a {O} at the {P}, {B} decided to give it to',
    'After {B} and {A} went to the {P}, {B} gave a {O} to',
    'While {B} and {A} were working at the {P}, {B} gave a {O} to',
    'Friends {B} and {A} found a {O} at the {P}. {B} gave it to',
    'Then, {B} and {A} had a long argument at the {P}, and afterwards {B} handed a {O} to',
    'Yesterday {B} and {A} met at the {P}, and {B} passed the {O} to',
    'At the {P}, {B} and {A} talked for an hour, then {B} sent a {O} to',
]
CAPITALS = {'France': 'Paris', 'Japan': 'Tokyo', 'Germany': 'Berlin', 'Italy': 'Rome', 'Spain': 'Madrid', 'Russia': 'Moscow',
            'China': 'Beijing', 'England': 'London', 'Egypt': 'Cairo', 'Greece': 'Athens', 'Austria': 'Vienna', 'Thailand': 'Bangkok',
            'Ireland': 'Dublin', 'Norway': 'Oslo', 'Sweden': 'Stockholm', 'Finland': 'Helsinki', 'Denmark': 'Copenhagen',
            'Portugal': 'Lisbon', 'Poland': 'Warsaw', 'Hungary': 'Budapest', 'Turkey': 'Ankara', 'Iran': 'Tehran', 'Iraq': 'Baghdad',
            'Cuba': 'Havana', 'Peru': 'Lima', 'Kenya': 'Nairobi', 'India': 'Delhi', 'Canada': 'Ottawa', 'Australia': 'Canberra',
            'Argentina': 'Buenos', 'Belgium': 'Brussels', 'Czechia': 'Prague', 'Indonesia': 'Jakarta', 'Philippines': 'Manila',
            'Vietnam': 'Hanoi', 'Lebanon': 'Beirut', 'Syria': 'Damascus', 'Jordan': 'Amman', 'Ukraine': 'Kyiv', 'Romania': 'Bucharest',
            'Serbia': 'Belgrade', 'Bulgaria': 'Sofia', 'Colombia': 'Bogota', 'Chile': 'Santiago', 'Venezuela': 'Caracas',
            'Scotland': 'Edinburgh', 'Wales': 'Cardiff', 'Nepal': 'Kathmandu', 'Afghanistan': 'Kabul', 'Pakistan': 'Islamabad',
            'Ethiopia': 'Addis', 'Morocco': 'Rabat', 'Tunisia': 'Tunis', 'Algeria': 'Algiers', 'Nigeria': 'Abuja', 'Ghana': 'Accra'}
CAPITAL_FRAMES = ['The capital of {C} is', 'The capital city of {C} is', 'In {C}, the capital city is',
                  'The name of the capital of {C} is', 'As everyone knows, the capital of {C} is',
                  'The largest government offices of {C} are in the capital, which is', '{C} has a capital, and that capital is']
NOUNS = [('key', 'keys'), ('dog', 'dogs'), ('cat', 'cats'), ('book', 'books'), ('car', 'cars'), ('teacher', 'teachers'),
         ('student', 'students'), ('doctor', 'doctors'), ('boy', 'boys'), ('girl', 'girls'), ('tree', 'trees'), ('house', 'houses'),
         ('river', 'rivers'), ('road', 'roads'), ('bird', 'birds'), ('table', 'tables'), ('chair', 'chairs'), ('door', 'doors'),
         ('window', 'windows'), ('flower', 'flowers'), ('farmer', 'farmers'), ('singer', 'singers'), ('writer', 'writers'),
         ('player', 'players'), ('worker', 'workers'), ('driver', 'drivers'), ('friend', 'friends'), ('neighbor', 'neighbors'),
         ('lawyer', 'lawyers'), ('painter', 'painters'), ('officer', 'officers'), ('soldier', 'soldiers'), ('engineer', 'engineers'),
         ('manager', 'managers'), ('visitor', 'visitors'), ('nurse', 'nurses'), ('horse', 'horses'), ('lamp', 'lamps'),
         ('bottle', 'bottles'), ('garden', 'gardens'), ('bridge', 'bridges'), ('picture', 'pictures'), ('letter', 'letters'),
         ('machine', 'machines'), ('computer', 'computers'), ('phone', 'phones'), ('shirt', 'shirts'), ('boat', 'boats'),
         ('train', 'trains'), ('plane', 'planes'), ('stone', 'stones'), ('apple', 'apples'), ('hat', 'hats'), ('cup', 'cups'),
         ('plate', 'plates'), ('box', 'boxes'), ('bus', 'buses'), ('wall', 'walls'), ('tower', 'towers'), ('shop', 'shops')]
AGREE_FRAMES = ['The {N} near the {X}', 'The {N} behind the {X}', 'The {N} on the {X}', 'The {N} by the {X}',
                'The {N} next to the {X}', 'The {N} beside the old {X}', 'The {N} in front of the {X}']

TASKS = {
    'kn_capital': {'part': '知识网络', 'label': '国家 → 首都', 'frames': CAPITAL_FRAMES, 'lockbox_frames': 2},
    'rs_ioi': {'part': '推理功能', 'label': '间接宾语识别', 'frames': IOI_FRAMES, 'lockbox_frames': 2},
    'gr_agree': {'part': '语法体系', 'label': '主谓一致', 'frames': AGREE_FRAMES, 'lockbox_frames': 2},
}


def split_fillers(words: Sequence[str], salt: str) -> Dict[str, List[str]]:
    """按填充词确定性切分（40 / 30 / 30），与分词器无关。"""
    order = sorted(words, key=lambda w: hashlib.sha256(f'{salt}|{w}'.encode('utf-8')).hexdigest())
    a, b = int(len(order) * 0.4), int(len(order) * 0.7)
    return {'discovery': order[:a], 'confirm': order[a:b], 'lockbox': order[b:]}


def frames_of(task: str, split: str) -> List[Tuple[int, str]]:
    spec = TASKS[task]
    frames = list(enumerate(spec['frames']))
    cut = len(frames) - spec['lockbox_frames']
    return frames[cut:] if split == 'lockbox' else frames[:cut]


def _item(task, split, frame, family, clean, corrupt, a, b, enc):
    ca, co = enc(clean), enc(corrupt)
    ta, tb = enc(' ' + a), enc(' ' + b)
    if len(ca) != len(co) or not ta or not tb or ta[0] == tb[0] or ca[-1] != co[-1]:
        return None
    return {'task': task, 'split': split, 'frame': frame, 'family': family, 'clean': clean, 'corrupt': corrupt,
            'answer': a, 'contrast': b, 'clean_ids': ca, 'corrupt_ids': co, 'a': ta[0], 'b': tb[0]}


def build_items(task: str, split: str, n: int, enc: Callable[[str], List[int]], *, seed: int = 0) -> Dict[str, Any]:
    """生成一个划分的材料。enc 是模型分词器；长度不等或答案首 token 相同的样本丢弃，并计数。"""
    rng = random.Random(f'{SCHEMA}|{task}|{split}|{seed}')
    frames = frames_of(task, split)
    items, seen, tried, dropped = [], set(), 0, 0
    if task == 'rs_ioi':
        names = split_fillers(NAMES, 'ioi')[split]
    elif task == 'kn_capital':
        countries = split_fillers(list(CAPITALS), 'capital')[split]
    else:
        nouns = split_fillers([p[0] for p in NOUNS], 'agree')[split]
        plural = dict(NOUNS)
    while len(items) < n and tried < n * 200:
        tried += 1
        fi, frame = frames[tried % len(frames)]
        if task == 'rs_ioi':
            a, b, c, d, e = rng.sample(names, 5)
            p, o = rng.choice(PLACES), rng.choice(OBJECTS)
            clean = frame.format(A=a, B=b, P=p, O=o)
            first, _, tail = frame.partition('{B}')     # ABC 破坏：三个位置都换名字，第二次出现的主语换成 e
            corrupt = first.format(P=p, O=o) + d + tail.format(A=c, B=e, P=p, O=o)
            key, family, ans, con = clean, a, a, b
        elif task == 'kn_capital':
            x, y = rng.sample(countries, 2)
            clean, corrupt = frame.format(C=x), frame.format(C=y)
            key, family, ans, con = clean + y, x, CAPITALS[x], CAPITALS[y]
        else:
            s, t = rng.sample(nouns, 2)
            attractor = rng.choice([t, plural[t]])
            sing = rng.random() < 0.5
            subj, other = (s, plural[s]) if sing else (plural[s], s)
            clean, corrupt = frame.format(N=subj, X=attractor), frame.format(N=other, X=attractor)
            key, family, ans, con = clean, s, 'is' if sing else 'are', 'are' if sing else 'is'
        if key in seen:
            continue
        seen.add(key)
        item = _item(task, split, fi, family, clean, corrupt, ans, con, enc)
        if item is None:
            dropped += 1
            continue
        items.append(item)
    return {'items': items, 'dropped': dropped, 'tried': tried, 'n_families': len({i['family'] for i in items})}


def design_sha(code_hash: str = '') -> str:
    body = json.dumps({'criteria': CRITERIA, 'budgets': BUDGETS, 'unit': UNIT_BUDGETS, 'dict_k': DICT_K, 'off': OFF_TARGET_BUDGETS,
                       'methods': METHODS, 'names': NAMES, 'capitals': CAPITALS, 'nouns': NOUNS,
                       'frames': {k: v['frames'] for k, v in TASKS.items()}, 'code': code_hash},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


# ---------------- 统计 ----------------

def budget_counts(n: int, budgets: Sequence[float]) -> List[int]:
    return [min(n, max(1, math.ceil(n * b))) for b in budgets]


def faithfulness(m: np.ndarray, full: np.ndarray, empty: np.ndarray) -> float:
    den = float(np.mean(full - empty))
    return float(np.mean(m - empty) / den) if den else float('nan')


def curve_area(budgets: Sequence[float], values: Sequence[float]) -> float:
    x = np.log10(np.asarray(budgets, dtype=float))
    y = np.asarray(values, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))


def auc_f(curve: Dict[float, np.ndarray], full, empty) -> float:
    b = sorted(curve)
    return curve_area(b, [faithfulness(curve[k], full, empty) for k in b])


def cmd(curve: Dict[float, np.ndarray], full, empty) -> float:
    b = sorted(curve)
    return curve_area(b, [abs(1 - faithfulness(curve[k], full, empty)) for k in b])


def cluster_boot_indices(clusters: Sequence[str], n_boot: int, seed: int) -> List[np.ndarray]:
    """按填充词整体重采样：同一词派生的样本不算独立。"""
    groups: Dict[str, List[int]] = {}
    for i, c in enumerate(clusters):
        groups.setdefault(str(c), []).append(i)
    keys = sorted(groups)
    arrays = [np.asarray(groups[k]) for k in keys]
    rng = np.random.default_rng(seed)
    return [np.concatenate([arrays[j] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(n_boot)]


def paired_auc_diff(curve_a, curves_b: List[Dict[float, np.ndarray]], full, empty, clusters, *, n_boot=None, seed=None) -> Dict[str, float]:
    """方法 a 减去 b（多次随机时取平均）的 auc 差及聚类自助区间。"""
    n_boot = n_boot or CRITERIA['n_boot']
    seed = CRITERIA['boot_seed'] if seed is None else seed

    def stat(idx):
        sub = lambda c: {k: v[idx] for k, v in c.items()}
        return auc_f(sub(curve_a), full[idx], empty[idx]) - float(np.mean([auc_f(sub(c), full[idx], empty[idx]) for c in curves_b]))

    point = stat(np.arange(len(full)))
    boots = np.array([stat(i) for i in cluster_boot_indices(clusters, n_boot, seed)])
    boots = boots[np.isfinite(boots)]
    a = (1 - CRITERIA['ci']) / 2
    return {'diff': point, 'lo': float(np.quantile(boots, a)), 'hi': float(np.quantile(boots, 1 - a)), 'n_clusters': len(set(map(str, clusters)))}


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def behavior_gate(ld_clean: np.ndarray, ld_corrupt: np.ndarray) -> Dict[str, Any]:
    acc = float(np.mean(ld_clean > 0))
    gap = float(np.mean(ld_clean) - np.mean(ld_corrupt))
    return {'clean_acc': acc, 'gap': gap, 'clean_ld': float(np.mean(ld_clean)), 'corrupt_ld': float(np.mean(ld_corrupt)),
            'pass': acc >= CRITERIA['min_clean_acc'] and gap >= CRITERIA['min_gap']}


def identity_gate(full_keep: Optional[np.ndarray], empty_keep: Optional[np.ndarray], full, empty) -> Dict[str, Any]:
    """全部保留必须回到干净输出，全部替换必须回到空集输出。缺记录时为待检。"""
    if full_keep is None or empty_keep is None:
        return {'status': 'pending', 'note': '缺全保留或全替换记录'}
    hi, lo = faithfulness(full_keep, full, empty), faithfulness(empty_keep, full, empty)
    ok = abs(hi - 1) <= CRITERIA['identity_tol'] and abs(lo) <= CRITERIA['identity_tol']
    return {'status': 'pass' if ok else 'fail', 'f_full': hi, 'f_empty': lo}


def parse_key(key: str) -> Dict[str, Any]:
    track, direction, ablation, method, budget, src, draw = key.split('|')
    return {'track': track, 'dir': direction, 'abl': ablation, 'method': method, 'budget': float(budget), 'src': src, 'draw': int(draw)}


def make_key(track, direction, ablation, method, budget, src='self', draw=0) -> str:
    return f'{track}|{direction}|{ablation}|{method}|{budget:g}|{src}|{draw}'


def collect(raw: Dict[str, np.ndarray], track='N', direction='keep', ablation='cf', method='atp', src='self', draw=0) -> Dict[float, np.ndarray]:
    out = {}
    for key, value in raw.items():
        if key.count('|') != 6:
            continue
        p = parse_key(key)
        if (p['track'], p['dir'], p['abl'], p['method'], p['src'], p['draw']) == (track, direction, ablation, method, src, draw):
            out[p['budget']] = value
    return out


def draws(raw, method, **kw) -> List[Dict[float, np.ndarray]]:
    out = []
    for d in range(max(CRITERIA['n_random'], CRITERIA['n_matched'])):
        c = collect(raw, method=method, draw=d, **kw)
        if c:
            out.append(c)
    return out


def score_task(raw: Dict[str, np.ndarray], clusters: Sequence[str], *, methods: Sequence[str] = METHODS, track='N',
               budgets: Sequence[float] = BUDGETS, n_boot=None) -> Dict[str, Any]:
    """一个模型 × 任务 × 划分的评分。raw 须含 'clean'、'empty|<消融>' 与各条件的逐样本 m。"""
    full = raw['clean']
    out: Dict[str, Any] = {'n': int(len(full)), 'n_clusters': len(set(map(str, clusters))), 'methods': {}}
    for abl in ABLATIONS:
        empty = raw.get(f'empty|{track}|{abl}')
        if empty is None:
            continue
        out.setdefault('empty_ld', {})[abl] = float(np.mean(empty))
    primary = raw.get(f'empty|{track}|{PRIMARY_ABLATION}')
    if primary is None:
        return {**out, 'status': 'pending', 'note': '缺主消融的空集记录'}
    ref = collect(raw, track=track, method=methods[0])
    out['identity'] = identity_gate(ref.get(1.0), raw.get(f'empty_keep|{track}|{PRIMARY_ABLATION}'), full, primary)
    rand = draws(raw, 'random', track=track)
    match = draws(raw, 'matched', track=track)
    expected = sorted(budgets)
    for m in methods:
        curve = collect(raw, track=track, method=m)
        if sorted(curve) != expected:
            out['methods'][m] = {'status': 'pending', 'note': f'缺预算：有 {len(curve)} / {len(expected)}'}
            continue
        row: Dict[str, Any] = {'status': 'done', 'auc_f': auc_f(curve, full, primary), 'cmd': cmd(curve, full, primary),
                               'curve': {f'{b:g}': faithfulness(curve[b], full, primary) for b in expected}}
        need = [b for b in expected if faithfulness(curve[b], full, primary) >= 0.8]
        row['budget_f80'] = need[0] if need else None
        if rand:
            row['vs_random'] = paired_auc_diff(curve, rand, full, primary, clusters, n_boot=n_boot)
        if match:
            row['vs_matched'] = paired_auc_diff(curve, match, full, primary, clusters, n_boot=n_boot)
        if m != 'act_diff' and collect(raw, track=track, method='act_diff'):
            row['vs_act_diff'] = paired_auc_diff(curve, [collect(raw, track=track, method='act_diff')], full, primary, clusters, n_boot=n_boot)
        removal = collect(raw, track=track, direction='remove', method=m)
        if removal:
            row['drop'] = {f'{b:g}': 1 - faithfulness(removal[b], full, primary) for b in sorted(removal)}
        for abl in ABLATIONS:
            e = raw.get(f'empty|{track}|{abl}')
            c = collect(raw, track=track, ablation=abl, method=m)
            if e is not None and sorted(c) == expected:
                row.setdefault('auc_by_ablation', {})[abl] = auc_f(c, full, e)
        out['methods'][m] = row
    if rand:
        out['random_auc'] = float(np.mean([auc_f(c, full, primary) for c in rand]))
    if match:
        out['matched_auc'] = float(np.mean([auc_f(c, full, primary) for c in match]))
    done = [m for m in methods if out['methods'][m].get('auc_by_ablation')]
    pairs = {}
    for a, b in (('cf', 'mean'), ('cf', 'zero'), ('mean', 'zero')):
        names = [m for m in done if a in out['methods'][m]['auc_by_ablation'] and b in out['methods'][m]['auc_by_ablation']]
        if len(names) >= 3:
            pairs[f'{a}~{b}'] = spearman([out['methods'][m]['auc_by_ablation'][a] for m in names],
                                         [out['methods'][m]['auc_by_ablation'][b] for m in names])
    out['ablation_rank'] = pairs
    return out


def specificity(target_drop: Optional[float], off_drops: Sequence[Optional[float]]) -> Dict[str, Any]:
    """off_drops 含其他任务上的 drop。任一缺失为待检。"""
    if target_drop is None or not off_drops or any(v is None for v in off_drops):
        return {'status': 'pending'}
    worst = max(0.0, max(off_drops))
    if target_drop < CRITERIA['specific_target_drop_min']:
        return {'status': 'no_effect', 'target_drop': target_drop, 'worst_off': worst}
    ratio = target_drop / worst if worst > 0 else float('inf')
    return {'status': 'specific' if ratio >= CRITERIA['specific_ratio_min'] else 'not_specific',
            'target_drop': target_drop, 'worst_off': worst, 'ratio': ratio}


def judge(score: Dict[str, Any], gate: Dict[str, Any]) -> Dict[str, Any]:
    """逐方法给出固定判定。行为门、仪器门未过时不判方法。"""
    if not gate.get('pass'):
        return {'status': 'behavior_failed', 'methods': {}}
    if (score.get('identity') or {}).get('status') != 'pass':
        return {'status': 'pending' if (score.get('identity') or {}).get('status') in (None, 'pending') else 'instrument_failed', 'methods': {}}
    out = {}
    for name, row in score['methods'].items():
        if row.get('status') != 'done':
            out[name] = {'verdict': '待检'}
            continue
        pos = lambda k: None if k not in row else row[k]['lo'] > 0
        out[name] = {'beats_random': pos('vs_random'), 'beats_matched': pos('vs_matched'), 'beats_act_diff': pos('vs_act_diff'),
                     'verdict': '待检' if pos('vs_random') is None or pos('vs_matched') is None else
                     ('支持' if pos('vs_random') and pos('vs_matched') else '未支持')}
    ranks = [v for v in (score.get('ablation_rank') or {}).values() if v is not None]
    robust = None if not ranks else min(ranks) >= CRITERIA['ablation_spearman_min']
    return {'status': 'judged', 'methods': out, 'ablation_robust': robust}


def dict_compare(raw: Dict[str, np.ndarray], ks: Sequence[int]) -> Dict[str, Any]:
    """同一层只替换这一层 MLP：保留前 k 个特征或神经元的差异，其余用反事实。特征一方的误差项始终是反事实。"""
    full, empty = raw['clean'], raw['empty']
    gap = float(np.mean(full - raw['corrupt']))
    sensitivity = float(np.mean(full - empty)) / gap if gap else 0.0
    out: Dict[str, Any] = {'sensitivity': sensitivity}
    if abs(sensitivity) < CRITERIA['dict_sensitivity_min']:
        return {**out, 'status': 'no_power', 'note': '整层替换几乎不改变这个行为，这一层无法比较'}
    f = lambda kind, k: faithfulness(raw[f'{kind}|{k}'], full, empty)
    rows = [{'k': k, 'feature': f('feature', k), 'neuron': f('neuron', k)} for k in ks]
    return {**out, 'status': 'compared', 'rows': rows, 'feature_all': f('feature', 'all'), 'neuron_all': f('neuron', 'all'),
            'feature_wins': sum(r['feature'] > r['neuron'] for r in rows), 'of': len(rows)}


def known_head_recall(scores: np.ndarray, known: Sequence[str] = KNOWN_IOI_HEADS) -> Dict[str, Any]:
    """scores: [L, H]。前 |known| 个头里已知电路占多少；描述，不是判据。"""
    L, H = scores.shape
    order = np.argsort(-scores.reshape(-1))[:len(known)]
    top = {f'{i // H}.{i % H}' for i in order}
    return {'hit': len(top & set(known)), 'of': len(known), 'missed': sorted(set(known) - top)}
