# -*- coding: utf-8 -*-
"""金丝雀一致性（canary.v1）。

每个 v3 模板包都附带同一组固定输入（CANARY_ITEMS，不参与统计）。runner 对每条金丝雀在每个
请求层位记录目标位置向量的摘要：L2 范数、均值、前 16 维。中心节点按
(model_id, model_rev, precision) 维护参考摘要：同一快照、同一精度下各节点的摘要应在容差内一致；
超出容差记为 drift，该结果不进入合并统计（仍保留，供排查）。

容差是工程经验值，不是统计检验：bf16 存在 kernel 选择差异，fp32 应几乎逐位一致。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 'canary.v1'
HEAD_DIMS = 16

CANARY_ITEMS: List[Dict[str, Any]] = [
    {'id': 'canary-zh-1', 'canary': True, 'cond': {}, 'split': 'canary',
     'text': '今天的天气很好，我们去公园散步。', 'target': '公园'},
    {'id': 'canary-zh-2', 'canary': True, 'cond': {}, 'split': 'canary',
     'text': '数学老师在黑板上写下了一道方程。', 'target': '方程'},
    {'id': 'canary-en-1', 'canary': True, 'cond': {}, 'split': 'canary',
     'text': 'The quick brown fox jumps over the lazy dog.', 'target': 'dog'},
    {'id': 'canary-en-2', 'canary': True, 'cond': {}, 'split': 'canary',
     'text': 'Water boils at one hundred degrees Celsius.', 'target': 'Celsius'},
]

TOLERANCE = {'fp32': 1e-4, 'float32': 1e-4, 'tf32': 2e-3, 'fp16': 1e-2,
             'float16': 1e-2, 'bf16': 3e-2, 'bfloat16': 3e-2}
DEFAULT_TOLERANCE = 3e-2


def model_key(model_id: str, model_rev: str, precision: str) -> str:
    return f'{model_id or "unknown"}|{model_rev or "-"}|{(precision or "-").lower()}'


def tolerance_for(precision: str) -> float:
    return TOLERANCE.get((precision or '').lower(), DEFAULT_TOLERANCE)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-6)


def _vec_rel(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return math.inf
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = max(math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b)), 1e-6)
    return num / den


def compare(reference: Dict[str, Any], candidate: Dict[str, Any], precision: str) -> Dict[str, Any]:
    """比较两份 canary.json 的 entries。返回 {status, max_rel, tolerance, detail}。

    status: ok | drift | incomplete（条目或层位不全，无法比较）。
    """
    tol = tolerance_for(precision)
    ref = (reference or {}).get('entries') or {}
    cand = (candidate or {}).get('entries') or {}
    if not ref or not cand:
        return {'status': 'incomplete', 'max_rel': None, 'tolerance': tol, 'detail': '缺少 canary 条目'}
    worst = 0.0
    worst_at = ''
    missing: List[str] = []
    for cid, layers in ref.items():
        if cid not in cand:
            missing.append(cid); continue
        for tag, stats in layers.items():
            other = cand[cid].get(tag)
            if not other:
                missing.append(f'{cid}@{tag}'); continue
            r = max(_rel(stats['norm'], other['norm']), _vec_rel(stats['head'], other['head']))
            if r > worst:
                worst, worst_at = r, f'{cid}@{tag}'
    if missing:
        return {'status': 'incomplete', 'max_rel': worst, 'tolerance': tol,
                'detail': '缺少：' + ', '.join(missing[:8])}
    return {'status': 'ok' if worst <= tol else 'drift', 'max_rel': worst, 'tolerance': tol,
            'detail': f'最大相对偏差 {worst:.3g} @ {worst_at}（容差 {tol:g}）'}


def summarize_vector(vec) -> Dict[str, Any]:
    """runner 与测试共用的摘要函数（vec 为一维序列）。"""
    vals = [float(x) for x in vec]
    n = len(vals)
    norm = math.sqrt(sum(v * v for v in vals))
    return {'norm': norm, 'mean': sum(vals) / max(n, 1), 'head': vals[:HEAD_DIMS], 'dim': n}


def evaluate(reference: Optional[Dict[str, Any]], candidate: Optional[Dict[str, Any]],
             precision: str) -> Tuple[str, Dict[str, Any]]:
    """中心节点入口：无参考时该结果成为参考（status=reference）。"""
    if not candidate or not candidate.get('entries'):
        return 'missing', {'status': 'missing', 'detail': '结果未附带 canary.json'}
    if not reference:
        return 'reference', {'status': 'reference', 'detail': '首个该快照/精度的金丝雀，登记为参考'}
    res = compare(reference, candidate, precision)
    return res['status'], res
