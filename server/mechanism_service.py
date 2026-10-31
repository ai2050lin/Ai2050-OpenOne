# -*- coding: utf-8 -*-
"""机制案例与方法校准报告的只读端点（mechanism_reports.v1）。

数据真源（研究脚本直接写出，本模块不计算、不改写任何数字）：
  - 机制案例：research/mechanism/<case>/report.json      （如 conflict_case.v1：上下文与记忆冲突）
  - 方法校准：research/calibration/<case>/report.json    （如 calib_ioi.v1、calib_qk.v1：GPT-2 上答案已知的校准）

端点：
  GET /api/mechanism/index        → {cases: [{id, kind, schema, set, model, passed, checks, ...}]}（摘要，不含大数组）
  GET /api/mechanism/case/{id}    → {id, kind, schema, report}（报告原样返回）

客户端研究工作台的「例句 / 模式与优先级 / 进展」只读这两个端点（经数据包，见 deploy/datapack.py）。
「通过」只描述报告里预注册判据的结果，不是证据等级。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from server.neuron_census_service import research_root

router = APIRouter(prefix='/api/mechanism', tags=['mechanism'])

KINDS = (('mechanism', 'mechanism'), ('calibration', 'calibration'))   # (research 子目录, kind)
_ID = re.compile(r'^[A-Za-z0-9._-]{1,80}$')


def report_files() -> List[tuple]:
    """[(kind, id, path)]：每个案例目录下的 report.json，按 kind、id 排序。"""
    out = []
    for sub, kind in KINDS:
        d = research_root() / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob('*/report.json')):
            if _ID.match(p.parent.name):
                out.append((kind, p.parent.name, p))
    return out


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


V2_GROUPS = ('C_context_writers', 'M_memory_writers', 'U_upstream')


def _v1_void(doc: Dict[str, Any]) -> Optional[str]:
    """conflict_case.v1 的必要性 / 充分性按留出平均 d 归一化；平均 d < 0 时符号失效（脚本 v2 段的注释），
    这时两条判据既不算通过也不算未通过，改看 v2。"""
    d = (doc.get('heads') or {}).get('heldout_mean_d')
    if isinstance(d, (int, float)) and d < 0:
        return f'留出平均 d = {d:.3f} < 0：v1 按平均 d 归一化，符号失效（见 v2）'
    return None


def _checks(doc: Dict[str, Any]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """报告里的布尔判据，统一成 ({名称: 是否通过}, {名称: 失效原因})。
    conflict_case 的判据分散在 heads / negated_control / v2 里。"""
    checks = {k: bool(v) for k, v in (doc.get('checks') or {}).items() if isinstance(v, bool)}
    void: Dict[str, str] = {}
    heads = doc.get('heads') or {}
    nec = heads.get('necessity') or {}
    suf = heads.get('sufficiency_last_position') or {}
    why = _v1_void(doc)
    for name, val in (('head_necessity', nec.get('necessary')), ('head_sufficiency_last_position', suf.get('sufficient'))):
        if isinstance(val, bool):
            if why:
                void[name] = why
            else:
                checks[name] = val
    neg = doc.get('negated_control') or {}
    if isinstance(neg.get('copying_candidate'), bool):
        checks['negated_control_not_copying'] = not neg['copying_candidate']   # 否定句里上下文仍胜 = 可能只是复制词
    v2 = doc.get('v2') or {}
    for cue, row in sorted((v2.get('by_cue') or {}).items()):
        for g in V2_GROUPS:
            eff = (row or {}).get(g, {}).get('effect') if isinstance((row or {}).get(g), dict) else None
            if isinstance(eff, bool):
                checks[f'v2_{cue}_{g}'] = eff
    hier = v2.get('hierarchy') or {}
    if isinstance(hier.get('holds'), bool):
        checks['v2_hierarchy_U_to_C'] = hier['holds']
    return checks, void


def summarize(kind: str, case_id: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    crit = doc.get('criteria') or {}
    checks, void = _checks(doc)
    heads = doc.get('heads') or {}
    suf = heads.get('sufficiency_last_position') or {}
    nec = heads.get('necessity') or {}
    passed = doc.get('passed')
    if not isinstance(passed, bool):
        passed = all(checks.values()) if checks else None
    return {
        'id': case_id, 'kind': kind, 'schema': crit.get('schema') or doc.get('schema'),
        'set': doc.get('set'), 'model': doc.get('model'),
        'model_snapshot': doc.get('model_snapshot') or doc.get('model_sha256'),
        'passed': passed, 'checks': checks, 'void_checks': void,
        'fidelity_void': _v1_void(doc),                # 头电路保真度（v1 保留比例）失效的原因；None = 有效
        'n_gated': doc.get('n_gated'), 'n_facts': doc.get('n_facts'),
        'n_heldout': doc.get('n_heldout'),
        'cues': sorted((doc.get('priority_table') or {}).keys()),
        'head_circuit_keep_fraction': suf.get('keep_fraction'),
        'head_necessity_drop': nec.get('drop_top'),
        'seconds': doc.get('seconds'),
        # 报告里有哪些部分（只描述字段齐全，不表示检验通过）；冲突案例五项齐全 = 一个完整案例
        'has': {
            'priority_table': bool(doc.get('priority_table')),
            'head_dla': isinstance(heads.get('dla_discovery'), list),
            'head_total_effect': isinstance(heads.get('total_effect_discovery'), list),
            'neurons': bool((doc.get('neurons') or {}).get('top')),
            'head_circuit_fidelity': isinstance(suf.get('keep_fraction'), (int, float)),
        },
    }


def index() -> Dict[str, Any]:
    cases = []
    for kind, case_id, p in report_files():
        doc = _load(p)
        if doc is None:
            cases.append({'id': case_id, 'kind': kind, 'error': '报告无法解析'})
            continue
        cases.append(summarize(kind, case_id, doc))
    return {'schema': 'mechanism_reports.v1', 'cases': cases}


def case(case_id: str) -> Dict[str, Any]:
    if not _ID.match(case_id or ''):
        raise HTTPException(status_code=400, detail='案例 id 不合法')
    for kind, cid, p in report_files():
        if cid == case_id:
            doc = _load(p)
            if doc is None:
                raise HTTPException(status_code=422, detail='报告无法解析')
            crit = doc.get('criteria') or {}
            return {'id': cid, 'kind': kind, 'schema': crit.get('schema') or doc.get('schema'), 'report': doc}
    raise HTTPException(status_code=404, detail='没有这个案例')


@router.get('/index')
def index_ep() -> Dict[str, Any]:
    return index()


@router.get('/case/{case_id}')
def case_ep(case_id: str) -> Dict[str, Any]:
    return case(case_id)
