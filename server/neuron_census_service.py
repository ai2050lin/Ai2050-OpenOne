# -*- coding: utf-8 -*-
"""原生 MLP 神经元的只读端点（C0）：普查索引、模板选择性汇总、单元卡（unit_card.v1）、检索。

数据真源：
  - 自然语料普查（mlp_census.v1）：<AI2050_NATIVE_DIR>/_census/<model_key>/<run>/{manifest.json, L<b>.npz}；
    语料 token 流在 <AI2050_NATIVE_DIR>/_census/corpus/<fp>/；
  - 模板选择性（neurons.v1）：research/neurons/GR-Q-*.json，单元的 d 值按需从节点本地 MLP 中间单元重算；
  - 预注册消融（INT-N-v1）：research/interventions/INT-N-v1.units.json 与 INT-N-v1-<model>.json。
本模块不导入 torch：上下文用 tokenizers 库从 token 流还原；前端不生成任何文本或标签。
"""
from __future__ import annotations

import glob
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
router = APIRouter(prefix='/api/neurons', tags=['neurons'])
TOKENS_PER_PAGE = 400
CTX_LEFT, CTX_RIGHT = 28, 8
N_SIDE, LO, HI = 32, 1e-3, 1e2
REGIMES = ('++', '--', '+-', '-+')
SRC_SHIFT = 40


def native_root() -> Path:
    return Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'))


def research_root() -> Path:
    return Path(os.environ.get('AI2050_RESEARCH_DIR', str(ROOT / 'research')))


# ---------------- 读取与缓存 ----------------

def census_runs() -> List[Dict[str, Any]]:
    out = []
    for m in sorted(glob.glob(str(native_root() / '_census' / '*' / '*' / 'manifest.json'))):
        p = Path(m)
        if p.parent.parent.name == 'corpus':
            continue
        try:
            man = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if man.get('schema') != 'mlp_census.v1':
            continue
        out.append({'model_key': man['model_key'], 'run': p.parent.name, 'corpus_fp': man['corpus_fp'],
                    'n_tokens': man.get('n_tokens_counted'), 'n_layers': man.get('n_layers'), 'd_ff': man.get('d_ff'),
                    'k': man.get('k'), 'skip_head': man.get('skip_head'), 'seq_len': man.get('seq_len'),
                    'sources': man.get('sources'), 'tok_per_s': man.get('tok_per_s')})
    return out


def pick_census(model_key: str, run: Optional[str]) -> Dict[str, Any]:
    runs = [r for r in census_runs() if r['model_key'] == model_key and (run is None or r['run'] == run)]
    if not runs:
        raise HTTPException(404, f'没有 {model_key} 的普查结果')
    return max(runs, key=lambda r: r['n_tokens'] or 0)


def _run_stamp(model_key: str, run: str) -> float:
    """普查运行的版本戳（manifest 修改时间）：后台子进程重新合并后，缓存随之失效。"""
    try:
        return (native_root() / '_census' / model_key / run / 'manifest.json').stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=4)
def _census_manifest(model_key: str, run: str, stamp: float) -> Dict[str, Any]:
    return json.loads((native_root() / '_census' / model_key / run / 'manifest.json').read_text(encoding='utf-8'))


def census_manifest(model_key: str, run: str) -> Dict[str, Any]:
    return _census_manifest(model_key, run, _run_stamp(model_key, run))


census_manifest.cache_clear = _census_manifest.cache_clear


@lru_cache(maxsize=8)
def _census_layer(model_key: str, run: str, block: int, stamp: float) -> Dict[str, np.ndarray]:
    p = native_root() / '_census' / model_key / run / f'L{block}.npz'
    if not p.is_file():
        raise HTTPException(404, f'普查没有第 {block} 块')
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def census_layer(model_key: str, run: str, block: int) -> Dict[str, np.ndarray]:
    return _census_layer(model_key, run, block, _run_stamp(model_key, run))


census_layer.cache_clear = _census_layer.cache_clear


@lru_cache(maxsize=2)
def _readout_index(model_key: str, run: str, n_layers: int, stamp: float):
    """各层的直接读出 token 与最大激活（只读这两个数组），供检索使用。"""
    out = []
    for b in range(n_layers):
        with np.load(native_root() / '_census' / model_key / run / f'L{b}.npz') as z:
            out.append((z['readout_top'], z['amax']))
    return out


def readout_index(model_key: str, run: str, n_layers: int):
    return _readout_index(model_key, run, n_layers, _run_stamp(model_key, run))


readout_index.cache_clear = _readout_index.cache_clear


@lru_cache(maxsize=2)
def corpus(fp: str):
    d = native_root() / '_census' / 'corpus' / fp
    man = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
    streams = []
    for s in man['sources']:
        with np.load(d / s['file']) as z:
            streams.append(z['tokens'])
    return man, streams


@lru_cache(maxsize=2)
def tokenizer(model_path: str):
    from tokenizers import Tokenizer
    p = Path(model_path)
    if not p.is_absolute():
        p = ROOT / p
    return Tokenizer.from_file(str(p / 'tokenizer.json'))


def _decode(tok, ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=False)


def context(tok, stream: np.ndarray, off: int) -> Dict[str, str]:
    """命中 token 前后的文本；字节级 token 解码出不完整字符时，向两侧扩展命中段直到可以完整解码。"""
    hs, he = off, off + 1
    for _ in range(3):                      # \u5148\u5411\u53f3\u8865\u9f50\u88ab\u5207\u5f00\u7684\u591a\u5b57\u8282\u5b57\u7b26
        if '\ufffd' not in _decode(tok, stream[hs:he]) or he >= len(stream):
            break
        he += 1
    for _ in range(3):                      # \u4ecd\u4e0d\u5b8c\u6574\u65f6\u5411\u5de6\u8865\u9f50
        if '\ufffd' not in _decode(tok, stream[hs:he]) or hs <= 0:
            break
        hs -= 1
    a = max(0, hs - CTX_LEFT)
    while a < hs and _decode(tok, stream[a:a + 1]).startswith('\ufffd'):
        a += 1
    return {'left': _decode(tok, stream[a:hs]), 'hit': _decode(tok, stream[hs:he]), 'right': _decode(tok, stream[he:he + CTX_RIGHT])}


def _per_page(rate: float) -> str:
    if rate <= 0:
        return '本语料中未出现'
    pages = 1.0 / (rate * TOKENS_PER_PAGE)
    if pages < 1:
        return f'约每页 {1 / pages:.1f} 次'
    return f'约每 {pages:.0f} 页一次' if pages < 1e4 else f'约每 {pages:.1e} 页一次'


def _bin_lower(i: int) -> float:
    """第 i 箱（0..64）的激活下边界（正侧）或上边界绝对值（负侧）。"""
    edges = np.geomspace(LO, HI, N_SIDE + 1)
    if i == N_SIDE:
        return 0.0
    return float(edges[i - N_SIDE - 1]) if i > N_SIDE else -float(edges[N_SIDE - 1 - i])


# ---------------- 模板选择性与干预 ----------------

def selectivity_runs(model_key: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for f in sorted(glob.glob(str(research_root() / 'neurons' / '*.json'))):
        try:
            r = json.loads(Path(f).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if r.get('schema') != 'neurons.v1':
            continue
        if model_key and r.get('model_key') != model_key:
            continue
        out.append({'file': Path(f).name, 'model_key': r.get('model_key'), 'tm_id': r.get('tm_id'), 'version': r.get('version'),
                    'factor': r.get('factor'), 'hi': r.get('hi'), 'lo': r.get('lo'), 'contribution': r.get('contribution', False),
                    'native_dirs': r.get('native_dirs'), 'criteria': r.get('criteria')})
    return out


@lru_cache(maxsize=4)
def _selectivity_json(name: str) -> Dict[str, Any]:
    return json.loads((research_root() / 'neurons' / name).read_text(encoding='utf-8'))


def unit_selectivity(run: Dict[str, Any], block: int, unit: int) -> List[Dict[str, Any]]:
    """从节点本地 MLP 中间单元重算该单元在模板对照上的 d（discovery 与留出）；本地数据缺失时只回报 top 列表中的记录。"""
    res = _selectivity_json(run['file'])
    out = []
    for t in res.get('tags', []):
        if t['block'] != block:
            continue
        rec = next((u for u in t.get('units', []) + t.get('units_by_contrib', []) if u['unit'] == unit), None)
        out.append({'run': run['file'], 'tm_id': run['tm_id'], 'version': run['version'], 'factor': run['factor'],
                    'hi': run['hi'], 'lo': run['lo'], 'readout': t['readout'], 'cut': t['cut'],
                    'in_top_lists': rec is not None, **({k: rec[k] for k in ('d_disc', 'd_held', 'pos_level', 'dominant_regime',
                                                                            'contrib', 'delta_A_disc') if k in rec} if rec else {})})
    try:
        from shared.dist_protocol import neurons as N
        dirs = [d for d in (run.get('native_dirs') or []) if Path(d).exists()]
        if dirs:
            corpus_ = json.loads((ROOT / 'distributed_data/bundles' / run['tm_id'] / f"v{run['version']}" / 'corpus.json').read_text(encoding='utf-8'))
            items = {it['id']: it for it in corpus_['items'] if not it.get('canary')}
            field = N.MlpField(dirs)
            lab, split, _, _ = N.design(field.ids, items, run['factor'], run['hi'], run['lo'] if run['lo'] != '其余水平' else None)
            disc = np.array([s == 'discovery' for s in split])
            held = np.array([s in N.HELD for s in split])
            for rec in out:
                G, V = field.GV(rec['readout'], block)
                A = N.silu(G[:, unit:unit + 1]) * V[:, unit:unit + 1]
                rec['d_disc'] = float(N._std_diff(A, (lab == 1) & disc, (lab == -1) & disc)[0])
                rec['d_held'] = float(N._std_diff(A, (lab == 1) & held, (lab == -1) & held)[0])
                rec['mean_hi'], rec['mean_lo'] = float(A[lab == 1].mean()), float(A[lab == -1].mean())
                rec['selected'] = bool(abs(rec['d_disc']) >= rec['cut'] and np.sign(rec['d_held']) == np.sign(rec['d_disc'])
                                       and abs(rec['d_held']) >= float(run['criteria']['held_min']))
                rec['source'] = '节点本地 MLP 中间单元重算'
    except (OSError, ValueError, KeyError, ImportError) as exc:
        for rec in out:
            rec['recompute_error'] = f'{type(exc).__name__}: {exc}'
    return out


def unit_interventions(model_key: str, block: int, unit: int) -> List[Dict[str, Any]]:
    out = []
    upath = research_root() / 'interventions' / 'INT-N-v1.units.json'
    if not upath.is_file():
        return out
    units = json.loads(upath.read_text(encoding='utf-8'))
    for model_id, m in units.get('models', {}).items():
        if m.get('model_key') != model_key:
            continue
        rpath = research_root() / 'interventions' / f'INT-N-v1-{model_id}.json'
        res = json.loads(rpath.read_text(encoding='utf-8')) if rpath.is_file() else None
        for seg, s in m['segments'].items():
            if not any(u['block'] == block and u['unit'] == unit for u in s['units']):
                continue
            rec = {'prereg': 'INT-N-v1', 'segment': seg, 'group_size': len(s['units']), 'role': '组内单元（20 个一起替换）'}
            if res:
                v = res['evaluation']['verdicts'][seg]
                e = res['evaluation']['segments'][seg]
                rec.update({'prereg_sha256': res.get('prereg_sha256'), 'causal_use_in_scope': v['causal_use_in_scope'],
                            'necessity': e['necessity_primary']['verdict'], 'necessity_mean': e['necessity_primary']['mean'],
                            'sufficiency': e['sufficiency_primary']['verdict'], 'sufficiency_mean': e['sufficiency_primary']['mean'],
                            'self_check_note': 'down_proj 输入与原生数据的相对误差最大值超过预注册门槛 0.02（见 Phase 254 §9）'})
            else:
                rec['causal_use_in_scope'] = '未运行'
            out.append(rec)
    return out


# ---------------- 单层因果归因（L<b>-v*：fp32 一阶归因 + 真实消融 + 条件化消融） ----------------

@lru_cache(maxsize=4)
def _layer_attr(path: str, stamp: float) -> Dict[str, Any]:
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    tot = np.array([u['total'] for u in d['units']])
    rank = np.empty(len(tot), dtype=np.int64)
    rank[np.argsort(-np.abs(tot))] = np.arange(1, len(tot) + 1)
    rows = {r['unit']: r for r in d['validation']['rows']}
    cond = {}
    cp = Path(path).with_name(Path(path).name.replace('-attribution.json', '-conditional.json'))
    if cp.is_file():
        c = json.loads(cp.read_text(encoding='utf-8'))
        cond = {'verdict': c.get('verdict'), 'rows': {r['unit']: r for r in c.get('rows', [])}}
    return {'d': d, 'rank': rank, 'rows': rows, 'cond': cond}


def unit_attribution(model_key: str, block: int, unit: int) -> List[Dict[str, Any]]:
    """只读通过预注册验证、可公开排序的运行（publish_ranking = true）；同一层取版本最高者。"""
    out = []
    files = sorted((research_root() / 'neurons').glob(f'L{block}-v*-attribution.json'),
                   key=lambda q: int(q.name.split('-v')[1].split('-')[0]))
    for f in reversed(files):
        a = _layer_attr(str(f), f.stat().st_mtime)
        d = a['d']
        if d.get('model_key') != model_key or not d.get('publish_ranking') or unit >= len(d['units']):
            continue
        u = d['units'][unit]
        rec = {'prereg': d.get('prereg'), 'prereg_sha256': d.get('prereg_sha256'), 'total': u['total'], 'direct': u['direct'],
               'indirect': u['indirect'], 'magnitude': u['magnitude'], 'rank': int(a['rank'][unit]), 'n_units': len(d['units']),
               'validation_verdict': d['validation']['verdict'],
               'note': '均值消融的一阶估计（fp32，普查语料）：正值 = 把该单元换成均值会使下一 token 损失上升'}
        r = a['rows'].get(unit)
        if r:
            rec['true_ablation'] = {'delta': r['true_delta'], 'z': r.get('z'), 'set': r['set']}
        c = a['cond'].get('rows', {}).get(unit) if a['cond'] else None
        if c:
            rec['conditional'] = {k: c.get(k) for k in ('n_fire', 'local_mean', 'p_local', 'sig_local', 'dlogp_promote', 'global')}
            rec['conditional']['run_verdict'] = a['cond'].get('verdict')
        out.append(rec)
        break
    return out


@lru_cache(maxsize=4)
def _layer_trigger(path: str, stamp: float) -> Dict[str, Any]:
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    return {'d': d, 'rows': {r['unit']: r for r in d.get('rows', [])}}


def unit_trigger(model_key: str, block: int, unit: int) -> List[Dict[str, Any]]:
    """触发点消融（L<b>-v*-trigger.json，layer_trigger.v1）：单元在命中众数 token 且前 1% 激活的位置被换成均值。"""
    out = []
    for f in sorted((research_root() / 'neurons').glob(f'L{block}-v*-trigger.json'), reverse=True):
        t = _layer_trigger(str(f), f.stat().st_mtime)
        r = t['rows'].get(unit)
        if not r:
            continue
        d = t['d']
        rec = {'prereg': d.get('prereg'), 'prereg_sha256': d.get('prereg_sha256'), 'n_trigger': r.get('n_trigger'), 'n_R': r.get('n_R'),
               'q99': r.get('q99'), 'dlogp_promote': r.get('dlogp_promote'),
               'primary_verdict': d['primary']['verdict'], 'secondary_verdict': d['secondary']['verdict'],
               'note': '触发点 = 命中正极众数 token 且激活位于本单元前 1%；R = 其中下一个 token 属于读出前 10 的位置'}
        for k in ('d_R', 'p_R', 'sig_p_R', 'ctrl_d_R', 'd_T', 'p_T', 'sig_p_T', 'ctrl_d_T'):
            if k in r:
                rec[k] = r[k]
        out.append(rec)
        break
    return out


# ---------------- 解释检验（EXPL-v1 / EXPL-v2） ----------------

EXPL_RUNS = (('EXPL-v1', 'EXPL-v1-results.json'), ('EXPL-v2', 'EXPL-v2-results.json'))
GRADES = ('文本可复现', '只在语料中可识别', '未解释')


def _expl_record(run_id: str, pre: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    """一条解释检验记录 → 统一格式与等级（文本可复现 / 只在语料中可识别 / 未解释）。"""
    text_out = r.get('text_outcome') or r.get('outcome')
    retrieval = r.get('retrieval')                       # EXPL-v1 没有语料检索
    if text_out == 'win':
        grade = '文本可复现'
    elif retrieval == '通过':
        grade = '只在语料中可识别'
    else:
        grade = '未解释'
    rec = {'run': run_id, 'prereg_sha256': pre.get('_sha'), 'code': r.get('code'), 'explanation': r.get('explanation'),
           'confidence': r.get('confidence'), 'text_outcome': text_out, 'retrieval': retrieval if retrieval else '未做',
           'recall': r.get('recall'), 'fpr': r.get('fpr'), 'grade': grade,
           'cause': r.get('cause'), 'cause_source': r.get('cause_source'), 'cause_reason': r.get('cause_reason'),
           'reason': r.get('reason')}
    if r.get('strong_rate'):
        rec['strong_rate'] = {k: r['strong_rate'].get(k) for k in ('new', 'near_miss', 'distant', 'control')}
    if r.get('median'):
        rec['median_s'] = {k: r['median'].get(k) for k in ('new', 'near_miss', 'distant', 'control')}
    return rec


@lru_cache(maxsize=1)
def _expl_index(stamp: tuple) -> Dict[tuple, List[Dict[str, Any]]]:
    out: Dict[tuple, List[Dict[str, Any]]] = {}
    for run_id, fname in EXPL_RUNS:
        rp = research_root() / 'interventions' / fname
        pp = research_root() / 'interventions' / f'{run_id}.prereg.json'
        if not rp.is_file() or not pp.is_file():
            continue
        pre = json.loads(pp.read_text(encoding='utf-8'))
        sp = pp.with_suffix('.sha256')
        pre['_sha'] = sp.read_text(encoding='utf-8').split()[0] if sp.is_file() else None
        mk = (pre.get('model') or {}).get('model_key')
        for r in json.loads(rp.read_text(encoding='utf-8')).get('units', []):
            out.setdefault((mk, int(r['block']), int(r['unit'])), []).append(_expl_record(run_id, pre, r))
    return out


def expl_index() -> Dict[tuple, List[Dict[str, Any]]]:
    stamp = tuple((research_root() / 'interventions' / f).stat().st_mtime if (research_root() / 'interventions' / f).is_file() else 0
                  for _, f in EXPL_RUNS)
    return _expl_index(stamp)


def explain_axis(records: List[Dict[str, Any]]) -> str:
    """以最新一次检验为准（EXPL-v2 晚于 v1）；未解释时附原因。"""
    if not records:
        return '未检验'
    r = records[-1]
    if r['grade'] != '未解释':
        return f"{r['grade']}（{r['run']}）"
    cause = r.get('cause') or ('无法解释' if r.get('reason') == '无法解释' else '未分类')
    return f"未解释：{cause}（{r['run']}）"


# ---------------- 端点 ----------------

@router.get('/index')
def index():
    return {'schema': 'neurons_index.v1', 'census': census_runs(), 'selectivity': selectivity_runs(),
            'note': '普查是自然语料上的描述统计；选择性是模板对照上的相关观测；两者都不说明单元被模型使用'}


@router.get('/selectivity')
def selectivity(run: str = Query(...), top: int = Query(10, ge=1, le=20)):
    if not re.fullmatch(r'[\w.\-]+\.json', run):
        raise HTTPException(400, '非法文件名')
    try:
        res = _selectivity_json(run)
    except OSError:
        raise HTTPException(404, '没有该选择性运行')
    keep = ('unit', 'd_disc', 'd_held', 'pos_level', 'dominant_regime', 'contrib', 'share')
    tags = [{**{k: t[k] for k in ('readout', 'block', 'n_selected', 'n_units', 'cut', 'null_max_d') if k in t},
             **({'share_selected': t['share_selected']} if 'share_selected' in t else {}),
             'top_by_d': [{k: u[k] for k in keep if k in u} for u in t.get('units', [])[:top]],
             'top_by_contrib': [{k: u[k] for k in keep if k in u} for u in t.get('units_by_contrib', [])[:top]]}
            for t in res['tags']]
    return {'run': run, 'model_key': res.get('model_key'), 'factor': res['factor'], 'hi': res['hi'], 'lo': res['lo'],
            'criteria': res['criteria'], 'contribution': res.get('contribution', False), 'tags': tags,
            'ranking_note': 'top_by_d 按标准化差值 |d| 排序（偏向低方差单元）；top_by_contrib 按残差贡献 |ΔĀ|·‖W_down 列‖ 排序'}


@router.get('/unit')
def unit_card(model_key: str = Query(...), block: int = Query(..., ge=0), unit: int = Query(..., ge=0),
              run: Optional[str] = Query(None), n_poles: int = Query(12, ge=1, le=32)):
    c = pick_census(model_key, run)
    man = census_manifest(model_key, c['run'])
    if block >= man['n_layers'] or unit >= man['d_ff']:
        raise HTTPException(404, '单元超出范围')
    L = census_layer(model_key, c['run'], block)
    cman, streams = corpus(man['corpus_fp'])
    tok = tokenizer(man['model_path'])
    n = float(L['n'])
    hist = L['hist'][unit].astype(np.int64)                                     # (65, 4)
    mean = float(L['s1'][unit] / n)
    std = float(np.sqrt(max(L['s2'][unit] / n - mean ** 2, 0.0)))
    amax, amin = float(L['amax'][unit]), float(L['amin'][unit])

    def rate_above(x: float) -> float:
        tot = hist.sum(1)
        return float(sum(tot[i] for i in range(N_SIDE + 1, 2 * N_SIDE + 1) if _bin_lower(i) >= x) / n)

    def poles(name: str):
        res = []
        for i in range(min(n_poles, L[f'{name}_idx'].shape[1])):
            g = int(L[f'{name}_idx'][unit, i])
            if g < 0:
                break
            src, off = g >> SRC_SHIFT, g & ((1 << SRC_SHIFT) - 1)
            G, V = float(L[f'{name}_g'][unit, i]), float(L[f'{name}_v'][unit, i])
            res.append({'a': float(L[f'{name}_val'][unit, i]), 'g': G, 'v': V,
                        'regime': ('++' if V > 0 else '+-') if G > 0 else ('--' if V < 0 else '-+'),
                        'source': cman['sources'][src]['source'], 'gidx': g, **context(tok, streams[src], off)})
        return res
    tot_by_regime = hist.sum(0)
    card = {
        'schema': 'unit_card.v1', 'id': f'mlp:{model_key}:B{block}:U{unit}', 'kind': 'mlp_neuron',
        'unit_label': 'MLP 中间单元（SwiGLU，A = silu(G)⊙V；不是残差流坐标）',
        'model_key': model_key, 'model_snapshot': man.get('model_snapshot', {}).get('sha256'), 'layer': block, 'unit': unit,
        'census': {'run': c['run'], 'corpus_fp': man['corpus_fp'], 'n_tokens': int(n),
                   'sources': [s['source'] for s in cman['sources']],
                   'hist': {'n_side': N_SIDE, 'lo': LO, 'hi': HI, 'bin_lower': [_bin_lower(i) for i in range(2 * N_SIDE + 1)],
                            'by_regime': {r: hist[:, k].tolist() for k, r in enumerate(REGIMES)}},
                   'regime_share': {r: float(tot_by_regime[k] / n) for k, r in enumerate(REGIMES)},
                   'mean': mean, 'std': std, 'max': amax, 'min': amin,
                   'fire_rate': {lvl: {'threshold': lvl_v * amax, 'rate': rate_above(lvl_v * amax), 'per_page': _per_page(rate_above(lvl_v * amax))}
                                 for lvl, lvl_v in (('gt_10pct_max', 0.1), ('gt_50pct_max', 0.5))} if amax > 0 else {},
                   'poles': {'pos': poles('top'), 'neg': poles('bot')},
                   'readout': {'promotes': [_decode(tok, [i]) for i in L['readout_top'][unit]],
                               'suppresses': [_decode(tok, [i]) for i in L['readout_bot'][unit]],
                               'note': 'W_U(γ⊙W_down[:, j]) 的前 / 后 10 个 token：直接通路近似，不含后续层'},
                   'excluded': f"每个 {man['seq_len']} token 窗口的前 {man['skip_head']} 个位置与文档分隔符不计入"},
        'template_selectivity': [rec for r in selectivity_runs(model_key) for rec in unit_selectivity(r, block, unit)],
        'interventions': unit_interventions(model_key, block, unit),
        'attribution': unit_attribution(model_key, block, unit),
        'trigger': unit_trigger(model_key, block, unit),
        'caveats': ['极值片段天然有选择偏差：它们是 top / bottom-k，不代表典型激活',
                    '语料以教育筛选网页与百科为主（中英），不含对话、代码与广告页',
                    '读出 token 只是直接通路近似'],
    }
    sel = card['template_selectivity']
    read = '留出复现' if any(r.get('selected') for r in sel) else ('有模板记录，未达入选判据' if sel else '未检验')
    iv = card['interventions']
    control = ('范围内干预（组级，20 单元一起替换）' if any(r.get('causal_use_in_scope') == '是' for r in iv)
               else ('组级干预未通过' if iv else '未检验'))
    att = card['attribution']
    trg = card['trigger']
    if control == '未检验' and trg and trg[0].get('sig_p_R') and trg[0]['primary_verdict'] == '支持':
        control = f"触发点均值消融（{trg[0]['prereg']}：读出为答案的位置损失 +{trg[0]['d_R']:.3f} nats）"
    elif control == '未检验' and att:
        ta = att[0].get('true_ablation')
        control = (f"单元级均值消融（{att[0]['prereg']}，|z|>2）" if ta and abs(ta.get('z') or 0) > 2
                   else f"一阶归因排序 #{att[0]['rank']}（{att[0]['prereg']}）")
    card['explanations'] = expl_index().get((model_key, block, unit), [])
    card['evidence_axes'] = {'read': read, 'control': control, 'explain': explain_axis(card['explanations'])}
    return card


@lru_cache(maxsize=4)
def _space(path: str, stamp: float) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def space_files() -> List[Path]:
    return sorted((research_root() / 'neurons').glob('L*-space.json'), key=lambda q: int(q.name[1:].split('-')[0]))


@router.get('/space/index')
def space_index():
    """有 3D 视图数据（layer_space.v1）的层。"""
    out = []
    for f in space_files():
        d = _space(str(f), f.stat().st_mtime)
        out.append({'model_key': d['model_key'], 'layer': d['layer'], 'n_units': d['n_units'], 'sources': d['sources']})
    return {'schema': 'layer_space_index.v1', 'layers': out}


@router.get('/space')
def space(model_key: str = Query(...), layer: int = Query(..., ge=0)):
    """单层神经元的属性空间与读出语义空间坐标（scripts/build_layer_space.py）。"""
    f = research_root() / 'neurons' / f'L{layer}-space.json'
    if not f.is_file():
        raise HTTPException(404, f'第 {layer} 层没有 3D 视图数据')
    d = _space(str(f), f.stat().st_mtime)
    if d['model_key'] != model_key:
        raise HTTPException(404, '该模型没有这一层的 3D 视图数据')
    return d


@router.get('/search')
def search(model_key: str = Query(...), q: str = Query(..., min_length=1, max_length=64), run: Optional[str] = Query(None),
           limit: int = Query(30, ge=1, le=100)):
    """「L23.U8056」「B23 8056」按单元定位；其他文本按直接读出（推高的 token）检索。"""
    c = pick_census(model_key, run)
    man = census_manifest(model_key, c['run'])
    m = re.fullmatch(r'\s*[LB]?(\d+)\s*[.:,\s]\s*U?(\d+)\s*', q, flags=re.I)
    if m:
        b, u = int(m.group(1)), int(m.group(2))
        if b < man['n_layers'] and u < man['d_ff']:
            return {'mode': 'unit', 'results': [{'block': b, 'unit': u}]}
        return {'mode': 'unit', 'results': []}
    tok = tokenizer(man['model_path'])
    ids = set(tok.encode(q, add_special_tokens=False).ids) | set(tok.encode(' ' + q, add_special_tokens=False).ids)
    res = []
    for b, (rt, amax) in enumerate(readout_index(model_key, c['run'], man['n_layers'])):
        hit = np.where(np.isin(rt[:, :3], list(ids)).any(1))[0]
        for u in hit:
            res.append({'block': b, 'unit': int(u), 'promotes': [_decode(tok, [i]) for i in rt[u][:5]], 'max': float(amax[u])})
    res.sort(key=lambda r: -r['max'])
    return {'mode': 'readout', 'query_token_ids': sorted(ids), 'results': res[:limit],
            'note': '按单元直接读出（W_U(γ⊙W_down)）的前 3 个推高 token 检索；结果按普查最大激活排序'}


@router.get('/explained')
def explained(model_key: str = Query(...), grade: Optional[str] = Query(None)):
    """做过解释检验的单元列表（EXPL-v1 / v2），按层排序；grade 可选 文本可复现 / 只在语料中可识别 / 未解释。"""
    if grade is not None and grade not in GRADES:
        raise HTTPException(400, f'grade 只能是 {GRADES}')
    rows = []
    for (mk, b, u), recs in sorted(expl_index().items(), key=lambda kv: (kv[0][1], kv[0][2])):
        if mk != model_key:
            continue
        r = recs[-1]
        if grade and r['grade'] != grade:
            continue
        rows.append({'block': b, 'unit': u, 'grade': r['grade'], 'run': r['run'], 'explanation': r['explanation'],
                     'confidence': r['confidence'], 'cause': r.get('cause'), 'axis': explain_axis(recs)})
    counts = {g: sum(1 for (mk, _, _), recs in expl_index().items() if mk == model_key and recs[-1]['grade'] == g) for g in GRADES}
    return {'model_key': model_key, 'counts': counts, 'results': rows,
            'note': '等级以最新一次检验为准：文本可复现 = 文本测试 win；只在语料中可识别 = 文本测试未 win 但语料检索验证通过（仅 EXPL-v2）；'
                    '未解释附失败原因（工具 / 解释 / 训练 / 递归泛化 / 叠加，或无法解释）。解释由子代理生成，证据等级见对应预注册。'}


# ---------------- 逐层神经元摘要（neuron_layers.v1）：v8「神经元」主线的层阵列与层塔 ----------------
# 摘要由后台子进程（deploy/snapshot_worker.py）按普查运行生成，写在 <普查运行>/neuron_layers/<代码哈希>/L<b>.json；
# 网站进程只读文件（不加载语料、不做分型计算）。解释等级与模板选择性在读取时合并（它们随研究文件更新）。

def layer_base_dir(model_key: str, run: str) -> Path:
    from shared.dist_protocol import neuron_layers as NL
    return native_root() / '_census' / model_key / run / 'neuron_layers' / NL.code_hash()


def build_layer_bases(model_key: str, run: str, log=print, guard=None) -> Dict[str, Any]:
    """生成该普查运行缺失的逐层摘要（每块一个文件，已有的跳过）。语料 token 流只加载一次。"""
    from shared.dist_protocol import neuron_layers as NL
    man = census_manifest(model_key, run)
    out_dir = layer_base_dir(model_key, run)
    todo = [b for b in range(man['n_layers']) if not (out_dir / f'L{b}.json').is_file()]
    if not todo:
        return {'computed': 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    _, streams = corpus(man['corpus_fp'])
    tok = tokenizer(man['model_path'])
    for b in todo:
        if guard is not None:
            guard.start(f'neuron_layers:{model_key}:{b}')
        with np.load(native_root() / '_census' / model_key / run / f'L{b}.npz') as z:
            base = NL.summarize({k: z[k] for k in z.files}, streams, lambda i: _decode(tok, [i]))
        p = out_dir / f'L{b}.json'
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(base, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        tmp.replace(p)
        log(f'[neuron_layers] {model_key} L{b}')
    return {'computed': len(todo)}


@lru_cache(maxsize=4)
def _layer_base(path: str, stamp: float) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def layer_base(model_key: str, run: str, block: int) -> Optional[Dict[str, Any]]:
    p = layer_base_dir(model_key, run) / f'L{block}.json'
    try:
        return _layer_base(str(p), p.stat().st_mtime)
    except OSError:
        return None


def _family_item(tm_id: str, factor: str) -> Optional[str]:
    from shared.dist_protocol import pieces as PC
    fam = tm_id[3:] if tm_id.startswith('TG-') else tm_id
    try:
        return PC.factor_item(PC.load_tree(), fam, factor)
    except Exception:
        return None


def selectivity_summary(model_key: str, top: int = 5) -> List[Dict[str, Any]]:
    """该模型的模板选择性：每个运行按读出位置给出逐块入选单元数与前列单元（d_held、偏向水平）。
    同一模板 × 因子有多个版本时只用最新版本（旧版本的冻结材料已被替换）。"""
    latest: Dict[tuple, Dict[str, Any]] = {}
    for r in selectivity_runs(model_key):
        key = (r['tm_id'], r['factor'], r['model_key'])
        if key not in latest or (r.get('version') or 0) > (latest[key].get('version') or 0):
            latest[key] = r
    out = []
    for r in sorted(latest.values(), key=lambda x: x['file']):
        res = selectivity(run=r['file'], top=20)
        by_ro: Dict[str, List[int]] = {}
        tops: Dict[str, Dict[str, List[list]]] = {}
        n_layers = max((t['block'] for t in res['tags']), default=-1) + 1
        for t in res['tags']:
            by_ro.setdefault(t['readout'], [0] * n_layers)[t['block']] = int(t.get('n_selected') or 0)
            tops.setdefault(t['readout'], {})[str(t['block'])] = [
                [int(u['unit']), round(float(u.get('d_held') or 0), 3), u.get('pos_level')] for u in t['top_by_d'][:top]]
        out.append({'file': r['file'], 'tm_id': r['tm_id'], 'version': r.get('version'), 'factor': r['factor'], 'hi': r['hi'], 'lo': r['lo'],
                    'item': _family_item(r['tm_id'] or '', r['factor'] or ''), 'by_readout': by_ro, 'top': tops})
    return out


def _explained_by_block(model_key: str) -> Dict[int, Dict[int, str]]:
    out: Dict[int, Dict[int, str]] = {}
    for (mk, b, u), recs in expl_index().items():
        if mk == model_key and recs:
            out.setdefault(b, {})[u] = recs[-1]['grade']             # 以最新一次检验为准
    return out


@router.get('/layer')
def neuron_layer(model_key: str = Query(...), block: int = Query(..., ge=0), run: Optional[str] = Query(None)):
    """一个块全部 MLP 中间单元的列式摘要（按真实索引）、解释等级与模板选择性前列。摘要还没生成时返回 pending。"""
    from shared.dist_protocol import neuron_layers as NL
    c = pick_census(model_key, run)
    man = census_manifest(model_key, c['run'])
    if block >= man['n_layers']:
        raise HTTPException(404, '块号超出范围')
    base = layer_base(model_key, c['run'], block)
    if base is None:
        return {'schema': NL.SCHEMA, 'pending': True, 'model_key': model_key, 'run': c['run'], 'block': block,
                'note': '逐层摘要正在后台生成（每块约十秒，首次全部生成需要几分钟），请稍后刷新。'}
    sel = []
    for s in selectivity_summary(model_key, top=20):                  # 阵列着色：每个运行的前 20 名全部标出
        sel.append({k: s[k] for k in ('file', 'tm_id', 'version', 'factor', 'hi', 'lo', 'item')}
                   | {'n_selected': {ro: v[block] if block < len(v) else 0 for ro, v in s['by_readout'].items()},
                      'top': {ro: v.get(str(block), []) for ro, v in s['top'].items()}})
    return {**NL.layer_doc(base, model_key=model_key, run=c['run'], block=block,
                           explained=_explained_by_block(model_key).get(block, {}), selectivity=sel), 'pending': False}


@router.get('/tower')
def neuron_tower(model_key: str = Query(...), run: Optional[str] = Query(None)):
    """全部块的叠加量（层塔着色用）：分型计数、强激活率中位数、已解释单元数、各模板逐块选择性单元数。"""
    from shared.dist_protocol import neuron_layers as NL
    c = pick_census(model_key, run)
    man = census_manifest(model_key, c['run'])
    bases = {b: layer_base(model_key, c['run'], b) for b in range(man['n_layers'])}
    bases = {b: v for b, v in bases.items() if v is not None}
    doc = NL.tower_doc(model_key, c['run'], man['n_layers'], man['d_ff'], bases, _explained_by_block(model_key),
                       selectivity_summary(model_key))
    return {**doc, 'pending': False, 'summaries_ready': len(bases)}
