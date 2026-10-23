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


@lru_cache(maxsize=4)
def census_manifest(model_key: str, run: str) -> Dict[str, Any]:
    return json.loads((native_root() / '_census' / model_key / run / 'manifest.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=8)
def census_layer(model_key: str, run: str, block: int) -> Dict[str, np.ndarray]:
    p = native_root() / '_census' / model_key / run / f'L{block}.npz'
    if not p.is_file():
        raise HTTPException(404, f'普查没有第 {block} 块')
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=2)
def readout_index(model_key: str, run: str, n_layers: int):
    """各层的直接读出 token 与最大激活（只读这两个数组），供检索使用。"""
    out = []
    for b in range(n_layers):
        with np.load(native_root() / '_census' / model_key / run / f'L{b}.npz') as z:
            out.append((z['readout_top'], z['amax']))
    return out


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
        'caveats': ['极值片段天然有选择偏差：它们是 top / bottom-k，不代表典型激活',
                    '语料以教育筛选网页与百科为主（中英），不含对话、代码与广告页',
                    '读出 token 只是直接通路近似'],
    }
    sel = card['template_selectivity']
    read = '留出复现' if any(r.get('selected') for r in sel) else ('有模板记录，未达入选判据' if sel else '未检验')
    iv = card['interventions']
    control = ('范围内干预（组级，20 单元一起替换）' if any(r.get('causal_use_in_scope') == '是' for r in iv)
               else ('组级干预未通过' if iv else '未检验'))
    card['evidence_axes'] = {'read': read, 'control': control, 'explain': '未检验'}
    return card


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
