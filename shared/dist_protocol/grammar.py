# -*- coding: utf-8 -*-
"""生成式语言模板（template_grammar.v1）。

一个模板族 = 若干框架（frame，带句式等因子水平）× 若干填充（filler，带知识/实体等因子水平
与槽位文本）。每个 (frame, filler) 组合生成一条测试样本；条件 cond = frame.levels ∪ filler.levels。

切分纪律（防近重复泄漏）：
- 按 **填充** 分层确定性切分 discovery / confirm / lockbox：同一填充的全部句式落在同一划分，
  lockbox 中的词汇在 discovery 中从未出现（未见词表）。
- ``heldout_frames`` 中的框架只进入 ``lockbox_frame``（未见句式），不参与发现。
- 切分只依赖 (family_id, salt, filler_id) 的 SHA256，可在任何节点复算。

spec 示例见 shared/dist_protocol/grammars/*.json。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 'template_grammar.v1'
SPLITS = ('discovery', 'confirm', 'lockbox')
FRAME_SPLIT = 'lockbox_frame'
ALL_SPLITS = SPLITS + (FRAME_SPLIT,)
DEFAULT_FRACTIONS = {'discovery': 0.6, 'confirm': 0.2, 'lockbox': 0.2}
DEFAULT_LAYERS = ['first', 'mid', 'last']
MAX_ITEMS_HARD = 50000
SLOT_RE = re.compile(r'\{([A-Za-z_][A-Za-z0-9_]*)\}')
ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')


class GrammarError(ValueError):
    """模板规格不合法（返回给客户端的可读错误列表）。"""

    def __init__(self, errors: List[str]):
        super().__init__('; '.join(errors))
        self.errors = errors


def cond_key(cond: Dict[str, Any]) -> str:
    """与 runner 相同的条件键：按因子名排序的 k=v，用 | 连接。"""
    return '|'.join(f'{k}={cond[k]}' for k in sorted(cond))


def _h(*parts: str) -> int:
    return int(hashlib.sha256('\x1f'.join(parts).encode('utf-8')).hexdigest()[:16], 16)


def _fill(pattern: str, slots: Dict[str, str]) -> str:
    return SLOT_RE.sub(lambda m: slots[m.group(1)], pattern)


def _fill_with_offset(pattern: str, slots: Dict[str, str], target: str) -> Tuple[str, int]:
    """填充并返回目标槽位（第一次出现）在成句中的字符起点，避免目标词是其他词子串时定位错误。"""
    out, pos, start = [], 0, -1
    for m in SLOT_RE.finditer(pattern):
        out.append(pattern[pos:m.start()])
        if m.group(1) == target and start < 0:
            start = sum(len(x) for x in out)
        out.append(slots[m.group(1)])
        pos = m.end()
    out.append(pattern[pos:])
    return ''.join(out), start


def validate(spec: Dict[str, Any]) -> List[str]:
    """返回错误列表；空列表 = 合法。只做结构与引用检查，不判断语言学合理性。"""
    errors: List[str] = []
    if not isinstance(spec, dict):
        return ['spec 必须是 JSON 对象']
    if spec.get('schema_version') != SCHEMA_VERSION:
        errors.append(f'schema_version 必须为 {SCHEMA_VERSION}')
    fam = spec.get('family_id')
    if not isinstance(fam, str) or not ID_RE.match(fam):
        errors.append('family_id 必须是 1–64 位字母数字/._- 标识')
    factors = spec.get('factors')
    if not isinstance(factors, list) or not factors or any(not isinstance(f, str) or not f for f in factors):
        errors.append('factors 必须是非空字符串列表')
        factors = []
    if len(set(factors)) != len(factors):
        errors.append('factors 不能重复')
    frames = spec.get('frames')
    fillers = spec.get('fillers')
    if not isinstance(frames, list) or not frames:
        errors.append('frames 必须是非空列表')
        frames = []
    if not isinstance(fillers, list) or not fillers:
        errors.append('fillers 必须是非空列表')
        fillers = []
    seen = set()
    for i, fr in enumerate(frames):
        where = f'frames[{i}]'
        if not isinstance(fr, dict):
            errors.append(f'{where} 必须是对象'); continue
        fid = fr.get('id')
        if not isinstance(fid, str) or not ID_RE.match(fid):
            errors.append(f'{where}.id 不合法')
        elif ('frame', fid) in seen:
            errors.append(f'{where}.id 重复：{fid}')
        seen.add(('frame', fid))
        if not isinstance(fr.get('text'), str) or not fr['text'].strip():
            errors.append(f'{where}.text 必须是非空字符串')
        if not isinstance(fr.get('target'), str):
            errors.append(f'{where}.target 必须是槽位名')
        elif isinstance(fr.get('text'), str) and '{' + fr['target'] + '}' not in fr['text']:
            errors.append(f'{where}.target 槽位 {{{fr["target"]}}} 不在 text 中')
        if not isinstance(fr.get('levels', {}), dict):
            errors.append(f'{where}.levels 必须是对象')
        probe = fr.get('probe')
        if probe is not None:
            if (not isinstance(probe, dict) or not isinstance(probe.get('prompt'), str)
                    or not isinstance(probe.get('answer'), str)):
                errors.append(f'{where}.probe 必须含 prompt 与 answer 字符串')
    for i, fl in enumerate(fillers):
        where = f'fillers[{i}]'
        if not isinstance(fl, dict):
            errors.append(f'{where} 必须是对象'); continue
        fid = fl.get('id')
        if not isinstance(fid, str) or not ID_RE.match(fid):
            errors.append(f'{where}.id 不合法')
        elif ('filler', fid) in seen:
            errors.append(f'{where}.id 重复：{fid}')
        seen.add(('filler', fid))
        if not isinstance(fl.get('slots'), dict) or not fl['slots']:
            errors.append(f'{where}.slots 必须是非空对象')
        elif any(not isinstance(v, str) or not v for v in fl['slots'].values()):
            errors.append(f'{where}.slots 的值必须是非空字符串')
        if not isinstance(fl.get('levels', {}), dict):
            errors.append(f'{where}.levels 必须是对象')
    # 因子覆盖：每个组合的 cond 必须恰好包含 factors
    if not errors:
        for fr in frames:
            for fl in fillers:
                cond = {**fr.get('levels', {}), **fl.get('levels', {})}
                overlap = set(fr.get('levels', {})) & set(fl.get('levels', {}))
                if overlap:
                    errors.append(f'frame {fr["id"]} 与 filler {fl["id"]} 同时定义因子 {sorted(overlap)}')
                    break
                if set(cond) != set(factors):
                    errors.append(f'frame {fr["id"]} × filler {fl["id"]} 的因子 {sorted(cond)} 与 factors {sorted(factors)} 不一致')
                    break
                needed = set(SLOT_RE.findall(fr['text']))
                if fr.get('probe'):
                    needed |= set(SLOT_RE.findall(fr['probe']['prompt'])) | set(SLOT_RE.findall(fr['probe']['answer']))
                missing = needed - set(fl['slots'])
                if missing:
                    errors.append(f'filler {fl["id"]} 缺少 frame {fr["id"]} 需要的槽位 {sorted(missing)}')
                    break
            if len(errors) > 20:
                break
    splits = spec.get('splits', {})
    if not isinstance(splits, dict):
        errors.append('splits 必须是对象')
    else:
        fr_ = splits.get('fractions', DEFAULT_FRACTIONS)
        if (not isinstance(fr_, dict) or set(fr_) != set(SPLITS)
                or any(not isinstance(v, (int, float)) or v < 0 for v in fr_.values())
                or abs(sum(fr_.values()) - 1.0) > 1e-6):
            errors.append('splits.fractions 必须给出 discovery/confirm/lockbox 三个非负比例且和为 1')
        held = splits.get('heldout_frames', [])
        frame_ids = {fr.get('id') for fr in frames if isinstance(fr, dict)}
        if not isinstance(held, list) or any(h not in frame_ids for h in held):
            errors.append('splits.heldout_frames 必须引用已定义的 frame id')
        elif frame_ids and set(held) == frame_ids:
            errors.append('splits.heldout_frames 不能包含全部 frame（否则没有发现材料）')
    layers = spec.get('layers', DEFAULT_LAYERS)
    if not isinstance(layers, list) or not layers or any(t not in ('first', 'mid', 'last') and not (isinstance(t, int)) for t in layers):
        errors.append('layers 只能是 first/mid/last 或整数层号')
    shards = spec.get('shards', 1)
    if not isinstance(shards, int) or shards < 1 or shards > 256:
        errors.append('shards 必须是 1–256 的整数')
    # 跨族概念预注册（atlas.v1）：{"因子=水平": "概念"}，声明该水平应与其他模式族里同概念的水平共享编码
    concepts = spec.get('concepts', {})
    if not isinstance(concepts, dict):
        errors.append('concepts 必须是对象 {"因子=水平": "概念"}')
    else:
        levels = {f: set() for f in (factors or [])}
        for it in list(frames) + list(fillers):
            if isinstance(it, dict) and isinstance(it.get('levels'), dict):
                for f, v in it['levels'].items():
                    levels.setdefault(f, set()).add(str(v))
        for k, v in concepts.items():
            f, _, lv = str(k).partition('=')
            if not isinstance(v, str) or not v.strip():
                errors.append(f'concepts[{k}] 必须是非空字符串')
            elif f not in levels or lv not in levels[f]:
                errors.append(f'concepts 键 {k} 没有对应的因子水平')
    return errors


def assign_splits(spec: Dict[str, Any]) -> Dict[str, str]:
    """按填充分层（填充自身的因子水平组合为一层）确定性分配划分。

    每层内按 SHA256(family, salt, filler_id) 排序，按比例取整分配；层大小 ≥3 时保证
    三个划分各至少 1 个填充，从而每个因子水平在每个划分都有材料。
    """
    fam = spec['family_id']
    salt = str(spec.get('splits', {}).get('salt', 'v1'))
    fractions = spec.get('splits', {}).get('fractions', DEFAULT_FRACTIONS)
    strata: Dict[str, List[str]] = {}
    for fl in spec['fillers']:
        strata.setdefault(cond_key(fl.get('levels', {})), []).append(fl['id'])
    out: Dict[str, str] = {}
    for _, ids in sorted(strata.items()):
        ids = sorted(ids, key=lambda i: _h(fam, salt, i))
        n = len(ids)
        counts = {s: int(math.floor(fractions[s] * n)) for s in SPLITS}
        if n >= 3:
            for s in SPLITS:
                if fractions[s] > 0 and counts[s] == 0:
                    counts[s] = 1
        # 余数按比例最大者补齐 / 超额从 discovery 以外回收
        while sum(counts.values()) < n:
            s = max(SPLITS, key=lambda k: fractions[k] * n - counts[k])
            counts[s] += 1
        while sum(counts.values()) > n:
            s = max(SPLITS, key=lambda k: counts[k] - fractions[k] * n)
            counts[s] -= 1
        cursor = 0
        for s in SPLITS:
            for fid in ids[cursor:cursor + counts[s]]:
                out[fid] = s
            cursor += counts[s]
    return out


def generate(spec: Dict[str, Any], *, include_canary: bool = True,
             per_cell_limit: Optional[int] = None) -> Dict[str, Any]:
    """生成 runner_v3 可读的 corpus（不含 tm_id/version，由注册方补齐）。

    per_cell_limit：试跑用，每个 (split, cond) 最多取前 k 个填充（按切分哈希序）。
    """
    errors = validate(spec)
    if errors:
        raise GrammarError(errors)
    from .canary import CANARY_ITEMS  # 局部导入避免循环

    split_of = assign_splits(spec)
    held = set(spec.get('splits', {}).get('heldout_frames', []))
    fam = spec['family_id']
    salt = str(spec.get('splits', {}).get('salt', 'v1'))
    items: List[Dict[str, Any]] = []
    warnings: List[str] = []
    per_cell: Dict[Tuple[str, str], int] = {}
    fillers = sorted(spec['fillers'], key=lambda f: _h(fam, salt, f['id']))
    for fr in spec['frames']:
        for fl in fillers:
            split = FRAME_SPLIT if fr['id'] in held else split_of[fl['id']]
            cond = {**fr.get('levels', {}), **fl.get('levels', {})}
            ck = cond_key(cond)
            if per_cell_limit is not None:
                if per_cell.get((split, ck), 0) >= per_cell_limit:
                    continue
            per_cell[(split, ck)] = per_cell.get((split, ck), 0) + 1
            slots = fl['slots']
            text, start = _fill_with_offset(fr['text'], slots, fr['target'])
            target = slots[fr['target']]
            if fr['text'].count('{' + fr['target'] + '}') > 1:
                warnings.append(f'{fr["id"]}: 目标槽位在句式中出现多次，取第一次出现')
            item = {
                'id': f'{fam}:{fr["id"]}:{fl["id"]}',
                'cond': cond,
                'split': split,
                'frame_id': fr['id'],
                'filler_id': fl['id'],
                'text': text,
                'target': target,
                'target_char': start,
            }
            if fr.get('probe'):
                item['probe'] = {'prompt': _fill(fr['probe']['prompt'], slots),
                                 'answer': _fill(fr['probe']['answer'], slots)}
            items.append(item)
    max_items = min(int(spec.get('max_items', MAX_ITEMS_HARD)), MAX_ITEMS_HARD)
    if len(items) > max_items:
        raise GrammarError([f'生成 {len(items)} 条，超过 max_items={max_items}；请拆分模板族或调高上限'])
    if include_canary:
        items.extend(dict(c) for c in CANARY_ITEMS)
    corpus = {
        'schema_version': 'corpus.v3',
        'protocol': 'dist_protocol.v3',
        'analysis': 'suffstats',
        'family_id': fam,
        'module': spec.get('module', ''),
        'name': spec.get('name', fam),
        'description': spec.get('description', ''),
        'factors': list(spec['factors']),
        'layers': list(spec.get('layers', DEFAULT_LAYERS)),
        'shards': int(spec.get('shards', 1)),
        'prereg': dict(spec.get('prereg', {})),
        'language_graph_target_ids': list(spec.get('language_graph_target_ids', [])),
        'concepts': dict(spec.get('concepts', {})),
        'grammar_sha256': spec_sha(spec),
        'items': items,
    }
    corpus['_warnings'] = sorted(set(warnings))[:50]
    return corpus


def spec_sha(spec: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def _chi2_quantile(p: float, df: int) -> float:
    """Wilson–Hilferty 近似的卡方分位数。"""
    z = NormalDist().inv_cdf(p)
    a = 2.0 / (9.0 * df)
    return df * (1 - a + z * math.sqrt(a)) ** 3


def mde_eta2(k_levels: int, n_total: int, *, dims: int = 2560, alpha: float = 0.05,
             power: float = 0.8) -> Optional[float]:
    """近似最小可检测 η²（单因子、Bonferroni 校正到 dims 个维度）。

    使用大样本卡方近似：λ 使 P(χ²(df, λ) > crit) ≈ power，非中心卡方用正态近似；
    η² = λ / (λ + N)。只用于设计阶段估算样本量，不作为统计检验。
    """
    df = k_levels - 1
    if df < 1 or n_total <= k_levels:
        return None
    a = alpha / max(dims, 1)
    crit = _chi2_quantile(1 - a, df)
    zp = NormalDist().inv_cdf(power)
    lo, hi = 0.0, 1e6
    for _ in range(100):
        lam = (lo + hi) / 2
        mean, sd = df + lam, math.sqrt(2 * (df + 2 * lam))
        if (mean - crit) / sd >= zp:
            hi = lam
        else:
            lo = lam
    return hi / (hi + n_total)


def preview(spec: Dict[str, Any], *, dims: int = 2560, sample: int = 12) -> Dict[str, Any]:
    """模板工作室预览：生成统计、逐格计数、功效估算、样例与告警（不写盘）。"""
    errors = validate(spec)
    if errors:
        return {'ok': False, 'errors': errors}
    try:
        corpus = generate(spec, include_canary=False)
    except GrammarError as exc:
        return {'ok': False, 'errors': exc.errors}
    items = corpus['items']
    factors = corpus['factors']
    cells: Dict[str, Dict[str, int]] = {}
    for it in items:
        cells.setdefault(cond_key(it['cond']), {s: 0 for s in ALL_SPLITS})[it['split']] += 1
    by_split = {s: sum(1 for it in items if it['split'] == s) for s in ALL_SPLITS}
    levels: Dict[str, List[str]] = {f: sorted({str(it['cond'][f]) for it in items}) for f in factors}
    power = []
    n_disc = by_split['discovery']
    for f in factors:
        k = len(levels[f])
        power.append({'factor': f, 'levels': k, 'n_discovery': n_disc,
                      'mde_eta2': mde_eta2(k, n_disc, dims=dims)})
    min_cell = int(spec.get('prereg', {}).get('min_cell_n', 8))
    thin = sorted(ck for ck, c in cells.items()
                  if c['discovery'] < min_cell and (c['discovery'] + c['confirm'] + c['lockbox']) > 0)
    warnings = list(corpus.get('_warnings', []))
    if thin:
        warnings.append(f'{len(thin)} 个条件格 discovery 样本少于 prereg.min_cell_n={min_cell}：{", ".join(thin[:6])}{" …" if len(thin) > 6 else ""}')
    if by_split['lockbox'] == 0:
        warnings.append('lockbox 为空：无法做未见词表预测检验')
    char_len = [len(it['text']) for it in items]
    step = max(1, len(items) // sample)
    return {
        'ok': True,
        'family_id': corpus['family_id'],
        'grammar_sha256': corpus['grammar_sha256'],
        'n_items': len(items),
        'n_frames': len(spec['frames']),
        'n_fillers': len(spec['fillers']),
        'factors': factors,
        'levels': levels,
        'by_split': by_split,
        'cells': [{'cell': ck, **cnt} for ck, cnt in sorted(cells.items())],
        'power': power,
        'probe_items': sum(1 for it in items if 'probe' in it),
        'text_chars': {'min': min(char_len), 'max': max(char_len),
                       'mean': round(sum(char_len) / len(char_len), 2)},
        'shards': corpus['shards'],
        'samples': [{k: it[k] for k in ('id', 'split', 'cond', 'text', 'target', 'target_char', 'probe') if k in it}
                    for it in items[::step][:sample]],
        'warnings': warnings,
        'note': '功效为大样本近似（Bonferroni 到隐藏维度数），仅用于设计阶段估算样本量；token 长度由 runner 按实际 tokenizer 记录。',
    }


def pilot_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """试跑版本：同一文法，单分片；生成时按每格上限抽样（见 generate per_cell_limit）。"""
    out = json.loads(json.dumps(spec))
    out['shards'] = 1
    return out
