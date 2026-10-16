# -*- coding: utf-8 -*-
"""生成式语言模板（template_grammar.v1）。

一个模板族 = 若干框架（frame，带句式等因子水平）× 若干填充（filler，带知识/实体等因子水平
与槽位文本）。每个 (frame, filler) 组合生成一条测试样本；条件 cond = frame.levels ∪ filler.levels。

切分纪律（防近重复泄漏）：
- 按 **填充** 分层确定性切分 discovery / confirm / lockbox：同一填充的全部句式落在同一划分，
  lockbox 中的词汇在 discovery 中从未出现（未见词表）。
- ``heldout_frames`` 中的框架只进入 ``lockbox_frame``（未见句式），不参与发现。
- 切分只依赖 (family_id, salt, filler_id) 的 SHA256，可在任何节点复算。

读出位置（readouts，缺省 ["target"]）：
- ``target``：目标槽位最后一个 token（frame.target_occurrence = first|last 选择槽位出现次序）；
- ``cue``：frame.cue 给出的句式片段（字面片段或 "{槽位}"）的最后一个 token；
- ``final``：整句最后一个 token；
- ``probe_answer``：probe.prompt 最后一个 token（预测答案首 token 的位置）。

因果可见性（causal_visibility）：因果注意力下，读出位置的状态只依赖其前缀。对每个句式因子，
找出仅在该因子上不同的句式对；若两者在某读出位置的模板前缀（句首目标 = 空前缀）相同，该因子
在此位置按构造不可见。所有读出位置都不可见 → validate 报错；部分位置不可见 → preview 告警。
答案槽（probe.answer 中除目标外的槽位）或类别标签槽出现在目标之前 → 告警（读出可能来自标签词）。
对照句式对一方有句首前缀、另一方没有（或长度相差很大）→ 告警「前缀 / 位置混杂」（prefix_warnings，不报错）。

行为 probe（frame.probe）：
- ``answer``：主答案（旧口径：首 token 是否 top-1，继续记录）；
- ``answers``：可接受答案集（同义词、带后缀形式，如 "{A}"、"{A}色"）；
- ``contrast``：对照答案集（字符串列表，或 auto:other / auto:same 从同一划分的其他填充自动生成）。
  有对照时行为门按完整答案对数概率做强制选择：max log P(可接受) > max log P(对照) 记为正确；
  没有对照时按"下一个 token 是否为任一可接受答案的首 token"判定。
- filler.answers_extra：该填充额外的可接受答案（近义词，如 冰→冷 另接受「凉」），不进入任何对照集。

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
LAYER_WORDS = ('first', 'mid', 'last', 'emb', 'all')
READOUTS = ('target', 'cue', 'final', 'probe_answer')
# probe.contrast 的自动模式：用同一划分中其他填充的可接受答案作对照（强制选择）。
#   auto:other —— 填充因子水平不同的填充（如 KN-ISA：其他类别的类别词）；
#   auto:same  —— 填充因子水平相同的其他填充（如 KN-ATTR：同一属性类型的其他属性值）。
CONTRAST_AUTO = ('auto:other', 'auto:same')
DEFAULT_READOUTS = ['target']
PREFIX_RATIO_WARN = 3.0     # 对照句式对的句首前缀长度比 ≥ 3 且相差 ≥ 4 字（或一方为空）时告警「前缀 / 位置混杂」
PREFIX_DIFF_WARN = 4
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


def _segments(pattern: str, slots: Optional[Dict[str, str]] = None) -> List[Tuple[int, int, int, int, Optional[str]]]:
    """把模板切成 (tpl_s, tpl_e, out_s, out_e, slot|None) 段；slots 为 None 时 out 坐标 = 模板坐标。"""
    segs, pos, out = [], 0, 0
    for m in SLOT_RE.finditer(pattern):
        if m.start() > pos:
            n = m.start() - pos
            segs.append((pos, m.start(), out, out + n, None))
            out += n
        n = len(slots[m.group(1)]) if slots is not None else m.end() - m.start()
        segs.append((m.start(), m.end(), out, out + n, m.group(1)))
        out += n
        pos = m.end()
    if pos < len(pattern):
        segs.append((pos, len(pattern), out, out + len(pattern) - pos, None))
    return segs


def _target_span(pattern: str, target: str, occurrence: str = 'first') -> Optional[Tuple[int, int]]:
    """目标槽位在模板中的 [起, 止)（first / last 次出现）。"""
    hits = [(m.start(), m.end()) for m in SLOT_RE.finditer(pattern) if m.group(1) == target]
    if not hits:
        return None
    return hits[-1] if occurrence == 'last' else hits[0]


def _cue_span(pattern: str, cue: str) -> Optional[Tuple[int, int]]:
    """cue 在模板中的 [起, 止)：「{槽位}」取该槽位第一次出现；字面片段取第一处不跨槽位的出现。"""
    m = SLOT_RE.fullmatch(cue)
    if m:
        return _target_span(pattern, m.group(1))
    start = 0
    while True:
        i = pattern.find(cue, start)
        if i < 0:
            return None
        if all(sl is None or e <= i or s >= i + len(cue) for s, e, _, _, sl in _segments(pattern)):
            return i, i + len(cue)
        start = i + 1


def _map_span(pattern: str, slots: Dict[str, str], span: Tuple[int, int]) -> Tuple[int, int]:
    """模板 [起, 止) → 成句字符 [起, 止)。槽位整体映射，字面片段按偏移映射。"""
    s0, e0 = span
    out_s = out_e = None
    for ts, te, os_, oe, sl in _segments(pattern, slots):
        if sl is not None:
            if ts == s0:
                out_s = os_
            if te == e0:
                out_e = oe
        else:
            if ts <= s0 < te:
                out_s = os_ + (s0 - ts)
            if ts < e0 <= te:
                out_e = os_ + (e0 - ts)
    if out_s is None or out_e is None:
        raise ValueError(f'span {span} 无法映射到成句')
    return out_s, out_e


def _probe_answers(probe: Dict[str, Any]) -> List[str]:
    """可接受答案模板：answers（含 answer 时保证 answer 在首位）或仅 answer。"""
    ans = probe.get('answers') if isinstance(probe.get('answers'), list) else []
    first = probe.get('answer')
    out = [first] if isinstance(first, str) else []
    out += [a for a in ans if isinstance(a, str) and a not in out]
    return out


def _probe_sets(probe: Dict[str, Any], fl: Dict[str, Any], pool: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """某个填充的 (可接受答案, 对照答案) 字符串。对照不含与可接受答案相同的字符串。"""
    answers = []
    for s in [_fill(t, fl['slots']) for t in _probe_answers(probe)] + list(fl.get('answers_extra', [])):
        if s not in answers:
            answers.append(s)
    con = probe.get('contrast')
    contrast: List[str] = []
    if isinstance(con, list):
        cands = [_fill(t, fl['slots']) for t in con]
    elif con in CONTRAST_AUTO:
        mine = fl.get('levels', {})
        others = [o for o in pool if o['id'] != fl['id']
                  and ((o.get('levels', {}) != mine) if con == 'auto:other' else (o.get('levels', {}) == mine))]
        cands = [_fill(t, o['slots']) for o in others for t in _probe_answers(probe)]
    else:
        cands = []
    for s in cands:
        if s not in answers and s not in contrast:
            contrast.append(s)
    return answers, sorted(contrast)


def readouts_of(spec: Dict[str, Any]) -> List[str]:
    r = spec.get('readouts', DEFAULT_READOUTS)
    return list(r) if isinstance(r, list) else list(DEFAULT_READOUTS)


def layers_of(spec: Dict[str, Any]) -> List[Any]:
    """规范化层位声明：字符串 "all" → ["all"]。"""
    layers = spec.get('layers', DEFAULT_LAYERS)
    return [layers] if isinstance(layers, str) else list(layers)


def _prefix(fr: Dict[str, Any], readout: str) -> Optional[str]:
    """读出位置在模板层面的前缀（含读出片段本身）；该句式没有这个读出位置时返回 None。"""
    text = fr['text']
    if readout == 'target':
        sp = _target_span(text, fr['target'], fr.get('target_occurrence', 'first'))
        return text[:sp[1]] if sp else None
    if readout == 'cue':
        sp = _cue_span(text, fr['cue']) if isinstance(fr.get('cue'), str) else None
        return text[:sp[1]] if sp else None
    if readout == 'final':
        return text
    if readout == 'probe_answer':
        return fr['probe']['prompt'] if fr.get('probe') else None
    return None


def label_slots(spec: Dict[str, Any]) -> List[str]:
    """类别标签槽：槽位文本由填充因子水平唯一决定、且不同水平取值不同（如 KN-ISA 的 {Y} = 类别词）。"""
    by_level: Dict[str, Dict[str, set]] = {}
    for fl in spec.get('fillers', []):
        if not isinstance(fl, dict) or not fl.get('levels') or not isinstance(fl.get('slots'), dict):
            continue
        key = cond_key(fl['levels'])
        for k, v in fl['slots'].items():
            by_level.setdefault(k, {}).setdefault(key, set()).add(v)
    out = []
    for slot, lv in by_level.items():
        vals = [next(iter(v)) for v in lv.values() if len(v) == 1]
        if len(lv) >= 2 and len(vals) == len(lv) and len(set(vals)) == len(vals):
            out.append(slot)
    return sorted(out)


def causal_visibility(spec: Dict[str, Any]) -> Dict[str, Any]:
    """因果可见性检查（模板片段近似，不依赖 tokenizer）。

    返回 {'matrix': {readout: {factor: bool}}, 'pairs': [...], 'errors': [...], 'warnings': [...]}。
    matrix[r][f] = True：f 的每个对照句式对在读出 r 的前缀都不同（填充因子：每个句式在 r 的前缀里都有槽位文本）。
    """
    frames = [fr for fr in spec.get('frames', []) if isinstance(fr, dict)]
    factors = list(spec.get('factors', []))
    readouts = readouts_of(spec)
    frame_factors = [f for f in factors if any(f in fr.get('levels', {}) for fr in frames)]
    matrix: Dict[str, Dict[str, bool]] = {r: {} for r in readouts}
    pairs_out, errors, warnings = [], [], []
    for f in factors:
        if f not in frame_factors:
            for r in readouts:
                pres = [_prefix(fr, r) for fr in frames]
                matrix[r][f] = all(p is not None and SLOT_RE.search(p) is not None for p in pres)
            continue
        others = [g for g in frame_factors if g != f]
        pairs = [(a, b) for i, a in enumerate(frames) for b in frames[i + 1:]
                 if a.get('levels', {}).get(f) != b.get('levels', {}).get(f)
                 and all(a.get('levels', {}).get(g) == b.get('levels', {}).get(g) for g in others)]
        if not pairs:
            warnings.append(f'因子 {f} 没有仅在该因子上不同的句式对，无法检查因果可见性')
            for r in readouts:
                matrix[r][f] = True
            continue
        all_vis = {r: True for r in readouts}
        for a, b in pairs:
            vis = {}
            for r in readouts:
                pa, pb = _prefix(a, r), _prefix(b, r)
                vis[r] = pa is not None and pb is not None and pa != pb
                all_vis[r] = all_vis[r] and vis[r]
            pairs_out.append({'factor': f, 'frames': [a['id'], b['id']], 'visible': vis})
            if not any(vis.values()):
                errors.append(f'因子 {f} 在读出位置不可见：句式 {a["id"]} 与 {b["id"]} 仅在 {f} 上不同，'
                              f'但在全部读出位置 {readouts} 的前缀相同（因果注意力下状态按构造相同）')
            else:
                for r in readouts:
                    if not vis[r]:
                        warnings.append(f'因子 {f} 在读出位置 {r} 不可见：句式 {a["id"]} 与 {b["id"]} 的前缀相同，'
                                        f'该位置上的 {f} 效应只能来自混杂')
        for r in readouts:
            matrix[r][f] = all_vis[r]
    prefix_warnings = []
    for p in pairs_out:
        a, b = (next(fr for fr in frames if fr['id'] == i) for i in p['frames'])
        la, lb = (len(_leading(fr['text'])) for fr in (a, b))
        lo, hi = min(la, lb), max(la, lb)
        if (lo == 0 and hi > 0) or (lo > 0 and hi / lo >= PREFIX_RATIO_WARN and hi - lo >= PREFIX_DIFF_WARN):
            prefix_warnings.append(f'因子 {p["factor"]}：句式 {a["id"]} 与 {b["id"]} 的句首前缀长度为 {la} 与 {lb} 字'
                                   f'（「{_leading(a["text"])}」/「{_leading(b["text"])}」），可能存在前缀 / 位置混杂')
    labels = set(label_slots(spec))
    befores: Dict[str, set] = {}
    for fr in frames:
        text, target = fr.get('text'), fr.get('target')
        if isinstance(text, str) and isinstance(target, str):
            sp = _target_span(text, target, fr.get('target_occurrence', 'first'))
            if sp:
                befores[fr.get('id')] = set(SLOT_RE.findall(text[:sp[0]])) - {target}
    # 在每个句式的目标之前都出现的普通答案槽不随条件变化（如 GR-ROLE 的 {N}{M} 都在复指目标之前），不构成泄漏
    always_before = set.intersection(*befores.values()) if befores else set()
    for fr in frames:
        if fr.get('id') not in befores:
            continue
        target = fr['target']
        before = befores[fr['id']]
        probe = fr.get('probe') if isinstance(fr.get('probe'), dict) else {}
        answer = set(SLOT_RE.findall(probe.get('answer', '') or '')) - {target} - always_before
        for slot in sorted(before & (answer | labels)):
            kind = '类别标签槽' if slot in labels else '答案槽'
            warnings.append(f'句式 {fr.get("id")}：{kind} {{{slot}}} 出现在目标 {{{target}}} 之前，'
                            f'目标位置的读出可能直接来自该词而非目标本身')
    return {'matrix': matrix, 'pairs': pairs_out, 'errors': errors, 'warnings': sorted(set(warnings)),
            'prefix_warnings': sorted(set(prefix_warnings))}


def _leading(text: str) -> str:
    """句首前缀：模板中第一个槽位之前的字面文本。"""
    m = SLOT_RE.search(text)
    return text[:m.start()] if m else text


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
        if fr.get('target_occurrence', 'first') not in ('first', 'last'):
            errors.append(f'{where}.target_occurrence 只能是 first 或 last')
        if 'cue' in fr and (not isinstance(fr['cue'], str) or not fr['cue'] or not isinstance(fr.get('text'), str)
                            or _cue_span(fr['text'], fr['cue']) is None):
            errors.append(f'{where}.cue 必须是 text 中槽位之外的字面片段，或 "{{槽位}}"')
        probe = fr.get('probe')
        if probe is not None:
            if (not isinstance(probe, dict) or not isinstance(probe.get('prompt'), str)
                    or not isinstance(probe.get('answer'), str)):
                errors.append(f'{where}.probe 必须含 prompt 与 answer 字符串')
            else:
                ans = probe.get('answers')
                if ans is not None and (not isinstance(ans, list) or not ans
                                        or any(not isinstance(a, str) or not a for a in ans)):
                    errors.append(f'{where}.probe.answers 必须是非空字符串列表（可接受答案集）')
                con = probe.get('contrast')
                if con is not None and not (con in CONTRAST_AUTO or (isinstance(con, list) and con and all(
                        isinstance(c, str) and c for c in con))):
                    errors.append(f'{where}.probe.contrast 必须是非空字符串列表，或 {list(CONTRAST_AUTO)} 之一')
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
        extra = fl.get('answers_extra')
        if extra is not None and (not isinstance(extra, list) or any(not isinstance(a, str) or not a for a in extra)):
            errors.append(f'{where}.answers_extra 必须是非空字符串列表（该填充额外的可接受答案，不进入对照集）')
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
                    for t in _probe_answers(fr['probe']) + (fr['probe']['contrast'] if isinstance(
                            fr['probe'].get('contrast'), list) else []):
                        needed |= set(SLOT_RE.findall(t))
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
    if layers == 'all':
        layers = ['all']
    if (not isinstance(layers, list) or not layers
            or any(not (t in LAYER_WORDS or (isinstance(t, int) and not isinstance(t, bool) and t >= 0)) for t in layers)
            or ('all' in layers and len(layers) > 1)):
        errors.append('layers 只能是 "all"，或 first/mid/last/emb/非负整数层号组成的列表')
    readouts = spec.get('readouts', DEFAULT_READOUTS)
    if (not isinstance(readouts, list) or not readouts or len(set(readouts)) != len(readouts)
            or any(r not in READOUTS for r in readouts)):
        errors.append(f'readouts 必须是 {list(READOUTS)} 中不重复的非空子集')
    else:
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            if 'cue' in readouts and 'cue' not in fr:
                errors.append(f'readouts 含 cue，但 frame {fr.get("id")} 没有声明 cue')
            if 'probe_answer' in readouts and not fr.get('probe'):
                errors.append(f'readouts 含 probe_answer，但 frame {fr.get("id")} 没有 probe')
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
    if not errors:
        errors.extend(causal_visibility(spec)['errors'])
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

    per_cell_limit：试跑用，只从 discovery 抽样（confirm / lockbox / lockbox_frame 不参与试跑与行为门筛选），
    每个 (cond, 句式) 最多取前 k 个填充（按切分哈希序）。
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
                # 按 (条件格, 句式) 计数：一个条件格有多个句式时，每个句式都进入试跑
                if split != 'discovery' or per_cell.get((split, ck, fr['id']), 0) >= per_cell_limit:
                    continue
            per_cell[(split, ck, fr['id'])] = per_cell.get((split, ck, fr['id']), 0) + 1
            slots = fl['slots']
            text = _fill(fr['text'], slots)
            occ = fr.get('target_occurrence', 'first')
            start = _map_span(fr['text'], slots, _target_span(fr['text'], fr['target'], occ))[0]
            target = slots[fr['target']]
            if fr['text'].count('{' + fr['target'] + '}') > 1 and 'target_occurrence' not in fr:
                warnings.append(f'{fr["id"]}: 目标槽位在句式中出现多次，取第一次出现（可用 target_occurrence 指定）')
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
            if isinstance(fr.get('cue'), str):
                cs, ce = _map_span(fr['text'], slots, _cue_span(fr['text'], fr['cue']))
                item['cue'], item['cue_char'] = text[cs:ce], cs
            if fr.get('probe'):
                item['probe'] = {'prompt': _fill(fr['probe']['prompt'], slots),
                                 'answer': _fill(fr['probe']['answer'], slots)}
                if 'answers' in fr['probe'] or 'contrast' in fr['probe']:
                    # 对照池只取同一划分的填充，避免发现材料的 probe 用到留出填充的答案词
                    pool = [o for o in spec['fillers'] if split_of[o['id']] == split_of[fl['id']]]
                    answers, contrast = _probe_sets(fr['probe'], fl, pool)
                    item['probe']['answers'] = answers
                    item['probe']['contrast'] = contrast
                    if 'contrast' in fr['probe'] and not contrast:
                        warnings.append(f'{fr["id"]}: 填充 {fl["id"]} 的对照答案为空，该条行为评分退回答案集首 token')
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
        'layers': layers_of(spec),
        'readouts': readouts_of(spec),
        'visibility': causal_visibility(spec)['matrix'],
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
    vis = causal_visibility(spec)
    warnings = list(corpus.get('_warnings', [])) + vis['warnings'] + vis['prefix_warnings']
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
        'readouts': corpus['readouts'],
        'layers': corpus['layers'],
        'visibility': {'matrix': vis['matrix'], 'pairs': vis['pairs']},
        'samples': [{k: it[k] for k in ('id', 'split', 'cond', 'text', 'target', 'target_char', 'cue', 'cue_char', 'probe')
                     if k in it}
                    for it in items[::step][:sample]],
        'warnings': warnings,
        'note': '功效为大样本近似（Bonferroni 到隐藏维度数），仅用于设计阶段估算样本量；token 长度由 runner 按实际 tokenizer 记录。',
    }


def pilot_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """试跑版本：同一文法，单分片；生成时按每格上限抽样（见 generate per_cell_limit）。"""
    out = json.loads(json.dumps(spec))
    out['shards'] = 1
    return out
