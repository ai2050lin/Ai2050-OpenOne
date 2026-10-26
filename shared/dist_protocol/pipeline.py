# -*- coding: utf-8 -*-
"""中心节点侧派生视图：合并、覆盖矩阵、审计自动检查、证据门链、总览漏斗、预测记分板。

全部为只读计算：输入是分布式 SQLite（templates / tasks / results / audits / canary_refs）与
内容寻址对象；输出给客户端。任何结果都不会因为这里的计算而自动成为 Evidence ——
审计通过只表示「这份执行产物可信、可进入合并」，证据等级仍由 Research OS 人工登记。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import canary as canary_mod
from . import analysis as analysis_mod
from .suffstats import SuffStats, replicate_consistency, scoreboard_rows, summarize

V3_ARTIFACTS = ('summary.json', 'suffstats.npz', 'behavior.json', 'canary.json')

# 证据门链（README「机制边闭合门」的执行顺序）。observational=分布式观测协议可以自动判定；
# 其余需要干预或生成实验，当前协议不判定，只显示「未运行」。
GATE_CHAIN = [
    {'id': 'behavior', 'label': '行为门', 'auto': True, 'desc': '模型在该条件格上确实完成任务（probe 准确率 ≥ 预注册阈值；有对照集时按完整答案对数概率强制选择，否则按答案集首 token）'},
    {'id': 'instrument', 'label': '仪器门', 'auto': True, 'desc': '金丝雀摘要与同快照/同精度参考一致；无漂移结果'},
    {'id': 'repeat', 'label': '重复', 'auto': True, 'desc': '同一分片被不同节点独立重跑且统计量一致'},
    {'id': 'confirm', 'label': '留出复现', 'auto': True, 'desc': 'discovery 方向在 confirm 划分（同分布未见填充）上复现'},
    {'id': 'lockbox', 'label': '未见词表预测', 'auto': True, 'desc': 'discovery 方向在 lockbox（未见词表）上达到预注册 cos 阈值'},
    {'id': 'lockbox_frame', 'label': '未见句式预测', 'auto': True, 'desc': 'discovery 方向在 heldout 句式上达到预注册 cos 阈值'},
    {'id': 'cross_model', 'label': '跨模型', 'auto': True, 'desc': '≥2 个模型快照各自通过未见词表预测（只比较功能结构，不对齐坐标）'},
    {'id': 'selectivity', 'label': '选择性/充分性', 'auto': False, 'desc': '干预只改变目标能力（需 patch/steering 实验）'},
    {'id': 'necessity', 'label': '必要性与负控制', 'auto': False, 'desc': '删除破坏对应能力，匹配 null 不出现同样效应'},
    {'id': 'generation', 'label': '自然生成', 'auto': False, 'desc': '在自由生成中产生预期结果'},
]

_GATE_LABEL = {g['id']: '最远到「' + g['label'] + '」' for g in GATE_CHAIN}
_CACHE: Dict[Tuple, Any] = {}
_CACHE_LOCK = threading.Lock()
# 重计算（读取并合并分片统计量、规律分析）同一进程内一次只做一个：多个页面请求同时到达时不叠加内存
_HEAVY_LOCK = threading.RLock()


# ---------------- 基础读取 ----------------

def mergeable_kinds() -> Tuple[str, ...]:
    """进入合并统计的结果类型。只有沙盒中心（环境变量 AI2050_DIST_ALLOW_SYNTHETIC=1，独立存储目录）
    才把 synthetic 结果当作可合并，用于方法与客户端全流程验证；其桶名带「合成·」前缀并在客户端醒目标注。"""
    return ('real', 'synthetic') if os.environ.get('AI2050_DIST_ALLOW_SYNTHETIC') == '1' else ('real',)


def _kinds_sql() -> str:
    return '(' + ','.join(f"'{k}'" for k in mergeable_kinds()) + ')'


def parse_blob(blob: bytes, manifest: List[dict]) -> Dict[str, bytes]:
    cursor = 4 + int.from_bytes(blob[:4], 'big')
    rest = blob[cursor:]
    files, off = {}, 0
    for f in manifest:
        size = int(f['size'])
        files[str(f['path'])] = rest[off:off + size]
        off += size
    return files


def paired_behavior(corpus_items: List[dict], behavior_items: List[dict], factor: str) -> Dict[str, Dict[str, int]]:
    """成对行为门（prereg.gate_mode=paired，定义见 grammar._gate_mode_errors）。

    同一划分、同一填充、其余因子水平相同、只在 factor 上不同的两条组成一对；两条的答案集与对照集须互换，
    一对判对当且仅当 m_a + m_b > 0（m = 强制选择间隔）。每一对同时计入两条所在的条件格。
    返回 {条件格: {'n': 对数, 'correct': 判对数, 'invalid': 答案集不互换或缺少间隔而无法配对的对数}}。"""
    meta = {it['id']: it for it in corpus_items}
    groups: Dict[Tuple, Dict[str, List[dict]]] = {}
    for b in behavior_items:
        it = meta.get(b.get('id'))
        if it is None or factor not in (it.get('cond') or {}):
            continue
        cond = it['cond']
        rest = tuple(sorted((k, str(v)) for k, v in cond.items() if k != factor))
        key = (it.get('split'), it.get('filler_id'), rest)
        groups.setdefault(key, {}).setdefault(str(cond[factor]), []).append({**b, '_probe': it.get('probe') or {}})
    out: Dict[str, Dict[str, int]] = {}

    def bump(cell, ok, invalid):
        rec = out.setdefault(cell, {'n': 0, 'correct': 0, 'invalid': 0})
        if invalid:
            rec['invalid'] += 1
        else:
            rec['n'] += 1
            rec['correct'] += int(ok)

    for by_level in groups.values():
        lv = sorted(by_level)
        for i, la in enumerate(lv):
            for lb in lv[i + 1:]:
                for a in by_level[la]:
                    for b in by_level[lb]:
                        pa, pb = a['_probe'], b['_probe']
                        swapped = (set(pa.get('answers') or []) == set(pb.get('contrast') or [])
                                   and set(pa.get('contrast') or []) == set(pb.get('answers') or []) and pa.get('contrast'))
                        ma, mb = a.get('margin'), b.get('margin')
                        invalid = not swapped or not isinstance(ma, (int, float)) or not isinstance(mb, (int, float))
                        ok = (not invalid) and ma + mb > 0
                        bump(a['cell'], ok, invalid)
                        bump(b['cell'], ok, invalid)
    return out


def result_files(data_dir: Path, sha: str, summary: dict, only=None) -> Dict[str, bytes]:
    """从对象文件中按清单取出各文件。only 给定时只读这些文件（按偏移跳读，不把整个对象读进内存）。"""
    p = Path(data_dir) / 'objects' / sha[:2] / sha
    if not p.is_file():
        return {}
    files = {}
    with p.open('rb') as fh:
        off = 4 + int.from_bytes(fh.read(4), 'big')
        for f in summary.get('_manifest') or []:
            size, name = int(f['size']), str(f['path'])
            if only is None or name in only:
                fh.seek(off)
                files[name] = fh.read(size)
            off += size
    return files


def _code_hash() -> str:
    """合并与汇总代码的内容哈希：代码改动后磁盘缓存自动失效。"""
    from . import suffstats as _ss
    h = hashlib.sha256()
    for f in (__file__, _ss.__file__):
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


def _agg_disk_path(data_dir: Path, key: Tuple) -> Path:
    raw = json.dumps([_code_hash(), *key], ensure_ascii=False, default=list)
    return Path(data_dir) / 'agg_cache' / (hashlib.sha256(raw.encode('utf-8')).hexdigest() + '.json')


def _agg_disk_get(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _agg_disk_put(path: Path, entry) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding='utf-8')
        tmp.replace(path)
    except (OSError, TypeError, ValueError):
        pass


def template_corpus(db: sqlite3.Connection, tm_id: str) -> dict:
    row = db.execute('SELECT bundle_dir FROM templates WHERE tm_id=?', (tm_id,)).fetchone()
    if not row:
        return {}
    p = Path(row['bundle_dir']) / 'corpus.json'
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def is_v3(corpus: dict) -> bool:
    return corpus.get('protocol') == 'dist_protocol.v3'


def summary_of(row) -> dict:
    try:
        return json.loads(row['summary_json'] or '{}')
    except (json.JSONDecodeError, TypeError):
        return {}


def precision_of(summary: dict) -> str:
    return str((summary.get('_upload_meta') or {}).get('precision') or summary.get('precision') or '')


def bucket_of(row, summary: dict) -> str:
    key = canary_mod.model_key(row['model_id'] or '', row['model_rev'] or '', precision_of(summary))
    return ('合成·' + key) if row['kind'] == 'synthetic' else key


def audit_status_map(db: sqlite3.Connection) -> Dict[str, dict]:
    try:
        rows = db.execute('SELECT result_sha,status,reviewer,note,created_at FROM audits').fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r['result_sha']: dict(r) for r in rows}


# ---------------- 合并（AGG-v2） ----------------

def _aggregate_bucket(db, data_dir, b, members, corpus, prereg, tpl, audits, gate, full, shards_expected):
    """一个模型桶的精确合并（读取分片统计量、行为门、独立重跑一致性）。"""
    excluded, primaries, replicas = [], {}, []
    for r, s in members:
        reason = None
        if tpl and r['design_sha'] != tpl['design_sha']:
            reason = 'design_sha 与当前模板不符'
        elif (s.get('_canary') or {}).get('status') == 'drift':
            reason = '金丝雀漂移'
        elif (audits.get(r['sha']) or {}).get('status') in ('rejected', 'needs_rerun'):
            reason = '审计未通过'
        if reason:
            excluded.append({'sha': r['sha'], 'reason': reason,
                             'audit_status': (audits.get(r['sha']) or {}).get('status')}); continue
        shard = str(s.get('shard') or '0/1')
        if shard in primaries:
            replicas.append((shard, r, s))
        else:
            primaries[shard] = (r, s)
    merged: Optional[SuffStats] = None
    behavior: Dict[str, Dict[str, int]] = {}
    gate_mode = prereg.get('gate_mode', 'absolute')
    beh_items: List[dict] = []
    loaded: Dict[str, Any] = {}                    # 分片 → 统计量；只为有独立重跑的分片保留（一致性比较用）
    replica_shards = {sh for sh, _, _ in replicas}
    for shard, (r, s) in sorted(primaries.items()):
        files = result_files(data_dir, r['sha'], s, only=('suffstats.npz', 'behavior.json'))
        if 'suffstats.npz' not in files:
            excluded.append({'sha': r['sha'], 'reason': '对象缺失'}); continue
        try:
            raw = files.pop('suffstats.npz')
            ss = SuffStats.from_npz_bytes(raw)
            if shard in replica_shards:            # 合并会改写 merged 本身：保留一份独立副本
                loaded[shard] = ss
                merged = SuffStats.from_npz_bytes(raw) if merged is None else merged.merge_inplace(ss)
            else:
                merged = ss if merged is None else merged.merge_inplace(ss)
                loaded[shard] = None
        except ValueError as exc:
            excluded.append({'sha': r['sha'], 'reason': f'统计量不可合并：{exc}'}); continue
        for cell, rec in (s.get('behavior_by_cell') or {}).items():
            agg = behavior.setdefault(cell, {'n': 0, 'correct': 0})
            agg['n'] += int(rec.get('n', 0)); agg['correct'] += int(rec.get('correct', 0))
        if gate_mode == 'paired' and 'behavior.json' in files:
            try:
                beh_items.extend(json.loads(files['behavior.json'].decode('utf-8')).get('items') or [])
            except (ValueError, UnicodeDecodeError):
                pass
    rep = []
    try:
        byte_reps = db.execute(
            'SELECT sha,node_id,model_id,model_rev,precision,uploaded_at FROM result_replicas WHERE sha IN (%s)'
            % ','.join('?' for _ in primaries), [r['sha'] for r, _ in primaries.values()]).fetchall() if primaries else []
    except sqlite3.OperationalError:
        byte_reps = []
    sha_shard = {r['sha']: sh for sh, (r, _) in primaries.items()}
    for br in byte_reps:
        rep.append({'shard': sha_shard.get(br['sha']), 'sha': br['sha'], 'node_id': br['node_id'],
                    'rel_diff': 0.0, 'byte_identical': True})
    for shard, r, s in replicas:
        if loaded.get(shard) is None:
            continue
        files = result_files(data_dir, r['sha'], s, only=('suffstats.npz',))
        try:
            other = SuffStats.from_npz_bytes(files['suffstats.npz'])
            rep.append({'shard': shard, 'sha': r['sha'], 'node_id': r['node_id'],
                        'rel_diff': replicate_consistency(loaded[shard], other)})
        except (KeyError, ValueError):
            rep.append({'shard': shard, 'sha': r['sha'], 'node_id': r['node_id'], 'rel_diff': None})
    min_acc = float(prereg.get('min_behavior_acc', 0.8))
    min_n = int(prereg.get('min_probe_n', 3))
    beh = {}
    failing = set()
    # 成对口径（prereg.gate_mode=paired）：按句子对判定；逐条绝对准确率仍记录为 absolute_acc
    paired = (paired_behavior(corpus.get('items') or [], beh_items, str(prereg.get('gate_factor')))
              if gate_mode == 'paired' else None)
    for cell, rec in sorted(behavior.items()):
        abs_acc = rec['correct'] / rec['n'] if rec['n'] else None
        if paired is not None:
            pr = paired.get(cell) or {'n': 0, 'correct': 0, 'invalid': 0}
            acc = pr['correct'] / pr['n'] if pr['n'] else None
            ok = acc is not None and pr['n'] >= min_n and acc >= min_acc
            beh[cell] = {'n': rec['n'], 'acc': acc, 'pass': ok, 'mode': 'paired', 'n_pairs': pr['n'],
                         'invalid_pairs': pr['invalid'], 'absolute_acc': abs_acc}
        else:
            ok = abs_acc is not None and rec['n'] >= min_n and abs_acc >= min_acc
            beh[cell] = {'n': rec['n'], 'acc': abs_acc, 'pass': ok}
        if not ok:
            failing.add(cell)
    payload = summarize(merged, exclude_conds=failing if gate else None, include_full=full) if merged else None
    entry = {
        'bucket': b, 'n_results': len(members), 'shards_expected': shards_expected,
        'shards_present': sorted(primaries), 'complete': len(primaries) >= shards_expected,
        'replicates': rep, 'excluded': excluded, 'behavior': beh,
        'primary_shas': [primaries[sh][0]['sha'] for sh in sorted(loaded)],
        'behavior_threshold': min_acc, 'gate_mode': gate_mode, 'gate_applied': gate, 'gate_excluded_cells': sorted(failing) if gate else [],
        'agg': payload,
        'synthetic': any(r['kind'] == 'synthetic' for r, _ in members),
    }
    return entry


def aggregate(db: sqlite3.Connection, data_dir: Path, tm_id: str, *, bucket: Optional[str] = None,
              gate: bool = True, full: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    corpus = template_corpus(db, tm_id)
    prereg = dict(corpus.get('prereg') or {})
    shards_expected = int(corpus.get('shards', 1) or 1)
    rows = db.execute(
        "SELECT sha,model_id,model_rev,kind,design_sha,summary_json,uploaded_at,node_id FROM results "
        f"WHERE tm_id=? AND kind IN {_kinds_sql()} ORDER BY uploaded_at", (tm_id,)).fetchall()
    tpl = db.execute('SELECT design_sha FROM templates WHERE tm_id=?', (tm_id,)).fetchone()
    audits = audit_status_map(db)
    groups: Dict[str, List[Tuple[Any, dict]]] = {}
    for r in rows:
        s = summary_of(r)
        paths = {str(f.get('path')) for f in s.get('_manifest') or []}
        if 'suffstats.npz' not in paths:
            continue
        b = bucket_of(r, s)
        if bucket and b != bucket:
            continue
        groups.setdefault(b, []).append((r, s))
    out_buckets = []
    for b, members in sorted(groups.items()):
        # 缓存键必须含当前模板的 design_sha：模板重新冻结（升版本）后，同一批旧结果要按新 design 重新判定为
        # 「design 不符」而排除；否则会沿用旧版本时「分片已齐」的缓存，跳过新版本的采集
        key = (tm_id, tpl['design_sha'] if tpl else None, b, gate, full, tuple(r['sha'] for r, _ in members),
               tuple((r['sha'], (audits.get(r['sha']) or {}).get('status')) for r, _ in members))
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            out_buckets.append(cached)
            continue
        # 合并结果按键落盘（<data_dir>/agg_cache）：进程重启后不必重新读取、合并全部分片；
        # 同一进程内一次只合并一个桶（多个请求同时到达时不叠加内存）
        disk = _agg_disk_path(data_dir, key)
        entry = _agg_disk_get(disk)
        if entry is None:
            with _HEAVY_LOCK:
                entry = _agg_disk_get(disk)
                if entry is None:
                    entry = _aggregate_bucket(db, data_dir, b, members, corpus, prereg, tpl, audits, gate, full,
                                              shards_expected)
                    _agg_disk_put(disk, entry)
        with _CACHE_LOCK:
            if len(_CACHE) > 8:                        # 合并结果已落盘；内存里只留少量，避免常驻占用
                _CACHE.clear()
            _CACHE[key] = entry
        out_buckets.append(entry)
    return {'tm_id': tm_id, 'agg_version': 'AGG-v2', 'prereg': prereg,
            'family_id': corpus.get('family_id'), 'factors': corpus.get('factors', []),
            'buckets': out_buckets,
            'policy': '只合并 kind=real、design_sha 一致、金丝雀未漂移、审计未驳回的结果；每个分片只计一次，'
                      '重复分片仅用于一致性检查；合并结果是观测统计，不自动成为 Evidence。'}


# ---------------- 证据门链 / 覆盖 / 记分板 ----------------

def gate_chain(agg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按模板（跨桶）判定每道门：pass / fail / partial / pending / not_run。"""
    buckets = [b for b in agg.get('buckets', []) if b.get('agg')]
    thr = (agg.get('prereg') or {}).get('min_lockbox_cos')
    status: Dict[str, Dict[str, Any]] = {}

    def put(gid, st, note):
        status[gid] = {'status': st, 'note': note}

    if not buckets:
        for g in GATE_CHAIN:
            put(g['id'], 'pending' if g['auto'] else 'not_run', '尚无合并统计' if g['auto'] else '需干预/生成实验，当前协议不判定')
        return [{**g, **status[g['id']]} for g in GATE_CHAIN]
    beh_cells = [c for b in buckets for c in b['behavior'].values()]
    if not beh_cells:
        put('behavior', 'pending', '未收到 probe 行为结果')
    elif all(c['pass'] for c in beh_cells):
        put('behavior', 'pass', f'{len(beh_cells)} 个格全部达标')
    else:
        bad = sum(1 for c in beh_cells if not c['pass'])
        put('behavior', 'partial', f'{bad}/{len(beh_cells)} 个格未达标，已从合并中排除')
    drift = [e for b in buckets for e in b['excluded'] if e['reason'] == '金丝雀漂移']
    unresolved = [e for e in drift if e.get('audit_status') not in ('rejected', 'needs_rerun')]
    if unresolved:
        put('instrument', 'fail', f'{len(unresolved)} 份金丝雀漂移结果尚未审计处理（已自动排除出合并）')
    else:
        put('instrument', 'pass', '纳入合并的结果金丝雀一致' + (f'；{len(drift)} 份漂移结果已驳回' if drift else ''))
    reps = [r['rel_diff'] for b in buckets for r in b['replicates'] if r['rel_diff'] is not None]
    if not reps:
        put('repeat', 'pending', '尚无同分片独立重跑')
    else:
        worst = max(reps)
        put('repeat', 'pass' if worst < 2e-2 else 'fail', f'{len(reps)} 次重跑，最大相对差 {worst:.2e}')

    prereg = agg.get('prereg') or {}
    target_factors = prereg.get('target_factors') or agg.get('factors') or []
    target_layers = prereg.get('target_layers')

    def rep_score(b, sp):
        """每个目标因子取目标层（未预注册时取各层最佳）的复现 cos，再取因子间最差者。"""
        layers = [L for L in (b['agg'] or {}).get('layers', []) if not target_layers or L['tag'] in target_layers]
        per = []
        for f in target_factors:
            vals = [((L.get('replication', {}).get(sp) or {}).get(f) or {}).get('mean_cos') for L in layers]
            vals = [v for v in vals if v is not None]
            if vals:
                per.append(max(vals))
        return min(per) if per else None

    for gid in ('confirm', 'lockbox', 'lockbox_frame'):
        scores = [s for s in (rep_score(b, gid) for b in buckets) if s is not None]
        if not scores:
            put(gid, 'pending', '该划分尚无足够样本')
        elif thr is None:
            put(gid, 'partial', f'最差因子 cos {min(scores):.3f}；未预注册阈值，不判定通过')
        else:
            put(gid, 'pass' if min(scores) >= float(thr) else 'fail',
                f'最差因子 cos {min(scores):.3f}（阈值 {thr}；每个因子取{"预注册层" if target_layers else "最佳层"}）')
    passing_models = {b['bucket'].split('|')[0] for b in buckets
                      if thr is not None and (rep_score(b, 'lockbox') or -1) >= float(thr)}
    if len(passing_models) >= 2:
        put('cross_model', 'pass', f'{len(passing_models)} 个模型通过未见词表预测')
    else:
        models = {b['bucket'].split('|')[0] for b in buckets}
        put('cross_model', 'pending', f'已有 {len(models)} 个模型；需 ≥2 个模型通过')
    for g in GATE_CHAIN:
        if not g['auto']:
            put(g['id'], 'not_run', '需干预/生成实验，当前分布式协议不判定')
    return [{**g, **status[g['id']]} for g in GATE_CHAIN]


def templates_v3(db: sqlite3.Connection) -> List[Tuple[sqlite3.Row, dict]]:
    out = []
    for t in db.execute('SELECT * FROM templates ORDER BY tm_id').fetchall():
        c = template_corpus(db, t['tm_id'])
        out.append((t, c))
    return out


def coverage(db: sqlite3.Connection, data_dir: Path) -> Dict[str, Any]:
    columns = [
        {'id': 'corpus', 'label': '语料'}, {'id': 'behavior', 'label': '行为门'},
        {'id': 'repr', 'label': '表示统计'}, {'id': 'confirm', 'label': '留出复现'},
        {'id': 'lockbox', 'label': '未见词表'}, {'id': 'cross_model', 'label': '跨模型'},
        {'id': 'audit', 'label': '审计'}, {'id': 'causal', 'label': '干预'},
    ]
    audits = audit_status_map(db)
    rows = []
    for t, corpus in templates_v3(db):
        tm = t['tm_id']
        res = db.execute('SELECT sha,kind,summary_json,model_id FROM results WHERE tm_id=?', (tm,)).fetchall()
        real = [r for r in res if r['kind'] in mergeable_kinds()]
        v3 = is_v3(corpus)
        cells: Dict[str, Dict[str, str]] = {}
        n_items = sum(1 for it in corpus.get('items', []) if not it.get('canary'))
        cells['corpus'] = {'status': 'done' if v3 else 'partial',
                           'note': f'{n_items} 条 · {"生成式 v3" if v3 else "手写 legacy"}'}
        if v3:
            agg = aggregate(db, data_dir, tm)
            chain = {g['id']: g for g in gate_chain(agg)}
            m = {'pass': 'done', 'partial': 'partial', 'fail': 'fail', 'pending': 'empty', 'not_run': 'empty'}
            cells['behavior'] = {'status': m[chain['behavior']['status']], 'note': chain['behavior']['note']}
            has = any(b.get('agg') for b in agg['buckets'])
            cells['repr'] = {'status': 'done' if has else ('partial' if real else 'empty'),
                             'note': f'{len(agg["buckets"])} 个模型桶' if has else f'{len(real)} 份 real 结果'}
            for gid in ('confirm', 'lockbox', 'cross_model'):
                cells[gid] = {'status': m[chain[gid]['status']], 'note': chain[gid]['note']}
        else:
            legacy_repr = any('means.npz' in {str(f.get('path')) for f in summary_of(r).get('_manifest') or []} for r in real)
            cells['behavior'] = {'status': 'na', 'note': 'legacy 模板无 probe'}
            cells['repr'] = {'status': 'done' if legacy_repr else ('partial' if real else 'empty'),
                             'note': f'{len(real)} 份 real 结果（单节点摘要，不可精确合并）'}
            for gid in ('confirm', 'lockbox'):
                cells[gid] = {'status': 'na', 'note': 'legacy 语料无划分'}
            models = {r['model_id'] for r in real if r['model_id']}
            cells['cross_model'] = {'status': 'partial' if len(models) >= 2 else 'empty', 'note': f'{len(models)} 个模型'}
        audited = [audits.get(r['sha'], {}).get('status') for r in real]
        n_pass = sum(1 for a in audited if a == 'passed')
        n_pending = sum(1 for a in audited if not a)
        cells['audit'] = {'status': 'done' if real and n_pending == 0 else ('partial' if n_pass else 'empty'),
                          'note': f'{n_pass} 通过 · {n_pending} 待审' if real else '无结果'}
        cells['causal'] = {'status': 'empty', 'note': '干预实验未接入分布式协议'}
        rows.append({'tm_id': tm, 'name': t['name'], 'version': t['version'],
                     'family_id': corpus.get('family_id'), 'module': corpus.get('module', ''),
                     'protocol': 'v3' if v3 else 'legacy', 'mode': corpus.get('mode', 'final' if v3 else ''),
                     'cells': cells})
    return {'schema': 'coverage.v1', 'columns': columns, 'rows': rows,
            'note': '每格独立计算自结果库与审计表；done/partial/fail/empty 只描述数据与门状态，不是证据等级。'}


def scoreboard(db: sqlite3.Connection, data_dir: Path) -> Dict[str, Any]:
    rows = []
    for t, corpus in templates_v3(db):
        if not is_v3(corpus):
            continue
        agg = aggregate(db, data_dir, t['tm_id'])
        chain = gate_chain(agg)
        reached = None
        for g in chain:
            if g['status'] == 'pass':
                reached = g['id']
            else:
                break
        blocking = next((g for g in chain if g['status'] != 'pass'), None)
        for b in agg['buckets']:
            if not b.get('agg'):
                continue
            for r in scoreboard_rows(b['agg'], agg['prereg']):
                rows.append({'tm_id': t['tm_id'], 'family_id': corpus.get('family_id'), 'bucket': b['bucket'], **r})
        rows_gate = {'tm_id': t['tm_id'], 'family_id': corpus.get('family_id'), 'name': t['name'],
                     'module': corpus.get('module', ''),
                     'language_graph_target_ids': corpus.get('language_graph_target_ids', []),
                     'gates': chain, 'reached': reached,
                     'blocking': {'id': blocking['id'], 'label': blocking['label'], 'note': blocking['note']} if blocking else None}
        rows.append({'_gate_row': rows_gate})
    gate_rows = [r['_gate_row'] for r in rows if '_gate_row' in r]
    score_rows = [r for r in rows if '_gate_row' not in r]
    return {'schema': 'scoreboard.v1', 'gate_chain': GATE_CHAIN, 'templates': gate_rows, 'scores': score_rows,
            'note': '分数 = discovery 因子水平方向与留出划分同一水平方向的平均 cos（同模型桶内）；'
                    '阈值来自模板 prereg.min_lockbox_cos，冻结后不可调整。'}


# ---------------- 审计 ----------------

def audit_checks(db: sqlite3.Connection, row, summary: dict) -> List[Dict[str, Any]]:
    checks = []

    def add(cid, label, st, detail):
        checks.append({'id': cid, 'label': label, 'status': st, 'detail': detail})

    tpl = db.execute('SELECT design_sha FROM templates WHERE tm_id=?', (row['tm_id'],)).fetchone()
    add('design', '冻结设计一致', 'pass' if tpl and tpl['design_sha'] == row['design_sha'] else 'fail',
        'design_sha 与模板一致' if tpl and tpl['design_sha'] == row['design_sha'] else '与当前模板 design_sha 不符')
    if row['kind'] == 'synthetic' and 'synthetic' in mergeable_kinds():
        add('kind', '真实运行', 'warn', '合成数据（沙盒中心，仅用于方法与流程验证，不进入研究记录）')
    else:
        add('kind', '真实运行', 'pass' if row['kind'] == 'real' else 'fail',
            '真实模型前向' if row['kind'] == 'real' else f'kind={row["kind"]}（smoke/合成结果不进入研究统计）')
    corpus = template_corpus(db, row['tm_id'])
    paths = {str(f.get('path')) for f in summary.get('_manifest') or []}
    if is_v3(corpus):
        missing = [p for p in V3_ARTIFACTS if p not in paths]
        add('artifacts', '协议产物齐全', 'fail' if missing else 'pass', '缺少 ' + ', '.join(missing) if missing else '4 个 v3 产物齐全')
        can = summary.get('_canary') or {}
        st = can.get('status')
        add('canary', '金丝雀一致', {'ok': 'pass', 'reference': 'pass', 'drift': 'fail'}.get(st, 'warn'),
            can.get('detail') or '未评估')
        beh = summary.get('behavior_by_cell') or {}
        thr = float((corpus.get('prereg') or {}).get('min_behavior_acc', 0.8))
        if beh:
            bad = [c for c, r in beh.items() if r.get('n') and r.get('correct', 0) / r['n'] < thr]
            paired_note = '；模板为成对口径，合并时按句子对重新判定' if (corpus.get('prereg') or {}).get('gate_mode') == 'paired' else ''
            add('behavior', '行为门（本分片）', 'warn' if bad else 'pass',
                (f'{len(bad)}/{len(beh)} 格逐条准确率低于 {thr:.0%}（合并时自动排除）' if bad else f'{len(beh)} 格均 ≥ {thr:.0%}') + paired_note)
        else:
            add('behavior', '行为门（本分片）', 'warn', '本分片无 probe 结果')
        shard = str(summary.get('shard') or '')
        expected = int(corpus.get('shards', 1) or 1)
        ok = shard.endswith('/' + str(expected))
        add('shard', '分片标识', 'pass' if ok else 'fail', f'shard={shard or "缺失"}，模板 shards={expected}')
        if summary.get('corpus_fingerprint'):
            add('fingerprint', '语料指纹', 'pass', summary['corpus_fingerprint'])
    else:
        add('artifacts', '协议产物', 'warn', 'legacy 模板：只有单节点摘要，无法精确合并或金丝雀校验')
    wu = summary.get('_research_work_unit') or {}
    add('work_unit', 'Research OS 绑定', 'pass' if wu.get('work_unit_id') else 'warn',
        wu.get('work_unit_id') or '未绑定 Work Unit（平台任务）')
    return checks


def audit_queue(db: sqlite3.Connection, *, status: str = 'pending', limit: int = 100) -> Dict[str, Any]:
    audits = audit_status_map(db)
    rows = db.execute(
        'SELECT r.sha,r.tm_id,r.node_id,r.model_id,r.model_rev,r.kind,r.design_sha,r.size,r.uploaded_at,'
        'r.summary_json,n.name AS node_name FROM results r LEFT JOIN nodes n ON n.node_id=r.node_id '
        'ORDER BY r.uploaded_at DESC LIMIT 1000').fetchall()
    items, counts = [], {'pending': 0, 'passed': 0, 'rejected': 0, 'needs_rerun': 0}
    for r in rows:
        a = audits.get(r['sha'])
        st = a['status'] if a else 'pending'
        counts[st] = counts.get(st, 0) + 1
        if status != 'all' and st != status:
            continue
        if len(items) >= limit:
            continue
        s = summary_of(r)
        checks = audit_checks(db, r, s)
        items.append({
            'sha': r['sha'], 'tm_id': r['tm_id'], 'node_id': r['node_id'], 'node_name': r['node_name'] or '',
            'model_id': r['model_id'], 'model_rev': r['model_rev'], 'precision': precision_of(s),
            'kind': r['kind'], 'size': r['size'], 'uploaded_at': r['uploaded_at'],
            'shard': s.get('shard'), 'status': st, 'audit': a,
            'checks': checks,
            'auto_verdict': 'fail' if any(c['status'] == 'fail' for c in checks)
                            else ('warn' if any(c['status'] == 'warn' for c in checks) else 'pass'),
        })
    return {'schema': 'audit_queue.v1', 'counts': counts, 'items': items,
            'policy': '审计结论只决定该执行产物能否进入合并统计；Evidence 等级仍在 Research OS 登记。'}


def record_audit(db: sqlite3.Connection, lock, sha: str, status: str, reviewer: str, note: str,
                 checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if status not in ('passed', 'rejected', 'needs_rerun'):
        raise ValueError('status must be passed | rejected | needs_rerun')
    now = time.time()
    with lock:
        db.execute('INSERT OR REPLACE INTO audits(result_sha,status,reviewer,note,checks_json,created_at) '
                   'VALUES(?,?,?,?,?,?)', (sha, status, reviewer, note, json.dumps(checks, ensure_ascii=False), now))
        db.commit()
    with _CACHE_LOCK:
        _CACHE.clear()
    return {'result_sha': sha, 'status': status, 'reviewer': reviewer, 'note': note, 'created_at': now}


# ---------------- 总览漏斗 ----------------

def overview(db: sqlite3.Connection, data_dir: Path, *, node_online_window: float = 600.0) -> Dict[str, Any]:
    now = time.time()
    tpls = templates_v3(db)
    v3 = [(t, c) for t, c in tpls if is_v3(c)]
    items_total = sum(sum(1 for it in c.get('items', []) if not it.get('canary')) for _, c in tpls)
    items_v3 = sum(sum(1 for it in c.get('items', []) if not it.get('canary')) for _, c in v3)
    tasks = {r['status']: r['n'] for r in db.execute('SELECT status,COUNT(*) AS n FROM tasks GROUP BY status')}
    res = {r['kind']: r['n'] for r in db.execute('SELECT kind,COUNT(*) AS n FROM results GROUP BY kind')}
    audits = audit_status_map(db)
    real_shas = [r['sha'] for r in db.execute(f"SELECT sha FROM results WHERE kind IN {_kinds_sql()}")]
    pending_audit = sum(1 for s in real_shas if s not in audits)
    drift = 0
    for r in db.execute(f"SELECT summary_json FROM results WHERE kind IN {_kinds_sql()}"):
        if (summary_of(r).get('_canary') or {}).get('status') == 'drift':
            drift += 1
    beh_cells = beh_pass = merged_cells = thin_cells = 0
    gates_reached: Dict[str, int] = {}
    for t, c in v3:
        agg = aggregate(db, data_dir, t['tm_id'])
        min_n = int((c.get('prereg') or {}).get('min_cell_n', 8))
        for b in agg['buckets']:
            beh_cells += len(b['behavior'])
            beh_pass += sum(1 for x in b['behavior'].values() if x['pass'])
            disc = ((b.get('agg') or {}).get('cells_by_split') or {}).get('discovery') or {}
            merged_cells += int(disc.get('cells', 0))
            if disc and int(disc.get('min_cell_n', 0)) < min_n:
                thin_cells += 1
        chain = gate_chain(agg)
        last = None
        for g in chain:
            if g['status'] == 'pass':
                last = g['id']
            else:
                break
        gates_reached[last or 'none'] = gates_reached.get(last or 'none', 0) + 1
    stale = db.execute("SELECT COUNT(*) AS n FROM tasks WHERE status='claimed' AND lease_until<?", (now,)).fetchone()['n']
    nodes_online = db.execute('SELECT COUNT(*) AS n FROM nodes WHERE last_seen>=?', (now - node_online_window,)).fetchone()['n']
    nodes_total = db.execute('SELECT COUNT(*) AS n FROM nodes').fetchone()['n']
    pilots = sum(1 for _, c in v3 if c.get('mode') == 'pilot')
    stages = [
        {'id': 'templates', 'label': '模板族', 'value': len(tpls), 'detail': f'{len(v3)} 个生成式 v3（含 {pilots} 个试跑）· {len(tpls) - len(v3)} 个 legacy'},
        {'id': 'items', 'label': '生成样本', 'value': items_total, 'detail': f'v3 {items_v3} 条；不含金丝雀'},
        {'id': 'behavior', 'label': '行为门', 'value': f'{beh_pass}/{beh_cells}' if beh_cells else '—', 'detail': '达标条件格 / 有 probe 的条件格（合并后）'},
        {'id': 'work_units', 'label': '任务', 'value': sum(tasks.values()),
         'detail': ' · '.join(f'{k} {v}' for k, v in sorted(tasks.items())) or '无任务'},
        {'id': 'merged', 'label': '已合并条件格', 'value': merged_cells, 'detail': f'discovery 划分；{thin_cells} 个桶有格样本不足 min_cell_n'},
        {'id': 'audit', 'label': '审计', 'value': f'{len(real_shas) - pending_audit}/{len(real_shas)}', 'detail': f'{pending_audit} 份{"可合并（real + 合成）" if "synthetic" in mergeable_kinds() else " real "}结果待审'},
        {'id': 'gates', 'label': '证据门', 'value': sum(v for k, v in gates_reached.items() if k != 'none'),
         'detail': '；'.join(f'{_GATE_LABEL.get(k, "未过任何门")} {v} 个模板' for k, v in sorted(gates_reached.items())) or '无 v3 模板'},
    ]
    health = [
        {'id': 'canary', 'label': '金丝雀漂移', 'value': drift, 'level': 'bad' if drift else 'ok'},
        {'id': 'stale', 'label': '过期租约', 'value': stale, 'level': 'warn' if stale else 'ok'},
        {'id': 'nodes', 'label': '在线节点', 'value': f'{nodes_online}/{nodes_total}', 'level': 'ok' if nodes_online else 'warn'},
        {'id': 'smoke', 'label': 'smoke 结果', 'value': res.get('smoke', 0), 'level': 'info'},
    ]
    decisions = []
    if pending_audit:
        decisions.append({'id': 'audit', 'label': f'{pending_audit} 份结果待审计', 'go': 'process', 'tab': 'audit'})
    for t, c in v3:
        if c.get('mode') == 'pilot':
            decisions.append({'id': f'pilot-{t["tm_id"]}', 'label': f'试跑 {t["tm_id"]}：查看行为门后决定是否冻结正式模板',
                              'go': 'process', 'tab': 'freeze'})
    if stale:
        decisions.append({'id': 'stale', 'label': f'{stale} 个租约已过期，需要重新派发', 'go': 'process', 'tab': 'dispatch'})
    if drift:
        decisions.append({'id': 'drift', 'label': f'{drift} 份结果金丝雀漂移，检查节点环境', 'go': 'process', 'tab': 'audit'})
    return {'schema': 'pipeline_overview.v1', 'server_time': now, 'stages': stages, 'health': health,
            'sandbox_synthetic': 'synthetic' in mergeable_kinds(),
            'decisions': decisions,
            'note': '各阶段计数来自不同数据表，彼此独立，不构成转化率。'}


# ---------------- 模式族规律分析（analysis.v1） ----------------

_ANALYSIS_RUNNING: Dict[str, float] = {}


def _analysis_code_hash() -> str:
    """analysis.py 的内容哈希：分析方法改动后，磁盘缓存自动失效。"""
    return hashlib.sha256(Path(analysis_mod.__file__).read_bytes()).hexdigest()[:16]


def _analysis_disk_path(data_dir: Path, key: Tuple) -> Path:
    raw = json.dumps([_analysis_code_hash(), *[list(k) if isinstance(k, tuple) else k for k in key]], ensure_ascii=False)
    return Path(data_dir) / 'analysis_cache' / (hashlib.sha256(raw.encode('utf-8')).hexdigest() + '.json')


def analyze_bucket(db: sqlite3.Connection, data_dir: Path, tm_id: str, *, bucket: Optional[str] = None,
                   gate: bool = True, n_perm: Optional[int] = None, background: bool = False,
                   as_path: bool = False):
    """在一个模型桶的已合并分片上运行 analysis.v1（逐条向量 items.npz）。没有逐条向量时返回 available=False。
    结果写入 <data_dir>/analysis_cache/（键含分片 sha、门控与 analysis.py 内容哈希），结果不可变，所以只算一次。
    background=True 时，未命中缓存就在后台线程计算并立即返回 pending（大模板首次可能要十几分钟）。
    as_path=True 且已有缓存时直接返回缓存文件路径（网站进程原样发送文件，不解析几十 MB 的 JSON）。
    结果只存磁盘、不在内存里常驻（单个结果解析后可达数百 MB）。"""
    agg = aggregate(db, data_dir, tm_id, bucket=bucket, gate=gate)
    buckets = [b for b in agg['buckets'] if b.get('agg')]
    if not buckets:
        return {'tm_id': tm_id, 'available': False, 'reason': '该模板还没有可合并的 real 结果', 'buckets': []}
    chosen = next((b for b in buckets if b['bucket'] == bucket), None) if bucket else \
        next((b for b in buckets if b['complete']), buckets[0])
    if chosen is None:
        return {'tm_id': tm_id, 'available': False, 'reason': f'模型桶 {bucket} 没有可合并结果',
                'buckets': [b['bucket'] for b in buckets]}
    key = (tm_id, chosen['bucket'], gate, tuple(chosen['primary_shas']), tuple(chosen['gate_excluded_cells']), n_perm)
    disk = _analysis_disk_path(data_dir, key)
    if disk.is_file():
        if as_path:
            return disk
        try:
            return json.loads(disk.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    base = {'tm_id': tm_id, 'bucket': chosen['bucket'], 'buckets': [b['bucket'] for b in buckets],
            'shards_present': chosen['shards_present'], 'shards_expected': chosen['shards_expected'],
            'complete': chosen['complete'], 'gate_applied': gate, 'gate_excluded_cells': chosen['gate_excluded_cells']}
    if background:
        with _CACHE_LOCK:
            started = _ANALYSIS_RUNNING.get(disk.name)

        def job():
            conn = sqlite3.connect(str(Path(data_dir) / 'dist.sqlite3'), check_same_thread=False, timeout=60)
            conn.row_factory = sqlite3.Row
            try:
                analyze_bucket(conn, data_dir, tm_id, bucket=chosen['bucket'], gate=gate, n_perm=n_perm)
            finally:
                conn.close()
        # 门控结果由后台计算子进程（warm_analyses）预热；只有不开门控的少数请求在本进程里补算
        if started is None and not gate:
            threading.Thread(target=job, name=f'analysis-{tm_id}', daemon=True).start()
        return {**base, 'available': False, 'pending': True, 'started_at': started or time.time(),
                'reason': '规律分析正在后台计算（大模板首次可能需要十几分钟），完成后自动显示；结果会缓存，之后打开即时显示。'}
    with _CACHE_LOCK:
        if disk.name in _ANALYSIS_RUNNING:          # 另一线程正在算同一份：不重复计算
            return {**base, 'available': False, 'pending': True, 'started_at': _ANALYSIS_RUNNING[disk.name],
                    'reason': '规律分析正在后台计算，完成后自动显示。'}
        _ANALYSIS_RUNNING[disk.name] = time.time()
    try:
        blobs = []
        for sha in chosen['primary_shas']:
            row = db.execute('SELECT summary_json FROM results WHERE sha=?', (sha,)).fetchone()
            files = result_files(data_dir, sha, summary_of(row), only=('items.npz',)) if row else {}
            if 'items.npz' in files:
                blobs.append(files['items.npz'])
        if not blobs:
            out = {**base, 'available': False, 'reason': '结果中没有逐条向量（items.npz）；需用 runner_v3 且 capture_items 未关闭'}
        else:
            corpus = template_corpus(db, tm_id)
            texts = {it['id']: it['text'] for it in corpus.get('items', []) if not it.get('canary')}
            try:
                items = analysis_mod.load_items(blobs)
                result = analysis_mod.analyze(items, list(corpus.get('factors') or []), texts=texts,
                                              exclude_cells=set(chosen['gate_excluded_cells']) if gate else None,
                                              n_perm=n_perm)
                out = {**base, 'available': True, 'items_from_shards': len(blobs), **result}
            except ValueError as exc:
                out = {**base, 'available': False, 'reason': f'逐条向量不可合并：{exc}'}
        out['policy'] = ('只用合并口径内的分片（kind=real、design_sha 一致、金丝雀未漂移、审计未驳回，每分片一次）；'
                         '判据固定写在 criteria 中；规律陈述是观测层面的候选，不自动成为 Evidence。')
        try:
            disk.parent.mkdir(parents=True, exist_ok=True)
            tmp = disk.with_suffix('.tmp')
            tmp.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
            tmp.replace(disk)
        except OSError:
            pass
        return out
    finally:
        with _CACHE_LOCK:
            _ANALYSIS_RUNNING.pop(disk.name, None)


def warm_analyses(db: sqlite3.Connection, data_dir: Path, skip=None, on_start=None) -> List[str]:
    """后台预热：为每个 v3 模板的每个有合并结果的模型桶算好规律分析（已缓存的跳过）。返回本次新算的。
    skip(name) 为真的步骤不算（后台进程记录的崩溃步骤）；on_start(name) 在每一步开始前调用。name = 'analysis:<tm>/<bucket>'。"""
    done = []
    for t, corpus in templates_v3(db):
        if not is_v3(corpus):
            continue
        agg = aggregate(db, data_dir, t['tm_id'])
        for b in agg['buckets']:
            if not b.get('agg'):
                continue
            key = (t['tm_id'], b['bucket'], True, tuple(b['primary_shas']), tuple(b['gate_excluded_cells']), None)
            if _analysis_disk_path(data_dir, key).is_file():
                continue
            name = f"analysis:{t['tm_id']}/{b['bucket']}"
            if skip and skip(name):
                continue
            if on_start:
                on_start(name)
            out = analyze_bucket(db, data_dir, t['tm_id'], bucket=b['bucket'], gate=True)
            if not out.get('pending'):
                done.append(f"{t['tm_id']}/{b['bucket']}")
    return done


# ---------------- 题库（catalog.v1）：所有用户看到的题目、类别与测试状态 ----------------

MODULE_LABEL = {'KN': '知识网络', 'GR': '语法系统', 'RS': '分析推理', 'SQ': '序列与多维', 'ST': '风格与语用',
                'TK': '任务功能'}
STATUS_LABEL = {'draft': '未冻结', 'pilot': '试跑中', 'open': '待测试', 'running': '测试中', 'done': '已完成'}


def _merge_shas(db: sqlite3.Connection, data_dir: Path, shas: List[str]) -> Optional[SuffStats]:
    merged = None
    for sha in shas:
        row = db.execute('SELECT summary_json FROM results WHERE sha=?', (sha,)).fetchone()
        files = result_files(data_dir, sha, summary_of(row), only=('suffstats.npz',)) if row else {}
        if 'suffstats.npz' in files:
            ss = SuffStats.from_npz_bytes(files['suffstats.npz'])
            merged = ss if merged is None else merged.merge_inplace(ss)
    return merged


def catalog(db: sqlite3.Connection, data_dir: Path, library: List[dict], *, node_online_window: float = 600.0,
            center_hint: str = '<center>') -> Dict[str, Any]:
    """题库：文法库中的每个模式族 + 已注册的模板，各自的类别、冻结版本、按模型的测试状态与可执行动作。"""
    now = time.time()
    audits = audit_status_map(db)
    tasks = {}
    for r in db.execute("SELECT tm_id,status,COUNT(*) AS n FROM tasks WHERE status IN ('claimed','queued') "
                        "AND (lease_until IS NULL OR lease_until>=? OR status='queued') GROUP BY tm_id,status", (now,)):
        tasks.setdefault(r['tm_id'], {})[r['status']] = r['n']
    tpls = templates_v3(db)
    by_family: Dict[str, Dict[str, Any]] = {}
    for t, c in tpls:
        if is_v3(c):
            slot = by_family.setdefault(c.get('family_id') or t['tm_id'], {})
            slot['pilot' if c.get('mode') == 'pilot' else 'final'] = (t, c)
    lib = {s.get('family_id'): s for s in library if isinstance(s, dict) and s.get('family_id')}
    items = []
    for fam in sorted(set(lib) | set(by_family)):
        spec = lib.get(fam) or {}
        slot = by_family.get(fam, {})
        final, pilot = slot.get('final'), slot.get('pilot')
        base_c = (final or pilot or (None, {}))[1] or {}
        module = spec.get('module') or base_c.get('module') or ''
        entry = {
            'id': fam, 'protocol': 'v3', 'name': spec.get('name') or base_c.get('name') or fam,
            'description': spec.get('description') or base_c.get('description') or '',
            'module': module, 'category': MODULE_LABEL.get(module, module or '未分类'),
            'factors': list(spec.get('factors') or base_c.get('factors') or []),
            'concepts': dict(spec.get('concepts') or base_c.get('concepts') or {}),
            'in_library': fam in lib, 'tm_id': final[0]['tm_id'] if final else None,
            'version': final[0]['version'] if final else None,
            'design_sha': final[0]['design_sha'] if final else None,
            'pilot_tm_id': pilot[0]['tm_id'] if pilot else None,
            'n_items': sum(1 for it in (final[1] if final else {}).get('items', []) if not it.get('canary')) if final else None,
            'shards': int((final[1] if final else {}).get('shards', 1) or 1) if final else None,
            'models': [], 'tasks_active': 0, 'audit_pending': 0, 'last_result_at': None,
        }
        if not final:
            entry['status'] = 'pilot' if pilot else 'draft'
        else:
            tm = final[0]['tm_id']
            agg = aggregate(db, data_dir, tm)
            for b in agg['buckets']:
                beh = list((b.get('behavior') or {}).values())
                entry['models'].append({
                    'bucket': b['bucket'], 'synthetic': b.get('synthetic', False), 'complete': b['complete'],
                    'shards_present': len(b['shards_present']), 'shards_expected': b['shards_expected'],
                    'behavior_pass': sum(1 for x in beh if x['pass']), 'behavior_cells': len(beh),
                    'excluded': len(b['excluded']), 'has_agg': bool(b.get('agg')),
                    'download': f'/api/export/{tm}?bucket={b["bucket"]}',
                })
            res = db.execute(f"SELECT sha,uploaded_at FROM results WHERE tm_id=? AND kind IN {_kinds_sql()}", (tm,)).fetchall()
            entry['audit_pending'] = sum(1 for r in res if not (audits.get(r['sha']) or {}).get('status'))
            entry['last_result_at'] = max((r['uploaded_at'] for r in res), default=None)
            entry['tasks_active'] = sum(tasks.get(tm, {}).values())
            if any(m['complete'] and m['has_agg'] for m in entry['models']):
                entry['status'] = 'done'
            elif entry['models'] or entry['tasks_active']:
                entry['status'] = 'running'
            else:
                entry['status'] = 'open'
            entry['claim_command'] = (f'python deploy/node_agent.py --server {center_hint} run --once --tm-id {tm} '
                                      f'--model-path <本机HF模型目录> --model-id <model_id> --dtype bf16')
        entry['status_label'] = STATUS_LABEL[entry['status']]
        items.append(entry)
    for t, c in tpls:                                   # legacy 手写模板
        if is_v3(c):
            continue
        res = db.execute(f"SELECT model_id,uploaded_at FROM results WHERE tm_id=? AND kind IN {_kinds_sql()}",
                         (t['tm_id'],)).fetchall()
        items.append({'id': t['tm_id'], 'protocol': 'legacy', 'name': t['name'], 'description': t['description'] or '',
                      'module': 'legacy', 'category': '早期模板（legacy）', 'factors': [], 'concepts': {},
                      'in_library': False, 'tm_id': t['tm_id'], 'version': t['version'], 'design_sha': t['design_sha'],
                      'models': [{'bucket': m, 'complete': True} for m in sorted({r['model_id'] for r in res if r['model_id']})],
                      'status': 'done' if res else 'open', 'status_label': STATUS_LABEL['done' if res else 'open'],
                      'last_result_at': max((r['uploaded_at'] for r in res), default=None), 'tasks_active': 0,
                      'audit_pending': 0, 'n_items': sum(1 for it in c.get('items', []) if not it.get('canary'))})
    cats: Dict[str, Dict[str, Any]] = {}
    for it in items:
        cat = cats.setdefault(it['category'], {'category': it['category'], 'module': it['module'], 'total': 0,
                                               **{k: 0 for k in STATUS_LABEL}})
        cat['total'] += 1
        cat[it['status']] += 1
    counts = {}
    for r in db.execute(f"SELECT node_id,COUNT(*) AS n,MAX(uploaded_at) AS last FROM results "
                        f"WHERE kind IN {_kinds_sql()} GROUP BY node_id"):
        counts[r['node_id']] = (r['n'], r['last'])
    contributors = []
    try:   # 节点归属账号（accounts.v1）：贡献者署名；旧库没有 users 表时退化为只列节点
        nodes = db.execute('SELECT n.node_id,n.name,n.gpu,n.model,n.dtype,n.last_seen,u.display_name AS owner '
                           'FROM nodes n LEFT JOIN users u ON u.user_id=n.owner_user_id ORDER BY n.last_seen DESC').fetchall()
    except sqlite3.OperationalError:
        nodes = db.execute('SELECT node_id,name,gpu,model,dtype,last_seen,NULL AS owner FROM nodes ORDER BY last_seen DESC').fetchall()
    for n in nodes:
        c, last = counts.get(n['node_id'], (0, None))
        contributors.append({'node_id': n['node_id'], 'name': n['name'], 'gpu': n['gpu'], 'model': n['model'],
                             'dtype': n['dtype'], 'results': c, 'last_result_at': last, 'owner': n['owner'],
                             'online': bool(n['last_seen'] and n['last_seen'] >= now - node_online_window)})
    return {'schema': 'catalog.v1', 'server_time': now, 'categories': sorted(cats.values(), key=lambda c: c['category']),
            'items': items, 'contributors': contributors, 'status_labels': STATUS_LABEL,
            'sandbox_synthetic': 'synthetic' in mergeable_kinds(),
            'note': '状态只描述数据：已完成 = 至少一个模型桶的分片齐全并可合并；它不是证据等级。'}


# ---------------- 研究包（respack.v1）：下载到本地观察与复算 ----------------

def _ascii_name(text: str) -> str:
    """下载文件名只用 ASCII（含中文的文件名会被部分浏览器忽略）；合成模型桶前缀记为 SYN_。"""
    import re
    text = str(text).replace('合成·', 'SYN_')
    return re.sub(r'_+', '_', re.sub(r'[^A-Za-z0-9._-]', '_', text)).strip('_') or 'pack'


def _zip(files: Dict[str, bytes], manifest: Dict[str, Any]) -> bytes:
    import hashlib
    import io
    import zipfile
    manifest = dict(manifest)
    manifest['files'] = [{'path': p, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for p, b in sorted(files.items())]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=1))
        for p, b in sorted(files.items()):
            z.writestr(p, b)
    return buf.getvalue()


RESPACK_README = """# AI2050 研究包（respack.v1）

一个模式族（模板）在一个模型桶上的合并结果，可离线观察与复算。

- `manifest.json`：模板、版本、design_sha、模型桶、生成时间、每个文件的 sha256。
- `corpus.json`：冻结语料（含划分、条件、行为 probe）。
- `agg.json`：AGG-v2 合并统计（η² 剖面、因子方向复现、条件结构、行为门）。
- `analysis.json`：analysis.v1 规律分析，含 3D 坐标（coords：层 → LDA/PCA 坐标）、层堆叠、规律陈述与判据。
- `suffstats.npz`：合并后的充分统计量（suffstats.v1：每个 划分×条件格×层×维 的 n、Σx、Σx²）。
- `items/<分片>.npz`：逐条向量（items.v1，可选）。

复算：
```python
import zipfile, io
from shared.dist_protocol import analysis
z = zipfile.ZipFile('研究包.zip')
items = analysis.load_items([z.read(n) for n in z.namelist() if n.startswith('items/')])
```
客户端：空间 → 实验响应 → 「打开本地研究包」载入 analysis.json，在 3D 响应空间中观察（不需要连接服务器）。
合成数据（模型桶带「合成·」前缀）只用于方法与流程验证，不代表任何真实模型。
"""


def export_pack(db: sqlite3.Connection, data_dir: Path, tm_id: str, *, bucket: Optional[str] = None,
                gate: bool = True, include_items: bool = True) -> Tuple[bytes, str]:
    corpus = template_corpus(db, tm_id)
    if not corpus:
        raise KeyError(tm_id)
    agg = aggregate(db, data_dir, tm_id, bucket=bucket, gate=gate)
    cands = [b for b in agg['buckets'] if b.get('agg')]
    chosen = next((b for b in cands if b['bucket'] == bucket), None) if bucket else \
        next((b for b in cands if b['complete']), cands[0] if cands else None)
    if chosen is None:
        raise LookupError('该模板没有可合并结果' if not bucket else f'模型桶 {bucket} 没有可合并结果')
    tpl = db.execute('SELECT version,design_sha FROM templates WHERE tm_id=?', (tm_id,)).fetchone()
    files: Dict[str, bytes] = {
        'README.md': RESPACK_README.encode('utf-8'),
        'corpus.json': json.dumps({k: v for k, v in corpus.items() if not k.startswith('_')}, ensure_ascii=False).encode('utf-8'),
        'agg.json': json.dumps({**{k: v for k, v in agg.items() if k != 'buckets'}, 'bucket': chosen},
                               ensure_ascii=False).encode('utf-8'),
    }
    an = analyze_bucket(db, data_dir, tm_id, bucket=chosen['bucket'], gate=gate)
    files['analysis.json'] = json.dumps(an, ensure_ascii=False).encode('utf-8')
    merged = _merge_shas(db, data_dir, chosen['primary_shas'])
    if merged is not None:
        files['suffstats.npz'] = merged.to_npz_bytes()
    if include_items:
        for sha in chosen['primary_shas']:
            row = db.execute('SELECT summary_json FROM results WHERE sha=?', (sha,)).fetchone()
            s = summary_of(row) if row else {}
            got = result_files(data_dir, sha, s, only=('items.npz',))
            if 'items.npz' in got:
                files[f'items/{str(s.get("shard") or "0/1").replace("/", "of")}.npz'] = got['items.npz']
    manifest = {'schema': 'respack.v1', 'tm_id': tm_id, 'family_id': corpus.get('family_id'),
                'version': tpl['version'] if tpl else None, 'design_sha': tpl['design_sha'] if tpl else None,
                'bucket': chosen['bucket'], 'synthetic': bool(chosen.get('synthetic')), 'complete': chosen['complete'],
                'gate_applied': gate, 'created_at': time.time(), 'source_results': chosen['primary_shas']}
    return _zip(files, manifest), f'respack_{_ascii_name(tm_id)}__{_ascii_name(chosen["bucket"])}.zip'


# ---------------- 语言谱图（atlas.v1） ----------------

_ATLAS_CACHE: Dict[Tuple, Any] = {}


def token_len_by_cell(db: sqlite3.Connection, shas: List[str]) -> Dict[str, float]:
    """各分片 summary.token_len_by_cell 的样本数加权平均：{条件格键: 平均 token 数}（谱图长度混杂标记用）。"""
    tot: Dict[str, List[float]] = {}
    for sha in shas:
        row = db.execute('SELECT summary_json FROM results WHERE sha=?', (sha,)).fetchone()
        if not row:
            continue
        s = json.loads(row[0] or '{}')
        conds = s.get('conditions') or {}
        for cell, v in (s.get('token_len_by_cell') or {}).items():
            m = v.get('mean') if isinstance(v, dict) else None
            if m is None:
                continue
            n = float(conds.get(cell, 1) or 1)
            t = tot.setdefault(cell, [0.0, 0.0])
            t[0] += float(m) * n
            t[1] += n
    return {c: t[0] / t[1] for c, t in tot.items() if t[1] > 0}


def atlas_payload(db: sqlite3.Connection, data_dir: Path, *, gate: bool = True) -> Dict[str, Any]:
    """所有模型桶 × 所有已冻结正式模式族 → atlas.v1。只用分片齐全的桶。"""
    from . import atlas as atlas_mod
    per_model: Dict[str, List[dict]] = {}
    sources, keys, todo = [], [], []
    for t, corpus in templates_v3(db):
        if not is_v3(corpus) or corpus.get('mode') == 'pilot':
            continue
        if (t['status'] or 'open') != 'open':
            # 已关闭的模板（如文法已升版本但新版本未进入正式冻结时的旧版本）不进入谱图
            sources.append({'tm_id': t['tm_id'], 'family_id': corpus.get('family_id'), 'bucket': None,
                            'complete': False, 'used': False, 'reason': f'模板已关闭（status={t["status"]}）'})
            keys.append((t['tm_id'], 'closed'))
            continue
        agg = aggregate(db, data_dir, t['tm_id'], gate=gate)
        for b in agg['buckets']:
            src = {'tm_id': t['tm_id'], 'family_id': corpus.get('family_id'), 'bucket': b['bucket'],
                   'complete': b['complete'], 'used': False}
            if not b.get('agg'):
                src['reason'] = '无可合并统计'
            elif not b['complete']:
                src['reason'] = f'分片未齐（{len(b["shards_present"])}/{b["shards_expected"]}）'
            else:
                src['used'] = True
                todo.append((t, corpus, b))
                keys.append((t['tm_id'], b['bucket'], tuple(b['primary_shas']), tuple(b['gate_excluded_cells'])))
            sources.append(src)
    key = (gate, tuple(keys))
    with _CACHE_LOCK:
        hit = _ATLAS_CACHE.get(key)
    if hit is not None:
        return hit
    for t, corpus, b in todo:
        ss = _merge_shas(db, data_dir, b['primary_shas'])
        if ss is None:
            continue
        fam = atlas_mod.family_features(corpus.get('family_id') or t['tm_id'], t['tm_id'], ss,
                                        list(corpus.get('factors') or []),
                                        exclude_conds=set(b['gate_excluded_cells']) if gate else None,
                                        concepts=corpus.get('concepts') or {},
                                        token_len=token_len_by_cell(db, list(b['primary_shas'])))
        per_model.setdefault(b['bucket'], []).append(fam)
    out = atlas_mod.build_atlas(per_model)
    out['sources'] = sources
    out['gate_applied'] = gate
    with _CACHE_LOCK:
        if len(_ATLAS_CACHE) > 8:
            _ATLAS_CACHE.clear()
        _ATLAS_CACHE[key] = out
    return out


def atlas_pack(payload: Dict[str, Any], model: Optional[str] = None) -> Tuple[bytes, str]:
    from . import atlas as atlas_mod
    sel = dict(payload)
    if model:
        sel['graphs'] = [g for g in payload['graphs'] if g['model'] == model]
        sel['models'] = [g['model'] for g in sel['graphs']]
    files = {'atlas.json': json.dumps(sel, ensure_ascii=False).encode('utf-8'),
             'digest.md': atlas_mod.digest(sel).encode('utf-8'),
             'prompt.md': atlas_mod.ai_prompt(sel).encode('utf-8')}
    manifest = {'schema': 'atlaspack.v1', 'models': sel['models'], 'created_at': time.time(),
                'criteria': payload.get('criteria'), 'gate_applied': payload.get('gate_applied')}
    return _zip(files, manifest), f'atlas_{_ascii_name(model or "all")}.zip'
