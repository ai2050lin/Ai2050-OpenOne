# -*- coding: utf-8 -*-
"""神经元普查分片任务（census_job.v1）：把 mlp_census.v1 拆成「token 切片 × 层组」的分片，由参与者节点执行、中心校验与合并。

一个作业（job）固定：模型快照（权重与 tokenizer 文件的 sha256）、语料指纹、每片 token 数、窗口长度、跳过的窗口开头位置数、
top-k 与层组划分。分片 = (切片 s, 层组 g)，编号 ``S<s>-G<g>``。同一切片的各层组共用一个 token 文件。

中心在建作业时就能从 token 流算出每个切片应计入的 token 数（窗口开头 skip_head 个位置与分隔符不计），节点结果必须逐层精确等于它；
直方图每个单元的总计数必须等于该数；正 / 负极的 token 位置必须落在该切片计入的位置集合里。按比例抽取的分片派给两个不同节点，
比较两份结果（bf16 在不同 GPU 上有小幅数值差异，比较用宽容差）。这些检查只说明执行产物可信、可以合并，不改变证据等级。

合并口径与 census.merge 相同：计数与和逐项相加、极值取极值、top-k 按数值合并，所以分片合并结果与单机一次累计相同。
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from shared.dist_protocol import census as C

SCHEMA = 'census_job.v1'
LAYER_KEYS = ('hist', 'n', 's1', 's2', 'amin', 'amax', 'top_val', 'top_idx', 'top_g', 'top_v', 'bot_val', 'bot_idx', 'bot_g', 'bot_v')
REPLICA_TOL = {'mean_z': 0.05, 'amax_rel': 0.05, 'top1_agree': 0.9, 'hist_l1': 0.02}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def shard_id(s: int, g: int) -> str:
    return f'S{s}-G{g}'


def parse_shard_id(text: str) -> Tuple[int, int]:
    a, b = text.split('-')
    if not (a.startswith('S') and b.startswith('G')):
        raise ValueError(f'bad shard id {text!r}')
    return int(a[1:]), int(b[1:])


def layer_groups(n_layers: int, group_size: int) -> List[List[int]]:
    return [list(range(i, min(i + group_size, n_layers))) for i in range(0, n_layers, group_size)]


def replica_required(job_id: str, sid: str, rate: float) -> int:
    """确定性抽样：分片是否需要第二个独立节点复算（与提交顺序无关，参与者无法预知以外的分片）。"""
    h = int(hashlib.sha256(f'{job_id}/{sid}'.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 2 if h < rate else 1


def slice_windows(streams: List[np.ndarray], s: int, tokens_per_slice: int, seq_len: int) -> Dict[str, np.ndarray]:
    """切片 s：每个来源取 [s·m, (s+1)·m)（m = tokens_per_slice / 来源数，按 seq_len 取整），切成不重叠窗口。"""
    m = (tokens_per_slice // len(streams)) // seq_len * seq_len
    if m <= 0:
        raise ValueError('tokens_per_slice 太小')
    wins, src, off = [], [], []
    for sid, toks in enumerate(streams):
        a, b = s * m, min((s + 1) * m, len(toks) // seq_len * seq_len)
        if b <= a:
            continue
        arr = np.asarray(toks[a:b], dtype=np.int32).reshape(-1, seq_len)
        wins.append(arr)
        src.append(np.full(len(arr), sid, dtype=np.int32))
        off.append(a + seq_len * np.arange(len(arr), dtype=np.int64))
    if not wins:
        return {'windows': np.zeros((0, seq_len), np.int32), 'src': np.zeros(0, np.int32), 'win_off': np.zeros(0, np.int64)}
    return {'windows': np.concatenate(wins), 'src': np.concatenate(src), 'win_off': np.concatenate(off)}


def n_slices(streams: List[np.ndarray], tokens_per_slice: int, seq_len: int) -> int:
    m = (tokens_per_slice // len(streams)) // seq_len * seq_len
    return max(1, max((len(t) // seq_len * seq_len + m - 1) // m for t in streams))


def counted_gidx(sl: Dict[str, np.ndarray], skip_head: int, sep: int) -> np.ndarray:
    """切片中计入统计的全局 token 位置（升序）。与节点 runner 的掩码口径完全一致。"""
    W = sl['windows']
    if len(W) == 0:
        return np.zeros(0, np.int64)
    pos = np.arange(W.shape[1])[None, :]
    mask = (pos >= skip_head) & (W != sep)
    off = sl['win_off'][:, None] + pos
    g = (sl['src'].astype(np.int64)[:, None] << C.SRC_SHIFT) | off
    return np.sort(g[mask])


def npz_bytes(arrays: Dict[str, np.ndarray], compressed: bool = False) -> bytes:
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    return buf.getvalue()


def build_job(job_id: str, *, model: Dict[str, Any], corpus_fp: str, corpus_manifest: Dict[str, Any],
              streams: List[np.ndarray], n_layers: int, d_ff: int, tokens_per_slice: int = 1_000_000,
              max_slices: Optional[int] = None, seq_len: int = 512, skip_head: int = 16, k: int = 32,
              group_size: int = 4, replica_rate: float = 0.1, batch: int = 8) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """生成作业规范与各切片 token 文件。返回 (job, {文件名: 字节})；job['shards'] 带每个分片应计入的 token 数。"""
    sep = int(corpus_manifest['separator_id'])
    total = n_slices(streams, tokens_per_slice, seq_len)
    S = min(total, max_slices) if max_slices else total
    groups = layer_groups(n_layers, group_size)
    files: Dict[str, bytes] = {}
    slices = []
    for s in range(S):
        sl = slice_windows(streams, s, tokens_per_slice, seq_len)
        name = f'tokens_S{s}.npz'
        files[name] = npz_bytes(sl)
        slices.append({'slice': s, 'file': name, 'sha256': sha256_bytes(files[name]), 'n_windows': int(len(sl['windows'])),
                       'expected_n': int(len(counted_gidx(sl, skip_head, sep)))})
    shards = []
    for s in range(S):
        for g, layers in enumerate(groups):
            sid = shard_id(s, g)
            shards.append({'shard': sid, 'slice': s, 'group': g, 'layers': layers, 'expected_n': slices[s]['expected_n'],
                           'readout': s == 0, 'replicas': replica_required(job_id, sid, replica_rate)})
    job = {'schema': SCHEMA, 'job_id': job_id, 'model': model, 'corpus_fp': corpus_fp, 'separator_id': sep,
           'sources': [s['source'] for s in corpus_manifest['sources']],
           'tokens_per_slice': int(tokens_per_slice), 'n_slices': S, 'n_slices_available': total, 'seq_len': int(seq_len),
           'skip_head': int(skip_head), 'k': int(k), 'batch': int(batch), 'n_layers': int(n_layers), 'd_ff': int(d_ff),
           'layer_groups': groups, 'replica_rate': float(replica_rate), 'replica_tol': REPLICA_TOL,
           'slices': slices, 'shards': shards,
           'note': '窗口独立前向，无 padding、无 KV 缓存；每个窗口前 skip_head 个位置与分隔符不计入；A = silu(G)⊙V；'
                   '读出 W_U(γ⊙W_down[:,j]) 只在切片 0 的分片上计算'}
    return job, files


def shard_spec(job: Dict[str, Any], sid: str) -> Dict[str, Any]:
    for sh in job['shards']:
        if sh['shard'] == sid:
            return sh
    raise KeyError(sid)


# ---------- 中心：结果校验、复算比较与合并 ----------

def load_shard_npz(data: bytes) -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, np.ndarray]] = {}
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        for name in z.files:
            head, key = name.split('__', 1)
            out.setdefault(int(head[1:]), {})[key] = z[name]
    return out


def validate_shard(job: Dict[str, Any], sid: str, layers: Dict[int, Dict[str, np.ndarray]],
                   counted: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """逐项检查节点结果；返回 {'ok': bool, 'errors': [...], 'checks': {...}}。counted 为该切片计入的全局位置（升序），给出时检查极值位置。"""
    sh = shard_spec(job, sid)
    F, k, n_exp = job['d_ff'], job['k'], sh['expected_n']
    errors: List[str] = []
    if sorted(layers) != sorted(sh['layers']):
        errors.append(f'层集合不符：{sorted(layers)} ≠ {sh["layers"]}')
    for b, d in sorted(layers.items()):
        miss = [key for key in LAYER_KEYS if key not in d]
        if sh['readout'] and not {'readout_top', 'readout_bot'} <= set(d):
            miss.append('readout_top/readout_bot')
        if miss:
            errors.append(f'L{b} 缺字段 {miss}')
            continue
        shapes = {'hist': (F, C.N_BINS, 4), 's1': (F,), 's2': (F,), 'amin': (F,), 'amax': (F,)}
        shapes.update({f'{p}_{f}': (F, k) for p in ('top', 'bot') for f in ('val', 'idx', 'g', 'v')})
        bad = [key for key, shp in shapes.items() if tuple(d[key].shape) != shp]
        if bad:
            errors.append(f'L{b} 形状不符 {bad}')
            continue
        n = int(d['n'])
        if n != n_exp:
            errors.append(f'L{b} 计入 token 数 {n} ≠ 应为 {n_exp}')
        if not np.all(d['hist'].reshape(F, -1).sum(1) == n):
            errors.append(f'L{b} 直方图总计数与 n 不一致')
        s1, s2 = d['s1'].astype(np.float64), d['s2'].astype(np.float64)
        if n > 0:
            if not np.all(np.isfinite(s1)) or not np.all(np.isfinite(s2)):
                errors.append(f'L{b} 和或平方和含非有限值')
            elif np.any(s2 + 1e-6 * np.abs(s2) + 1e-6 < s1 ** 2 / n):
                errors.append(f'L{b} 平方和小于和的平方 / n')
            if np.any(d['top_val'][:, 0] < d['top_val'][:, -1]) or np.any(d['bot_val'][:, 0] > d['bot_val'][:, -1]):
                errors.append(f'L{b} 极值列表未按数值排序')
            if not np.allclose(d['top_val'][:, 0], d['amax'], rtol=1e-3, atol=1e-3) or \
                    not np.allclose(d['bot_val'][:, 0], d['amin'], rtol=1e-3, atol=1e-3):
                errors.append(f'L{b} 极值列表首项与 max / min 不一致')
            if counted is not None:
                idx = np.concatenate([d['top_idx'].ravel(), d['bot_idx'].ravel()])
                idx = idx[idx >= 0]
                pos = np.searchsorted(counted, idx)
                inside = (pos < len(counted)) & (counted[np.minimum(pos, len(counted) - 1)] == idx)
                if not np.all(inside):
                    errors.append(f'L{b} 有 {int((~inside).sum())} 个极值位置不在该切片计入的 token 中')
    return {'ok': not errors, 'errors': errors[:20], 'checks': {'layers': sorted(layers), 'expected_n': n_exp}}


def compare_replicas(a: Dict[int, Dict[str, np.ndarray]], b: Dict[int, Dict[str, np.ndarray]],
                     tol: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """同一分片两份独立结果的一致性（宽容差：不同 GPU 上 bf16 计算有小幅差异）。"""
    tol = tol or REPLICA_TOL
    rows, ok = [], True
    for L in sorted(set(a) & set(b)):
        x, y = a[L], b[L]
        n = max(int(x['n']), 1)
        mx, my = x['s1'] / n, y['s1'] / n
        sd = np.sqrt(np.maximum(x['s2'] / n - mx ** 2, 1e-12))
        mean_z = float(np.max(np.abs(mx - my) / sd))
        amax_rel = float(np.max(np.abs(x['amax'] - y['amax']) / np.maximum(np.abs(x['amax']), 1e-3)))
        top1 = float(np.mean(x['top_idx'][:, 0] == y['top_idx'][:, 0]))
        hx = x['hist'].reshape(len(mx), -1).astype(np.float64) / n
        hy = y['hist'].reshape(len(mx), -1).astype(np.float64) / n
        hist_l1 = float(np.max(np.abs(hx - hy).sum(1)))
        r = {'layer': int(L), 'n_equal': int(x['n']) == int(y['n']), 'mean_z': mean_z, 'amax_rel': amax_rel,
             'top1_agree': top1, 'hist_l1': hist_l1}
        r['ok'] = (r['n_equal'] and mean_z <= tol['mean_z'] and amax_rel <= tol['amax_rel']
                   and top1 >= tol['top1_agree'] and hist_l1 <= tol['hist_l1'])
        ok = ok and r['ok']
        rows.append(r)
    if set(a) != set(b):
        ok = False
    return {'ok': bool(ok and rows), 'layers': rows, 'tol': tol}


def merge_job(job: Dict[str, Any], shard_layers: Iterable[Tuple[str, Dict[int, Dict[str, np.ndarray]]]]) -> Dict[int, Dict[str, np.ndarray]]:
    """每个分片取一份已接受结果，逐层合并全部切片；读出取自切片 0。返回 {层: 普查数组}（与 run_mlp_census 的 L<b>.npz 同格式）。"""
    merged: Dict[int, Dict[str, np.ndarray]] = {}
    readout: Dict[int, Dict[str, np.ndarray]] = {}
    for sid, layers in shard_layers:
        for L, d in layers.items():
            core = {key: d[key] for key in LAYER_KEYS}
            core['hist'] = core['hist'].astype(np.int64)
            merged[L] = core if L not in merged else C.merge(merged[L], core)
            if 'readout_top' in d:
                readout[L] = {'readout_top': d['readout_top'], 'readout_bot': d['readout_bot']}
    for L in merged:
        merged[L].update(readout.get(L, {}))
    return merged


def merged_manifest(job: Dict[str, Any], merged: Dict[int, Dict[str, np.ndarray]], files: List[Dict[str, Any]],
                    model_path: str, n_results: int) -> Dict[str, Any]:
    n_tokens = int(next(iter(merged.values()))['n']) if merged else 0
    return {'schema': 'mlp_census.v1', 'model_snapshot': {'sha256': job['model']['snapshot_sha256']},
            'model_key': job['model']['model_key'], 'model_path': model_path, 'corpus_fp': job['corpus_fp'],
            'sources': [{'source_id': i, 'source': s} for i, s in enumerate(job['sources'])],
            'max_tokens': job['tokens_per_slice'] * job['n_slices'], 'seq_len': job['seq_len'], 'skip_head': job['skip_head'],
            'k': job['k'], 'd_ff': job['d_ff'], 'n_layers': job['n_layers'], 'n_tokens_counted': n_tokens,
            'hist': {'n_side': C.N_SIDE, 'lo': C.LO, 'hi': C.HI, 'regimes': list(C.REGIMES)},
            'gidx': f'source_id << {C.SRC_SHIFT} | 来源 token 流内偏移', 'files': files, 'dtype': 'bf16',
            'distributed': {'job_id': job['job_id'], 'n_shards': len(job['shards']), 'n_results_merged': n_results,
                            'replica_rate': job['replica_rate']},
            'note': job['note'] + '；分布式分片合并（census_job.v1）'}
