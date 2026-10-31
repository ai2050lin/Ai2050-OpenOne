# -*- coding: utf-8 -*-
"""机械可解释性 v3 · A2 + G' 层：把 A1 的「头」和普查的「单元」在残差流里接起来。

A1 的结论是：**权重几何不能解释 DLA**（corr(w_letter_max,|t|)=0.349，显著头仅 1.15× 非显著）。
所以这里不沿用「与首奇异向量算 cos」，改用方案 §3 G' 的修正定义：

  A2 单元分组（D2 基的经验代理）
      普查 L{b}.npz 每单元有 top_idx = 激活最大的 32 个全局位置（gidx = source<<40 | offset）。
      把 (unit, window) 建成稀疏二值矩阵，tf-idf 加权后 SVD→KMeans，得到「在语料里一起亮的单元组」。
      这是 VPD「参数子组件」的零训练代理：不优化行为保持，只优化统计同现。
      必须验证：组内同现强度是否显著高于随机配对（否则分组无结构，D2 作废）。

  G' 读取增益（取代「与首奇异向量 cos」）
      单元 i 的写方向 u_i = down_proj 第 i 列（单位化）。下游头 h 的读取增益
          gain(i,h) = ‖W_K^h u_i‖ · ‖W_V^h u_i‖
      逐头用 256 个随机单位向量做零模型得 (mu, sd)，报 z = (gain-mu)/sd。
      z 已按头自身归一，跨层可比——这是 A1 那版用原始 cos 时做不到的。

  产物是「组 × 头」亲和矩阵，即 v3 的桥：24 组 × 12 个头上层 × 32 头，
  每个单元层只有 9k 个候选对，比 9728×1152 小三个数量级。

端点：
  POST /api/mech/pair/cluster_build?blocks=2,14,24,30,34&k=24   后台线程，约 3 min
  POST /api/mech/pair/gain_build?blocks=...&head_lo=24&head_hi=35  后台线程，约 3–6 min
  GET  /api/mech/pair/status
  GET  /api/mech/pair/clusters?block=
  GET  /api/mech/pair/gain?block=&head_layer=&head=
"""
from __future__ import annotations

import json
import math
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import APIRouter

try:
    from server import live_analysis_service as las
    from server import mech_service as msvc
    from server import neuron_census_service as ncs
except ImportError:                      # 独立脚本
    import live_analysis_service as las   # type: ignore
    import mech_service as msvc           # type: ignore
    import neuron_census_service as ncs   # type: ignore

ROOT = Path(__file__).resolve().parents[1]
ME = ROOT / 'data' / 'mech'
CLUST_JSON = ME / 'unit_clusters.json'
CLUST_NPZ = ME / 'unit_clusters.npz'
GAIN_JSON = ME / 'pair_gain.json'
GAIN_NPZ = ME / 'pair_gain.npz'

router = APIRouter(prefix='/api/mech/pair', tags=['mech-pair'])

_STATE: Dict[str, Any] = {'job': '', 'state': 'idle', 'error': '', 'done': 0, 'total': 0,
                          'elapsed_s': 0.0, 'note': ''}
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None

SEQ_LEN = 512            # 普查的前向窗口长度，用于把 offset 折成窗口号
_Z_HI = 6.0              # 单元特异度阈（同 kv 组内逐单元标准化，9728 个单元随机最大约 3.9）


# ------------------------------------------------------------------ 公用
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _set(**kw) -> None:
    with _LOCK:
        _STATE.update(kw)


def _pick_run(n_layers: int) -> Dict[str, Any]:
    """选与当前模型层数一致、token 数最多的普查 run。"""
    runs = [r for r in ncs.census_runs() if int(r.get('n_layers') or 0) == n_layers]
    if not runs:
        raise RuntimeError('没有 n_layers=%d 的普查结果；先跑自然语料普查' % n_layers)
    runs.sort(key=lambda r: int(r.get('n_tokens') or 0), reverse=True)
    return runs[0]


def _layer_npz(run: Dict[str, Any], block: int) -> Path:
    return ncs.native_root() / '_census' / run['model_key'] / run['run'] / ('L%d.npz' % block)


def _wait_ready(timeout: float = 300.0):
    """后台线程里不能直接用 _ensure_ready（加载中会抛 503），改成轮询到 ready。"""
    t0 = time.time()
    while True:
        st = getattr(las, '_state', {}).get('status')
        if st == 'ready':
            return msvc._ready()
        if st == 'error':
            raise RuntimeError('模型不可用：%s' % las._state.get('error'))
        if st == 'idle':
            las._load_async()
        if time.time() - t0 > timeout:
            raise RuntimeError('模型加载超时（%.0fs）' % timeout)
        time.sleep(2.0)


# ------------------------------------------------------------------ A2 单元分组
def _window_ids(top_idx: np.ndarray) -> np.ndarray:
    """gidx = source_id<<40 | offset  →  紧凑窗口号（source, offset//SEQ_LEN）。"""
    src = top_idx >> 40
    off = top_idx & ((1 << 40) - 1)
    w = off // SEQ_LEN
    return src.astype(np.int64) * 40_000_000 + w.astype(np.int64)


def _cluster_one_layer(run: Dict[str, Any], block: int, k: int) -> Dict[str, Any]:
    from scipy.sparse import csr_matrix
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    with np.load(_layer_npz(run, block)) as z:
        top_idx = z['top_idx'].astype(np.int64)          # [d_ff, 32]
        amax = z['amax'].astype(np.float32)
        readout = z['readout_top'].astype(np.int64)      # [d_ff, 10]
    n_unit = top_idx.shape[0]

    win = _window_ids(top_idx)                           # [n_unit, 32]
    uniq, inv = np.unique(win.ravel(), return_inverse=True)
    inv = inv.reshape(win.shape)
    n_win = int(uniq.size)

    rows = np.repeat(np.arange(n_unit, dtype=np.int64), win.shape[1])
    X = csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, inv.ravel())),
                   shape=(n_unit, n_win))
    X.sum_duplicates()
    X.data[:] = 1.0                                      # 二值：单元在窗口内是否亮

    df = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    idf = np.log((1.0 + n_win) / (1.0 + df)) + 1.0
    D = csr_matrix((idf, (np.arange(n_unit), np.arange(n_unit))), shape=(n_unit, n_unit))
    X = D @ X
    # 行 L2 归一：只比「在哪亮」，不比「亮多少」（幅度受 SwiGLU 规范自由度污染）
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    nrm[nrm == 0] = 1.0
    X = csr_matrix((X.data / np.repeat(nrm, np.diff(X.indptr)), X.indices, X.indptr),
                   shape=X.shape)

    kk = min(32, max(2, k))
    svd = TruncatedSVD(n_components=kk, random_state=20261030)
    U = svd.fit_transform(X)
    ev = svd.explained_variance_ratio_
    Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)

    km = KMeans(n_clusters=k, n_init=10, random_state=20261030)
    lab = km.fit_predict(Un)

    # --- 验证 1：组内同现是否高于随机配对（在原始二值矩阵上算 Jaccard）
    rng = np.random.default_rng(20261030)
    Xb = csr_matrix((np.ones_like(rows, dtype=np.float32), (rows, inv.ravel())),
                    shape=(n_unit, n_win))
    Xb.sum_duplicates()
    Xb.data[:] = 1.0
    sets = [Xb.indices[Xb.indptr[i]:Xb.indptr[i + 1]] for i in range(n_unit)]

    def _jac(a: int, b: int) -> float:
        sa, sb = sets[a], sets[b]
        inter = np.intersect1d(sa, sb, assume_unique=True).size
        uni = sa.size + sb.size - inter
        return inter / uni if uni else 0.0

    within = []
    for c in range(k):
        idx = np.flatnonzero(lab == c)
        if idx.size < 2:
            continue
        for _ in range(30):
            i, j = rng.choice(idx, 2, replace=False)
            within.append(_jac(int(i), int(j)))
    rand = [_jac(int(a), int(b)) for a, b in
            rng.integers(0, n_unit, size=(3000, 2)) if a != b]
    w_mean = float(np.mean(within)) if within else 0.0
    r_mean = float(np.mean(rand)) if rand else 0.0
    lift = w_mean / r_mean if r_mean > 0 else float('nan')

    # --- 验证 2：组内「共推同一批词」是否高于随机（readout_top 前 5 词集合的 Jaccard）
    rs = [set(readout[i, :5].tolist()) for i in range(n_unit)]

    def _tj(a: int, b: int) -> float:
        inter = len(rs[a] & rs[b])
        uni = len(rs[a] | rs[b])
        return inter / uni if uni else 0.0

    coh = []
    for c in range(k):
        idx = np.flatnonzero(lab == c)
        if idx.size < 2:
            continue
        for _ in range(20):
            i, j = rng.choice(idx, 2, replace=False)
            coh.append(_tj(int(i), int(j)))
    base = [_tj(int(a), int(b)) for a, b in rng.integers(0, n_unit, size=(3000, 2)) if a != b]
    coh_mean = float(np.mean(coh)) if coh else 0.0
    base_share = float(np.mean(base)) if base else 0.0
    tok_lift = coh_mean / base_share if base_share > 0 else float('nan')

    # --- 每组的画像
    tok_of = {}
    for c in range(k):
        idx = np.flatnonzero(lab == c)
        if idx.size == 0:
            continue
        flat = readout[idx].ravel()
        u, c2 = np.unique(flat, return_counts=True)
        order = np.argsort(-c2)[:8]
        tok_of[int(c)] = [[int(u[o]), int(c2[o])] for o in order]
        # 该组最典型的单元（离质心最近的 5 个）
        cen = km.cluster_centers_[c]
        dist = np.linalg.norm(Un[idx] - cen, axis=1)
        near = idx[np.argsort(dist)[:5]].tolist()
        tok_of[int(c)] = {'tokens': tok_of[int(c)], 'near_units': [int(x) for x in near]}

    return {
        'block': block, 'n_unit': n_unit, 'n_window': n_win,
        'k': k, 'svd_k': kk,
        'svd_ev': [round(float(x), 5) for x in ev],
        'svd_ev_sum': round(float(ev.sum()), 4),
        'within_jaccard': round(w_mean, 5), 'random_jaccard': round(r_mean, 5),
        'lift': None if not np.isfinite(lift) else round(float(lift), 3),
        'readout_coh': round(coh_mean, 4), 'readout_coh_base': round(base_share, 4),
        'readout_lift': None if not np.isfinite(tok_lift) else round(float(tok_lift), 3),
        'sizes': {int(c): int((lab == c).sum()) for c in range(k)},
        'cluster_tokens': tok_of,
        '_lab': lab.astype(np.int16),
        '_amax': amax,
    }


def _run_cluster(blocks: List[int], k: int) -> None:
    t0 = __import__('time').time()
    try:
        model_key_note = ''
        layers: List[Dict[str, Any]] = []
        n_layers = None
        for i, b in enumerate(blocks):
            if n_layers is None:
                # 用第一层的 manifest 推层数
                runs_all = ncs.census_runs()
                if not runs_all:
                    raise RuntimeError('没有普查结果')
                n_layers = int(max(runs_all, key=lambda r: int(r.get('n_tokens_counted') or 0))['n_layers'])
            run = _pick_run(n_layers)
            model_key_note = '%s/%s' % (run['model_key'], run['run'])
            _set(done=i, total=len(blocks), note='L%d 共现聚类' % b)
            layers.append(_cluster_one_layer(run, b, k))
        _write_npz(CLUST_NPZ, **{
            'lab_%d' % d['block']: d.pop('_lab') for d in layers
        })
        for d in layers:
            d.pop('_amax', None)
        _write_json(CLUST_JSON, {
            'schema': 'mech_unit_clusters.v1', 'census': model_key_note,
            'n_layers': n_layers, 'k': k, 'seq_len': SEQ_LEN,
            'note': 'tf-idf 二值 (unit×window) 矩阵 → TruncatedSVD(32) → KMeans(k)。'
                    'lift = 组内平均窗口 Jaccard / 随机配对平均，<1.5 视为无结构；'
                    'readout_lift = 组内 readout_top 前 5 词集合 Jaccard / 随机配对，>1.5 说明组有共同读出；'
                    'svd_ev_sum 是前 32 个奇异值的方差占比，偏低说明共现结构是高维的（组只是近似）。',
            'layers': layers,
        })
        el = __import__('time').time() - t0
        _set(job='cluster', state='done', done=len(blocks), elapsed_s=round(el, 1),
             note='%d 层聚类完成' % len(blocks))
    except Exception as e:                      # 失败必须可见
        _set(job='cluster', state='error', error='%s: %s' % (type(e).__name__, e),
             note=traceback.format_exc()[-800:])


# ------------------------------------------------------------------ G' 读取增益
def _run_gain(blocks: List[int], head_lo: int, head_hi: int) -> None:
    import time as _t
    t0 = _t.time()
    try:
        model, tok, device, meta, model_name = _wait_ready()
        L = int(meta['n_layers'])
        H, Hkv, dh = msvc._attn_slices(model)
        d = int(model.config.hidden_size)
        group = max(1, H // Hkv)
        head_hi = min(head_hi, L - 1)

        rng = torch.Generator(device='cpu').manual_seed(20261030)
        G = torch.randn(256, d, generator=rng).to(device)
        G = G / G.norm(dim=1, keepdim=True).clamp_min(1e-9)

        t_ans = None
        try:
            z = np.load(msvc.POS_NPZ)
            t_ans = z['dla_net_t'][:, :, 2]      # 答案位，层内归一化后的 t
        except Exception:
            t_ans = None

        hl_list = list(range(max(head_lo, 0), head_hi + 1))
        out_layers: List[Dict[str, Any]] = []
        top_rec: List[Dict[str, Any]] = []
        corr_rows: List[Dict[str, Any]] = []
        coh_rows: List[Dict[str, Any]] = []

        # A2 的分组标签（若已跑过 cluster_build）
        lab_by_block: Dict[int, Any] = {}
        if CLUST_JSON.exists():
            try:
                cj = json.loads(CLUST_JSON.read_text(encoding='utf-8'))
                cz = np.load(CLUST_NPZ)
                for rec in cj['layers']:
                    key = 'lab_%d' % rec['block']
                    if key in cz.files:
                        lab_by_block[rec['block']] = (cz[key].astype(np.int64), int(rec['k']))
            except Exception:
                lab_by_block = {}

        for bi, b in enumerate(blocks):
            _set(done=bi, total=len(blocks), note='L%d 单元写入方向 → 下游头读取增益' % b)
            Wd = model.model.layers[b].mlp.down_proj.weight.detach().float().to(device)   # [d, d_ff]
            U = Wd.t().contiguous()                                                       # [d_ff, d]
            U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-9)
            n_unit = int(U.shape[0])

            # 组级权重验证：同组的写方向是否比随机配对更像（VPD 的「子组件写同一方向」假设）
            if b in lab_by_block:
                lab_k, K = lab_by_block[b]
                rg = np.random.default_rng(20261030 + b)
                Wg = model.model.layers[b].mlp.gate_proj.weight.detach().float().to(device)  # [d_ff, d]
                Rg = Wg / Wg.norm(dim=1, keepdim=True).clamp_min(1e-9)
                per_c, per_cr = [], []
                for c in range(K):
                    idx = np.flatnonzero(lab_k == c)
                    if idx.size < 4:
                        continue
                    a = rg.choice(idx, size=min(40, idx.size), replace=False)
                    ti = torch.as_tensor(a, device=device)
                    Vc = U[ti]
                    S = (Vc @ Vc.t()).abs()
                    iu = torch.triu_indices(len(a), len(a), offset=1)
                    per_c.append(float(S[iu[0], iu[1]].mean()))
                    Rc = Rg[ti]
                    Sr2 = (Rc @ Rc.t()).abs()
                    per_cr.append(float(Sr2[iu[0], iu[1]].mean()))
                a2 = rg.choice(n_unit, size=80, replace=False)
                t2 = torch.as_tensor(a2, device=device)
                Vr = U[t2]
                Sr = (Vr @ Vr.t()).abs()
                iur = torch.triu_indices(len(a2), len(a2), offset=1)
                base_c = float(Sr[iur[0], iur[1]].mean())
                Rr = Rg[t2]
                Srr = (Rr @ Rr.t()).abs()
                base_r = float(Srr[iur[0], iur[1]].mean())
                w_c = float(np.mean(per_c)) if per_c else 0.0
                w_r = float(np.mean(per_cr)) if per_cr else 0.0
                coh_rows.append({'unit_block': b,
                                 'write_within_abs_cos': round(w_c, 4),
                                 'write_random_abs_cos': round(base_c, 4),
                                 'write_lift': round(w_c / base_c, 2) if base_c > 0 else None,
                                 'read_within_abs_cos': round(w_r, 4),
                                 'read_random_abs_cos': round(base_r, 4),
                                 'read_lift': round(w_r / base_r, 2) if base_r > 0 else None,
                                 'null_abs_cos_1_over_sqrt_d': round(1.0 / math.sqrt(d), 4)})
                del Vc, S, Vr, Sr, Rc, Sr2, Rr, Srr, Wg, Rg

            per_hl = {}
            for l2 in hl_list:
                if l2 <= b:
                    continue
                att = model.model.layers[l2].self_attn
                Wk = att.k_proj.weight.detach().float().to(device)    # [Hkv*dh, d]
                Wv = att.v_proj.weight.detach().float().to(device)
                with torch.no_grad():
                    kn = (U @ Wk.t()).view(-1, Hkv, dh).norm(dim=2)   # [d_ff, Hkv]
                    vn = (U @ Wv.t()).view(-1, Hkv, dh).norm(dim=2)
                    gain = (kn * vn).double().cpu().numpy()            # [d_ff, Hkv]
                    kn0 = (G @ Wk.t()).view(-1, Hkv, dh).norm(dim=2)
                    vn0 = (G @ Wv.t()).view(-1, Hkv, dh).norm(dim=2)
                    g0 = (kn0 * vn0).double().cpu().numpy()            # [256, Hkv]
                per_hl[l2] = (gain, g0)
                del kn, vn, kn0, vn0

            # GQA 下 K/V 只有 Hkv 个独立通道，32 个 query 头里每 group 个共享同一读取增益，
            # 所以统计一律落在 kv 组上，不假装有 32 个独立观测。
            for l2, (gain, g0) in per_hl.items():
                mu_n, sd_n = g0.mean(0), np.maximum(g0.std(0), 1e-9)
                head_pref = (gain.mean(0) - mu_n) / sd_n                    # [Hkv] 头级：读子空间是否偏好 MLP 写方向
                us = (gain - gain.mean(0)) / np.maximum(gain.std(0), 1e-9)  # [d_ff,Hkv] 单元特异
                cnt_hi = (us > _Z_HI).sum(0).astype(int)
                rec = {'unit_block': b, 'head_layer': l2,
                       'n_unit_spec': cnt_hi.tolist(),
                       'max_unit_spec': [round(float(x), 2) for x in us.max(0)],
                       'head_pref_vs_random': [round(float(x), 2) for x in head_pref],
                       'top_units': {}}
                for k in range(Hkv):
                    idx = np.argsort(-us[:, k])[:5]
                    rec['top_units'][str(k)] = [[int(i), round(float(us[i, k]), 2)] for i in idx]
                if b in lab_by_block:
                    lab, K = lab_by_block[b]
                    agg = np.zeros((K, Hkv), dtype=np.float64)
                    for c in range(K):
                        idx = np.flatnonzero(lab == c)
                        if idx.size:
                            agg[c] = us[idx].mean(axis=0)
                    rec['cluster_affinity'] = [[round(float(x), 2) for x in row] for row in agg]
                top_rec.append(rec)

                if t_ans is not None:
                    tt = np.abs(t_ans[l2]).reshape(Hkv, -1).mean(axis=1)     # 每个 kv 组的平均 |t|
                    corr_rows.append({
                        'unit_block': b, 'head_layer': l2, 'n_kv': int(Hkv),
                        'corr_nspec_absT': round(float(np.corrcoef(cnt_hi, tt)[0, 1]), 3),
                        'corr_headpref_absT': round(float(np.corrcoef(head_pref, tt)[0, 1]), 3),
                        'absT_by_kv': [round(float(x), 2) for x in tt],
                        'n_unit_spec_total': int(cnt_hi.sum()),
                    })
            out_layers.append({'unit_block': b, 'head_layers': sorted(per_hl.keys())})
            del U, Wd
            torch.cuda.empty_cache()

        _write_json(GAIN_JSON, {
            'schema': 'mech_pair_gain.v1', 'model_name': model_name,
            'n_layers': L, 'n_heads': H, 'n_kv_heads': Hkv, 'kv_group': group,
            'head_dim': dh, 'd_model': d,
            'z_hi': _Z_HI, 'blocks': blocks, 'head_lo': head_lo, 'head_hi': head_hi,
            'note': 'gain(i,k)=‖W_K^k u_i‖·‖W_V^k u_i‖，u_i 为 down_proj 第 i 列单位化，k 为 KV 组。'
                    'head_pref_vs_random = 该 KV 组对 MLP 写方向的偏好（对 256 个随机单位向量标准化）；'
                    'unit_spec = 同组内逐单元标准化，问的是「这个单元被这组读得比别的单元强多少」，'
                    '>6 记显著。head_pref_vs_random 与 unit_spec 用两套零模型，不可混用。'
                    '问题：Qwen3-4B 是 GQA（32 query 头 / 8 KV 头），同组 4 个 query 头读取增益完全相同，'
                    '所以头级分辨率上限是 8 个 KV 组，不是 32 个头。',
            'pairs': top_rec,
            'vs_dla': corr_rows,
            'cluster_write_coherence': coh_rows,
        })
        el = _t.time() - t0
        _set(job='gain', state='done', done=len(blocks), elapsed_s=round(el, 1),
             note='%d 个单元层 × %d 个头上层' % (len(blocks), len(hl_list)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        _set(job='gain', state='error', error='%s: %s' % (type(e).__name__, e),
             note=traceback.format_exc()[-800:])


# ------------------------------------------------------------------ 端点
def _spawn(job: str, fn, *args) -> Dict[str, Any]:
    global _THREAD
    with _LOCK:
        if _STATE['state'] == 'running':
            return {'accepted': False, 'reason': '已有任务在跑：%s' % _STATE['job']}
        _STATE.update({'job': job, 'state': 'running', 'error': '', 'done': 0,
                       'total': 1, 'elapsed_s': 0.0, 'note': '启动'})
    _THREAD = threading.Thread(target=fn, args=args, daemon=True)
    _THREAD.start()
    return {'accepted': True, 'job': job}


@router.post('/cluster_build')
def cluster_build(blocks: str = '2,14,24,30,34', k: int = 24) -> dict:
    bl = [int(x) for x in blocks.split(',') if x.strip()]
    return _spawn('cluster', _run_cluster, bl, int(k))


@router.post('/gain_build')
def gain_build(blocks: str = '2,14,24,30,34', head_lo: int = 24, head_hi: int = 35) -> dict:
    bl = [int(x) for x in blocks.split(',') if x.strip()]
    return _spawn('gain', _run_gain, bl, int(head_lo), int(head_hi))


@router.get('/status')
def status() -> dict:
    with _LOCK:
        st = dict(_STATE)
    st['files'] = {p.name: p.stat().st_size for p in (CLUST_JSON, CLUST_NPZ, GAIN_JSON, GAIN_NPZ)
                   if p.exists()}
    st['schema'] = 'mech_pair_status.v1'
    return st


def _decode(ids: List[int]) -> List[str]:
    """用已加载的 tokenizer 解词；模型没加载时返回空（前端只显示 id）。"""
    st = getattr(las, '_state', None) or {}
    tok = st.get('tokenizer')
    if tok is None:
        return []
    out = []
    for i in ids:
        try:
            out.append(tok.decode([int(i)]))
        except Exception:
            out.append('')
    return out


@router.get('/clusters')
def clusters(block: int = -1) -> dict:
    if not CLUST_JSON.exists():
        return {'available': False, 'reason': '先 POST /api/mech/pair/cluster_build'}
    d = json.loads(CLUST_JSON.read_text(encoding='utf-8'))
    layers = d['layers'] if block < 0 else [r for r in d['layers'] if r['block'] == block]
    for rec in layers:
        for c, info in list(rec['cluster_tokens'].items()):
            ids = [t[0] for t in info['tokens']]
            info['words'] = _decode(ids)
    return {'available': True, 'schema': d['schema'], 'k': d['k'],
            'census': d['census'], 'note': d['note'], 'layers': layers}


@router.get('/gain')
def gain(unit_block: int = -1, head_layer: int = -1, kv: int = -1) -> dict:
    if not GAIN_JSON.exists():
        return {'available': False, 'reason': '先 POST /api/mech/pair/gain_build'}
    d = json.loads(GAIN_JSON.read_text(encoding='utf-8'))
    sel = [r for r in d['pairs']
           if (unit_block < 0 or r['unit_block'] == unit_block)
           and (head_layer < 0 or r['head_layer'] == head_layer)]
    if kv >= 0:
        for r in sel:
            k = str(kv)
            r['n_unit_spec'] = [r['n_unit_spec'][kv]]
            r['max_unit_spec'] = [r['max_unit_spec'][kv]]
            r['head_pref_vs_random'] = [r['head_pref_vs_random'][kv]]
            r['top_units'] = {k: r['top_units'].get(k, [])}
            if 'cluster_affinity' in r:
                r['cluster_affinity'] = [row[kv] for row in r['cluster_affinity']]
    return {'available': True, 'schema': d['schema'], 'z_hi': d['z_hi'],
            'n_kv_heads': d['n_kv_heads'], 'kv_group': d['kv_group'],
            'note': d['note'], 'vs_dla': d['vs_dla'],
            'cluster_write_coherence': d.get('cluster_write_coherence', []),
            'pairs': sel}
