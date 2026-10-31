# -*- coding: utf-8 -*-
"""机械可解释性 v3 · A1 层：把「单元 / 头」换成「残差流方向」所需的三张底表。

方案见 `design/lang_code_joint_attribution_v3.md` §3 的 R 流水线。本模块只做 A1，不做因果：

  R1 头方向  每头的 OV 主方向（写）与 QK 主方向（读），纯权重运算，秒级。
             全部在残差流 d_model 维的同一坐标系里，这是 v3「方向对」原子的前提。
  R3 头类型  induction / prev-token / duplicate / sink / diffuse 的自动打分，
             用重复随机序列探针，一次前向。族的定义要用「类型 × 层带」，不能靠头编号。
  R0 位置    v2 最大的缺口：只读末位等于只看见汇聚点。这里在
             题干末 / 各选项末 / 答案位 三处分别取注意力分布与 DLA，
             用来检验「早层在选项位、晚层在答案位」这个分层假设（v3 预测 P6）。

纪律（与本项目既有口径一致）：
  · 零模型基线 |cos| ≈ 1/√d_model = 0.0198，任何「对齐」必须先超过它才算有结构；
  · DLA 用层内归一化，跨层只比归一化后的量；
  · 没有测到的量不出现在结果里，不用示意数据填格子；
  · 失败必须可见：每个 build 端点返回 accepted / reason，后台任务的错误写进 status。

端点：
  POST /api/mech/heads_build          算并落盘头方向（秒级，同步）
  GET  /api/mech/heads?layer=&top=    读头方向表
  POST /api/mech/types_build          算并落盘头类型（秒级，同步，需一次前向）
  GET  /api/mech/types?layer=
  POST /api/mech/pos_run?dataset=&n=  位置分辨读出（后台线程，169 题约 1–2 分钟）
  GET  /api/mech/pos
  GET  /api/mech/status
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
    # 必须走包内导入：裸名会加载出第二份模块副本（独立 _state），见 lang_atlas_service 的注释。
    from server import live_analysis_service as las
    from server import lang_atlas_service as latlas
except ImportError:                      # 独立脚本 / 以 server 为根启动时退回裸名
    import live_analysis_service as las
    import lang_atlas_service as latlas

ROOT = Path(__file__).resolve().parents[1]
ME = ROOT / 'data' / 'mech'
HEADS_NPZ = ME / 'heads_dir.npz'
HEADS_JSON = ME / 'heads_dir.json'
TYPES_NPZ = ME / 'head_types.npz'
TYPES_JSON = ME / 'head_types.json'
POS_NPZ = ME / 'pos_profile.npz'
POS_JSON = ME / 'pos_profile.json'

router = APIRouter(prefix='/api/mech', tags=['mech'])

_NULL_COS = 1.0 / math.sqrt(2560.0)      # 仅作参考展示；实际用模型维度重算

_state: Dict[str, Any] = {'job': '', 'state': 'idle', 'error': '', 'done': 0, 'total': 0,
                          'elapsed_s': 0.0, 'note': ''}
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None


def _write_json(path: Path, obj: Any) -> None:
    ME.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:   # newline='' 防 Windows 换行污染
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _write_npz(path: Path, **arrays) -> None:
    ME.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _ready():
    """取已加载的模型；未加载则触发后台加载并等待（首次约 40 s）。

    `_ensure_ready()` 在 loading 态会直接抛 503，不能用于同步 build 端点，
    所以这里与 lang_atlas_service 用同一套轮询：
    注意：直接调用会读到旧状态，必须每次读 `_state['status']`。
    """
    deadline = time.time() + 900
    while time.time() < deadline:
        st = las._state['status']
        if st == 'ready':
            break
        if st == 'idle':
            las._load_async()
        if st == 'error':
            raise RuntimeError(las._state['error'] or '模型不可用')
        time.sleep(1.0)
    else:
        raise RuntimeError('等待模型加载超时（15 分钟）')
    with las._LOCK:
        st = las._state
        return st['model'], st['tokenizer'], st['device'], st['meta'], Path(st['dir'] or '').name


def _attn_slices(model):
    """返回每层 (q_proj, k_proj, v_proj, o_proj) 的权重与 GQA 分组。"""
    cfg = model.config
    H = int(getattr(cfg, 'num_attention_heads'))
    Hkv = int(getattr(cfg, 'num_key_value_heads', H) or H)
    dh = int(getattr(cfg, 'head_dim', None) or (cfg.hidden_size // H))
    return H, Hkv, dh


def _svd_left_times(A: torch.Tensor, B: torch.Tensor, k: int):
    """算 M = A @ B 的前 k 个左奇异向量与经济奇异值，不显式构造 d×d 矩阵。

    A [d, r]，B [r, d]。A = Ua Sa Vha^T  =>  M = Ua (Sa Vha^T B) = Ua C。
    C 的 SVD 给 Uc Sc Vhc^T，于是 M = (Ua Uc) Sc Vhc^T，左奇异向量 = Ua @ Uc。
    经济 SVD，d=2560 / r=80 时单次约 1 ms，1152 个头秒级完成。
    """
    Ua, Sa, Vha = torch.linalg.svd(A, full_matrices=False)
    C = (Sa.unsqueeze(1) * Vha) @ B
    Uc, Sc, Vhc = torch.linalg.svd(C, full_matrices=False)
    U = Ua @ Uc
    return U[:, :k], Sc[:k], Sc, Vhc


def _participation_ratio(s: torch.Tensor) -> float:
    s2 = (s.float() ** 2)
    denom = float((s2 ** 2).sum())
    if denom <= 0:
        return 0.0
    return float((s2.sum() ** 2) / denom)


def _letter_dirs(model, tok) -> Dict[str, torch.Tensor]:
    """五个选项字母的 unembedding 方向（乘 final norm 增益），与 DLA 的读出方向同口径。"""
    Wu = model.get_output_embeddings().weight.detach().float()
    g = model.model.norm.weight.detach().float()
    ids = latlas._letter_token_ids(tok)
    out = {}
    for letter in 'ABCDE':
        t = next(iter(ids.get(letter) or []), None)
        if t is None:
            continue
        v = (g * Wu[t]).float()
        out[letter] = v / (v.norm() + 1e-12)
    return out


# ------------------------------------------------------------------ R1 头方向
def build_heads() -> dict:
    model, tok, device, meta, model_name = _ready()
    L = int(meta['n_layers'])
    H, Hkv, dh = _attn_slices(model)
    group = max(1, H // Hkv)
    d = int(model.config.hidden_size)
    letters = _letter_dirs(model, tok)
    lmat = torch.stack(list(letters.values())) if letters else None       # [k, d] 单位向量
    lkeys = list(letters.keys())

    w_dir = np.zeros((L, H, 3, d), dtype=np.float16)
    r_dir = np.zeros((L, H, 3, d), dtype=np.float16)
    w_sing = np.zeros((L, H, dh), dtype=np.float16)
    r_sing = np.zeros((L, H, dh), dtype=np.float16)
    scal: List[dict] = []

    with torch.no_grad():
        for l in range(L):
            att = model.model.layers[l].self_attn
            Wo = att.o_proj.weight.detach().float()      # [d, H*dh]
            Wv = att.v_proj.weight.detach().float()      # [Hkv*dh, d]
            Wq = att.q_proj.weight.detach().float()      # [H*dh, d]
            Wk = att.k_proj.weight.detach().float()      # [Hkv*dh, d]
            for h in range(H):
                kv = h // group
                O = Wo[:, h * dh:(h + 1) * dh]           # [d, dh]
                V = Wv[kv * dh:(kv + 1) * dh, :]         # [dh, d]
                Q = Wq[h * dh:(h + 1) * dh, :].t()       # [d, dh]
                K = Wk[kv * dh:(kv + 1) * dh, :]         # [dh, d]

                Uw, sw, sw_all, _ = _svd_left_times(O, V, 3)     # OV 左奇异 = 写方向
                Uq, sq, sq_all, Vhc = _svd_left_times(Q, K, 3)   # Q^T K 左奇异 = 查询侧读方向

                w_dir[l, h] = Uw.t().cpu().numpy().astype(np.float16)     # [k, d]
                r_dir[l, h] = Uq.t().cpu().numpy().astype(np.float16)
                w_sing[l, h] = sw_all.cpu().numpy().astype(np.float16)
                r_sing[l, h] = sq_all.cpu().numpy().astype(np.float16)

                w0 = Uw[:, 0]
                r0 = Uq[:, 0]
                rec = {'block': l, 'head': h, 'kv': kv,
                       'ov_pr': round(_participation_ratio(sw_all), 2),
                       'qk_pr': round(_participation_ratio(sq_all), 2),
                       's1': round(float(sw_all[0]), 4), 's2': round(float(sw_all[1]), 4)}
                if lmat is not None:
                    cw = (w0[None, :] @ lmat.t()).squeeze(0)     # [k]
                    cr = (r0[None, :] @ lmat.t()).squeeze(0)
                    rec['w_letter'] = {k2: round(float(x), 4) for k2, x in zip(lkeys, cw)}
                    rec['r_letter'] = {k2: round(float(x), 4) for k2, x in zip(lkeys, cr)}
                    # 写方向的读出：OV 在五个字母子空间上的能量占比（不是单方向 cos）
                    P = lmat.t() @ lmat                          # [d,d] 投影（k 小，可显式算）
                    num = float((w0 @ P).norm())
                    rec['w_letter_energy'] = round(num, 4)
                    rec['w_letter_max'] = round(float(cw.abs().max()), 4)
                scal.append(rec)

    _write_npz(HEADS_NPZ, w_dir=w_dir, r_dir=r_dir, w_sing=w_sing, r_sing=r_sing)
    out = {
        'schema': 'mech_heads.v1', 'model_name': model_name, 'n_layers': L, 'n_heads': H,
        'n_kv_heads': Hkv, 'head_dim': dh, 'd_model': d,
        'null_cos': round(1.0 / math.sqrt(d), 4),
        'note': 'w_dir = OV 前 3 个左奇异向量（写），r_dir = Q^T K 前 3 个左奇异向量（查询侧读）；'
                '两者都在残差流同一坐标系，可直接算 cos。ov_pr/qk_pr 为有效秩（奇异值平方的参与比），'
                '低有效秩说明这个头只沿少数方向工作。',
        'heads': scal,
    }
    _write_json(HEADS_JSON, out)
    return {'accepted': True, 'schema': 'mech_heads.v1', 'n_heads': len(scal),
            'n_layers': L, 'head_dim': dh, 'null_cos': out['null_cos'],
            'path': str(HEADS_JSON)}


# ------------------------------------------------------------------ R3 头类型
_TYPE_KEYS = ['induction', 'prev', 'dup', 'sink', 'self', 'entropy']


def _probe_batch(tok, device, n_seq: int = 4, m: int = 48, seed: int = 20261030):
    rng = np.random.default_rng(seed)
    lo, hi = 3000, 20000
    x = rng.integers(lo, hi, size=(n_seq, m), dtype=np.int64)
    seq = np.concatenate([x, x], axis=1)                     # 重复一次 => 后半段可做 induction
    return torch.tensor(seq, device=device), m


def build_types() -> dict:
    model, tok, device, meta, model_name = _ready()
    L = int(meta['n_layers'])
    H, Hkv, dh = _attn_slices(model)
    ids, m = _probe_batch(tok, device)
    n_seq, T = ids.shape

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_attentions=True)
        atts = out.attentions                                    # tuple of [n, H, T, T]

    acc = np.zeros((L, H, len(_TYPE_KEYS)), dtype=np.float64)
    cnt = 0
    for s in range(n_seq):
        x = ids[s].cpu().numpy()
        for i in range(m + 1, T):                                # 只统计后半段
            prev_pos = i - m                                     # 同一 token 上一次出现
            tgt_ind = prev_pos + 1
            if tgt_ind >= T:
                continue
            row_key = {
                'induction': tgt_ind,
                'prev': i - 1,
                'sink': 0,
                'self': i,
            }
            same = [j for j in range(i) if x[j] == x[i]]
            for l in range(L):
                A = atts[l][s].float().cpu().numpy()             # [H, T]
                for name, j in row_key.items():
                    acc[l, :, _TYPE_KEYS.index(name)] += A[:, i, j]
                if same:
                    acc[l, :, _TYPE_KEYS.index('dup')] += A[:, i, same].mean()
                else:
                    acc[l, :, _TYPE_KEYS.index('dup')] += 0.0
                ent = -(A[:, i, :i + 1] * np.log(A[:, i, :i + 1] + 1e-12)).sum(axis=1)
                acc[l, :, _TYPE_KEYS.index('entropy')] += ent / math.log(i + 2)
            cnt += 1
    if cnt == 0:
        raise RuntimeError('探针没有产生任何统计位置')
    acc /= cnt

    # 类型判定：阈值写死在返回里，允许前端按自己的阈值重判
    # 注意 sink 不是「结构类型」而是近乎普遍的行为（实测 sink 得分中位 0.484、
    # 66% 的头 ≥0.3），若按「sink 优先」判定会把所有结构头吞掉。
    # 所以主标签只在 induction / prev / dup 之间取 argmax，sink 仅作独立标记。
    THR = {'induction': 0.20, 'prev': 0.20, 'dup': 0.20, 'sink_flag': 0.60, 'local_entropy': 0.75}
    heads = []
    for l in range(L):
        for h in range(H):
            v = {k: round(float(acc[l, h, i]), 4) for i, k in enumerate(_TYPE_KEYS)}
            struct = {'induction': v['induction'], 'prev-token': v['prev'], 'duplicate': v['dup']}
            bname = max(struct, key=struct.get)
            if struct[bname] >= THR[{'induction': 'induction', 'prev-token': 'prev',
                                     'duplicate': 'dup'}[bname]]:
                label = bname
            elif v['sink'] >= THR['sink_flag']:
                label = 'sink'
            elif v['entropy'] < THR['local_entropy']:
                label = 'local'
            else:
                label = 'diffuse'
            heads.append({'block': l, 'head': h, 'type': label,
                          'sink_flag': bool(v['sink'] >= THR['sink_flag']), **v})

    _write_npz(TYPES_NPZ, score=acc.astype(np.float32))
    from collections import Counter
    by_type = Counter(h['type'] for h in heads)
    by_layer = {}
    for h in heads:
        by_layer.setdefault(str(h['block']), Counter())[h['type']] += 1
    out = {'schema': 'mech_head_types.v1', 'model_name': model_name,
           'n_layers': L, 'n_heads': H, 'keys': _TYPE_KEYS, 'thresholds': THR,
           'probe': {'n_seq': int(n_seq), 'm': m, 'seed': 20261030,
                     'note': '随机 token 重复序列；induction = 当前 token 上一次出现之后那位；'
                             'prev = 前一位；sink = 第 0 位；dup = 同 token 历史均值；'
                             'entropy = 归一化注意力熵（<0.75 记 local）。'},
           'counts': dict(by_type),
           'by_layer': {k: dict(v) for k, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]))},
           'heads': heads}
    _write_json(TYPES_JSON, out)
    return {'accepted': True, 'schema': 'mech_head_types.v1', 'counts': dict(by_type),
            'n_positions': cnt, 'path': str(TYPES_JSON)}


# ------------------------------------------------------------------ R0 位置分辨
_POS_KEYS = ['q_end', 'opt_end', 'ans']
_SEG_KEYS = ['instr', 'q', 'own', 'others', 'ans']


def _measure_pos(model, tok, device, q: dict, letter_ids, max_len: int = 640) -> Optional[dict]:
    """一题一次前向，在三个读出位分别取：注意力分段质量、归一化熵、各头 DLA 净推力。

    与 atlas 的差别只有两点：① 读出位从 1 个变 3 个；② DLA 也在选项位算一遍。
    这两点是 v3 预测 P6（早层在选项位、晚层在答案位）的唯一证据来源。
    """
    prompt, lines, _ = latlas._build_prompt(q)
    enc = tok(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0][:max_len]
    T = int(enc.shape[0])
    spans, seg_total = latlas._spans(tok, lines)
    shift = T - seg_total
    if abs(shift) > 4:
        return None

    def adj(name):
        lo, hi = spans[name]
        return max(0, lo + shift), min(T, hi + shift)

    letters = list('ABCDE'[:len(q['choices'])])
    opt = {}
    for ch in letters:
        lo, hi = adj(f'opt{ch}')
        if hi - 1 > lo:
            opt[ch] = (lo, hi)
    if len(opt) < len(letters) or q['answer'] not in opt:
        return None
    ans_tok = next((c for c in letter_ids.get(q['answer'], [])), None)
    if ans_tok is None:
        return None

    q_lo, q_hi = adj('q')
    q_end = max(q_lo, q_hi - 1)
    ans_pos = T - 1
    pos_list = [q_end] + [opt[ch][1] - 1 for ch in letters] + [ans_pos]
    pos_idx = torch.tensor(pos_list, device=device, dtype=torch.long)

    head_at: Dict[int, torch.Tensor] = {}
    handles = []
    for li, layer in enumerate(model.model.layers):
        def mk(i):
            def hook(_m, args):
                head_at[i] = args[0].detach()[0].index_select(0, pos_idx).float()   # [n_pos, H*dh]
            return hook
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mk(li)))

    try:
        with las._RUN_LOCK, torch.no_grad():
            out = model(input_ids=enc.unsqueeze(0).to(device), use_cache=False, output_attentions=True)
            atts = out.attentions
            L = len(atts)
            H = int(atts[0].shape[1])

            # ---- 注意力：三个读出位 × 五类分段（层在外层，每层只做一次 CPU 转换）
            mass = np.zeros((L, H, len(_POS_KEYS), len(_SEG_KEYS)), dtype=np.float32)
            ent = np.zeros((L, H, len(_POS_KEYS)), dtype=np.float32)
            seg_ranges = {'instr': adj('instr'), 'q': (q_lo, q_hi), 'ans': adj('ans')}
            plists = {'q_end': [q_end],
                      'opt_end': [opt[ch][1] - 1 for ch in letters],
                      'ans': [ans_pos]}
            cntp = np.array([max(1, len(plists[p])) for p in _POS_KEYS], dtype=np.float32)
            for l in range(L):
                A = atts[l][0].float().cpu().numpy()                       # [H, T, T]
                for pi, pname in enumerate(_POS_KEYS):
                    for pp in plists[pname]:
                        row = A[:, pp, :]                                  # [H, T]
                        for si, sname in enumerate(_SEG_KEYS):
                            if sname in ('own', 'others'):
                                if pname == 'q_end':
                                    continue
                                cur = next((ch for ch in letters if opt[ch][1] - 1 == pp), None)
                                if sname == 'own':
                                    if cur is None:          # 答案位没有「自己的选项」
                                        continue
                                    lo, hi = opt[cur]
                                    mass[l, :, pi, si] += row[:, lo:hi].sum(axis=1)
                                else:
                                    # 选项末：除自己以外的选项；答案位：全部选项文本
                                    accv = np.zeros(H, dtype=np.float32)
                                    for ch in letters:
                                        if ch == cur:
                                            continue
                                        lo, hi = opt[ch]
                                        accv += row[:, lo:hi].sum(axis=1)
                                    mass[l, :, pi, si] += accv
                            else:
                                lo, hi = seg_ranges[sname]
                                if hi > lo:
                                    mass[l, :, pi, si] += row[:, lo:hi].sum(axis=1)
                        p = np.clip(row[:, :pp + 1], 1e-12, None)
                        ent[l, :, pi] += (-(p * np.log(p)).sum(axis=1)) / math.log(pp + 2)
            mass /= cntp.reshape(1, 1, -1, 1)
            ent /= cntp.reshape(1, 1, -1)

            # ---- DLA：三个读出位各一份
            Wu = model.lm_head.weight.detach()
            ng = model.model.norm.weight.detach().float()
            cols = []
            for ch in letters:
                t = next(iter(letter_ids.get(ch) or []), None)
                if t is None:
                    return None
                cols.append(Wu[t].float())
            uk = torch.stack(cols) * ng                                  # [k, d]
            dla = np.zeros((L, H, len(_POS_KEYS), len(letters)), dtype=np.float32)
            npos = len(pos_list)
            base = {'q_end': 0, 'opt_end': list(range(1, 1 + len(letters))), 'ans': npos - 1}
            for l in range(L):
                Wo = model.model.layers[l].self_attn.o_proj.weight.detach().float()
                hd = Wo.shape[1] // H
                v = head_at[l].view(npos, H, hd)                          # [npos, H, hd]
                c = torch.einsum('dhk,nhk->ndh', Wo.view(Wo.shape[0], H, hd), v)   # [npos, d, H]
                d_l = torch.einsum('ndh,kd->nhk', c, uk).cpu().numpy()    # [npos, H, k]
                for pi, pname in enumerate(_POS_KEYS):
                    if pname == 'q_end':
                        dla[l, :, pi, :] = d_l[base['q_end']]
                    elif pname == 'ans':
                        dla[l, :, pi, :] = d_l[base['ans']]
                    else:
                        dla[l, :, pi, :] = d_l[base['opt_end']].mean(axis=0)
    finally:
        for h in handles:
            h.remove()

    ai = letters.index(q['answer'])
    dis = [i for i in range(len(letters)) if i != ai]
    net = dla[..., ai] - dla[..., dis].mean(axis=-1)                   # [L,H,3]
    return {'L': L, 'H': H, 'T': T, 'mass': mass, 'ent': ent, 'net': net}


def _run_pos(dataset: str, n: int, seed: int) -> None:
    try:
        model, tok, device, meta, _ = _ready()
        letter_ids = latlas._letter_token_ids(tok)
        rows = [q for q in latlas._load_questions() if (q.get('source') or '') == dataset]
        if n > 0 and len(rows) > n:
            rng = np.random.default_rng(seed)
            pick = rng.permutation(len(rows))[:n]
            rows = [rows[i] for i in pick]
        with _LOCK:
            _state.update(job='pos', state='running', error='', done=0, total=len(rows),
                          elapsed_s=0.0, note=f'{dataset}')
            t0 = time.perf_counter()

        accM = accE = accN = accN2 = None
        L = H = None
        ok = 0
        for i, q in enumerate(rows):
            r = _measure_pos(model, tok, device, q, letter_ids)
            if r is None:
                continue
            if accM is None:
                L, H = r['L'], r['H']
                accM = np.zeros_like(r['mass']); accE = np.zeros_like(r['ent'])
                accN = np.zeros_like(r['net']); accN2 = np.zeros_like(r['net'])
            accM += r['mass']; accE += r['ent']
            accN += r['net']; accN2 += r['net'] ** 2
            ok += 1
            with _LOCK:
                _state['done'] = i + 1
                _state['elapsed_s'] = round(time.perf_counter() - t0, 1)
        if not ok:
            raise RuntimeError('没有一道题能对齐 token 边界，无法测量。')

        mean = accN / ok
        var = np.maximum(accN2 / ok - mean ** 2, 0.0)
        sd = np.sqrt(var * ok / max(ok - 1, 1))
        tval = mean / (sd / math.sqrt(ok) + 1e-12)

        _write_npz(POS_NPZ,
                   attn_mass=(accM / ok).astype(np.float32),
                   attn_ent=(accE / ok).astype(np.float32),
                   dla_net_mean=mean.astype(np.float32),
                   dla_net_t=tval.astype(np.float32),
                   n=np.array([ok]))
        # 摘要：每个读出位按 |t| 排前 15 的头
        top = {}
        for pi, pname in enumerate(_POS_KEYS):
            lst = [{'block': l, 'head': h, 't': round(float(tval[l, h, pi]), 2),
                    'net': round(float(mean[l, h, pi]), 3),
                    'entropy': round(float(accE[l, h, pi] / ok), 3)}
                   for l in range(L) for h in range(H)]
            lst.sort(key=lambda x: -abs(x['t']))
            top[pname] = lst[:15]
        out = {'schema': 'mech_pos.v1', 'dataset': dataset, 'n': ok, 'n_layers': L, 'n_heads': H,
               'pos_keys': _POS_KEYS, 'seg_keys': _SEG_KEYS,
               'top_heads': top,
               'layer_hist': {p: {str(b): int(sum(1 for l in range(L) for h in range(H)
                                                  if l == b and abs(tval[l, h, pi]) > 4.0))
                                  for b in range(L)} for pi, p in enumerate(_POS_KEYS)},
               'note': 'attn_mass/attn_ent/dla_net_* 全量在 pos_profile.npz；'
                       'seg 的 own/others 在答案位分别表示「无」与「全部选项文本」，'
                       '在选项位表示「本选项」与「其他选项」。'
                       't 为跨题 t（net 未做层内归一化，故与 atlas 的 t 绝对值不同，排序应一致），'
                       '|t|>4 视为显著（1152 头 Bonferroni 口径）。'}
        _write_json(POS_JSON, out)
        with _LOCK:
            _state.update(state='done', note=f'{dataset} n={ok}')
    except Exception as exc:
        with _LOCK:
            _state.update(state='error', error=f'{exc}')
        traceback.print_exc()


# ------------------------------------------------------------------ 端点
@router.post('/heads_build')
def heads_build() -> dict:
    try:
        return build_heads()
    except Exception as exc:
        traceback.print_exc()
        return {'accepted': False, 'reason': f'{exc}'}


@router.get('/heads')
def heads(layer: int = -1, top: int = 20) -> dict:
    if not HEADS_JSON.exists():
        return {'schema': 'mech_heads.v1', 'available': False,
                'reason': '还没建表，先 POST /api/mech/heads_build'}
    try:
        d = json.loads(HEADS_JSON.read_text(encoding='utf-8'))
    except Exception as exc:
        return {'schema': 'mech_heads.v1', 'available': False, 'reason': f'{exc}'}
    hs = d.get('heads') or []
    if layer >= 0:
        hs = [h for h in hs if h['block'] == layer]
    else:
        hs = sorted(hs, key=lambda h: -abs(h.get('w_letter_max') or 0))[:top]
    return {'schema': 'mech_heads.v1', 'available': True, 'model_name': d.get('model_name'),
            'n_layers': d.get('n_layers'), 'n_heads': d.get('n_heads'),
            'null_cos': d.get('null_cos'), 'note': d.get('note'), 'heads': hs}


@router.post('/types_build')
def types_build() -> dict:
    try:
        return build_types()
    except Exception as exc:
        traceback.print_exc()
        return {'accepted': False, 'reason': f'{exc}'}


@router.get('/types')
def types(layer: int = -1) -> dict:
    if not TYPES_JSON.exists():
        return {'schema': 'mech_head_types.v1', 'available': False,
                'reason': '还没建表，先 POST /api/mech/types_build'}
    d = json.loads(TYPES_JSON.read_text(encoding='utf-8'))
    if layer >= 0:
        d = {**d, 'heads': [h for h in d['heads'] if h['block'] == layer]}
    return {**d, 'available': True}


@router.post('/pos_run')
def pos_run(dataset: str = 'C-Eval val', n: int = 0, seed: int = 20261030) -> dict:
    global _THREAD
    with _LOCK:
        if _state['state'] == 'running':
            return {'accepted': False, 'reason': '已有一轮在跑', 'job': _state['job']}
        _state.update(job='pos', state='running', error='', done=0, total=0, elapsed_s=0.0)
    _THREAD = threading.Thread(target=_run_pos, args=(dataset, int(n), int(seed)),
                               daemon=True, name='mech-pos')
    _THREAD.start()
    return {'accepted': True, 'dataset': dataset, 'n': n}


@router.get('/pos')
def pos() -> dict:
    if not POS_JSON.exists():
        return {'schema': 'mech_pos.v1', 'available': False,
                'reason': '还没跑，先 POST /api/mech/pos_run'}
    d = json.loads(POS_JSON.read_text(encoding='utf-8'))
    return {**d, 'available': True}


@router.get('/status')
def status() -> dict:
    with _LOCK:
        d = dict(_state)
    d['schema'] = 'mech_status.v1'
    d['files'] = {p.name: (p.exists() and p.stat().st_size) for p in
                  (HEADS_JSON, HEADS_NPZ, TYPES_JSON, TYPES_NPZ, POS_JSON, POS_NPZ)}
    return d
