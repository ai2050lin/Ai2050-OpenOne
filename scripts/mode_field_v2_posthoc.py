# -*- coding: utf-8 -*-
"""MF-v2 事后诊断（不进判定）。
1  真正的局部基线（修正 A 的设计错误）：逐层局部窗口允许信息沿层逐级接力（每层后退 3 个位置，27 层可覆盖整段），只是减慢而非阻断远程信息。
   这里改为只把深共用位置 t 之前的最后 w 个 token 输入模型（w = 4, 8, 16），在读出层 r 用同一读出器读风格，与完整上下文比较。
2  开关的选择性：全部前缀位置 6σ 的写入本身就会扰动输出、抬高稀有风格的质量（随机方向可达 +13 nat）。
   选择性 = Δlog M_目标 − 其他非来源风格中最大的 Δlog M；与随机方向的同一量比较。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from shared.dist_protocol import mode_field as F  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import mode_field_v2 as MF2  # noqa: E402
import path_scan_v1 as PS  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402

STY = F.STYLES


def selectivity(model_path, int8, crit):
    _, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    rep = json.loads(MF2.report_path(model_path, int8).read_text(encoding='utf-8'))
    out, n_ok = {}, 0
    for c in range(4):
        vals = np.load(od / f'mf_v2_B_{STY[c]}.npy')
        spec = rep['run']['B']['by_source'][STY[c]]['spec']
        ix = {s: j for j, s in enumerate(spec)}
        tg = [s for s in range(4) if s != c]
        for s in tg:
            others = [o for o in tg if o != s]
            sel = lambda j: float((vals[:, j, s] - vals[:, j][:, others].max(1)).mean())
            op = sel(ix[f'{STY[s]}+6'])
            rnd = [sel(ix[f'rand{j}+6']) for j in range(crit['n_random'])]
            ok = bool(op >= 1.0 and op > max(rnd))
            n_ok += ok
            out[f'{STY[s]}<-{STY[c]}'] = {'selectivity_k6': op, 'random_selectivity_k6': rnd, 'selective': ok}
    return {'combos': out, 'n_selective': n_ok}


def truncation(model_path, int8, crit):
    import torch
    _, key = XM.key_of(model_path, int8)
    z = MF1.load_fields(key)
    v2 = json.loads(MF2.report_path(model_path, int8).read_text(encoding='utf-8'))
    v1 = json.loads(MF1.report_path(model_path, int8).read_text(encoding='utf-8'))
    G = [int(x) for x in z['grid']]
    r = v2['run']['readout_layer']
    y, tok, sp, pos, text = z['style'], z['tok'], z['split'], z['pos'], z['text']
    D = sp == 'disc'
    reader = F.ridge_multi(z['Hg'][D][:, G.index(r)].astype(np.float32), y[D], 4, v1['H1']['lambda_rel'])
    deep = MF2.deep_mask(z, crit['deep_n'])
    sel = np.where((sp == 'conf') & deep & np.isin(y, [1, 2, 3]))[0]
    for k_ in ('Hg', 'Hs', 'M'):
        z.pop(k_)
    tok_, model = PS.load(model_path, int8)
    R = XM.Runner(tok_, model)
    C, ids = MF1.tokenize(tok_)
    out = {}
    for w in (4, 8, 16, None):
        H = []
        for j in range(0, len(sel), 32):
            seqs = []
            for g in sel[j:j + 32]:
                full = ids[int(text[g])][:int(pos[g]) + 1]
                seqs.append(full if w is None else full[-w:])
            st = R.last_states(seqs)[:, r]
            H.append(st)
        S = F.scores(reader, np.concatenate(H).astype(np.float32))
        pa = F.pairwise_acc(S, y[sel], tok[sel])
        accs = {f'{STY[a]}-{STY[b]}': pa[f'{a}-{b}'] for a, b in MF2.PAIRS3}
        out['full' if w is None else str(w)] = {'pair_acc': accs, 'mean_acc': float(np.nanmean(list(accs.values())))}
        print('truncation', w, {k: round(v, 3) for k, v in accs.items()}, flush=True)
    return {'readout_layer': r, 'n_positions': int(len(sel)), 'by_window': out}


def main(model_path='models/hf/qwen3-4b', int8=False):
    crit = json.loads(MF2.PREREG.read_text(encoding='utf-8'))['criteria']
    out = {'note': '事后诊断，不进 MF-v2 判定', 'selectivity': selectivity(model_path, int8, crit)}
    print('selective', out['selectivity']['n_selective'], '/ 12', flush=True)
    out['truncation'] = truncation(model_path, int8, crit)
    MF2.save(model_path, int8, {'posthoc': out})


if __name__ == '__main__':
    main(*(sys.argv[1:2] or []), int8='--int8' in sys.argv)
