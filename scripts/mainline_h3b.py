# -*- coding: utf-8 -*-
"""主线 H3b（MA-H3b，Phase 347）：在注意力头上找“小而充分”的范围载体，用经过校准的充分性检验。

为什么：MA-H3（Phase 342–343）把 MLP 计入预算并换成反事实值，这种保留检验在 GPT-2 IOI 上连已知电路也找不回
（Phase 345）。改用在 IOI 上校准过的口径（C2b，Phase 347）：载体只在注意力头上选；非载体头换成同一条
现实范围反事实的激活（cf），MLP 全部保留并重新计算。IOI 上该口径让已知 26 头电路恢复 0.69、随机 ≤ 0.04。

材料  与 MA-H3-v4 相同：MA-H1 discovery 按实体分 fit / eval 两半；问法 v3（check_v3 的排除格不变）。
排序  头的分数取 MA-H3-v4 discover 在 fit 半上的 exact（逐个组件换成反事实后 logit 差的变化，rankings.npz），
      atp、atp_ig 作对比；不重跑 discover。
步骤  prereg   写预注册（判据随文件冻结）
      evalfit  在 eval 半 320 对上：各方法前 k 个头（k 按头数比例）保留 / 移除，3 组随机头、3 组按层匹配随机头
      freeze   按预注册规则冻结载体（方法固定为 exact），写 frozen_h3b.json
      （confirm 在新材料 MA-H1b 上做，另行开封）
用法  python scripts/mainline_h3b.py prereg|evalfit|freeze --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

import arbitration_v1 as CF  # noqa: E402
import mainline_h3 as H3  # noqa: E402
from shared.dist_protocol import mech_bench as B  # noqa: E402

OUT = ROOT / 'research/mainline/H3b'
VERSION = 'v1'
PREREG = OUT / f'MA-H3b-{VERSION}.prereg.json'
FROZEN = OUT / 'frozen_h3b.json'
RANKINGS = ROOT / 'research/mainline/H3/rankings.npz'
L, H = 36, 32
NH = L * H
FRACS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
METHODS = ['exact', 'atp', 'atp_ig']

CRITERIA = {
    'metric': '保留 = 载体头取资料范围（干净句）的激活，其余头换成同一条现实范围反事实的激活，MLP 全部保留并重新计算；'
              '移除 = 只把载体头换成反事实激活。f = (LD − LD_∅) / (LD_全 − LD_∅)，∅ = 全部头换成反事实、MLP 保留',
    'calibration': 'C2b（GPT-2 IOI，同一口径）：已知 26 头电路 0.69、3 组随机 ≤ 0.04、exact 前 26 头 0.72（Phase 347）',
    'ranking': 'MA-H3-v4 discover（fit 半 256 对）的 exact 头分数；atp、atp_ig 只作对比',
    'fracs': FRACS, 'n_evalfit': H3.CRITERIA['n_evalfit'], 'batch': H3.CRITERIA['batch'], 'vram_fraction': H3.CRITERIA['vram_fraction'],
    'freeze_rule': '方法固定为 exact；在 eval 半上平均 f_keep ≥ 0.6 的最小比例；没有则记“≤ 20% 的头不充分”，不冻结载体',
    'confirm_predictions': {
        'P1b_sufficiency': '新材料 MA-H1b 上 f(C) ≥ 0.6，且比 3 组随机与 3 组按层匹配随机头中最好的高 ≥ 0.3，实体 bootstrap 95% 区间下界 > 0',
        'P2b_necessity': '只把 C 换成反事实激活时，f_remove ≤ 0.5',
        'P3b_replication': '|f_MA-H1b(C) − f_eval(C)| ≤ 0.15',
    },
    'power': 'f 曲线不是逐题二值结果，不用 McNemar 口径；按实体 bootstrap（2000 次）报告区间，与 MA-H3 相同',
    'bootstrap': 2000,
}


def prereg_doc():
    return {'id': f'MA-H3b-{VERSION}', 'schema': 'mainline_prereg.v1', 'created': '2026-11-08', 'stage': 'H3b',
            'basis': 'Phase 342–343（MA-H3 载体占一半组件）；Phase 345（GPT-2 IOI 上 H3 的保留检验找不回已知电路，atp_ig 在 Qwen3 上偏重前段）；Phase 347（C2b 校准）',
            'question': H3.Q_CUR, 'materials': 'MA-H1 discovery 的 fit / eval 两半（与 MA-H3-v4 相同）；确认用新实体 MA-H1b，另建另开封',
            'methods': METHODS, 'criteria': CRITERIA,
            'rankings_sha256': hashlib.sha256(RANKINGS.read_bytes()).hexdigest()}


def do_prereg():
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    d = hashlib.sha256(text.encode()).hexdigest()
    PREREG.with_suffix('.sha256').write_text(d + '\n', encoding='utf-8', newline='\n')
    print('prereg', d)


def head_order(z, m):
    return np.argsort(-z[f'score_{m}'][:NH])


def mask(heads, n_nodes, keep=True):
    """keep=True：只有这些头和全部 MLP 取干净激活；keep=False：只把这些头换成反事实。"""
    import torch
    out = torch.zeros(n_nodes, dtype=torch.bool) if keep else torch.ones(n_nodes, dtype=torch.bool)
    idx = torch.as_tensor(np.asarray(sorted(heads), dtype=np.int64))
    if keep:
        if len(idx):
            out[idx] = True
        out[NH:] = True          # MLP 全部保留
    else:
        if len(idx):
            out[idx] = False
    return out


def do_evalfit(model_path):
    import mech_bench_run as R
    _, eng, model, tok = H3.setup(model_path)
    palette, forms = CF.palette_forms()
    _, ev = H3.discovery_pairs(model, tok, palette, forms)
    items = H3.balanced(ev, CRITERIA['n_evalfit'], 'evalfit')
    z = np.load(RANKINGS)
    n, rng = eng.n_nodes, np.random.default_rng(0)
    ks = {f: max(1, int(round(f * NH))) for f in FRACS}
    plan = [('none', mask([], n)), ('allheads', mask(range(NH), n))]
    for f, k in ks.items():
        for m in METHODS:
            top = head_order(z, m)[:k]
            plan += [(f'keep|{m}|{f}', mask(top, n)), (f'remove|{m}|{f}', mask(top, n, keep=False))]
            if m == 'exact':
                layers = np.bincount(top // H, minlength=L)
                for d in range(3):
                    matched = [int(l * H + h) for l in range(L) for h in rng.choice(H, size=layers[l], replace=False)]
                    plan.append((f'keep|matched{d}|{f}', mask(matched, n)))
        for d in range(3):
            plan.append((f'keep|random{d}|{f}', mask(rng.choice(NH, size=k, replace=False), n)))
    t0 = time.time()
    raw = H3.run_plan(R, eng, items, plan, 'h3b-evalfit')
    base, full = raw['none'], raw['allheads']
    fv = lambda key, ix=None: B.faithfulness(raw[key] if ix is None else raw[key][ix], full if ix is None else full[ix], base if ix is None else base[ix])
    clusters = [p['family'] for p in items]
    res = {'n_items': len(items), 'seconds': round(time.time() - t0, 1), 'gate': B.behavior_gate(raw['clean'], raw['corrupt']),
           'ld': {'clean': float(raw['clean'].mean()), 'corrupt': float(raw['corrupt'].mean()), 'all_heads_cf_mlp_kept': float(base.mean()), 'all_heads_kept': float(full.mean())},
           'ks': ks, 'curves': {}}
    for m in METHODS + [f'random{d}' for d in range(3)] + [f'matched{d}' for d in range(3)]:
        entry = {'f_keep': [round(fv(f'keep|{m}|{f}'), 4) for f in FRACS]}
        if m in METHODS:
            entry['f_remove'] = [round(fv(f'remove|{m}|{f}'), 4) for f in FRACS]
            entry['keep_ci'] = [[round(x, 4) for x in H3.boot_ci(raw, lambda ix, key=f'keep|{m}|{f}': fv(key, ix), clusters, 500)] for f in FRACS]
        res['curves'][m] = entry
    (OUT / 'evalfit_h3b.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez(OUT / 'evalfit_h3b_raw.npz', **{k.replace('|', '__'): v for k, v in raw.items()})
    print('h3b evalfit', res['gate'], res['seconds'], 's')
    for m, e in res['curves'].items():
        print(m.ljust(9), 'keep', e['f_keep'], 'remove', e.get('f_remove'))


def do_freeze():
    if FROZEN.exists():
        raise SystemExit('已冻结：不得重选')
    ev = json.loads((OUT / 'evalfit_h3b.json').read_text(encoding='utf-8'))
    fk = ev['curves']['exact']['f_keep']
    hit = [f for f, v in zip(FRACS, fk) if v >= 0.6]
    z = np.load(RANKINGS)
    doc = {'id': 'MA-H3b-frozen', 'created': datetime.now().isoformat(timespec='seconds'), 'method': 'exact', 'reached_0p6': bool(hit),
           'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'rankings_sha256': hashlib.sha256(RANKINGS.read_bytes()).hexdigest(),
           'prediction_rule': CRITERIA['confirm_predictions']}
    if hit:
        f = hit[0]; k = ev['ks'][str(f)]
        heads = [int(i) for i in head_order(z, 'exact')[:k]]
        doc.update({'frac': f, 'k': k, 'f_eval': fk[FRACS.index(f)], 'carrier': heads, 'carrier_names': [f'L{i // H}H{i % H}' for i in heads]})
    else:
        doc.update({'frac': None, 'note': '≤ 20% 的头在该口径下不充分，不冻结载体'})
    text = json.dumps(doc, ensure_ascii=False, indent=1) + '\n'
    FROZEN.write_text(text, encoding='utf-8', newline='\n')
    FROZEN.with_suffix('.sha256').write_text(hashlib.sha256(text.encode()).hexdigest() + '\n', encoding='utf-8', newline='\n')
    print('frozen', {k: v for k, v in doc.items() if k not in ('carrier', 'prediction_rule')})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'evalfit', 'freeze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    {'prereg': do_prereg, 'evalfit': lambda: do_evalfit(a.model), 'freeze': do_freeze}[a.cmd]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
