# -*- coding: utf-8 -*-
"""AM-v1 事后诊断（看到结果后设计，不属于预注册）：属性任务族的图谱预测是否只是「过度自信」——
在发现集交叉拟合预测上拟合一个温度 τ（预测 / τ，使发现集 KL 最小），再用于确认集。
用法：python scripts/g5_assembly_v1_posthoc.py；输出 research/assembly/AM-v1-qwen3-4b/diagnostic_posthoc.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from shared.dist_protocol import assembly as S  # noqa: E402
import entity_atlas_run as V1  # noqa: E402
import entity_atlas_v2 as V2  # noqa: E402
import g5_assembly_v1 as AM  # noqa: E402

key = '6be1034f0b55a6df-bf16'
v2rep = json.loads((ROOT / 'research/entity_atlas/EA-v2-qwen3-4b/report.json').read_text(encoding='utf-8'))['G2_confirm']
z2 = np.load(V2.out_dir(key) / 'data.npz')
v1 = np.load(V1.native_dir(key) / 'codes.npz')
taus = np.exp(np.linspace(np.log(0.25), np.log(8), 41))
out = {'note': '事后诊断（看到 AM-v1 结果后设计），不属于预注册；温度只在发现集上选', 'families': {}}
for lang in ('zh', 'en'):
    Xd, Xc = v1[lang].astype(np.float64), z2[f'conf_{lang}_codes'].astype(np.float64)
    for attr in AM.ATTRS:
        Yd, Yc = z2[f'disc_{lang}_{attr}'].astype(float), z2[f'conf_{lang}_{attr}'].astype(float)
        l, lam = v2rep[attr][lang]['layer'], v2rep[attr][lang]['lambda_rel']
        Pd, Pc = AM.ridge_predictions(Xd[:, l], Yd, Xc[:, l], lam)
        f = S.kl_categorical if attr == 'color' else S.kl_bernoulli
        kd = [float(f(Yd, Pd / t).mean()) for t in taus]
        t = float(taus[int(np.argmin(kd))])
        u = float(f(Yc, np.zeros_like(Yc)).mean())
        raw, cal = float(f(Yc, Pc).mean()), float(f(Yc, Pc / t).mean())
        out['families'][f'{attr}/{lang}'] = {'tau': t, 'kl_raw': raw, 'kl_calibrated': cal, 'uniform': u,
                                             'recovered_raw': 1 - raw / u, 'recovered_calibrated': 1 - cal / u}
        print(attr, lang, {k: round(v, 3) for k, v in out['families'][f'{attr}/{lang}'].items()})
dst = ROOT / 'research/assembly/AM-v1-qwen3-4b/diagnostic_posthoc.json'
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
