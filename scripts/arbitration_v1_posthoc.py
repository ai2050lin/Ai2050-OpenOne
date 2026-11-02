# -*- coding: utf-8 -*-
"""CF-v2 事后诊断（看到结果后设计，不属于预注册）：按提问范围分开拟合仲裁规律，报告确认集 R²（预注册的合并 R² 主要由范围主效应贡献）。
用法：python scripts/arbitration_v1_posthoc.py；输出 research/arbitration/CF-v2-qwen3-4b/diagnostic_posthoc.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import arbitration_v1 as CF  # noqa: E402
from shared.dist_protocol import arbitration as A  # noqa: E402

key = '6be1034f0b55a6df-bf16'
z = np.load(CF.out_dir(key) / 'data.npz')
conds = [tuple(c) for c in json.loads(CF.PREREG.read_text(encoding='utf-8'))['conditions']]
out = {'note': '事后诊断（看到 CF-v2 结果后设计），不属于预注册', 'by_scope': {}}
for lang in ('zh', 'en'):
    Sh = dict(zip(('disc', 'conf'), CF.s_hat(key, lang)))
    D = {}
    for s in ('disc', 'conf'):
        LO, PK = z[f'{s}_{lang}_LO'], z[f'{s}_{lang}_PK']
        P, K = PK[:, 0].astype(int), PK[:, 1].astype(int)
        D[s] = (LO, PK[:, 2], Sh[s][np.arange(len(P)), P] - Sh[s][np.arange(len(P)), K])
    for S, name in ((0, 'world'), (1, 'book')):
        js = [j for j, (src, r, SS) in enumerate(conds) if SS == S and r > 0]

        def mk(s, which):
            LO, sv, sh = D[s]
            X = [[1, src == 'ency', np.log(r), (sv if which == 's' else sh)[i]] for j in js for (src, r, _) in [conds[j]] for i in range(len(LO))]
            y = [LO[i, j] for j in js for i in range(len(LO))]
            return np.array(X, float), np.array(y)
        for which in ('s', 's_hat'):
            Xd, yd = mk('disc', which); Xc, yc = mk('conf', which)
            b, bc = A.fit(Xd, yd), A.fit(Xd[:, :3], yd)
            out['by_scope'][f'{lang}/{name}/{which}'] = {'R2_conf': A.r2(yc, Xc @ b), 'R2_conf_cues_only': A.r2(yc, Xc[:, :3] @ bc),
                                                          'coef': dict(zip(['1', 'src', 'logr', 'strength'], map(float, b)))}
    j = conds.index(('none', 0, 0))
    out[f'{lang}_corr_LO_nocontext_world_vs_s'] = float(np.corrcoef(D['conf'][0][:, j], D['conf'][1])[0, 1])
dst = ROOT / 'research/arbitration/CF-v2-qwen3-4b/diagnostic_posthoc.json'
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
