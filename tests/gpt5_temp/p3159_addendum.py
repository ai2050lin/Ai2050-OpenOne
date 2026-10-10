# -*- coding: utf-8 -*-
# 3159 addendum (zero-GPU): 逐层消耗曲线解剖 —— top 方向被动力学消耗的层位与形状
# 从 sealed collect.npz (SHARE (3,NA,ND,NALPHA,NH), KL, BUD) 现场渲染:
#   (1) per-layer share_top 中位曲线 3 臂 -> 消耗层位 L90(top 份额首破 10% 的层)
#   (2) per-anchor: pnorm 与 top 消耗(gain) 的 Pearson; massive 锚(pnorm>50) vs 普通锚
#   (3) bottom 臂 re-emerge 曲线形状(线性 vs 末端跳变: 增量分布)
#   (4) KL 曲线 per-arm(中位 over anchors x dirs)
import json, os
import numpy as np

RDIR = os.path.join(r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913',
                    'phase3159', 'g4p2_equivalence_dynamics')
out = {}
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    z = np.load(os.path.join(RDIR, m, 'collect.npz'))
    r = json.load(open(os.path.join(RDIR, m, 'result.json'), encoding='utf-8'))
    SH = z['SHARE'].astype(np.float64)          # (3, NA, ND, NALPHA, NH)
    KL = z['KL'].astype(np.float64)             # (3, NA, ND, NALPHA)
    alphas = z['alphas']
    l_mid = int(z['l_mid'])
    NH = SH.shape[4]
    NL = NH - 1
    am = json.loads(str(z['anchor_meta']))
    # (1) per-layer median share curve (over anchors/dirs/alphas)
    med_curve = np.median(SH, axis=(1, 2, 3))   # (3, NH)
    top_curve = med_curve[2]
    l90 = None
    for l in range(l_mid, NH):
        if top_curve[l] < 0.10:
            l90 = l
            break
    # 消耗速度: share 从 L_mid+1 到 L90 的每层降幅
    drops = np.diff(top_curve[l_mid:])
    big_drop_layer = int(l_mid + 1 + int(np.argmin(drops))) if len(drops) else None
    # (2) per-anchor gain vs pnorm
    gains = SH[2, :, :, :, NL] .mean(axis=(1, 2)) - SH[2, :, :, :, l_mid].mean(axis=(1, 2))
    pnorms = np.array([a['pnorm'] for a in am])
    corr_pg = float(np.corrcoef(pnorms, gains)[0, 1]) if len(pnorms) > 2 else None
    # (3) bottom re-emerge 曲线增量
    bot_curve = med_curve[0][l_mid:]
    incs = np.diff(bot_curve)
    frac_tail = float(incs[-3:].sum() / (incs.sum() + 1e-18))
    # (4) per-arm KL median curve
    kl_med = np.median(KL, axis=(1, 2))         # (3, NALPHA)
    out[m] = dict(l_mid=l_mid, NH=NH, l90=l90, big_drop_layer=big_drop_layer,
                  share_seq=[round(float(x), 4) for x in top_curve[l_mid::4]],
                  corr_pnorm_gain=corr_pg,
                  gain_massive=float(gains[pnorms > np.median(pnorms)].mean()),
                  gain_normal=float(gains[pnorms <= np.median(pnorms)].mean()),
                  bot_tail_frac=round(frac_tail, 3),
                  bot_gain_per_layer=[round(float(x), 4) for x in np.diff(bot_curve)[::4]],
                  kl_med=[dict(arm=arm, vals=[round(float(x), 4) for x in kl_med[i]])
                          for i, arm in enumerate(('null', 'rand', 'top'))],
                  seal=r['seal_sha8'])

blob = json.dumps(out, ensure_ascii=False, indent=1)
p = os.path.join(RDIR, 'qwen3-4b', 'result_addendum.json')
json.dump(json.loads(blob), open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
import hashlib
print('saved', p, 'sha8=%s' % hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8])
txt = []
for m, d in out.items():
    txt.append('%s: L90=%s big_drop_layer=%s corr_pnorm_gain=%s gain(mass %.1f vs normal %.1f)' % (
        m, d['l90'], d['big_drop_layer'], d['corr_pnorm_gain'], d['gain_massive'], d['gain_normal']))
    txt.append('  top share seq(every 4 layers from L_mid): %s' % d['share_seq'])
    txt.append('  bottom tail_frac=%.3f incs(every 4): %s' % (d['bot_tail_frac'], d['bot_gain_per_layer']))
    txt.append('  kl_med null=%s' % d['kl_med'][0]['vals'])
    txt.append('  kl_med top =%s' % d['kl_med'][2]['vals'])
open(os.path.join(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp', 'p3159_addendum.txt'),
     'w', encoding='utf-8').write('\n'.join(txt))
print('written')
