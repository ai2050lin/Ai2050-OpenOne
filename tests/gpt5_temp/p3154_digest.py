# -*- coding: utf-8 -*-
# p3154_digest.py: 提取三模型+summary result 关键字段 -> digest txt
import json, os, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3154', 'g1p4_mfd_multifactor_disentangle')
out = []
for mode, fname in [('qwen3-4b', 'result.json'), ('qwen3-14b', 'result.json'),
                    ('glm4', 'result.json'), ('summary', 'result_summary.json')]:
    p = os.path.join(RDIR, mode, fname)
    r = json.load(open(p, encoding='utf-8'))
    out.append('== %s ==' % mode)
    out.append('verdict: %s' % r['verdict'])
    out.append('res_sha8=%s seal=%s design=%s runtime=%ss' %
               (r.get('res_sha8'), r.get('seal_sha8'), r.get('design_sha'), r.get('runtime_s')))
    if mode != 'summary':
        out.append('npz_sha8=%s det=%s' % (r.get('npz_sha8'), r.get('determinism_note')))
        out.append('d_cov=%s' % json.dumps(r['d_covariate'], ensure_ascii=False))
        out.append('shares_kout=%s' % json.dumps(r['shares_kout']))
        out.append('shares_kstar=%s' % json.dumps(r['shares_kstar']))
        out.append('shares_emb=%s' % json.dumps(r['shares_emb']))
        cG = r['curve_G']
        top3 = sorted(((float(v), int(k)) for k, v in cG.items()), reverse=True)[:3]
        out.append('curve_G_top3(share,layer)=%s curve_G_argmax=%s' % (top3, r['curve_G_argmax_layer']))
        cL = r['curve_L']
        topL = sorted(((float(v), int(k)) for k, v in cL.items()), reverse=True)[:2]
        out.append('curve_L_top2=%s' % topL)
        iv = r['intervention_kout']
        out.append('iv_kout: stat=%.3f p=%.5f flip=%.3f null=%.3f alpha=%.3f' %
                   (iv['stat'], iv['p_stat'], iv['flip_real'], iv['flip_null_mean'], iv['alpha']))
        ivk = r['intervention_kstar']
        out.append('iv_kstar: stat=%.3f p=%.5f flip=%.3f null=%.3f' %
                   (ivk['stat'], ivk['p_stat'], ivk['flip_real'], ivk['flip_null_mean']))
        out.append('heldout=%s' % json.dumps(r['heldout_kout']))
        out.append('gates=%s' % json.dumps(r['gates']))
    else:
        out.append('fp_pairs=%s' % json.dumps(r['fp_pairs']))
        out.append('fp_min=%s consistent=%s ho_all=%s logic_all=%s' %
                   (r['fp_min_kout'], r['fingerprint_consistent'], r['ho_all_pass'], r['logic_axis_sig_all']))
        out.append('shares_mean_kout=%s' % json.dumps(r['shares_mean_kout']))
        for mn, pm in r['per_model'].items():
            out.append('per %s: argmaxG=%s hoacc=%.3f ivstat=%.3f' %
                       (mn, pm['curve_G_argmax'], pm['heldout_acc'], pm['iv_kout']['stat']))
    out.append('')

# 磁盘 sha8 独立复核
out.append('== disk sha8 (independent re-read) ==')
for mode, fname in [('qwen3-4b', 'result.json'), ('qwen3-14b', 'result.json'),
                    ('glm4', 'result.json'), ('summary', 'result_summary.json')]:
    p = os.path.join(RDIR, mode, fname)
    out.append('%s/%s disk_sha8=%s' % (mode, fname,
               hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]))
for mode in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    npz = os.path.join(RDIR, mode, 'collect.npz')
    out.append('%s/collect.npz disk_sha8=%s' % (mode,
               hashlib.sha256(open(npz, 'rb').read()).hexdigest()[:8]))
dst = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3154_digest.txt')
open(dst, 'w', encoding='utf-8').write('\n'.join(out))
print('written', dst)
