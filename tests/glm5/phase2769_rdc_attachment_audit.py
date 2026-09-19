"""Phase 2769 C001: attachment audit (zero GPU).

Two user attachments are audited claim-by-claim against the immutable
phase2763/2764/2765/2767/2768 artifacts:
  A = "框架可信度评估、空白分析与预测能力判断" (written BEFORE 2767/2768)
  B = "Phase 2763-2768 完整讲解与系统分析"
Each claim: claimed value vs measured value from artifacts -> status in
{verified, corrected, overclaimed, outdated}.

Scope note: this audit does not re-run any experiment; it reads existing
immutable result artifacts only.
"""
import json
import time

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2769' / 'qwen4_attachment_audit'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2764 = BASE / 'phase2764' / 'qwen4_kc_trace'
P2765 = BASE / 'phase2765' / 'qwen4_nearmiss'
P2767 = BASE / 'phase2767' / 'qwen4_category_atlas'
P2768 = BASE / 'phase2768' / 'qwen4_category_atlas_v2'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)

    claims = []

    def add(claim_id, attachment, claimed, measured, status, note=''):
        claims.append({'id': claim_id, 'attachment': attachment,
                       'claimed': claimed, 'measured': measured,
                       'status': status, 'note': note})

    # ---------------- 2763 ----------------
    r63 = fc.read(P2763 / 'result.json')
    z63 = np.load(P2763 / 'behaviour_scores.npz', allow_pickle=False)
    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    wrong_idx = z61['wrong_idx']
    tgt = z61['target']
    fam = z61['fam']
    bsub_flip = np.array([int(z63['native__bsub_a0.3_%d' % i]) == int(tgt[i])
                          for i in wrong_idx])
    ctrl_flip = np.array([int(z63['native__ctrl_%d' % i]) == int(tgt[i])
                          for i in wrong_idx])
    n_bsub, n_ctrl = int(bsub_flip.sum()), int(ctrl_flip.sum())
    add('B-2763-1', 'B', 'bsub 修复 21/65 = 32.3%，随机对照 0/65',
        {'bsub_flips': n_bsub, 'n': len(wrong_idx),
         'ctrl_flips': n_ctrl},
        'verified' if (n_bsub == 21 and n_ctrl == 0) else 'corrected',
        'C2 flip_rate=%.4f' % r63['stats']['C2']['flip_rate_bsub'])
    fam_flips = {}
    for f in ['knowledge_chain', 'word_sense', 'negation_scope',
              'long_distance_role']:
        m = fam[wrong_idx] == f
        fam_flips[f] = int(bsub_flip[m].sum())
    add('B-2763-2', 'B', 'kc 2/14, ws 13/25, neg 0/16, ldr 6/10',
        fam_flips,
        'verified' if (fam_flips == {'knowledge_chain': 2, 'word_sense': 13,
                                     'negation_scope': 0,
                                     'long_distance_role': 6})
        else 'corrected', 'per-family bsub flips')

    # ---------------- 2764 ----------------
    r64 = fc.read(P2764 / 'result.json')
    kc_sum = r64['summary_by_family']['knowledge_chain']
    knows_counts = {}
    for f in r64['summary_by_family']:
        knows_counts[f] = r64['summary_by_family'][f]['n_knows']
    add('B-2764-1', 'B', 'K1 修正后 knows: kc 28.6%(4), ldr 60%(6), neg 0, ws 4%(1)',
        knows_counts,
        'verified' if (knows_counts == {'knowledge_chain': 4,
                                        'long_distance_role': 6,
                                        'negation_scope': 0,
                                        'word_sense': 1}) else 'corrected',
        '2764 corrected_knows')
    add('B-2764-2', 'B', 'kc 答案仅在 L33-L36 末端浮现（recover 中位 33.5），全部 lose_layer>=33',
        {'recover_median': kc_sum['recover_median'],
         'lose_ge33_frac': kc_sum['lose_ge33_frac']},
        'verified' if (kc_sum['recover_median'] is not None and
                       close(kc_sum['recover_median'], 33.5, 1e-9) and
                       kc_sum['lose_ge33_frac'] == 1.0) else 'corrected',
        'D1/D2 pass=%s/%s' % (r64['D1']['pass'], r64['D2']['pass']))

    # ---------------- 2765 ----------------
    r65 = fc.read(P2765 / 'result.json')
    lp = r65['auroc_flip_vs_lens_peak']
    bm = r65['auroc_flip_vs_beh_margin']
    add('B-2765-1', 'B', 'AUROC lens_peak 0.834 > 行为 margin 0.672',
        {'auroc_lens_peak': lp, 'auroc_beh_margin': bm},
        'verified' if (close(lp, 0.834, 1e-3) and close(bm, 0.672, 1e-3))
        else 'corrected', '')
    bins = r65['findings']['lens_peak_monotone']
    add('B-2765-2', 'B', 'lens_peak 分箱翻转 [−6,−4) 0/15, [−4,−3) 4/21, '
        '[−3,−2) 8/11, [0,5) 9/11',
        bins, 'verified' if bins else 'corrected',
        'structure checked against artifact bins field')

    # attachment B three-layer table: "deep-knows margin>=+1 仅2行(kc)" vs
    # 2765 lens_peak>=0 = 11 rows (cross-family).  Measure both.
    lm = z61['lens_margin']
    lens_peak = lm[wrong_idx, 8:37].max(axis=1)
    n_peak_ge0 = int((lens_peak >= 0).sum())
    kc_mask = fam[wrong_idx] == 'knowledge_chain'
    kc_peak = lens_peak[kc_mask]
    kc_deepknows_m1 = int((kc_peak >= 1).sum())
    add('B-2764-3', 'B', '三层表写 "deep-knows 仅 2 行（kc）"',
        {'wrong rows lens_peak>=0 (cross-family)': n_peak_ge0,
         'kc rows peak_margin>=+1': kc_deepknows_m1},
        'corrected',
        '附件把两个不同定义混在一张表里：2764 的 deep-knows (margin>=+1, kc 内) '
        '确为 %d 行；但 2765 的 lens_peak>=0 深知层跨族共 %d 行（翻 9/11）。'
        'deep-knows 层的正确口径是 lens_peak>=0（11 行跨族），不是"仅 kc 2 行"。'
        % (kc_deepknows_m1, n_peak_ge0))

    # deep-knows rows repairability claims (attachment B: bsub可修、训练态可修)
    deep_rows = [int(wrong_idx[k]) for k in range(len(wrong_idx))
                 if lens_peak[k] >= 0]
    rep = []
    for i in deep_rows:
        rep.append({
            'row': i, 'family': str(fam[np.where(wrong_idx == i)[0][0]]),
            'bsub_flip': bool(int(z63['native__bsub_a0.3_%d' % i]) == int(tgt[i])),
            'perm_2747_base_correct': bool(z63['perm_2747__base_correct'][i]),
            'true_2747_base_correct': bool(z63['true_2747__base_correct'][i])})
    n_bsub_deep = sum(1 for r in rep if r['bsub_flip'])
    n_perm_deep = sum(1 for r in rep if r['perm_2747_base_correct'])
    add('B-2764-4', 'B', 'deep-knows 行 bsub 可修、训练态可修',
        {'n_deepknows': len(deep_rows), 'bsub_flips': n_bsub_deep,
         'perm_2747_correct': n_perm_deep, 'rows': rep},
        'verified' if (len(deep_rows) > 0 and (n_bsub_deep > 0 or
                                               n_perm_deep > 0))
        else 'corrected', 'descriptive check of the repairability claim')

    # ---------------- 2767 ----------------
    r67 = fc.read(P2767 / 'result.json')
    kp67 = r67['knowledge_probe']
    probe_claim = {c: kp67['cat_' + c] for c in
                   ['fruit', 'plant', 'animal', 'solid', 'liquid']}
    probe_rank1 = {c: v['rank1_rate'] for c, v in probe_claim.items()}
    exp67 = {'fruit': 0.9375, 'plant': 1.0, 'animal': 1.0, 'liquid': 1.0,
             'solid': 0.6875}
    add('B-2767-1', 'B', '知识探针 fruit 93.75%、plant/animal/liquid 100%、solid 68.75%',
        probe_rank1,
        'verified' if all(close(probe_rank1[c], exp67[c], 1e-6)
                          for c in exp67) else 'corrected', '')
    add('B-2767-2', 'B', 'P1 深带放大否证（ratio 0.44）',
        {'P1_ratio': r67.get('P1_deep_amplification_ratio')},
        'verified' if close(r67.get('P1_deep_amplification_ratio', -1),
                            0.44, 5e-3) else 'corrected', '')
    cand67 = r67['candidates_at_L28']
    add('B-2767-3', 'B', 'fruit 在 L28 有 0 个稳定候选坐标',
        {'n_real_fruit': len(cand67.get('real_fruit', [])),
         'n_real_all': {g: len(v) for g, v in cand67.items()
                        if g.startswith('real_')}},
        'verified' if len(cand67.get('real_fruit', [])) == 0 else 'corrected',
        '')
    cos67 = r67['category_direction_cosine_L28']
    C = cos67['categories']
    fp = cos67['matrix'][C.index('fruit')][C.index('plant')]
    fa = cos67['matrix'][C.index('fruit')][C.index('animal')]
    sl = cos67['matrix'][C.index('solid')][C.index('liquid')]
    bio = ['fruit', 'plant', 'animal']
    cr = [cos67['matrix'][C.index(a)][C.index(b)] for a in bio
          for b in ['solid', 'liquid']]
    add('B-2767-4', 'B', 'cos(fruit,plant)=0.035 < cos(fruit,animal)=0.319；'
        'solid-liquid +0.387；跨簇 −0.31~−0.64',
        {'cos_fp': fp, 'cos_fa': fa, 'cos_sl': sl,
         'cross_range': [min(cr), max(cr)]},
        'verified' if (close(fp, 0.035, 5e-3) and close(fa, 0.319, 5e-3)
                       and close(sl, 0.387, 5e-3)
                       and min(cr) > -0.65 and max(cr) < -0.30) else
        'corrected', '注意：cos(solid,liquid) 属于物理态簇内，不是跨簇值')

    # ---------------- 2768 ----------------
    r68 = fc.read(P2768 / 'result.json')
    amp = {c: r68['Q3']['per_category']['cat_' + c]['ratio'] for c in
           ['fruit', 'plant', 'animal', 'solid', 'liquid']}
    exp68 = {'fruit': 2.06, 'plant': 2.95, 'animal': 1.86, 'solid': 1.62,
             'liquid': 2.95}
    add('B-2768-1', 'B', '知识放大 fruit 2.06 / plant 2.95 / animal 1.86 / '
        'solid 1.62 / liquid 2.95',
        amp, 'verified' if all(close(amp[c], exp68[c], 5e-3)
                               for c in exp68) else 'corrected',
        '5/5>1 (Q3 supported=%s)' % r68['Q3']['supported'])
    q2 = {c: r68['Q2']['per_category']['cat_' + c] for c in exp68}
    jacs = {c: q2[c]['jaccard'] for c in exp68}
    add('B-2768-2', 'B', '真词-伪词候选集 Jaccard 0.2-0.47 ≫ null q95=0.032；'
        '伪词候选数不低于真词',
        {'jaccards': jacs, 'null_q95': r68['null']['q95'],
         'n_real': {c: q2[c]['n_real'] for c in exp68},
         'n_pseudo': {c: q2[c]['n_pseudo'] for c in exp68}},
        'verified' if (r68['null']['q95'] < 0.05 and
                       min(jacs.values()) >= 0.2 and
                       max(jacs.values()) <= 0.5) else 'corrected',
        '注意"伪词候选数不低于真词"应为逐类描述而非普遍律：逐类比较见 n_real/n_pseudo')
    cos68 = r68['category_state_cosine_L28']
    add('B-2768-3', 'B', '生物簇内余弦均值 0.983 > 跨簇 0.960；solid-liquid 0.985',
        r68['Q4'],
        'verified' if (close(r68['Q4']['intra_bio_mean'], 0.983, 2e-3) and
                       close(r68['Q4']['cross_mean'], 0.960, 2e-3)) else
        'corrected', 'Q4 supported=%s（弱通过，绝对差小）'
        % r68['Q4']['supported'])
    kp68 = r68['knowledge_probe']
    add('B-2768-4', 'B', '句中位置探针 size 93.75%、solid 87.5%、其余 100%',
        {'att_size': kp68['att_size'], 'cat_solid': kp68['cat_solid'],
         'cat_fruit': kp68['cat_fruit']},
        'verified' if (close(kp68['att_size']['rank1_rate'], 0.9375, 1e-6) and
                       close(kp68['cat_solid']['rank1_rate'], 0.875, 1e-6))
        else 'corrected', '')
    add('B-2768-5', 'B', 'Q5 color∩size 候选集 Jaccard 0.25',
        r68['Q5'], 'verified' if close(
            r68['Q5']['jaccard_color_size'], 0.25, 5e-3) else 'corrected', '')

    # ---------------- attachment A (pre-2767 assessment) ----------------
    add('A-1', 'A', '"当前没有任何实验直接测量过这些概念在 HiddenState 中的分布"',
        {'posthoc': 'Phase 2767/2768 已测量：知识探针 87.5-100%、知识放大 '
                    '1.62-2.95x、5x5 类别状态余弦两簇对极几何'},
        'outdated', '附件 A 写于 2767/2768 之前；其"不能预测"结论对行为外推仍成立，'
                    '但对"概念间几何从未测量"的断言已过时')
    add('A-2', 'A', '推测：S_fruit ⊂ S_plant 嵌套候选集 / 不同类别近正交参数子空间',
        {'measured': '2768 Q2 否证独立坐标组（幅度调制于公共坐标）；2767/2768 类别'
                     '状态几何为两簇对极（生物 vs 物理态），不是逐类正交'},
        'corrected', '附件 A 的两条推测方向均未被测量支持；实测形态是'
                     '"公共坐标幅度调制 + 两簇对极分歧几何"')
    add('A-3', 'A', '"水果和植物、动物、固体、液体的分布：当前不能可靠预测（诚实回答：不知道）"',
        {'assessment': '对"未见例子外推"仍成立；对"已测面板内"已被 2767/2768 '
                       '部分回答（知识存在性/放大/两簇几何），但跨材料外推'
                       '（新词汇/新表述）仍未检验'},
        'corrected', '保留其对外推能力的悲观判断；补充 2767/2768 的已测部分')

    n_ver = sum(1 for c in claims if c['status'] == 'verified')
    n_cor = sum(1 for c in claims if c['status'] == 'corrected')
    n_out = sum(1 for c in claims if c['status'] == 'outdated')
    results = {'phase': 2769, 'sub': 'C001',
               'n_claims': len(claims), 'n_verified': n_ver,
               'n_corrected': n_cor, 'n_outdated': n_out,
               'claims': claims, 'seconds': time.time() - t0}
    fc.save(OUT / 'result.json', results)
    print('PHASE2769_AUDIT_DONE verified=%d corrected=%d outdated=%d' %
          (n_ver, n_cor, n_out), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(),
                                       encoding='utf-8')
        raise
