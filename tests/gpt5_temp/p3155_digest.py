# -*- coding: utf-8 -*-
import json, os
RDIR = os.path.join(r'tests\glm5\result\rdc_query_construction_20260913\phase3155',
                    'g2p1_relation_family_operator_separability')
lines = []
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    r = json.load(open(os.path.join(RDIR, m, 'result.json'), encoding='utf-8'))
    ko = r['k2_shares_kout']
    lines.append('%s: k2int=%.4f shares[E=%.3f C=%.3f EintC=%.3f T=%.3f EintT=%.3f CintT=%.3f ECT=%.3f] argmax_k=%d' % (
        m, ko['EintC'], ko['E'], ko['C'], ko['EintC'], ko['T'], ko['EintT'], ko['CintT'], ko['ECT'],
        r['curve_EintC_argmax_layer']))
    ab = r['bidirectional_ablation'][str(r['readout'])]
    ec = [(k, v) for k, v in ab['entity_readout'].items() if '->' in k]
    ew = [v for k, v in ab['entity_readout'].items() if 'within' in k]
    lines.append('  ablation: ent_cross=%.3f (%s) within=%.3f | rel cross=%.3f within=%.3f | pca_keep=%.4f' % (
        sum(v for _, v in ec) / len(ec),
        ' '.join('%s:%.2f' % (k, v) for k, v in ec),
        sum(ew) / len(ew), ab['relation_readout']['cross_entity'],
        ab['relation_readout']['within_entity'], ab['energy_kept']))
    sv = r['subspace_angles'][str(r['readout'])]
    lines.append('  subspace: pairs=%s ent-vs-rel max=%.3f min=%.3f' % (
        {k: round(v['top1'], 3) for k, v in sv['entity_subspace_pairs'].items()},
        sv['entity_vs_relation']['sv_max'], sv['entity_vs_relation']['sv_min']))
    hf = r['heldout_folds'][str(r['readout'])]
    lines.append('  folds(KOUT): %s | gates=%s' % (
        {k: round(v['ratio'], 3) for k, v in hf.items()}, r['gates']))
    lines.append('  det=%s npz=%s res=%s seal=%s runtime=%ss' % (
        r['determinism_note'][:20], r['npz_sha8'], r['res_sha8'], r['seal_sha8'], r['runtime_s']))
s = json.load(open(os.path.join(RDIR, 'summary', 'result_summary.json'), encoding='utf-8'))
lines.append('SUMMARY: %s' % s['verdict'])
lines.append('  fp_pairs=%s' % {k: round(v['pearson_kout'], 4) for k, v in s['fp_pairs'].items()})
lines.append('  kstar_pairs=%s' % {k: round(v['pearson_kstar'], 4) for k, v in s['fp_pairs'].items()})
lines.append('  shares_mean_kout=%s' % {k: round(v, 4) for k, v in s['shares_mean_kout'].items()})
lines.append('  per_model_ho_ratios=%s' % {m: {k: round(vv, 3) for k, vv in pm['ho_fold_ratios'].items()}
                                           for m, pm in s['per_model'].items()})
lines.append('  ent_cross_by_model=%s' % {m: round(pm['ent_cross'], 3) for m, pm in s['per_model'].items()})
lines.append('  rel_cross_by_model=%s' % {m: round(pm['rel_cross'], 3) for m, pm in s['per_model'].items()})
lines.append('  entvsrel_svmax=%s' % {m: round(pm['ent_vs_rel_sv_max'], 3) for m, pm in s['per_model'].items()})
lines.append('  summary_res=%s seal=%s' % (s['res_sha8'], s['seal_sha8']))
open(r'tests\gpt5_temp\p3155_digest.txt', 'w', encoding='utf-8').write(chr(10).join(lines))
print('written')
