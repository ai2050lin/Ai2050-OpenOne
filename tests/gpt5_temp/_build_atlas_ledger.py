# -*- coding: utf-8 -*-
"""Build atlas_ledger.json v1 from existing immutable phase artifacts.

Populates axes / blocks / headsets / measurements / growth_curve / linkage
with SHA256-8 of referenced files. Idempotent: rerun rebuilds the same file.
"""
import hashlib
import io
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def find(phase, fname):
    d = os.path.join(BASE, 'phase%d' % phase)
    for root, dirs, files in os.walk(d):
        if fname in files:
            p = os.path.join(root, fname)
            return {'path': os.path.relpath(p, BASE).replace('\\', '/'),
                    'sha256_8': sha8(p)}
    return {'path': None, 'sha256_8': None, 'missing': True}


def ref(phase, fname, npz_key=None):
    r = find(phase, fname)
    r['phase'] = phase
    if npz_key:
        r['key'] = npz_key
    return r


ledger = {
    'format': 'tma-atlas-ledger',
    'version': 1,
    'model': 'qwen3-4b',
    'protocol_family': 'LPF v5.3',
    'axes': [
        {'axis_id': 'class', 'kind': 'taxonomic',
         'words_n': 80, 'labels_n': 10,
         'words_source': ref(2806, 'execution.json'),
         'census_source': ref(2846, 'census_full.npz'),
         'status': 'active'},
        {'axis_id': 'attr', 'kind': 'attributional',
         'words_n': 28, 'labels_n': 10,
         'axes_names': ['size', 'speed', 'temp', 'age', 'weight', 'strength',
                        'brightness', 'moisture', 'height', 'fullness'],
         'census_source': ref(2872, 'attr_census.npz'),
         'geometry_source': ref(2870, 'result.json'),
         'status': 'active'},
        {'axis_id': 'e200', 'kind': 'taxonomic',
         'words_n': 199, 'status': 'quarantined',
         'notes': '2858 G2 offdiag 0.494 > 0.488 gate; census deferred'},
    ],
    'blocks': [
        {'block_id': 'B1_unembed', 'axis_id': 'class',
         'kind': 'unembed_proj', 'shape': [80, 10],
         'src': ref(2867, 'word_coords_v1.npz', 'B1'),
         'notes': 'constructive-circularity caveat: vocab built from unembed '
                  'class centers (declared in phase2867)'},
        {'block_id': 'B2_causal', 'axis_id': 'class',
         'kind': 'causal_spectrum', 'shape': [80, 1152],
         'src': ref(2867, 'word_coords_v1.npz', 'B2')},
        {'block_id': 'B3_mlp', 'axis_id': 'class',
         'kind': 'mlp_response', 'shape': [80, 10],
         'src': ref(2867, 'word_coords_v1.npz', 'B3'),
         'notes': 'g_direct_true from phase2861, L26-35'},
        {'block_id': 'B2s_class43', 'axis_id': 'class',
         'kind': 'causal_spectrum_sparse', 'shape': [80, 43],
         'headsets': 'class43',
         'src': ref(2868, 'growth_v2.npz', 'B2s')},
        {'block_id': 'B_attr_causal', 'axis_id': 'attr',
         'kind': 'causal_spectrum', 'shape': [28, 1152],
         'src': ref(2873, 'growth_axis2.npz', 'X_attr')},
    ],
    'headsets': [
        {'set_id': 'top64_causal', 'definition': 'top-64 mean drop, 80 words',
         'n': 64, 'load_share_pos': 0.516,
         'src': ref(2846, 'census_full.npz'), 'phase': 2846},
        {'set_id': 'class43', 'definition': 'per-word eta2 F-test p<0.01, '
         '80 words (correct tail, 2871 erratum applied)',
         'n': 43, 'src': ref(2868, 'growth_v2.npz', 'sig_mask'),
         'phase': 2868},
        {'set_id': 'stable_top10', 'definition': '2859 S2 bootstrap-stable '
         'front edge', 'n': 10,
         'src': ref(2859, 'result.json'), 'phase': 2859},
        {'set_id': 'triple_core', 'definition': 'heads in >=2 of '
         '{top64, class43, stable_top10}', 'n': 15,
         'members_core4': ['L5H25', 'L23H7', 'L23H29', 'L26H4'],
         'src': ref(2871, 'result.json'), 'phase': 2871},
    ],
    'measurements': [
        {'meas_id': 'M2857_drift', 'type': 'vocab_stability',
         'verdict': 'subset_incompatibility; prototype gradient real',
         'source': ref(2857, 'result.json')},
        {'meas_id': 'M2859_stability', 'type': 'frontedge_stability',
         'verdict': 'S1=false(rank noise)/S2=true/S3=true/S4=true; '
                    'bipolar orthogonality most robust',
         'source': ref(2859, 'result.json')},
        {'meas_id': 'M2861_true_wp', 'type': 'window_response',
         'verdict': 'E2=no_amplification; max|g|=0.947; window line closed',
         'source': ref(2861, 'result.json')},
        {'meas_id': 'M2862_ov', 'type': 'ov_static_vs_causal',
         'verdict': 'M1=ov_uncorrelated(rho=-0.026); top64 pure formers; '
                    'L13H30 OV gain 0.0075 (1/13 of drop)',
         'source': ref(2862, 'result.json')},
        {'meas_id': 'M2863_class_slices', 'type': 'class_mean_signal',
         'verdict': 'J1/J2/J3 all false at 8 words/class (null-corrected)',
         'source': ref(2863, 'result.json')},
        {'meas_id': 'M2864_variance', 'type': 'class_variance_decomp',
         'verdict': 'V1=true(max eta2 0.393)/V3=false(R2 0.122)/'
                    'eta2 axis perpendicular to causal axis',
         'source': ref(2864, 'result.json')},
        {'meas_id': 'M2866_word_coords_v0', 'type': 'word_level_margin',
         'verdict': 'W1=true(margin 0.0976 vs null 0.0170)/W2=true/'
                    'W4=no shared mechanism(min nn-cos 0.144)/W5=true',
         'source': ref(2866, 'result.json')},
        {'meas_id': 'M2867_word_coords_v1', 'type': 'three_block_complement',
         'verdict': 'T1=true(rho<=0.242)/B3 retrieval 0.875 (2.7x B2)/'
                    'B1=1.0 with circularity caveat',
         'source': ref(2867, 'result.json')},
        {'meas_id': 'M2868_naive_fusion', 'type': 'fusion_naive',
         'verdict': 'G1=false(B23 0.7375 < B3 0.875)/G2=true/G3=true '
                    '(43 heads 26.8x compression, acc up 0.425>0.325)',
         'source': ref(2868, 'result.json')},
        {'meas_id': 'M2869_density_fusion', 'type': 'fusion_density',
         'verdict': 'H1=true(alpha*=0.25, acc 0.8875)/H2=monotone in alpha/'
                    'growth_v3=0.10 boundary',
         'source': ref(2869, 'result.json')},
        {'meas_id': 'M2870_attr_geometry', 'type': 'axis_orthogonality',
         'verdict': 'P1=true(max cos 0.0755)/P3=true(0.995 out-of-class)/'
                    'P2 literal false but same-axis pairs 0.46-0.59 = '
                    'axis_structure_confirmed (criterion flaw noted)',
         'source': ref(2870, 'result.json')},
        {'meas_id': 'M2871_triple', 'type': 'headset_enrichment',
         'verdict': 'U1=true(A&B p=3.6e-4, A&C p~0, B&C p=3.0e-4)/'
                    'U2 zero amplifiers/U3 algebraic share invalid '
                    '(negatives); pos-normalized core2=23.1% load',
         'source': ref(2871, 'result.json')},
        {'meas_id': 'M2872_attr_census', 'type': 'axis2_census',
         'verdict': 'X1=false(no shared frontedge)/X2=true(rho 0.055, '
                    'mechanism sides separate)/X3=independent_populations',
         'source': ref(2872, 'result.json')},
        {'meas_id': 'M2873_growth_axis2', 'type': 'growth_point',
         'verdict': 'C1=not_replicated(power-limited, null p95 0.3571)/'
                    'C2=axis_specific(0/3 overlap)/C3=reuse_signal weak',
         'source': ref(2873, 'result.json')},
    ],
    'growth_curve': [
        {'point_id': 'G_axis1_full', 'axis_id': 'class',
         'block': 'B2_causal', 'components': 1152, 'acc': 0.325,
         'phase': 2867},
        {'point_id': 'G_axis1_sparse', 'axis_id': 'class',
         'block': 'B2s_class43', 'components': 43, 'acc': 0.425,
         'phase': 2868},
        {'point_id': 'G_axis1_mlp', 'axis_id': 'class',
         'block': 'B3_mlp', 'components': 10, 'acc': 0.875, 'phase': 2867},
        {'point_id': 'G_axis1_naive_fusion', 'axis_id': 'class',
         'block': 'B23', 'components': 1162, 'acc': 0.7375, 'phase': 2868},
        {'point_id': 'G_axis1_density_fusion', 'axis_id': 'class',
         'block': 'B3+0.25*B2s', 'components': 53, 'acc': 0.8875,
         'phase': 2869},
        {'point_id': 'G_axis2_attr', 'axis_id': 'attr',
         'block': 'B_attr_causal', 'components': 3, 'acc': 0.2143,
         'shared_with_prev': 0, 'new': 3, 'phase': 2873,
         'notes': 'power-limited at 28 words; not a zero-proof'},
    ],
    'linkage': [
        {'link_id': 'L1_l13h30_mlp_indirect',
         'from': {'headset': 'top64_causal', 'head': 'L13H30'},
         'to': {'kind': 'mlp_response', 'block': 'B3_mlp'},
         'evidence': 'G_mlp/G_attn=4.7 per word; e_ff=-0.23 (~30x OV '
                     'static 0.0075); rotation layer word-specific',
         'phase': 2865, 'status': 'confirmed'},
        {'link_id': 'L2_gate_backbone',
         'from': {'kind': 'qk_gate_pool', 'n_heads': 15, 'share': 0.09},
         'to': {'kind': 'backbone_passthrough', 'n_heads': 497,
                'share': 0.91},
         'evidence': 'gate pool explicit modulation vs passive backbone, '
                     '60% compensatory negatives',
         'phase': 2844, 'status': 'confirmed'},
        {'link_id': 'L3_density_gating',
         'from': {'block': 'B3_mlp', 'dim': 10},
         'to': {'block': 'B2_causal', 'dim': 1152},
         'evidence': 'naive concat harmful (-0.1375); density-matched '
                     '1:4 gain +0.0125; carriers stratify by info density',
         'phase': 2869, 'status': 'confirmed'},
        {'link_id': 'L4_axis_independence',
         'from': {'axis': 'class'},
         'to': {'axis': 'attr'},
         'evidence': 'geometry cos<=0.0755 (2870); mechanism rho=0.055 '
                     '(2872); causal overlap 0/3 (2873)',
         'phase': 2873, 'status': 'confirmed'},
    ],
    'errata_ledger': [
        {'phase': 2860, 'corrects': [2853, 2855],
         'note': 'pseudo-residual workpoint mixing; L32-34 gain ~7 artifact'},
        {'phase': 2861, 'corrects': [2853],
         'note': 'true-workpoint re-test: no window amplification'},
        {'phase': 2871, 'corrects': [2864],
         'note': '102-head set was wrong-tail vector; correct = p<0.01 43'},
    ],
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with io.open(OUT, 'w', encoding='utf-8') as f:
    json.dump(ledger, f, indent=2, ensure_ascii=False)
print('WROTE %s' % OUT)
