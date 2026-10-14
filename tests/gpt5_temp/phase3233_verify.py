# -*- coding: utf-8 -*-
"""Phase 3233 independent verification (read-only; sealed artifacts untouched)."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = r'tests\glm5\result\rdc_query_construction_20260913'
RD = os.path.join(SRC, 'phase3233', 'g5g30_gap5_upstream_lever_design')

fails = []


def chk(name, ok, got, exp):
    print('%-46s %s  got=%s exp=%s' % (name, 'PASS' if ok else 'FAIL',
                                       got, exp))
    if not ok:
        fails.append(name)


def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


# V1 result identity (hash domain: res WITH res_sha8 set back to None,
# matching the main script which hashed before assigning)
res = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
r1 = dict(res)
r1['res_sha8'] = None
chk('V1a res_sha8 recompute', csha8(r1) == res['res_sha8'],
    csha8(r1), res['res_sha8'])
chk('V1b res_sha8 width 8', len(res['res_sha8']) == 8, len(res['res_sha8']), 8)
chk('V1c verdict', res['verdict'] == 'upstream_lever_design_registered',
    res['verdict'], 'upstream_lever_design_registered')
chk('V1d freeze sha matches result', res['freeze_sha8'] == '7e201c78',
    res['freeze_sha8'], '7e201c78')
chk('V1e design content sha matches result', res['design_sha8'] == 'd74b6484',
    res['design_sha8'], 'd74b6484')

# V2 design doc identity + structure
docp = os.path.join(RD, 'g5g30_upstream_lever_design_v1.json')
doc = json.load(open(docp, encoding='utf-8'))
d1 = dict(doc)
d1.pop('sha8')
chk('V2a doc content sha recompute', csha8(d1) == doc['sha8'],
    csha8(d1), doc['sha8'])
chk('V2b doc file sha == result.design_file_sha8',
    s8(docp) == res['design_file_sha8'], s8(docp), res['design_file_sha8'])
chk('V2c doc status', doc['status'] == 'DESIGN_SEALED_NOT_OBSERVED',
    doc['status'], 'DESIGN_SEALED_NOT_OBSERVED')
chk('V2d doc phase/name', (doc['phase'], doc['name']) ==
    ('3233', 'g5g30_gap5_upstream_lever_design'), (doc['phase'], doc['name']),
    ('3233', 'g5g30_gap5_upstream_lever_design'))
chk('V2e candidates == result.candidates',
    sorted(doc['candidate_levers_pre_registered'].keys()) == res['candidates'],
    sorted(doc['candidate_levers_pre_registered'].keys()), res['candidates'])
chk('V2f gates == result.gates',
    sorted(doc['gates_pre_registered'].keys()) == res['gates'],
    sorted(doc['gates_pre_registered'].keys()), res['gates'])
chk('V2g scope_limits count', len(doc['scope_limits']) == 5,
    len(doc['scope_limits']), 5)
chk('V2h honesty_points count', len(doc['honesty_points']) == 3,
    len(doc['honesty_points']), 3)
chk('V2i chain keys', sorted(doc['chain'].keys()) == ['3234', '3235'],
    sorted(doc['chain'].keys()), ['3234', '3235'])

# V3 freeze identity
fz = json.load(open(os.path.join(RD, 'execution.json'), encoding='utf-8'))
f1 = dict(fz)
f1.pop('sha8')
chk('V3a freeze sha recompute', csha8(f1) == fz['sha8'],
    csha8(f1), fz['sha8'])
chk('V3b freeze zero_gpu', 'design-only' in fz['zero_gpu'], fz['zero_gpu'][:20],
    'design-only')
chk('V3c freeze prereg points to 3232 tail',
    'Phase 3232 tail' in fz['prereg'], fz['prereg'][:40], 'Phase 3232 tail')

# V4 data anchors re-verified
for tag, exp in fz['data_basis_pinned']['caches'].items():
    rel = {'p3154_4b': r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-4b\collect.npz',
           'p3154_14b': r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-14b\collect.npz',
           'p3154_glm4': r'phase3154\g1p4_mfd_multifactor_disentangle\glm4\collect.npz',
           'p3169_4b': r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz',
           'p3169_14b': r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz',
           'p3169_glm4': r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'}[tag]
    chk('V4 ' + tag, s8(os.path.join(SRC, rel)) == exp,
        s8(os.path.join(SRC, rel)), exp)

# V5 k1 anchor independent re-read
k1 = json.load(open(os.path.join(
    SRC, r'phase3152\g1p2_tri_model_k1\summary\result_summary.json'),
    encoding='utf-8'))
for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    chk('V5 kstar ' + m, k1['k1_3model'][m]['kstar'] == 3,
        k1['k1_3model'][m]['kstar'], 3)
    chk('V5 NL ' + m, k1['k1_3model'][m]['NL'] == res['k1_anchor'][m]['NL'],
        k1['k1_3model'][m]['NL'], res['k1_anchor'][m]['NL'])
chk('V5 UL_loc band text [1,5]', '[1, 5]' in
    doc['gates_pre_registered']['UL_loc'], '[1, 5]',
    'UL_loc band text')

# V6 chain state untouched
led = json.load(open(os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
                     encoding='utf-8'))
chk('V6a ledger n==384', len(led['measurements']) == 384,
    len(led['measurements']), 384)
chk('V6b ledger post', led['ledger_sha256_8'] == '0e47bfd9',
    led['ledger_sha256_8'], '0e47bfd9')
chk('V6c registry v1.31 untouched',
    s8(os.path.join(SRC, r'phase3231\g5g28_registry_decision'
                    r'\atlas_registry_v1_31.json')) == 'ef5189e7',
    s8(os.path.join(SRC, r'phase3231\g5g28_registry_decision'
                    r'\atlas_registry_v1_31.json')), 'ef5189e7')
chk('V6d gap v1.10 untouched',
    s8(os.path.join(SRC, r'phase3232\g5g29_gd_amp_class_robustness'
                    r'\gap_ledger_v1_10.json')) == '378de1cd',
    s8(os.path.join(SRC, r'phase3232\g5g29_gd_amp_class_robustness'
                    r'\gap_ledger_v1_10.json')), '378de1cd')

# V7 SMOKE artifacts agree with FULL on frozen content
sm = json.load(open(os.path.join(RD, 'smoke_result.json'), encoding='utf-8'))
chk('V7a smoke freeze == full freeze', sm['freeze_sha8'] == res['freeze_sha8'],
    sm['freeze_sha8'], res['freeze_sha8'])
chk('V7b smoke design_sha == full', sm['design_sha8'] == res['design_sha8'],
    sm['design_sha8'], res['design_sha8'])
smz = json.load(open(os.path.join(RD, 'smoke_execution.json'),
                     encoding='utf-8'))
chk('V7c smoke freeze sha == full freeze sha', smz['sha8'] == fz['sha8'],
    smz['sha8'], fz['sha8'])

# V8 no observation leakage (design-only arm must contain no per-model data)
leak_keys = [k for k in res if k.startswith(('qwen', 'glm', 'flip', 'marg'))]
chk('V8 no observation keys in result', leak_keys == [], leak_keys, [])

print()
print('TOTAL FAILS: %d' % len(fails))
