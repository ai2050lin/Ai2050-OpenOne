# -*- coding: utf-8 -*-
"""Phase 3227 independent verify: 18 checks, re-derives from disk."""
import hashlib
import json
import os

OK = []
BAD = []


def chk(n, name, cond):
    (OK if cond else BAD).append((n, name))
    print('%2d %s %s' % (n, 'PASS' if cond else 'FAIL', name))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def jd_safe(a, b):
    return json.dumps(a, ensure_ascii=False, indent=1, sort_keys=True) == \
        json.dumps(b, ensure_ascii=False, indent=1, sort_keys=True)


def csha8(o):
    return hashlib.sha256(json.dumps(o, ensure_ascii=False, indent=1,
                                     sort_keys=True).encode('utf-8')
                          ).hexdigest()[:8]


P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3227'
      r'\g5g26_e3_intervention_design')
B = r'tests\glm5\result\rdc_query_construction_20260913'
P225 = B + r'\phase3225\g5g25_a1_behavioral_band_compute'
P226 = B + r'\phase3226\g5g25_a2_gate_decision'
LED = r'research\gpt5\atlas\atlas_ledger.json'
MEMO = r'research\gpt5\docs\AGI_GPT5_MEMO.md'

# 1 verdict + res content sha
res = json.load(open(os.path.join(P5, 'result.json'), encoding='utf-8'))
_no = dict(res); _no['res_sha8'] = None  # recorded sha was computed with the key present as None
chk(1, 'verdict e3_intervention_design_registered + content sha f374377b',
    res['verdict'] == 'e3_intervention_design_registered'
    and res['res_sha8'] == csha8(_no))
# 2 freeze before observation
fz = json.load(open(os.path.join(P5, 'execution.json'), encoding='utf-8'))
chk(2, 'freeze 82dbb8b1 sealed before any observation',
    fz['sha8'] == '82dbb8b1' and fz['sha8'] == res['freeze_sha8']
    and 'before any observation' in fz['prereg'])
# 3 design doc sealed, content/file sha
doc = json.load(open(os.path.join(P5, 'e3_intervention_design_v1.json'),
                     encoding='utf-8'))
chk(3, 'design doc a883c4da/32968738, DESIGN_SEALED_NOT_OBSERVED',
    doc['sha8'] == 'a883c4da'
    and sha8_file(os.path.join(P5, 'e3_intervention_design_v1.json')) == '32968738'
    and doc['status'] == 'DESIGN_SEALED_NOT_OBSERVED'
    and res['design_sha8'] == 'a883c4da')
# 4 caches x6 re-verified against independent disk re-read
exp = {'p3154_4b': 'bb99438f', 'p3154_14b': '6d142b9d', 'p3154_glm4': 'cf2659f1',
       'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575', 'p3169_glm4': '9cc3f8b4'}
got = {}
for tag, rel in [
    ('p3154_4b', r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-4b\collect.npz'),
    ('p3154_14b', r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-14b\collect.npz'),
    ('p3154_glm4', r'phase3154\g1p4_mfd_multifactor_disentangle\glm4\collect.npz'),
    ('p3169_4b', r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    ('p3169_14b', r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    ('p3169_glm4', r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz')]:
    got[tag] = sha8_file(os.path.join(B, rel))
chk(4, 'caches x6 independent re-read match design anchors', got == exp)
# 5 A1 inputs untouched (license source)
chk(5, 'A1 sealed npz ac5d4d74 untouched + verdict intact',
    sha8_file(os.path.join(P225, 'a1_behavioral_band.npz')) == 'ac5d4d74'
    and json.load(open(os.path.join(P225, 'result.json'),
                       encoding='utf-8'))['verdict'] == 'a1_behavioral_band_sealed')
# 6 exactness + license wording in math
chk(6, 'math: exact patched logits at any alpha + interpretation license '
       'scoped to gate scale',
    'ground truth for any a' in doc['math']['patched_logits_exact']
    and 'gate scale' in doc['data_basis']['w_u_and_license']
    ['a1_linearity_license'])
# 7-10 gates pre-registered verbatim + taxonomy
g = doc['gates_pre_registered']
chk(7, 'GD_flip threshold 0.15 >=3/4 folds x2 models w/ null controls',
    '0.15' in g['GD_flip'] and '3/4 folds x 2 models' in g['GD_flip']
    and any('random norm-matched' in x for x in
            doc['data_basis']['protocol_operationalization']['nulls']))
chk(8, 'GD_dose monotone alpha {0.5,1,2}', 'monotone' in g['GD_dose']
    and doc['data_basis']['protocol_operationalization']['alphas']
    == [0.5, 1.0, 2.0])
chk(9, 'GD_spec cross-patch <= 1/3 matched', '1/3' in g['GD_spec'])
chk(10, 'GD_amp rank-corr >= 0.6 + n=4 small-n scope recorded',
    '0.6' in g['GD_amp']
    and any('n=4' in x for x in doc['scope_limits']))
# 11 taxonomy: E3_PASS iff three gates; FAIL -> F ledger
chk(11, 'taxonomy E3_PASS iff GD_flip and GD_dose and GD_spec; FAIL->F ledger',
    'E3_PASS iff GD_flip and GD_dose and GD_spec' in g['verdict_taxonomy']
    and 'FAIL' in g['verdict_taxonomy'] and 'F ledger' in g['verdict_taxonomy'])
# 12 E3 honesty: in-protocol boundary verbatim
chk(12, 'E3 taxonomy note: in-protocol only, no full-regeneration claim',
    'in-protocol causality' in g['e3_taxonomy_note']
    and 'no full-regeneration' in g['e3_taxonomy_note']
    and 'bypasses upstream' in g['e3_taxonomy_note']
    and any('no upstream recomputation' in x for x in doc['scope_limits']))
# 13 upgrade targets + chain
chk(13, 'upgrade targets FTR-22 family + FTR-26 -> E3 via #11 registry v1.30',
    'FTR-22' in g['upgrade_targets'] and 'FTR-26' in g['upgrade_targets']
    and '#11' in g['upgrade_targets'] and 'v1.30' in g['upgrade_targets'])
chk(14, 'chain 3228 compute+seal / 3229 gate+decision',
    '3228' in doc['chain'] and '3229' in doc['chain']
    and 'compute' in doc['chain']['3228'] and 'decision' in doc['chain']['3229'])
# 15 prerequisite state intact: 3226 outputs untouched
chk(15, 'registry v1.29 (85b8dc1f) + gap v1.7 (4f45fda0) untouched',
    sha8_file(os.path.join(P226, 'atlas_registry_v1_29.json')) == '85b8dc1f'
    and sha8_file(os.path.join(P226, 'gap_ledger_v1_7.json')) == '4f45fda0')
# 16 ledger #379 with rollback-equal bak
led = json.load(open(LED, encoding='utf-8'))
bak = json.load(open(LED + '.bak_3227', encoding='utf-8'))
e379 = led['measurements'][378]
back = dict(led)
back['measurements'] = led['measurements'][:378]
back['ledger_sha256_8'] = bak['ledger_sha256_8']
chk(16, 'ledger #379 (phase 3227) post f21108e7, bak_3227 rollback-equal',
    len(led['measurements']) == 379 and e379['phase'] == 3227
    and led['ledger_sha256_8'] == 'f21108e7'
    and bak['ledger_sha256_8'] == '62b2d7d6' and jd_safe(bak, back))
# 17 MEMO 3227 section + prereg 3228
memo = open(MEMO, encoding='utf-8').read()
chk(17, 'MEMO 3227 section w/ gates + honest log + prereg 3228',
    '## Phase 3227 — G5-G26' in memo and 'GD_flip' in memo
    and 'glm4' in memo and '预注册 Phase 3228' in memo)
# 18 arm honesty: design-only (no observation fields in res)
chk(18, 'design-only honesty: no gate RESULTS recorded (no observation)',
    all(k not in res for k in ('gate_table', 'flip_rate', 'measured'))
    and res['arm'].startswith('design-only'))



print('RECHECK: %s (%d pass / %d fail)' % (
    'CLEAN' if not BAD else 'DIRTY', len(OK), len(BAD)))
