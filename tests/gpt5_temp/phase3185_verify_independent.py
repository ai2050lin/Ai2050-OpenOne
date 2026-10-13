# -*- coding: utf-8 -*-
"""Phase 3185 independent verification (disk read-back; verify-only, never
touches sealed artifacts). 15 checks."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913', 'phase3185',
                   'g5b10_rrel_predictive')
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
P80 = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3179_g5b4_ftr23_registration.py'

ok = 0


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def check(name, cond):
    global ok
    assert cond, 'VERIFY FAIL: %s' % name
    ok += 1
    print('PASS %2d: %s' % (ok, name))


fin = json.load(open(os.path.join(RES, 'final_result.json'), encoding='utf-8'))

# 1-3 sha anchors
check('registry v1.12 sha8 269a37c0',
      fin['registry_v112_sha8'] == '269a37c0' and
      sha8_file(os.path.join(RES, 'atlas_registry_v1_12.json')) == '269a37c0')
check('atlas v1.13 sha8 a3bd2206',
      fin['html_sha8'] == 'a3bd2206' and
      sha8_file(os.path.join(RES, 'atlas_v1_13.html')) == 'a3bd2206')
check('final seal 21c67b80 / res 686391c5',
      fin['seal_sha8'] == '21c67b80' and fin['res_sha8'] == '686391c5')

# 4 res/seal internal chain of final_result
# NOTE: final dict embeds the ARM result's res8/seal8 (df958cc9/7559b27b) under
# res_sha8/seal_sha8; f_res8 = sha(dumps(final-as-frozen)); f_seal8 = sha(dumps(
# final-as-frozen with res_sha8 replaced by f_res8)). On disk the two keys are
# overwritten by the seal values, so rebuild by substituting them back.
ARM_RES, ARM_SEAL = 'df958cc9', '7559b27b'
base = dict(fin, res_sha8=ARM_RES, seal_sha8=ARM_SEAL)
raw = json.dumps(base, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
r8 = hashlib.sha256(raw).hexdigest()[:8]
mid = json.dumps(dict(base, res_sha8=r8), ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
s8 = hashlib.sha256(mid).hexdigest()[:8]
check('final res/seal chain recomputed (arm anchors substituted)',
      r8 == fin['res_sha8'] and s8 == fin['seal_sha8'])

# 5-6 registry content: F19 append + 23 legacy byte-preserved
reg21 = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                                    'rdc_query_construction_20260913',
                                    'phase3184', 'g5b9_14b_marginal_loo',
                                    'atlas_registry_v1_11.json'), encoding='utf-8'))
reg22 = json.load(open(os.path.join(RES, 'atlas_registry_v1_12.json'), encoding='utf-8'))
f19 = reg22['failures'][-1]
check('F19 appended (failures 18->19, phase 3185)',
      len(reg22['failures']) == 19 and f19['id'] == 'F19' and f19['phase'] == 3185)
check('failures prefix-preserved (F1-F18 byte-equal)',
      json.dumps(reg21['failures'], ensure_ascii=False, sort_keys=True) ==
      json.dumps(reg22['failures'][:18], ensure_ascii=False, sort_keys=True))

# 7 FTR-24 counter_evidence append-only, E1 unchanged
f24o = [f for f in reg21['features'] if f['id'] == 'FTR-24'][0]
f24n = [f for f in reg22['features'] if f['id'] == 'FTR-24'][0]
check('FTR-24 counter_evidence append-only + E1 unchanged',
      f24n['counter_evidence'][:-1] == f24o['counter_evidence'] and
      len(f24n['counter_evidence']) == len(f24o['counter_evidence']) + 1 and
      '3185' in f24n['counter_evidence'][-1] and
      f24n['evidence_level'] == f24o['evidence_level'] == 'E1_repeatable')

# 8 other 23 features byte-preserved
check('23 legacy features byte-preserved',
      all(json.dumps(a, ensure_ascii=False, sort_keys=True) ==
          json.dumps(b, ensure_ascii=False, sort_keys=True)
          for a, b in ((f, [x for x in reg22['features'] if x['id'] == f['id']][0])
                       for f in reg21['features'] if f['id'] != 'FTR-24')))

# 9 upgrade_log/provenance keys unchanged
check('upgrade_log(6)/taxonomy/principles unchanged',
      len(reg22['upgrade_log']) == 6 and
      all(json.dumps(reg21.get(k), ensure_ascii=False, sort_keys=True) ==
          json.dumps(reg22.get(k), ensure_ascii=False, sort_keys=True)
          for k in ('upgrade_log', 'evidence_level_taxonomy', 'principles', 'schema',
                    'models_mainline', 'models_pending', 'provenance', 'created')))

# 10 HTML data-k full field check: rebuild expectation via 3185 flatten_v113
import sys
sys.path.insert(0, os.path.dirname(P80))
import phase3179_g5b4_ftr23_registration as p79
sys.path.insert(0, os.path.join(ROOT, 'tests', 'glm5'))
import phase3185_g5b10_rrel_predictive as p85
gl15 = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                                   'rdc_query_construction_20260913',
                                   'phase3175', 'g5a12_residual_arms',
                                   'gap_ledger_v1_5.json'), encoding='utf-8'))
nodes = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                                    'rdc_query_construction_20260913',
                                    'phase3162', 'g5a1_atlas_foundation',
                                    'atlas_registry.json'),
                       encoding='utf-8'))['audit']['nodes']
html = open(os.path.join(RES, 'atlas_v1_13.html'), encoding='utf-8').read()
expected = p85.flatten_v113(reg22, nodes, gl15, {'reg22': '269a37c0'})
# meta.created is a render-time stamp: p85.CREATED_STR re-evaluates at import in
# THIS process, so substitute the value frozen in the HTML itself (format-checked)
import re
mc = re.search(r'data-k="meta\.created">([^<]+)<', html)
assert mc and re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', mc.group(1)), 'created format'
expected['meta.created'] = mc.group(1)
v = p79.verify_html(html, expected)
check('HTML data-k full view (736 fields via flatten_v113) 0/0/0/0',
      not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups'])
check('HTML field count 736/736', fin['field_check']['n_found'] == 736)

# 12-13 ledger
led = json.load(open(LED, encoding='utf-8'))
check('ledger n=337, #337=3185, pre sha ad566a5c',
      len(led['measurements']) == 337 and
      led['measurements'][-1]['phase'] == 3185 and
      sha8_file(LED) == '4fff0f4a')
bak = json.load(open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                                  'atlas_ledger_backup_pre_phase3185.json'),
                     encoding='utf-8'))
check('ledger prefix equivalence vs backup',
      json.dumps(bak['measurements'], ensure_ascii=False, sort_keys=True) ==
      json.dumps(led['measurements'][:336], ensure_ascii=False, sort_keys=True))

# 14 MEMO
mt = open(MEMO, encoding='utf-8').read()
sec = mt[mt.rfind('## Phase 3185'):]
check('MEMO Phase 3185 section (ec73b057, verdict threefold + prereg 3186)',
      hashlib.sha256(mt.encode('utf-8')).hexdigest()[:8] == 'ec73b057' and
      sec.count('verdict') >= 3 and 'verdict（重复二）' in sec and
      'verdict（重复三）' in sec and '预注册 3186' in sec and
      'F19' in sec and '269a37c0' in sec)

# 15 smoke/full numeric identity (A/B readings)
sm = json.load(open(os.path.join(RES, 'smoke_result.json'), encoding='utf-8'))
fu = json.load(open(os.path.join(RES, 'result.json'), encoding='utf-8'))
same = all(
    abs(sm['per_model'][m]['predA'][f]['r_A'] - fu['per_model'][m]['predA'][f]['r_A']) == 0 and
    abs(sm['per_model'][m]['predB'][r]['R2_B'] - fu['per_model'][m]['predB'][r]['R2_B']) == 0
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')
    for f in ('isa', 'hasa') for r in ('isa', 'hasa'))
check('SMOKE/FULL per-fold readings byte-identical', same and
      fu['verdict'] == sm['verdict'] and 'rejected' in fu['verdict'])

print('ALL PASS: %d/15 checks' % ok)
