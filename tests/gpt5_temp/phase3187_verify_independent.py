# -*- coding: utf-8 -*-
"""Phase 3187 independent verification (disk read-back; verify-only). 15 checks."""
import hashlib
import json
import os
import re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913', 'phase3187',
                   'g5b12_alpha_calibration')
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')

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
check('registry v1.14 sha8 95fe55f5',
      fin['registry_v114_sha8'] == '95fe55f5' and
      sha8_file(os.path.join(RES, 'atlas_registry_v1_14.json')) == '95fe55f5')
check('atlas v1.15 sha8 a29022d5',
      fin['html_sha8'] == 'a29022d5' and
      sha8_file(os.path.join(RES, 'atlas_v1_15.html')) == 'a29022d5')
check('final seal 135a3755 / res 76d308ae',
      fin['seal_sha8'] == '135a3755' and fin['res_sha8'] == '76d308ae')

# 4 res/seal chain (arm anchors substituted: res 10140b94 / seal cb1bce5b)
ARM_RES, ARM_SEAL = '10140b94', 'cb1bce5b'
base = dict(fin, res_sha8=ARM_RES, seal_sha8=ARM_SEAL)
raw = json.dumps(base, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
r8 = hashlib.sha256(raw).hexdigest()[:8]
mid = json.dumps(dict(base, res_sha8=r8), ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
s8 = hashlib.sha256(mid).hexdigest()[:8]
check('final res/seal chain recomputed (arm anchors substituted)',
      r8 == fin['res_sha8'] and s8 == fin['seal_sha8'])

# 5-7 registry content: statement-only append, no anchors/ce/failures change
reg23 = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                                    'rdc_query_construction_20260913',
                                    'phase3186', 'g5b11_second_order',
                                    'atlas_registry_v1_13.json'), encoding='utf-8'))
reg24 = json.load(open(os.path.join(RES, 'atlas_registry_v1_14.json'), encoding='utf-8'))
check('failures unchanged (19) + upgrade_log unchanged (6)',
      len(reg24['failures']) == 19 and len(reg24['upgrade_log']) == 6 and
      json.dumps(reg23['failures'], ensure_ascii=False, sort_keys=True) ==
      json.dumps(reg24['failures'], ensure_ascii=False, sort_keys=True))
f24o = [f for f in reg23['features'] if f['id'] == 'FTR-24'][0]
f24n = [f for f in reg24['features'] if f['id'] == 'FTR-24'][0]
check('FTR-24 statement append-only + anchors/counter_evidence unchanged',
      f24n['statement'].startswith(f24o['statement']) and
      f24n['anchors'] == f24o['anchors'] and
      f24n['counter_evidence'] == f24o['counter_evidence'] and
      f24n['evidence_level'] == f24o['evidence_level'] == 'E1_repeatable')
check('downgrade note content (direction-level, alpha withdrawn)',
      '方向级陈述' in f24n['statement'] and 'α 撤销' in f24n['statement'] and
      '+0.7602' in f24n['statement'] and '3187' in f24n['statement'])

# 8 other 23 features byte-preserved
check('23 legacy features byte-preserved',
      all(json.dumps(a, ensure_ascii=False, sort_keys=True) ==
          json.dumps(b, ensure_ascii=False, sort_keys=True)
          for a, b in ((f, [x for x in reg24['features'] if x['id'] == f['id']][0])
                       for f in reg23['features'] if f['id'] != 'FTR-24')))

# 9 provenance keys unchanged
check('taxonomy/principles/schema/provenance/created unchanged',
      all(json.dumps(reg23.get(k), ensure_ascii=False, sort_keys=True) ==
          json.dumps(reg24.get(k), ensure_ascii=False, sort_keys=True)
          for k in ('evidence_level_taxonomy', 'principles', 'schema',
                    'models_mainline', 'models_pending', 'provenance', 'created')))

# 10-11 HTML full field check via flatten_v115
import sys
sys.path.insert(0, os.path.join(ROOT, 'tests', 'glm5'))
import phase3179_g5b4_ftr23_registration as p79
import phase3187_g5b12_alpha_calibration as p87
gl15 = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                                   'rdc_query_construction_20260913',
                                   'phase3175', 'g5a12_residual_arms',
                                   'gap_ledger_v1_5.json'), encoding='utf-8'))
nodes = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                                    'rdc_query_construction_20260913',
                                    'phase3162', 'g5a1_atlas_foundation',
                                    'atlas_registry.json'),
                       encoding='utf-8'))['audit']['nodes']
html = open(os.path.join(RES, 'atlas_v1_15.html'), encoding='utf-8').read()
expected = p87.flatten_v115(reg24, nodes, gl15, {'reg24': '95fe55f5'})
mc = re.search(r'data-k="meta\.created">([^<]+)<', html)
assert mc and re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', mc.group(1)), 'created format'
expected['meta.created'] = mc.group(1)
v = p79.verify_html(html, expected)
check('HTML data-k full view (736 fields via flatten_v115) 0/0/0/0',
      not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups'])
check('HTML field count 736/736', fin['field_check']['n_found'] == 736)

# 12-13 ledger
led = json.load(open(LED, encoding='utf-8'))
check('ledger n=339, #339=3187, pre sha 17c4f3e6',
      len(led['measurements']) == 339 and
      led['measurements'][-1]['phase'] == 3187 and
      sha8_file(LED) == 'da29d7d6')
bak = json.load(open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                                  'atlas_ledger_backup_pre_phase3187.json'),
                     encoding='utf-8'))
check('ledger prefix equivalence vs backup',
      json.dumps(bak['measurements'], ensure_ascii=False, sort_keys=True) ==
      json.dumps(led['measurements'][:338], ensure_ascii=False, sort_keys=True))

# 14 MEMO
mt = open(MEMO, encoding='utf-8').read()
sec = mt[mt.rfind('## Phase 3187'):]
check('MEMO Phase 3187 section (b49c04d3, verdict threefold + prereg 3188)',
      hashlib.sha256(mt.encode('utf-8')).hexdigest()[:8] == 'b49c04d3' and
      sec.count('verdict') >= 3 and 'verdict（重复二）' in sec and
      'verdict（重复三）' in sec and '预注册 3188' in sec and
      'FULL-1 crash' in sec and '95fe55f5' in sec)

# 15 smoke/full numeric identity (alpha readings) + arm res cross-run identity
sm = json.load(open(os.path.join(RES, 'smoke_result.json'), encoding='utf-8'))
fu = json.load(open(os.path.join(RES, 'result.json'), encoding='utf-8'))
same = all(
    abs(sm['per_model'][m]['alpha'][r]['alpha_hat'] -
        fu['per_model'][m]['alpha'][r]['alpha_hat']) == 0 and
    abs(sm['per_model'][m]['alpha'][r]['explained_rate'] -
        fu['per_model'][m]['alpha'][r]['explained_rate']) == 0
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b') for r in ('isa', 'hasa'))
check('SMOKE/FULL alpha readings byte-identical + verdict UNSTABLE',
      same and 'unstable' in fu['verdict'] and
      fu['res_sha8'] == '10140b94' and fu['seal_sha8'] == 'cb1bce5b')

print('ALL PASS: %d/15 checks' % ok)
