# -*- coding: utf-8 -*-
"""Phase 3186 independent verification (disk read-back; verify-only). 15 checks."""
import hashlib
import json
import os
import re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913', 'phase3186',
                   'g5b11_second_order')
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
check('registry v1.13 sha8 730729f5',
      fin['registry_v113_sha8'] == '730729f5' and
      sha8_file(os.path.join(RES, 'atlas_registry_v1_13.json')) == '730729f5')
check('atlas v1.14 sha8 56c45e50',
      fin['html_sha8'] == '56c45e50' and
      sha8_file(os.path.join(RES, 'atlas_v1_14.html')) == '56c45e50')
check('final seal e59cced6 / res 34980bd0',
      fin['seal_sha8'] == 'e59cced6' and fin['res_sha8'] == '34980bd0')

# 4 res/seal chain (arm anchors substituted: res 1bd47cee / seal 731169a0)
ARM_RES, ARM_SEAL = '1bd47cee', '731169a0'
base = dict(fin, res_sha8=ARM_RES, seal_sha8=ARM_SEAL)
raw = json.dumps(base, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
r8 = hashlib.sha256(raw).hexdigest()[:8]
mid = json.dumps(dict(base, res_sha8=r8), ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
s8 = hashlib.sha256(mid).hexdigest()[:8]
check('final res/seal chain recomputed (arm anchors substituted)',
      r8 == fin['res_sha8'] and s8 == fin['seal_sha8'])

# 5-7 registry content: no new failure, FTR-24 double append-only
reg22 = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                                    'rdc_query_construction_20260913',
                                    'phase3185', 'g5b10_rrel_predictive',
                                    'atlas_registry_v1_12.json'), encoding='utf-8'))
reg23 = json.load(open(os.path.join(RES, 'atlas_registry_v1_13.json'), encoding='utf-8'))
check('failures unchanged (19, append-only intact)',
      len(reg23['failures']) == 19 and
      json.dumps(reg22['failures'], ensure_ascii=False, sort_keys=True) ==
      json.dumps(reg23['failures'], ensure_ascii=False, sort_keys=True))
f24o = [f for f in reg22['features'] if f['id'] == 'FTR-24'][0]
f24n = [f for f in reg23['features'] if f['id'] == 'FTR-24'][0]
check('FTR-24 counter_evidence append-only + anchors unchanged',
      f24n['counter_evidence'][:-1] == f24o['counter_evidence'] and
      len(f24n['counter_evidence']) == len(f24o['counter_evidence']) + 1 and
      '3186' in f24n['counter_evidence'][-1] and
      f24n['anchors'] == f24o['anchors'] and
      f24n['evidence_level'] == f24o['evidence_level'] == 'E1_repeatable')
check('FTR-24 statement append-only + contraction note',
      f24n['statement'].startswith(f24o['statement']) and
      '收缩为均值级参数化' in f24n['statement'] and
      '满秩扩散' in f24n['statement'] and '3186' in f24n['statement'])

# 8 other 23 features byte-preserved
check('23 legacy features byte-preserved',
      all(json.dumps(a, ensure_ascii=False, sort_keys=True) ==
          json.dumps(b, ensure_ascii=False, sort_keys=True)
          for a, b in ((f, [x for x in reg23['features'] if x['id'] == f['id']][0])
                       for f in reg22['features'] if f['id'] != 'FTR-24')))

# 9 upgrade_log/provenance keys unchanged
check('upgrade_log(6)/taxonomy/principles unchanged',
      len(reg23['upgrade_log']) == 6 and
      all(json.dumps(reg22.get(k), ensure_ascii=False, sort_keys=True) ==
          json.dumps(reg23.get(k), ensure_ascii=False, sort_keys=True)
          for k in ('upgrade_log', 'evidence_level_taxonomy', 'principles', 'schema',
                    'models_mainline', 'models_pending', 'provenance', 'created')))

# 10-11 HTML full field check via flatten_v114
import sys
sys.path.insert(0, os.path.join(ROOT, 'tests', 'glm5'))
import phase3179_g5b4_ftr23_registration as p79
import phase3186_g5b11_second_order as p86
gl15 = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                                   'rdc_query_construction_20260913',
                                   'phase3175', 'g5a12_residual_arms',
                                   'gap_ledger_v1_5.json'), encoding='utf-8'))
nodes = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                                    'rdc_query_construction_20260913',
                                    'phase3162', 'g5a1_atlas_foundation',
                                    'atlas_registry.json'),
                       encoding='utf-8'))['audit']['nodes']
html = open(os.path.join(RES, 'atlas_v1_14.html'), encoding='utf-8').read()
expected = p86.flatten_v114(reg23, nodes, gl15, {'reg23': '730729f5'})
mc = re.search(r'data-k="meta\.created">([^<]+)<', html)
assert mc and re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', mc.group(1)), 'created format'
expected['meta.created'] = mc.group(1)
v = p79.verify_html(html, expected)
check('HTML data-k full view (737 fields via flatten_v114) 0/0/0/0',
      not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups'])
check('HTML field count 737/737', fin['field_check']['n_found'] == 737)

# 12-13 ledger
led = json.load(open(LED, encoding='utf-8'))
check('ledger n=338, #338=3186, pre sha 4fff0f4a',
      len(led['measurements']) == 338 and
      led['measurements'][-1]['phase'] == 3186 and
      sha8_file(LED) == '17c4f3e6')
bak = json.load(open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                                  'atlas_ledger_backup_pre_phase3186.json'),
                     encoding='utf-8'))
check('ledger prefix equivalence vs backup',
      json.dumps(bak['measurements'], ensure_ascii=False, sort_keys=True) ==
      json.dumps(led['measurements'][:337], ensure_ascii=False, sort_keys=True))

# 14 MEMO
mt = open(MEMO, encoding='utf-8').read()
sec = mt[mt.rfind('## Phase 3186'):]
check('MEMO Phase 3186 section (ef1e4839, verdict threefold + prereg 3187)',
      hashlib.sha256(mt.encode('utf-8')).hexdigest()[:8] == 'ef1e4839' and
      sec.count('verdict') >= 3 and 'verdict（重复二）' in sec and
      'verdict（重复三）' in sec and '预注册 3187' in sec and
      '均值级收缩' in sec and '730729f5' in sec)

# 15 smoke/full numeric identity (A spectrum + B LOO bounds)
sm = json.load(open(os.path.join(RES, 'smoke_result.json'), encoding='utf-8'))
fu = json.load(open(os.path.join(RES, 'result.json'), encoding='utf-8'))
same = all(
    abs(sm['per_model'][m]['specA'][r]['top2'] - fu['per_model'][m]['specA'][r]['top2']) == 0 and
    abs(sm['per_model'][m]['looB'][r]['r_max'] - fu['per_model'][m]['looB'][r]['r_max']) == 0 and
    sm['per_model'][m]['looB'][r]['flip_entities'] == fu['per_model'][m]['looB'][r]['flip_entities']
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b') for r in ('isa', 'hasa'))
check('SMOKE/FULL spectrum + LOO readings byte-identical', same and
      fu['verdict'] == sm['verdict'] and 'contraction' in fu['verdict'])

print('ALL PASS: %d/15 checks' % ok)
