# -*- coding: utf-8 -*-
"""Phase 3223 independent verify (18 checks). Read-only: verifies sealed
artifacts + closeout writes; fixes go to THIS script only, never to sealed
products."""
import hashlib
import json
import os

import numpy as np


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3223'
      r'\g5g25_presoftmax_band_design')
SRC_DIR = r'tests\glm5\result\rdc_query_construction_20260913'
ROOT = '.'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))
    print('#%-2d %-58s %s' % (i, name, 'PASS' if cond else 'FAIL'))


# artifacts
res = json.load(open(os.path.join(P5, 'result.json'), encoding='utf-8'))
doc = json.load(open(os.path.join(P5, 'presoftmax_band_design_v1.json'),
                     encoding='utf-8'))
raw_doc = open(os.path.join(P5, 'presoftmax_band_design_v1.json'), 'rb').read()
c2 = json.loads(raw_doc.decode('utf-8'))
c2.pop('sha8', None)
content_sha = hashlib.sha256(
    json.dumps(c2, ensure_ascii=False, indent=1, sort_keys=True)
    .encode('utf-8')).hexdigest()[:8]

# 1 verdict
chk(1, 'verdict registered', res['verdict'] == 'presoftmax_band_design_registered')
# 2 D1 anchors
z1 = {m: sha8_file(os.path.join(SRC_DIR, 'phase3154',
                                'g1p4_mfd_multifactor_disentangle', sub,
                                'collect.npz'))
      for m, sub in [('qwen3-4b', 'qwen3-4b'), ('qwen3-14b', 'qwen3-14b'),
                     ('glm4-9b', 'glm4')]}
chk(2, 'D1 npz sha anchors x3 on disk',
    z1 == {'qwen3-4b': 'bb99438f', 'qwen3-14b': '6d142b9d',
           'glm4-9b': 'cf2659f1'})
# 3 D1 independent reread (shape)
H = np.load(os.path.join(SRC_DIR, 'phase3154', 'g1p4_mfd_multifactor_disentangle',
                         'qwen3-4b', 'collect.npz'), allow_pickle=True)['H']
chk(3, 'D1 independent reread H(208,37,2560) float16',
    H.shape == (208, 37, 2560) and str(H.dtype) == 'float16')
# 4 D2 anchors
z2 = {m: sha8_file(h[0]) for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')
      for h in [__import__('glob').glob(os.path.join(
          SRC_DIR, 'phase3169', '*', 'collect_%s.npz' % m))]}
chk(4, 'D2 OOV npz sha anchors x3 on disk',
    z2 == {'qwen3-4b': '36eb4ff0', 'qwen3-14b': '97f98575',
           'glm4-9b': '9cc3f8b4'})
# 5 D3 finding recorded
d3 = doc['data_basis']['w_u_basis']
chk(5, 'D3 W_U finding: statuses recorded honestly',
    d3['qwen3-4b']['status'] == 'no_snapshot_dir'
    and d3['qwen3-14b']['status'] == 'snapshot_incomplete_no_safetensors')
# 6 D4 finding
chk(6, 'D4 w_u_locally_usable False + glm4 absent',
    d3['status_summary']['w_u_locally_usable'] is False
    and d3['status_summary']['glm4_weights_in_hub'] is False)
# 7 design content sha 87e51bcd
chk(7, 'design content sha 87e51bcd (recomputed)',
    content_sha == '87e51bcd' and doc['sha8'] == '87e51bcd')
# 8 file sha 1a1c4978
chk(8, 'design file sha 1a1c4978', sha8_file(
    os.path.join(P5, 'presoftmax_band_design_v1.json')) == '1a1c4978')
# 9 gates verbatim pre-registered
g = doc['gates_pre_registered']
chk(9, 'gates GA/GB/GC pre-registered verbatim',
    '0.70' in g['GA_behavioral_heldout'] and '0.25' in g['GB_band']
    and '0.8' in g['GC_rank_sufficiency']
    and 'BEHAVIORAL_BAND_PASS' in g['verdict_taxonomy'])
# 10 upgrade path + chain
chk(10, 'upgrade path E2_behavioral + chain 3224/3225',
    'E2_behavioral' in doc['upgrade_path']
    and 'upgrade_log #10' in doc['upgrade_path']
    and set(doc['chain'].keys()) == {'3224', '3225'}
    and 'weights acquisition' in doc['chain']['3224'])
# 11 math fields
m = doc['math']
chk(11, 'math: exact displacement + linearized predictor + rmsnorm Jacobian',
    'W_U' in m['true_displacement'] and 'J' in m['linearized_predictor']
    and 'rms' in m['rmsnorm'])
# 12 scope_limits weights precondition
chk(12, 'scope_limits: 3224 weights precondition + glm4 deferred',
    any('NOT LOCALLY AVAILABLE' in s for s in doc['scope_limits'])
    and any('glm4' in s for s in doc['scope_limits']))
# 13 prereg deviation honest
chk(13, 'prereg deviation honestly recorded',
    'user selected' in doc['arm'] and 'standby' in doc['arm'])
# 14 D6 + ledger untouched in result
chk(14, 'result: ledger untouched n=374 a1075729',
    res['ledger_untouched'] == {'n': 374, 'post': 'a1075729'})
# 15 ledger #375
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
e = led['measurements'][374]
chk(15, 'ledger #375 phase/name/verdict prefix',
    len(led['measurements']) == 375 and e['phase'] == 3223
    and e['name'] == 'g5g25_presoftmax_band_design'
    and e['verdict'].startswith('presoftmax_band_design_registered')
    and e['chain']['design_sha8'] == '87e51bcd')
# 16 rollback-equality
bako = json.loads(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3223',
                       'rb').read().decode('utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:374]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
chk(16, 'rollback-equality bak_3223 vs truncated head',
    eq and len(bako['measurements']) == 374
    and bako['ledger_sha256_8'] == 'a1075729')
# 17 MEMO section + honest log + prereg
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
chk(17, 'MEMO 3223 section + honest log + prereg 3224',
    '## Phase 3223 — G5-G25' in memo
    and 'run2 DIRTY(3)' in memo and '跨盘' in memo
    and '### 预注册 Phase 3224（两支候选）' in memo
    and '87e51bcd' in memo and '#375' in memo)
# 18 daily + MEMORY
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily 3223 guard + MEMORY 3224 prereg + 3223 row',
    '### Phase 3223' in daily and '3224 已预注册' in mem
    and '87e51bcd' in mem and '5111392c' in mem)

n_fail = sum(1 for _, _, c in ok if not c)
print('RECHECK: %s (%d/18 PASS)' % ('CLEAN' if n_fail == 0 else 'FAIL',
                                    18 - n_fail))
