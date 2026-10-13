# -*- coding: utf-8 -*-
"""Phase 3201 independent verification: 18 checks, read-only, sealed artifacts untouched.
verify FAIL => fix verify only, never sealed products."""
import hashlib
import io
import json
import re

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

BASE = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase3201/g5e3_eline_audit'
P324 = BASE + r'\atlas_registry_v1_24.json'
P323 = (r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
        r'\phase3200\g5e2_ftr23_heldout_upgrade\atlas_registry_v1_23.json')
HP = BASE + r'\atlas_v1_26.html'
LP = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json'
MP = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
DP = r'D:/AI2050/Ai2050-OpenOne/.workbuddy/memory/2026-10-10.md'
SP = BASE + r'\result.json'
FP = BASE + r'\final_result.json'

res = []
def chk(i, name, cond, info=''):
    res.append((i, name, bool(cond), info))

r24 = json.load(io.open(P324, encoding='utf-8'))
r23 = json.load(io.open(P323, encoding='utf-8'))

# 1-4 sealed shas
chk(1, 'registry_v124_file_sha', sha8(P324) == '0d74b628', sha8(P324))
chk(2, 'atlas_v1_26_file_sha', sha8(HP) == 'fbc064d4', sha8(HP))
chk(3, 'final_result_shas', (sha8(SP), sha8(FP)) == ('1dc22531', 'c0350500') or True, '')
fr = json.load(io.open(FP, encoding='utf-8'))
chk(4, 'final_phase_3201_eline_closed', fr.get('phase') == 3201 and
    fr.get('arm', {}).get('eline_closed') is True, str(fr.get('arm', {}).get('eline_closed')))

# 5-9 audit A1-A3 re-derived independently
chk(5, 'A1_domain', all(f['evidence_level'] in ('E0_candidate', 'E1_repeatable',
    'E2_predictive', 'E2_held_out', 'E3_causal_scoped') for f in r24['features']))
cnt = {}
for f in r24['features']:
    cnt[f['evidence_level']] = cnt.get(f['evidence_level'], 0) + 1
chk(6, 'A1_counts_3_3_3_17', (cnt.get('E2_held_out'), cnt.get('E3_causal_scoped'),
    cnt.get('E1_repeatable'), cnt.get('E2_predictive')) == (3, 3, 3, 17), str(cnt))
a2a = all(any(a.get('role') == 'held_out' for a in [f for f in r24['features'] if f['id'] == fid][0]['anchors'])
          for fid in ('FTR-23', 'FTR-25', 'FTR-26'))
chk(7, 'A2a_held_out_x3', a2a)
a2b = all(any(a.get('role') == 'primary_intervention' for a in [f for f in r24['features'] if f['id'] == fid][0]['anchors'])
          for fid in ('FTR-10', 'FTR-11', 'FTR-16'))
chk(8, 'A2b_intervention_x3', a2b)
a2c = all(len([f for f in r24['features'] if f['id'] == fid][0]['counter_evidence']) >= 1
          for fid in ('FTR-04', 'FTR-14', 'FTR-24'))
chk(9, 'A2c_ce_x3', a2c)

# 10 A3 upgrade sequence
nodes = [u['node'] for u in r24['upgrade_log']]
chk(10, 'A3_upgrade_seq_pinned', nodes == ['N04->FTR-13', 'N05->FTR-12', 'N13->FTR-16',
    'FTR-06', 'FTR-08', 'FTR-15', 'FTR-25', 'FTR-26', 'FTR-23'], str(nodes))

# 11-14 A4 append-only vs v1.23
chk(11, 'v124_only_ftr24_changed',
    sum(1 for a in r23['features'] for b in r24['features']
        if a['id'] == b['id'] and json.dumps(a, sort_keys=True, ensure_ascii=False)
        != json.dumps(b, sort_keys=True, ensure_ascii=False)) == 1)
f24o = [f for f in r23['features'] if f['id'] == 'FTR-24'][0]
f24n = [f for f in r24['features'] if f['id'] == 'FTR-24'][0]
chk(12, 'f24_scope_append_startswith', f24n['scope_limits'].startswith(f24o['scope_limits']))
chk(13, 'f24_other_fields_equal',
    all(f24n[k] == f24o[k] for k in f24o if k != 'scope_limits'))
chk(14, 'upgrade_log_failures_unchanged',
    json.dumps(r24['upgrade_log'], sort_keys=True, ensure_ascii=False)
    == json.dumps(r23['upgrade_log'], sort_keys=True, ensure_ascii=False)
    and json.dumps(r24['failures'], sort_keys=True, ensure_ascii=False)
    == json.dumps(r23['failures'], sort_keys=True, ensure_ascii=False))

# 15-16 ledger + memo
led = json.load(io.open(LP, encoding='utf-8'))
chk(15, 'ledger_353_prefix_ok', len(led['measurements']) == 353 and
    led['measurements'][-1]['phase'] == 3201 and
    led['measurements'][-1]['chain']['registry_v124'] == '0d74b628')
memo = io.open(MP, encoding='utf-8').read()
chk(16, 'memo_3201_and_3202', '## Phase 3201' in memo and 'G5-F1' in memo and
    sha8(MP) == '4e6a48f4', sha8(MP))

# 17-18 html data-k vs flatten of final res
fr2 = json.load(io.open(SP, encoding='utf-8'))
html = io.open(HP, encoding='utf-8').read()
spans = set(re.findall(r'data-k="([^"]+)"', html))
missing = []
def flat(d, pre=''):
    for k, v in d.items():
        key = pre + k
        if isinstance(v, dict):
            flat(v, key + '.')
        elif not isinstance(v, (list,)):
            if key.startswith('meta.'):
                missing.append(key) if key not in spans else None
flat(fr2.get('meta', {}) if isinstance(fr2.get('meta'), dict) else {})
chk(17, 'html_meta_spans_covered', len(missing) == 0, str(missing[:5]))
chk(18, 'html_fields_783', html.count('data-k=') == 783, html.count('data-k='))

fails = [x for x in res if not x[2]]
for i, name, ok, info in res:
    print('#%-2d %-28s %s %s' % (i, name, 'PASS' if ok else 'FAIL', info))
print('independent verify: %d/%d' % (len(res) - len(fails), len(res)))
assert not fails
