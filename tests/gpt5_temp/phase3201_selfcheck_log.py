# -*- coding: utf-8 -*-
"""Phase 3201 self-check: disk read-back of sealed artifacts (post-closeout)."""
import hashlib
import io
import json

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

BASE = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase3201/g5e3_eline_audit'
RP = BASE + r'\atlas_registry_v1_24.json'
HP = BASE + r'\atlas_v1_26.html'
checks = []

reg = json.load(io.open(RP, encoding='utf-8'))
checks.append(('registry_v124_sha8', sha8(RP), '0d74b628'))
checks.append(('version', reg['version'], '1.24'))
checks.append(('n_features', len(reg['features']), 26))
lv = {}
for f in reg['features']:
    lv[f['evidence_level']] = lv.get(f['evidence_level'], 0) + 1
checks.append(('counts', (lv.get('E2_held_out'), lv.get('E3_causal_scoped'),
                          lv.get('E1_repeatable'), lv.get('E2_predictive')),
               (3, 3, 3, 17)))
checks.append(('n_upgrade_log', len(reg['upgrade_log']), 9))
checks.append(('upgrade_seq_tail', reg['upgrade_log'][-1]['node'], 'FTR-23'))
f24 = [f for f in reg['features'] if f['id'] == 'FTR-24'][0]
checks.append(('f24_scope_has_3201', '3201 G5-E3' in f24['scope_limits'], True))
checks.append(('f24_e_level', f24['evidence_level'], 'E1_repeatable'))
f23 = [f for f in reg['features'] if f['id'] == 'FTR-23'][0]
checks.append(('f23_e_level', f23['evidence_level'], 'E2_held_out'))

html = io.open(HP, encoding='utf-8').read()
checks.append(('atlas_v1_26_sha8', sha8(HP), 'fbc064d4'))
import re
n_spans = len(re.findall(r'data-k=', html))
checks.append(('html_datak_count_ge', n_spans >= 783, True))
checks.append(('html_has_v124_span', 'data-k="meta.registry_v124_sha8"' in html, True))
checks.append(('html_has_v120_span', 'data-k="meta.registry_v120_sha8"' in html, True))

LP = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json'
led = json.load(io.open(LP, encoding='utf-8'))
checks.append(('ledger_n', len(led['measurements']), 353))
checks.append(('ledger_last_phase', led['measurements'][-1]['phase'], 3201))
checks.append(('ledger_post', sha8(LP), 'ba3d1bcc'))

MP = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
memo = io.open(MP, encoding='utf-8').read()
checks.append(('memo_sha8', sha8(MP), '4e6a48f4'))
checks.append(('memo_has_3201', '## Phase 3201' in memo, True))
checks.append(('memo_has_3202_prereg', '3202 预注册' in memo, True))

ok = 0
for name, got, want in checks:
    good = got == want
    ok += good
    print('%-24s %-8s got=%s want=%s' % (name, 'PASS' if good else 'FAIL', got, want))
print('self-check: %d/%d' % (ok, len(checks)))
assert ok == len(checks)
