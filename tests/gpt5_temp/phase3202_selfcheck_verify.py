# -*- coding: utf-8 -*-
"""Phase 3202 self-check (disk read-back) + independent verify (18 checks)."""
import hashlib
import io
import json
import re

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

BASE = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase3202/g5f1_ftr04_recheck'
B324 = (r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
        r'\phase3201\g5e3_eline_audit')
RP = BASE + r'\atlas_registry_v1_25.json'
HP = BASE + r'\atlas_v1_27.html'
LP = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/atlas/atlas_ledger.json'
MP = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'

res = []
def chk(i, name, cond, info=''):
    res.append((i, name, bool(cond), info))

r25 = json.load(io.open(RP, encoding='utf-8'))
r24 = json.load(io.open(B324 + r'\atlas_registry_v1_24.json', encoding='utf-8'))

# 1-4 sealed shas + verdict
chk(1, 'registry_v125_file_sha', sha8(RP) == '3bb33a24', sha8(RP))
chk(2, 'atlas_v1_27_file_sha', sha8(HP) == '739faeb7', sha8(HP))
fr = json.load(io.open(BASE + r'\final_result.json', encoding='utf-8'))
chk(3, 'final_phase_3202_registered', fr.get('phase') == 3202 and
    fr.get('arm', {}).get('recheck_clean') is True)
chk(4, 'final_decomposition', abs(fr['arm']['decomposition']['loeo_gap'] - 0.1181) < 5e-5
    and abs(fr['arm']['decomposition']['margin_ratio'] - 1.9661) < 5e-5
    and fr['arm']['decomposition']['geometry_band_ok'] is True,
    str(fr.get('arm', {}).get('decomposition')))

# 5-9 recheck re-derived independently from sealed 3181
s81 = json.load(io.open((r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
    r'\phase3181\g5b6_ftr04_glm4_attribution\result.json'), encoding='utf-8'))
pm = s81['per_model']
chk(5, 'R1_drift0_x3', all(
    round(pm[m]['loeo_acc'], 4) == p[0] and round(pm[m]['mean_margin'], 4) == p[1]
    and round(pm[m]['kappa'], 4) == p[2] for m, p in
    (('qwen3-4b', (0.6019, 2.6614, 48118411.9811)),
     ('qwen3-14b', (0.5981, 1.4164, 46016538.1628)),
     ('glm4-9b', (0.4800, 5.2326, 44953766.4873)))))
chk(6, 'R3_loeo_gap', round(min(pm['qwen3-4b']['loeo_acc'], pm['qwen3-14b']['loeo_acc'])
    - pm['glm4-9b']['loeo_acc'], 4) >= 0.10)
chk(7, 'R3_margin_ratio', round(pm['glm4-9b']['mean_margin']
    / max(pm['qwen3-4b']['mean_margin'], pm['qwen3-14b']['mean_margin']), 4) >= 1.5)
chk(8, 'R3_geometry_band', abs(pm['glm4-9b']['kappa'] / pm['qwen3-4b']['kappa'] - 1) < 0.25
    and abs(pm['glm4-9b']['pr'] / pm['qwen3-4b']['pr'] - 1) < 0.2
    and int(pm['glm4-9b']['e90']) == int(pm['qwen3-4b']['e90']) == 8)
chk(9, 'content_lock_3181', s81['res_sha8'] == '1e95a19d' and s81['seal_sha8'] == '462fd221')

# 10-14 A4 append-only vs v1.24
chk(10, 'version_1_25_n26', r25['version'] == '1.25' and len(r25['features']) == 26)
chk(11, 'v125_only_ftr04_changed',
    sum(1 for a in r24['features'] for b in r25['features']
        if a['id'] == b['id'] and json.dumps(a, sort_keys=True, ensure_ascii=False)
        != json.dumps(b, sort_keys=True, ensure_ascii=False)) == 1)
f04o = [f for f in r24['features'] if f['id'] == 'FTR-04'][0]
f04n = [f for f in r25['features'] if f['id'] == 'FTR-04'][0]
chk(12, 'f04_scope_append_startswith', f04n['scope_limits'].startswith(f04o['scope_limits'])
    and '3202 G5-F1' in f04n['scope_limits'])
chk(13, 'f04_other_fields_equal',
    all(f04n[k] == f04o[k] for k in f04o if k != 'scope_limits')
    and f04n['evidence_level'] == 'E1_repeatable')
chk(14, 'upgrade_log_failures_unchanged',
    json.dumps(r25['upgrade_log'], sort_keys=True, ensure_ascii=False)
    == json.dumps(r24['upgrade_log'], sort_keys=True, ensure_ascii=False)
    and json.dumps(r25['failures'], sort_keys=True, ensure_ascii=False)
    == json.dumps(r24['failures'], sort_keys=True, ensure_ascii=False))

# 15-16 ledger + memo
led = json.load(io.open(LP, encoding='utf-8'))
chk(15, 'ledger_354_prefix_ok', len(led['measurements']) == 354 and
    led['measurements'][-1]['phase'] == 3202 and
    led['measurements'][-1]['chain']['registry_v125'] == '3bb33a24' and
    sha8(LP) == '4074e4bc', sha8(LP))
memo = io.open(MP, encoding='utf-8').read()
chk(16, 'memo_3202_and_3203', '## Phase 3202' in memo and 'G5-F2' in memo and
    sha8(MP) == '2a3a5a8f', sha8(MP))

# 17-18 html data-k integrity
html = io.open(HP, encoding='utf-8').read()
chk(17, 'html_fields_784', html.count('data-k=') == 784, html.count('data-k='))
spans = set(re.findall(r'data-k="([^"]+)"', html))
chk(18, 'html_v125_span', 'meta.registry_v125_sha8' in spans
    and 'meta.p3181_result' in spans
    and 'meta.registry_v120_sha8' not in spans)

fails = [x for x in res if not x[2]]
for i, name, ok, info in res:
    print('#%-2d %-28s %s %s' % (i, name, 'PASS' if ok else 'FAIL', info))
print('self-check + independent verify: %d/%d' % (len(res) - len(fails), len(res)))
assert not fails
