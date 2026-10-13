# -*- coding: utf-8 -*-
"""Phase 3204 self-check + independent verify (disk re-read; 18 checks).
Verify-only: sealed 3158/3180, gap v1.5/v1.6, v1.25 NOT touched.
"""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(B + r'\phase3204\g5g1_ftr15_heldout_recheck\result.json', encoding='utf-8'))
smoke = json.load(open(B + r'\phase3204\g5g1_ftr15_heldout_recheck\smoke_result.json', encoding='utf-8'))
fin = json.load(open(B + r'\phase3204\g5g1_ftr15_heldout_recheck\final_result.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('ftr15_heldout_recheck_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE first-run clean', smoke['recheck']['recheck_clean'] is True
    and smoke['gates']['n_contradictions'] == 0)
chk(4, 'zero_gpu + name', res.get('zero_gpu') is True and res.get('name') == 'g5g1_ftr15_heldout_recheck')
chk(5, 'final content locks', fin['res_sha8'] == 'f9ee4152' and fin['seal_sha8'] == '59100c6d')

# H1 independent: 3158 ratios bitwise vs pins + locks
s58 = json.load(open(B + r'\phase3158\g4p1_output_equivalence_class\summary\result_summary.json', encoding='utf-8'))
PINS = {'qwen3-4b': 1.0283477199188433, 'qwen3-14b': 1.3316768638966026, 'glm4': 1.2472595295618603}
chk(6, 'H1 3158 ratios bitwise x3 + locks',
    all(s58['ratios'][k] == v for k, v in PINS.items())
    and s58['res_sha8'] == 'fa5cca12' and s58['seal_sha8'] == 'e0c60629')
# H2 independent: 3180 chain
s80 = json.load(open(B + r'\phase3180\g5b5_ftr15_heldout\result.json', encoding='utf-8'))
bands = {'qwen3-4b': 0.0300, 'qwen3-14b': 0.0278, 'glm4': 0.0182}
ncgt1 = {'qwen3-4b': 10, 'qwen3-14b': 16, 'glm4': 15}
h2 = all(
    s80['per_model'][m]['ratio_full_recompute'] == PINS[m]
    and s80['per_model'][m]['drift'] == 0.0
    and round(s80['per_model'][m]['max_band'], 4) == bands[m]
    and s80['per_model'][m]['n_c_gt1'] == ncgt1[m]
    and len(s80['folds_full'][m]) == 16
    and all(f['same_class'] for f in s80['folds_full'][m])
    for m in PINS)
chk(7, 'H2 3180 chain (bitwise/band/folds/drift)', h2)
chk(8, 'H2 gates + content lock',
    s80['gates'] == {'G1_device': True, 'G2_class': 3.0, 'G2_heldout': True, 'G2_log_band': 0.25}
    and s80['res_sha8'] == 'f1689e94' and s80['seal_sha8'] == 'a3c1b53e')
# H3: registry v1.26 anchors + preservation
reg26 = json.load(open(B + r'\phase3204\g5g1_ftr15_heldout_recheck\atlas_registry_v1_26.json', encoding='utf-8'))
reg25 = json.load(open(B + r'\phase3202\g5f1_ftr04_recheck\atlas_registry_v1_25.json', encoding='utf-8'))
f15 = [f for f in reg26['features'] if f['id'] == 'FTR-15'][0]
f15o = [f for f in reg25['features'] if f['id'] == 'FTR-15'][0]
a58 = [a for a in f15['anchors'] if a['src'] == 'p3158'][0]['asserts']
a80 = [a for a in f15['anchors'] if a['src'] == 'p3180'][0]['asserts']
chk(9, 'H3 v1.26 FTR-15 anchors bitwise',
    all(a58['ratios.%s' % k] == v for k, v in PINS.items())
    and all(a80['ratio_full_recompute.%s' % k] == v for k, v in PINS.items())
    and all(a80['max_log_band.%s' % k] == s80['per_model'][k]['max_band'] for k in PINS))
chk(10, 'scope_limits append-only semantics',
    f15['scope_limits'].startswith(f15o['scope_limits'])
    and len(f15['scope_limits']) > len(f15o['scope_limits'])
    and f15['evidence_level'] == f15o['evidence_level'] == 'E2_predictive'
    and f15['statement'] == f15o['statement'] and f15['anchors'] == f15o['anchors'])
chk(11, '25 legacy features + upgrade_log + failures byte-preserved',
    all(json.dumps(f, ensure_ascii=False, sort_keys=True) ==
        json.dumps([x for x in reg26['features'] if x['id'] == f['id']][0],
                   ensure_ascii=False, sort_keys=True)
        for f in reg25['features'] if f['id'] != 'FTR-15')
    and json.dumps(reg26['upgrade_log'], sort_keys=True) == json.dumps(reg25['upgrade_log'], sort_keys=True)
    and json.dumps(reg26['failures'], sort_keys=True) == json.dumps(reg25['failures'], sort_keys=True))
chk(12, 'v1.26 version/supersedes + 26 features',
    reg26['version'] == '1.26' and len(reg26['features']) == 26
    and '3bb33a24' in reg26['supersedes'])
chk(13, 'v1.25/v1.26 sha refs on disk',
    sha8(B + r'\phase3202\g5f1_ftr04_recheck\atlas_registry_v1_25.json') == '3bb33a24'
    and sha8(B + r'\phase3204\g5g1_ftr15_heldout_recheck\atlas_registry_v1_26.json') == '112a46ee')
chk(14, 'atlas_v1_28 sha + 784/784 fields',
    sha8(B + r'\phase3204\g5g1_ftr15_heldout_recheck\atlas_v1_28.html') == 'cfc389a8'
    and fin['field_check'] == {'n_expected': 784, 'n_found': 784,
                               'missing': 0, 'extra': 0, 'mismatch': 0, 'dups': 0})
# ledger
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3204', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:355]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
blob = json.dumps(ledold, ensure_ascii=False, indent=1)
blob = json.dumps(dict(back, measurements=led['measurements'][:355] + [led['measurements'][355]]),
                  ensure_ascii=False, indent=1)
e = led['measurements'][355]
chk(15, 'ledger #356 present + rollback equivalence',
    len(led['measurements']) == 356 and e['phase'] == 3204
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
# post-sha convention: sha(dumps(entry-appended blob with OLD sha inside)) == field
tmp = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
blob2 = json.dumps(tmp, ensure_ascii=False, indent=1)
chk(16, 'ledger post-sha convention', hashlib.sha256(blob2.encode('utf-8')).hexdigest()[:8] == led['ledger_sha256_8'])
chk(17, 'ledger chain refs', e['chain']['registry_v126'] == '112a46ee'
    and e['chain']['atlas_v1_28'] == 'cfc389a8'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8'])
chk(18, 'MEMO + daily + MEMORY updated',
    'Phase 3204 — G5-G1' in open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
    and 'Phase 3204' in open(r'.workbuddy\memory\2026-10-10.md', encoding='utf-8').read()
    and '3205 已预注册' in open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read())

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
