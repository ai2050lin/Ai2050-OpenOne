# -*- coding: utf-8 -*-
"""Phase 3205 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(B + r'\phase3205\g5g3_ftr14_attribution_recheck\result.json', encoding='utf-8'))
smoke = json.load(open(B + r'\phase3205\g5g3_ftr14_attribution_recheck\smoke_result.json', encoding='utf-8'))
fin = json.load(open(B + r'\phase3205\g5g3_ftr14_attribution_recheck\final_result.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('ftr14_attribution_recheck_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE second run clean', smoke['recheck']['recheck_clean'] is True)
chk(4, 'zero_gpu + name', res.get('zero_gpu') is True and res.get('name') == 'g5g3_ftr14_attribution_recheck')
chk(5, 'final content locks', fin['res_sha8'] == '9d25e49b' and fin['seal_sha8'] == 'fbcf7098')

s77 = json.load(open(B + r'\phase3177\g5b2_ftr14_heldout\result.json', encoding='utf-8'))
s78 = json.load(open(B + r'\phase3178\g5b3_ftr14_a2_attribution\result.json', encoding='utf-8'))
PINS_R = {'qwen3-4b': 0.0918, 'qwen3-14b': 0.0951, 'glm4-9b': 0.1289}
A3157 = {'qwen3-4b': 0.5629908442497253, 'qwen3-14b': 0.5614360570907593, 'glm4-9b': 0.5640953183174133}
chk(6, 'K1 r_m 4dp >= 0.05 x3 + locks',
    all(round(s78['arm']['per_model'][m]['r_m'], 4) == v and s78['arm']['per_model'][m]['r_m'] >= 0.05
        for m, v in PINS_R.items())
    and s78['res_sha8'] == '4f521d9b' and s78['seal_sha8'] == '1f166645')
chk(7, 'K1 p_m aggregate bitwise x3',
    all((s78['arm']['per_model'][m]['resid']['isa']['p']
         + s78['arm']['per_model'][m]['resid']['hasa']['p']) / 2.0 ==
        [a for a in [f for f in json.load(open(B + r'\phase3205\g5g3_ftr14_attribution_recheck\atlas_registry_v1_27.json', encoding='utf-8'))['features'] if f['id'] == 'FTR-23'][0]['anchors'] if a['src'] == 'p3178'][0]['asserts']['p_m.%s' % m]
        for m in PINS_R))
chk(8, 'K1 drift 0 + gates',
    all(s78['arm']['per_model'][m]['tc_full_drift'] == 0.0 and s78['arm']['per_model'][m]['exch_drift'] == 0.0
        for m in PINS_R)
    and s78['gates'] == {'calibration_bar': False, 'device_3157': True, 'r_tol': 0.05, 'replication_3177': True}
    and s78['arm']['exch_mean'] == 1.2728174525072664)
chk(9, 'K2 3177 gates + cross-bitwise',
    s77['arm']['gate_a1'] is True and s77['arm']['gate_a2'] is False and s77['arm']['gate_exch_band'] is True
    and all(s77['arm']['per_model'][m]['cos_full_a2'] == s78['arm']['per_model'][m]['cos_full_a2']
            and s77['arm']['per_model'][m]['tc_full'] == A3157[m]
            and s77['arm']['per_model'][m]['exchange_obs'] == s78['arm']['per_model'][m]['exchange_obs']
            for m in PINS_R)
    and s77['res_sha8'] == 'd451030f' and s77['seal_sha8'] == '7b746eb2')
reg27 = json.load(open(B + r'\phase3205\g5g3_ftr14_attribution_recheck\atlas_registry_v1_27.json', encoding='utf-8'))
reg26 = json.load(open(B + r'\phase3204\g5g1_ftr15_heldout_recheck\atlas_registry_v1_26.json', encoding='utf-8'))
f14 = [f for f in reg27['features'] if f['id'] == 'FTR-14'][0]
f14o = [f for f in reg26['features'] if f['id'] == 'FTR-14'][0]
chk(10, 'scope_limits append-only semantics',
    f14['scope_limits'].startswith(f14o['scope_limits'])
    and len(f14['scope_limits']) > len(f14o['scope_limits'])
    and f14['evidence_level'] == f14o['evidence_level'] == 'E1_repeatable'
    and f14['anchors'] == f14o['anchors'] and f14['counter_evidence'] == f14o['counter_evidence'])
chk(11, '25 legacy + upgrade_log + failures preserved',
    all(json.dumps(f, ensure_ascii=False, sort_keys=True) ==
        json.dumps([x for x in reg27['features'] if x['id'] == f['id']][0], ensure_ascii=False, sort_keys=True)
        for f in reg26['features'] if f['id'] != 'FTR-14')
    and json.dumps(reg27['upgrade_log'], sort_keys=True) == json.dumps(reg26['upgrade_log'], sort_keys=True)
    and json.dumps(reg27['failures'], sort_keys=True) == json.dumps(reg26['failures'], sort_keys=True))
chk(12, 'v1.27 version/supersedes + 26 features',
    reg27['version'] == '1.27' and len(reg27['features']) == 26 and '112a46ee' in reg27['supersedes'])
chk(13, 'v1.26/v1.27 sha refs on disk',
    sha8(B + r'\phase3204\g5g1_ftr15_heldout_recheck\atlas_registry_v1_26.json') == '112a46ee'
    and sha8(B + r'\phase3205\g5g3_ftr14_attribution_recheck\atlas_registry_v1_27.json') == 'd5ce1c19')
chk(14, 'atlas_v1_29 sha + 785/785 fields',
    sha8(B + r'\phase3205\g5g3_ftr14_attribution_recheck\atlas_v1_29.html') == 'b2841d0c'
    and fin['field_check'] == {'n_expected': 785, 'n_found': 785,
                               'missing': 0, 'extra': 0, 'mismatch': 0, 'dups': 0})
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3205', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:356]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(15, 'ledger #357 rollback equivalence',
    len(led['measurements']) == 357 and led['measurements'][356]['phase'] == 3205
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(16, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8] == led['ledger_sha256_8'])
e = led['measurements'][356]
chk(17, 'ledger chain refs', e['chain']['registry_v127'] == 'd5ce1c19'
    and e['chain']['atlas_v1_29'] == 'b2841d0c'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8'])
chk(18, 'MEMO + daily + MEMORY updated',
    'Phase 3205 — G5-G3' in open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
    and 'Phase 3205' in open(r'.workbuddy\memory\2026-10-10.md', encoding='utf-8').read()
    and '3206 已预注册' in open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read())

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
