# -*- coding: utf-8 -*-
"""Phase 3206 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(B + r'\phase3206\g5g5_ftr22_chain_recheck\result.json', encoding='utf-8'))
smoke = json.load(open(B + r'\phase3206\g5g5_ftr22_chain_recheck\smoke_result.json', encoding='utf-8'))
fin = json.load(open(B + r'\phase3206\g5g5_ftr22_chain_recheck\final_result.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('ftr22_chain_recheck_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE first run clean', smoke['recheck']['recheck_clean'] is True)
chk(4, 'zero_gpu + name', res.get('zero_gpu') is True and res.get('name') == 'g5g5_ftr22_chain_recheck')
chk(5, 'final content locks', fin['res_sha8'] == '7ad6c7d4' and fin['seal_sha8'] == 'f899b8cd')

s72 = json.load(open(B + r'\phase3172\g5a9_port_calibration\result.json', encoding='utf-8'))
s69 = json.load(open(B + r'\phase3169\g5a6_oov_panel\result.json', encoding='utf-8'))
s71 = json.load(open(B + r'\phase3171\g5a8_collapse_mechanism\result.json', encoding='utf-8'))
PIN_POOLED = {'0': 2.5388114997805062, '1': 2.031459140015479, '2': 1.9429485491548533,
              '4': 1.8766676450829491, '8': 1.800176217907229}
PIN_K8 = {'qwen3-4b': 1.9485993218753288, 'qwen3-14b': 1.7100461949026464,
          'glm4-9b': 1.767394689531845}
chk(6, 'L1 pooled bitwise x5 + overall + locks',
    all(s72['pooled'][k]['ratio'] == v for k, v in PIN_POOLED.items())
    and s72['overall']['ratio_kmax'] == PIN_POOLED['8']
    and s72['overall']['main_cls'] == 'borderline_partial_recovery'
    and s72['res_sha8'] == '463f42c8' and s72['seal_sha8'] == 'cdb85525')
chk(7, 'L1 kcurves.8.ratio bitwise x3',
    all(s72['per_model'][m]['kcurves']['8']['ratio'] == v for m, v in PIN_K8.items()))
chk(8, 'L2 3169 gate bitwise == pooled.0 (k0 drift 0)',
    s69['gate']['ratio_B'] == PIN_POOLED['0']
    and s69['gate']['pooled_E_oov_B'] == s72['pooled']['0']['E_oov']
    and s69['gate']['pooled_E_seen_B'] == s72['pooled']['0']['E_seen']
    and s69['gate']['verdict'] == 'collapse_confirmed_gap4_open'
    and s69['res_sha8'] == '49430a39' and s69['seal_sha8'] == '018c6024')
chk(9, 'L2 3171 ratio_S bitwise x3 + mixed_across_models',
    all(s71['per_model'][m]['slots']['k_main']['ratio_S'] ==
        [0.7511424513908266, 0.7166531333803954, 0.8205301477290912][i]
        for i, m in enumerate(['qwen3-4b', 'qwen3-14b', 'glm4-9b']))
    and s71['overall']['main_cls'] == 'mixed_across_models'
    and s71['res_sha8'] == '6a29201c' and s71['seal_sha8'] == 'cc7ccedd')
reg28 = json.load(open(B + r'\phase3206\g5g5_ftr22_chain_recheck\atlas_registry_v1_28.json', encoding='utf-8'))
reg27 = json.load(open(B + r'\phase3205\g5g3_ftr14_attribution_recheck\atlas_registry_v1_27.json', encoding='utf-8'))
f22 = [f for f in reg28['features'] if f['id'] == 'FTR-22'][0]
f22o = [f for f in reg27['features'] if f['id'] == 'FTR-22'][0]
a72 = [a for a in f22['anchors'] if a['src'] == 'p3172'][0]['asserts']
a69 = [a for a in f22['anchors'] if a['src'] == 'p3169'][0]['asserts']
a71 = [a for a in f22['anchors'] if a['src'] == 'p3171'][0]['asserts']
chk(10, 'L3 FTR-22 three anchors bitwise vs sealed',
    all(a72['pooled.%s.ratio' % k] == v for k, v in PIN_POOLED.items())
    and all(a72['per_model.%s.kcurves.8.ratio' % m] == v for m, v in PIN_K8.items())
    and a69['gate.ratio_B'] == PIN_POOLED['0'] and a69['k0_drift_ratio'] == 0.0
    and a69['k0_drift_E_oov'] == 0.0 and a69['k0_drift_E_seen'] == 0.0
    and all(a71['per_model.%s.slots.k_main.ratio_S' % m] == v
            for m, v in [('qwen3-4b', 0.7511424513908266), ('qwen3-14b', 0.7166531333803954),
                         ('glm4-9b', 0.8205301477290912)]))
chk(11, 'scope_limits append-only semantics',
    f22['scope_limits'].startswith(f22o['scope_limits'])
    and len(f22['scope_limits']) > len(f22o['scope_limits'])
    and f22['evidence_level'] == f22o['evidence_level'] == 'E2_predictive'
    and f22['anchors'] == f22o['anchors'] and f22['counter_evidence'] == f22o['counter_evidence']
    and f22['values'] == f22o['values']
    and 'closed@3175' in f22['scope_limits'] and 'port_residual_dominant' in f22['scope_limits'])
chk(12, '25 legacy + upgrade_log + failures preserved',
    all(json.dumps(f, ensure_ascii=False, sort_keys=True) ==
        json.dumps([x for x in reg28['features'] if x['id'] == f['id']][0], ensure_ascii=False, sort_keys=True)
        for f in reg27['features'] if f['id'] != 'FTR-22')
    and json.dumps(reg28['upgrade_log'], sort_keys=True) == json.dumps(reg27['upgrade_log'], sort_keys=True)
    and json.dumps(reg28['failures'], sort_keys=True) == json.dumps(reg27['failures'], sort_keys=True))
chk(13, 'v1.28 version/supersedes + 26 features + sha refs on disk',
    reg28['version'] == '1.28' and len(reg28['features']) == 26
    and 'd5ce1c19' in reg28['supersedes']
    and sha8(B + r'\phase3205\g5g3_ftr14_attribution_recheck\atlas_registry_v1_27.json') == 'd5ce1c19'
    and sha8(B + r'\phase3206\g5g5_ftr22_chain_recheck\atlas_registry_v1_28.json') == '5995bbe9')
chk(14, 'atlas_v1_30 sha + 787/787 fields',
    sha8(B + r'\phase3206\g5g5_ftr22_chain_recheck\atlas_v1_30.html') == 'ebc1e122'
    and fin['field_check'] == {'n_expected': 787, 'n_found': 787,
                               'missing': 0, 'extra': 0, 'mismatch': 0, 'dups': 0})
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3206', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:357]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(15, 'ledger #358 rollback equivalence',
    len(led['measurements']) == 358 and led['measurements'][357]['phase'] == 3206
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(16, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'] == 'fecd8e62')
e = led['measurements'][357]
chk(17, 'ledger chain refs', e['chain']['registry_v128'] == '5995bbe9'
    and e['chain']['atlas_v1_30'] == 'ebc1e122'
    and e['chain']['registry_v127'] == 'd5ce1c19'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8'])
chk(18, 'MEMO + daily + MEMORY updated',
    'Phase 3206 — G5-G5' in open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
    and 'Phase 3206' in open(r'.workbuddy\memory\2026-10-10.md', encoding='utf-8').read()
    and '3207 已预注册' in open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read())

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
