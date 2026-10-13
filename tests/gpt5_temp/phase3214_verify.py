# -*- coding: utf-8 -*-
"""Phase 3214 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P4 = B + r'\phase3214\g5g14_ftr15_heldout_closed'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P4 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P4 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P4 + r'\execution.json', encoding='utf-8'))

chk(1, 'FULL verdict closed', res['verdict'].startswith('ftr15_heldout_closed'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + name + design',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('name') == 'g5g14_ftr15_heldout_closed'
    and res.get('design_sha8') == 'b8052e94')
chk(5, 'FULL content locks', res['res_sha8'] == 'a48005fa' and res['seal_sha8'] == '926be348')

r4b = json.load(open(B + r'\phase3158\g4p1_output_equivalence_class\qwen3-4b\result.json', encoding='utf-8'))
r14b = json.load(open(B + r'\phase3158\g4p1_output_equivalence_class\qwen3-14b\result.json', encoding='utf-8'))
rglm = json.load(open(B + r'\phase3158\g4p1_output_equivalence_class\glm4\result.json', encoding='utf-8'))
r3180 = json.load(open(B + r'\phase3180\g5b5_ftr15_heldout\result.json', encoding='utf-8'))
r3204 = json.load(open(B + r'\phase3204\g5g1_ftr15_heldout_recheck\result.json', encoding='utf-8'))
reg = json.load(open(B + r'\phase3206\g5g5_ftr22_chain_recheck\atlas_registry_v1_28.json', encoding='utf-8'))
f15 = [f for f in reg['features'] if f['id'] == 'FTR-15'][0]
a3158 = [a for a in f15['anchors'] if a['src'] == 'p3158'][0]
a3180 = [a for a in f15['anchors'] if a['src'] == 'p3180'][0]
ratios = {'qwen3-4b': r4b['ratio'], 'qwen3-14b': r14b['ratio'], 'glm4': rglm['ratio']}
mk = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4': 'glm4'}

chk(6, 'H1 3158 ratios bitwise == p3158 anchor x3 (disk re-read)',
    all(a3158['asserts']['ratios.' + m] == v for m, v in ratios.items())
    and ratios == {'qwen3-4b': 1.0283477199188433, 'qwen3-14b': 1.3316768638966026,
                   'glm4': 1.2472595295618603})
chk(7, 'H2 3180 locks + verdict + chain bitwise',
    r3180['res_sha8'] == 'f1689e94' and r3180['seal_sha8'] == 'a3c1b53e'
    and r3180['verdict'].startswith('ftr15_upgraded_E2_predictive')
    and all(r3180['per_model'][mk[m]]['ratio_full_recompute'] == v
            and a3180['asserts']['ratio_full_recompute.' + m] == v
            and r3180['per_model'][mk[m]]['drift'] == 0.0
            and a3180['asserts']['max_log_band.' + m] == r3180['per_model'][mk[m]]['max_band'] < 0.25
            and r3180['per_model'][mk[m]]['all_same_class'] is True
            for m, v in ratios.items()))
chk(8, 'H2 folds 16x3 all same_class + n_c_gt1 [10,16,15] + gates',
    all(len(r3180['folds_full'][mk[m]]) == 16
        and all(fd['same_class'] for fd in r3180['folds_full'][mk[m]])
        for m in mk)
    and [r3180['per_model']['qwen3-4b']['n_c_gt1'],
         r3180['per_model']['qwen3-14b']['n_c_gt1'],
         r3180['per_model']['glm4']['n_c_gt1']] == [10, 16, 15]
    and r3180['gates']['G1_device'] is True and r3180['gates']['G2_heldout'] is True)
chk(9, 'H3 3204 locks + verdict + recheck clean',
    r3204['res_sha8'] == '263934c8' and r3204['seal_sha8'] == '7bb722b1'
    and r3204['design_sha8'] == '588dcf87'
    and r3204['verdict'].startswith('ftr15_heldout_recheck_registered')
    and r3204['recheck']['recheck_clean'] is True)
chk(10, 'H3 registry scope_limits 3204 append verbatim + E2_held_out note',
    all(frag in f15['scope_limits'] for frag in [
        '3204 G5-G1 FTR-15 held_out 补强复读',
        'ratio_full_recompute 逐位 == 3158 ×3（device drift 0 ×3）',
        'max_log_band 0.0300/0.0278/0.0182 <0.25',
        'E2_held_out 为 E2_predictive 强化档注记，不新增独立 evidence_level'])
    and f15['evidence_level'] == 'E2_predictive')

memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
i3204 = memo.find('## Phase 3204'); j3204 = memo.find('\n## ', i3204 + 1)
chk(11, 'MEMO 3204 verbatim anchors on disk',
    i3204 >= 0 and '## Phase 3204 — G5-G1 FTR-15 held_out 补强复读臂' in memo[i3204:j3204]
    and 'E2_held_out' in memo[i3204:j3204] and 'held_out' in memo[i3204:j3204])
chk(12, 'disposition closed@3204 on result',
    res['recheck']['dispositions']['FTR-15 held_out']['status'] == 'closed_at_3204'
    and res['recheck']['dispositions']['FTR-15 held_out']['recommendation'].startswith('no registry'))
chk(13, 'A4/H4 untouched assertion recorded in FULL run log',
    'H4: ledger untouched re-assertion OK (n=365, 3bd36b2a)'
    in open(P4 + r'\run_log.txt', encoding='utf-8').read())

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3214', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:365]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(14, 'ledger #366 rollback equivalence',
    len(led['measurements']) == 366 and led['measurements'][365]['phase'] == 3214
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(15, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][365]
chk(16, 'ledger entry chain refs',
    e['chain']['res_sha8'] == 'a48005fa' and e['chain']['seal_sha8'] == '926be348'
    and e['chain']['design_sha8'] == 'b8052e94'
    and e['chain']['r3180'].startswith('cb811310')
    and e['chain']['r3204'].startswith('b1021387')
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G14')
chk(17, 'prior tail intact (n=365 pre, 3bd36b2a) + sealed sources untouched',
    len(ledold['measurements']) == 365 and led['measurements'][364]['phase'] == 3213
    and ledold['ledger_sha256_8'] == '3bd36b2a'
    and sha8(B + r'\phase3180\g5b5_ftr15_heldout\result.json') == 'cb811310'
    and sha8(B + r'\phase3204\g5g1_ftr15_heldout_recheck\result.json') == 'b1021387')
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'MEMO + daily + MEMORY updated (single append, 3215 prereg)',
    '## Phase 3214 — G5-G14' in memo and 'G5-G15 GAP-4' in memo
    and daily.count('### Phase 3214') == 1 and '3215 已预注册' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
