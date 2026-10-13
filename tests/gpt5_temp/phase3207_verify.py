# -*- coding: utf-8 -*-
"""Phase 3207 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(B + r'\phase3207\g5g6_appendix_ledger_recheck\result.json', encoding='utf-8'))
smoke = json.load(open(B + r'\phase3207\g5g6_appendix_ledger_recheck\smoke_result.json', encoding='utf-8'))

chk(1, 'FULL verdict report_only_clean',
    res['verdict'].startswith('appendix_ledger_recheck_report_only_clean'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE second run clean', smoke['recheck']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + name',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('name') == 'g5g6_appendix_ledger_recheck')
chk(5, 'FULL content locks', res['res_sha8'] == '97b25b57' and res['seal_sha8'] == '20d84260')

D = B + r'\..'
gl15 = json.load(open(B + r'\phase3175\g5a12_residual_arms\gap_ledger_v1_5.json', encoding='utf-8'))
gl16 = json.load(open(B + r'\phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json', encoding='utf-8'))
chk(6, 'A1 appendix 5 entries + version',
    gl15['version'] == '1.5' and len(gl15['appendix']) == 5
    and [a['id'] for a in gl15['appendix']] ==
        ['APPX-1', 'APPX-2', 'APPX-3', 'APPX-4', 'APPX-5'])
chk(7, 'A2 anchor shas == anchor_sha8 fields',
    sha8(B + r'\phase3166\g5a3b_logic_direction\result.json') == '448af595'
    and sha8(B + r'\phase3165\g5a3_family_alignment\result.json') == '511d9b13'
    and sha8(B + r'\phase3155\g2p1_relation_family_operator_separability\summary\result_summary.json') == '46a5034d'
    and gl15['appendix'][0]['anchor_sha8'] == '448af595'
    and gl15['appendix'][1]['anchor_sha8'] == '511d9b13'
    and gl15['appendix'][2]['anchor_sha8'] == '46a5034d')
r66 = json.load(open(B + r'\phase3166\g5a3b_logic_direction\result.json', encoding='utf-8'))
r65 = json.load(open(B + r'\phase3165\g5a3_family_alignment\result.json', encoding='utf-8'))
s55 = json.load(open(B + r'\phase3155\g2p1_relation_family_operator_separability\summary\result_summary.json', encoding='utf-8'))
s66 = json.dumps(r66, ensure_ascii=False)
eff = []
i = 0
while True:
    i = s66.find('"eff_ge05": ', i)
    if i < 0:
        break
    eff.append(int(s66[i + 12:s66.find(',', i)]))
    i = s66.find(',', i)
chk(8, 'A2 3166 eff_ge05 sum==2 one-hot + locks',
    sum(eff) == 2 and eff.count(2) == 1
    and r66['res_sha8'] == '89b3f320' and r66['seal_sha8'] == '263de0ab')
chk(9, 'A2 3165 K_entity_eff_ge05==0 glm4/14b + locks',
    r65['cross_model_K']['glm4']['K_entity_eff_ge05'] == 0
    and r65['cross_model_K']['qwen3-14b']['K_entity_eff_ge05'] == 0
    and r65['res_sha8'] == '9b0fe9c9' and r65['seal_sha8'] == 'e966a4e5')
chk(10, 'A2 3155 kout share bitwise x3',
    s55['k2_int_share_kout'] == {'qwen3-4b': 0.15187579769212284,
                                 'qwen3-14b': 0.1824533831239199,
                                 'glm4': 0.16411745881744746})
chk(11, 'A3 gl16 version 1.6 + appendix append-only bitwise == gl15',
    gl16['version'] == '1.6'
    and json.dumps(gl16['appendix'], ensure_ascii=False, sort_keys=True) ==
        json.dumps(gl15['appendix'], ensure_ascii=False, sort_keys=True)
    and sha8(B + r'\phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json') == '0f6423e0')
ag = json.load(open(r'research\deepseek\atlas\a_gate_closure_v1.json', encoding='utf-8'))
cc = {c['id']: c for c in ag['corrections_C1_C6']}
chk(12, 'A4 agate sha 24c60160 + C1-C6 x6',
    sha8(r'research\deepseek\atlas\a_gate_closure_v1.json') == '24c60160'
    and len(ag['corrections_C1_C6']) == 6)
chk(13, 'A4 C4/C6 pending_cross_line + 未施加 + corrections file on disk',
    cc['C4']['status'] == 'pending_cross_line' and cc['C6']['status'] == 'pending_cross_line'
    and '未施加' in cc['C4']['enactment'] and '未施加' in cc['C6']['enactment']
    and sha8(r'research\deepseek\atlas\ledger_corrections_v1.json') == 'beb775bd')
chk(14, 'no registry/gap-ledger/HTML mutation by 3207',
    not any('3207' in d for d in
            [dd for dd in __import__('os').listdir(B + r'\phase3207\g5g6_appendix_ledger_recheck')
             if ('registry' in dd or 'gap_ledger' in dd or dd.endswith('.html'))])
    and gl15['version'] == '1.5' and gl16['version'] == '1.6')
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3207', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:358]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(15, 'ledger #359 rollback equivalence',
    len(led['measurements']) == 359 and led['measurements'][358]['phase'] == 3207
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(16, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'] == 'efe4a6de')
e = led['measurements'][358]
chk(17, 'ledger chain refs', e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['chain']['gl15'] == '4fb3f37d' and e['chain']['gl16'] == '0f6423e0'
    and e['chain']['agate'] == '24c60160' and e['chain']['ledger_corrections'] == 'beb775bd')
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
daily = open(r'.workbuddy\memory\2026-10-10.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'MEMO + daily (once) + MEMORY updated',
    '## Phase 3207 — G5-G6' in memo
    and daily.count('### Phase 3207 G5-G6') == 1
    and '3208 已预注册' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
