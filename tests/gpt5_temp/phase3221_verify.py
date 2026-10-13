# -*- coding: utf-8 -*-
"""Phase 3221 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P5 = B + r'\phase3221\g5g23_ui_freeze_archival'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P5 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P5 + r'\smoke_result.json', encoding='utf-8'))
cd = json.load(open(P5 + r'\ui_client_version_chain_v1.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('ui_version_chain_archival_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + ui_line_frozen + name + design',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('ui_line_frozen') is True
    and res.get('name') == 'g5g23_ui_freeze_archival'
    and res.get('design_sha8') == '09f68ccd')
chk(5, 'FULL content locks', res['res_sha8'] == '5941ed71' and res['seal_sha8'] == '880a8bfa')
chk(6, 'chain doc sealed (content ca1b24ca, file 660043ab)',
    res['chain_content_sha8'] == 'ca1b24ca' and cd['sha8'] == 'ca1b24ca'
    and sha8(P5 + r'\ui_client_version_chain_v1.json') == '660043ab')

# F1 independent re-read of all five clients
exp = [(r'phase3209\g5g8_ui_v16_client', 'bd230988', 'eac7fe87', 137),
       (r'phase3211\g5g10_ui_v17_anchor_values', '62ba1bd6', 'ee976567', 611),
       (r'phase3213\g5g13_ui_v18_block_anchor_details', 'ca76ec49', 'a398c290', 670),
       (r'phase3217\g5g19_ui_v19_chain_ledger_details', '9c5b48da', '41efe346', 698),
       (r'phase3219\g5g21_ui_v110_standing_block', 'def433e0', '6bdc1127', 727)]
f1 = True
for d, csha, rsha, n in exp:
    rp = B + '\\' + d + r'\result.json'
    rj = json.load(open(rp, encoding='utf-8'))
    if not (sha8(B + '\\' + d + r'\atlas_client_v1.html') == csha
            and rj['res_sha8'] == rsha
            and rj['field_check']['n_expected'] == n):
        f1 = False
chk(7, 'F1 five clients disk re-read (client sha + content res + n_fields)',
    f1)
chk(8, 'chain doc generations match disk values',
    len(cd['generations']) == 5
    and [g['client_sha8'] for g in cd['generations']]
    == ['bd230988', '62ba1bd6', 'ca76ec49', '9c5b48da', 'def433e0']
    and cd['field_evolution'] == '137 -> 611 -> 670 -> 698 -> 727')
chk(9, 'F3 freeze declaration',
    cd['frozen_at'].startswith('v1.10') and 'FROZEN' in cd['governance']
    and res['ui_line_frozen'] is True)
chk(10, 'zero-script constraint in governance text',
    'zero-script' in cd['governance'])

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3221', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:372]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(11, 'ledger #373 rollback equivalence',
    len(led['measurements']) == 373 and led['measurements'][372]['phase'] == 3221
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(12, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][372]
chk(13, 'ledger entry chain refs',
    e['chain']['res_sha8'] == '5941ed71' and e['chain']['seal_sha8'] == '880a8bfa'
    and e['chain']['design_sha8'] == '09f68ccd'
    and e['chain']['chain_doc'] == 'ca1b24ca (file 660043ab)'
    and e['chain']['clients'] == 'bd230988/62ba1bd6/ca76ec49/9c5b48da/def433e0'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G23')
chk(14, 'prior tail intact (n=372 pre, 87b28799)',
    len(ledold['measurements']) == 372 and led['measurements'][371]['phase'] == 3220
    and ledold['ledger_sha256_8'] == '87b28799')
chk(15, 'F4 untouched assertion in FULL log',
    'F4: ledger untouched (n=372, 87b28799): OK'
    in open(P5 + r'\run_log.txt', encoding='utf-8').read())
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
chk(16, 'MEMO 3221 section with version table + honest log',
    '## Phase 3221 — G5-G23' in memo
    and 'ui_version_chain_archival_registered' in memo
    and '137 → 611 → 670 → 698 → 727' in memo
    and '文件字节 sha' in memo)
chk(17, 'MEMO prereg 3222 candidates present',
    '预注册 Phase 3222（两支候选）' in memo)
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily + MEMORY updated (single append, 3222 prereg)',
    daily.count('### Phase 3221') == 1 and '3222 已预注册' in mem
    and 'UI 迭代线自 v1.10 冻结' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
