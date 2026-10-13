# -*- coding: utf-8 -*-
"""Phase 3210 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P0 = B + r'\phase3210\g5g9_standing_items_audit'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P0 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P0 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P0 + r'\execution.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('standing_items_audit_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE final run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + name + design',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('name') == 'g5g9_standing_items_audit'
    and res.get('design_sha8') == 'e678dc42')
chk(5, 'FULL content locks', res['res_sha8'] == '3ae23744' and res['seal_sha8'] == '23008571')

rc = res['recheck']
chk(6, 'D1 provenance MEMO-anchored 5 phases + ledger absence finding',
    not any(c[0] == 'D1_provenance' for c in rc['contradictions'])
    and rc['dispositions']['ledger_absence_finding']
    ['phases_absent_from_shared_ledger'] == [2767, 2800, 2802, 2810, 2833])
chk(7, 'D2 memo anchors verbatim (no contradictions)',
    not any(c[0] == 'D2_memo_anchors' for c in rc['contradictions']))
chk(8, 'D3 standing list first phase == 3177',
    rc['dispositions']['standing_list_first_phase'] == 3177)
chk(9, 'dispositions both recommend close',
    rc['dispositions']['水果类']['recommendation'].startswith('close')
    and rc['dispositions']['K4']['recommendation'].startswith('close')
    and rc['dispositions']['K4']['status'] == 'criterion_falsified_and_superseded')
chk(10, 'execution.json frozen before observation (design e678dc42)',
    exe['design_sha8'] == 'e678dc42' and exe['frozen_before'] == 'any observation')

memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
i2833 = memo.find('## Phase 2833')
j2833 = memo.find('\n## ', i2833 + 1)
chk(11, 'D2 anchors physically on disk (2833 segment)',
    i2833 >= 0 and 'attention-sink/位置通用通道支配' in memo[i2833:j2833]
    and 'P-K2/P-K4 双否证' in memo[i2833:j2833])
chk(12, 'fruit/K4 first list occurrence physically in 3177 tail',
    '水果类、K4' in memo
    and memo.rfind('## Phase 3177', 0, memo.find('水果类、K4')) > 0)
chk(13, 'MEMO 3210 section + prereg 3211 written',
    '## Phase 3210 — G5-G9' in memo
    and '## Phase 3210 \u2014 G5-G9' in memo
    and 'G5-G10 UI v1.7' in memo and 'G5-G11 FTR-04' in memo)

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3210', encoding='utf-8'))
chk(14, 'ledger #362 appended, prior tail intact (n=361, 46d53545)',
    len(led['measurements']) == 362 and led['measurements'][361]['phase'] == 3210
    and led['measurements'][360]['phase'] == 3209
    and len(ledold['measurements']) == 361)
back = dict(led)
back['measurements'] = led['measurements'][:361]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(15, 'ledger #362 rollback equivalence',
    json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
    json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(16, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][361]
chk(17, 'ledger entry chain refs',
    e['chain']['res_sha8'] == '3ae23744' and e['chain']['seal_sha8'] == '23008571'
    and e['chain']['design_sha8'] == 'e678dc42'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G9')
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily + MEMORY updated (single append, 3211 prereg)',
    daily.count('### Phase 3210') == 1 and 'ledger #362' in daily
    and '3211 已预注册' in mem and '用户确认文档编辑待办' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
