# -*- coding: utf-8 -*-
"""Phase 3217 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P5 = B + r'\phase3217\g5g19_ui_v19_chain_ledger_details'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P5 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P5 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P5 + r'\execution.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('ui_v19_chain_ledger_details_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE final run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + non_ledger + name + design',
    res.get('zero_gpu') is True and res.get('non_ledger') is True
    and res.get('name') == 'g5g19_ui_v19_chain_ledger_details'
    and res.get('design_sha8') == '4313f47d')
chk(5, 'FULL content locks', res['res_sha8'] == '41efe346' and res['seal_sha8'] == '5ae84bf8')

hp = P5 + r'\atlas_client_v1.html'
html = open(hp, encoding='utf-8').read()

chk(6, 'client sealed sha8 9c5b48da + byte count',
    sha8(hp) == '9c5b48da'
    and len(html.encode('utf-8')) == 142904)
chk(7, 'C1 zero script on disk client',
    '<script' not in html.lower()
    and 'javascript:' not in html.lower())
chk(8, 'v1.9 new fields present (chain notes x8 + ledger tail pre/post)',
    all(('data-k="chain.%s.note"' % v) in html
        for v in ('v1.28', 'v1.27', 'v1.26', 'v1.25', 'v1.24', 'v1.23', 'v1.22', 'v1.21'))
    and all(('data-k="led.%d.%s"' % (i, k)) in html
            for i in range(363, 368) for k in ('phase', 'name', 'verdict', 'pre', 'post')))
chk(9, 'ledger tail last row post == 3cf5938b (idx 367)',
    'data-k="led.367.post"' in html and '3cf5938b' in html)
chk(10, 'v1.8 carried fields sampled (feat/gap/appx/ndraft)',
    'data-k="feat.FTR-01.elevel"' in html and 'data-k="feat.FTR-26.statement"' in html
    and 'data-k="gap.GAP-4.mechanism_note"' in html
    and 'data-k="appx.APPX-5.anchor_sha8"' in html
    and 'data-k="ndraft.meta.governance"' in html)

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3217', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:368]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(11, 'ledger #369 rollback equivalence',
    len(led['measurements']) == 369 and led['measurements'][368]['phase'] == 3217
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(12, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][368]
chk(13, 'ledger entry chain refs',
    e['chain']['res_sha8'] == '41efe346' and e['chain']['seal_sha8'] == '5ae84bf8'
    and e['chain']['design_sha8'] == '4313f47d'
    and e['chain']['client_sha8'] == '9c5b48da'
    and e['chain']['base_client'] == 'ca76ec49 (v1.8, phase 3213)'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G19')
chk(14, 'prior tail intact (n=368 pre, 3cf5938b)',
    len(ledold['measurements']) == 368 and led['measurements'][367]['phase'] == 3216
    and ledold['ledger_sha256_8'] == '3cf5938b')
chk(15, 'C5 untouched assertion recorded in FULL run log',
    'C5: ledger untouched (n=368, 3cf5938b): OK'
    in open(P5 + r'\run_log.txt', encoding='utf-8').read())
memo_txt = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
chk(16, 'MEMO 3217 section present with honest log',
    '## Phase 3217 — G5-G19' in memo_txt
    and '要求 expected 值为字符串' in memo_txt)
chk(17, 'MEMO prereg 3218 candidates present',
    '预注册 Phase 3218（两支候选）' in memo_txt)
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily + MEMORY updated (single append, 3218 prereg)',
    daily.count('### Phase 3217') == 1 and '3218 已预注册' in mem
    and 'UI v1.9' in mem)


for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
