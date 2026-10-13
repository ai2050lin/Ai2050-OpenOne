# -*- coding: utf-8 -*-
"""Phase 3219 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P5 = B + r'\phase3219\g5g21_ui_v110_standing_block'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P5 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P5 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P5 + r'\execution.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('ui_v110_standing_block_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE final run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + non_ledger + name + design',
    res.get('zero_gpu') is True and res.get('non_ledger') is True
    and res.get('name') == 'g5g21_ui_v110_standing_block'
    and res.get('design_sha8') == '16d99a7e')
chk(5, 'FULL content locks', res['res_sha8'] == '6bdc1127' and res['seal_sha8'] == '5819a1b3')

hp = P5 + r'\atlas_client_v1.html'
html = open(hp, encoding='utf-8').read()

chk(6, 'client sealed sha8 def433e0 + byte count',
    sha8(hp) == 'def433e0' and len(html.encode('utf-8')) == 146547)
chk(7, 'C1 zero script on disk client',
    '<script' not in html.lower() and 'javascript:' not in html.lower())
chk(8, 'standing block spans on disk (closed 15 / open 8 / pending 2 / ready 2)',
    all(('data-k="stand.closed.%d.%s"' % (i, k)) in html
        for i in range(5) for k in ('item', 'closed_at', 'evidence'))
    and all(('data-k="stand.open.%d.%s"' % (i, k)) in html
            for i in range(4) for k in ('item', 'why'))
    and 'data-k="stand.pending.0.item"' in html and 'data-k="stand.pending.0.why"' in html
    and 'data-k="stand.ready.nline"' in html and 'data-k="stand.ready.appx5"' in html)
chk(9, 'readiness values (ready_append_type / not_ready_stale_baseline)',
    'ready_append_type' in html and 'not_ready_stale_baseline' in html)
chk(10, 'meta new anchors + carried fields sampled',
    'data-k="meta.v2_sha8"' in html and 'data-k="meta.checklist_sha8"' in html
    and 'data-k="feat.FTR-01.elevel"' in html
    and 'data-k="gap.GAP-4.mechanism_note"' in html
    and 'data-k="chain.v1.28.note"' in html)

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3219', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:370]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(11, 'ledger #371 rollback equivalence',
    len(led['measurements']) == 371 and led['measurements'][370]['phase'] == 3219
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(12, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][370]
chk(13, 'ledger entry chain refs',
    e['chain']['res_sha8'] == '6bdc1127' and e['chain']['seal_sha8'] == '5819a1b3'
    and e['chain']['design_sha8'] == '16d99a7e'
    and e['chain']['client_sha8'] == 'def433e0'
    and e['chain']['base_client'] == '9c5b48da (v1.9, phase 3217)'
    and e['chain']['v2'] == 'cd105b50' and e['chain']['checklist'] == 'd3f8c544'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G21')
chk(14, 'prior tail intact (n=370 pre, d85d6792)',
    len(ledold['measurements']) == 370 and led['measurements'][369]['phase'] == 3218
    and ledold['ledger_sha256_8'] == 'd85d6792')
chk(15, 'C5 untouched assertion recorded in FULL run log',
    'C5: ledger untouched (n=370, d85d6792): OK'
    in open(P5 + r'\run_log.txt', encoding='utf-8').read())
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
chk(16, 'MEMO 3219 section present with honest log',
    '## Phase 3219 — G5-G21' in memo
    and 'not_ready_stale_baseline' in memo
    and 'ui_v110_standing_block_registered' in memo)
chk(17, 'MEMO prereg 3220 candidates present',
    '预注册 Phase 3220（两支候选）' in memo)
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily + MEMORY updated (single append, 3220 prereg)',
    daily.count('### Phase 3219') == 1 and '3220 已预注册' in mem
    and 'UI v1.10' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
