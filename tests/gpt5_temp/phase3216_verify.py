# -*- coding: utf-8 -*-
"""Phase 3216 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P5 = B + r'\phase3216\g5g16_standing_ledger_v2'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P5 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P5 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P5 + r'\execution.json', encoding='utf-8'))
v2 = json.load(open(P5 + r'\standing_ledger_v2.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('standing_ledger_v2_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE final run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + name + design',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('name') == 'g5g16_standing_ledger_v2'
    and res.get('design_sha8') == '4a1bda9a')
chk(5, 'FULL content locks', res['res_sha8'] == 'e46b2e51' and res['seal_sha8'] == '09e6908a')

g15 = B + r'\phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'
g16 = B + r'\phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json'
draft = B + r'\phase3208\g5g7_nline_backfill_draft\n_line_backfill_draft_v1.json'
lcorr = r'research\deepseek\atlas\ledger_corrections_v1.json'

flog = open(P5 + r'\run_log.txt', encoding='utf-8').read()
chk(6, 'M1 closure sweep 3210-3215 x6 in FULL log',
    'M1: closure sweep 3210-3215 governance entries x6: OK' in flog
    and 'M1 closures 3210-3215 x6 on ledger' in flog)
appx = res['recheck']['dispositions']['appendix_rows']
chk(7, 'M2 appendix x5 all open (GPU/cross-line)',
    isinstance(appx, list) and len(appx) == 5
    and all(str(r.get('status', '')).startswith('appendix_open') for r in appx))
d = json.load(open(draft, encoding='utf-8'))
lc = json.load(open(lcorr, encoding='utf-8'))
chk(8, 'M3 N-line draft (5, pending) + lcorr single record ops x4',
    d['status'] == 'draft_pending_cross_line' and len(d['entries']) == 5
    and isinstance(lc, dict) and len(lc.get('operations', [])) == 4)
chk(9, 'M3 count recorded on result', res['recheck']['dispositions']['n_ledger_corrections'] == 4)

memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
i77 = memo.find('## Phase 3177')
seg77 = memo[i77:memo.find('\n## ', i77 + 1)]
chk(10, 'M4 水果类/K4 verbatim in 3177 tail', '水果类、K4' in seg77)
no_sha = {k: v for k, v in v2.items() if k != 'sha8'}
content_sha = hashlib.sha256(json.dumps(
    no_sha, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')).hexdigest()[:8]
chk(11, 'standing_ledger_v2 sealed (content sha 207f2fa6, file cd105b50)',
    sha8(P5 + r'\standing_ledger_v2.json') == 'cd105b50'
    and v2['sha8'] == '207f2fa6' and content_sha == '207f2fa6')
chk(12, 'v2 disposition sets (closed 5 / open 4 / pending 1)',
    len(v2['closed']) == 5 and len(v2['open']) == 4
    and len(v2['pending_user_confirmation']) == 1
    and v2['provenance']['ledger'].startswith('bb1474e6')
    and v2['provenance']['closure_ledger_entries'] == [362, 363, 364, 365, 366, 367])

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3216', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:367]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(13, 'ledger #368 rollback equivalence',
    len(led['measurements']) == 368 and led['measurements'][367]['phase'] == 3216
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(14, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][367]
chk(15, 'ledger entry chain refs',
    e['chain']['res_sha8'] == 'e46b2e51' and e['chain']['seal_sha8'] == '09e6908a'
    and e['chain']['design_sha8'] == '4a1bda9a'
    and e['chain']['standing_ledger_v2'] == '207f2fa6'
    and e['chain']['lcorr'] == 'beb775bd' and e['chain']['draft'] == 'f29da2de'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G16')
chk(16, 'prior tail intact (n=367 pre, bb1474e6)',
    len(ledold['measurements']) == 367 and led['measurements'][366]['phase'] == 3215
    and ledold['ledger_sha256_8'] == 'bb1474e6')
chk(17, 'MEMO 3216 section present with honest log',
    '## Phase 3216 — G5-G16' in memo
    and 'operations' in memo and 'standing_ledger_v2_registered' in memo)
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily + MEMORY updated (single append, 3217 prereg)',
    daily.count('### Phase 3216') == 1 and '3217 已预注册' in mem
    and '挂账总账 v2 终审' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
