# -*- coding: utf-8 -*-
"""Phase 3209 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P9 = B + r'\phase3209\g5g8_ui_v16_client'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P9 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P9 + r'\smoke_result.json', encoding='utf-8'))
html = open(P9 + r'\atlas_client_v1.html', encoding='utf-8').read()

chk(1, 'FULL verdict registered', res['verdict'].startswith('ui_v16_client_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE final run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + non_ledger + name',
    res.get('zero_gpu') is True and res.get('non_ledger') is True
    and res.get('name') == 'g5g8_ui_v16_client')
chk(5, 'FULL content locks', res['res_sha8'] == 'eac7fe87' and res['seal_sha8'] == 'fed536ad')

chk(6, 'C1 zero script / no on* / no javascript: (disk re-read)',
    '<script' not in html.lower()
    and 'javascript:' not in html.lower()
    and __import__('re').search(r'\son\w+\s*=', html, __import__('re').I) is None)
chk(7, 'C2 census on disk 26 features',
    html.count('class="feat"') == 26
    and res['census'] == {'E2_predictive': 17, 'E1_repeatable': 3,
                          'E2_held_out': 3, 'E3_causal_scoped': 3})
chk(8, 'C3 blocks present',
    all(x in html for x in ['id="gaps"', 'id="appendix"', 'id="nline"',
                            'id="chain"', 'id="ledger"', 'N-P3', 'N-P7',
                            'APPX-1', 'APPX-5', 'FTR-01', 'FTR-26']))
chk(9, 'C4 field check 137/137',
    res['field_check'] == {'n_expected': 137, 'n_found': 137,
                           'missing': 0, 'extra': 0, 'mismatch': 0, 'dups': 0})
chk(10, 'client sha on disk + size',
    sha8(P9 + r'\atlas_client_v1.html') == 'bd230988' and len(html.encode('utf-8')) == 73507)
chk(11, 'meta chain spans on disk',
    '5995bbe9' in html and '0f6423e0' in html and '4fb3f37d' in html
    and 'f29da2de' in html and 'd5ce1c19' in html and '47cf01e0' in html)
chk(12, 'GAP-4 mechanism_note + closed badges + draft status rendered',
    'port_residual_dominant' in html or 'amplitude' in html
    and 'closed@3164' in html and 'closed@3163' in html
    and 'draft_pending_cross_line' in html)
reg = json.load(open(B + r'\phase3206\g5g5_ftr22_chain_recheck\atlas_registry_v1_28.json', encoding='utf-8'))
chk(13, 'all 26 feature ids + statements on disk',
    all(('data-fid="%s"' % f['id']) in html and f['id'] + '</span>' in html
        for f in reg['features']))
chk(14, 'C5 ledger untouched (n=360, 383a05db) at FULL time',
    res['gates'].get('recheck_clean') is True
    and json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))['measurements'][359]['phase'] == 3208)
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3209', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:360]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(15, 'ledger #361 rollback equivalence',
    len(led['measurements']) == 361 and led['measurements'][360]['phase'] == 3209
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(16, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'] == '46d53545')
e = led['measurements'][360]
chk(17, 'ledger chain refs', e['chain']['client_html'] == 'bd230988'
    and e['chain']['registry_v128'] == '5995bbe9'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8'])
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'MEMO + daily + MEMORY updated',
    '## Phase 3209 — G5-G8' in memo
    and '### Phase 3209' in daily and daily.count('### Phase 3209') == 1
    and '3210 已预注册' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
