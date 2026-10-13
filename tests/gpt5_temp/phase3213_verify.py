# -*- coding: utf-8 -*-
"""Phase 3213 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json
import re


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P3 = B + r'\phase3213\g5g13_ui_v18_block_anchor_details'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P3 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P3 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P3 + r'\execution.json', encoding='utf-8'))
html = open(P3 + r'\atlas_client_v1.html', encoding='utf-8').read()

chk(1, 'FULL verdict registered', res['verdict'].startswith('ui_v18_block_anchor_details_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE final run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + non_ledger + name + design',
    res.get('zero_gpu') is True and res.get('non_ledger') is True
    and res.get('name') == 'g5g13_ui_v18_block_anchor_details'
    and res.get('design_sha8') == 'e5a9ed86')
chk(5, 'FULL content locks', res['res_sha8'] == 'a398c290' and res['seal_sha8'] == '095624ed')

chk(6, 'C1 zero script / no on* / no javascript: (disk re-read)',
    '<script' not in html.lower()
    and 'javascript:' not in html.lower()
    and re.search(r'\son\w+\s*=', html, re.I) is None)
chk(7, 'C2 census on disk 26 features',
    html.count('class="feat"') == 26
    and res['census'] == {'E2_predictive': 17, 'E1_repeatable': 3,
                          'E2_held_out': 3, 'E3_causal_scoped': 3})
chk(8, 'C3 blocks present',
    all(x in html for x in ['id="gaps"', 'id="appendix"', 'id="nline"',
                            'N-P3', 'N-P7', 'APPX-1', 'APPX-5',
                            'FTR-01', 'FTR-26']))
chk(9, 'C4 field check 670/670',
    res['field_check'] == {'n_expected': 670, 'n_found': 670,
                           'missing': 0, 'extra': 0, 'mismatch': 0, 'dups': 0})
chk(10, 'client sha on disk + size + v1.8 title',
    sha8(P3 + r'\atlas_client_v1.html') == 'ca76ec49'
    and len(html.encode('utf-8')) == 141577
    and 'Atlas Client v1.8' in html)
chk(11, 'v1.8 block details rendered on disk',
    'data-k="gap.GAP-1.ea0"' in html and 'data-k="gap.GAP-4.anch.q03"' in html
    and 'data-k="appx.APPX-1.anchor_sha8"' in html
    and 'data-k="ndraft.N-P3.src.e2_report"' in html
    and 'data-k="ndraft.meta.governance"' in html
    and 'APPX-4/5 同型：只补不改' in html)
chk(12, 'v1.7 feature anchor view carried over',
    'feat.FTR-01.a0.src' in html and 'shares_mean_kout.C' in html
    and '>true<' in html and '>True<' not in html)
gl16 = json.load(open(B + r'\phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json', encoding='utf-8'))
gl15 = json.load(open(B + r'\phase3175\g5a12_residual_arms\gap_ledger_v1_5.json', encoding='utf-8'))
dr = json.load(open(B + r'\phase3208\g5g7_nline_backfill_draft\n_line_backfill_draft_v1.json', encoding='utf-8'))
n_new = (sum(len(g['evidence_anchors']) + len(g['anchor_sha8']) for g in gl16['gaps'])
         + 2 * len(gl15['appendix'])
         + sum(1 + len(e['sources']) for e in dr['entries']) + 3)
chk(13, 'increment field count == 59 (11+15+10+20+3)', n_new == 59)
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3213', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:364]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(14, 'ledger #365 rollback equivalence',
    len(led['measurements']) == 365 and led['measurements'][364]['phase'] == 3213
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(15, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][364]
chk(16, 'ledger entry chain refs',
    e['chain']['res_sha8'] == 'a398c290' and e['chain']['seal_sha8'] == '095624ed'
    and e['chain']['client_html'] == 'ca76ec49' and e['chain']['client_v17'] == '62ba1bd6'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G13')
chk(17, 'prior tail intact (n=364 pre, 63a42cbc) + C5 logged',
    len(ledold['measurements']) == 364 and led['measurements'][363]['phase'] == 3212
    and ledold['ledger_sha256_8'] == '63a42cbc'
    and 'A4' not in open(P3 + r'\run_log.txt', encoding='utf-8').read()
    and 'C5: ledger untouched (n=364, 63a42cbc): OK'
    in open(P3 + r'\run_log.txt', encoding='utf-8').read())
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'MEMO + daily + MEMORY updated (single append, 3214 prereg)',
    '## Phase 3213 — G5-G13' in memo and 'G5-G14 FTR-15' in memo
    and daily.count('### Phase 3213') == 1 and '3214 已预注册' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
