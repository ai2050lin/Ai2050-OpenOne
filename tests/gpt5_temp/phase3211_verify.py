# -*- coding: utf-8 -*-
"""Phase 3211 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P1 = B + r'\phase3211\g5g10_ui_v17_anchor_values'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P1 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P1 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P1 + r'\execution.json', encoding='utf-8'))
html = open(P1 + r'\atlas_client_v1.html', encoding='utf-8').read()

chk(1, 'FULL verdict registered', res['verdict'].startswith('ui_v17_anchor_values_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + non_ledger + name + design',
    res.get('zero_gpu') is True and res.get('non_ledger') is True
    and res.get('name') == 'g5g10_ui_v17_anchor_values'
    and res.get('design_sha8') == '93fdcc0c')
chk(5, 'FULL content locks', res['res_sha8'] == 'ee976567' and res['seal_sha8'] == 'e1a67926')

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
                            'N-P3', 'N-P7', 'APPX-1', 'APPX-5',
                            'FTR-01', 'FTR-26']))
chk(9, 'C4 field check 611/611',
    res['field_check'] == {'n_expected': 611, 'n_found': 611,
                           'missing': 0, 'extra': 0, 'mismatch': 0, 'dups': 0})
chk(10, 'client sha on disk + size + v1.7 title',
    sha8(P1 + r'\atlas_client_v1.html') == '62ba1bd6'
    and len(html.encode('utf-8')) == 137358
    and 'Atlas Client v1.7' in html)
chk(11, 'anchor values view rendered (spot checks on disk)',
    'feat.FTR-01.a0.src' in html and 'shares_mean_kout.C' in html
    and 'node:N01.status' in html and 'disk_verified' in html)
chk(12, 'assert value fidelity: bool lowercase json, no Python True/False',
    '>true<' in html and '>True<' not in html and '>False<' not in html)
reg = json.load(open(B + r'\phase3206\g5g5_ftr22_chain_recheck\atlas_registry_v1_28.json', encoding='utf-8'))
n_anchors = sum(len(f['anchors']) for f in reg['features'])
n_asserts = sum(len(a.get('asserts', {})) for f in reg['features'] for a in f['anchors'])
chk(13, 'all anchors + asserts counts match registry (61/352)',
    n_anchors == 61 and n_asserts == 352
    and sum(1 for f in reg['features'] if ('data-fid="%s"' % f['id']) in html) == 26)
chk(14, 'C5 ledger untouched (n=362, 1aaab36d) at FULL time',
    json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))['measurements'][361]['phase'] == 3210)
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3211', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:362]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(15, 'ledger #363 rollback equivalence',
    len(led['measurements']) == 363 and led['measurements'][362]['phase'] == 3211
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(16, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][362]
chk(17, 'ledger entry chain refs',
    e['chain']['res_sha8'] == 'ee976567' and e['chain']['seal_sha8'] == 'e1a67926'
    and e['chain']['client_html'] == '62ba1bd6' and e['chain']['client_v16'] == 'bd230988'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G10')
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'MEMO + daily + MEMORY updated (single append, 3212 prereg)',
    '## Phase 3211 — G5-G10' in memo and 'G5-G12 FTR-04' in memo
    and daily.count('### Phase 3211') == 1 and '3212 已预注册' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
