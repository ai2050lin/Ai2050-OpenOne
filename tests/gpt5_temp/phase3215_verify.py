# -*- coding: utf-8 -*-
"""Phase 3215 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P5 = B + r'\phase3215\g5g15_gap4_closed'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P5 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P5 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P5 + r'\execution.json', encoding='utf-8'))

chk(1, 'FULL verdict confirmed', res['verdict'].startswith('gap4_closed_loop_confirmed'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE final run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + name + design',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('name') == 'g5g15_gap4_closed'
    and res.get('design_sha8') == 'ecfead62')
chk(5, 'FULL content locks', res['res_sha8'] == '1d2f037e' and res['seal_sha8'] == '254bff7c')

r3169 = json.load(open(B + r'\phase3169\g5a6_oov_panel\result.json', encoding='utf-8'))
r3172 = json.load(open(B + r'\phase3172\g5a9_port_calibration\result.json', encoding='utf-8'))
r3175 = json.load(open(B + r'\phase3175\g5a12_residual_arms\result.json', encoding='utf-8'))
r3192 = json.load(open(B + r'\phase3192\g5c5_gap4_amplitude_attribution\result.json', encoding='utf-8'))
gl16 = json.load(open(B + r'\phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json', encoding='utf-8'))
g4 = [g for g in gl16['gaps'] if g['id'] == 'GAP-4'][0]

chk(6, 'K1 3169/3172/3175/3192 res+seal locks (disk re-read)',
    (r3169['res_sha8'], r3169['seal_sha8']) == ('49430a39', '018c6024')
    and (r3172['res_sha8'], r3172['seal_sha8']) == ('463f42c8', 'cdb85525')
    and (r3175['res_sha8'], r3175['seal_sha8']) == ('cb76a177', '9e1a2f08')
    and (r3192['res_sha8'], r3192['seal_sha8']) == ('6221fe00', '58c4f183'))
chk(7, 'K1 scalars bitwise (ratios/curve/delta/amp)',
    r3169['per_model']['qwen3-4b']['B']['ratio'] == 2.8291123613715263
    and r3169['per_model']['qwen3-14b']['B']['ratio'] == 2.125753723276223
    and r3169['per_model']['glm4-9b']['B']['ratio'] == 2.7142111478689683
    and r3172['pooled']['0']['ratio'] == 2.5388114997805062
    and r3172['pooled']['8']['ratio'] == 1.800176217907229
    and r3175['delta_gate']['delta'] == 0.06603206261005456
    and r3192['per_model']['qwen3-4b']['amp_ratio'] == 10.598382949829102
    and r3192['per_model']['qwen3-14b']['amp_ratio'] == 7.110019207000732
    and r3192['per_model']['glm4-9b']['amp_ratio'] == 8.890898704528809)
chk(8, 'K2 GAP-4 status quantified_collapse + six-stage chain on disk',
    g4['status'] == 'quantified_collapse'
    and all(frag in g4['mechanism_note'] for frag in [
        '3171 机制定位', 'borderline_partial_recovery',
        'port_residual_dominant',
        '机制链闭环：3169 量化 -> 3171 encoding_missing 否定 -> '
        '3172 端口部分恢复 -> 3175 残留=端口固有',
        'u*_amplitude_carrier'])
    and set(g4['anchor_sha8']) == {'q03', 'p3151', 'p3169', 'p3171', 'p3172', 'p3175'}
    and g4['anchor_sha8']['p3169'] == '5b51c2c1')
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
segs = {}
for ph in (3169, 3175, 3192):
    i = memo.find('## Phase %d' % ph)
    segs[ph] = memo[i:memo.find('\n## ', i + 1)]
chk(9, 'K3 MEMO 3169/3175/3192 verbatim anchors on disk',
    'ratio_B=2.5388' in segs[3169] and 'collapse_confirmed' in segs[3169]
    and 'port_residual_dominant' in segs[3175] and '0.0660' in segs[3175]
    and '10.5984' in segs[3192] and 'u*' in segs[3192])
chk(10, 'disposition mechanism_chain_closed on result',
    res['recheck']['dispositions']['GAP-4']['status'] == 'mechanism_chain_closed_on_the_books'
    and res['recheck']['dispositions']['GAP-4']['recommendation'].startswith('no registry'))
chk(11, 'K4 untouched assertion recorded in FULL run log',
    'K4: ledger untouched re-assertion OK (n=366, f30394dc)'
    in open(P5 + r'\run_log.txt', encoding='utf-8').read())
chk(12, 'gl16 untouched (0f6423e2 prefix? no — exact 0f6423e0)',
    sha8(B + r'\phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json') == '0f6423e0'
    and sha8(B + r'\phase3169\g5a6_oov_panel\result.json') == '5b51c2c1')

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3215', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:366]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(13, 'ledger #367 rollback equivalence',
    len(led['measurements']) == 367 and led['measurements'][366]['phase'] == 3215
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(14, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][366]
chk(15, 'ledger entry chain refs',
    e['chain']['res_sha8'] == '1d2f037e' and e['chain']['seal_sha8'] == '254bff7c'
    and e['chain']['design_sha8'] == 'ecfead62'
    and e['chain']['r3169'].startswith('5b51c2c1')
    and e['chain']['gl16'] == '0f6423e0'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G15')
chk(16, 'prior tail intact (n=366 pre, f30394dc)',
    len(ledold['measurements']) == 366 and led['measurements'][365]['phase'] == 3214
    and ledold['ledger_sha256_8'] == 'f30394dc')
chk(17, 'MEMO 3215 section present with honest log',
    '## Phase 3215 — G5-G15' in memo
    and 'pooled 实为 k-字符串键 dict' in memo)
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily + MEMORY updated (single append, 3216 prereg)',
    daily.count('### Phase 3215') == 1 and '3216 已预注册' in mem
    and 'G5-G16 挂账总账 v2' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
