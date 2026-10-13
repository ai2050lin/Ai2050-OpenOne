# -*- coding: utf-8 -*-
"""Phase 3220 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P5 = B + r'\phase3220\g5g22_appendix_anchors_reread'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P5 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P5 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P5 + r'\execution.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('appendix_anchors_reread_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + no_status_change + name + design',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('no_status_change') is True
    and res.get('name') == 'g5g22_appendix_anchors_reread'
    and res.get('design_sha8') == '4288f690')
chk(5, 'FULL content locks', res['res_sha8'] == '122adaf5' and res['seal_sha8'] == '47f9213e')
chk(6, 'anchor map recorded', res['anchors'] == {'APPX-1': '448af595', 'APPX-2': '511d9b13',
                                                 'APPX-3': '46a5034d', 'APPX-4/5': '24c60160'})

# A1 on-disk re-derivation (independent of main run)
B2 = B + r'\phase3166\g5a3b_logic_direction\result.json'
r1 = json.load(open(B2, encoding='utf-8'))
r3 = json.load(open(B + r'\phase3155\g2p1_relation_family_operator_separability'
                    r'\summary\result_summary.json', encoding='utf-8'))
chk(7, 'A1 p3166 sha + scalars (census mixed, cls_agree False)',
    sha8(B2) == '448af595' and r1['overall']['census_cls'] == 'mixed'
    and r1['overall']['cls_agree'] is False)
chk(8, 'A1 p3155 sha + k2_int_share bitwise x3',
    sha8(B + r'\phase3155\g2p1_relation_family_operator_separability'
          r'\summary\result_summary.json') == '46a5034d'
    and r3['k2_int_share_kout']['qwen3-4b'] == 0.15187579769212284
    and r3['k2_int_share_kout']['qwen3-14b'] == 0.1824533831239199
    and r3['k2_int_share_kout']['glm4'] == 0.16411745881744746)
chk(9, 'A1 p3165 sha + cross_model_K present',
    sha8(B + r'\phase3165\g5a3_family_alignment\result.json') == '511d9b13'
    and 'cross_model_K' in json.load(open(B + r'\phase3165\g5a3_family_alignment'
                                              r'\result.json', encoding='utf-8')))
gl15 = json.load(open(B + r'\phase3175\g5a12_residual_arms\gap_ledger_v1_5.json', encoding='utf-8'))
appx = {a['id']: a for a in gl15['appendix']}
chk(10, 'A2 statuses appendix_open x3 + anchor_sha8 match on-disk',
    all(appx[i]['status'] == 'appendix_open' for i in ('APPX-1', 'APPX-2', 'APPX-3'))
    and appx['APPX-1']['anchor_sha8'] == '448af595'
    and appx['APPX-2']['anchor_sha8'] == '511d9b13'
    and appx['APPX-3']['anchor_sha8'] == '46a5034d')
chk(11, 'A3 agate sha + draft/lcorr unchanged',
    sha8(r'research\deepseek\atlas\a_gate_closure_v1.json') == '24c60160'
    and json.load(open(B + r'\phase3208\g5g7_nline_backfill_draft'
                          r'\n_line_backfill_draft_v1.json', encoding='utf-8'))['status']
    == 'draft_pending_cross_line'
    and json.load(open(r'research\deepseek\atlas\ledger_corrections_v1.json',
                       encoding='utf-8'))['status'] == 'SEALED_BUT_NOT_APPLIED')
chk(12, 'A4 untouched assertion in FULL log',
    'A4: ledger untouched (n=371, 5cff05f6): OK'
    in open(P5 + r'\run_log.txt', encoding='utf-8').read())

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3220', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:371]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(13, 'ledger #372 rollback equivalence',
    len(led['measurements']) == 372 and led['measurements'][371]['phase'] == 3220
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(14, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][371]
chk(15, 'ledger entry chain refs',
    e['chain']['res_sha8'] == '122adaf5' and e['chain']['seal_sha8'] == '47f9213e'
    and e['chain']['design_sha8'] == '4288f690'
    and e['chain']['p3166'] == '448af595' and e['chain']['p3155'] == '46a5034d'
    and e['chain']['agate'] == '24c60160'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G22')
chk(16, 'prior tail intact (n=371 pre, 5cff05f6)',
    len(ledold['measurements']) == 371 and led['measurements'][370]['phase'] == 3219
    and ledold['ledger_sha256_8'] == '5cff05f6')
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
chk(17, 'MEMO 3220 section + prereg 3221 present',
    '## Phase 3220 — G5-G22' in memo
    and 'appendix_anchors_reread_registered' in memo
    and '预注册 Phase 3221（两支候选）' in memo)
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily + MEMORY updated (single append, 3221 prereg)',
    daily.count('### Phase 3220') == 1 and '3221 已预注册' in mem
    and '在盘锚复读' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
