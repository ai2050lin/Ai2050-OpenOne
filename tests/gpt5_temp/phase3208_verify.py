# -*- coding: utf-8 -*-
"""Phase 3208 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P8 = B + r'\phase3208\g5g7_nline_backfill_draft'
NB = r'tests\deepseek_temp'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P8 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P8 + r'\smoke_result.json', encoding='utf-8'))
draft = json.load(open(P8 + r'\n_line_backfill_draft_v1.json', encoding='utf-8'))

chk(1, 'FULL verdict draft_registered', res['verdict'].startswith('n_line_backfill_draft_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE first run clean', smoke['recheck']['recheck_clean'] is True)
chk(4, 'zero_gpu + draft_only + name',
    res.get('zero_gpu') is True and res.get('draft_only') is True
    and res.get('name') == 'g5g7_nline_backfill_draft')
chk(5, 'FULL content locks', res['res_sha8'] == '33cc81ac' and res['seal_sha8'] == '35b0345e')

SHA = {
    'Phase3\\e2_report.txt': 'd6de235f', 'Phase3\\memo_append_e2.md': '1d56f814',
    'Phase3\\wlog_append_e2.md': 'd0810382', 'Phase4\\N1_design_seal.json': 'bd05ba88',
    'Phase4\\memo_append_n1.md': 'cbe8452e', 'Phase4\\n1v2_report_qwen3-4b.txt': 'ad05f1f2',
    'Phase5\\memo_append_n2.md': 'b40de352', 'Phase5\\n2f_report_qwen3-4b.txt': '248f0a46',
    'Phase5\\n2g_report_qwen3-4b.txt': 'd20cac53', 'Phase6\\N2h1_design_seal.json': '4d32b1aa',
    'Phase6\\memo_append_n2h1.md': '373571ed', 'Phase6\\n2h1c_report_qwen3-4b.txt': '4fb86141',
    'Phase7\\N3_design_seal.json': '66afc202', 'Phase7\\memo_append_n3.md': '952f8833',
    'Phase7\\n3_report_qwen3-4b.txt': '51e19b7f',
}
chk(6, 'B1 15 sealed sources bitwise on disk re-read',
    all(sha8(NB + '\\' + k) == v for k, v in SHA.items()))
r6 = open(NB + r'\Phase6\n2h1c_report_qwen3-4b.txt', encoding='utf-8', errors='replace').read()
r7 = open(NB + r'\Phase7\n3_report_qwen3-4b.txt', encoding='utf-8', errors='replace').read()
m5 = open(NB + r'\Phase5\memo_append_n2.md', encoding='utf-8', errors='replace').read()
s6 = json.load(open(NB + r'\Phase6\N2h1_design_seal.json', encoding='utf-8'))
s7 = json.load(open(NB + r'\Phase7\N3_design_seal.json', encoding='utf-8'))
s4 = json.load(open(NB + r'\Phase4\N1_design_seal.json', encoding='utf-8'))
r3 = open(NB + r'\Phase3\e2_report.txt', encoding='utf-8', errors='replace').read()
chk(7, 'B2 P6 report anchors', 'B_cat=+10.192' in r6 and 'B/A=0.958' in r6 and 'B_cat=+0.071' in r6)
chk(8, 'B2 P7 report + seals + P5 memo + P3 head + P4 seal',
    'F2 color B/A=0.925 (lstar=L27)' in r7
    and 'F1_is_a' in json.dumps(s7) and 'B_cat_loo' in json.dumps(s7)
    and '置换向量消融' in s6.get('title', '')
    and '没有任何单头' in m5 and 'L31\u2013L35' in m5
    and 'E2 条件化比较探针' in r3
    and s4['arm_A']['name'] == 'sense_split' and s4['arm_B']['name'] == 'ontology_margin')
chk(9, 'B3 draft file structure + status',
    draft['schema'] == 'n_line_backfill_draft_v1'
    and draft['status'] == 'draft_pending_cross_line'
    and len(draft['entries']) == 5
    and [e['phase'] for e in draft['entries']] ==
        ['N-P3', 'N-P4', 'N-P5', 'N-P6', 'N-P7']
    and all(e['status'] == 'draft_pending_cross_line' for e in draft['entries']))
chk(10, 'draft sha on disk + chain refs',
    sha8(P8 + r'\n_line_backfill_draft_v1.json') == 'f29da2de'
    and draft['phase3208']['res_sha8'] == '33cc81ac'
    and draft['phase3208']['seal_sha8'] == '35b0345e')
ANCH = {
    'p3_e2_report': ('Phase3\\e2_report.txt', 'd6de235f'),
    'p3_memo': ('Phase3\\memo_append_e2.md', '1d56f814'),
    'p3_wlog': ('Phase3\\wlog_append_e2.md', 'd0810382'),
    'p4_seal': ('Phase4\\N1_design_seal.json', 'bd05ba88'),
    'p4_memo': ('Phase4\\memo_append_n1.md', 'cbe8452e'),
    'p4_report': ('Phase4\\n1v2_report_qwen3-4b.txt', 'ad05f1f2'),
    'p5_memo': ('Phase5\\memo_append_n2.md', 'b40de352'),
    'p5_report_f': ('Phase5\\n2f_report_qwen3-4b.txt', '248f0a46'),
    'p5_report_g': ('Phase5\\n2g_report_qwen3-4b.txt', 'd20cac53'),
    'p6_seal': ('Phase6\\N2h1_design_seal.json', '4d32b1aa'),
    'p6_memo': ('Phase6\\memo_append_n2h1.md', '373571ed'),
    'p6_report': ('Phase6\\n2h1c_report_qwen3-4b.txt', '4fb86141'),
    'p7_seal': ('Phase7\\N3_design_seal.json', '66afc202'),
    'p7_memo': ('Phase7\\memo_append_n3.md', '952f8833'),
    'p7_report': ('Phase7\\n3_report_qwen3-4b.txt', '51e19b7f'),
}
chk(11, 'draft source_sha_chain == script SHA_ANCHOR (15 keys, shas match paths)',
    draft['source_sha_chain'] == {k: v for k, (_, v) in ANCH.items()}
    and len(draft['source_sha_chain']) == 15
    and all(sha8(NB + '\\' + ANCH[k][0]) == v for k, (_, v) in ANCH.items()))
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3208', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:359]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(12, 'ledger #360 rollback equivalence',
    len(led['measurements']) == 360 and led['measurements'][359]['phase'] == 3208
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(13, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'] == '383a05db')
e = led['measurements'][359]
chk(14, '#359 (3207) untouched + #360 chain refs',
    led['measurements'][358]['phase'] == 3207
    and led['measurements'][358]['verdict'].startswith('g5g6_appendix_ledger_recheck')
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['chain']['draft_file'] == 'f29da2de')
chk(15, 'backfill NOT applied (no N-P entries in ledger measurements)',
    not any(str(m.get('phase', '')).startswith('N-P') for m in led['measurements'][:359]))
chk(16, 'P3-P7 phases were never G-line numbered (no 3xxx collision)',
    all(not str(m.get('phase', '')).startswith('N-') or m['phase'] in
        ('N-P3', 'N-P4', 'N-P5', 'N-P6', 'N-P7')
        for m in led['measurements'][359:]))
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(17, 'MEMO + daily(10-11) + MEMORY updated',
    '## Phase 3208 — G5-G7' in memo
    and '### Phase 3208' in daily and daily.count('### Phase 3208') == 1
    and '3209 已预注册' in mem)
chk(18, 'execution.json frozen + design sha',
    json.load(open(P8 + r'\execution.json', encoding='utf-8'))['design_sha8'] == '2e96dae7'
    and res['design_sha8'] == '2e96dae7')

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
