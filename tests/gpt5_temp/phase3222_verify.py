# -*- coding: utf-8 -*-
"""Phase 3222 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P5 = B + r'\phase3222\g5g24_gline_session_summary'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P5 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P5 + r'\smoke_result.json', encoding='utf-8'))
sd = json.load(open(P5 + r'\gline_session_summary_v1.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('gline_session_summary_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + name + design',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('name') == 'g5g24_gline_session_summary'
    and res.get('design_sha8') == '7081779b')
chk(5, 'FULL content locks', res['res_sha8'] == 'a1de6a81' and res['seal_sha8'] == '3b9cb4e6')
chk(6, 'summary sealed (content 52ba5d3a, file 17dfb237)',
    res['summary_content_sha8'] == '52ba5d3a' and sd['sha8'] == '52ba5d3a'
    and sha8(P5 + r'\gline_session_summary_v1.json') == '17dfb237')

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3222', encoding='utf-8'))

# G1 independent sequence re-derivation
exp = [(362, 3210), (363, 3211), (364, 3212), (365, 3213), (366, 3214),
       (367, 3215), (368, 3216), (369, 3217), (370, 3218), (371, 3219),
       (372, 3220), (373, 3221)]
seq_ok = all(led['measurements'][i - 1]['phase'] == ph
             and ('|REGISTERED' in led['measurements'][i - 1]['verdict']
                  or led['measurements'][i - 1]['verdict'].split('|')[0].endswith('_registered'))
             for i, ph in exp)
chk(7, 'G1 twelve entries #362-#373 phases sequential, all REGISTERED', seq_ok)
chk(8, 'G2 census in summary doc (closure 3 / ui 5 / audit 4)',
    sd['census'] == {'audit_inventory': 4, 'closure': 3, 'ui_iteration': 5}
    and len(sd['rounds']) == 12)
chk(9, 'closing_state in summary doc (v2/ready/frozen/appendix)',
    sd['closing_state']['standing_ledger_v2'].startswith('207f2fa6')
    and 'ready_append_type' in sd['closing_state']['apply_readiness']['nline']
    and 'not_ready' in sd['closing_state']['apply_readiness']['appx5']
    and 'FROZEN at v1.10' in sd['closing_state']['ui_line'])
chk(10, 'pending_user_confirmation items in summary doc',
    len(sd['pending_user_confirmation']) == 2
    and any('水果类' in s for s in sd['pending_user_confirmation']))
chk(11, 'G4 untouched assertion in FULL log',
    'G4: ledger untouched (n=373, ff363d67): OK'
    in open(P5 + r'\run_log.txt', encoding='utf-8').read())

back = dict(led)
back['measurements'] = led['measurements'][:373]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(12, 'ledger #374 rollback equivalence',
    len(led['measurements']) == 374 and led['measurements'][373]['phase'] == 3222
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(13, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][373]
chk(14, 'ledger entry chain refs',
    e['chain']['res_sha8'] == 'a1de6a81' and e['chain']['seal_sha8'] == '3b9cb4e6'
    and e['chain']['design_sha8'] == '7081779b'
    and e['chain']['summary_doc'] == '52ba5d3a (file 17dfb237)'
    and e['chain']['window'] == 'phases 3210-3221, ledger #362-#373'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G24')
chk(15, 'prior tail intact (n=373 pre, ff363d67)',
    len(ledold['measurements']) == 373 and led['measurements'][372]['phase'] == 3221
    and ledold['ledger_sha256_8'] == 'ff363d67')
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
chk(16, 'MEMO 3222 section with census + honest log',
    '## Phase 3222 — G5-G24' in memo
    and 'gline_session_summary_registered' in memo
    and '在盘账本为唯一事实源' in memo)
chk(17, 'MEMO prereg 3223 candidates present',
    '预注册 Phase 3223（两支候选）' in memo
    and '静默待命' in memo)
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily + MEMORY updated (single append, 3223 prereg)',
    daily.count('### Phase 3222') == 1 and '3223 已预注册' in mem
    and '静默待命' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
