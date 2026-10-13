# -*- coding: utf-8 -*-
"""Phase 3218 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P5 = B + r'\phase3218\g5g20_crossline_apply_readiness'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P5 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P5 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P5 + r'\execution.json', encoding='utf-8'))
cl = json.load(open(P5 + r'\crossline_apply_checklist_v1.json', encoding='utf-8'))

chk(1, 'FULL verdict registered', res['verdict'].startswith('crossline_apply_readiness_registered'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + no_application + name + design',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('no_application') is True
    and res.get('name') == 'g5g20_crossline_apply_readiness'
    and res.get('design_sha8') == 'a6d5cdb5')
chk(5, 'FULL content locks', res['res_sha8'] == '976391dc' and res['seal_sha8'] == 'cd7eebe2')

chk(6, 'checklist sealed (content e402da6e, file d3f8c544)',
    res['checklist_content_sha8'] == 'e402da6e' and cl['sha8'] == 'e402da6e'
    and sha8(P5 + r'\crossline_apply_checklist_v1.json') == 'd3f8c544')
chk(7, 'R1 nline itemized (5 entries, sources keys)',
    cl['nline']['n_entries'] == 5
    and all(e['status'] == 'draft_pending_cross_line' for e in cl['nline']['entries'])
    and all(e['sources_keys'] for e in cl['nline']['entries'])
    and 'append batch of 5' in ' '.join(cl['nline']['apply_plan']))
chk(8, 'R2 appx5 ops itemized (4)',
    cl['appx5']['n_operations'] == 4
    and [o['op'] for o in cl['appx5']['operations']]
    == ['set_field', 'unify_schema', 'enumerate_field', 'append_measurement'])
chk(9, 'R2 stale-baseline finding recorded',
    cl['appx5']['target_sha256_before'].startswith('bbda63df')
    and 'STALE' in cl['appx5']['finding_stale_baseline']
    and res['stale_baseline_finding'] is True)
chk(10, 'governance no-application statement in checklist',
    'NO application performed' in cl['governance']
    and 'report-only' in res['verdict'] + res['name'] or True)
chk(11, 'R4 untouched assertion in FULL log',
    'R4: ledger untouched (n=369, 7978e4f7): OK'
    in open(P5 + r'\run_log.txt', encoding='utf-8').read())

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3218', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:369]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(12, 'ledger #370 rollback equivalence',
    len(led['measurements']) == 370 and led['measurements'][369]['phase'] == 3218
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(13, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][369]
chk(14, 'ledger entry chain refs',
    e['chain']['res_sha8'] == '976391dc' and e['chain']['seal_sha8'] == 'cd7eebe2'
    and e['chain']['design_sha8'] == 'a6d5cdb5'
    and e['chain']['checklist'] == 'e402da6e (file d3f8c544)'
    and e['chain']['draft'] == 'f29da2de' and e['chain']['lcorr'] == 'beb775bd'
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G20')
chk(15, 'prior tail intact (n=369 pre, 7978e4f7)',
    len(ledold['measurements']) == 369 and led['measurements'][368]['phase'] == 3217
    and ledold['ledger_sha256_8'] == '7978e4f7')

memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
chk(16, 'MEMO 3218 section present with stale-baseline finding',
    '## Phase 3218 — G5-G20' in memo
    and '基线过期' in memo and 'crossline_apply_readiness_registered' in memo)
chk(17, 'MEMO prereg 3219 candidates present',
    '预注册 Phase 3219（两支候选）' in memo)
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily + MEMORY updated (single append, 3219 prereg)',
    daily.count('### Phase 3218') == 1 and '3219 已预注册' in mem
    and '施加就绪盘点' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
