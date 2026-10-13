# -*- coding: utf-8 -*-
"""Phase 3212 self-check + independent verify (disk re-read; 18 checks)."""
import hashlib
import json


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

B = r'tests\glm5\result\rdc_query_construction_20260913'
P2 = B + r'\phase3212\g5g12_ftr04_attribution_closed'
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))


res = json.load(open(P2 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(P2 + r'\smoke_result.json', encoding='utf-8'))
exe = json.load(open(P2 + r'\execution.json', encoding='utf-8'))

chk(1, 'FULL verdict closed', res['verdict'].startswith('ftr04_glm4_attribution_closed'))
chk(2, 'gates clean 0 contradictions',
    res['gates']['recheck_clean'] is True and res['gates']['n_contradictions'] == 0)
chk(3, 'SMOKE final run clean', smoke['gates']['recheck_clean'] is True)
chk(4, 'zero_gpu + report_only + name + design',
    res.get('zero_gpu') is True and res.get('report_only') is True
    and res.get('name') == 'g5g12_ftr04_attribution_closed'
    and res.get('design_sha8') == 'b2e6a642')
chk(5, 'FULL content locks', res['res_sha8'] == '07041998' and res['seal_sha8'] == '73024f1a')

rc = res['recheck']
r3181 = json.load(open(B + r'\phase3181\g5b6_ftr04_glm4_attribution\result.json', encoding='utf-8'))
r3202 = json.load(open(B + r'\phase3202\g5f1_ftr04_recheck\result.json', encoding='utf-8'))
reg = json.load(open(B + r'\phase3206\g5g5_ftr22_chain_recheck\atlas_registry_v1_28.json', encoding='utf-8'))
f04 = [f for f in reg['features'] if f['id'] == 'FTR-04'][0]
a3181 = [a for a in f04['anchors'] if a['src'] == 'p3181'][0]
pm = r3181['per_model']

chk(6, 'A1 3181 locks + verdict + anchor bitwise (disk re-read)',
    r3181['res_sha8'] == '1e95a19d' and r3181['seal_sha8'] == '462fd221'
    and r3181['design_sha8'] == '7c567972'
    and r3181['verdict'].startswith('noise_level')
    and all(a3181['asserts'][k] == v for k, v in [
        ('kappa.qwen3-4b', pm['qwen3-4b']['kappa']),
        ('kappa.qwen3-14b', pm['qwen3-14b']['kappa']),
        ('kappa.glm4-9b', pm['glm4-9b']['kappa']),
        ('proj_ratio.glm4-9b', pm['glm4-9b']['proj_ratio']),
        ('mean_margin.glm4-9b', pm['glm4-9b']['mean_margin'])])
    and a3181['asserts']['gate_Ga_geometry'] is False
    and a3181['asserts']['gate_Gb_readout'] is False)
chk(7, 'A2 3202 locks + decomposition re-derived 1e-12',
    r3202['res_sha8'] == 'af81e9f4' and r3202['seal_sha8'] == '480bd271'
    and abs(r3202['recheck']['decomposition']['loeo_gap']
            - (pm['qwen3-14b']['loeo_acc'] - pm['glm4-9b']['loeo_acc'])) <= 1e-12
    and abs(r3202['recheck']['decomposition']['margin_ratio']
            - pm['glm4-9b']['mean_margin'] / pm['qwen3-4b']['mean_margin']) <= 1e-12
    and r3202['recheck']['decomposition']['geometry_band_ok'] is True)
chk(8, 'A2 registry scope_limits 3202 append verbatim',
    all(frag in f04['scope_limits'] for frag in [
        '3202 G5-F1 FTR-04 glm4 弱特异性单元级分解复读',
        'LOEO 留出读出准确率最低 0.4800 vs 0.6019/0.5981（gap 0.1181 ≥0.10 钉定 bar）',
        '支撑 3181 noise_level 定判']))
chk(9, 'A3 E1_repeatable maintained + chain counter_evidence',
    f04['evidence_level'] == 'E1_repeatable'
    and any('3176 arm (a) FAILED gate' in c for c in f04['counter_evidence'])
    and any(c.startswith('3181 归因定判') for c in f04['counter_evidence']))
chk(10, 'A3 dispositions closed@3202 on result',
    rc['dispositions']['FTR-04 glm4 attribution']['status'] == 'closed_at_3202'
    and rc['dispositions']['FTR-04 glm4 attribution']['recommendation'].startswith('no registry'))
chk(11, 'MEMO 3181/3202 verbatim anchors on disk',
    'noise_level_not_attributable' in memo_seg(r'research\gpt5\docs\AGI_GPT5_MEMO.md', 3181)
    if False else True)  # placeholder replaced below

memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
i3181 = memo.find('## Phase 3181'); j3181 = memo.find('\n## ', i3181 + 1)
i3202 = memo.find('## Phase 3202'); j3202 = memo.find('\n## ', i3202 + 1)
ok.pop()
chk(11, 'MEMO 3181/3202 verbatim anchors on disk',
    'noise_level_not_attributable' in memo[i3181:j3181]
    and all(kw in memo[i3202:j3202] for kw in
            ['drift==0', '1.9661', '0.1181', '读出不一致', '单元级分解']))

led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
ledold = json.load(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3212', encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:363]
back['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(12, 'ledger #364 rollback equivalence',
    len(led['measurements']) == 364 and led['measurements'][363]['phase'] == 3212
    and json.dumps(ledold, ensure_ascii=False, indent=1, sort_keys=True) ==
        json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True))
tmp = json.loads(json.dumps(led))
tmp['ledger_sha256_8'] = ledold['ledger_sha256_8']
chk(13, 'ledger post-sha convention',
    hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])
e = led['measurements'][363]
chk(14, 'ledger entry chain refs',
    e['chain']['res_sha8'] == '07041998' and e['chain']['seal_sha8'] == '73024f1a'
    and e['chain']['design_sha8'] == 'b2e6a642'
    and e['chain']['r3181'].startswith('5ab51de6')
    and e['chain']['r3202'].startswith('af7d8a70')
    and e['chain']['ledger_pre'] == ledold['ledger_sha256_8']
    and e['prereg_id'] == 'G5-G12')
chk(15, 'prior tail intact (n=363 pre, d637e53f)',
    len(ledold['measurements']) == 363 and led['measurements'][362]['phase'] == 3211
    and ledold['ledger_sha256_8'] == 'd637e53f')
chk(16, 'A4 untouched assertion recorded in FULL run log',
    'A4: ledger untouched re-assertion OK (n=363, d637e53f)'
    in open(P2 + r'\run_log.txt', encoding='utf-8').read())
chk(17, 'sealed sources untouched (5ab51de6 / af7d8a70)',
    sha8(B + r'\phase3181\g5b6_ftr04_glm4_attribution\result.json') == '5ab51de6'
    and sha8(B + r'\phase3202\g5f1_ftr04_recheck\result.json') == 'af7d8a70')
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'MEMO + daily + MEMORY updated (single append, 3213 prereg)',
    '## Phase 3212 — G5-G12' in memo and 'UI v1.8' in memo and 'FTR-15' in memo
    and daily.count('### Phase 3212') == 1 and '3213 已预注册' in mem)

for i, name, c in ok:
    print('#%-2d %s %s' % (i, 'OK  ' if c else 'FAIL', name))
print('PASS %d/18' % sum(c for _, _, c in ok))
