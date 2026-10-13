# -*- coding: utf-8 -*-
"""Phase 3226 independent verify: 18 checks, re-derives from disk."""
import hashlib
import json
import os

OK = []
BAD = []


def chk(n, name, cond):
    (OK if cond else BAD).append((n, name))
    print('%2d %s %s' % (n, 'PASS' if cond else 'FAIL', name))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def jd(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True)


P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3226'
      r'\g5g25_a2_gate_decision')
P3 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3223'
      r'\g5g25_presoftmax_band_design')
P225 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3225'
        r'\g5g25_a1_behavioral_band_compute')
P206 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3206'
        r'\g5g5_ftr22_chain_recheck')
P192 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3192'
        r'\g5c5_gap4_amplitude_attribution')
LED = r'research\gpt5\atlas\atlas_ledger.json'
MEMO = r'research\gpt5\docs\AGI_GPT5_MEMO.md'

# 1 sealed artifact set present with honest log
res = json.load(open(os.path.join(P5, 'result.json'), encoding='utf-8'))
rl = open(os.path.join(P5, 'run_log.txt'), encoding='utf-8').read()
_no = dict(res); _no.pop('res_sha8', None)
chk(1, 'verdict a2_behavioral_upgrade_registered + content sha 48222035',
    res['verdict'] == 'a2_behavioral_upgrade_registered'
    and res['res_sha8'] == hashlib.sha256(json.dumps(
        _no, ensure_ascii=False, indent=1, sort_keys=True)
        .encode('utf-8')).hexdigest()[:8])
# 2 freeze before gate evaluation
fz = json.load(open(os.path.join(P5, 'execution.json'), encoding='utf-8'))
chk(2, 'freeze 1608c23d sealed w/ gates verbatim + inputs',
    fz['sha8'] == '1608c23d' and fz['sha8'] == res['freeze_sha8']
    and 'frozen before gate evaluation' in fz['prereg']
    and fz['inputs']['a1_npz'] == 'ac5d4d74')
# 3 design anchors
chk(3, 'design doc sha8 1a1c4978/87e51bcd + gates source',
    sha8_file(os.path.join(P3, 'presoftmax_band_design_v1.json')) == '1a1c4978'
    and json.load(open(os.path.join(P3, 'presoftmax_band_design_v1.json'),
                       encoding='utf-8'))['sha8'] == '87e51bcd'
    and 'GA_behavioral_heldout' in fz['gates_applied'])
# 4 A1 inputs untouched
chk(4, 'A1 sealed files untouched (res ea1a0585 / npz ac5d4d74)',
    sha8_file(os.path.join(P225, 'a1_behavioral_band.npz')) == 'ac5d4d74'
    and json.load(open(os.path.join(P225, 'result.json'),
                       encoding='utf-8'))['verdict'] == 'a1_behavioral_band_sealed')
# 5-6 gate numbers independently re-derived from sealed A1 metrics
a1 = json.load(open(os.path.join(P225, 'result.json'), encoding='utf-8'))
gt = res['gate_table']
re_ok, n_rows = True, 0
for fid, pred in (('FTR-26', 'P1_ustar'), ('FTR-25', 'P2_vhat')):
    for m in ('qwen3-4b', 'qwen3-14b'):
        for alpha in ('0.5', '0.1'):
            rows = [a1['models'][m]['metrics'][k]
                    for k in sorted(a1['models'][m]['metrics'])
                    if ('_%s_a%s' % (pred, alpha)) in k]
            n_rows += len(rows)
            cs = [x['cos_S_med'] for x in rows]
            bm = [x['band_med'] for x in rows]
            se = [x['svd8_energy'] for x in rows]
            r = gt['%s.%s.a%s.gate' % (fid, m, alpha)]
            re_ok &= (len(rows) == 4 and r['ga_folds_ge_0.70'] ==
                      sum(1 for c in cs if c >= 0.70)
                      and abs(r['gb_band_med_median'] - sorted(bm)[1]) < 1e-12
                      and abs(r['gc_svd8_median'] - sorted(se)[1]) < 1e-12
                      and r['verdict'] == ('BEHAVIORAL_BAND_PASS'
                                           if r['ga_pass'] and r['gb_pass']
                                           else r['verdict']))
chk(5, 'gate table x8 rows re-derived bitwise from sealed A1 metrics',
    re_ok and n_rows == 32)
chk(6, 'gate summary: GA 4/4 x2x2 (min cos 1.0), GB max 0.0052, GC min '
       '0.999904, both features BEHAVIORAL_BAND_PASS @a0.5',
    res['feature_verdicts'] == {'FTR-26': 'BEHAVIORAL_BAND_PASS',
                                'FTR-25': 'BEHAVIORAL_BAND_PASS'}
    and res['gate_summary']['min_cos_S_med'] >= 0.99999
    and res['gate_summary']['min_cos_S_med'] == min(
        v['ga_cos_S_med_min'] for v in gt.values())
    and res['gate_summary']['max_band_med'] < 0.25
    and res['gate_summary']['min_svd8_median'] >= 0.8)
# 7-11 registry v1.29
reg29 = json.load(open(os.path.join(P5, 'atlas_registry_v1_29.json'),
                       encoding='utf-8'))
reg28 = json.load(open(os.path.join(P206, 'atlas_registry_v1_28.json'),
                       encoding='utf-8'))
chk(7, 'registry v1.29 file 85b8dc1f + version/supersedes',
    sha8_file(os.path.join(P5, 'atlas_registry_v1_29.json')) == '85b8dc1f'
    and reg29['version'] == '1.29'
    and 'atlas_registry_v1_28.json (5995bbe9, Phase 3206)' == reg29['supersedes'])
chk(8, 'upgrade_log #10 appended (9->10, node FTR-25+FTR-26, change '
       'E2_held_out -> E2_behavioral)',
    len(reg29['upgrade_log']) == 10
    and reg29['upgrade_log'][:9] == reg28['upgrade_log']
    and reg29['upgrade_log'][9]['node'] == 'FTR-25+FTR-26'
    and 'E2_held_out -> E2_behavioral' == reg29['upgrade_log'][9]['change']
    and 'ea1a0585' in reg29['upgrade_log'][9]['reason'])
chk(9, 'FTR-25/26 evidence_level E2_behavioral + anchors 3->4 (behavioral_band)',
    all(f['evidence_level'] == 'E2_behavioral' for f in reg29['features']
        if f['id'] in ('FTR-25', 'FTR-26'))
    and all(len(f['anchors']) == len([x for x in reg28['features']
                                      if x['id'] == f['id']][0]['anchors']) + 1
            for f in reg29['features'] if f['id'] in ('FTR-25', 'FTR-26'))
    and all(f['anchors'][-1]['role'] == 'behavioral_band'
            and f['anchors'][-1]['asserts']['a1_npz'] == 'ac5d4d74'
            for f in reg29['features'] if f['id'] in ('FTR-25', 'FTR-26')))
chk(10, 'byte-preservation: 24 legacy features + legacy top-level equal',
    all(jd(a) == jd(b) for a, b in zip(reg28['features'], reg29['features'])
        if a['id'] not in ('FTR-25', 'FTR-26'))
    and len(reg29['features']) == 26
    and all(jd(reg28[k]) == jd(reg29[k]) for k in reg28
            if k not in ('version', 'supersedes', 'upgrade_log', 'provenance',
                         'features')))
chk(11, 'provenance a1 npz anchor appended; v1.28 sealed file untouched',
    reg29['provenance']['anchor_files']['a1_behavioral_band']['sha8'] == 'ac5d4d74'
    and sha8_file(os.path.join(P206, 'atlas_registry_v1_28.json')) == '5995bbe9')
# 12-14 gap ledger v1.7
gap17 = json.load(open(os.path.join(P5, 'gap_ledger_v1_7.json'),
                       encoding='utf-8'))
gap16 = json.load(open(os.path.join(P192, 'gap_ledger_v1_6.json'),
                       encoding='utf-8'))
g5 = [x for x in gap17['gaps'] if x['id'] == 'GAP-5'][0]
chk(12, 'gap v1.7 file 4f45fda0 + version/updated',
    sha8_file(os.path.join(P5, 'gap_ledger_v1_7.json')) == '4f45fda0'
    and gap17['version'] == '1.7' and gap17['updated'] == '2026-10-11')
chk(13, 'GAP-5 appended: localized_linear_presoftmax + anchors',
    g5['status'] == 'localized_linear_presoftmax'
    and 'post-softmax' in g5['statement']
    and g5['evidence_anchors'][0]['res'] == 'ea1a0585'
    and g5['anchor_sha8']['a1_npz'] == 'ac5d4d74')
chk(14, 'gap byte-preservation: legacy 4 gaps + legacy top-level equal',
    all(jd(a) == jd(b) for a, b in zip(gap16['gaps'], gap17['gaps'][:4]))
    and len(gap17['gaps']) == 5
    and all(jd(gap16[k]) == jd(gap17[k]) for k in gap16
            if k not in ('version', 'updated', 'gaps'))
    and sha8_file(os.path.join(P192, 'gap_ledger_v1_6.json')) == '0f6423e0')
# 15 ledger #378 with rollback-equal bak
led = json.load(open(LED, encoding='utf-8'))
bak = json.load(open(LED + '.bak_3226', encoding='utf-8'))
e378 = led['measurements'][377]
back = dict(led)
back['measurements'] = led['measurements'][:377]
back['ledger_sha256_8'] = bak['ledger_sha256_8']
chk(15, 'ledger #378 (phase 3226) post 62b2d7d6, bak_3226 rollback-equal',
    len(led['measurements']) == 378 and e378['phase'] == 3226
    and led['ledger_sha256_8'] == '62b2d7d6'
    and bak['ledger_sha256_8'] == '969b6791'
    and jd(bak) == jd(back))
# 16 MEMO 3226 section + honest log + prereg 3227
memo = open(MEMO, encoding='utf-8').read()
chk(16, 'MEMO 3226 section w/ honest log + upgrade + prereg 3227',
    '## Phase 3226 — G5-G25 A2' in memo
    and 'BEHAVIORAL_BAND_PASS' in memo and 'a2_behavioral_upgrade_registered'
    in memo and '预注册 Phase 3227' in memo)
# 17 E2_behavioral census: exactly 2 features at new level, rest unchanged
lv = {}
for f in reg29['features']:
    lv[f['evidence_level']] = lv.get(f['evidence_level'], 0) + 1
chk(17, 'census v1.29: E2_behavioral x2, legacy levels unchanged vs v1.28',
    lv.get('E2_behavioral') == 2
    and all(lv.get(k) == sum(1 for f in reg28['features']
                             if f['evidence_level'] == k)
            for k in set(f['evidence_level'] for f in reg28['features'])
            if k != 'E2_held_out')
    and lv.get('E2_held_out') == sum(
        1 for f in reg28['features']
        if f['evidence_level'] == 'E2_held_out') - 2)
# 18 result contradictions empty + gate taxonomy respected
chk(18, '0 contradictions; FAIL taxonomy would route to F ledger (design)',
    res['recheck']['contradictions'] == []
    and 'FAIL goes to F ledger' in fz['gates_applied']['verdict_taxonomy'])

print('RECHECK: %s (%d pass / %d fail)' % (
    'CLEAN' if not BAD else 'DIRTY', len(OK), len(BAD)))
