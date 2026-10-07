# -*- coding: utf-8 -*-
"""p3147 independent disk verify."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D47 = (RDIR + r'\phase3147'
       + r'\omega_p145_tbsym_co50exk_'
       + r'negmech_v1micro')
D46 = (RDIR + r'\phase3146'
       + r'\omega_p144_taillocus_'
       + r'histfield_pcresdose_cleanclip')
PASS = []
FAIL = []


def chk(name, cond):
    (PASS if cond else FAIL).append(name)


# 1. result.json
p = D47 + r'\result.json'
raw = io.open(p, 'rb').read()
r = json.loads(raw.decode('utf-8'))
chk('result_exists', True)
chk('result_sha8',
    hashlib.sha256(raw).hexdigest()[:8]
    == '63b9885d')
chk('result_seal',
    r['seal_sha8'] == '66a7b2d9')
chk('result_smoke_false',
    r['smoke'] is False)
chk('result_runtime',
    abs(r['runtime_s'] - 13802.0) < 60)
V = r['verdict']
for tag in ('a_3146_ok', 'repro_bit_11',
            'repro_bit_ok',
            'dvec19_repro_6a0332',
            'field_self_ok', 'z35_cos17_ok',
            'resid_anchor_ok',
            'tail_sign_cross',
            'tb15_sym_pos',
            'tail_sub_bot10',
            'co50ex_threshold',
            'co50ex_locus_top',
            'enrich_inverse_absent',
            'neg_late_format_shift',
            'neg_repro_3146',
            'v1_micro_continuous',
            'v1_amp_stable', 'xphase_ok',
            'coverage_full'):
    chk('verdict_%s' % tag, tag in V)

# 2. part values
chk('xphase_both_1',
    r['part_a']['xphase_P'] == 1.0
    and r['part_a']['xphase_A1'] == 1.0)
chk('dvec19_sha',
    r['part_field']['dvec19_sha8']
    == '6a0332a6')
chk('bit11_all_match',
    all(v['match'] for v in
        r['part_bits']['bit_anchors']
        .values())
    and len(r['part_bits']
            ['bit_anchors']) == 11)
ps = r['part_s']
chk('tail_cross_vals',
    abs(ps['tail_pos']['2'] - 0.2578125)
    < 1e-9
    and abs(ps['tail_neg']['4']
            - 0.3046875) < 1e-9)
chk('tb15_pos_d2',
    abs(ps['tb15_pos_d2'] - 0.359375)
    < 1e-9)
chk('tb15_neg_d2',
    abs(ps['tb15_neg_d2'] - 0.09375)
    < 1e-9)
chk('sub_bot10',
    abs(ps['chg_tb10_d4'] - 0.2109375)
    < 1e-9
    and abs(ps['chg_tb5_d4'] - 0.15625)
    < 1e-9)
chk('tailpos2_repro',
    ps['tailpos_d2_repro'] is True)
px = r['part_x']
chk('x_dose',
    abs(px['dose_curve']['0.5']
        - 0.140625) < 1e-9
    and abs(px['dose_curve']['4']
            - 0.9921875) < 1e-9)
chk('x_ktop15',
    abs(px['k_top']['15'] - 0.953125)
    < 1e-9)
chk('x_order_ex_len',
    len(px['order_ex']) == 50
    and px['order_ex'][0] == 2530)
pn = r['part_n']
chk('n_repro',
    abs(pn['chg_neg_d05'] - 0.2421875)
    < 1e-9
    and pn['n_repro_ok'] is True)
chk('n_fsteps',
    pn['fsteps'] == [4, 6, 4, 4, 4, 4, 6, 4])
chk('n_dlog_step2',
    abs(pn['dlog_traj']['2']
        - 1.4527831124141812) < 1e-6)
chk('n_wdn_flat',
    all(abs(v) < 0.05 for v in
        pn['wdn_traj'].values()))
pv = r['part_v']
chk('v_micro_cont',
    abs(pv['micro']['v_base_a005']
        - 0.0703125) < 1e-9
    and pv['micro_min'] > 0.05)

# 3. seal + run log
chk('seal_file',
    os.path.exists(D47
                   + r'\design_seal.json'))
chk('run_log',
    os.path.exists(D47 + r'\run_log.txt'))
chk('npz',
    os.path.exists(D47
                   + r'\p145_readout.npz'))
lg = io.open(D47 + r'\run_log.txt',
             encoding='utf-8').read()
chk('log_verdict', 'VERDICT:' in lg)
chk('log_done', 'DONE' in lg)
chk('log_no_traceback',
    'Traceback' not in lg)
chk('ckpt_cleaned',
    not os.path.exists(D47
                       + r'\p145_ckpt.pkl'))

# 4. ledger
LP = (ROOT + r'\research\gpt5\atlas'
      + r'\atlas_ledger.json')
led = json.load(io.open(LP,
                        encoding='utf-8'))
chk('ledger_n284',
    len(led['measurements']) == 284)
chk('ledger_last_3147',
    led['measurements'][-1]['phase']
    == 3147)
chk('ledger_sha',
    led['measurements'][-1]['result_sha8']
    == '63b9885d')

# 5. MEMO
MEMO = (ROOT + r'\research\gpt5\docs'
        + r'\AGI_GPT5_MEMO.md')
tm = io.open(MEMO, encoding='utf-8').read()
chk('memo_3147', '## Phase 3147:' in tm)
chk('memo_3148_prereg',
    '3148（Ω-P146）预注册' in tm)
chk('memo_3147_sha',
    '63b9885d' in tm)

# 6. daily log
DP = (ROOT + r'\.workbuddy\memory'
      + r'\2026-09-30.md')
td = io.open(DP, encoding='utf-8').read()
chk('daily_3147', 'P145）闭环' in td)
chk('daily_tail_clean',
    not td.rstrip().endswith(chr(92)
                             + 'n'))

# 7. MEMORY.md
MP = (ROOT + r'\.workbuddy\memory'
      + r'\MEMORY.md')
tmm = io.open(MP, encoding='utf-8').read()
chk('mem_max_3147',
    '- max=3147，' in tmm)
for tag in ('tail_sign_cross',
            'co50ex_threshold',
            'co50ex_locus_top',
            'neg_late_format_shift',
            'v1_micro_continuous',
            '63b9885d'):
    chk('mem_%s' % tag, tag in tmm)

# 8. 3146 untouched
raw46 = io.open(D46 + r'\result.json',
                'rb').read()
chk('res46_untouched',
    hashlib.sha256(raw46).hexdigest()[:8]
    == 'e5ed3181')

print('PASS=%d FAIL=%d' % (len(PASS),
                           len(FAIL)))
for f in FAIL:
    print('FAIL:', f)
io.open(
    r'D:\AI2050\Ai2050-OpenOne\tests'
    r'\gpt5_temp\p3147_verify_out.txt',
    'w', encoding='utf-8').write(
    'PASS=%d FAIL=%d\n' % (len(PASS),
                           len(FAIL))
    + '\n'.join(FAIL))
