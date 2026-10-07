# -*- coding: utf-8 -*-
"""p3148 independent disk verify."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913'
        + r'\phase3148'
        + r'\omega_p146_xcross_'
        + r'headsrc_uncancel_v1sym')
SCRIPT = (ROOT + r'\tests\glm5'
          + r'\phase3148_omega_p146_xcross_'
          + r'headsrc_uncancel_v1sym.py')
MEMO = (ROOT + r'\research\gpt5\docs'
        + r'\AGI_GPT5_MEMO.md')
DAILY = (ROOT + r'\.workbuddy\memory'
         + r'\2026-09-30.md')
MEM = (ROOT + r'\.workbuddy\memory'
       + r'\MEMORY.md')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          + r'\atlas_ledger.json')

ok = 0
bad = []


def chk(name, cond):
    global ok
    if cond:
        ok += 1
        print('PASS %s' % name)
    else:
        bad.append(name)
        print('FAIL %s' % name)


# 1 result.json
p = BASE + r'\result.json'
raw = io.open(p, 'rb').read()
sha = hashlib.sha256(raw).hexdigest()[:8]
chk('result_sha8_29327d0e',
    sha == '29327d0e')
r = json.loads(raw.decode('utf-8'))
chk('result_phase', r['phase'] == 3148)
chk('result_not_smoke',
    r['smoke'] is False)
v = r['verdict']
for tag in ('a_3147_ok', 'repro_bit_18',
            'repro_bit_ok',
            'dvec19_repro_6a0332',
            'field_self_ok',
            'z35_cos17_ok',
            'resid_anchor_ok',
            'tail_xcross_located',
            'xcross_fstep_poslate',
            'h_pair_sub',
            'head_conc_moderate',
            'src_top_partial',
            'uncancel_insufficient',
            'v1_sym_mixed', 'xphase_ok',
            'coverage_full'):
    chk('verdict_%s' % tag, tag in v)
chk('seal_sha8', r['seal_sha8'] ==
    '87842206')
chk('xphase_pair',
    r['part_a']['xphase_P'] == 1.0
    and r['part_a']['xphase_A1'] == 1.0)
ba = r['part_bits']['bit_anchors']
chk('bit_count_18', len(ba) == 18)
chk('bit_all_match',
    all(ba[k]['match'] is True
        for k in ba))
chk('res_link_63b9885d',
    r['part_a']['res47_sha8'] ==
    '63b9885d')
# key numbers
x2 = r['part_x2']
chk('x2_interval',
    x2['xcross_interval'] == [3.0, 3.5])
chk('x2_pos25',
    abs(x2['pos_curve']['2.5']
        - 0.265625) < 1e-9)
chk('x2_neg35',
    abs(x2['neg_curve']['3.5']
        - 0.2578125) < 1e-9)
chk('x2_diff35_neg',
    x2['diff_curve']['3.5'] < -0.09)
chk('x2_poslate',
    x2['flip_pos_d1']['med_fstep'] == 4.0
    and x2['flip_neg_d4']['med_fstep']
    == 0.0)
h = r['part_h']
chk('h_pair_sub',
    abs(h['pair_d2'] - 0.1484375) < 1e-9
    and h['h_pair'] == 'h_pair_sub')
chk('h_share_0174',
    abs(h['share_top2_vs_co50ex_d2']
        - 0.1743119266055046) < 1e-9)
chk('h_frac_0269',
    abs(h['frac_top2']
        - 0.2689969539642334) < 1e-9)
chk('h_rank2530_10',
    h['rank_dv19']['2530'] == 10)
u = r['part_u']
chk('u_cancel_0820',
    abs(u['chg_cancel']
        - 0.8203125) < 1e-9)
chk('u_rand_0203',
    abs(u['chg_rand'] - 0.203125) < 1e-9)
chk('u_wdn_0992',
    abs(u['chg_wdn']
        - 0.9921875) < 1e-9)
v2 = r['part_v2']
chk('v2_amp',
    abs(v2['chg_amp025'] - 0.1875) < 1e-9
    and abs(v2['chg_amp050']
            - 0.3828125) < 1e-9)
chk('v2_sym',
    abs(v2['sym25']
        - 0.631578947368421) < 1e-9
    and abs(v2['sym50']
            - 0.5764705882352941) < 1e-9)

# 2 seal
sp = BASE + r'\design_seal.json'
seal = json.load(io.open(sp,
                         encoding='utf-8'))
chk('seal_phase', seal['phase'] == 3148)
chk('seal_not_smoke',
    seal['smoke'] is False)
chk('seal_head_coords',
    seal['constants']['HEAD_COORDS']
    == [2530, 3755])
chk('seal_nbit',
    seal['anchors']['n_bit_tot'] == 18)

# 3 npz
import numpy as np
npz = np.load(BASE + r'\p146_readout.npz',
              allow_pickle=False)
chk('npz_keys',
    all(k in npz.files for k in (
        'head_contrib', 'order_head',
        'pos_curve', 'neg_curve',
        'sym_ratios', 'pc1_res')))
chk('npz_head_shape',
    npz['head_contrib'].shape
    == (32, 32, 2))
chk('npz_pos',
    abs(float(npz['pos_curve'][1])
        - 0.2578125) < 1e-9)

# 4 run_log
rl = io.open(BASE + r'\run_log.txt',
             encoding='utf-8').read()
chk('log_verdict',
    'VERDICT: ' + v in rl)
chk('log_bit18',
    'V2 replay: +2 bit anchors (total '
    '18/18)' in rl)
chk('log_n2_frozen',
    'N2 frozen-list match ok (3147)'
    in rl)
chk('log_xgate',
    'tail_xcross_located (interval '
    '[3.0, 3.5])' in rl)
chk('log_no_ckpt',
    not os.path.exists(BASE
                       + r'\p146_ckpt.pkl'))

# 5 smoke dir
smoke = BASE + r'\smoke'
chk('smoke_result',
    os.path.exists(smoke
                   + r'\result.json'))

# 6 script
sc = io.open(SCRIPT, encoding='utf-8').read()
chk('script_rev3148a',
    'rev-3148a patch1' in sc)
chk('script_ncap4',
    'NCAP = 4 if SMOKE else 128' in sc)
chk('script_ledger_285',
    'LEDGER_N = 284' in sc)

# 7 ledger
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
chk('ledger_n_285',
    len(led['measurements']) == 285)
last = led['measurements'][-1]
chk('ledger_last_3148',
    last['phase'] == 3148
    and last['result_sha8'] == '29327d0e')

# 8 MEMO
mt = io.open(MEMO, encoding='utf-8').read()
chk('memo_3148_title',
    '## Phase 3148: ' in mt)
chk('memo_3149_prereg',
    '3149（Ω-P147）预注册' in mt)
chk('memo_3148_sha',
    '29327d0e' in mt)
chk('memo_one_3148',
    mt.count('## Phase 3148:') == 1)

# 9 daily
dt = io.open(DAILY, encoding='utf-8').read()
chk('daily_3148',
    'P146）闭环' in dt)
chk('daily_3148_once',
    dt.count('P146）闭环') == 1)

# 10 MEMORY
mm = io.open(MEM, encoding='utf-8').read()
chk('mem_max_3148',
    '- max=3148，' in mm)
chk('mem_no_3147_head',
    '- max=3147，' not in mm)
chk('mem_xcross',
    'tail_xcross_located' in mm)
chk('mem_uncancel',
    'uncancel_insufficient' in mm)

print('---')
print('VERIFY %d/%d PASS'
      % (ok, ok + len(bad)))
if bad:
    print('FAILED:', bad)
    raise SystemExit(1)
print('DISK VERIFY 3148 ALL PASS')
