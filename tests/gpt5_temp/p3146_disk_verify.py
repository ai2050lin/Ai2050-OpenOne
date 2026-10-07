# -*- coding: utf-8 -*-
"""p3146 disk verify: independent
re-verification of all closeout writes
from the real disk. Read-only."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D46 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3146'
       r'\omega_p144_taillocus_histfield_'
       r'pcresdose_cleanclip')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
DLOG = (ROOT + r'\.workbuddy\memory'
        r'\2026-09-30.md')
WMEM = (ROOT + r'\.workbuddy\memory'
        r'\MEMORY.md')
SCRIPT = (ROOT + r'\tests\glm5'
          r'\phase3146_omega_p144_'
          r'taillocus_histfield_'
          r'pcresdose_cleanclip.py')

results = []


def chk(name, cond):
    results.append((name, bool(cond)))


# 1. result.json integrity
raw = io.open(
    D46 + r'\result.json', 'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
chk('res_sha8_e5ed3181',
    sha8 == 'e5ed3181')
r = json.loads(raw.decode('utf-8'))
chk('res_seal_ed8c8ede',
    r['seal_sha8'] == 'ed8c8ede')
chk('res_smoke_false',
    r['smoke'] is False)
chk('res_verdict_bit14',
    'repro_bit_14|repro_bit_ok'
    in r['verdict'])
chk('res_verdict_full',
    all(t in r['verdict'] for t in (
        'tail_locus_bot15',
        'l39_jump_downstream',
        'pcres_gap_mono',
        'neg_flip_late',
        'v1_no_clean_window',
        'hist_onset_l29',
        'xphase_ok',
        'coverage_full')))
chk('res_bit14_all_match',
    all(v['match'] for v in
        r['part_cleanclip']
        ['bit_anchors'].values())
    and len(r['part_cleanclip']
            ['bit_anchors']) == 14)
chk('res_xphase_1',
    r['part_a']['xphase_P'] == 1.0
    and r['part_a']['xphase_A1'] == 1.0)
chk('res_dvec19_sha',
    r['part_field']
    ['dvec19_sha8'] == '6a0332a6')
chk('res_gap_mono_vals',
    abs(r['part_pcres']['gap_by_dose']
        ['0.25'] - 0.1484375) < 1e-9
    and abs(r['part_pcres']
            ['gap_by_dose']['0.5']
            - 0.5390625) < 1e-9
    and abs(r['part_pcres']
            ['gap_by_dose']['1.0']
            - 0.5625) < 1e-9)
chk('res_locus_vals',
    abs(r['part_dose']
        ['chg_ttop10_d4']
        - 0.2421875) < 1e-9
    and abs(r['part_dose']
            ['chg_tbot15_d4']
            - 0.359375) < 1e-9)
chk('res_windows_dirty',
    r['part_cleanclip']['windows']
    == {'a025': 0.296875,
        'a050': 0.6640625,
        'deco': 0.78125})
# npz + seal + log exist
chk('npz_exists', os.path.exists(
    D46 + r'\p144_readout.npz'))
chk('seal_exists', os.path.exists(
    D46 + r'\design_seal.json'))
logt = io.open(
    D46 + r'\run_log.txt',
    encoding='utf-8').read()
chk('log_done', 'DONE' in logt)
chk('log_no_traceback',
    'Traceback' not in logt)
chk('log_bit14',
    'B1 3142 repro: 3/3 bit-match'
    in logt)
chk('ckpt_cleaned',
    not os.path.exists(
        D46 + r'\p144_ckpt.pkl'))
# script on disk
st = io.open(SCRIPT,
             encoding='utf-8').read()
chk('script_no_rows_bug',
    '_rows' not in st.replace(
        'div_rows', '').replace(
        'tf_rows', '').replace(
        'tf_pc1_rows', '').replace(
        'rows_', ''))
chk('script_d1_naming',
    "e_pcres38_d1.0_pos" in st)

# 2. ledger
led = json.load(io.open(
    LEDGER, encoding='utf-8'))
chk('ledger_n_283',
    len(led['measurements']) == 283)
e3146 = [m for m in
         led['measurements']
         if m.get('phase') == 3146]
chk('ledger_entry_3146',
    len(e3146) == 1
    and e3146[0]['res_sha8'] == sha8)

# 3. MEMO
mt = io.open(MEMO,
             encoding='utf-8').read()
chk('memo_3146_section',
    '## Phase 3146:' in mt)
chk('memo_3146_prereg',
    '3147（Ω-P145）预注册' in mt)
chk('memo_3146_anchors',
    'e5ed3181' in mt
    and 'ed8c8ede' in mt
    and 'ledger n=283' in mt)
chk('memo_3146_tags',
    all(k in mt for k in (
        'tail_locus_bot15',
        'l39_jump_downstream',
        'pcres_gap_mono',
        'v1_no_clean_window')))

# 4. daily log
dt = io.open(DLOG,
             encoding='utf-8').read()
chk('daily_3146',
    'Phase 3146（Ω-P144）闭环' in dt)
chk('daily_3146_sha',
    'e5ed3181' in dt)

# 5. MEMORY.md
wm = io.open(WMEM,
             encoding='utf-8').read()
chk('wmem_3146_line',
    'max=3146，下一 3147' in wm)
chk('wmem_tags',
    'tail_locus_bot15' in wm
    and 'v1_no_clean_window' in wm)
chk('wmem_no_3145_stale',
    'max=3145' not in wm)

npass = sum(1 for _, ok in results
            if ok)
for name, ok in results:
    print('%-26s %s'
          % (name, 'PASS' if ok
             else 'FAIL'))
print('%d/%d PASS'
      % (npass, len(results)))
