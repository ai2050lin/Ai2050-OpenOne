# -*- coding: utf-8 -*-
"""Phase 3114 disk verification: every closeout write
re-checked on the real disk."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3114'
        r'\omega_p112_write_erase_ablation')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory\2026-09-23.md'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory\2026-09-23.md')
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
SCR_MAIN = ROOT + (r'\tests\glm5'
                   r'\phase3114_omega_p112_write_'
                   r'erase_ablation.py')
SCR_CLOSE = ROOT + (r'\tests\gpt5_temp'
                    r'\phase3114_closeout.py')

ok = []
fail = []


def chk(name, cond):
    (ok if cond else fail).append(name)


def has_any(t, *alts):
    return any(a in t for a in alts)


# 1. result.json
r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
chk('r01 verdict', r['verdict'] ==
    'head_write_not_causal|mlp_write_partial|'
    'erase_not_active')
chk('r02 n_records=2016', r['n_records'] == 2016)
chk('r03 smoke=False', r['smoke'] is False)
chk('r04 d_head +0.0101',
    abs(r['dmp_rel']['L28_top8_head']
        - 0.01014466739237411) < 1e-9)
chk('r05 lin_pred +0.0316',
    abs(r['dmp_rel']['L28_top8_head_linear_pred']
        - 0.03163098640774666) < 1e-9)
chk('r06 d_mlp28 -0.1644',
    abs(r['dmp_rel']['L28_mlp']
        - (-0.16443887776338192)) < 1e-9)
chk('r07 d_mlp24 +0.3460',
    abs(r['dmp_rel']['L24_mlp']
        - 0.3460303786765555) < 1e-9)
chk('r08 d_mlp32 +0.1030',
    abs(r['dmp_rel']['L32_mlp']
        - 0.10298860230091159) < 1e-9)
chk('r09 selfcheck<0.01',
    max(r['selfcheck_rel'].values()) < 0.01)
chk('r10 top8 frozen',
    r['top8_heads']['L28'] ==
    [30, 22, 31, 4, 20, 1, 21, 23])
chk('r11 gates kept',
    r['gates']['head'].startswith('abl_L28_top8'))

# 2. design_seal.json
s = json.load(io.open(OUTD + r'\design_seal.json',
                      encoding='utf-8'))
chk('s01 seal top8', s['top8_head_L28'] ==
    [30, 22, 31, 4, 20, 1, 21, 23])
chk('s02 seal gates', 'le -0.15' in
    s['gates']['head'].replace('<= -0.15', 'le -0.15')
    or '-0.15' in s['gates']['head'])
chk('s03 seal ablations 5',
    s['ablations'] == ['baseline',
                       'abl_L28_top8_head',
                       'abl_L24_mlp', 'abl_L28_mlp',
                       'abl_L32_mlp'])
chk('s04 seal note L24 mlp-only',
    'MLP-only control' in s['note'])

# 3. run_log.txt
rl = io.open(OUTD + r'\run_log.txt',
             encoding='utf-8').read()
chk('l01 baseline mpair 8.2458',
    'mpair=8.2458' in rl)
chk('l02 VERDICT line',
    'VERDICT: head_write_not_causal|'
    'mlp_write_partial|erase_not_active' in rl)
chk('l03 selfcheck head lin=0',
    'zero=1 lin=0.00e+00' in rl)
chk('l04 clean snapshot',
    'clean snapshot stored (3 layers)' in rl)
chk('l05 2016 rebuilt',
    'records rebuilt: 2016 (672 pairs)' in rl)

# 4. npz artifact
npz = OUTD + r'\ablation_readout.npz'
chk('n01 npz exists>0',
    os.path.exists(npz)
    and os.path.getsize(npz) > 1e5)

# 5. closeout_log.txt
cl = io.open(OUTD + r'\closeout_log.txt',
             encoding='utf-8').read()
chk('c01 five lines', len(cl.strip()
    .splitlines()) == 5)
chk('c02 memory updated<3000',
    'memory updated 2930 chars' in cl)

# 6. ledger
led = json.load(io.open(LEDGER, encoding='utf-8'))
m14 = [m for m in led['measurements']
       if m.get('phase') == 3114]
chk('g01 meas3114 exists', len(m14) == 1)
chk('g02 meas verdict', m14[0]['verdict'] ==
    'head_write_not_causal|mlp_write_partial|'
    'erase_not_active')
chk('g03 L14 connects',
    'meas3114_omega_p112_write_erase_ablation'
    in [l for l in led['linkage']
        if l.get('link_id')
        == 'L14_readout_spectrum_cross_model']
    [0]['connects'])
led2 = dict(led)
sha = led2.pop('ledger_sha256_8')
chk('g04 sha self-consistent',
    hashlib.sha256(json.dumps(
        led2, sort_keys=True,
        ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8] == sha)
chk('g05 n_measurements',
    len(led['measurements']) == 251)

# 7. MEMO
memo = io.open(MEMO, encoding='utf-8').read()
tail = memo[-6000:]
chk('m01 header', '## Phase 3114:' in memo)
chk('m02 title', '写/擦两相因果消融' in tail)
chk('m03 key numbers', all(
    has_any(tail, v, v.replace('-', '\u2212'))
    for v in ('+0.0101', '-0.1644', '+0.3460',
              '+0.1030', '8.2458', '11.0991',
              '6.8898', '9.0950')))
chk('m04 frozen heads', '[30,22,31,4,20,1,21,23]'
    in tail)
chk('m05 adversarial', '对抗性写入' in tail)
chk('m06 prereg 3115',
    '联合消融' in tail and 'argnext' in tail)
chk('m07 unicode minus ok',
    has_any(tail, '\u22120.1644', '-0.1644'))
chk('m08 append-only (no 3115 yet)',
    '## Phase 3115:' not in memo)

# 8. wlogs
wd = io.open(WLOG_D, encoding='utf-8').read()
wc = io.open(WLOG_C, encoding='utf-8').read()
chk('w01 wlog D', 'Phase 3114 Omega-P112' in wd)
chk('w02 wlog C', 'Phase 3114 Omega-P112' in wc)

# 9. MEMORY.md
mem = io.open(MEMO_W, encoding='utf-8').read()
chk('e01 len<3000', len(mem) < 3000)
chk('e02 max=3114', 'max=3114' in mem)
chk('e03 chain 3114',
    '## 机制链状态（3114）' in mem)
chk('e04 next 3115',
    '下一 3115' in mem and '联合 MLP 消融' in mem)
chk('e05 old 3113 compressed',
    '伪迹分离+写入端：配对内中位 0.952' in mem)

# 10. scripts exist
chk('p01 main script',
    os.path.exists(SCR_MAIN)
    and os.path.getsize(SCR_MAIN) > 15000)
chk('p02 closeout script',
    os.path.exists(SCR_CLOSE))

rep = ['VERIFY %d/%d PASS'
       % (len(ok), len(ok) + len(fail))]
rep += ['PASS ' + n for n in ok]
rep += ['FAIL ' + n for n in fail]
with io.open(OUTD + r'\disk_verify_out.txt', 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(rep) + '\n')
print('\n'.join(rep[:3]))
