# -*- coding: utf-8 -*-
import io, json, os, hashlib
import numpy as np
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3139\omega_p137_'
       'idinteract_portconsume_rewrite_xmat')
out = []
ok = 0
fail = 0
def chk(name, cond, detail=''):
    global ok, fail
    if cond:
        ok += 1
        out.append('PASS %s %s'
                   % (name, detail))
    else:
        fail += 1
        out.append('FAIL %s %s'
                   % (name, detail))

# 1 artifacts on disk
for f in ('result.json', 'design_seal.json',
          'run_log.txt', 'p137_readout.npz'):
    chk('file:' + f,
        os.path.exists(os.path.join(OUT, f)))
raw = io.open(OUT + r'\result.json', 'rb').read()
chk('result_sha8',
    hashlib.sha256(raw).hexdigest()[:8]
    == '7b57b15b')
r = json.loads(raw.decode('utf-8'))
chk('phase', r['phase'] == 3139)
chk('smoke_false', r['smoke'] is False)
chk('xphase_1.0',
    r['part_e']['xphase'] == 1.0)
chk('port_I_med',
    abs(r['part_d']['port_I_med']
        - 0.45175) < 0.01)
chk('iinj_l17',
    abs(r['part_e']['trials']
        ['iinj_l17_d1.0']['chg']
        - 0.328125) < 1e-9)
z = np.load(OUT + r'\p137_readout.npz',
            allow_pickle=False)
chk('npz_keys_11', len(z.files) == 11,
    str(len(z.files)))

# 2 ledger real readback
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas'
    r'\atlas_ledger.json',
    encoding='utf-8'))
e39 = [e for e in led['measurements']
       if e.get('phase') == 3139]
chk('ledger_n276',
    len(led['measurements']) == 276,
    str(len(led['measurements'])))
chk('ledger_3136_entry', len(e39) == 1)
if e39:
    chk('ledger_sha',
        e39[0]['hashes']['result_sha256_8']
        == '7b57b15b')
    chk('ledger_verdict',
        e39[0]['verdict'] == r['verdict'])

# 3 MEMO real readback
m = io.open(ROOT + r'\research\gpt5\docs'
            r'\AGI_GPT5_MEMO.md',
            encoding='utf-8').read()
i = m.rfind('## Phase 3139:')
chk('memo_section', i > 0)
if i > 0:
    sec = m[i:]
    chk('memo_T4_22',
        'T4 第22 Phase' in sec[:80])
    for frag in ('ish2_hard75_fail',
                 'own_over_I', 'cinj_active',
                 'iinj', '0.328', '0.458',
                 '7b57b15b', 'c9af477d',
                 '3140（Ω-P138）预注册',
                 '层位谱相反'):
        chk('memo_has:' + frag[:14],
            frag in sec)
    chk('memo_x3_head2',
        sec.count('不特异于本行身份') >= 3)
    chk('memo_x3_head3',
        sec.count('层位谱相反') >= 3)
    chk('memo_no_placeholder',
        '__HHMM__' not in sec
        and '__VERDICT__' not in sec)

# 4 wlog real readback
wl = io.open(ROOT + r'\.workbuddy\memory'
             r'\2026-09-28.md',
             encoding='utf-8').read()
chk('wlog_closed',
    '闭环：正式跑 2700' in wl)

# 5 workspace MEMORY real readback
mm = io.open(ROOT + r'\.workbuddy\memory'
             r'\MEMORY.md',
             encoding='utf-8').read()
chk('wmem_3139_line', '3139（T4）' in mm)
chk('wmem_max3139',
    'max=3139，下一 3140' in mm)
chk('wmem_old_next_gone',
    'max=3138，下一 3139' not in mm)

out.append('SUMMARY ok=%d fail=%d'
           % (ok, fail))
io.open(ROOT + r'\tests\gpt5_temp'
        r'\p3139_disk_verify.txt', 'w',
        encoding='utf-8').write('\n'.join(out))
print('OK')
