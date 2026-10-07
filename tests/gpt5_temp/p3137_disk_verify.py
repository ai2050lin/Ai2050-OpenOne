# -*- coding: utf-8 -*-
import io, json, os, hashlib
import numpy as np
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3137\omega_p135_modecoop_'
       r'k0anat_coorddecomp_l35recheck')
out = []
ok = 0
fail = 0
def chk(name, cond, detail=''):
    global ok, fail
    if cond:
        ok += 1
        out.append('PASS %s %s' % (name, detail))
    else:
        fail += 1
        out.append('FAIL %s %s' % (name, detail))

# 1 result.json + seal + npz on disk
for f in ('result.json', 'design_seal.json',
          'run_log.txt', 'p135_readout.npz'):
    p = os.path.join(OUT, f)
    chk('file:' + f, os.path.exists(p))
raw = io.open(OUT + r'\result.json', 'rb').read()
chk('result_sha8', hashlib.sha256(raw).hexdigest()[:8] == '4f8055bb')
r = json.loads(raw.decode('utf-8'))
chk('phase', r['phase'] == 3137)
chk('smoke_false', r['smoke'] is False)
chk('verdict_tags', r['verdict'].count('|') == 8)
z = np.load(OUT + r'\p135_readout.npz', allow_pickle=False)
chk('npz_keys_29', len(z.files) == 29, str(len(z.files)))
chk('npz_k0only_sha', hashlib.sha256(z['fstep_w8_k0only'].tobytes()).hexdigest()[:8] == '9ba70f6f')
chk('npz_k0only_eq_po17s1', hashlib.sha256(z['fstep_w8_k0only'].tobytes()).hexdigest() == hashlib.sha256(z['fstep_po_l17_s1.0'].tobytes()).hexdigest())

# 2 ledger real readback
led = json.load(io.open(ROOT + r'\research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
e37 = [e for e in led['measurements'] if e.get('phase') == 3137]
chk('ledger_n274', len(led['measurements']) == 274, str(len(led['measurements'])))
chk('ledger_3136_entry', len(e37) == 1)
if e37:
    chk('ledger_sha', e37[0]['hashes']['result_sha256_8'] == '4f8055bb')
    chk('ledger_verdict_head', e37[0]['verdict'] == r['verdict'])

# 3 MEMO real readback
m = io.open(ROOT + r'\research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
i = m.rfind('## Phase 3137:')
chk('memo_section', i > 0)
if i > 0:
    sec = m[i:]
    chk('memo_T4_20', 'T4 第20 Phase' in sec[:80])
    for frag in ('mode_gap_all', 'k0_negative_confirmed', 'co36_l17_flat',
                 '0.089286', '9ba70f6f', '4f8055bb', '3138 预注册',
                 'FINGERPRINT_PARADIGM_PLAN'):
        chk('memo_has:' + frag[:16], frag in sec)
    chk('memo_x3_head1', sec.count('模式×层位×剂量定律') >= 3)
    chk('memo_x3_head2', sec.count('k0 负贡献确认') >= 3)
    chk('memo_x3_head3', sec.count('错配坐标平坦') >= 3)

# 4 wlog real readback
wl = io.open(ROOT + r'\.workbuddy\memory\2026-09-28.md', encoding='utf-8').read()
chk('wlog_closed', '闭环：正式跑 13998' in wl)

# 5 workspace MEMORY real readback
mm = io.open(ROOT + r'\.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk('wmem_3137_line', '3137（T4）' in mm)
chk('wmem_max3137', 'max=3137，下一 3138' in mm)
chk('wmem_next_updated', 'Absolute State Bank' in mm)
chk('wmem_old_next_gone', 'prompt-only×allstep×层位×剂量对照' not in mm)

out.append('SUMMARY ok=%d fail=%d' % (ok, fail))
io.open(ROOT + r'\tests\gpt5_temp\p3137_disk_verify.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
