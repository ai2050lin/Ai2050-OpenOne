# -*- coding: utf-8 -*-
# p3126_meta_fix.py
# Fix seal.json filename typo (actual file is
# design_seal.json) in: (1) ledger artifacts of
# meas3126 + ledger_sha256_8 recompute, (2) MEMO
# Phase-3126 artifact list. Dual wlogs get an
# APPENDED meta-fix line (history preserved; the
# new sha8 string satisfies verify F_wlogs).
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOGS = [
    ROOT + r'\.workbuddy\memory'
    r'\2026-09-24.md',
    (r'C:\Users\Admin\WorkBuddy'
     r'\2026-09-17-01-30-05'
     r'\.workbuddy\memory'
     r'\2026-09-24.md')]
VF = (ROOT + r'\tests\gpt5_temp'
      r'\p3126_meta_fix_out.txt')
o = []

MID = ('meas3126_omega_p124_glm4_'
       'anchoredlast_regen_writechain')
OLD_SEAL = ('phase3126/omega_p124_glm4_'
            'anchoredlast_regen_'
            'writechain/seal.json')
NEW_SEAL = ('phase3126/omega_p124_glm4_'
            'anchoredlast_regen_'
            'writechain/design_seal.json')

# ---- 1. ledger: artifacts.seal + sha8 ----
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
m6 = [m for m in led['measurements']
      if m.get('meas_id') == MID]
o.append('meas found: %d' % len(m6))
assert len(m6) == 1
old_v = m6[0]['artifacts']['seal']
o.append('old seal path: %s' % old_v)
assert old_v == OLD_SEAL, old_v
m6[0]['artifacts']['seal'] = NEW_SEAL
old_sha = led.get('ledger_sha256_8')
o.append('old ledger sha8: %s' % old_sha)
led.pop('ledger_sha256_8', None)
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
new_sha = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_sha
o.append('new ledger sha8: %s' % new_sha)
with io.open(LEDGER, 'w',
             encoding='utf-8') as f:
    json.dump(led, f, ensure_ascii=False,
              indent=1)
o.append('ledger written (indent=1)')

# ---- 2. MEMO: artifact list filename ----
memo = io.open(MEMO, encoding='utf-8').read()
OLD_L = (u'\uff08result.json\u3001seal.json'
         u'\u3001run_log.txt'
         u'\u3001p124_readout.npz\uff09')
NEW_L = (u'\uff08result.json'
         u'\u3001design_seal.json'
         u'\u3001run_log.txt'
         u'\u3001p124_readout.npz\uff09')
cnt2 = memo.count(OLD_L)
o.append('memo old list count: %d' % cnt2)
assert cnt2 == 1, cnt2
memo = memo.replace(OLD_L, NEW_L)
io.open(MEMO, 'w', encoding='utf-8').write(
    memo)
o.append('memo updated (1 replace)')

# ---- 3. dual wlogs: APPEND meta-fix line ----
fix_line = ('- Phase 3126 meta-fix: closeout '
            'wrote seal.json in ledger '
            'artifacts/MEMO but actual file '
            'is design_seal.json; artifacts '
            'path fixed in place, MEMO '
            'artifact list fixed in place, '
            'ledger sha8 %s -> %s '
            '(filename typo only, no '
            'scientific content change); '
            'this line appended for '
            'append-only history.\n'
            % (old_sha, new_sha))
n_wl = 0
for wl in WLOGS:
    try:
        t = io.open(wl, encoding='utf-8').read()
    except IOError as e:
        o.append('wlog read fail %s: %r'
                 % (wl, e))
        continue
    has_old = ('sha=' + old_sha) in t
    o.append('wlog %s has_old_sha=%s'
             % (wl, has_old))
    if new_sha not in t:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(fix_line)
        n_wl += 1
        o.append('wlog appended: %s' % wl)
    else:
        o.append('wlog already has new sha: %s'
                 % wl)
o.append('wlogs appended: %d' % n_wl)

o.append('META_FIX_OK')
io.open(VF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('meta fix done')
