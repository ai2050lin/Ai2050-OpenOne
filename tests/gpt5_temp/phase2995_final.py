# -*- coding: utf-8 -*-
"""Phase 2995 final disk verification."""
import hashlib
import io
import json
import os

out = []

# ledger self-consistency
led = json.load(io.open(
    r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
    r'\atlas_ledger.json', encoding='utf-8'))
hc = led.pop('ledger_sha256_8')
calc = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
out.append('ledger self=%s n=%d L14=%d p2995=%s' % (
    calc == hc, len(led['measurements']),
    len(l14['connects']),
    any(isinstance(c, dict) and c.get('phase') == 2995
        for c in l14['connects'])))

# memo tail
memo = io.open(
    r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
    r'\AGI_GPT5_MEMO.md', encoding='utf-8').read()
tail = memo.split('## Phase 2995:')[1]
out.append('memo2995 hashes(3)=%s residue=%s title=%s' % (
    all(h in tail for h in ['52073515', '2cf6da32',
                            '9e98c847', 'a18f1e0a']),
    '%(' in tail,
    tail.split('\n')[0][:60]))

# artifacts on disk
base = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913\phase2995'
        r'\omega_f1_glm4_panel')
files = ['execution.json', 'result.json',
         'omega_f1_glm4_panel.npz', 'seal.json',
         'run_log.txt']
out.append('files=%s' % all(
    os.path.isfile(os.path.join(base, f))
    for f in files))

# workspace log
ws = io.open(
    r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
    r'\.workbuddy\memory\2026-09-20.md',
    encoding='utf-8').read()
out.append('wslog2995=%s hashline=%s' % (
    'Phase 2995' in ws, 'a18f1e0a' in ws))

# memory
mm = io.open(
    r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
    r'\.workbuddy\memory\MEMORY.md',
    encoding='utf-8').read()
out.append('memory chars=%d max2995=%s next2996=%s '
           'dblperm_rule=%s' % (
    len(mm), 'max=2995' in mm, '**2996**' in mm,
    '双置换' in mm))

io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_final2995.txt', 'w',
        encoding='utf-8').write('\n'.join(out))
print('ok')
