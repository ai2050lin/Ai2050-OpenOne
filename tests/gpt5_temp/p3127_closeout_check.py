import io, json, os
out = []
# 1) ledger
led = json.load(io.open(r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
last = led['entries'][-1] if 'entries' in led else None
if last is None:
    # fallback: find list-valued key
    for k, v in led.items():
        if isinstance(v, list) and v and isinstance(v[-1], dict):
            last = v[-1]
            out.append('listkey=' + k)
            break
out.append('ledger_n=%s' % led.get('n'))
out.append('l14=%s' % led.get('l14_count', led.get('l14')))
out.append('sha8=%s' % led.get('sha8'))
if last:
    out.append('last_phase=%s' % str(last.get('phase', last.get('id', '?'))))
    arts = last.get('artifacts', {})
    out.append('last_art_keys=%s' % sorted(arts.keys())[:12])
    out.append('last_sha8=%s' % str(arts.get('result_json', ''))[:20])
# 2) MEMO tail
t = io.open(r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
import re
heads = re.findall(r'^## Phase 3127[^\n]*', t, re.M)
out.append('memo_3127_heads=%s' % heads)
out.append('memo_lines=%d' % t.count(chr(10)))
tail = t[-3000:]
out.append('memo_tail_has_verdict=%s' % ('|'.join(str(x) for x in ['lag46_notsig' in tail, 'a1_short_range_intrinsic' in tail, 'qwen_write_not' in tail, 'glm4_write_not' in tail or 'c_write' in tail, 'regen_replay_bit_exact' in tail, 'coverage_full' in tail])))
out.append('memo_tail_has_sha=%s' % ('sha8' in tail))
# 3) wlogs
for wl in (r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-09-24.md',
           r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05\.workbuddy\memory\2026-09-24.md'):
    try:
        w = io.open(wl, encoding='utf-8').read()
        n3127 = w.count('3127')
        nver = w.count('CLOSEOUT_OK') + w.count('closeout')
        out.append('wlog %s : 3127x%d closeoutx%d' % (wl[:12], n3127, nver))
    except IOError as e:
        out.append('wlog %s MISSING: %s' % (wl[:12], e))
# 4) MEMORY.md
m = io.open(r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
out.append('memory_len=%d' % len(m))
out.append('memory_has_3127=%s' % ('3127' in m))
out.append('memory_next=%s' % ('3128' in m))
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp\p3127_closeout_verify.txt', 'w', encoding='utf-8').write(chr(10).join(out))
print('CHECK_OK', len(out))
