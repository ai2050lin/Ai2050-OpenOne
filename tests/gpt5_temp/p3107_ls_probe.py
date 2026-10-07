# -*- coding: utf-8 -*-
"""Probe qwen3-4b model dir (junction) + safetensors keys."""
import json
import os

d = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
lines = []
lines.append('exists=%s islink=%s' % (os.path.exists(d),
                                      os.path.islink(d)))
try:
    entries = os.listdir(d)
    lines.append('n_entries=%d' % len(entries))
    for e in entries:
        lines.append('  %s (%d bytes)' % (
            e, os.path.getsize(os.path.join(d, e))
            if os.path.isfile(os.path.join(d, e))
            else -1))
except Exception as ex:
    lines.append('listdir fail: %r' % ex)
    # try resolving junction target
    try:
        import subprocess
        out = subprocess.run(
            ['cmd', '/c', 'dir', '/AL', os.path.dirname(d)],
            capture_output=True, text=True).stdout
        lines.append(out)
    except Exception as e2:
        lines.append(repr(e2))

# safetensors key probe
try:
    from safetensors import safe_open
    for fn in sorted(entries):
        if fn.endswith('.safetensors'):
            p = os.path.join(d, fn)
            with safe_open(p, framework='np') as f:
                keys = list(f.keys())
                hit = [k for k in keys
                       if 'lm_head' in k
                       or 'embed_tokens' in k]
                lines.append('%s: %d keys; head-ish: %s'
                             % (fn, len(keys), hit[:5]))
except Exception as ex:
    lines.append('safetensors probe fail: %r' % ex)

io_ = open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
           r'\p3107_ls_out.txt', 'w', encoding='utf-8')
io_.write('\n'.join(lines) + '\n')
io_.close()
print('done')
