# -*- coding: utf-8 -*-
"""p3120_patch6: fix patch5 wrong codepoint for
'tu' in nei-rong-rao-dong.  Use correct U+6260 U+52A8
and a longer unique tail anchor."""
import io

MEMO = (r'D:\AI2050\Ai2050-OpenOne'
        r'\research\gpt5\docs\AGI_GPT5_MEMO.md')
LOG = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp\p3120_patch6_log.txt')
o = []
src = io.open(MEMO, encoding='utf-8').read()
old = (u'\u6270\u52a8** 2026-09-23 16:57\n')
new = (u'\u6270\u52a8** [2026-09-23 16:57]\n')
c = src.count(old)
o.append('old count=%d' % c)
if c == 1:
    src = src.replace(old, new)
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(src)
    back = io.open(MEMO, encoding='utf-8').read()
    o.append('disk has bracketed=%s'
             % (new in back))
elif new in src:
    o.append('already applied')
else:
    # diagnose: show actual tail of the 3120 title
    i = src.rindex('## Phase 3120:')
    line = src[i:].split('\n')[0]
    o.append('tail repr=%r' % line[-40:])
io.open(LOG, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
