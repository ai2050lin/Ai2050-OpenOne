# -*- coding: utf-8 -*-
"""p3090_memo_append.py — Phase 3090 追加（append-only）
读取草稿 -> 填时间戳 -> 追加到 MEMO 尾部 -> 复核顺序与落盘。"""
import io
from datetime import datetime

DRAFT = (r'D:\AI2050\Ai2050-OpenOne'
         r'\tests\gpt5_temp\p3090_memo_draft.md')
MEMO = (r'D:\AI2050\Ai2050-OpenOne'
        r'\research\gpt5\docs\AGI_GPT5_MEMO.md')
LOGF = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\gpt5_temp\p3090_append_log.txt')

o = []
txt = io.open(DRAFT, encoding='utf-8').read()
ts = datetime.now().strftime('%Y-%m-%d %H:%M')
assert 'TIMESTAMP_PLACEHOLDER' in txt
txt = txt.replace('TIMESTAMP_PLACEHOLDER', ts)

memo = io.open(MEMO, encoding='utf-8').read()
assert '## Phase 3090:' not in memo, 'already appended'
assert '## Phase 3089:' in memo
p3089 = memo.rindex('## Phase 3089:')
assert memo.rindex('## Phase 3088:') < p3089

if not memo.endswith('\n'):
    memo += '\n'
memo += '\n' + txt
if not memo.endswith('\n'):
    memo += '\n'
with io.open(MEMO, 'w', encoding='utf-8') as f:
    f.write(memo)

# 复核：重读磁盘，顺序与存在性
memo2 = io.open(MEMO, encoding='utf-8').read()
i3089 = memo2.rindex('## Phase 3089:')
i3090 = memo2.rindex('## Phase 3090:')
assert i3089 < i3090, 'order broken'
assert ts in memo2
o.append('APPEND_OK ts=%s' % ts)
o.append('p3089_at=%d p3090_at=%d' % (i3089, i3090))
o.append('memo_chars=%d' % len(memo2))

with io.open(LOGF, 'w', encoding='utf-8') as f:
    f.write('\n'.join(o))
print('APPEND_OK %s' % ts)
