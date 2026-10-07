# -*- coding: utf-8 -*-
import io
from transformers import AutoTokenizer

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
tok = AutoTokenizer.from_pretrained(MD)

BODIES = (
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',)
TARGETS = ('so', 'because', 'therefore', 'however',
           'while', 'yet', 'although', 'thus')
NEW_BODIES = (
    'The game was delayed, so',
    'She stayed home because',
    'The engine failed, therefore',
    'He kept smiling, although',)
NEW_TARGETS = ('so', 'because', 'therefore',
               'although')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')

out = []
word_tok = {}
for w in set(TARGETS) | set(NEW_TARGETS):
    wi = tok(' ' + w, add_special_tokens=False)[
        'input_ids']
    word_tok[w] = wi
    out.append('word %r -> %r' % (w, wi))

for tag, bodies, targets in (
        ('OLD', BODIES, TARGETS),
        ('NEW', NEW_BODIES, NEW_TARGETS)):
    for bi in range(len(bodies)):
        for ci in range(len(PREFIXES)):
            s = (PREFIXES[ci] + ' ' + bodies[bi]) \
                if PREFIXES[ci] else bodies[bi]
            ids = tok(s, add_special_tokens=False)[
                'input_ids']
            t = word_tok[targets[bi]]
            if len(t) != 1 or ids.count(t[0]) != 1:
                out.append('BAD %s b%d c%d: s=%r '
                           'toks=%r t=%r count=%d'
                           % (tag, bi, ci, s, ids, t,
                              ids.count(t[0])
                              if len(t) == 1 else -1))
out.append('probe done')

# also dump the raw file's BODIES/TARGETS block to
# verify script content matches expectation
src = io.open(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
              r'\phase3046_omega_p43_kfield_'
              r'injection_qwen.py',
              encoding='utf-8').read()
i0 = src.find('BODIES = (')
i1 = src.find('NEW_TARGETS')
out.append('--- script constants block ---')
out.append(src[i0:i1 + 200])
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\probe3046b_result.txt', 'w',
        encoding='utf-8').write('\n'.join(out))
print('ok')
