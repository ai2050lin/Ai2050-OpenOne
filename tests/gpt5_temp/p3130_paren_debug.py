# -*- coding: utf-8 -*-
"""Track paren depth line by line for
phase3130_closeout.py to locate the
imbalance. Brackets inside string literals
are ignored via a simple state machine."""
import io
import tokenize

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\gpt5_temp\phase3130_closeout.py')
src = io.open(SRC, encoding='utf-8').read()
lines = src.split('\n')

# use tokenize to find OP tokens only
depth = 0
events = []
try:
    toks = list(tokenize.generate_tokens(
        io.StringIO(src).readline))
except Exception as e:
    events.append('TOKENIZE ERROR: %s' % e)
    toks = []
lastline = 0
for t in toks:
    if t.type == tokenize.OP:
        if t.string in '([{':
            depth += 1
        elif t.string in ')]}':
            depth -= 1
            if depth < 0:
                events.append(
                    'NEGATIVE at line %d: %r'
                    % (t.start[0], t.string))
                depth = 0
    lastline = max(lastline, t.end[0])
    # record depth at end of each logical line
events.append('FINAL depth after tokenize: %d'
              % depth)

# per-line depth at line end
depth = 0
linedepth = {}
try:
    for t in tokenize.generate_tokens(
            io.StringIO(src).readline):
        if t.type == tokenize.OP:
            if t.string in '([{':
                depth += 1
            elif t.string in ')]}':
                depth -= 1
        linedepth[t.end[0]] = depth
except Exception:
    pass

out = ['FINAL: %d' % depth]
for ln in sorted(linedepth):
    out.append('%4d d=%d | %s'
               % (ln, linedepth[ln],
                  lines[ln - 1][:70]))
with io.open(SRC.replace(
        'phase3130_closeout.py',
        'p3130_paren_debug.txt'), 'w',
        encoding='utf-8') as f:
    f.write('\n'.join(out))
print('DEBUG_WRITTEN')
