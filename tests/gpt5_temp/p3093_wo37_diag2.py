# -*- coding: utf-8 -*-
"""Diagnostic 2: replay p3093_patch_a2.py
but right before the BAD assertion, dump
every 'Wo<digits>' occurrence in the
RENDERED DOC template (proves whether the
Wo%d/Wo%d placeholders picked up the right
LB/LP values)."""
import io
import re

PATCH = (r'D:\AI2050\Ai2050-OpenOne'
         r'\tests\gpt5_temp'
         r'\p3093_patch_a2.py')
CTX = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\p3093_wo37_ctx2.txt')
src = io.open(PATCH, encoding='utf-8').read()
anchor = ("bad_left = [b for b in BAD "
          "if b in src]")
assert anchor in src
dump = (
    "import io as _io, re as _re\n"
    "_lines = []\n"
    "for _m in _re.finditer('Wo[0-9]+', DOC):\n"
    "    _lines.append('DOC %s :: %s' % (\n"
    "        _m.group(0),\n"
    "        DOC[max(0, _m.start()-60):_m.end()"
    "+30].replace(chr(10), ' | ')))\n"
    "for _m in _re.finditer('Wo[0-9]+', src):\n"
    "    _lines.append('SRC %s :: %s' % (\n"
    "        _m.group(0),\n"
    "        src[max(0, _m.start()-60):_m.end()"
    "+30].replace(chr(10), ' | ')))\n"
    "_io.open(r'__CTX__', 'w', "
    "encoding='utf-8')"
    ".write(chr(10).join(_lines))\n"
).replace('__CTX__', CTX)
src2 = src.replace(anchor, dump + anchor)
try:
    exec(compile(src2, PATCH, 'exec'),
         {'__name__': '__main__'})
    print('NO_ASSERT')
except AssertionError as e:
    print('EXPECTED_ASSERT %s' % e)
print(io.open(CTX, encoding='utf-8')
      .read()[:2000])
