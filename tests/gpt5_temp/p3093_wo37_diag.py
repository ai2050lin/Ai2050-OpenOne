# -*- coding: utf-8 -*-
"""Diagnostic: replay p3093_patch_a2.py but
dump the context around the remaining 'Wo37'
right before the BAD assertion.  The original
patch writes DST only AFTER the BAD assert, so
catching the AssertionError here never writes
any artifact."""
import io

PATCH = (r'D:\AI2050\Ai2050-OpenOne'
         r'\tests\gpt5_temp'
         r'\p3093_patch_a2.py')
CTX = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\p3093_wo37_ctx.txt')
src = io.open(PATCH, encoding='utf-8').read()
anchor = ("bad_left = [b for b in BAD "
          "if b in src]")
assert anchor in src
dump = (
    "import io as _io\n"
    "_i = src.find('Wo37')\n"
    "_io.open(r'%s', 'w', encoding='utf-8')"
    ".write('POS=%%d\\n' %% _i + "
    "(src[max(0, _i-260):_i+260] if _i >= 0 "
    "else 'NOT_FOUND'))\n" % CTX)
src2 = src.replace(anchor, dump + anchor)
try:
    exec(compile(src2, PATCH, 'exec'),
         {'__name__': '__main__'})
    print('NO_ASSERT (patch completed?!)')
except AssertionError as e:
    print('EXPECTED_ASSERT %s' % e)
ctx = io.open(CTX, encoding='utf-8').read()
print(ctx[:600])
