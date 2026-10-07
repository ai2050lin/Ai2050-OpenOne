# -*- coding: utf-8 -*-
"""Fix English placeholder strings in verify script
-> proper Chinese check strings."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
     r'\p3122_disk_verify.py')
src = io.open(P, encoding='utf-8').read()

o1 = ("for s in ('grammar makes content readable'\n"
      "          .replace('grammar makes content '\n"
      "                   'readable',\n"
      "                   'grammar makes content '\n"
      "                   'readable'),")
n1 = u"for s in (u'\u8bed\u6cd5\u4f7f\u5185\u5bb9\u53ef\u8bfb',"
c1 = src.count(o1)
assert c1 == 1, 'o1 %d' % c1
src = src.replace(o1, n1)

o2 = "          'persistent trajectory anchor',"
n2 = u"          u'\u7f3a\u5931\u6301\u4e45\u8f68\u8ff9\u951a\u70b9',"
c2 = src.count(o2)
assert c2 == 1, 'o2 %d' % c2
src = src.replace(o2, n2)

o3 = ("ck('F.l3122', 'grammar makes content "
      "readable' in mtxt\n"
      "   and 'persistent trajectory anchor' "
      "in mtxt)")
n3 = (u"ck('F.l3122', "
      u"u'\u8bed\u6cd5\u4f7f\u5185\u5bb9\u53ef\u8bfb' in mtxt\n"
      u"   and u'\u7f3a\u5931\u6301\u4e45\u8f68\u8ff9\u951a\u70b9' "
      u"in mtxt)")
c3 = src.count(o3)
assert c3 == 1, 'o3 %d' % c3
src = src.replace(o3, n3)

assert 'grammar makes content' not in src
assert 'persistent trajectory anchor' not in src
io.open(P, 'w', encoding='utf-8').write(src)
import py_compile
py_compile.compile(P, doraise=True)
print('verify-fix OK')
