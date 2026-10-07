# -*- coding: utf-8 -*-
"""vfix2: line-based repair of verify check strings.
ASCII-only source; all Chinese via \\u escapes."""
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
     r'\p3122_disk_verify.py')
lines = io.open(P, encoding='utf-8').read() \
    .splitlines(True)
out = []
nrep = 0
for ln in lines:
    st = ln.strip()
    if st.startswith("'L35 large negative write"):
        out.append(u"          u'L35 \\u5927\\u8d1f"
                   u"\\u5199 \\u221211.4',\n")
        nrep += 1
        continue
    if st.startswith("'### 1. Three major"):
        out.append(u"          u'### 1. \\u4e09\\u5927"
                   u"\\u53d1\\u73b0\\uff08\\u91cd\\u590d"
                   u"\\u4e09\\u904d\\uff09',\n")
        nrep += 1
        continue
    if st.startswith("'### 2. Key numerical"):
        out.append(u"          u'### 2. \\u5173\\u952e"
                   u"\\u6570\\u503c',\n")
        nrep += 1
        continue
    if st.startswith("'### 3. Weaknesses"):
        out.append(u"          u'### 3. \\u786c"
                   u"\\u4f24',\n")
        nrep += 1
        continue
    if st.startswith("'### 4. Mechanism puzzle"):
        out.append(u"          u'### 4. \\u673a\\u5236"
                   u"\\u62fc\\u56fe\\u66f4\\u65b0',\n")
        nrep += 1
        continue
    if st.startswith("'### 5. 3123 prereg"):
        out.append(u"          u'### 5. 3123 "
                   u"\\u9884\\u6ce8\\u518c',\n")
        nrep += 1
        continue
    if 'p3122_patch1.py' in st \
            and 'endswith' not in st \
            and 'ck(' not in st:
        out.append(u"       u'`p3122_patch4.py`"
                   u"\\u3002'))\n")
        nrep += 1
        continue
    if st.startswith("ck('F.title'"):
        out.append("_title3122 = [l for l in "
                   "mtxt.splitlines()\n")
        out.append("              if l.startswith"
                   "('## ')\n")
        out.append("              and '3122' in l\n")
        out.append("              and u'\\u673a\\u5236"
                   "\\u94fe\\u72b6\\u6001' in l]\n")
        out.append("ck('F.title', "
                   "len(_title3122) == 1)\n")
        nrep += 1
        continue
    if st.startswith("ck('F.l3122'"):
        out.append(u"ck('F.l3122', u'\\u8bed\\u6cd5"
                   u"\\u4f7f\\u5185\\u5bb9\\u53ef"
                   u"\\u8bfb' in mtxt\n")
        out.append(u"   and u'\\u7f3a\\u6301\\u4e45"
                   u"\\u8f68\\u8ff9\\u951a\\u70b9' "
                   u"in mtxt)\n")
        nrep += 1
        # skip the continuation line
        out.append('__SKIP__')
        continue
    out.append(ln)

# drop the continuation line after F.l3122 block
res = []
skip = False
for ln in out:
    if ln == '__SKIP__':
        skip = True
        continue
    if skip:
        skip = False
        if st is None:
            pass
        continue
    res.append(ln)

src = ''.join(res)
assert nrep == 9, 'nrep %d' % nrep
assert 'Three major' not in src
assert 'Key numerical' not in src
assert 'Weaknesses' not in src
assert 'Mechanism puzzle' not in src
assert '3123 prereg' not in src
assert 'large negative write' not in src
assert '_title3122' in src
py_compile.compile(P, doraise=True)
io.open(P, 'w', encoding='utf-8').write(src)
print('vfix2 OK (%d replacements)' % nrep)
