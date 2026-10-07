import io

P = (r'D:\AI2050\Ai2050-OpenOne'
     r'\research\gpt5\docs'
     r'\AGI_GPT5_MEMO.md')
NEW = ('## Phase 3127: \u03a9-P125 '
       '\u5199\u5165\u94fe\u5355\u5c42\u6d88'
       '\u878d\u5426\u5b9a + A1 \u77ed\u7a0b'
       '\u5185\u5728 + \u5168\u91cf\u53cd'
       '\u4e8b\u5b9e\u4ea4\u4e92\uff08T4 '
       '\u7b2c10Phase\uff09'
       '[2026-09-24 20:53]')
t = io.open(P, encoding='utf-8').read()
ls = t.splitlines()
idx = [i for i, l in enumerate(ls)
       if l.startswith('## Phase 3127:')]
assert len(idx) == 1, idx
old = ls[idx[0]]
ls[idx[0]] = NEW
with io.open(P, 'w', encoding='utf-8') as f:
    f.write(chr(10).join(ls) + chr(10))
# verify on disk
t2 = io.open(P, encoding='utf-8').read()
ls2 = t2.splitlines()
hit = [l for l in ls2
       if l.startswith('## Phase 3127:')]
out = ['OLD_LEN=%d' % len(old),
       'NEW_LEN=%d' % len(hit[0]),
       'NEW=%s' % hit[0],
       'n_3127_heads=%d' % len(hit),
       'memo_lines_before=%d after=%d'
       % (len(ls), len(ls2))]
io.open(r'D:\AI2050\Ai2050-OpenOne'
        r'\gpt5_temp\p3127_title_fix.txt',
        'w', encoding='utf-8').write(
    chr(10).join(out))
print('TITLE_FIX_OK')
