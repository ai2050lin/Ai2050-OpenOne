# -*- coding: utf-8 -*-
"""Phase 2970 patch5b: 2969 pairing per head/layer (correct anchors)."""
import ast
import io
import shutil
import os

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2970_delay_carrier_localization.py'
t = io.open(P, encoding='utf-8').read()

old0 = """        log('total en/L pairs: %d' % len(pairs_all), lines)"""
new0 = """        log('total en/L pairs: %d' % len(pairs_all), lines)

        def pairs_2969(pk):
            \"\"\"2969 pairing口径 verbatim: filter peak words
            first, then cidx (last L per concept among peak
            words). 2887 is 4-language: cidx[cid][lang] = i
            overwrite picks the LAST matching word.\"\"\"
            c = {}
            for i in range(n_words):
                if pk[i] is None:
                    continue
                c.setdefault(words[i][1], {})[lab_lang[i]] = i
            return [(d[0], d[1]) for d in c.values()
                    if 0 in d and 1 in d]"""
assert old0 in t, 'pairs_all anchor not found'
t = t.replace(old0, new0, 1)

old1 = """            pk = [peak_loc(s_grid, curve[:, i])
                  for i in range(n_words)]
            d = [pk[fr] - pk[en] for en, fr in pairs_all
                 if pk[en] is not None
                 and pk[fr] is not None]
            if len(d) < PAIRS_MIN_TOTAL:
                continue"""
new1 = """            pk = [peak_loc(s_grid, curve[:, i])
                  for i in range(n_words)]
            pl = pairs_2969(pk)
            d = [pk[fr] - pk[en] for en, fr in pl]
            if len(d) < PAIRS_MIN_TOTAL:
                continue"""
assert old1 in t, 'T2 pairing anchor not found'
t = t.replace(old1, new1, 1)

old2 = """                for li in valid_l:
                    pk = pk_cache[li]
                    d = np.array([pk[fr] - pk[en]
                                  for en, fr in pairs_all
                                  if pk[en] is not None
                                  and pk[fr] is not None])
                    worst = max(worst, abs(
                        float((sg[:len(d)] * d).mean())))"""
new2 = """                for li in valid_l:
                    pk = pk_cache[li]
                    pl = pairs_2969(pk)
                    d = np.array([pk[fr] - pk[en]
                                  for en, fr in pl])
                    worst = max(worst, abs(
                        float((sg[:len(d)] * d).mean())))"""
assert old2 in t, 'T2 maxT anchor not found'
t = t.replace(old2, new2, 1)

old3 = """            for k in valid_h:
                pk = pk_cache_h[k]
                d = np.array([pk[fr] - pk[en]
                              for en, fr in pairs_all
                              if pk[en] is not None
                              and pk[fr] is not None])
                d_arr[k] = d
                groups[len(d)].append(k)"""
new3 = """            for k in valid_h:
                pk = pk_cache_h[k]
                pl = pairs_2969(pk)
                d = np.array([pk[fr] - pk[en]
                              for en, fr in pl])
                d_arr[k] = d
                groups[len(d)].append(k)"""
assert old3 in t, 'T1 grouped anchor not found'
t = t.replace(old3, new3, 1)

old4 = """            d = [pk[fr] - pk[en] for en, fr in pairs_all
                 if pk[en] is not None
                 and pk[fr] is not None]
            if len(d) >= PAIRS_MIN_HEAD:"""
new4 = """            pl = pairs_2969(pk)
            d = [pk[fr] - pk[en] for en, fr in pl]
            if len(d) >= PAIRS_MIN_HEAD:"""
assert old4 in t, 'T1 validity anchor not found'
t = t.replace(old4, new4, 1)

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
ast.parse(t2)
res = ['pairs_2969 refs: %d' % t2.count('pairs_2969('),
       'old pairing refs left: %d'
       % t2.count('for en, fr in pairs_all\n')]
d = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
    r'\rdc_query_construction_20260913\phase2970'
if os.path.exists(d):
    shutil.rmtree(d)
res.append('phase2970 cleaned: %s' % (not os.path.exists(d)))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p2970_fix5.txt', 'w',
        encoding='utf-8').write('\n'.join(res) + '\n')
print('patched')
