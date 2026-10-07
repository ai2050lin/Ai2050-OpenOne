# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3048_omega_p45_kvpos_full_replay_qwen.py')
s = io.open(P, encoding='utf-8').read()

# 1) drop find_sub (dead after realignment fix)
old1 = """def find_sub(hay, needle):
    for s in range(len(hay) - len(needle) + 1):
        if list(hay[s:s + len(needle)]) \\
                == list(needle):
            return s
    return -1


assembled = []"""
new1 = """assembled = []"""
assert s.count(old1) == 1, ('find_sub',
                            s.count(old1))
s = s.replace(old1, new1)

# 2) realign offsets: tail alignment + first-token
#    semantic check (BPE leading-space on the
#    first body token inside the prefix prompt)
old2 = """for i in range(n_pr):
    ci = assembled[i]['cond']
    if ci == 0:
        assembled[i]['off'] = 0
    else:
        bid = assembled[idx_of[(0, assembled[i]
                               ['body'],
                               assembled[i]['new'])]][
            'ids']
        o = find_sub(assembled[i]['ids'], bid)
        assert o > 0 and o + len(bid) \\
            == len(assembled[i]['ids']), i
        assembled[i]['off'] = o"""
new2 = """for i in range(n_pr):
    ci = assembled[i]['cond']
    if ci == 0:
        assembled[i]['off'] = 0
    else:
        bid = assembled[idx_of[(0, assembled[i]
                               ['body'],
                               assembled[i]['new'])]][
            'ids']
        pid = assembled[i]['ids']
        off = len(pid) - len(bid)
        assert off > 0, i
        # tail alignment: body tokens 1..end match
        assert list(pid[off + 1:]) \\
            == list(bid[1:]), i
        # first body token: leading-space variant of
        # the same word (BPE boundary effect)
        w0b = tok.decode([bid[0]]).strip()
        w0p = tok.decode([pid[off]]).strip()
        assert w0b == w0p, (i, w0b, w0p)
        assembled[i]['off'] = off"""
assert s.count(old2) == 1, ('offset', s.count(old2))
s = s.replace(old2, new2)

# 3) register correction
old3 = """    'corrections': 'run1: none; prereg frozen '
                   'before any observation',"""
new3 = """    'corrections': 'run1 crashed pre-anchor on '
                   'the prompt-assembly alignment '
                   'assertion: BPE leading-space '
                   'boundary effect (first body '
                   'token is the no-space variant in '
                   'the base prompt but the '
                   'leading-space variant inside the '
                   'prefix prompt, so naive '
                   'subsequence matching fails); '
                   'alignment redefined as tail '
                   'matching (pref ids end with base '
                   'ids[1:]) plus a first-token '
                   'semantic strip-equality check; '
                   'offset = length difference; '
                   'statistics unchanged; run2 '
                   'authoritative',"""
assert s.count(old3) == 1, ('corr', s.count(old3))
s = s.replace(old3, new3)

# 4) run label
old4 = "'run': 'run1 authoritative (fp32)',"
new4 = ("'run': 'run2 authoritative (fp32; run1 "
        "crashed pre-anchor on a tokenization "
        "alignment assertion, see corrections)',")
assert s.count(old4) == 1, ('run', s.count(old4))
s = s.replace(old4, new4)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3048b_result.txt', 'w',
        encoding='utf-8').write(
    'patched ok; compile ok\n')
print('ok')
