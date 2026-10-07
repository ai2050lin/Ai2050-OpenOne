# -*- coding: utf-8 -*-
"""p3143 patch1: fix _injv condition
typo, remove placeholder dead code,
remove no-op loop."""
import io
import py_compile

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\glm5\phase3143_omega_'
       r'p141_d19field_readout_topk_'
       r'newI.py')

txt = io.open(SRC, encoding='utf-8').read()

# fix 1: _injv condition typo
old1 = '''                if _hit if False else (
                        _m == 'allstep'
                        or _st['step'] == 0
                        or _st['step']
                        == _m):
                    o2[:, -1, :] += _dv'''
new1 = '''                if (_m == 'allstep'
                        or _st['step'] == 0
                        or _st['step']
                        == _m):
                    o2[:, -1, :] += _dv'''
n1 = txt.count(old1)
assert n1 == 1, 'old1 count %d' % n1
txt = txt.replace(old1, new1)

# fix 2: placeholder dead code in D1
old2 = '''z35_c17_med = np.array(
    [float(np.nanmedian(z35['cos_17'][r]))
     for r in range(20, 40)])
ours_c17 = {}
for r in range(20, 40):
    dh = (base_states[r] * 0.0)  # placeholder
    ours_c17[r] = None
# injection captures (conduction)'''
new2 = '''z35_c17_med = np.array(
    [float(np.nanmedian(z35['cos_17'][r]))
     for r in range(20, 40)])
# injection captures (conduction)'''
n2 = txt.count(old2)
assert n2 == 1, 'old2 count %d' % n2
txt = txt.replace(old2, new2)

# fix 3: no-op loop in make_materials
old3 = '''    for dc in DIRS:
        pass
    return {'texts': texts, 'PID_T': PID_T,
            'DOT': DOT}'''
new3 = '''    return {'texts': texts, 'PID_T': PID_T,
            'DOT': DOT}'''
n3 = txt.count(old3)
assert n3 == 1, 'old3 count %d' % n3
txt = txt.replace(old3, new3)

io.open(SRC, 'w', encoding='utf-8').write(txt)

# disk verify
txt2 = io.open(SRC, encoding='utf-8').read()
assert txt2.count(new1) == 1
assert txt2.count(new2) == 1
assert txt2.count(new3) == 1
assert '_hit if False' not in txt2
assert '# placeholder' not in txt2
py_compile.compile(SRC, doraise=True)
lines = txt2.splitlines()
print('patch1 OK, compiled, lines = %d'
      % len(lines))
