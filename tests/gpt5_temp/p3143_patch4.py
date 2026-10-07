# -*- coding: utf-8 -*-
"""p3143 patch4: restore BOS generation
prefix (root cause of xphase 0/128 and
pc1 anchor failure) + selective ckpt
keep.

3142 _pad_batch: chunk = PREFIX_IDS + p,
where PREFIX_IDS = tok(text)[:len-len(
PID_T[0])] = [BOS] (tok called WITHOUT
add_special_tokens=False -> BOS prepended).
3143 dropped this -> generation baseline
diverged (xphase 0/128) -> every
generation chg incomparable with
3141/3142 anchors (pc1 0.4062/0.5859 vs
0.203125). Teacher-forced captures do
NOT go through _pad_batch: dvec19
(sha 6a0332a6, drift 0), field
(cos 1.0000), cond19/cond17 are
BOS-independent and VALID -> kept.
Gen-stage ckpts (bases, pc1 trials)
dropped and recomputed."""
import io
import py_compile

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\glm5\phase3143_omega_'
       r'p141_d19field_readout_topk_'
       r'newI.py')

txt = io.open(SRC, encoding='utf-8').read()

# fix 1: PREFIX_IDS definition
old1 = '''matg = make_materials(tok_g, gens_g)
pids_P_all = matg['PID_T']['P']
pids_A1_all = matg['PID_T']['A1']
log('materials ready (672x2)')'''
new1 = '''matg = make_materials(tok_g, gens_g)
pids_P_all = matg['PID_T']['P']
pids_A1_all = matg['PID_T']['A1']
_genenc = tok_g(
    matg['texts']['P'][pks[0]])['input_ids']
PREFIX_IDS = [int(t) for t in
              _genenc[:len(_genenc)
                      - len(matg['PID_T']
                            ['P'][0])]]
log('gen-prefix ready (%d ids; 3142 '
    'BOS semantics)' % len(PREFIX_IDS))
log('materials ready (672x2)')'''
n1 = txt.count(old1)
assert n1 == 1, 'old1 count %d' % n1
txt = txt.replace(old1, new1)

# fix 2: _pad_batch prefix restore
old2 = '''def _pad_batch(rows):
    # generation rows: prompt ids only
    chunk = [list(p) for p in rows]'''
new2 = '''def _pad_batch(rows):
    chunk = [list(PREFIX_IDS) + list(p)
             for p in rows]'''
n2 = txt.count(old2)
assert n2 == 1, 'old2 count %d' % n2
txt = txt.replace(old2, new2)

# fix 3: selective ckpt keep on meta
# mismatch (BOS-independent captures)
old3 = '''CKM = {'smoke': SMOKE, 'ncap': NCAP,
       'n19': N19_ROWS}
if CK['meta'] and CK['meta'] != CKM:
    log('CKPT meta mismatch -> discard')
    CK = {'done': [], 'data': {},
          'meta': {}}
CK['meta'] = CKM'''
new3 = '''CKM = {'smoke': SMOKE, 'ncap': NCAP,
       'n19': N19_ROWS,
       'gen_prefix': True}
_KEEP_BOS_FREE = ('dvec19', 'field',
                  'cond19', 'cond17')
if CK['meta'] and CK['meta'] != CKM:
    _nd = {k: v for k, v in
           CK['data'].items()
           if k in _KEEP_BOS_FREE}
    _nold = len(CK['data'])
    CK = {'done': sorted(_nd.keys()),
          'data': _nd, 'meta': {}}
    log('CKPT meta mismatch -> kept %d '
        'BOS-independent capture stages, '
        'dropped %d gen stages'
        % (len(_nd), _nold - len(_nd)))
CK['meta'] = CKM'''
n3 = txt.count(old3)
assert n3 == 1, 'old3 count %d' % n3
txt = txt.replace(old3, new3)

io.open(SRC, 'w', encoding='utf-8').write(txt)
txt2 = io.open(SRC, encoding='utf-8').read()
assert txt2.count(new1) == 1
assert txt2.count(new2) == 1
assert txt2.count(new3) == 1
assert 'gen_prefix' in txt2
assert 'prompt ids only' not in txt2
py_compile.compile(SRC, doraise=True)
print('patch4 OK, compiled')
