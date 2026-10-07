# -*- coding: utf-8 -*-
"""Patch phase3123: add build_prompt def; exact
value-pooling for L35/L30Q gates."""
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3123_omega_p121_dirfit_anchor_'
     r'l35loc_syntax_trace.py')
src = io.open(P, encoding='utf-8').read()


def rep(tag, old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, ('[%s] expect %d got %d'
                    % (tag, n, c))
    src = src.replace(old, new)


# ---- 1. add build_prompt before texts loop ----
rep('P1',
    """log('== PART C: layer tracing ==')
p2r = mat5['pair2rel']
""",
    """log('== PART C: layer tracing ==')


def build_prompt(mat, s, o, lrel, qrel):
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    rng2 = _rnd.Random(zlib.crc32(
        ('%d_%d_ord5' % (s, o)).encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents[ls], PREDS[lr], ents[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents[s], PREDS[qrel], ents[o]))
    return text


p2r = mat5['pair2rel']
""")

# ---- 2. L35 exact pooling ----
rep('P2',
    """    w35 = w[35][:, :N_NEW]
    am = (ann == 9)
    l35_stats[dc] = {
        'ans_mean': float(w35[am].mean()),
        'oth_mean': float(w35[~am].mean()),
        'n_ans': int(am.sum()),
        'n_oth': int((~am).sum())}
""",
    """    w35 = w[35][:, :N_NEW]
    am = (ann == 9)
    l35_stats[dc] = {
        'ans_mean': float(w35[am].mean()),
        'oth_mean': float(w35[~am].mean()),
        'ans_sum': float(w35[am].sum()),
        'oth_sum': float(w35[~am].sum()),
        'n_ans': int(am.sum()),
        'n_oth': int((~am).sum())}
""")
rep('P3',
    """ans_all = np.mean([l35_stats[d]['ans_mean']
                   for d in ('P', 'A1')])
oth_all = np.mean([l35_stats[d]['oth_mean']
                   for d in ('P', 'A1')])
""",
    """_ans_sum = sum(l35_stats[d]['ans_sum']
               for d in ('P', 'A1'))
_oth_sum = sum(l35_stats[d]['oth_sum']
               for d in ('P', 'A1'))
_n_ans = sum(l35_stats[d]['n_ans']
             for d in ('P', 'A1'))
_n_oth = sum(l35_stats[d]['n_oth']
             for d in ('P', 'A1'))
ans_all = _ans_sum / max(_n_ans, 1)
oth_all = _oth_sum / max(_n_oth, 1)
""")

# ---- 3. L30Q exact pooling ----
rep('P4',
    """        st = l30q_stats.setdefault(key, {
            'q_vals': [], 'nq_vals': []})
        st['q_vals'].append(
            float(wq[qm].mean()))
        st['nq_vals'].append(
            float(wq[~qm].mean()))
""",
    """        st = l30q_stats.setdefault(key, {
            'q_sum': 0.0, 'q_n': 0,
            'nq_sum': 0.0, 'nq_n': 0,
            'q_mean_per_dir': [],
            'nq_mean_per_dir': []})
        st['q_sum'] += float(wq[qm].sum())
        st['q_n'] += int(qm.sum())
        st['nq_sum'] += float(wq[~qm].sum())
        st['nq_n'] += int((~qm).sum())
        st['q_mean_per_dir'].append(
            float(wq[qm].mean()))
        st['nq_mean_per_dir'].append(
            float(wq[~qm].mean()))
""")
rep('P5',
    """q_all = np.mean(l30q_stats['L30']['q_vals']
                + l30q_stats['L32']['q_vals'])
nq_all = np.mean(l30q_stats['L30']['nq_vals']
                 + l30q_stats['L32']['nq_vals'])
""",
    """_q_sum = sum(l30q_stats[k]['q_sum']
             for k in ('L30', 'L32'))
_q_n = sum(l30q_stats[k]['q_n']
           for k in ('L30', 'L32'))
_nq_sum = sum(l30q_stats[k]['nq_sum']
              for k in ('L30', 'L32'))
_nq_n = sum(l30q_stats[k]['nq_n']
            for k in ('L30', 'L32'))
q_all = _q_sum / max(_q_n, 1)
nq_all = _nq_sum / max(_nq_n, 1)
""")

py_compile.compile(P, doraise=True)
io.open(P, 'w', encoding='utf-8').write(src)
print('patch3123-1 OK; COMPILE_OK')
