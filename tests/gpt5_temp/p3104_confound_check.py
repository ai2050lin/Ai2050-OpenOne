# -*- coding: utf-8 -*-
"""3104 confound quantification: d_rel frequency cue.

cue = 1 if any distractor line's relation == the record's crit_rel.
Under the design, TRUE conditions have cue=1 ALWAYS (d_rel is under
the pair's true relation r); FALSE conditions have cue=1 only when a
neutral distractor collides with ri (7/8 chance each, 2 draws).
If AUC(cue -> truth) approximates the probe AUC (~0.75-0.80), the
T2 probe signal is explained by this surface cue, not relational
truth encoding.
"""
import io
import json
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3104\omega_p102_relation_vs_endpoint')
mat = json.load(io.open(OUTD + r'\material.json',
                        encoding='utf-8'))
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
distr = mat['distractors']
false_rels = mat['false_rels']
pair2rel = mat['pair2rel']

# rebuild record-level table (main records only, same order as
# capture: train/val/test/testE x pairs x 3 conds)
recs = []
for split in ('train', 'val', 'test', 'testE'):
    for p in mat['splits'][split]:
        key = '%d_%d' % tuple(p)
        r = pair2rel[key]
        (ri1, ri2) = false_rels[key]
        for cond, crel, lab in (('true', r, 1),
                                ('false1', ri1, 0),
                                ('false2', ri2, 0)):
            D = distr[key]
            cue = int(any(d[1] == crel for d in D))
            n_match = sum(1 for d in D if d[1] == crel)
            recs.append({'split': split, 'cond': cond,
                         'truth': lab, 'cue': cue,
                         'n_match': n_match,
                         'pair': key})
truth = np.array([x['truth'] for x in recs])
cue = np.array([x['cue'] for x in recs])


def auc(y, s):
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    return float((ranks[y == 1].sum()
                  - n1 * (n1 + 1) / 2.0) / (n1 * n0))


te_idx = np.array([i for i, x in enumerate(recs)
                   if x['split'] == 'test'])
teE_idx = np.array([i for i, x in enumerate(recs)
                    if x['split'] == 'testE'])
out = {
    'cue_auc_test': auc(truth[te_idx], cue[te_idx]),
    'cue_auc_testE': auc(truth[teE_idx], cue[teE_idx]),
    'cue_auc_all': auc(truth, cue),
    'cue_rate_true': float(cue[truth == 1].mean()),
    'cue_rate_false': float(cue[truth == 0].mean()),
    'probe_auc_test_reference':
        res['gates']['t2_best_auc_test'],
    'probe_auc_testE_reference':
        res['gates']['t2_best_auc_testE'],
    'm_auc_test_reference': res['gates']['m_auc_test'],
}
# perfect-cue upper bound: cue=1 -> predict true
acc_cue = float(((cue == 1) == (truth == 1)).mean())
out['cue_acc_all'] = acc_cue
with io.open(OUTD + r'\confound_check.json', 'w',
             encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
