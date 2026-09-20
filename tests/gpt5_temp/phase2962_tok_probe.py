# -*- coding: utf-8 -*-
# Phase 2962 pre-freeze reachability probe: tokenizer
# single-token check ONLY (no outcome statistics observed).
import json
import os

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p2962_tok_probe.txt'
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'

CONCRETE = ['apple', 'river', 'mountain', 'chair', 'dog',
            'car', 'tree', 'house', 'book', 'rain', 'stone',
            'bread', 'horse', 'hand', 'clock']
ABSTRACT = ['freedom', 'justice', 'memory', 'truth', 'hope',
            'love', 'fear', 'thought', 'reason', 'fate',
            'peace', 'doubt', 'glory', 'idea', 'time']
FUNCTION = ['the', 'of', 'and', 'but', 'in', 'on', 'with',
            'because', 'if', 'however', 'although', 'since',
            'while', 'then', 'also']
BACKUP = {'concrete': ['ocean', 'window', 'spoon', 'lamp',
                       'needle'],
          'abstract': ['wisdom', 'virtue', 'logic', 'mercy',
                       'anger'],
          'function': ['yet', 'nor', 'or', 'than', 'unless']}

lines = []


def check(tok, words, gname, lines):
    ok, bad = [], []
    for w in words:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids2 = tok(w, add_special_tokens=False)[
                'input_ids']
        else:
            ids2 = ids
        if len(ids) == 1:
            ok.append((w, int(ids[0])))
        elif len(ids2) == 1:
            ok.append((w, int(ids2[0])))
            lines.append('  NOTE %s: single only without '
                         'leading space' % w)
        else:
            bad.append((w, [int(i) for i in ids2]))
    lines.append('%s: ok %d/15 bad=%s'
                 % (gname, len(ok), bad))
    return ok, bad


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    lines.append('tokenizer loaded')
    allw = CONCRETE + ABSTRACT + FUNCTION
    assert len(set(allw)) == 45, 'duplicate words in list'
    res = {}
    for gname, ws in (('concrete', CONCRETE),
                      ('abstract', ABSTRACT),
                      ('function', FUNCTION)):
        ok, bad = check(tok, ws, gname, lines)
        res[gname] = ok
        if bad:
            lines.append('  backups for %s:' % gname)
            for w in BACKUP[gname]:
                ids = tok(' ' + w, add_special_tokens=False)[
                    'input_ids']
                lines.append('   %s -> %s'
                             % (w, [int(i) for i in ids]))
    # cross-check against the 57-word list tids (overlap?
    # any word already used as a key token)
    import numpy as np
    z = np.load(r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
                r'\rdc_query_construction_20260913\phase2887'
                r'\language_axis_mlp\language_axis_mlp.npz',
                allow_pickle=True)
    old_words = [str(w).split(':')[2] for w in z['words']]
    old_tids = set()
    for w in old_words:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) == 1:
            old_tids.add(int(ids[0]))
    overlap = [(w, t) for g in res.values() for (w, t) in g
               if t in old_tids]
    lines.append('overlap with old 57-word tid set: %s'
                 % overlap)
    lines.append('old words sample: %s' % old_words[:12])
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('probe done')


if __name__ == '__main__':
    main()
