# -*- coding: utf-8 -*-
# Phase 2963 pre-freeze probe: tids + fresh-list check ONLY
# (design variables; no outcome statistics observed).
import io

import numpy as np

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p2963_cand.txt'
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'

FUNCTION = ['from', 'that', 'this', 'it', 'at', 'by', 'for',
            'as', 'or', 'so', 'when', 'which', 'about',
            'into', 'over', 'than', 'under', 'after',
            'before', 'because']
COMMON = ['people', 'water', 'money', 'world', 'school',
          'food', 'music', 'light', 'power', 'body',
          'friend', 'family', 'animal', 'city', 'game',
          'word', 'story', 'fire', 'ground', 'color']
RARE = ['tiger', 'guitar', 'pencil', 'bridge', 'garden',
        'pocket', 'ladder', 'candle', 'anchor', 'basket',
        'feather', 'hammer', 'kettle', 'mirror', 'ribbon',
        'saddle', 'tunnel', 'pepper', 'drawer', 'beetle']

lines = []


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    # prior words (2962 lists + old 2887 57-word list)
    z = np.load(r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
                r'\rdc_query_construction_20260913\phase2887'
                r'\language_axis_mlp\language_axis_mlp.npz',
                allow_pickle=True)
    prior = set(str(w).split(':')[2] for w in z['words'])
    prior |= {'apple', 'river', 'mountain', 'chair', 'dog',
              'car', 'tree', 'house', 'book', 'rain',
              'stone', 'bread', 'horse', 'hand', 'clock',
              'freedom', 'justice', 'memory', 'truth',
              'hope', 'love', 'fear', 'thought', 'reason',
              'fate', 'peace', 'doubt', 'glory', 'idea',
              'time', 'the', 'of', 'and', 'but', 'in',
              'on', 'with', 'because', 'if', 'however',
              'although', 'since', 'while', 'then', 'also'}
    for gname, ws in (('FUNCTION', FUNCTION),
                      ('COMMON', COMMON),
                      ('RARE', RARE)):
        rows = []
        for w in ws:
            ids = tok(' ' + w, add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                lines.append('%s: %s MULTI %s'
                             % (gname, w, [int(i) for i in ids]))
                continue
            rows.append((int(ids[0]), w, w in prior))
        rows.sort()
        lines.append('%s single-token %d/%d' %
                     (gname, len(rows), len(ws)))
        for t, w, p in rows:
            lines.append('  %6d %-10s prior=%s' % (t, w, p))
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('probe done')


if __name__ == '__main__':
    main()
