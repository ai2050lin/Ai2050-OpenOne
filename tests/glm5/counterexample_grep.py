# -*- coding: utf-8 -*-
"""F4 counterexample propagation grep.
Given a demoted/refuted claim keyword, scan the
docs tree for references and emit a checklist.
"""
import io
import os
import sys


DOCS = os.path.join(
    r'D:\AI2050\Ai2050-OpenOne',
    'research', 'gpt5', 'docs')


def grep_claim(keyword, docs=DOCS):
    hits = []
    for fn in sorted(os.listdir(docs)):
        if not fn.endswith('.md'):
            continue
        p = os.path.join(docs, fn)
        try:
            txt = io.open(p, encoding=
                'utf-8').read()
        except Exception:
            continue
        for i, ln in enumerate(
                txt.splitlines(), 1):
            if keyword in ln:
                hits.append((fn, i,
                             ln.strip()[:120]))
    return hits


if __name__ == '__main__':
    kw = sys.argv[1]
    for fn, i, ln in grep_claim(kw):
        print('%s:%d %s' % (fn, i, ln))
    print('%d hits for %r' %
          (len(grep_claim(kw)), kw))
