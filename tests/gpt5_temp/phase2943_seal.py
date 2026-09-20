# -*- coding: utf-8 -*-
"""Phase 2943 seal: forensic snapshot + SHA256-8 registry."""
import hashlib
import json
import os
import time

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2943'
       r'\gamma_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2943_seal_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def main():
    lines = []
    res = json.load(open(os.path.join(OUT, 'result.json'),
                         encoding='utf-8'))
    exec_j = json.load(open(os.path.join(OUT, 'execution.json'),
                            encoding='utf-8'))
    lines.append('phase 2943 seal @ %s'
                 % time.strftime('%Y-%m-%dT%H:%M:%S'))
    lines.append('verdict: %s' % res['final_verdict'])
    lines.append('execution created: %s' % exec_j['created'])
    lines.append('runtime_s: %s' % res['runtime_s'])
    hashes = {}
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p):
            hashes[fn] = sha8(p)
            lines.append('  %-24s %s %d bytes'
                         % (fn, hashes[fn],
                            os.path.getsize(p)))
    sp = os.path.join(OUT, 'gamma_anatomy.npz')
    lines.append('npz present: %s' % os.path.exists(sp))
    script = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
              r'\phase2943_gamma_anatomy.py')
    lines.append('script sha8: %s' % sha8(script))
    lines.append('script mtime: %s'
                 % time.strftime('%Y-%m-%dT%H:%M:%S',
                                 time.localtime(
                                     os.path.getmtime(script))))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK seal written')


if __name__ == '__main__':
    main()
