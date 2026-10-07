# -*- coding: utf-8 -*-
import os, hashlib, io
ROOT = r'D:\AI2050\Ai2050-OpenOne'
memo = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
src = os.path.join(ROOT, 'gpt5_temp', 'memo_append_n1.md')
out = []
add = open(src, encoding='utf-8').read()
b0 = open(memo, 'rb').read()
n0 = b0.decode('utf-8').count('\n') + 1
with open(memo, 'a', encoding='utf-8', newline='') as f:
    f.write(add)
b1 = open(memo, 'rb').read()
T = b1.decode('utf-8'); L = T.splitlines()
out.append('memo before bytes=%d lines=%d' % (len(b0), n0))
out.append('memo after  bytes=%d lines=%d' % (len(b1), len(L)))
out.append('delta_bytes=%d appended_utf8=%d' % (len(b1) - len(b0), len(add.encode('utf-8'))))
out.append('prefix_identical=%s' % (b1[:len(b0)] == b0))
hdr = '## 探索性探针 N1: 主轴三段分工'
out.append('hdr_count=%d' % T.count(hdr))
idx = [i + 1 for i, l in enumerate(L) if l.startswith(hdr)]
out.append('hdr_lines=%s' % idx)
out.append('utf8_ok=%s' % bool(T.encode('utf-8')))
out.append('tail6:')
for l in L[-6:]:
    out.append('  ' + l[:150])
out.append('')
out.append('--- artifact hashes ---')
for p in ['research/gpt5/docs/MAIN_AXIS_VERDICT_v1.md',
          'tests/gpt5_temp/N1_design_seal.json',
          'tests/gpt5_temp/e3_embed_feature_audit.py', 'tests/gpt5_temp/e3b_embed_followup.py',
          'tests/gpt5_temp/n1_v2_main_axis_scan.py', 'tests/gpt5_temp/n1b_ontology_readout.py',
          'tests/gpt5_temp/n1c_ontology_cloze.py',
          'tests/gpt5_temp/e3_report.txt', 'tests/gpt5_temp/e3b_report.txt']:
    fp = os.path.join(ROOT, p.replace('/', os.sep))
    if os.path.exists(fp):
        d = open(fp, 'rb').read()
        out.append('  %-45s bytes=%8d sha8=%s' % (os.path.basename(p), len(d), hashlib.sha256(d).hexdigest()[:8]))
    else:
        out.append('  %-45s MISSING' % p)
for pat in ['n1v2_report_', 'n1b_report_', 'n1c_report_']:
    d = os.path.join(ROOT, 'tests', 'gpt5_temp')
    for f in sorted(os.listdir(d)):
        if f.startswith(pat) and f.endswith('.txt'):
            dd = open(os.path.join(d, f), 'rb').read()
            out.append('  %-45s bytes=%8d sha8=%s' % (f, len(dd), hashlib.sha256(dd).hexdigest()[:8]))
open(os.path.join(ROOT, 'gpt5_temp', 'verify_append_n1.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('done')
