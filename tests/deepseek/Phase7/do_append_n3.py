# -*- coding: utf-8 -*-
import os, hashlib, io
root = r'D:\AI2050\Ai2050-OpenOne'
memo = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
src = os.path.join(root, 'gpt5_temp', 'memo_append_n3.md')
add = open(src, encoding='utf-8').read()
add = add.replace('\r\n', '\n').replace('\n', '\r\n')
if not add.startswith('\r\n'):
    add = '\r\n' + add

b0 = open(memo, 'rb').read()
b0p = b0
with open(memo, 'ab') as f:
    f.write(add.encode('utf-8'))
b1 = open(memo, 'rb').read()
T = b1.decode('utf-8-sig')
L = T.splitlines()
out = []
out.append('before_bytes %d  after_bytes %d  delta %d' % (len(b0), len(b1), len(b1) - len(b0)))
out.append('appended_chars %d' % len(add.encode('utf-8')))
out.append('prefix_byte_identical %s' % (b1[:len(b0p)] == b0p))
out.append('bom_preserved %s' % b1.startswith(b'\xef\xbb\xbf'))
out.append('crlf %d  lf_only %d' % (b1.count(b'\r\n'), b1.count(b'\n') - b1.count(b'\r\n')))
out.append('utf8_valid True')
out.append('lines %d' % len(L))
for h in ['## Phase 3', '## Phase 4', '## Phase 5', '## Phase 6', '## Phase 7']:
    out.append('  hdr %-12s -> %d' % (h, T.count(h)))
out.append('last_line %r' % L[-1][:140])
out.append('')
out.append('--- artifacts sha256 ---')
for p in [os.path.join(root, 'tests', 'gpt5_temp', 'n3_subspace_generality.py'),
          os.path.join(root, 'tests', 'gpt5_temp', 'N3_design_seal.json'),
          os.path.join(root, 'tests', 'gpt5_temp', 'n3_report_qwen3-4b.txt'),
          os.path.join(root, 'tests', 'gpt5_temp', 'n3_report_qwen2.5-3b-instruct.txt'),
          os.path.join(root, 'tests', 'gpt5_temp', 'n3_report_glm4-9b-chat-hf.txt')]:
    bb = open(p, 'rb').read()
    out.append('  %-42s %7d B  sha8=%s' % (os.path.basename(p), len(bb), hashlib.sha256(bb).hexdigest()[:8]))
bb = open(memo, 'rb').read()
out.append('  %-42s %7d B  sha8=%s' % ('AGI_DEEPSEEK_MEMO.md', len(bb), hashlib.sha256(bb).hexdigest()[:8]))
open(os.path.join(root, 'gpt5_temp', 'verify_append_n3.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('done')
