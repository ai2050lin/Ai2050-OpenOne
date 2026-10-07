# -*- coding: utf-8 -*-
import os, hashlib, io

root = r'D:\AI2050\Ai2050-OpenOne'
memo = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
src = os.path.join(root, 'gpt5_temp', 'memo_append_n2h1.md')
chk = os.path.join(root, 'gpt5_temp', 'verify_append_n2h1.txt')

b0 = open(memo, 'rb').read()
sha0 = hashlib.sha256(b0).hexdigest()[:8]
add_txt = open(src, encoding='utf-8').read()
# 统一换行为 CRLF（与既有文件一致）
if not add_txt.endswith('\n'):
    add_txt += '\n'
# 目标文件为 BOM+CRLF：追加内容不含 BOM，行尾用 CRLF
with open(memo, 'ab') as f:
    f.write(add_txt.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8'))

b1 = open(memo, 'rb').read()
sha1 = hashlib.sha256(b1).hexdigest()[:8]

out = []
out.append('memo before_bytes %d sha8 %s' % (len(b0), sha0))
out.append('memo after_bytes  %d sha8 %s' % (len(b1), sha1))
out.append('delta %d ; appended_chars %d' % (len(b1) - len(b0), len(add_txt.encode('utf-8'))))
out.append('prefix_unchanged %s' % (b1[:len(b0)] == b0))
out.append('starts_with_bom %s' % (b1[:3] == b'\xef\xbb\xbf'))
try:
    T = b1.decode('utf-8')
    out.append('utf8_decode OK')
except Exception as e:
    out.append('utf8_decode FAIL %r' % e)
    T = b1.decode('utf-8', errors='replace')
out.append('crlf_count %d ; bare_lf %d' % (T.count('\r\n'), T.count('\n') - T.count('\r\n')))
hdr = '## Phase 6'
out.append('new_hdr_occurrences %d' % T.count(hdr))
lines = T.splitlines()
idx = [i + 1 for i, l in enumerate(lines) if l.startswith(hdr)]
out.append('new_hdr_lines %s' % idx)
out.append('all_phase_hdrs: ' + ' | '.join(l[:34] for l in lines if l.startswith('## Phase')))
out.append('last_line %r' % lines[-1][:110])

# 产物哈希
for rel in ['gpt5_temp/memo_append_n2h1.md',
            'tests/gpt5_temp/N2h1_design_seal.json',
            'tests/gpt5_temp/n2h1_permutation_ablation.py',
            'tests/gpt5_temp/n2h1b_subspace_align.py',
            'tests/gpt5_temp/n2h1c_cross_material.py',
            'tests/gpt5_temp/n2h1_report_qwen3-4b.txt',
            'tests/gpt5_temp/n2h1b_report_qwen2.5-3b-instruct.txt',
            'tests/gpt5_temp/n2h1b_report_glm4-9b-chat-hf.txt',
            'tests/gpt5_temp/n2h1c_report_qwen3-4b.txt',
            'tests/gpt5_temp/n2h1c_report_glm4-9b-chat-hf.txt']:
    p = os.path.join(root, rel.replace('/', os.sep))
    if os.path.exists(p):
        b = open(p, 'rb').read()
        out.append('ART %-52s %8d B  %s' % (os.path.basename(p), len(b), hashlib.sha256(b).hexdigest()[:8]))
    else:
        out.append('ART %-52s MISSING' % os.path.basename(p))

open(chk, 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
