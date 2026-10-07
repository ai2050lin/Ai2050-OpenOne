import os, hashlib, time

root = r'D:\AI2050\Ai2050-OpenOne'
memo = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
src = os.path.join(root, 'gpt5_temp', 'memo_append_e2.md')

raw = open(src, 'rb').read()
txt = raw.decode('utf-8')
add = txt.lstrip('\r\n')
add = add.replace('\r\n', '\n').replace('\n', '\r\n')
if not add.endswith('\r\n'):
    add = add + '\r\n'
blob = add.encode('utf-8')

b0 = open(memo, 'rb').read()
with open(memo, 'ab') as f:
    f.write(blob)
b1 = open(memo, 'rb').read()

T1 = b1.decode('utf-8')
L1 = T1.split('\r\n')
hdr = '## Phase 3： 分析'

out = []
out.append('time %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
out.append('before bytes %d after bytes %d delta %d' % (len(b0), len(b1), len(b1) - len(b0)))
out.append('appended bytes %d (expect %d)' % (len(blob), len(b1) - len(b0)))
out.append('before sha8 %s' % hashlib.sha256(b0).hexdigest()[:8])
out.append('after  sha8 %s' % hashlib.sha256(b1).hexdigest()[:8])
out.append('prefix_unchanged %s' % (b1[:len(b0)] == b0))
out.append('strict_utf8_ok %s' % True)
out.append('lines_after %d' % (len(L1) - 1))
out.append('hdr_count %d' % T1.count(hdr))
idx = T1.find(hdr)
out.append('hdr_char_offset %d' % idx)
out.append('hdr_line %d' % (T1[:idx].count('\r\n') + 1) if idx >= 0 else 'NA')
out.append('has_k4 %s' % ('死线 K4' in T1))
out.append('has_R11 %s' % ('0.1758' in T1))
out.append('--- all ## titles ---')
for l in L1:
    if l.startswith('## '):
        out.append('   ' + l[:110])
out.append('--- tail 4 ---')
for l in L1[-6:]:
    out.append('   T ' + l[:130])

open(os.path.join(root, 'gpt5_temp', 'verify_append_e2.txt'), 'w', encoding='utf-8').write('\r\n'.join(out))
print('ok')
