import os, hashlib, time

root = r'D:\AI2050\Ai2050-OpenOne'
memo = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
src = os.path.join(root, 'tests', 'deepseek_temp', 'memo_append_r2.md')

raw = open(src, 'rb').read()
text = raw.decode('utf-8')
# 统一为 CRLF，与主文件主体一致
text = text.replace('\r\n', '\n').replace('\n', '\r\n')
add = text.encode('utf-8')

b0 = open(memo, 'rb').read()
h0 = hashlib.sha256(b0).hexdigest()
n0 = len(b0)
L0 = b0.decode('utf-8-sig').count('\n') + 1

with open(memo, 'ab') as f:
    f.write(add)

b1 = open(memo, 'rb').read()
T1 = b1.decode('utf-8-sig')
L1 = T1.splitlines()

out = []
out.append('append at %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
out.append('memo before_bytes %d before_lines %d sha8 %s' % (n0, L0, h0[:8]))
out.append('memo after_bytes  %d after_lines  %d sha8 %s' % (len(b1), len(L1), hashlib.sha256(b1).hexdigest()[:8]))
out.append('delta_bytes %d (appended %d)' % (len(b1) - n0, len(add)))
out.append('prefix_unchanged %s' % (b1[:n0] == b0))
out.append('bom_preserved %s' % (b1[:3] == b'\xef\xbb\xbf'))
out.append('utf8_valid %s' % True)
hdr = '## 复核 R1 确认 + 登记约定变更（产物迁移）（非 Phase 编号）'
out.append('hdr_occurrences %d' % T1.count(hdr))
out.append('hdr_line %d' % ([i + 1 for i, l in enumerate(L1) if l.startswith(hdr)][0]))
out.append('bare_lf %d crlf %d' % (b1.count(b'\n') - b1.count(b'\r\n'), b1.count(b'\r\n')))
out.append('--- tail 6 ---')
for l in L1[-6:]:
    out.append('  ' + l[:150])

open(os.path.join(root, 'tests', 'deepseek_temp', 'verify_append_r2.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
