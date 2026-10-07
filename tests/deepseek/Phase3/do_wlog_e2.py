import os, hashlib, time

root = r'D:\AI2050\Ai2050-OpenOne'
dst = os.path.join(root, '.workbuddy', 'memory', '2026-10-01.md')
src = os.path.join(root, 'gpt5_temp', 'wlog_append_e2.md')

add = open(src, encoding='utf-8').read()
b0 = open(dst, 'rb').read()
with open(dst, 'a', encoding='utf-8', newline='') as f:
    f.write(add)
b1 = open(dst, 'rb').read()
T = b1.decode('utf-8')

mem = os.path.join(root, '.workbuddy', 'memory', 'MEMORY.md')
M = open(mem, encoding='utf-8').read()

out = []
out.append('time %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
out.append('wlog before %d after %d delta %d' % (len(b0), len(b1), len(b1) - len(b0)))
out.append('wlog prefix_unchanged %s' % (b1[:len(b0)] == b0))
out.append('wlog has_section %s' % ('## E2 探针：条件化概念比较' in T))
out.append('wlog ends_ok %s' % T.rstrip().endswith('本轮 GPU 独占无冲突。'))
out.append('mem has_deepseek_line %s' % ('AGI_DEEPSEEK_MEMO.md' in M))
out.append('mem has_e2 %s' % ('E2' in M))
out.append('mem chars %d' % len(M))
open(os.path.join(root, 'gpt5_temp', 'verify_wlog_e2.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
