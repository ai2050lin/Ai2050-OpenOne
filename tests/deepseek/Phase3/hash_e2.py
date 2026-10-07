import os, hashlib, time

root = r'D:\AI2050\Ai2050-OpenOne'
out = []
out.append('now %s' % time.strftime('%Y-%m-%d %H:%M:%S'))

paths = [
    r'tests\gpt5_temp\e2_context_conditioned_probe_20261001.py',
    r'tests\gpt5_temp\e2_report.txt',
    r'research\deepseek\docs\AGI_DEEPSEEK_MEMO.md',
    r'tests\gpt5_temp\e1_embed_probe_20260930.py',
    r'tests\gpt5_temp\e1_embed_probe_report.txt',
    r'research\gpt5\docs\EMBED_ANCHOR_VERDICT_v1.md',
]
for rp in paths:
    p = os.path.join(root, rp)
    if os.path.exists(p):
        b = open(p, 'rb').read()
        T = b.decode('utf-8', errors='replace')
        out.append('%s bytes=%d lines=%d sha8=%s' % (os.path.basename(p), len(b), T.count('\n') + 1, hashlib.sha256(b).hexdigest()[:8]))
    else:
        out.append('%s MISSING' % rp)

# deepseek memo 结构
p = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
T = open(p, encoding='utf-8', errors='replace').read()
L = T.splitlines()
out.append('--- deepseek memo ---')
out.append('bytes %d lines %d' % (len(T.encode('utf-8')), len(L)))
out.append('sections: ' + ' || '.join([l[:70] for l in L if l.startswith('## ')]))
out.append('tail_last %r' % L[-1][:120])
out.append('tail_2nd_last %r' % L[-2][:120])

# 主 MEMO 状态
p5 = os.path.join(root, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
T5 = open(p5, encoding='utf-8', errors='replace').read()
L5 = T5.splitlines()
out.append('--- gpt5 memo ---')
out.append('bytes %d lines %d last_line %r' % (len(T5.encode('utf-8')), len(L5), L5[-1][:100]))

open(os.path.join(root, 'gpt5_temp', 'hash_e2.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
