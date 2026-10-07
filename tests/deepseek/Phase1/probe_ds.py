import os, time, json

root = r'D:\AI2050\Ai2050-OpenOne'
out = []
out.append('now %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
out.append('now_iso %s' % time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()))

d = os.path.join(root, 'research', 'deepseek')
out.append('research/deepseek exists %s' % os.path.isdir(d))
if os.path.isdir(d):
    for dp, dn, fn in os.walk(d):
        rel = dp.replace(root, '')
        out.append('DIR %s  files=%d dirs=%d' % (rel, len(fn), len(dn)))
        for f in fn[:60]:
            out.append('    ' + f)

p = os.path.join(d, 'docs', 'AGI_DEEPSEEK_MEMO.md')
out.append('memo exists %s' % os.path.exists(p))
if os.path.exists(p):
    T = open(p, encoding='utf-8', errors='replace').read()
    out.append('memo bytes %d lines %d' % (len(T.encode('utf-8')), T.count('\n') + 1))
    for l in T.splitlines()[:40]:
        out.append('  H ' + l[:150])
    out.append('  --- tail ---')
    for l in T.splitlines()[-25:]:
        out.append('  T ' + l[:150])

rd = os.path.join(root, 'research')
out.append('--- research dirs ---')
out.append(', '.join(sorted([x for x in os.listdir(rd) if os.path.isdir(os.path.join(rd, x))])))
out.append('--- research/deepseek siblings ---')
if os.path.isdir(d):
    out.append(', '.join(sorted(os.listdir(d))))
    dd = os.path.join(d, 'docs')
    if os.path.isdir(dd):
        out.append('docs: ' + ', '.join(sorted(os.listdir(dd))))

# 主 MEMO 尾部状态（确认 3149 是否已 closeout）
memo5 = os.path.join(root, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
T5 = open(memo5, encoding='utf-8', errors='replace').read()
L5 = T5.splitlines()
out.append('--- main MEMO ---')
out.append('gpt5 memo bytes %d lines %d' % (len(T5.encode('utf-8')), len(L5)))
out.append('has Phase 3149 %s' % ('## Phase 3149' in T5))
out.append('has deepseek memo ref %s' % ('AGI_DEEPSEEK_MEMO' in T5))
for l in L5[-6:]:
    out.append('  M ' + l[:150])

# 3149 结果状态
p3149 = os.path.join(root, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3149', 'omega_p147_carrier_dlogit_poslate_kdose_v3amp')
out.append('--- phase3149 ---')
if os.path.isdir(p3149):
    for f in sorted(os.listdir(p3149)):
        fp = os.path.join(p3149, f)
        if os.path.isfile(fp):
            out.append('  %s %.1fKB mtime=%s' % (f, os.path.getsize(fp) / 1024,
                       time.strftime('%m-%d %H:%M:%S', time.localtime(os.path.getmtime(fp)))))
        else:
            out.append('  %s/ dir' % f)
else:
    out.append('  dir missing')

open(os.path.join(root, 'gpt5_temp', 'probe_ds.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
