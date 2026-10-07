# -*- coding: utf-8 -*-
import os, re, glob, time, json
ROOT = r'D:\AI2050\Ai2050-OpenOne'
out = []
def w(s): out.append(str(s))
w('now %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
w('')
w('--- 1) 被删节对应的独立文档是否还在 ---')
for rel in ['research/gpt5/docs/RDC_TESTPLAN_v1.md', 'research/gpt5/docs/EMBED_ANCHOR_VERDICT_v1.md',
            'research/gpt5/docs/MEMO_AUDIT_2750_3148.md', 'research/gpt5/docs/MAIN_AXIS_VERDICT_v1.md']:
    p = os.path.join(ROOT, rel.replace('/', os.sep))
    if os.path.exists(p):
        st = os.stat(p)
        w('  OK   %-46s %7d B  mtime=%s' % (os.path.basename(rel), st.st_size,
          time.strftime('%m-%d %H:%M', time.localtime(st.st_mtime))))
    else:
        w('  GONE %s' % rel)
w('')
w('--- 2) 谁在改写 AGI_GPT5_MEMO.md（扫描脚本源码）---')
hits = []
for base in ['tests', 'scripts', 'research', 'server', 'shared', 'gpt5_temp']:
    d = os.path.join(ROOT, base)
    if not os.path.isdir(d): continue
    for dp, dn, fn in os.walk(d):
        if 'node_modules' in dp: continue
        for f in fn:
            if not f.endswith('.py'): continue
            fp = os.path.join(dp, f)
            try:
                t = open(fp, encoding='utf-8', errors='ignore').read()
            except Exception:
                continue
            if 'AGI_GPT5_MEMO' in t:
                mode = []
                if re.search(r"open\([^)]*['\"]a['\"]", t): mode.append('append')
                if re.search(r"open\([^)]*['\"]w['\"]", t): mode.append('WRITE/TRUNC')
                if 'compress' in f or 'memcompress' in f: mode.append('compressor')
                hits.append((fp.replace(ROOT + os.sep, ''), ','.join(mode) or '?',
                             time.strftime('%m-%d %H:%M', time.localtime(os.stat(fp).st_mtime))))
w('  files referencing MEMO: %d' % len(hits))
for h in sorted(hits, key=lambda x: x[2])[-25:]:
    w('   %-62s %-18s %s' % h)
w('')
w('--- 3) 备份候选 ---')
bps = []
for pat in ['**/AGI_GPT5_MEMO*.md*', '**/*MEMO*.bak', '**/*memo*backup*']:
    bps += glob.glob(os.path.join(ROOT, pat), recursive=True)
bps = [b for b in bps if 'node_modules' not in b and 'frontend' not in b]
for b in bps[:30]:
    try:
        w('  %-72s %8d B  %s' % (b.replace(ROOT + os.sep, ''), os.path.getsize(b),
          time.strftime('%m-%d %H:%M', time.localtime(os.stat(b).st_mtime))))
    except Exception: pass
w('')
w('--- 4) docs 目录近 3 小时内改动 ---')
d = os.path.join(ROOT, 'research', 'gpt5', 'docs')
if os.path.isdir(d):
    rows = []
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isfile(fp):
            rows.append((os.stat(fp).st_mtime, f, os.path.getsize(fp)))
    rows.sort(reverse=True)
    for m, f, s in rows[:15]:
        w('  %s  %-40s %8d B' % (time.strftime('%m-%d %H:%M', time.localtime(m)), f, s))
open(os.path.join(ROOT, 'gpt5_temp', 'probe_memo_loss.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
