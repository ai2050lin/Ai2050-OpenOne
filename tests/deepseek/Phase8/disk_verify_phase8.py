# -*- coding: utf-8 -*-
"""Phase 8 独立磁盘复核（新进程，不引用运行期内存）。"""
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
o = []
def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def tree(d):
    out = []
    for rt, ds, fs in os.walk(d):
        for f in sorted(fs):
            p = os.path.join(rt, f)
            out.append((os.path.relpath(p, ROOT).replace('\\', '/'), os.path.getsize(p), sha8(p)))
    return sorted(out)

o.append('[1] Phase 8 目录')
for d in ['tests/deepseek/Phase8', 'tests/deepseek_temp/Phase8']:
    p = os.path.join(ROOT, d)
    o.append('  %s (%d files)' % (d, sum(len(f) for _, _, f in os.walk(p))))
    for rel, sz, h in tree(p):
        o.append('    %-62s %8d  %s' % (rel, sz, h))

o.append('')
o.append('[2] 备忘录 Phase 8 节')
mp = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
b = open(mp, 'rb').read(); T = b.decode('utf-8-sig'); L = T.splitlines()
o.append('  bytes %d lines %d sha8 %s bom %s crlf %d bare_lf %d' %
         (len(b), len(L), sha8(mp), b[:3] == b'\xef\xbb\xbf', b.count(b'\r\n'),
          b.count(b'\n') - b.count(b'\r\n')))
o.append('  Phase8 hdr lines %s' % [i + 1 for i, l in enumerate(L) if l.startswith('## Phase 8')])
o.append('  Phase1-7 hdr present %s' % all(any(l.startswith('## Phase %d' % k) for l in L) for k in range(1, 8)))
bl = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json'), encoding='utf-8'))
o.append('  baseline match: bytes %s sha256 %s' % (bl['bytes'] == len(b), bl['sha256'] == hashlib.sha256(b).hexdigest()))

o.append('')
o.append('[3] Ledger')
lp = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
Lg = json.load(io.open(lp, encoding='utf-8'))
o.append('  measurements %d ; last phase %s ; verdict %s' %
         (len(Lg['measurements']), Lg['measurements'][-1]['phase'], Lg['measurements'][-1]['verdict']))
o.append('  migration_history %d ; last note %s' %
         (len(Lg['migration_history']), Lg['migration_history'][-1]['note'][:110]))
bk = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'atlas_ledger_backup_pre_phase8.json')
o.append('  backup sha8 %s (== pre-append %s)' % (sha8(bk), Lg['migration_history'][-1]['backup_sha256_8']))

o.append('')
o.append('[4] 判决/结果一致性')
R = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'result_phase8.json'), encoding='utf-8'))
J = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'judgement_phase8.json'), encoding='utf-8'))
o.append('  result.verdict=%s ; judgement.verdict=%s ; equal %s' % (R['verdict'], J['verdict'], R['verdict'] == J['verdict']))
o.append('  gate G1=%s G2=%s ; max_head share_v=%.4f ; mlp share_v=%.4f ; I_nl=%.3f ; jump L%s(%s)' %
         (R['gates']['G1_distributed'], R['gates']['G2_localized'], R['amend1']['max_head_share_v'],
          R['amend1']['share_v']['mlp'], R['amend1']['I_nl'], R['jump_layer'], R['jump_flag']))
o.append('  confirmation same_band=%s' % R['confirmation']['same_band'])

o.append('')
o.append('[5] wlog / MEMORY')
wp = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
T2 = io.open(wp, encoding='utf-8').read()
o.append('  wlog bytes %d ; has Phase8 section %s ; has 0.4717 %s' %
         (os.path.getsize(wp), '## Phase 8 / N2h1-alpha' in T2, '0.4717' in T2))
mp2 = io.open(os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md'), encoding='utf-8').read()
for k in ['N2h1-α（Phase 8', 'I_nl', '向量预算', 'memo_baseline.json', '9 臂 + 26 条坑', 'Phase 9 阈值增益']:
    o.append('  MEMORY has %-22s %s' % (k, k in mp2))
sk = io.open(r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md', encoding='utf-8').read()
for k in ['N2h1-α 写入端组件预算', '向量预算', '26 条已实测的坑', '9 个可复用臂', '门裕度必须写进报告']:
    o.append('  SKILL has %-22s %s' % (k, k in sk))

io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'disk_verify_phase8.txt'), 'w',
        encoding='utf-8').write('\n'.join(o))
print('\n'.join(o))
