# -*- coding: utf-8 -*-
"""补丁 9b：修正补丁 9 在 closeout_docs wlog 条目中**格式符与实参的数/序不一致**
（原写 9 个占位符 vs 传入 10 个实参 ⇒ 到收尾链才炸）。就地改正为「占位符 = 实参、次序语义对齐」。
并加一道**静态自检**：从已打好补丁的源码里用 AST 抽出 `%` 渲染点，用审计件实参试渲染，
把「格式错」从收尾时刻提前到现在。
"""
import io
import os
import re
import ast
import json

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20'
INFRA = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_infra'
n = 0


def rep(path, pairs):
    global n
    s = io.open(path, encoding='utf-8').read()
    for old, new in pairs:
        assert old != new, 'no-op replace: %r' % old[:60]
        c = s.count(old)
        assert c == 1, '%s :: count=%d :: %r' % (path, c, old[:90])
        s = s.replace(old, new); n += 1
    io.open(path, 'w', encoding='utf-8', newline='\n').write(s)


rep(BASE + r'\closeout_docs_phase20.py', [
    ("'`[YYYY-MM-DD HH:MM]`，逐条 +11 B、合计 **+%d B**（与实盘差逐位一致，残差 %+d B），行数不变（%d）、无文本丢失；'",
     "'`[YYYY-MM-DD HH:MM]`，逐条 +11 B、合计 **+%d B**（实测 **+%d B**，残差 %+d B），行数不变（%d）、无文本丢失；'"),
])
print('PATCHED %d spots' % n)

# ---------------------------------------------------------------- 静态自检
DRIFT = json.load(io.open(os.path.join(INFRA, 'memo_drift_phase19_postbaseline.json'), encoding='utf-8'))
fails = []
rendered = {}


def try_render(label, tmpl, args):
    if tmpl is None:
        fails.append('%s: 未定位到 % 渲染点' % label)
        return
    try:
        out = tmpl % tuple(args)
    except Exception as e:
        fails.append('%s: %s: %s' % (label, type(e).__name__, e))
        return
    rendered[label] = out
    # 渲染后不应再有「看起来像占位符」的残留（%s/%d/%+d 等）
    m = re.search(r'%[-+ 0-9.]*[sdrf]', out)
    if m:
        fails.append('%s: 渲染后仍残留占位符 %r' % (label, out[max(0, m.start() - 30):m.start() + 20]))


def find_mod_binop(path, needle):
    src = io.open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            seg = ast.get_source_segment(src, node) or ''
            if needle in seg:
                return ast.literal_eval(node.left)
    return None


ARG_W = [DRIFT['prev_baseline']['bytes'], str(DRIFT['prev_baseline']['sha8']),
         str(DRIFT['prev_baseline']['frozen_at']), str(DRIFT['memo_mtime']),
         len(DRIFT['normalized_phases']), DRIFT['predicted_delta_bytes'],
         DRIFT['observed_delta_bytes'], DRIFT['residual_bytes'],
         DRIFT['memo_lines_at_audit'], '2026-10-02 07:40:59']

try_render('closeout_docs wlog',
           find_mod_binop(os.path.join(BASE, 'closeout_docs_phase20.py'), 'post-append-phase19` 基线'),
           ARG_W)
try_render('gen_present 勘误',
           find_mod_binop(os.path.join(BASE, 'gen_present_phase20.py'), 'E-baseline'),
           [str(DRIFT['prev_baseline']['bytes']), str(DRIFT['prev_baseline']['sha8']),
            str(DRIFT['memo_mtime']), str(len(DRIFT['normalized_phases'])),
            str(DRIFT['observed_delta_bytes'])])

src3 = io.open(os.path.join(BASE, 'gen_memo_phase20.py'), encoding='utf-8').read()
if 'E-baseline' not in src3:
    fails.append('gen_memo: 未含 E-baseline 条目')

print('STATIC CHECKS: %s' % ('OK' if not fails else 'FAILED'))
for f in fails:
    print('  !!', f)
for k, v in rendered.items():
    print('--- render(%s) ---' % k)
    print(v)

OUT = os.path.join(INFRA, 'patch_p20_9_fmtcheck.txt')
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(
    'PATCHED %d spots\nSTATIC CHECKS: %s\n' % (n, 'OK' if not fails else 'FAILED')
    + ''.join('  !! %s\n' % f for f in fails)
    + ''.join('--- render(%s) ---\n%s\n' % (k, v) for k, v in rendered.items()))
print('WROTE', OUT)
