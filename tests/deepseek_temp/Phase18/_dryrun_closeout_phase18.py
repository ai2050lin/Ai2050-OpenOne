# -*- coding: utf-8 -*-
"""Phase 18 收尾脚本预演：用 SMOKE result 作为输入，验证 gen_memo / gen_present 的字段访问。"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P = os.path.join(ROOT, 'tests', 'deepseek', 'Phase18')

for fn, out_repl in (
        ('gen_memo_phase18.py', ("memo_append_phase18.md", "_dryrun_memo_phase18.md")),
        ('gen_present_phase18.py', ("present_phase18.html", "_dryrun_present_phase18.html"))):
    src = io.open(os.path.join(P, fn), encoding='utf-8').read()
    src = src.replace("os.path.join(P18T, 'result_phase18.json')",
                      "os.path.join(P18T, 'result_phase18_smoke.json')")
    src = src.replace("'%s'" % out_repl[0], "'%s'" % out_repl[1])
    src = src.replace("A0, A1, A2 = ARMS", "A0, A1, A2 = (list(ARMS) + list(ARMS) + list(ARMS))[:3]")
    try:
        exec(compile(src, fn + ':dry', 'exec'), {'__name__': '__dry__'})
        print('[OK] %s 干跑成功' % fn)
    except Exception as e:
        import traceback
        print('[FAIL] %s -> %r' % (fn, e))
        traceback.print_exc()
