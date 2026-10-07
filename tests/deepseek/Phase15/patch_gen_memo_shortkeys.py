# -*- coding: utf-8 -*-
"""把 gen_memo_phase15.py 里还剩的长臂名 dict 键一律换成 SHORT（可读性 + 与表头一致）。
每个替换都 assert count==1，改成 0 处即报错，避免幻影编辑。"""
import io, hashlib, os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen_memo_phase15.py')
s = io.open(P, encoding='utf-8', newline='').read()
orig = s
before = hashlib.sha256(s.encode('utf-8')).hexdigest()

REPL = [
    # §5 结论 4：top3_j / null95_j
    ("json.dumps({a: f(g(E5C, a, 'top3_j'), 4) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E5C, a, 'top3_j'), 4) for a in ARMS_ALL}, ensure_ascii=False)"),
    ("json.dumps({a: f(g(E5C, a, 'null_j', 'null95'), 4) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E5C, a, 'null_j', 'null95'), 4) for a in ARMS_ALL}, ensure_ascii=False)"),
    # §5 结论 5：spearman(xhalf) / spearman(J) / XH_RANGE
    ("json.dumps({a: f(g(E5C, a, 'spearman_xh_depth'), 4) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E5C, a, 'spearman_xh_depth'), 4) for a in ARMS_ALL}, ensure_ascii=False)"),
    ("json.dumps({a: f(g(E5C, a, 'spearman_J_depth'), 4) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E5C, a, 'spearman_J_depth'), 4) for a in ARMS_ALL}, ensure_ascii=False)"),
    ("json.dumps({a: f(g(E4S, a, 'XH_RANGE'), 4) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E4S, a, 'XH_RANGE'), 4) for a in ARMS_ALL}, ensure_ascii=False)"),
    # §9：spearman(J) / XH_RANGE / margin_j / margin_x
    ("json.dumps({a: f(g(E5C, a, 'spearman_J_depth'), 3) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E5C, a, 'spearman_J_depth'), 3) for a in ARMS_ALL}, ensure_ascii=False)"),
    ("json.dumps({a: f(g(E4S, a, 'XH_RANGE'), 4) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E4S, a, 'XH_RANGE'), 4) for a in ARMS_ALL}, ensure_ascii=False)"),
    ("json.dumps({a: f(g(E5C, a, 'margin_j'), 4) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E5C, a, 'margin_j'), 4) for a in ARMS_ALL}, ensure_ascii=False)"),
    ("json.dumps({a: f(g(E5C, a, 'margin_x'), 4) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E5C, a, 'margin_x'), 4) for a in ARMS_ALL}, ensure_ascii=False)"),
    # §9：pred 列表
    ("json.dumps({a: f(g(E5C, a, 'margin_x'), 4) for a in ARMS_ALL}, ensure_ascii=False)",
     "json.dumps({SHORT[a]: f(g(E5C, a, 'margin_x'), 4) for a in ARMS_ALL}, ensure_ascii=False)"),
]

log = []
for old, new in REPL:
    if old == new:
        continue
    c = s.count(old)
    if c == 0:
        log.append('SKIP(count=0) %s' % old[:70])
        continue
    s = s.replace(old, new)
    log.append('REPLACED x%d %s' % (c, old[:70]))

after = hashlib.sha256(s.encode('utf-8')).hexdigest()
assert s != orig, 'nothing changed'
io.open(P, 'w', encoding='utf-8', newline='').write(s)

# 落盘复核
s2 = io.open(P, encoding='utf-8', newline='').read()
assert s2 == s, 'disk readback mismatch'

print('PATCH OK')
print('  sha256 before = %s' % before[:16])
print('  sha256 after  = %s' % after[:16])
print('  bytes %d -> %d' % (len(orig.encode('utf-8')), len(s.encode('utf-8'))))
for l in log:
    print('  ' + l)
# 残留长键检查
import re
leftover = [m.start() for m in re.finditer(r"json\.dumps\(\{a:", s)]
print('  leftover_longkey_dicts = %d' % len(leftover))
