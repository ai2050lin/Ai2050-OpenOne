# -*- coding: utf-8 -*-
"""修正 closeout_docs_phase19.py 的 5 处渲染缺陷，并把 wlog 回滚到 P19 追加前再重跑。

缺陷：
  (1) `U_ℓ` 秩误写 len(classes)=6 -> 应为 n_classes-1=5
  (2) 主结果1 标签 'P=PASS' -> '= PASS'
  (3) 主结果1 尾部 '**DEEP**/四臂 3/3 深端' 措辞
  (4) 主结果2 的 nf4/bf16 侧标签接线错误（A1 的 nf4 值被当成 bf16），且 A1 未四舍五入
  (5) 限界 '③ `A1_bf16`·bf16' 冗余
"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P = os.path.join(ROOT, 'tests', 'deepseek', 'Phase19', 'closeout_docs_phase19.py')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')

s = io.open(P, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, 'count=%d for %r' % (s.count(old), old[:60])
    s = s.replace(old, new)


# (1) 秩
rep("     EX['neighbourhood_width'], R['bootstrap']['BP']))",
    "     EX['neighbourhood_width'], R['bootstrap']['BP']))")  # no-op guard (existence)
rep("len(EX['instances_all']), len(EX['classes']),",
    "len(EX['instances_all']), len(EX['classes']) - 1,")

# (2) 标签
rep('（P2·同模型 P=%s / P3·跨家族 holdout P=%s）', '（P2·同模型 = %s / P3·跨家族 holdout = %s）')

# (3) 尾部措辞
rep('⇒ **`%s`/`%s`**。\'\n  % (\'PASS\' if PC[\'P2\']', "⇒ **`%s`**（%s）。'\n  % ('PASS' if PC['P2']")
rep("JV['Q6_joint'], '四臂 3/3 深端' if JV['Q6_joint'].endswith('ALL') else ''))",
    "JV['Q6_joint'], '四臂 bf16 与 nf4 同判深端' if JV['Q6_joint'].endswith('ALL') else ''))")

# (4) nf4/bf16 侧接线 + A1 四舍五入
rep("     pk('A0_nf4|A0_bf16', 'share_mlp_nb_nf4'), pk('A1_nf4|A1_bf16', 'share_mlp_nb_nf4'),",
    "     pk('A0_nf4|A0_bf16', 'share_mlp_nb_nf4'), pk('A0_nf4|A0_bf16', 'share_mlp_nb_bf16'),")
rep("     v1 := V[A1n]['share_mlp_nb'], V[A1b]['share_mlp_nb'], fn(FL['MLP_DOM_MIN'], 2), JV['Q5_joint'],",
    "     fn(V[A1n]['share_mlp_nb'], 4), fn(V[A1b]['share_mlp_nb'], 4), fn(FL['MLP_DOM_MIN'], 2), JV['Q5_joint'],")

# (5) 限界冗余
rep("③ `%s`·bf16 需 CPU offload", "③ `%s` 需 CPU offload")

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
s2 = io.open(P, encoding='utf-8').read()
assert "len(EX['classes']) - 1," in s2 and "v1 :=" not in s2 and "share_mlp_nb_bf16')," in s2
print('SOURCE PATCH OK  bytes=%d' % len(s2.encode('utf-8')))

# ---- wlog 回滚到 P19 追加前 ----
b = open(WLOG, 'rb').read()
txt = b.decode('utf-8')
key = '\r\n\r\n## Phase 19 / N2h1-'
i = txt.find(key)
assert i > 0, 'wlog 未找到 P19 段'
pre = txt[:i]
if len(pre.encode('utf-8')) + 2 == 72596:
    pre += '\r\n'
elif len(pre.encode('utf-8')) == 72596:
    pass
else:
    raise AssertionError('回滚后长度异常: %d' % len(pre.encode('utf-8')))
open(WLOG, 'wb').write(pre.encode('utf-8'))
b2 = open(WLOG, 'rb').read()
assert len(b2) == 72596, 'wlog 回滚失败 len=%d' % len(b2)
assert '## Phase 19 /' not in b2.decode('utf-8')
print('WLOG ROLLBACK OK  bytes=%d -> %d' % (len(b), len(b2)))
