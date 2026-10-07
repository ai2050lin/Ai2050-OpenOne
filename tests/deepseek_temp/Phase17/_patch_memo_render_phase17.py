# -*- coding: utf-8 -*-
"""一次性补丁：把 gen_memo_phase17.py 中手工转录的数字改为现场渲染（铁律 (ae)）。
涉及 §1 动机锚、§5 读数、§6 A0 对齐段、§10 E4。
（幻影编辑缺陷 -> 用 Python patch + assert count==1 + 回读复核。）"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase17\gen_memo_phase17.py'
s = io.open(P, encoding='utf-8').read()

REPS = []

# ---- 0) 顶部引入 v2 留痕（E4 的「修正前」com_V 亦现场读取）
REPS.append((
    "ARMS = list(RES['arms'].keys())\nSH = {",
    "ARMS = list(RES['arms'].keys())\n"
    "V2P = os.path.join(P17T, 'result_phase17_v2_preintervalfix.json')\n"
    "V2 = json.loads(open(V2P, 'rb').read().decode('utf-8')) if os.path.exists(V2P) else None\n"
    "_V2CV0 = float(V2['verdict'][ARMS[0]]['Q3_com_V']) if V2 else None\n"
    "assert _V2CV0 is not None, 'v2 留痕缺失，§10 E4 无法现场渲染'\n"
    "SH = {",
))

# ---- 1) §1 动机：A0/A1/A2 行为质心由锚现场渲染
REPS.append((
    "  'Phase 16 实测的行为质心是 A0 **23.0** / A1 **18.2** / A2 **8.4** 层 —— 这三个层，谁都没查过。')\n",
    "  'Phase 16 实测的行为质心是 A0 **%s** / A1 **%s** / A2 **%s** 层 —— 这三个层，谁都没查过。'\n"
    "  % tuple(f(SEAL['anchor_values'][a]['com_layer_x'], 1) for a in ARMS))\n",
))

# ---- 2) §5 读数：探针/生产 com_V 均现场渲染；删去手工的 24.47
REPS.append((
    "A('读数：探针（A0，24 对）com_V = **26.15**；正式运行（A0，24 对）com_V = **%s** —— '\n"
    "  '**逐位一致**（首版生产实现取位点单层质量而非区间求和，给出 24.47；修正后与探针完全吻合，见 §10 E4）。'\n"
    "  '三个臂的 `w_ℓ` **都在 L28–L34 附近达到峰值**，浅端在写入窗附近另有一个次峰。'\n"
    "  % f(V[ARMS[0]]['Q3_com_V'], 3))\n",
    "A('读数：探针（A0，24 对）com_V = **%s**；正式运行（A0，24 对）com_V = **%s** —— '\n"
    "  '**逐位一致**（首版生产实现取位点单层质量而非区间求和，见 §10 E4；修正后与探针完全吻合）。'\n"
    "  '三个臂的 `w_ℓ` **都在 L28–L34 附近达到峰值**，浅端在写入窗附近另有一个次峰。'\n"
    "  % (f(V[ARMS[0]]['Q3_com_V'], 2), f(V[ARMS[0]]['Q3_com_V'], 3)))\n",
))

# ---- 3) §6 A0 对齐段：24.5 / 1.46 / 23.0 / 8.4 全部现场渲染
REPS.append((
    "A('**A0 是唯一「对齐」的臂**（min(d) = %s 层 < 4）：A0 的 `com_layer(x)` 恰在深端（23.0），'\n"
    "  '与向量质心（24.5）只差 1.46 层。这**正是 Phase 16 P6 否证的镜像**：'\n"
    "  'A0 上「行为质心深」与「向量质心深」同时成立是**碰巧对齐**，一旦换到 A2（同家族 3.5×）'\n"
    "  '行为质心跑到浅端（8.4）而向量质心仍在深端（24.5），两者立刻分开。'\n"
    "  % f(V['A0_calib_qwen3-4b-nf4']['Q4_min_d'], 2))\n",
    "A('**A0 是唯一「对齐」的臂**（min(d) = %s 层 < %s）：A0 的 `com_layer(x)` 恰在深端（%s），'\n"
    "  '与向量质心（%s）相差 %s 层。这**正是 Phase 16 P6 否证的镜像**：'\n"
    "  'A0 上「行为质心深」与「向量质心深」同时成立是**碰巧对齐**，一旦换到 A2（同家族 3.5×）'\n"
    "  '行为质心跑到浅端（%s）而向量质心仍在深端（%s），两者立刻分开。'\n"
    "  % (f(V[ARMS[0]]['Q4_min_d'], 2), FL['CENTROID_SEP_MIN'],\n"
    "     f(SEAL['anchor_values'][ARMS[0]]['com_layer_x'], 1), f(V[ARMS[0]]['Q3_com_V'], 3),\n"
    "     f(V[ARMS[0]]['Q4_min_d'], 2),\n"
    "     f(SEAL['anchor_values'][ARMS[2]]['com_layer_x'], 1), f(V[ARMS[0]]['Q3_com_V'], 3)))\n",
))

# ---- 4) §10 E4：修正前/后 com_V 与差值全部现场渲染（并入同一 % 元组）
REPS.append((
    "  '两者在 A0 上差 **1.68 层**（24.466 vs 26.150）。修正为区间求和后，A0 的 `com_V` 与**探针**'\n"
    "  '（独立脚本、24 对）的 **26.15 逐位相同** —— 这同时是一次**独立实现之间的交叉验证**。'\n",
    "  '两者在 A0 上差 **%s 层**（%s vs %s）。修正为区间求和后，A0 的 `com_V` 与**探针**'\n"
    "  '（独立脚本、24 对）的 **%s 逐位相同** —— 这同时是一次**独立实现之间的交叉验证**。'\n",
))
REPS.append((
    "  '`min(d)` 与 `spearman` 方向不变（P3/P4/P6 判决不变）。'\n"
    "  % (f(V[ARMS[0]]['Q3_com_V'], 3), f(V[ARMS[1]]['Q3_com_V'], 3), f(V[ARMS[2]]['Q3_com_V'], 3)))\n",
    "  '`min(d)` 与 `spearman` 方向不变（P3/P4/P6 判决不变）。'\n"
    "  % (f(abs(_V2CV0 - V[ARMS[0]]['Q3_com_V']), 2), f(_V2CV0, 3), f(V[ARMS[0]]['Q3_com_V'], 3),\n"
    "     f(V[ARMS[0]]['Q3_com_V'], 2),\n"
    "     f(V[ARMS[0]]['Q3_com_V'], 3), f(V[ARMS[1]]['Q3_com_V'], 3), f(V[ARMS[2]]['Q3_com_V'], 3)))\n",
))

for i, (old, new) in enumerate(REPS):
    n = s.count(old)
    print('REP[%d] count=%d' % (i, n))
    assert n == 1, 'REP[%d] expect 1, got %d' % (i, n)
    s = s.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)

# 回读复核：确认不再有手工数字，且新渲染片段存在
t = io.open(P, encoding='utf-8').read()
for bad in ('A0 **23.0** / A1 **18.2**', '给出 24.47；', '（24.5）只差 1.46 层', '**1.68 层**（24.466'):
    assert bad not in t, 'still present: %s' % bad
for good in ('_V2CV0', "SEAL['anchor_values'][a]['com_layer_x'], 1", "FL['CENTROID_SEP_MIN']"):
    assert good in t, 'missing: %s' % good
print('PATCH OK')
