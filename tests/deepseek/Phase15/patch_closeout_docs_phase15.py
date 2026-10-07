# -*- coding: utf-8 -*-
"""closeout_docs_phase15.py 的三处修补：
(a) 主结果 1 里 4B 的「x 集中窗 L28-L34 / J 集中窗 L7-L10」改为从 E5[A0] 现场取值（原值 1 处手写有偏移）；
(b) 新增「6 格零假设校准只有 2 格存活」这条自我修正（本轮最重要结论之一，原 wlog 缺）；
(c) 死线的第二/第三候选改为与 MEMO §8 一致。
每处 assert count==1 并落盘复核。
"""
import io, hashlib, os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'closeout_docs_phase15.py')
s = io.open(P, encoding='utf-8', newline='').read()
orig = s
b0 = hashlib.sha256(s.encode('utf-8')).hexdigest()

# ---- (a) 4B 集中窗改现场取值
old_a = ("p('- **主结果 1（Q2 两坐标 argmax 距离）**：Phase 13 在 qwen3-4b 上给 `MODE_X = %s` / `MODE_J = %s`（差 **13** 个窗口单位，'\n"
         "  'x 集中窗 L28–L34、J 集中窗 L7–L10）。本轮跨模型：%s ⇒ 合取判决 **`%s`**。'\n"
         "  % (JUD['reference_4B']['MODE_X_13'], JUD['reference_4B']['MODE_J_13'],\n")
new_a = ("p('- **主结果 1（Q2 两坐标 argmax 距离）**：Phase 13 在 qwen3-4b 上给 `MODE_X = %s` / `MODE_J = %s`（差 **13** 个窗口单位；'\n"
         "  'x 集中窗 = w%s [%s..%s]、J 集中窗 = w%s [%s..%s]）。本轮跨模型：%s ⇒ 合取判决 **`%s`**。'\n"
         "  % (JUD['reference_4B']['MODE_X_13'], JUD['reference_4B']['MODE_J_13'],\n"
         "     (E5[A0]['win_sem_x'] or {}).get('w'), (E5[A0]['win_sem_x'] or {}).get('a'),\n"
         "     (E5[A0]['win_sem_x'] or {}).get('b'),\n"
         "     (E5[A0]['win_sem_j'] or {}).get('w'), (E5[A0]['win_sem_j'] or {}).get('a'),\n"
         "     (E5[A0]['win_sem_j'] or {}).get('b'),\n")
assert s.count(old_a) == 1, 'patch(a) anchor not unique: %d' % s.count(old_a)
s = s.replace(old_a, new_a)

# ---- (b) 新增 6 格零假设校准 bullet（插在「主结果 2」之后、剖面形状之前）
anchor_b = "p('- **剖面形状（Q5，描述性）**：%s。' %\n"
new_b = (
    "p('- **⚠️ 本轮最重要的自我修正（零假设校准的逐格复核）**：对 `J` 坐标施加**同一套**置换零假设检验后，'\n"
    "  '**6 个「臂 × 坐标」格里只有 %d 格超过各自的 null 95 分位**（%s）⇒ 三臂上**不存在跨模型稳健的集中度判据**；'\n"
    "  '「坐标依赖」须再升一级为「**坐标 × 模型**双重依赖」，且**不得再说「某个坐标才是有区分力的坐标」**。'\n"
    "  '（`J` 裕度 = %s；`xhalf` 裕度 = %s）'\n"
    "  % (sum(1 for a in ARMS if (E5[a]['margin_j'] or -1) > 0 or (E5[a]['margin_x'] or -1) > 0),\n"
    "     ' / '.join('%s·%s（裕度 %+.4f）' % (a, 'J' if (E5[a]['margin_j'] or -1) > 0 else 'xhalf',\n"
    "                                       E5[a]['margin_j'] if (E5[a]['margin_j'] or -1) > 0 else E5[a]['margin_x'])\n"
    "                for a in ARMS if (E5[a]['margin_j'] or -1) > 0 or (E5[a]['margin_x'] or -1) > 0),\n"
    "     ' / '.join('%s %+.4f' % (a, E5[a]['margin_j']) for a in ARMS),\n"
    "     ' / '.join('%s %+.4f' % (a, E5[a]['margin_x']) for a in ARMS)))\n"
)
assert s.count(anchor_b) == 1, 'patch(b) anchor not unique: %d' % s.count(anchor_b)
s = s.replace(anchor_b, new_b + anchor_b)

# ---- (c) 死线的第二/第三候选改为与 MEMO §8 一致
old_c = ("  '第二候选：**α 网格不变量的正式化**（`xhalf` 已证为网格不变量、`J` 已证不是 ⇒ 用 dense 网格在 glm4/14B 上复算 `XH_RANGE` 的网格稳定性）。'\n"
         "  '第三候选：**位置通道容量的上界测量**（多模板 × 多位置，含 3 位置模板）—— 仍挂账。')\n")
new_c = ("  '**但不得直接沿用本轮的 `J` 窗口索引**：`argmax_w_j` 恰是本轮唯一被量化噪声换掉的量（nf4 vs bf16 换窗）。'\n"
         "  '第二候选（因「6 格只活 2 格」而升为并列最高优先）：**集中度统计量的重设计** —— `top3_share` 这种'\n"
         "  '「极值型 3-窗口占比」在两个坐标上都缺跨模型稳健性，需换**对排序 / 单窗口不敏感**的量'\n"
         "  '（谱熵、窗口加权质心、或深度去势后的残差集中度），并在三臂 × 双坐标上重算 Phase 12/13/14 的全部集中度表。'\n"
         "  '第三候选：**家族 vs 规模的解耦**（补第三个同家族不同规模 / 第二个 GLM 规模点）。'\n"
         "  '仍挂账：N2h1-α-1 权重级定位、N2h1-β 水果类崩塌解剖、R1 对照补强、K4 处置、N 线 Phase 3–7 补登 Ledger。')\n")
assert s.count(old_c) == 1, 'patch(c) anchor not unique: %d' % s.count(old_c)
s = s.replace(old_c, new_c)

assert s != orig
io.open(P, 'w', encoding='utf-8', newline='').write(s)
s2 = io.open(P, encoding='utf-8', newline='').read()
assert s2 == s, 'disk readback mismatch'
print('PATCH closeout_docs OK')
print('  sha256 before %s' % b0[:16])
print('  sha256 after  %s' % hashlib.sha256(s.encode('utf-8')).hexdigest()[:16])
print('  bytes %d -> %d' % (len(orig.encode('utf-8')), len(s.encode('utf-8'))))
