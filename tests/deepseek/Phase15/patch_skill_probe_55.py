# -*- coding: utf-8 -*-
"""给 rdc-main-axis-probe 追加坑 55（极值型集中度统计量跨模型不可移植）。"""
import io, hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
s = io.open(P, encoding='utf-8').read()
b0 = hashlib.sha256(s.encode('utf-8')).hexdigest()

ANCHOR = '\n## 5 代码骨架要点'
assert s.count(ANCHOR) == 1, 'anchor not unique: %d' % s.count(ANCHOR)

NEW = '''

55. **「极值型 3-窗口占比」这类集中度统计量可能**在两个坐标、三个模型上同时失效** —— 任何集中度断言必须同时声明**坐标 + 模型 + 零假设分位**（Phase 15 实证：6 格只活 2 格）**：口径 = `top3_share = max_w |Σ_{j=w}^{w+2} jumps_j| / range`，零假设 = 把同一组 `jumps` **幅度重排**后重算取 95 分位（零额外前向），裕度 = `share − null95`。Phase 15 在 **3 臂 × 2 坐标 = 6 格**上逐格做零假设校准，结果只有 **2 格**过线：
    - A0·`J` `0.8319` vs null `0.7262`（裕度 **+0.1057**）；A1·`xhalf` `0.5741` vs `0.5384`（**+0.0357**）；
    - A0·`xhalf` `0.5454` vs `0.6982`（**−0.1528**）；A1·`J` `0.5904` vs `0.6159`（**−0.0256**）；A2·`xhalf` `0.3454` vs `0.4367`（**−0.0913**）；A2·`J` `0.5375` vs `0.6592`（**−0.1217**）。
    **两种翻转同时存在**：① **换坐标翻转**（同一臂内 `xhalf` 与 `J` 结论可相反，如 A0 的 `J` 超而 `xhalf` 不超）；② **换模型翻转**（同一坐标在不同臂上「超 / 不超」互换，如 A1 的 `xhalf` 超而 `J` 不超）。⇒ Phase 13/14 的「坐标依赖」必须再升一级为「**坐标 × 模型**双重依赖」，**不得再说「某个坐标才是有区分力的坐标」**。
    **对策（三条）**：① `null95` 与裕度**逐格同报**，禁止只报 `share`（死线条款）；② 换用**对排序 / 单窗口不敏感**的量（谱熵、窗口加权质心、深度去势后的残差集中度），并在**多臂 × 双坐标**上一次性重算历史集中度表；③ **不要把「argmax 距离 ≥3」与「两窗集中度都显著」混为一谈** —— 前者（`d_argmax`）是**窗的位置**问题，本 Phase 三臂都过（14/10/5 ≥ 3）；后者是**显著性**问题，本 Phase 大面积不过。Q2 判「层栈性质」时**只能**指向前者。
    **同时注意「窗的位置」在量化口径下不稳健**：A0 的 `argmax_w_j` 在 nf4 下 = 0、bf16 下 = 1（`xhalf` 的 `argmax_w_x` 两口径都是 14）⇒ **校准门要写在最稳健的量（`xhalf`）上**，跨 Phase 复用的层索引须在**原口径（bf16）**下重测。
'''

s = s.replace(ANCHOR, NEW + ANCHOR)
io.open(P, 'w', encoding='utf-8').write(s)
s2 = io.open(P, encoding='utf-8').read()
assert s2 == s, 'disk readback mismatch'
print('OK bytes 63746 -> %d' % len(s.encode('utf-8')))
print('sha256 before %s after %s' % (b0[:16], hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]))
print('pit55 present:', '55. **「极值型 3-窗口占比」' in s2)
