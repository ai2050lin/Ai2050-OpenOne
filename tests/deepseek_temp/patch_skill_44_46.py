# -*- coding: utf-8 -*-
"""向 rdc-phase-closeout 追加教训 44–46（Q04 实测），文件化补丁 + count 断言 + 回读。"""
import hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
raw = open(P, 'rb').read()
txt = raw.decode('utf-8')
before_sha = hashlib.sha256(raw).hexdigest()[:8]

ANCHOR = '## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）'
assert txt.count(ANCHOR) == 1, 'anchor count=%d' % txt.count(ANCHOR)

NEW = """44. **预注册哈希域只能含不变量 —— 混入时间戳会让冻结每次重跑都「DESIGN DRIFT」（Q04 实证，2026-10-03）**
   (a) Q04 首次把 `created=<now>` 放进被哈希的 `design` 字典 ⇒ 第二次运行同一脚本立刻 `AssertionError: DESIGN DRIFT`，
       冻结形同废纸（写完执行脚本、跑过一次 SMOKE 之后才暴露）。
   (b) **固定做法**：`design`（进哈希）只放不变量（面板/网格/阈值/口径/预测器/门）；**易变字段**（时刻、运行机、耗时、日志路径）
       一律置于哈希域之外，只写进 run_log 与 `result`。
   (c) 幂等自检：同一 (模型, SMOKE) 参数条件下，第二次运行必须打印 `execution.json match (sha ...)`；这是冻结是否合格的最快判据。
   (d) 一旦哈希域改过（哪怕只是"移出时间戳"）⇒ 旧 `execution.json` 必须删除重冻结，并在 MEMO 如实记"首版冻结被工程性取代"。

45. **新 KPI 首次建装置时，必须做「阈值量纲体检」：门是否**真的可失败**？（Q04 实证）**
   (a) Q04 给 E_ar 的 S1 门沿用了 5% 门族阈 **0.05**，但 E_ar 是**原始 logit** 量纲（实测 3–4）⇒ 阈被超 **88×**，
       "形式上可失败、实际上必过"，**不具否证力**——这类门写进预注册会制造"看着有门其实无门"的假安全感。
   (b) **纪律**：装置冻结前，先问三件事——① 该量的**自然量纲**是什么？② 阈值是否与被测量同量纲？③ 在预期量级下，
       门**可能**FAIL 吗？若答案为否，改**相对形式**（如 `E_ar/scale`，scale 取 held-out 目标量 std）。
   (c) **阈值冻结后不得事后调门**（防事后调门）。正确处置：**保留原门**并降级为"装置灵敏度门"（证明装置能看见信号），
       另立**相对形式**作为**下一 Phase 的科学门**，且必须在**下一 Phase 观测之前**冻结——在结果里写 `q0X_prereg` 区块随产物封存。
   (d) 类推：`E_read` 的 0.05 是**归一化 MSE**（比值），`E_ar` 的 logit L1 不是比值 —— **跨 KPI 借用阈值必须先核对量纲**。

46. **截断哈希的比对函数必须同样截断 —— 64 位 vs 8 位会伪报「self-hash 不符」（Q04 实证）**
   (a) 项目多处用 `*_sha256_8`（8 位截断）。Q04 写的 `content_hash()` 返回**完整 64 位** hexdigest，
       却与文件里的 8 位 `content_sha256_8` 相比 ⇒ 断言必错，且**误报方向**是"文件坏了，不是函数错了"。
   (b) **固定做法**：比较函数返回值与 `assert` 目标**同宽**；哈希工具函数在命名上就标明宽度（`sha8()` / `content_hash8()`）。
   (c) 排错提示：当 self-hash 突然不符、而你刚确认过文件 sha8 没变，**先怀疑比对函数的宽度/编码/是否 sort_keys**，
       用一次独立的最小复算（`json.dumps(c, ensure_ascii=False, indent=1)`，**不** `sort_keys`）就能定位。

"""

txt2 = txt.replace(ANCHOR, NEW + ANCHOR)
assert txt2 != txt and txt2.count(NEW) == 1, 'insert failed'
open(P, 'wb').write(txt2.encode('utf-8'))

rb = open(P, 'rb').read()
t2 = rb.decode('utf-8')
print('skill bytes %d/%s -> %d/%s' % (len(raw), before_sha, len(rb), hashlib.sha256(rb).hexdigest()[:8]))
print('  lesson44=%s lesson45=%s lesson46=%s anchor_intact=%s'
      % ('44.' in t2, '45.' in t2, '46.' in t2, t2.count(ANCHOR) == 1))
print('DONE')
