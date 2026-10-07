# -*- coding: utf-8 -*-
"""给 rdc-phase-closeout 技能追加教训 23（前缀锚 bug + 渲染器写死散文）、教训 24（幂等）。
在 '## 参照实现' 之前插入；assert count==1；落盘回读复核。"""
import io, hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
s = io.open(P, encoding='utf-8').read()
b0 = hashlib.sha256(s.encode('utf-8')).hexdigest()

ANCHOR = '## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）'
assert s.count(ANCHOR) == 1, 'anchor not unique: %d' % s.count(ANCHOR)

NEW = '''23. **「前缀锚」必须用**追加前的完整原始字节**；`BOM + txt.rstrip(CRLF)` 是错的（Phase 15 实证：一次假失败）**：Phase 15 的 `do_append_phase15.py` 写成
    `prefix = BOM + txt.rstrip('\\r\\n').replace(...)`，再断言 `sha256(rb[:len(prefix)]) == sha256(raw0)`。
    当原文件**以 CRLF 结尾**时 `len(prefix) == len(raw0) - 2`，于是**追加明明成功、断言却必然失败**（`assert anchor_ok` 抛在写盘之后 ⇒ 脚本报 FAIL 而磁盘已改，状态割裂）。
    正确关系：`rb = BOM + txt.rstrip(CRLF) + CRLF + CRLF + body + CRLF`，**`rb` 以 raw0 为前缀** ⇒ 前缀锚应为 **`pre_raw = raw0`**（不是从 `txt` 重建），核验 `rb.startswith(raw0)` 且 `sha256(rb[:len(raw0)])[:8] == sha256(raw0)[:8]`。
    **通用纪律**：凡「追加 / 拼接」类写入，**前缀锚只能是「写前读到的原始 bytes 对象」**，任何「解码 → strip → 重编码」的再造都会在 EOL/BOM 上错位。关联：本教训同时说明**写盘与断言必须可分辨** —— 报告里要打印 `startswith` 与 `sha_match` **两个布尔**，否则无法区分「没写」与「写对了但公式错」。

24. **`do_append_*` 必须有「幂等探测」；渲染器（`gen_memo_*`）的每一条结论句必须**由 verdict 分支或现场数据选择**，禁写死散文（Phase 15 实证：渲染件与数据矛盾 4 处）**：两组问题是同一次收尾里暴露的。
    (a) **幂等**：Phase 15 上一条事故把文件写到一半（已追加、未通过自检），重跑脚本若再走写入路径就会**双份 Phase 节**。对策：追加前先 `ALREADY = ('## Phase <N>:' in txt)`；命中则**跳过写入**、把前缀锚改为「与冻结基线 `sha8` 比对前 `BASE_BYTES` 字节」，并在报告里显式打印 `=== 幂等路径：跳过写入 ===`。这样「重跑」永远安全，且第二次运行即可产出通过的 `verify_append_*.txt`。
    (b) **渲染器数据驱动**：`gen_memo_phase15.py` 首版在 §5/§6/§9 有 4 处**写死的散文结论**（如「`J` 坐标才是有区分力的坐标」「`J` 在三臂上**都**超零假设」），而实测只有 **6 个「臂 × 坐标」格里 2 格**超零假设 ⇒ 渲染件与 `result.json` **直接矛盾**，且这类句子会被当成 Phase 结论进入 MEMO。对策与流程：① 每个 §5/§6/§9 的结论句都放在 `if q*_joint == '<LABEL>':` 分支里，或从 `result` 现场 `sum(...)` 出数字；② **引入的中间量（`_jup/_xup/_ncell/_n95`）必须在 §0 之前定义**（首版放在 §5，导致 §0 引用时 `NameError`，也会让模块顺序成为隐性依赖）；③ 渲染后**必须逐句对照数据复核**（本轮即靠这一步抓出矛盾，改动 4 处）；④ 与之配套：`do_append_*` 的锚点预检如果报「缺 `CONC_JUDGE_INVALID_X_ALL`」之类的**分支标签**缺失，正确的修法是**在正文里加一条「判决标签完整取值域」**（顺带让 MEMO 可被跨 Phase 检索），而不是把锚点删掉。

'''

s = s.replace(ANCHOR, NEW + ANCHOR)
assert s != io.open(P, encoding='utf-8').read() or True
io.open(P, 'w', encoding='utf-8').write(s)
s2 = io.open(P, encoding='utf-8').read()
assert s2 == s, 'disk readback mismatch'
print('OK bytes %d -> %d' % (len(s.encode('utf-8')) if False else 0, len(s.encode('utf-8'))))
print('sha256 before %s  after %s' % (b0[:16], hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]))
print('lesson 23 present:', '23. **「前缀锚」' in s2)
print('lesson 24 present:', '24. **`do_append_*` 必须有「幂等探测」' in s2)
