# -*- coding: utf-8 -*-
"""R6: 技能 rdc-phase-closeout 追加「教训 40」（插在 ## 参照实现 之前）。"""
import os, hashlib

SKILL = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_skill40_r6_report.txt"
def sh8b(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

LESSON40 = r"""
40. **跨目录归位后，脚本内嵌的输出路径会变成死路径（R6 实证，2026-10-03）**：

```text
(a) 两步分离：归位脚本只搬文件，补丁脚本只改脚本文本。一次执行里既搬又改，失败时无法判断坏在哪一半。
(b) 白名单精确串，禁全局 regex：用 "tests","deepseek_temp","_review" / tests/deepseek_temp/_review /
    tests\deepseek_temp\_review / tests/deepseek/_review/ 逐条替换；**绝不要对 `_review` 做全局替换**
    —— `propositions_review` / `propositions_new` 这类变量名会被误伤（本轮哨兵计数 6 处不变）。
(c) 替换顺序：先替换「带引号的整串」，再替换裸路径；否则后者的改写会把前者的模式重新切碎。
(d) 反向复核：改完必须 grep 确认「残留死路径 = 0」，并统计一个已知哨兵词计数不变，用以证明没误伤。
(e) 产物哈希会变：改写 generated_by 这类**会写进产物正文**的字符串时，重跑生成的产物哈希必然改变；
    冻结产物的哈希一律以归位**前**的 MEMO 记录为准。
(f) 大段并入的标题必须降级：把整篇 .md 并入 MEMO 成 `## Phase {N}` 时，正文标题降一级（# -> ##），
    并**跳过代码围栏内的 #**；同时**剥掉文档自身的一级标题**，否则它会与 `## Phase` 同级、污染 Phase 目录。
(g) 只读冻结件先备份再动：并入后要删除的原件，先复制到备份目录并**逐字节比对哈希**，再删；删前再校验一次。
```

"""
ANCH = "## 参照实现（Phase 3125"
st = open(SKILL, "rb").read().decode("utf-8-sig")
assert st.count(ANCH) == 1, "anchor count=%d" % st.count(ANCH)
assert "40. **跨目录归位" not in st, "lesson 40 already present"
st2 = st.replace(ANCH, LESSON40.lstrip("\n") + "\n" + ANCH, 1)
open(SKILL, "w", encoding="utf-8", newline="").write(st2)
sb = open(SKILL, "rb").read().decode("utf-8")
assert sb == st2, "readback mismatch"
assert "40. **跨目录归位" in sb
assert "39. **一次脚本多处写入" in sb
out = "[SKILL] %d -> %d chars  sha8 %s -> %s  lesson40=%s" % (
    len(st), len(sb), sh8b(SKILL) if False else hashlib.sha256(st.encode()).hexdigest()[:8], sh8b(SKILL), "40. **跨目录归位" in sb)
open(REPORT, "w", encoding="utf-8").write(out)
print(out)
print("SKILL40_OK")
