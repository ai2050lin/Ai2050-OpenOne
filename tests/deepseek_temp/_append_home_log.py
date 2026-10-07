# -*- coding: utf-8 -*-
import io, os

p = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-07.md"
entry = u"""
## /rdc-fusion 设为默认首页（2026-10-07 03:2x）

- `frontend/src/main.jsx` 路由分支重排：`/` 与 `/rdc-fusion` 等价渲染融合驾驶舱；未匹配路径兜底也落融合页。
- 旧版首页（App.jsx + 右下角 rdc-query 浮标）整体移到 `/legacy`，**原样保留、无入口链接**；其余 rdc-* 子页路径不变。
- playwright 验证（tests/fusion_verify_home.mjs → workspace 跑）：4/4 PASS（/ 为融合页、/rdc-fusion 正常、/legacy 渲染旧版、/rdc-query 不受影响）；分路径错误探测：`/` 0 错误，/legacy 的 snapshot.json 请求失败为旧版自身旧有行为。截图 tests/fusion_home_default.png。
"""
with io.open(p, "r", encoding="utf-8") as f:
    cur = f.read()
if u"设为默认首页" not in cur:
    with io.open(p, "a", encoding="utf-8") as f:
        f.write(entry)
print("LOG_OK", os.path.getsize(p))
