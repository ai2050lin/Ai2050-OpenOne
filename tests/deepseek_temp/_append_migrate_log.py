# -*- coding: utf-8 -*-
# 追加 2026-10-07 工作区日志：三 tab 从研发透镜迁移到路线透镜
import io

LOG = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-07.md"

entry = """
## 三 tab 迁移：研发透镜回退 v2 → 路线透镜 v3（14:06）

**用户纠正**：业界新闻/平台进度/当前机器进度三按钮应加在**路线界面**而非研发界面。

**回退**（LensProcess.jsx + CSS）：
- LensProcess.jsx 恢复 v2（删 view 状态/tab 条/节点横幅/platform+news 视图/import distributedData），多模型配置等 v2 成果完整保留
- CSS：`.fw-process` 恢复原样（删 align-content:start 与 `.fw-view.on.fw-process{display:grid}`，回到 v2 实际 flex 布局）；三 tab 样式（fw-pt-*/fw-node-*/fw-pl-*/fw-tmx-*/fw-agg-*/fw-news-*）保留给路线界面，`.fw-pt-tabs` 加 margin-bottom:14px

**迁移**（LensProgress.jsx 全量重写 v3）：
- 顶层三 tab（用户口序）：业界新闻 | 平台进度 | 当前机器进度；默认 = 当前机器进度
- machine tab = 本机节点横幅（NODE-A3F7 · TM-07 · 312/462 cells）+ 自己的 RDC 主线时间轴（10 节点）+ 路线图逻辑注脚 + 回到当前任务
- platform tab = 统计 5 卡 + 节点表 8 行 + 模板矩阵 12 格（TM-07 高亮）+ 聚合管线 5 步 + 6 维度进展 + 平台发布时间轴（6 节点）
- news tab = 业界新闻 7 卡 + 业界主流对比卡 5 张
- 抽出 Timeline 复用组件（主线/发布两条时间轴共用）；distributedData.js 数据源不变

**验证**：playwright 10/10 PASS、0 console 错误（研发 tab=0 回退干净 + v2 完好 display=flex；路线三 tab 顺序正确、machine 默认、platform 5+8+12+5+6、news 7+5、切回正常）。教训：JSX 单行渲染时 textContent 无 \\n，断言用 childNodes[0].textContent。截图 prog_{machine,platform,news}_tab.png。
"""

with io.open(LOG, "r", encoding="utf-8") as f:
    old = f.read()
marker = "## 三 tab 迁移：研发透镜回退 v2 → 路线透镜 v3（14:06）"
assert marker not in old, "already appended"
with io.open(LOG, "a", encoding="utf-8", newline="") as f:
    f.write(entry)
with io.open(LOG, "r", encoding="utf-8") as f:
    assert marker in f.read(), "write failed"
print("LOG OK")
