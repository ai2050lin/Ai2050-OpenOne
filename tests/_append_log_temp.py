# -*- coding: utf-8 -*-
"""Append a dated note to workspace memory log (file-based to avoid bash shim quoting bugs)."""
import os, io

LOG = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-05.md"
note = (
    "\n## 12:55 多条件分解方法论设计（答复用户提问，未执行实验）\n"
    "- 问题：上下文同时含语言/风格/逻辑/标点距离多维信息，如何分解到神经元。\n"
    "- 方案（五步管线）：条件网格最小对（因子化解混杂，≥1e4 句）→ 全层 hook 采集"
    "（残差流 h_l + 纯位置基线 p_l(pos) 扣 RoPE）→ 坐标级线性混合模型归因"
    "（h_i = mu + bL*1[L] + bS*1[S] + bR*1[R] + g*d + 交互 + eps，eta^2 效应热图，FDR）"
    "→ 方向级提取（类均值差 u^L/u^S/u^R，cos 正交性 + 子空间方差占比 + 分组 CV 探针上下界）"
    "→ 因果验证（消融/注入方向，行为切换率 + 附带损伤，随机方向对照）。\n"
    "- 判据：单条件 eta^2 高 = 专用坐标；多条件中等 = 混合编码；主效应 0 + 交互显著 = "
    "条件齿轮候选；方向 cos≈0 = 独立子空间，cos 高 = 共享子空间。\n"
    "- 衔接：复用 tests/deepseek 采集框架；第 5 步直接复用 Q06 C_steer 口径"
    "（steered 成功率 + 附带损伤）；P17 教训已内建（坐标级与方向级分开报告，不混比）。\n"
    "- 硬伤：模板生态效度（需自然语料回归检验推广）；内容残差混杂（实体置换家族缓解）；"
    "交互只做到二阶；激活级干预≠权重级证明（限界③）；标点距离与句法边界混杂（同词异位最小对分离）。\n"
    "- 状态：待用户确认后立项执行。\n"
)
os.makedirs(os.path.dirname(LOG), exist_ok=True)
existed = os.path.exists(LOG)
with io.open(LOG, "a", encoding="utf-8") as f:
    f.write(note)
print("APPENDED", existed, os.path.getsize(LOG))
