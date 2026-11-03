# -*- coding: utf-8 -*-
"""Q9-A 收尾记录：verify 全绿 + 勘误 + Q9-B 预注册，幂等 append 到今日 wlog。"""
import hashlib
from pathlib import Path

ROOT = Path(r"D:/AI2050/Ai2050-OpenOne")
W = ROOT / ".workbuddy/memory/2026-11-02.md"
MARK = "Q9-A 收尾链：独立磁盘复核全绿"
rep = []

b0 = W.read_bytes()
txt = b0.decode("utf-8")
if MARK in txt:
    rep.append(f"idempotent skip: marker already present (count={txt.count(MARK)})")
else:
    entry = f"""

## {MARK}（2026-11-02 18:35）

- 独立磁盘复核 `tests/gpt5_temp/q9a_disk_verify.py`：**111 检查 / 0 FAIL**——从冻结 npz（qwen 31a0ca8b / glm 0dafd223）逐位重放恒等式（A/B 两位置）、acc_resid 全剖面（78 次 LR 重拟合逐位一致）、acc_posB、perm_null l=0（default_rng(20261101) 同调用序列逐位重放）、P3 组件带 + 配对置换 p（rng 快进 3×200 置换后 zh→en 逐带重放）、NC 迁移曲线；对账 result 自声明 sha8（3f470a5b / 6df1b476 逐位复算）、账本链步（#468 → 5944a2d0 逐位复现）、规则 4 条在场、MEMO/wlog/MEMORY marker。
- 复核抓出 1 个真问题：**GLM v1 应为 AMBIGUOUS**（inc 0.33333 < 3×null_sd 0.33545，贴门差 0.0021），MEMO 散文误写两模型 STYLE_HAS_COMPUTED_INCREMENT → 勘误三连（MEMO 尾 + PAT_Q9A_FINDINGS + 账本 errata #12，新 head **38b4bd0b** 复算自洽；报告 `q9a_erratum_report.txt`）。另 2 个 FAIL 为复核脚本硬编码笔误（qwen v3 en2zh 应 53/60、glm l8 阈值过严：GLM l8=0.983、l20 才 1.000），修正后全绿。
- 并发写者痕迹两处（记录不改）：MEMORY.md 的 Q9-A 行被后续会话覆盖 → 幂等补回（前缀逐字节不变，7772→8170 B）；并发账本步链式复算 735ac5f8 ≠ f1671034（NOTE）。
- **Q9-B 预注册冻结**：`research/gpt5/docs/PAT_Q9B_PREREG_v1.md`——读写分离 patch 检验（W1 中段 MLP 语域轴干预充分性 / W2 消融必要性 / W3 注意力同轴对照 / W4 GLM 末段语言回写 / W5 GLM bf16 精度桥清算 8-bit 挂账）；N_PERM=200，MDE 翻转率 0.11、acc 下降 0.07；方向 disc 估计、conf 干预零重叠。
"""
    add = entry.encode("utf-8")
    W.write_bytes(b0 + add)
    b1 = W.read_bytes()
    ok = b1[: len(b0)] == b0 and MARK in b1.decode("utf-8")
    rep.append(f"appended: bytes {len(b0)}->{len(b1)} sha8 {hashlib.sha256(b0).hexdigest()[:8]}->"
               f"{hashlib.sha256(b1).hexdigest()[:8]} prefix_ok={b1[:len(b0)] == b0} ok={ok}")

p = ROOT / "tests/gpt5_temp/q9a_wlog_closeout.txt"
p.write_text("\n".join(rep), encoding="utf-8", newline="\n")
print("written", p)
