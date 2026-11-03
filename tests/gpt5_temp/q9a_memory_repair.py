# -*- coding: utf-8 -*-
"""Q9-A closeout 幂等补检：MEMORY.md 的 Q9-A 索引行被后续会话重写丢失，幂等补回（不动他人内容）。"""
import json, hashlib
from pathlib import Path

ROOT = Path(r"D:/AI2050/Ai2050-OpenOne")
MEM = ROOT / ".workbuddy/memory/MEMORY.md"
rep = []

raw = MEM.read_bytes()
before_bytes, before_sha8 = len(raw), hashlib.sha256(raw).hexdigest()[:8]
txt = raw.decode("utf-8")
n_before = txt.count("Q9-A")
rep.append(f"before: bytes={before_bytes} sha8={before_sha8} q9a_count={n_before}")

if n_before == 0:
    line = ("- **Q9-A 已收官（G 线，2026-11-01）**：中段 MLP 主写语域（raw |a|:|m| 0.11-0.39 / normed 0.33-0.74）；"
            "hs[0] 带 2/3、前 8 层补完计算增量；语域轴 l≈8 跨语言抽象化（zh↔en 迁移 1.0）；"
            "GLM 末段注意力反超 + en→zh 迁移塌回 0.5（8-bit 挂账）。"
            "ledger #468 head 5944a2d0；规则 +R-MET-37/38/39 +R-MTH-28。读写分离假设交 Q9-B patch。")
    new_txt = txt.rstrip("\n") + "\n" + line + "\n"
    MEM.write_bytes(new_txt.encode("utf-8"))
    raw2 = MEM.read_bytes()
    n_after = raw2.decode("utf-8").count("Q9-A")
    rep.append(f"after: bytes={len(raw2)} sha8={hashlib.sha256(raw2).hexdigest()[:8]} q9a_count={n_after}")
    ok = (raw2.startswith(raw) and n_after == 1)
    rep.append(f"prefix_unchanged={raw2.startswith(raw)} repaired={ok}")
else:
    rep.append(f"already present (count={n_before}) — skip（幂等路径）")

p = ROOT / "tests/gpt5_temp/q9a_memory_repair.txt"
p.write_text("\n".join(rep), encoding="utf-8", newline="\n")
print("written", p)
