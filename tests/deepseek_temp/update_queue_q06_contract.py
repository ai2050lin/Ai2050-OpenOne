# -*- coding: utf-8 -*-
"""Q06 C_steer 合同冻结脚本：
1) 把误写到 research/deepseek/result/ 的预注册设计搬到 tests/deepseek/result/（Q05 惯例位置）
2) 注入 frozen_at（当前时间）
3) 计算 sha256 -> sha8
4) 更新 phase_queue_v1.json（Q06 status=contract_frozen + prereg 字段）
5) 写验证报告
"""
import hashlib
import io
import json
import os
from datetime import datetime

WS = r"D:\AI2050\Ai2050-OpenOne"
SRC = os.path.join(WS, "research", "deepseek", "result", "q06_prereg_design_v1.json")
DST = os.path.join(WS, "tests", "deepseek", "result", "q06_prereg_design_v1.json")
QUEUE = os.path.join(WS, "research", "deepseek", "atlas", "phase_queue_v1.json")
REPORT = os.path.join(WS, ".workbuddy", "tmp_q06_freeze_report.txt")

lines = []

# 1) move if needed
if os.path.isfile(SRC) and not os.path.isfile(DST):
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with io.open(SRC, "r", encoding="utf-8") as f:
        payload = f.read()
    with io.open(DST, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)
    os.remove(SRC)
    lines.append("moved: research/deepseek/result -> tests/deepseek/result")
elif os.path.isfile(DST):
    lines.append("dst already exists, skip move")
else:
    raise SystemExit("design file missing at both locations")

# 2) inject frozen_at
with io.open(DST, "r", encoding="utf-8") as f:
    design = json.load(f)
now = datetime.now().strftime("%Y-%m-%d %H:%M")
design["frozen_at"] = now
with io.open(DST, "w", encoding="utf-8", newline="\n") as f:
    json.dump(design, f, ensure_ascii=False, indent=2)
    f.write("\n")

# 3) sha
raw = io.open(DST, "rb").read()
sha_full = hashlib.sha256(raw).hexdigest()
sha8 = sha_full[:8]
lines.append("frozen_at=%s" % now)
lines.append("sha256=%s" % sha_full)
lines.append("sha8=%s" % sha8)
lines.append("size=%d" % len(raw))

# 4) queue update
with io.open(QUEUE, "r", encoding="utf-8") as f:
    queue = json.load(f)
q6 = next(e for e in queue["queue"] if e.get("id") == "Q06")
assert q6.get("status") == "pending", "Q06 status unexpected: %r" % q6.get("status")
q6["status"] = "contract_frozen"
q6["contract_frozen_at"] = now
q6["prereg_design_sha"] = sha8
q6["prereg_ref"] = "tests/deepseek/result/q06_prereg_design_v1.json"
q6["note"] = "预注册合同已冻结（frozen before any Q06 observation incl. SMOKE）；执行 Phase 须以 design_sha drift 断言对齐本哈希，并先加载 rdc-main-axis-probe / rdc-phase-closeout 技能。"
queue["status_updated_at"] = now
queue["status_updated_by"] = "tests/deepseek_temp/update_queue_q06_contract.py"
with io.open(QUEUE, "w", encoding="utf-8", newline="\n") as f:
    json.dump(queue, f, ensure_ascii=False, indent=1)
    f.write("\n")
lines.append("queue: Q06 -> contract_frozen, prereg_design_sha=%s" % sha8)

# 5) report
with io.open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("FREEZE_DONE")
