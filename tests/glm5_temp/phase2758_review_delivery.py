"""Append the completed review, retaining the previous memo byte-for-byte."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/rdc_unified_structure_review_20260923"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
old = MEMO.read_bytes()
phase_ids = re.findall(r"^## Phase (\d+):", old.decode("utf-8"), re.M)
assert int(phase_ids[-1]) == 2757, "Reconcile the latest phase before appending."
assert not (OUT / "append_receipt.json").exists()
body = (OUT / "report_body.md").read_text(encoding="utf-8")
assert not any(ord(c) < 32 and c not in "\n\r\t" for c in body)
assert body.count("$$") % 2 == 0
stamp = datetime.now().astimezone()
report = (
    f"## Phase 2758: 统一数学结构猜想、六大拼图证据复核与三图谱实施路线 "
    f"[{stamp:%Y-%m-%d %H:%M}]\n\n{body.rstrip()}\n"
)
(OUT / "phase_report.md").write_text(report, encoding="utf-8")
prefix = "\n" if old.endswith(b"\n") else "\n\n"
addition = (prefix + report).encode("utf-8")
assert MEMO.read_bytes() == old
with MEMO.open("ab") as f:
    f.write(addition)
after = MEMO.read_bytes()
assert after == old + addition
sha = lambda b: hashlib.sha256(b).hexdigest()
receipt = {
    "phase": 2758, "created_local": stamp.isoformat(),
    "memo_line": old.count(b"\n") + prefix.count("\n") + 1,
    "append_only_verified": True, "before_sha256": sha(old),
    "after_sha256": sha(after), "report_sha256": sha(report.encode("utf-8")),
    "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p.read_bytes())}
              for p in sorted(OUT.iterdir()) if p.is_file()],
}
(OUT / "append_receipt.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in receipt.items() if k != "files"},
                 ensure_ascii=False, indent=2))
