"""Deliver the completed six-puzzle audit and append a byte-preserving memo entry."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/rdc_six_puzzles_audit_20260923"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    receipt_path = OUT / "delivery_receipt.json"
    if receipt_path.exists():
        raise SystemExit("Delivery receipt already exists; refusing duplicate append.")
    body = (OUT / "report_body.md").read_text(encoding="utf-8")
    claims = []
    for line in body.splitlines():
        if re.match(r"^\| \d+ ", line):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            num, title = cells[0].split(" ", 1)
            assert len(cells) == 3
            claims.append({"id": int(num), "claim": title,
                           "assessment": cells[1], "reason": cells[2]})
    assert [row["id"] for row in claims] == list(range(1, 31))
    assert body.count("$$") % 2 == 0
    assert not any(ord(c) < 32 and c not in "\n\r\t" for c in body)
    before = MEMO.read_bytes()
    old_text = before.decode("utf-8-sig")
    phases = re.findall(r"^## Phase (\d+):", old_text, re.MULTILINE)
    assert phases[-1] == "2758", phases[-1]
    stamp = dt.datetime.now().astimezone()
    heading = ("## Phase 2759: 六大拼图群逐项审查、原生读出接口与可读性复算 "
               f"[{stamp:%Y-%m-%d %H:%M}]")
    report = heading + "\n\n" + body.rstrip() + "\n"
    (OUT / "phase_report.md").write_text(report, encoding="utf-8", newline="\n")
    (OUT / "claim_ledger.json").write_text(
        json.dumps({"phase": 2759, "claims": claims}, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8")
    separator = b"\n\n" if not before.endswith(b"\n") else b"\n"
    addition = separator + report.encode("utf-8")
    with MEMO.open("ab") as stream:
        stream.write(addition)
    after = MEMO.read_bytes()
    assert after == before + addition
    line_no = after[:after.index(heading.encode("utf-8"))].count(b"\n") + 1
    evidence_files = sorted(p for p in OUT.iterdir() if p.is_file())
    evidence_files += [ROOT / "tests/glm5/phase2759_six_puzzles_audit.py",
                       ROOT / "tests/glm5_temp/phase2759_interface_checks.py",
                       Path(__file__).resolve()]
    receipt = {
        "phase": 2759, "timestamp": stamp.isoformat(),
        "memo": str(MEMO), "memo_line": line_no,
        "before_sha256": sha(before), "after_sha256": sha(after),
        "before_bytes": len(before), "after_bytes": len(after),
        "historical_bytes_preserved": True, "claims": len(claims),
        "new_gpu_runs": 0,
        "artifacts": [{"path": str(p.relative_to(ROOT)),
                       "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                      for p in evidence_files]
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)
                            + "\n", encoding="utf-8")
    print(json.dumps({"phase": 2759, "claims": len(claims),
                      "memo_line": line_no, "timestamp": stamp.isoformat(),
                      "historical_bytes_preserved": True,
                      "report": str(OUT / "phase_report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
