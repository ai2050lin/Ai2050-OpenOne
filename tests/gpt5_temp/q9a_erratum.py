# -*- coding: utf-8 -*-
"""Q9-A 勘误三连：MEMO 尾 append + findings 文档 append + atlas 账本 errata 条目。
全部 bytes 级写入，前缀不变断言；账本先 .bak 并按既有链式配方重算 head。"""
import json, hashlib, shutil
from pathlib import Path

ROOT = Path(r"D:/AI2050/Ai2050-OpenOne")
rep = []

stamp = "2026-11-02 18:40"

ERRATUM = f"""

## 勘误: Q9-A GLM v1 判决与迁移曲线精化（独立磁盘复核发现）[{stamp}]

独立磁盘复核（`tests/gpt5_temp/q9a_disk_verify.py`，109 检查项）逐位重放 Q9-A 冻结 npz 与 result 后，发现两处散文超claim；冻结产物本身无误，更正如下：

1. **GLM v1 判决应为 AMBIGUOUS 而非 STYLE_HAS_COMPUTED_INCREMENT**：`192208_pat_q9a_glm.json` payload `v1.verdict=AMBIGUOUS`——max_increment=0.33333 < 3×null_sd=0.33545（差 0.0021，预注册 V1 门）。Qwen 门内通过（3×null_sd=0.30416 < 0.33333）。「hs[0] 带 2/3 + 计算增量早期补齐」对两模型方向上仍成立（GLM：LR 曲线 l=2 已 0.983、NC 迁移 l=4 起 0.633 → l=8 0.983），但按预注册纪律，**GLM 的 v1 正式标签记 AMBIGUOUS（贴门）**，其对「早期计算增量」的支持降级为方向性证据。
2. **「语域轴 l≈8 起两模型迁移 1.000」精化**：Qwen 精确成立（l=8 双向 1.000）；GLM l=8 为 0.983/0.983，l=20 起才双 1.000（l=12–16 为 0.967–0.983）。结论「语域轴中段跨语言抽象化」不变，层位表述改为「qwen l≈8 / glm l≈8(0.983)–20(1.000)」。

复核 3 个 FAIL 中其余 2 个为复核脚本硬编码笔误（qwen v3 的 en2zh 应为 53/60、glm l8 阈值过严），修正后全绿。账本本条 erratum 后 head 按既有链式配方重算；并发写者此前步骤的链式复算差异（735ac5f8 vs f1671034）已记录、未回改。
"""


def append_bytes(path: Path, text: str, label: str):
    b0 = path.read_bytes()
    add = text.replace("\n", "\r\n").encode("utf-8") if b"\r\n" in b0[-200:] else text.encode("utf-8")
    path.write_bytes(b0 + add)
    b1 = path.read_bytes()
    ok = b1[: len(b0)] == b0 and len(b1) - len(b0) == len(add)
    rep.append(f"{label}: bytes {len(b0)}->{len(b1)} sha8 {hashlib.sha256(b0).hexdigest()[:8]}->"
               f"{hashlib.sha256(b1).hexdigest()[:8]} prefix_ok={b1[:len(b0)] == b0} appended_ok={ok}")
    return ok


ok1 = append_bytes(ROOT / "research/gpt5/docs/AGI_GPT5_MEMO.md", ERRATUM, "MEMO")
ok2 = append_bytes(ROOT / "research/gpt5/docs/PAT_Q9A_FINDINGS_20261101.md", ERRATUM, "FINDINGS")

# 账本 errata 条目
lp = ROOT / "research/gpt5/atlas/atlas_ledger.json"
shutil.copy2(lp, str(lp) + ".bak_q9a_erratum")
led = json.loads(lp.read_text(encoding="utf-8"))
prev_head = led.get("ledger_sha256_8")
n_err_before = len(led["errata_ledger"])
entry = {"date": "2026-11-02", "kind": "erratum", "phase": "Q9-A", "name": "pat_q9a_v1_glm_relabel",
         "line": "G",
         "note": ("Disk verify (tests/gpt5_temp/q9a_disk_verify.py, 109 checks) relabels GLM v1 to "
                  "AMBIGUOUS: max_increment 0.33333 < 3*null_sd 0.33545 (margin 0.0021, frozen V1 gate); "
                  "qwen v1 STYLE_HAS_COMPUTED_INCREMENT stands (3*null_sd 0.30416). Transfer-curve precision: "
                  "glm l8 0.983/0.983, first double-1.000 at l20; qwen l8 1.000/1.000. Frozen artifacts "
                  "unchanged; prose-only overclaim corrected in MEMO erratum.")}
led["errata_ledger"].append(entry)
led["ledger_sha256_8"] = prev_head
new_head = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1).encode("utf-8")).hexdigest()[:8]
led["ledger_sha256_8"] = new_head
lp.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")
# 回读复算
led2 = json.loads(lp.read_text(encoding="utf-8"))
led2["ledger_sha256_8"] = prev_head
rec = hashlib.sha256(json.dumps(led2, ensure_ascii=False, indent=1).encode("utf-8")).hexdigest()[:8]
b1 = lp.read_bytes()
rep.append(f"LEDGER: errata {n_err_before}->{len(led2['errata_ledger'])} head {prev_head}->{new_head} "
           f"recompute_match={rec == new_head} bytes={len(b1)}")
ok3 = rec == new_head

rep.append(f"ALL_OK={ok1 and ok2 and ok3}")
p = ROOT / "tests/gpt5_temp/q9a_erratum_report.txt"
p.write_text("\n".join(rep), encoding="utf-8", newline="\n")
print("written", p, "ALL_OK=", ok1 and ok2 and ok3)
