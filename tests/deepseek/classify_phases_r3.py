# -*- coding: utf-8 -*-
"""R3 续研：把 I1（验收函数）操作化为一个可计算的数。
启发式分类：一个 Phase 若报告了「共享指标的小数变化」（同一指标名 X 的 A -> B），
记为 advance-proxy；否则记为 catalog-proxy。
所有数字现场从 MEMO 解析，禁手工转录。"""
import os, re, json, hashlib

root = r"D:\AI2050\Ai2050-OpenOne"
p = os.path.join(root, "research/gpt5/docs/AGI_GPT5_MEMO.md")
b = open(p, "rb").read()
t = b.decode("utf-8-sig")
lines = t.split("\n")

# ---- 切分 Phase 节（## 开头且含 "Phase"）----
idx = [i for i, l in enumerate(lines) if l.startswith("## ") and "Phase" in l]
secs = []
for j, i in enumerate(idx):
    end = idx[j + 1] if j + 1 < len(idx) else len(lines)
    secs.append((lines[i], "\n".join(lines[i:end])))

DEC = r"\d+\.\d+"
METRIC = r"(?:误差|error|rel-?L2|AUC|cos|余弦|相关系数|rho)"
DIRW = r"(?:→|->|降到|降至|下降|降低|减少|改善|恶化|上升|增至|提升|拉近)"
# advance-proxy：指标名 + 小数A + 方向 + 小数B，同一句内
adv_re = re.compile(METRIC + r"[^。\n]{0,45}?(" + DEC + r")[^。\n]{0,14}?" + DIRW + r"[^。\n]{0,14}?(" + DEC + r")")
# 窄词汇变体（去掉 rho / 相关系数）：用于报告词汇敏感区间
METRIC_N = r"(?:误差|error|rel-?L2|AUC|cos|余弦)"
adv_nar = re.compile(METRIC_N + r"[^。\n]{0,45}?(" + DEC + r")[^。\n]{0,14}?" + DIRW + r"[^。\n]{0,14}?(" + DEC + r")")
# 更宽松：任意小数 -> 小数
arrow_re = re.compile(r"(" + DEC + r")\s*(?:→|->)\s*(" + DEC + r")")
verdict_re = re.compile(r"(?:判决\s*=|verdict\s*[=:])")
err_re = re.compile(METRIC)

n = len(secs)
adv = []
advn = []
loose = []
verd = []
err_any = []
for head, body in secs:
    if adv_re.search(body):
        adv.append(head)
    if adv_nar.search(body):
        advn.append(head)
    if arrow_re.search(body):
        loose.append(head)
    if verdict_re.search(body):
        verd.append(head)
    if err_re.search(body):
        err_any.append(head)

# ---- 判决标签统计（复用率）----
vlist = re.findall(r"(?:judgement|verdict)\s*[=:]\s*([A-Za-z0-9_|]+)", t)
uniq = set()
for s in vlist:
    for part in s.split("|"):
        part = part.strip()
        if part:
            uniq.add(part)

# ---- 逐 Phase 小节长度 ----
lens = sorted([len(body.split("\n")) for _, body in secs])
med = lens[len(lens) // 2] if lens else 0

out = {
    "source": {"path": "research/gpt5/docs/AGI_GPT5_MEMO.md", "bytes": len(b),
               "lines": b.count(b"\n") + 1, "sha8": hashlib.sha256(b).hexdigest()[:8]},
    "phases": {
        "n_sections": n,
        "n_with_metric_delta_strict": len(adv),
        "n_with_metric_delta_narrow": len(advn),
        "n_with_arrow_delta_loose": len(loose),
        "n_with_verdict": len(verd),
        "n_mention_any_error_metric": len(err_any),
        "advance_proxy_share_strict": round(len(adv) / n, 4) if n else None,
        "catalog_proxy_strict": n - len(adv),
        "median_section_lines": med,
    },
    "verdict_vocab_inline": {"uses": len(vlist), "unique": len(uniq),
                             "reuse": round(len(vlist) / len(uniq), 3) if uniq else None},
    "samples": {
        "advance_strict": [h[:120] for h in adv[:12]],
        "loose_only": [h[:120] for h in loose if h not in adv][:8],
    },
}

op = os.path.join(root, "tests/deepseek/result/phase_classify_r3.json")
with open(op, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

# 人类可读摘要
rep = []
rep.append("=== Phase 分类（I1 验收函数操作化，启发式）===")
rep.append("源: %s  %d B  %d 行  sha8=%s" % (
    out["source"]["path"], out["source"]["bytes"], out["source"]["lines"], out["source"]["sha8"]))
rep.append("Phase 节数 N = %d" % n)
rep.append("  报告「指标名 + 小数→小数」严格变化 = %d  (%.2f%%)" % (
    len(adv), 100.0 * len(adv) / n if n else 0))
rep.append("    其中窄词汇（无 rho/相关系数）      = %d  (%.2f%%)" % (
    len(advn), 100.0 * len(advn) / n if n else 0))
rep.append("  仅含「小数→小数」宽松变化     = %d" % len(loose))
rep.append("  含「判决=/verdict=」           = %d" % len(verd))
rep.append("  提及任一误差/相似度指标        = %d" % len(err_any))
rep.append("  中位小节行数                   = %d" % med)
rep.append("判决标签: uses=%d unique=%d reuse=%.3f" % (
    len(vlist), len(uniq), (len(vlist) / len(uniq)) if uniq else 0))
rep.append("")
rep.append("--- 严格 advance 示例 ---")
for h in adv[:10]:
    rep.append("  " + h)
rep.append("")
rep.append("--- 仅宽松（无严格共享指标 delta）示例 ---")
for h in [h for h in loose if h not in adv][:8]:
    rep.append("  " + h)
open(os.path.join(root, "tests/deepseek/result/phase_classify_r3.txt"), "w",
     encoding="utf-8").write("\n".join(rep))
print("OK phases=%d adv=%d loose=%d verdict=%d" % (n, len(adv), len(loose), len(verd)))
