# -*- coding: utf-8 -*-
"""Q9-A 收尾前置探针：核对冻结产物与收尾链现状（只读，不写任何文件）。"""
import json, hashlib, os
from pathlib import Path

ROOT = Path(r"D:/AI2050/Ai2050-OpenOne")
out = []
def say(s):
    out.append(str(s))

def sha8(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]

# 1) 冻结 npz / result json
for tag in ["qwen", "glm"]:
    npz = ROOT / f"tests/gpt5/result/pat_q9a_{tag}.npz"
    say(f"npz[{tag}]: exists={npz.exists()} size={npz.stat().st_size if npz.exists() else 0} sha8={sha8(npz) if npz.exists() else '-'}")
    if npz.exists():
        import numpy as np
        z = np.load(npz)
        say(f"  keys={list(z.keys())} R{z['R'].shape} A{z['A'].shape} M{z['M'].shape}")

for d in sorted((ROOT / "results" / "2026-11-01").glob("*pat_q9a*")):
    say(f"result: {d.name} sha8={sha8(d)}")

res_path = ROOT / "results/2026-11-01/192208_pat_q9a_glm.json"
for name in ["191456_pat_q9a_qwen.json", "192208_pat_q9a_glm.json"]:
    p = ROOT / "results/2026-11-01" / name
    if p.exists():
        r = json.loads(p.read_text(encoding="utf-8"))
        say(f"{name} keys={list(r.keys())[:14]}")

# 2) verify summary
vp = ROOT / "tests/gpt5/result/pat_q9a_verify_summary.json"
say(f"verify_summary: exists={vp.exists()} sha8={sha8(vp) if vp.exists() else '-'}")
if vp.exists():
    v = json.loads(vp.read_text(encoding="utf-8"))
    say(f"  top keys={list(v.keys())}")

# 3) ledger
lp = ROOT / "research/gpt5/atlas/atlas_ledger.json"
led = json.loads(lp.read_text(encoding="utf-8"))
meas, negs = led["measurements"], led["negatives"]
say(f"ledger: n={len(meas)} neg={len(negs)} self_sha8={led.get('ledger_sha256_8')}")
q9m = [m for m in meas if "Q9-A" in json.dumps(m, ensure_ascii=False)]
q9n = [n for n in negs if "Q9-A" in json.dumps(n, ensure_ascii=False)]
say(f"Q9-A entries: meas={len(q9m)} neg={len(q9n)}")
for m in q9m:
    say("  meas: " + json.dumps(m, ensure_ascii=False)[:300])
for n in q9n:
    say("  neg: " + json.dumps(n, ensure_ascii=False)[:220])

# 4) rules
rp = ROOT / "data/rules/ledger.json"
rules = json.loads(rp.read_text(encoding="utf-8"))["rules"]
import collections
c = collections.Counter(r.get("status") for r in rules)
say(f"rules: n={len(rules)} status={dict(c)}")
newids = ["R-MET-37", "R-MET-38", "R-MET-39", "R-MTH-28"]
say("new rules: " + str({rid: sum(1 for r in rules if r.get("id") == rid) for rid in newids}))

# 5) MEMO marker
memo = ROOT / "research/gpt5/docs/AGI_GPT5_MEMO.md"
mb = memo.read_bytes()
say(f"MEMO: bytes={len(mb)} sha8={hashlib.sha256(mb).hexdigest()[:8]} marker_count={mb.decode('utf-8', 'replace').count('Phase Q9-A:')}")

# 6) wlogs（两位置）
for w in [ROOT / ".workbuddy/memory/2026-11-01.md",
          ROOT / ".workbuddy/memory/2026-11-02.md",
          Path(r"C:/Users/Admin/.workbuddy/user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal/memory/2026-11-01.md"),
          Path(r"C:/Users/Admin/.workbuddy/user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal/memory/2026-11-02.md")]:
    if w.exists():
        t = w.read_text(encoding="utf-8")
        say(f"wlog {w}: bytes={len(w.read_bytes())} q9a_marker={t.count('Q9-A')}")
    else:
        say(f"wlog {w}: MISSING")

# 7) MEMORY.md
mem = ROOT / ".workbuddy/memory/MEMORY.md"
mt = mem.read_text(encoding="utf-8")
say(f"MEMORY.md: bytes={len(mem.read_bytes())} has_5944a2d0={'5944a2d0' in mt} has_Q9A={'Q9-A' in mt}")

# 8) 既有结果里的关键数（供 verify 硬编码）
qres = ROOT / "results/2026-11-01/191456_pat_q9a_qwen.json"
gres = ROOT / "results/2026-11-01/192208_pat_q9a_glm.json"
if qres.exists():
    r = json.loads(qres.read_text(encoding="utf-8"))
    say("qwen res keys: " + str(list(r.keys())))
    for k in ["acc_resid", "acc_residB", "component", "transfer", "perm_null", "res"]:
        if k in r:
            say(f"  {k} = {json.dumps(r[k], ensure_ascii=False)[:400]}")
if gres.exists():
    r = json.loads(gres.read_text(encoding="utf-8"))
    say("glm res keys: " + str(list(r.keys())))
    for k in ["acc_resid", "acc_residB", "component", "transfer", "perm_null", "res"]:
        if k in r:
            say(f"  {k} = {json.dumps(r[k], ensure_ascii=False)[:400]}")

# 9) prereg 文档
pr = ROOT / "research/gpt5/docs/PAT_Q9A_PREREG_v1.md"
say(f"Q9A prereg: exists={pr.exists()} sha8={sha8(pr) if pr.exists() else '-'}")
prb = ROOT / "research/gpt5/docs/PAT_Q9B_PREREG_v1.md"
say(f"Q9B prereg: exists={prb.exists()}")

txt = "\n".join(out)
p = ROOT / "tests/gpt5_temp/q9a_state_probe.txt"
p.write_text(txt, encoding="utf-8", newline="\n")
print("written", p, len(txt))
