# -*- coding: utf-8 -*-
"""R8 探针 2：账本归属取证 + C 表目标可达性 + 环境。"""
import os, re, json, hashlib, collections

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek_temp\_seal_r8_probe2.txt")
r = []
def add(s): r.append(s)
def sha8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8] if os.path.exists(p) else "MISSING"

LG = os.path.join(ROOT, r"research\gpt5\atlas\atlas_ledger.json")
SPEC = os.path.join(ROOT, r"research\gpt5\atlas\ATLAS_LEDGER_SPEC.md")
PL = os.path.join(ROOT, r"tests\glm5\result\rdc_query_construction_20260913\phase3103\omega_p101_formula_audit\proposition_ledger.json")

add("=== A) atlas_ledger.json 归属取证 ===")
lg = json.loads(open(LG, "rb").read().decode("utf-8-sig"))
add("  format=%r version=%r model=%r" % (lg.get("format"), lg.get("version"), lg.get("model")))
add("  ledger_sha256_8 (记录值) = %r" % lg.get("ledger_sha256_8"))
add("  schema_version=%r  protocol_family=%r" % (lg.get("schema_version"), lg.get("protocol_family")))
ms = lg["measurements"]
add("  n=%d" % len(ms))
ph = collections.Counter(m.get("phase") for m in ms)
add("  phase 分布 = %s" % dict(sorted(ph.items(), key=lambda x: (x[0] is None, x[0]))))
nm = collections.Counter()
for m in ms:
    nm[m.get("name","")[:6]] += 1
add("  name 前缀 top = %s" % nm.most_common(12))
ev = collections.Counter(m.get("evidence_level") for m in ms)
add("  evidence_level = %s" % dict(ev))
add("  has meas_id = %d / has phase = %d / both = %d"
    % (sum(1 for m in ms if "meas_id" in m), sum(1 for m in ms if "phase" in m),
       sum(1 for m in ms if "meas_id" in m and "phase" in m)))
# 是否含明显属于 G 线（gpt5）的条目？看 model_scope
sc = collections.Counter()
for m in ms:
    s = str(m.get("model_scope",""))
    for k in ["qwen3-4b","qwen3-14b","glm4-9b","gemma","ds7b","llama"]:
        if k.lower() in s.lower(): sc[k]+=1
add("  model_scope 关键词计数 = %s" % dict(sc))

add("")
add("=== B) ATLAS_LEDGER_SPEC.md 归属声明（前 45 行）===")
if os.path.exists(SPEC):
    t = open(SPEC, "rb").read().decode("utf-8-sig","replace").replace("\r\n","\n").split("\n")
    for i,l in enumerate(t[:45],1):
        add("  %03d|%s" % (i,l[:150]))
else:
    add("  MISSING")

add("")
add("=== C) proposition_ledger.json ===")
add("  path=%s exists=%s sha8=%s" % (PL, os.path.exists(PL), sha8(PL)))
if os.path.exists(PL):
    d = json.loads(open(PL, "rb").read().decode("utf-8-sig"))
    add("  top keys=%s" % list(d.keys()))
    for k in d:
        if re.search(r"(sha|hash|count|grade|ledger)", k, re.I):
            add("  [%s] = %s" % (k, json.dumps(d[k], ensure_ascii=False)[:300]))

add("")
add("=== D) metric_dict.json（I1 KPI 定义）===")
MD = os.path.join(ROOT, r"research\deepseek\atlas\metric_dict.json")
d = json.loads(open(MD, "rb").read().decode("utf-8-sig"))
add("  keys=%s" % list(d.keys()))
add(json.dumps(d, ensure_ascii=False)[:1200])

add("")
add("=== E) 环境：GPU / 模型 ===")
try:
    import subprocess
    o = subprocess.run([r"C:\Windows\System32\nvidia-smi.exe","--query-gpu=name,memory.total,memory.used",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    add("  nvidia-smi rc=%d out=%r err=%r" % (o.returncode, o.stdout.strip()[:400], o.stderr.strip()[:200]))
except Exception as e:
    add("  nvidia-smi EXC %s" % e)
try:
    import torch
    add("  torch=%s cuda_avail=%s ndev=%s" % (torch.__version__, torch.cuda.is_available(),
        torch.cuda.device_count() if torch.cuda.is_available() else 0))
except Exception as e:
    add("  torch EXC %s" % e)
for sub in [r"models\hf", r"models"]:
    p = os.path.join(ROOT, sub)
    if os.path.isdir(p):
        add("  DIR %s: %s" % (sub, sorted(os.listdir(p))[:20]))

open(OUT, "w", encoding="utf-8").write("\n".join(r))
print("WROTE %s (%d lines)" % (OUT, len(r)))
