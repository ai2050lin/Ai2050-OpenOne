# -*- coding: utf-8 -*-
"""Phase 3278 closeout: five-write (ledger #430 / wlog+mirror / G-line MEMO /
MEMORY / daily log) + impact-preview addendum. Idempotent. Verdict honest
BLOCKED (G-C top1 tolerance FAIL, quantization perturbation real).
Templates use __PLACEHOLDER__ replacement (bare-% lesson)."""
import hashlib
import json
import os
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
WLOG = os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
PV = os.path.join(ROOT, "research", "gpt5", "docs", "PENDING_ITEMS_IMPACT_PREVIEW.md")
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
DAILY = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-21.md")
PREV_POST = "76783ddd"
PREV_N = 435
TODAY = datetime.date.today().isoformat()

_led0 = json.load(open(LED, encoding="utf-8"))
assert _led0["ledger_sha256_8"] == PREV_POST and len(_led0["measurements"]) == PREV_N

# ---------------- 1. ledger append #430 (idempotent) ----------------
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
if any(str(r.get("phase")) == "3278" for r in ms):
    new_post = led["ledger_sha256_8"]
    print("ledger: 3278 present, head", new_post, "n", len(ms))
else:
    rec = {
        "n": len(ms) + 1,
        "phase": "3278",
        "name": "g5h41_appx1_exec",
        "line": "G",
        "date": TODAY,
        "verdict": "appx1_blocked",
        "res_sha8": "588d36c9",
        "note": ("APPX-1 execution (owner approved nf4 baseline): Arm a "
                 "zero-GPU anchors reproduced bitwise-grade (26.641/2, "
                 "71.796/0, 70.235/0) and the two shared dims LOCALIZED "
                 "(cos 0.8938 R_logic row0 x K_ent rows0+1; cos 0.7149 "
                 "R_logic rows5/3/7 x K_ent row2; last-layer emergence "
                 "37-40; vocab: discourse-connective family / "
                 "truth-judgment family). Arm b nf4: G-B fidelity PASS "
                 "(rowspace 0.0001 deg); G-C conservation top1 FAIL "
                 "30.791 vs 26.641+3.0 tolerance (diff 4.15 deg; "
                 "perturbation carried by R_logic side; eff=2 and shared "
                 "cos diff <=0.035 hold). Honest BLOCKED: localization + "
                 "explanation complete; quantitative nf4 conservation "
                 "below preregistered strength. fp16-device first run "
                 "(res fa1858f9) archived as *_fp16device.*; bf16 rerun "
                 "confirms perturbation is quantization-real, not dtype "
                 "device artifact."),
        "prev_post": PREV_POST,
    }
    led["measurements"] = ms + [rec]
    probe = json.loads(json.dumps(led, ensure_ascii=False))
    probe["ledger_sha256_8"] = PREV_POST
    new_post = hashlib.sha256(json.dumps(probe, ensure_ascii=False, indent=1)
                              .encode("utf-8")).hexdigest()[:8]
    led["ledger_sha256_8"] = new_post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    print("ledger #430 appended, post", new_post, "n", len(led["measurements"]))

# ---------------- 2. wlog + mirror ----------------
wtxt = ("## Phase 3278 (G5-H41): APPX-1 execution (nf4 baseline)\n\n"
        "- **3278（G5-H41）appx1_blocked**（owner 批准 nf4 基线，G-A/G-B 过、"
        "G-C top1 容差 FAIL，诚实 BLOCKED）：**Arm a 零 GPU 定位**（封存锚逐值"
        "复现 26.641/2、71.796/0、70.235/0，tol 0.05°）——eff_ge05=2 的两个共享维"
        "定位：**dim0 cos=0.8938**（R_logic row0 主对比方向 × K_ent rows0+1，"
        "层 37–40 末段新生；词表=转折连词族：但由于/但他/But/其次是）；"
        "**dim1 cos=0.7149**（R_logic rows5/3/7 × K_ent row2；词表=真值判断族："
        "错/True/False/判断）。K_ent row2 真值成分可溯源 3157 面板极性臂"
        "（isa/hasa 否定结构）。**Arm b nf4**（Qwen3-14B-bnb-nf4，866 行重采，"
        "bf16 口径对齐原版）：G-B 保真 PASS（lm_head fp16 未量化，K_readout "
        "rowspace 角 0.0001°；K_ent_fp×K_read_nf4=70.234≈锚）；G-C 守恒 top1 "
        "FAIL（30.791 vs 26.641+3.0，差 4.15°；扰动 R_logic 侧承载：R_nf4×"
        "K_ent_fp=31.44；eff=2 与共享 cos diff≤0.035 仍守恒）。**诚实日志**："
        "首跑 fp16 装置口径（res fa1858f9）存档 *_fp16device.*；bf16 重采确认"
        "扰动为量化真实信号非口径伪影；V×V Gram 172 GiB 内存爆→D 空间 GPU "
        "fp32 eigh（rowspace 0.0001° 等价验证）；freeze 分文件（3263 教训）"
        "与 torch 负步长切片各修一次。freeze 7b5554a7 / res 588d36c9 / ledger "
        "#430 post '__POST__'（n=436）。APPX-1 状态：定位+解释完成入册，"
        "nf4 定量守恒未达预注册强度——处置留 owner（按定位/解释关闭或留待"
        "全精度硬件复测）。\n")
if "G5-H41" not in open(WLOG, encoding="utf-8").read():
    with open(WLOG, "a", encoding="utf-8", newline="") as f:
        f.write(wtxt.replace("__POST__", new_post))
    print("wlog appended")

mir = os.path.join(ROOT, "tests", "gpt5_temp", "WORKLOG_gpt5_mirror.md")
mir_txt = open(mir, encoding="utf-8").read() if os.path.exists(mir) else ""
if "G5-H41" not in mir_txt:
    with open(mir, "a", encoding="utf-8", newline="") as f:
        f.write(wtxt.replace("__POST__", new_post))
    print("mirror appended")

# ---------------- 3. G-line MEMO ----------------
mtxt = ("\n### 3278（G5-H41）APPX-1 execution（appx1_blocked，owner 批准 nf4 "
        "基线）\nArm a 零 GPU 定位（封存锚 26.641/2、71.796/0、70.235/0 逐值"
        "复现）：两个共享维——dim0 cos=0.8938（R_logic row0×K_ent rows0+1，层 "
        "37–40 末段新生，词表=转折连词族）；dim1 cos=0.7149（R_logic rows5/3/7×"
        "K_ent row2，词表=真值判断族；K_ent row2 可溯源 3157 面板极性臂）。"
        "Arm b nf4（bf16 口径，866 行重采）：G-B 保真 PASS（rowspace 0.0001°）；"
        "G-C top1 FAIL（30.791 vs 26.641+3.0，差 4.15°，R_logic 侧承载扰动；"
        "eff=2 与 cos diff≤0.035 守恒）。诚实 BLOCKED：定位+解释完成，nf4 定量"
        "守恒未达预注册强度。fp16 首跑（fa1858f9）存档。res 588d36c9 / ledger "
        "#430 post __POST__（n=436）。\n")
if "G5-H41" not in open(MEMO, encoding="utf-8").read():
    with open(MEMO, "a", encoding="utf-8", newline="") as f:
        f.write(mtxt.replace("__POST__", new_post))
    print("MEMO appended")

# ---------------- 4. MEMORY (idempotent, <=3000) ----------------
mem = open(MEMORY, encoding="utf-8").read()
if "- **3278**" not in mem:
    if len(mem) + 320 > 3000:
        lines = mem.split("\n")
        out = []
        for l in lines:
            if l.startswith("- **3276**"):
                l = ("- **3276**：APPX-2/3 执行相（协议 ce73984d）：S 族×K_readout "
                     "separable 跨 6 模型守恒；K3 死线 3/3 触发、K2 3/3 不触发。"
                     "res 8d5bba36。#428。")
            out.append(l)
        mem = "\n".join(out)
    mem = mem.rstrip("\n") + ("\n- **3278**：APPX-1 执行（nf4 基线，owner 批准）："
                              "Arm a 定位 2 共享维（0.894 R_logic r0×K_ent r0+1 层"
                              "37–40 新生=转折连词族；0.715 r5/3/7×r2=真值族，可溯"
                              "源面板极性臂）；nf4 G-C top1 FAIL（30.79 vs 26.64+3，"
                              "量化扰动 R_logic 侧）→ 诚实 BLOCKED；fp16 首跑存档。"
                              "res 588d36c9。#430。\n")
    assert len(mem) <= 3000, len(mem)
    with open(MEMORY, "w", encoding="utf-8", newline="") as f:
        f.write(mem)
    print("MEMORY updated, len", len(mem))
else:
    print("MEMORY already has 3278")

# ---------------- 5. daily log ----------------
dtxt = ("- 3278 G5-H41 APPX-1 执行相收官（appx1_blocked，G-A/G-B 过 G-C top1 "
        "FAIL，owner 批准 nf4 基线）：Arm a 零 GPU 封存锚逐值复现（26.641/2、"
        "71.796/0、70.235/0）+ 两共享维定位（dim0 cos 0.8938 R_logic row0×K_ent "
        "rows0+1，层 37–40 末段新生，词表转折连词族；dim1 cos 0.7149 rows5/3/7×"
        "row2，真值判断族，可溯源 3157 面板极性臂）；Arm b nf4 bf16 重采 866 行"
        "——G-B 保真 PASS（rowspace 0.0001°）、G-C 守恒 top1 FAIL（30.791 vs "
        "26.641+3.0，差 4.15°，量化扰动 R_logic 侧承载，eff=2 与 cos diff≤0.035 "
        "守恒）。诚实日志：fp16 口径首跑（fa1858f9）存档 *_fp16device.*；bf16 "
        "重跑证实扰动为量化真实信号；V×V Gram 172GiB→D 空间 GPU fp32 eigh 等价"
        "修正；freeze 分文件+torch 负步长切片各修一次。freeze 7b5554a7 / res "
        "588d36c9 / ledger #430 post __POST__（n=436）。APPX-1 处置留 owner。\n")
dl = open(DAILY, encoding="utf-8").read() if os.path.exists(DAILY) else ""
if "3278 G5-H41" not in dl:
    with open(DAILY, "a", encoding="utf-8", newline="") as f:
        f.write(dtxt.replace("__POST__", new_post))
    print("daily log appended")

# ---------------- 6. impact preview addendum ----------------
pv = open(PV, encoding="utf-8").read()
if "addendum-3278" not in pv:
    txt = ("\n<!-- addendum-3278 -->\n③ APPX-1 已执行（Phase 3278，2026-10-21，"
           "owner 批准 nf4 基线）：定位+解释完成（两共享维 0.894/0.715，末段层"
           "新生，话语/真值语义族在案）；nf4 定量守恒 top1 差 4.15° 超预注册"
           "容差 3.0°（量化扰动 R_logic 侧，eff=2 与 cos diff≤0.035 守恒）"
           "→ verdict appx1_blocked（res 588d36c9，ledger #430）。处置（按"
           "定位/解释关闭 vs 留待全精度硬件复测）留 owner。剩余待决：APPX-1 "
           "关闭口径一项。\n")
    with open(PV, "a", encoding="utf-8", newline="") as f:
        f.write(txt)
    print("impact preview addendum appended")

print("CLOSEOUT DONE, head post", new_post)
