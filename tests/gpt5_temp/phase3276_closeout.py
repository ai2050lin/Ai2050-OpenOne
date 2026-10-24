# -*- coding: utf-8 -*-
"""Phase 3276 closeout: five writes + impact-preview addendum."""
import hashlib
import json
import os

BASE = r"D:\AI2050\Ai2050-OpenOne"
LED = os.path.join(BASE, "research", "gpt5", "atlas", "atlas_ledger.json")
WLOG = os.path.join(BASE, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
MIRR = os.path.join(BASE, "research", "deepseek", "docs",
                    "WORKLOG_gpt5_mirror.md")
MEMO = os.path.join(BASE, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
PREVIEW = os.path.join(BASE, "research", "gpt5", "docs",
                       "PENDING_ITEMS_IMPACT_PREVIEW.md")
MEMORIA = os.path.join(BASE, ".workbuddy", "memory", "MEMORY.md")
DAILY = os.path.join(BASE, ".workbuddy", "memory", "2026-10-21.md")
OUT = os.path.join(BASE, "tests", "glm5", "result",
                   "rdc_query_construction_20260913", "phase3276",
                   "g5h39_appx23_exec")
RES = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
MARK = "## Phase 3276 (G5-H39):"


def append_once(path, text, marker):
    cur = open(path, encoding="utf-8").read()
    if marker in cur:
        print("skip (idempotent):", os.path.basename(path))
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)


# 1 ledger receipt (#428, n 433->434)
L = json.load(open(LED, encoding="utf-8"))
ms = L["measurements"]
if ms[-1]["phase"] == 3276:
    print("skip: ledger receipt already present")
    new_post = L["ledger_sha256_8"]
else:
    assert len(ms) == 433 and L["ledger_sha256_8"] == "923806e0", (
        len(ms), L["ledger_sha256_8"])
    rec = {"phase": 3276, "name": "g5h39_appx23_exec",
           "verdict": "appx23_exec_registered",
           "evidence_level": "statistical",
           "res_sha8": RES["res_sha8"],
           "note": "APPX-2/3 execution per protocol ce73984d (owner-named "
                   "2026-10-21): APPX-2 all-5-model S_class/S_attr/S_syntax "
                   "x K_readout separable (58.5-73.6 deg, conservation); "
                   "APPX-3 K3 deathline FIRES 3/3 models (top-50 single-coord "
                   "coverage 1.9-5.9% << 30%), K2 does not fire 3/3 (cos "
                   "drop 1.2-1.4%); X-gates 5/5; landing via 3277 (G5/H5)",
           "prev_post": L["ledger_sha256_8"]}
    Ld = dict(L)
    Ld["measurements"] = ms + [rec]
    Ld["ledger_sha256_8"] = L["ledger_sha256_8"]
    new_post = hashlib.sha256(json.dumps(Ld, ensure_ascii=False, indent=1)
                              .encode("utf-8")).hexdigest()[:8]
    Ld["ledger_sha256_8"] = new_post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(Ld, f, ensure_ascii=False, indent=1)
    print("write1 ledger: #428 post", new_post,
          "(n=%d)" % len(Ld["measurements"]))

# 2 wlogs
wtxt = ("- **3276（G5-H39）appx23_exec_registered**（owner 点名执行相，协议 ce73984d，"
        "GATE 全过 + X-gate 5/5）：**ARM a/APPX-2**——5 候选模型（qwen2.5-3b/qwen2-7b/"
        "r1-distill-7b/glm4-9b/gemma-3-4b）S1 词表坐标重建（attr 8/10 轴全过 VG1；"
        "syntax 存活 3/1/0/1/4，tokenizer-face 分歧如实记录）+ S3 静态 K_readout "
        "（3158 Gram top64）×S_class/S_attr/S_syntax 主角判定：**全部 separable "
        "（58.5–73.6°）**——S 族轴对读出空间的可分离性跨 6 模型守恒（含 4b 现役）。"
        "**ARM b/APPX-3**——K2 直接测量（冻结操作化 phi_l(c)=QC∪QEC，解析读出接口 "
        "logits=W_U@h，3158 位等价）：cos 降 1.18–1.39% ≪50% 且 first-token 一致率 "
        "0.509–0.639 <0.8 → **K2 死线 3/3 不触发**（诚实注记：条件成分低幅但行为承重，"
        "准则双条件均不满足）；K3 覆盖率（η² top-50）**1.9–5.9% ≪30% → 死线 3/3 触发**"
        "——单坐标机制目标按 p35 死线废弃（≤10 维联合+端口类）。X-gate：4b S1 逐位=="
        "2874/2878 封存、K_readout≈3158、S_class 主角 66.99°≈67.0 现役、ExC 份额"
        "逐位==3155。freeze bda93f83 / res __RES__ / ledger #428 post '__POST__'（n=434）。"
        "落册按 G5/H5 留待 3277 registry 相；剩余待决 1 项：APPX-1 三选一。\n"
        .replace("__RES__", RES["res_sha8"]).replace("__POST__", new_post))
append_once(WLOG, "\n" + MARK + " APPX-2/3 执行相\n\n" + wtxt, MARK)
append_once(MIRR, "\n" + MARK + " APPX-2/3 执行相\n\n" + wtxt, MARK)

# 3 G-line MEMO
memo_txt = ("\n" + MARK + " APPX-2/3 执行相（owner 点名，协议 ce73984d）\n\n"
            "- 裁决：owner 于 2026-10-21 点名 APPX-2/3 执行相（PENDING_ITEMS_"
            "IMPACT_PREVIEW ④）。\n"
            "- ARM a/APPX-2（5 候选模型逐模型，H4 不池化）：S1 按 2874/2878 规则"
            "在 W_U 行上重建 attr/syntax 词表坐标（attr 8/10 轴全模型过 VG1；"
            "syntax 存活 3/1/0/1/4——tokenizer-face 分歧如实记录，从不池化）；"
            "S2 3157 式 128 行收集（gemma-3 fp16 溢出教训→fp32 存储，slot 范数"
            "记录）→ K_entity top8 vs K_readout 主角 64.5–79.0°（descriptive）；"
            "S3 K_readout=3158 同配方静态 Gram top64 + S_class=2881 配方逐模型；"
            "主角判定 **全部 separable（58.5–73.6°）**：S 族轴与读出子空间的"
            "可分离性为跨 6 模型守恒性质。\n"
            "- ARM b/APPX-3（4b/14b/glm4，封存 3155 collects 复用，零新前向）："
            "T1 冻结操作化 phi_l(c):=QC∪QEC（3155 平衡正交基逐字构造），分离投影"
            "后响应 cos 降 1.18–1.39%；T2 解析读出接口 logits=W_U@h（3158 位等价"
            "协议）first-token 一致率 0.509–0.639；**K2 死线 3/3 不触发**（诚实"
            "注记：条件成分幅低但行为承重，双条件均不满足）；T3 η² top-50 单坐标"
            "覆盖率 1.9–5.9%，**K3 死线 3/3 触发 → 单坐标机制目标按 p35 废弃"
            "（≤10 维联合+端口类）**。\n"
            "- X-gate 5/5：4b S1 重建逐位==2874/2878 封存 dW；K_readout≈3158 "
            "top64；S_class×K_readout 66.99°≈3165 现役 67.0；ExC 份额逐位=="
            "3155 封存。X-gate 教训：S_class 质心须用未截断全量单 token 词"
            "（3165 原文），截断仅用于 material 断言；投影基为样本空间 Q@(QᵀX)。\n"
            "- freeze bda93f83 / res __RES__ / ledger #428。落册（gap/registry 注记）"
            "按 G5/H5 留待 3277 registry 相。剩余待决 1 项：APPX-1 量化三选一。\n\n"
            .replace("__RES__", RES["res_sha8"]))
append_once(MEMO, memo_txt, MARK)

# 3b impact preview addendum
PVMARK = "addendum-3276"
pv_txt = ("\n\n<!-- %s -->\n④ APPX-2/3 执行相已实施（Phase 3276，2026-10-21，"
          "owner 点名）：APPX-2 全 5 模型 S 族轴×K_readout 判定 separable（守恒）；"
          "APPX-3 K3 死线 3/3 触发（弃单坐标机制目标）、K2 3/3 不触发。gap/registry "
          "落册留待 3277（协议 G5/H5）。剩余待决 1 项：③ APPX-1 量化三选一。\n"
          % PVMARK)
append_once(PREVIEW, pv_txt, PVMARK)

# 4 MEMORY: add 3276 line
m = open(MEMORIA, encoding="utf-8").read()
o = "- **3275**：**水果类/K4 结清勘误**（owner 批准，append-only）：deepseek MEMO " \
    "尾 dated 注记（3210/3216 三锚，35 行 52 处注记），前缀字节不动；MEMO 现 " \
    "`fb813f57`。#427。"
n = o + ("\n- **3276**：**APPX-2/3 执行相**（owner 点名，协议 ce73984d）：APPX-2 全 5 模型 "
         "S 族×K_readout separable（58.5–73.6°，守恒）；APPX-3 **K3 死线 3/3 触发**"
         "（单坐标覆盖 1.9–5.9%<30%→弃单坐标目标）、K2 3/3 不触发；落册留 3277。#428。")
if "**3276" not in m:
    if o in m:
        m = m.replace(o, n)
    else:
        m = m.rstrip("\n") + "\n- **3276**：APPX-2/3 执行相（协议 ce73984d）；K3 死线 3/3 触发。#428。"
    m = m.replace("挂账：APPX-1 量化待确认；APPX-2/3 待点名（协议 ce73984d）。"
                  "APPX-4/5、水果类/K4 已结清（3274/3275）。",
                  "挂账：APPX-1 量化待确认。APPX-2/3 已执行（3276，落册 3277）；"
                  "APPX-4/5、水果类/K4 已结清（3274/3275）。")
    open(MEMORIA, "w", encoding="utf-8", newline="").write(m)
print("MEMORY len(chars):", len(m))
assert len(m) <= 3000, "MEMORY OVER: %d" % len(m)

# 5 daily
append_once(DAILY,
            "\n- 3276 G5-H39 APPX-2/3 执行相收官（appx23_exec_registered，协议 ce73984d，"
            "owner 点名 2026-10-21）：ARM a/APPX-2 五候选模型逐模型 S1 词表坐标重建"
            "（attr 8/10 轴全过；syntax 存活 3/1/0/1/4 tokenizer-face 分歧如实记录）+ "
            "S3 静态 K_readout×S_class/S_attr/S_syntax 主角判定全 separable（58.5–73.6°，"
            "跨 6 模型守恒）；S2 K_entity 轮廓 64.5–79.0°（gemma fp16 溢出→fp32 教训）。"
            "ARM b/APPX-3 封存 3155 collects 复用零新前向：K2（phi_l(c)=QC∪QEC 冻结操作化，"
            "解析读出 logits）cos 降 1.18–1.39% 且一致率 0.509–0.639 → 死线 3/3 不触发"
            "（条件成分低幅但行为承重）；K3 η² top-50 覆盖 1.9–5.9% → 死线 3/3 触发，"
            "单坐标机制目标废弃（≤10 维联合+端口类）。X-gate 5/5（4b 逐位复现）。"
            "freeze bda93f83 / res __RES__ / ledger #428 post __POST__（n=434）。落册留 3277；"
            "剩余待决 1 项：APPX-1 三选一。\n"
            .replace("__RES__", RES["res_sha8"]).replace("__POST__", new_post),
            "3276 G5-H39")
print("closeout done; #428 post:", new_post)
