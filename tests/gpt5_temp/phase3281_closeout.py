# -*- coding: utf-8 -*-
"""Phase 3281 closeout: five-write (ledger #439 / wlog+mirror / G-line MEMO /
MEMORY compact rewrite / daily log). Idempotent. Same template as 3280.
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LED = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
WLOG = os.path.join(ROOT, "research", "gpt5", "docs", "WORKLOG_gpt5.md")
WLOG_M = os.path.join(ROOT, "tests", "gpt5_temp", "WORKLOG_gpt5_mirror.md")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
DAILY = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-31.md")
RES_DIR = os.path.join(ROOT, "tests", "glm5", "result",
                       "rdc_query_construction_20260913",
                       "phase3281", "g5h44_consumer_attribution_norm")

PREV_POST = "a6bf3af5"
PREV_N = 438


def today():
    import datetime
    return datetime.date.today().isoformat()


res = json.load(open(os.path.join(RES_DIR, "result.json"), encoding="utf-8"))
RES_SHA = res["res_sha8"]
assert res["verdict"] == "consumer_attribution_norm_registered" and res["gate_all"] is True
assert res["gates_k"]["R_logic"]["overlap"] == 15 and res["gates_k"]["K_entity"]["overlap"] == 15

# ---------------------------------------------------------------- 1. ledger
led = json.load(open(LED, encoding="utf-8"))
ms = led["measurements"]
if any(str(m.get("phase")) == "3281" for m in ms):
    new_post = led["ledger_sha256_8"]
    print("ledger: 3281 present, head", new_post, "n", len(ms))
else:
    assert led["ledger_sha256_8"] == PREV_POST and len(ms) == PREV_N, \
        (led["ledger_sha256_8"], len(ms))
    rec = {
        "n": len(ms) + 1,
        "phase": "3281",
        "name": "g5h44_consumer_attribution_norm",
        "line": "G",
        "date": today(),
        "verdict": "consumer_attribution_norm_registered",
        "res_sha8": RES_SHA,
        "note": ("Share-normalized consumer attribution (correction "
                 "preregistered in #432). Static arm identical to 3280 "
                 "(anchors 45.845/73.528/69.823 reproduced; top8 == 3280). "
                 "Dynamic D_norm=mean_rows(||proj||/||c||), silent rows "
                 "<1e-6 excluded (frozen pre-run). K-A overlap 15/15 (>=10 "
                 "both) and K-A2 spearman 0.340/0.362 (>=0.25 both) PASS — "
                 "static write strength predicts dynamic share. Arm c "
                 "confound evidence: spearman(D_abs, mean||c||)=0.973/0.962 "
                 "vs spearman(S_rel, mean||c||)=0.069/0.097 — the 3280 "
                 "absolute-norm confound quantified; 3280-style gates "
                 "recomputed 9/11 + 0.137/0.179 verbatim == #432 "
                 "(continuity). Share top heads: R_logic ~0.13-0.15 of head "
                 "output (28,6)/(30,17)/(23,10)/(23,9)/(28,7); K_entity "
                 "~0.19-0.24 (35,5)/(29,27)/(30,17)/(23,9)/(35,22) — static "
                 "relay family confirmed dynamically. Late-block share "
                 "0.45/0.61. Next prereg: 3282 PR-integral head screening."),
        "prev_post": PREV_POST,
    }
    led["measurements"] = ms + [rec]
    probe = json.loads(json.dumps(led))
    probe["ledger_sha256_8"] = PREV_POST
    new_post = hashlib.sha256(json.dumps(probe, ensure_ascii=False, indent=1)
                              .encode("utf-8")).hexdigest()[:8]
    led["ledger_sha256_8"] = new_post
    with open(LED, "w", encoding="utf-8", newline="") as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    print("ledger #439 appended, post", new_post, "n", len(led["measurements"]))

# ---------------------------------------------------------------- 2. wlog + mirror
wtxt = (
    "\n## Phase 3281 (G5-H44) g5h44_consumer_attribution_norm — __DATE__\n"
    "- 份额归一化修正相（#432 预注册；同判据）。Arm a 静态与 3280 逐位一致"
    "（top8 (24,19)/(23,10)/(23,9)/(23,11)…，gA 锚 45.845/73.528/69.823 复现）。\n"
    "- Arm b D_norm=mean_rows(||proj||/||c||)（静默行 <1e-6 剔除，预运行冻结）："
    "**K-A overlap 15/15（需双 ≥10）、K-A2 spearman 0.340/0.362（需 ≥0.25）"
    "双过 → verdict consumer_attribution_norm_registered**。静态写入强度预测"
    "动态份额——3280 的 FAIL 闭环为度量混淆而非机制缺失。\n"
    "- Arm c（报告制）混淆定量：spearman(D_abs, mean||c||)=0.973/0.962 vs "
    "spearman(S_rel, mean||c||)=0.069/0.097；3280 旧口径复算 9/11 + "
    "0.137/0.179 与 #432 逐字一致（连续性）。份额 top：R_logic ~0.13–0.15"
    "（(28,6)/(30,17)/(23,10)/(23,9)/(28,7)）；K_entity ~0.19–0.24"
    "（(35,5)/(29,27)/(30,17)/(23,9)/(35,22)）——静态中继头族动态确认；"
    "late_share 0.45/0.61。\n"
    "- freeze 7807b62c / res __RES__ / ledger #439 post '__POST__'（n=439）。"
    "下一 Phase 预注册：3282 路由侧 PR 积分筛选相（G5-DP01 P2 通用筛）。\n"
).replace("__DATE__", today()).replace("__RES__", RES_SHA)
for p in (WLOG, WLOG_M):
    cur = open(p, encoding="utf-8").read()
    if "g5h44_consumer_attribution_norm" not in cur:
        with open(p, "a", encoding="utf-8", newline="") as f:
            f.write(wtxt.replace("__POST__", new_post))
print("wlog+mirror done")

# ---------------------------------------------------------------- 3. G-line MEMO
mtxt = (
    "\n### Phase 3281 (G5-H44) 消费方归因修正相（份额归一化）— __DATE__\n"
    "- verdict `consumer_attribution_norm_registered`（5/5 闸门全过；#432 预注册"
    "的修正，同判据）。Arm a 静态与 3280 逐位一致（gA 锚 45.845/73.528/69.823 "
    "复现）。\n"
    "- **K-A overlap 15/15、K-A2 spearman 0.340/0.362 双过**——D_norm="
    "mean_rows(||proj||/||c||)（静默行 <1e-6 剔除，预运行冻结）下，静态写入强度"
    "预测动态份额；3280 K-A/K-A2 FAIL 闭环为**度量混淆**而非机制缺失。\n"
    "- Arm c 报告制混淆定量：spearman(D_abs, mean||c||)=0.973/0.962 vs "
    "spearman(S_rel, mean||c||)=0.069/0.097（静态量本就不受幅度混淆）；3280 旧"
    "口径复算 9/11 + 0.137/0.179 与 #432 逐字一致。\n"
    "- **份额量级**：K_entity top 头输出的 19–24% 落在 8 维载荷子空间"
    "（(35,5)/(29,27)/(30,17)/(23,9)/(35,22)），R_logic top 13–15%"
    "（(28,6)/(30,17)/(23,10)/(23,9)/(28,7)）——静态中继头族获动态确认，"
    "3280 绝对范数 top 里的 (34,15)/(35,0) 未进份额 top8（幅度混淆受害者）。\n"
    "- res __RES__ / ledger #439 post '__POST__'（n=439）。下一 Phase 预注册："
    "3282 路由侧 PR 积分筛选相（G5-DP01 P2 通用筛；H_content + R_h/W_h 入原子"
    "注册表）。\n"
).replace("__DATE__", today()).replace("__RES__", RES_SHA)
cur = open(MEMO, encoding="utf-8").read()
if "Phase 3281 (G5-H44)" not in cur:
    with open(MEMO, "a", encoding="utf-8", newline="") as f:
        f.write(mtxt.replace("__POST__", new_post))
print("memo done")

# ---------------------------------------------------------------- 4. MEMORY compact rewrite
mtxt2 = (
    "# RDC/LPF 双线项目纪律（工作区索引）\n"
    "> 权威日志：deepseek 线 `research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`"
    "（P1–40 收官）；gpt5/G 线 `research\\gpt5\\docs\\AGI_GPT5_MEMO.md`（现役）。"
    "细节回查 MEMO/技能/用户级记忆。\n"
    "- 「好的，继续」= AI 主导续研；关键发现重复 3 次。Ledger "
    "`research\\gpt5\\atlas\\atlas_ledger.json`（跨线共享，他线并发只 append）。\n"
    "\n"
    "## 0 现役纪律（G 线五段式）\n"
    "- 每轮：freeze（DESIGN 静态、hash 排除 created）→ SMOKE → 正式 → verdict → "
    "closeout 五写 → 独立复核（verify FAIL 只修 verify）→ present → 预注册下一 Phase。\n"
    "- sha8 锚：锚值实测探针禁凭记忆；内容 sha ≠ 文件 sha；registry/gap vN immutable，"
    "升级 append only，legacy 字节保全断言；链头复算含全部 measurements 且 "
    "field=prev_post；closeout rec 的 phase 用字符串。\n"
    "- E 级 taxonomy：E0/E1/E2/E3；失败必入 F 账本；读得出≠控得住、指纹≠机制。"
    "写盘 newline=''（Windows 换行陷阱）；closeout 模板禁裸 % 格式化。\n"
    "- G5-DP01 P0 地基已落：provenance 五字段强制（tools/provenance.py）、"
    "atom_registry.v1 冻结（research/gpt5/atlas/）、三类功效池 pools.v1（n=1263/"
    "1411/1087）、power_analysis（AGENTS.md §13：预注册三栏强制）。\n"
    "\n"
    "## 1 深线（AGI_DEEPSEEK，P1–40 收官）\n"
    "主轴三段/写入端分布式/REACH 域；P35 A 闸门 seal。挂账：无。\n"
    "\n"
    "## 2 G 线现状（registry v1.39 ab14e586 + gap v1.16 12961b7b）\n"
    "- 26 特征×3 模型全 scope；F1–F20；GAP-1/2/3 closed、GAP-4/5 收官；"
    "E 线 CLOSED@3201；APPX 10 项。转向：Cammarata 对齐（F20 主攻）。\n"
    "\n"
    "## 3 phase 索引\n"
    "3154–3231 回 MEMO；3238–3257 收线链收官；3267–3270 协议 v2；3278–3279 "
    "APPX-1（nf4 G-C FAIL 逐字入账）。\n"
    "\n"
    "## 4 现役与挂账\n"
    "- 链头：ledger #439 post '__POST__'（n=439）。\n"
    "- APPX-1 处置留 owner；其余全结清。\n"
    "- **下一 Phase 预注册：3282 路由侧 PR 积分筛选相**（1152 头 PR−身份对照 → "
    "H_content；R_h/W_h 基入 atom_registry head_dir 类；中继头族并入标注）。\n"
    "\n"
    "## 5 本机缺陷（Windows，全量见用户级记忆）\n"
    "bash shim 坏/反引号被吃/内联 stdout 丢 ⇒ Python 文件化写 .txt 再 Read；"
    "Edit 幻影 ⇒ 关键写后回读；沙箱 curl localhost 须 --noproxy '*'；V×V 大 Gram "
    "内存爆改 D 空间 GPU eigh；qwen3-4b tied embeddings。\n"
    "\n"
    "## 6 近期 phase 锚（细节回 daily log/MEMO）\n"
    "- 3280：P0 消费方归因 v1（BLOCKED）：静态中继头族层 23–30 ~9×；动态绝对范数"
    "混淆 K-A/K-A2 FAIL。res ce0c400b。#432。\n"
    "- **3281：份额归一化修正（REGISTERED 5/5）**：D_norm=mean_rows(||proj||/"
    "||c||)，K-A 15/15、K-A2 0.340/0.362 双过——静态写入强度预测动态份额；"
    "混淆定量 D_abs~‖c‖ ρ=0.97 vs S_rel~‖c‖ ρ≈0.07；份额 top：K_entity 头输出 "
    "19–24% 落载荷子空间（(35,5)/(29,27)/(30,17)/(23,9)/(35,22)），R_logic "
    "13–15%；(34,15)/(35,0) 未进份额 top8（幅度受害者）。res a8f1e066。#439。\n"
    "- 语言编码机制线（v3）：几何筛证伪、共现聚类作废、统计单元+一阶归因排序、"
    "格式单元陷阱（R-MET-11/12）——细节见 MEMO。\n"
    "- 通道测量纪律：概念=低维子空间非单方向；跨层 cos 不可用；范数预测可读性"
    "不预测一致性；SwiGLU 规范自由度⇒范数非内禀量。\n"
).replace("__POST__", new_post)
assert len(mtxt2) <= 3000, len(mtxt2)
with open(MEMORY, "w", encoding="utf-8", newline="") as f:
    f.write(mtxt2)
print("MEMORY rewritten, len", len(mtxt2))

# ---------------------------------------------------------------- 5. daily log
dtxt = (
    "- 3281 G5-H44 消费方归因修正相（份额归一化，#432 预注册，同判据；REGISTERED "
    "5/5）：Arm a 静态与 3280 逐位一致（gA 锚 45.845/73.528/69.823）。Arm b "
    "D_norm=mean_rows(||proj||/||c||)（静默行 <1e-6 剔除，预运行冻结）——K-A "
    "overlap 15/15（需双≥10）、K-A2 spearman 0.340/0.362（需≥0.25）双过：静态"
    "写入强度预测动态份额，3280 FAIL 闭环为度量混淆而非机制缺失。Arm c 报告制："
    "spearman(D_abs, mean||c||)=0.973/0.962 vs spearman(S_rel, mean||c||)=0.069/"
    "0.097（混淆定量坐实）；3280 旧口径复算 9/11+0.137/0.179 与 #432 逐字一致。"
    "份额量级：K_entity top 头输出 19–24% 落 8 维载荷子空间（(35,5)/(29,27)/"
    "(30,17)/(23,9)/(35,22)），R_logic 13–15%（(28,6)/(30,17)/(23,10)/(23,9)/"
    "(28,7)）——静态中继头族动态确认；(34,15)/(35,0) 未进份额 top8（幅度受害者）。"
    "late_share 0.45/0.61。freeze 7807b62c / res a8f1e066 / ledger #439 post "
    "'__POST__'（n=439）。下一 Phase 预注册：3282 路由侧 PR 积分筛选相（G5-DP01 "
    "P2 通用筛，H_content+R_h/W_h 入 atom_registry）。\n"
).replace("__POST__", new_post)
cur = open(DAILY, encoding="utf-8").read()
if "3281 G5-H44" not in cur:
    with open(DAILY, "a", encoding="utf-8", newline="") as f:
        f.write(dtxt)
print("daily log done")
print("CLOSEOUT DONE, head post", new_post)
