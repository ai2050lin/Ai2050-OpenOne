# -*- coding: utf-8 -*-
"""R4 落盘：技能教训 37 + MEMORY.md 重写 + 当日 wlog。幂等 + 断言 + 回读复核。"""
import os, hashlib, shutil

ROOT = r"D:\AI2050\Ai2050-OpenOne"
TMP  = os.path.join(ROOT, "tests", "deepseek_temp", "_review")
MP   = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
DL   = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-03.md")
SK   = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
REP  = []
def A(s=""): REP.append(str(s))
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

# ================= 1) 技能：插入教训 37 =================
sk = open(SK, "rb").read().decode("utf-8")
SK0 = len(sk.encode("utf-8"))
LESSON = """37. **元层账本必须做「单一真源审计」（Q01/Q02 实证，2026-10-03）**：

```text
五类陷阱（全部实测）：
(a) 计数口径必须显式命名。同一 62 条复合 grade 账本，component（拆分量，和 94）
    与 atomic_*（每条一值，和 62）给出不同表；MEMO 与 TESTPLAN 各用一半 -> 「对不上」。
    规范：count_mode 必填；本线 canonical = component，并列报 atomic_best。
(b) 禁止「分量数 ÷ 条数」。MEMO 的「约 55%（A+B）」= 34/62、「63%」= 39/62 都是
    分母错配；同量纲唯一成立值 = 39/94 = 41.5%。
(c) 文件内自声明哈希在追加后必然失效。atlas_ledger 声明 41d65a13 vs 实际 bbda63df，
    18 种候选语义全不命中（= 追加前历史值）。修法：
      sha256(json.dumps(d 去掉该字段, ensure_ascii=False, indent=1))[:8]
    排除自身 => 回写后不变（实测不变量 0dc6e57a）。
(d) 元层审计必须「只读」。发现的不自洽一律落「更正表 + 待 seal」，不得顺手改
    MEMO/TESTPLAN/Ledger。就地升版前先确认历史断言（P3150 只查 len(metrics)==7
    与 meta_rules 含 F2/F7 => 加字段安全；先备份 v1 并在 supersedes 记录其 sha8）。
(e) KPI 冻结必须带完整来源指纹。E_read 口径 = k1_model_report.b4_rel_readout_mean3seed，
    须连同 result.json 文件 sha8 + res_sha8 + held-out 载体 npz sha8 一起冻结，
    否则「同一口径」不可复现。
```

"""
if "37. **元层账本必须做" in sk:
    A("[SKIP] 教训 37 已存在")
else:
    anchor = "## 参照实现（Phase 3125"
    assert sk.count(anchor) == 1, "锚点 %s 计数 %d" % (anchor, sk.count(anchor))
    sk = sk.replace(anchor, LESSON + anchor)
    open(SK, "wb").write(sk.encode("utf-8"))
    A("[OK] 技能插入教训 37：%d -> %d B" % (SK0, len(sk.encode("utf-8"))))

# ================= 2) MEMORY.md 重写 =================
raw = open(MP, "rb").read()
shutil.copy(MP, os.path.join(TMP, "_memory_backup_r4_pre_rewrite.md"))
OLD = raw.decode("utf-8")

NEW = """# RDC/LPF 项目纪律（工作区长期记忆）
> 权威记录：`research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（N 线）、`research\\gpt5\\docs\\AGI_GPT5_MEMO.md`（G 线）。本文件仅跨轮索引，细节一律回查 MEMO / 技能。

## 0 约定与落点
- deepseek 线只写 deepseek 备忘录（append-only、UTF-8+**BOM**+**CRLF**、`bare_lf 0`）。
- 产物 v2：脚本→`tests\\deepseek\\Phase{N}\\`；报告/seal/`verify_*`/memo·wlog 节源码→`tests\\deepseek_temp\\Phase{N}\\`；非 Phase→`_review\\`/`_infra\\`。**禁堆根下**。
- 收尾链（**十四次 P8–P21**）：探针→seal→amend→exec→SMOKE→正式→判决→Ledger→MEMO→基线→wlog→MEMORY→技能→独立复核→present。
- 标题 `## Phase {N}: 短标题（线-P）[hh:mm]`；纠错 append 不回改（**唯一例外：交付前生成件缺陷⇒逐字节回滚基线重跑链**）。
- Ledger `atlas_ledger.json` n=**304** / sha8 **`bbda63df`**（P8–P21 各 1；**N 线 P3–P7 待补**）。基线 `post-append-phase21`=497,406 B/4,774 行（`4ba5e22f`）。

## 1 装置铁律（26 条全文在 skill `rdc-main-axis-probe` 坑集，现 **63 坑**）
最易违反：(a) 份额只用**精确可加向量预算**。(b) SMOKE 必做**必看数字**。(o) 同消息多 Edit 静默丢⇒Python 补丁+`assert count==1`+回读。(ad) 实现与 seal **逐字一致**。(ae) 数字**一律 result 现场渲染**。(ac) 插值读数不设单一硬门⇒分层。(af) **冻结锚重验**、如实报「陈旧」；字典键不得截断生成。
其余：patch 剔末层；`h_ℓ=hidden_states[ℓ+1]`；写入窗=相邻最大增量；`jump/max|effect|≥0.5`；剔实例 token 同剔类别 token；绝对/相对双剂量 `x*·r_ℓ`；固定基报 `overlap`；GQA 禁 `hidden/n_heads`；换口径=新自由度⇒校准臂；cos 与 rel-L2 双报；平均秩；paired `margin` 负=候选优。

## 2 N 线主线（P4→P21，关键数）
- **P4–P7** 主轴三段（嵌入=词典/层=开关/**权重绑定定 is-a 落点**）；单头 share_max **3.0%**；读位槽=**G−1=5 维**充分必要；**跨族近正交⇒无通用类别算子**；R1 K_d 降级。
- **P8–P11** 写入端**分布式**（向量预算 MLP 0.472/单头 0.074）；L6 内**无阈值增益**、非线性在其**之后**（S 形 `x*`≈0.6）⇒ **栈=软门**；深端塌陷主因**方向失配**。
- **P12–P16** `rho(xhalf,depth)=−0.783`；**P16 预算否证⇒P12/13/14 物理深度表述撤回**；新域 `REACH={ℓ:ρ≥0.10}`。
- **P17** `w_ℓ`+`com_V` **26.15/26.70/26.68** ≫ median(REACH)；`MLP_DOMINANT` 0.740/0.975/0.824；`spearman(w,J)` −0.55/−0.79/−0.60。
- **P18** `share_mlp_beh` 0.66/0.96/0.75；`spearman(w,|b|)` **正**（与 P17 **反号⇒P17 P6 对象错配**）。
- **P19–P21** 跨精度：向量侧 5/5；行为+剖面 8/9（P9 FAIL=域歧义，不改判）；`share_v`+权重级 `W` **7/9**，A0_bf16 **逐位复现 P8 锚**、**G1_core 四臂全 True**⇒「分布式搬运」非 nf4 kernel 产物；Δ`share_v(mlp)`≤**0.0234**、ρ≥**0.9920**。**P8 与 P16–P18 精度缺口已闭合**。

## 3 挂账与限界
- R1/R2：10 条成立；P1 纠错 1（K_d）+降级 4+挂账 5。
- **限界**：① 否定臂基线不成立；② 留一只覆盖「未见实例」，「未见类别」仍崩（水果 0.04/0.05）；③ **激活级干预 ≠ 权重级证明**；④ 跨深度固定基衰减须扣基旋转；⑤ P13 确认集只 3 对；⑥ P14/P15 n≤18；⑦ P15 家族×规模混杂；⑧ P17 `com_V`≠`com_layer`；⑨ P16 exec `bootstrap.seeds` 错记；⑩ P18 `b` 不可加；⑪ P19 只覆盖向量侧；⑫ P20 只两模型+offload、**不**升格 `com_layer`；⑬ P21 argmax 可在**并列带内**跨精度翻转（前三差<0.004）、**跨模型地板不可比**。
- **外部方案裁决**：三图谱骨架/统一实验卡/三集划分/竞争性解释/「预测未见现象」→ 全采纳；「每族一组特征+拼图还原+统一阈值+通用算子」→ 全改写。

## 4 G 线（gpt5）
3151 k3_only；3152 k1_not_triggered（k* 定律）；3153 fingerprint_consistent_coverage_partial；3105–3150：真值=记录级一阶矩广播、写入头组 L20–28 主写/L32 擦除；**判决符号=rev-3151b（负=优）**。**3154 已预注册**。**⚠ K1 判定层问题与其元层对账见 §9。**

## 5 本机缺陷（Windows）
- bash shim 劣化：`ls`/`rm`/`tail` 坏、内联 `python -c` stdout 常丢、**反引号被吃** ⇒ Python 文件化+写 `.txt` 再 Read。
- **`Edit` 幻影**+IME 吞汉字 ⇒ 大段中文走 Python 补丁；调用 `cd <root> && .venv/Scripts/python.exe tests/...`。关键写入后 Grep 复核；GPU 逐模型防 OOM。
- **`Read` 视图可能陈旧**（大文件）⇒「改了没有」一律用 Grep/Python 判定。
- **P16–P21 逐 Phase 教训**在 skill `rdc-phase-closeout` 教训 26–35（P21：identity-probe 取权重 / argmax 并列内翻转 / 跨模型地板不可比 / 复核脚本 `sha8`-vs-64hex 错配⇒修脚本不动产物）。

## 6 工作方式
「好的，继续」= AI 主导不停、深度自主续研；结构化输出；**关键发现重复 3 次**。

## 7 下一步（死线优先级）
- **Phase 22** = P8 `share_v` 与 P16/P17 逐层 `w_ℓ` 在**同一精度**下逐位对接，消除跨 Phase 精度不确定性。并列：邻域 ±2 敏感性；P9 判据补域；P17 `P6` 的 MEMO 改判。
- **其他挂账**：N2h1-α-1 权重级；N2h1-β 水果类崩塌；N3-β→ε；R1 补强；K4；**P3–P7 补登 Ledger**。
- **A 闸门进度**：**Q01/Q02 已完成**（§9 末）；Q09（死线双轨）/Q12（D/E 引用审计）可零 GPU 直做；Q03–Q07 为 KPI 装置。
- **⚠ 待用户 seal**：Q01 更正表 **C1–C6**；Q08（K1 改判，两种读法已备）；I1/I9（宪法 `01df6398` + 队列 `675836fd` 已生成）。

## 8 技能
`rdc-main-axis-probe`（15 臂 + **63 坑**）、`rdc-phase-closeout`（**37 教训**/十四次链）、`rdc-dual-arm-phase-template`。

## 9 元层循环诊断与 A 闸门（2026-10-02/03，非 Phase）
产物：`LOOP_DIAGNOSIS_AND_EXIT_v1.md`（`013d08a9`）+ `RDC_RESEARCH_CONSTITUTION_v1.md`（`01df6398`）+ `atlas/phase_queue_v1.json`（`675836fd`，Q01–Q30）；数据 `loop_stats_r3.json`；页 `loop_diagnosis_r3.html` / `q01q02_gate_r4.html`。
- **诊断核心**：gpt5 MEMO 1.90 MB/14,901 行/**402 Phase**；判决 **685** 次但**判决串 47→48 唯一标签（复用率 1.10）⇒ 不可累积**；A 级 5/62(8%)；「接续」**380** 处。**改判主张**：`k1_not_triggered…operator_line_kept` **应改判**——判据挂在 k*（≈7.5%）⇒ above5=**1/3** 未触发；但**行为读出层 0.3316/0.3986/0.3898 = 5% 门的 6.6×**；3153 读出层：交互格 **0.0–0.3%**、加性残差 34–38%、**模板内高秩散布 56.8–60.4%**（谱相关 0.968–0.991）。四机制（验收错位/死线免疫/自催化/吸收证伪）与 I1–I11 全文见诊断与宪法。
- **教训落地**：**397 Phase 仅 26（6.55%）报告共享指标变化**（窄词表 18）⇒ 93.45% 是「目录条目」。
- **A 闸门 Q01（元层对账）**：五处不自洽全定位根因（原 3 + 新 2：MEMO 记 `proposition_ledger`=`add57ba7`≠实际 `9a3c6ff4`；`measurements` 两套 schema 且 `evidence_level` **304/304 恒为常量**）。**口径**：真源 62 条（review57+new5），按**分量计**复现 MEMO A5/B34/C14/D21/E20（和 94）；**TESTPLAN 的 B6/C10 五口径全不可复现**；「55%/63%」= 分量数÷条数**量纲混淆**（同量纲唯一成立值 41.5%=39/94）。**哈希**：自声明 `41d65a13`=追加前旧值（18 候选全不命中）⇒ 自指结构性失效；自洽值 `0dc6e57a`（content_excluding_self，回写不变）。产物 `META_SINGLE_SOURCE_Q01.md`+`meta_single_source_v4.json`。**C1–C6 待 seal**。
- **A 闸门 Q02（KPI 冻结）**：`metric_dict.json` **v1→v2**（`469c0ad1`→`03887e51`，v1 备份在册）；新增 `global_kpis`——E_read=`k1_model_report.b4_rel_readout_mean3seed` **0.33162/0.39860/0.38984（0/3 过 5% 门）**、E_ar(k)/C_steer 口径冻结未测；`metrics`7 项 + `meta_rules` **原样** ⇒ 兼容 P3150 历史断言。**复核 44/0 + 43/0**。
"""
# 断言锚点齐全
ANCH = ["# RDC/LPF 项目纪律", "## 0 约定与落点", "## 1 装置铁律", "## 2 N 线主线", "## 3 挂账与限界",
        "## 4 G 线", "## 5 本机缺陷", "## 6 工作方式", "## 7 下一步", "## 8 技能", "## 9 元层循环诊断",
        "63 坑", "37 教训", "304", "bbda63df", "01df6398", "675836fd", "03887e51", "0dc6e57a",
        "41d65a13", "add57ba7", "9a3c6ff4", "C1–C6"]
miss = [a for a in ANCH if a not in NEW]
assert not miss, "MEMORY 新文本缺锚点: %s" % miss
open(MP, "wb").write(NEW.encode("utf-8"))
nb = open(MP, "rb").read()
A("[OK] MEMORY.md 重写：%d -> %d B (%d -> %d 字符) sha8 %s -> %s  LF-only=%s BOM=%s"
  % (len(raw), len(nb), len(OLD), len(NEW), sha8(raw), sha8(nb), (b"\r" not in nb), nb[:3] == b"\xef\xbb\xbf"))

# ================= 3) 当日 wlog =================
WLOG = """
## R4：A 闸门 Q01 + Q02（元层单一真源 + KPI 口径冻结，零 GPU）
- **Q01**：五处元层不自洽全定位根因（原 3 + 新 2：MEMO 记 `proposition_ledger`=`add57ba7`≠实际 `9a3c6ff4`；`measurements` 两套 schema 且 `evidence_level` 304/304 恒为常量）。口径量纲混淆（55%/63% = 分量数÷条数）；哈希自指结构性失效（自洽值 `0dc6e57a`）。产物 `research/gpt5/docs/META_SINGLE_SOURCE_Q01.md`(`95366ca9`) + `research/gpt5/atlas/meta_single_source_v4.json`(`f9d7ede6`)；独立复核 **44/0 ALL_PASS**。
- **Q02**：`metric_dict.json` v1→v2（`469c0ad1`→`03887e51`；备份 `metric_dict_v1_backup.json`），新增 `global_kpis`（E_read=`b4_rel_readout_mean3seed` 0.33162/0.39860/0.38984，**0/3 过 5% 门**；E_ar/C_steer 口径冻结未测）+ 登记规则；`metrics`7 + `meta_rules` 原样 ⇒ 兼容 P3150 历史断言。报告 `research/gpt5/docs/METRIC_DICT_Q02.md`(`ba518fea`)；复核 **43/0 ALL_PASS**。
- 汇报页 `tests/deepseek_temp/_review/q01q02_gate_r4.html`(`d5fbf42b`)。
- 脚本入 `tests/deepseek/_review/`（probe_q01/q01b/q02/q02b/q02c、gen_q01/gen_q02/gen_q01q02_html、disk_verify_q01/q02），报告入 `tests/deepseek_temp/_review/`。
- **未改动**任何 MEMO 原文 / TESTPLAN / Ledger / 3152 产物；6 条更正 C1–C6 **待 seal**。
- 技能 `rdc-phase-closeout` 新增**教训 37**（元层单一真源审计 5 条）；MEMORY.md 重写压缩并纳入本节。
"""
db = open(DL, "rb").read().decode("utf-8")
if "R4：A 闸门 Q01 + Q02" in db:
    A("[SKIP] wlog R4 已存在")
else:
    open(DL, "wb").write((db.rstrip("\n") + "\n" + WLOG).encode("utf-8"))
    A("[OK] wlog 追加：%d -> %d B" % (len(db.encode("utf-8")), os.path.getsize(DL)))

open(os.path.join(TMP, "_patch_r4_report.txt"), "wb").write(("\n".join(REP) + "\n").encode("utf-8"))
print("\n".join(REP))
print("OK")
