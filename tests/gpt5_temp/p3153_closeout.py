# -*- coding: utf-8 -*-
"""Phase 3153 closeout：Ledger + MEMO + daily + MEMORY 压缩重写（幂等）。"""
import json, io, os, time, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
LEDGER = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
MEMO = os.path.join(ROOT, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")
DAILY = os.path.join(ROOT, ".workbuddy", "memory", "2026-10-01.md")
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
out = []

def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

# ============ 1. Ledger ============
led = json.load(io.open(LEDGER, encoding="utf-8"))
ms = led["measurements"]
if any(m.get("phase") == 3153 for m in ms):
    out.append("ledger: 3153 already present (n=%d), skip" % len(ms))
else:
    entry = {
        "phase": 3153, "name": "g1p3_failure_mode_anatomy",
        "seal_sha8": "8236f974", "result_sha8": "b9846738",
        "evidence_level": "statistical",
        "model_scope": "qwen3-4b + qwen3-14b + glm4-9b",
        "n_rows": 738, "prereg_id": "p3153", "superseded_by": None,
        "verdict": "g1p3_fingerprint_consistent_coverage_partial",
        "rev_note": ("all-zero-GPU recompute from 3151/3152 collect.npz; anchors bit-equal: "
                     "worst20 drift 0.0 x3, V5 drift 0.0 (4b/14b), M1@k* s7 margins "
                     "-0.0040/-0.0432/-0.0289 vs 3152 gates; patch2 fixed M1 gate block "
                     "(Mte/e_m1 were inside t-loop); fingerprint fp_corr "
                     "0.999/0.995/0.998 spec_corr 0.968/0.991/0.977 all>=0.8 -> dead-line "
                     "not triggered; ANOVA readout [class 5-6.5, additive 34-38, "
                     "interact 0.0-0.3, within 57-60]% -> (i,c) grid energy ~0 closes "
                     "3152 rank10 mystery; M1 factors ~= Pc(k*) principal 0.979/0.889/0.977 "
                     "vs Pc(KOUT) 0.142/0.335/0.378 (4th independent confirmation); "
                     "per-model res: 4b c222c7ff/dad158d2, 14b a0049e59/a600c4b5, "
                     "glm4 18f2ef01/d36c7627"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ms.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode("utf-8")
    led["ledger_sha256_8"] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    out.append("ledger: appended 3153 (n=%d) sha8=%s" % (len(ms), led["ledger_sha256_8"]))

# ============ 2. MEMO（幂等检查标题） ============
memo_txt = io.open(MEMO, encoding="utf-8").read()
sec = """
## Phase 3153: 失败模态解剖（G1-P3）[17:34]

**主判决：`g1p3_fingerprint_consistent_coverage_partial`——读出层高秩残差谱指纹三模型一致（死线未触发，"交互=类子空间"叙事保住），覆盖率门部分（cov_mean 0.83 达标、4b 0.60 未达）。**

### 设计与执行
- 全零 GPU 复算：4b/14b 读 3152 collect.npz、glm4 读 3151 npz、SMOKE 读 3152 smoke npz（1.2s 全链）。解剖对象=S1_s7 test 行 B4@读出层残差 R（147×D），k* 层同构对照。
- 四类锚全过（位级）：worst20 drift=0.0 ×3 模型；S2 vs 3152 v3 per_fold（4b/14b）；V5 drift=0.0（e_ks 0.9230/0.5186、e_ko 0.1490/0.1798）；M1@k* s7 margin −0.0040/−0.0432/−0.0289 与 3152 gates 位级一致。修 1 次（patch2）：门段 Mte/Yte/e_m1 误置于 t 循环内（2/3 行 M1=0 → margin 假爆 +2.71）；ALS/Rg 与 3152 逐字符一致由 debug 脚本独立证明（Rhat identical=True、同 Rg margin −0.0017）。
- 网格锚说明：3152 m1_grid 只含 KGRID 层（k 不含 k*=3），k* 网格值作为新记录（4b 0.00464/14b 0.03741/glm4 0.01836，ALS40it）。

### 四发现（重复强调）
1. **读出层残差载体三模型定量一致（ANOVA 嵌套增量 [类间,加性|类,交互(i,c),模板内]）**：4b [5.0,34.3,0.3,60.4]%、14b [6.2,35.9,0.0,59.0]%、glm4 [6.5,37.7,0.0,56.8]% —— **(i,c) 交互网格能量≈0（0.0–0.3%）**。读出层组合非线性 = 实体×类加性残余（34–38%）+ 模板内高秩散布（57–60%），不存在可被 rank-5 网格捕获的交互结构 → **3152 "读出层 rank10 顶格"之谜闭合：rank 越高越好的原因是残差根本没有网格坐标**。
2. **谱指纹跨模型一致（主验收过）**：ANOVA 份额 Pearson 0.999/0.995/0.998、spec_cell 谱形（σj/σ1 top10）0.968/0.991/0.977 —— 3 对全 ≥0.8 门 → 死线未触发，G1 保住。k* 对照谱尖锐（4b σ2/σ1=0.126 vs 读出层 0.615）。
3. **M1 rank-5 因子列空间 ≈ Pc(k\*)（"k* 交互=类子空间"第 4 次独立确证）**：主角谱 top1 0.979/0.889/0.977（vs Pc(k*)）vs 0.142/0.335/0.378（vs Pc(KOUT)）——三模型机制层/读出层二分再确证。
4. **worst20 失败模态分布模型特异**：glm4 错配主导（13/20；水果 S2=1.223 仍最差）、4b 散布主导（M3 8/20，E_g 0.27–0.42）、14b 均衡（难类 7+网格特异 7；颜色 S2=1.011）。样本级 Jaccard 0.14–0.29（对级最大 0.35）。覆盖率门 partial：cov_mean 0.83、4b 0.60<0.8（M3 定义 E_g<0.5 结构不可归类，其 proj_pcko 0.01–0.60 混合=非类、非网格、非模板均值的第 4 种散布）。

### 锚
4b res **c222c7ff** seal dad158d2；14b res **a0049e59** seal a600c4b5；glm4 res **18f2ef01** seal d36c7627；summary res **b9846738** seal 8236f974（execution 95541415）。ledger n=**290**。产物 `phase3153\\g1p3_failure_mode_anatomy\\{qwen3-4b,qwen3-14b,glm4,summary}\\`。runtime 51.6/105.6/88.4/0.0s（全零 GPU）。

### 预注册 Phase 3154：G2-P1 多关系族与算子可分离性（K2 死线）
(1)新面板采集（GPU）：3 关系族（是-a is-a / 有-a has-a / 制成-of made-of）×(实体×类)×3 模板，qwen3-4b + qwen3-14b + glm4-9b 全隐层采集（3151 collect.npz 协议复用）；(2)K2 检验：h_ℓ(i,c) 分解为 W_ℓ v_i + φ_ℓ(c) 的可分离性（实体方向响应是否关系无关、类方向是否关系无关；双向消融+子空间角）——**死线 K2：φ_ℓ(c) 与 W_ℓ 不可分离（交互份额>50%）→ 弃"条件门"独立结构**；(3)held-out 关系泛化门：2 关系训练 → 第 3 关系预测（err ≤ 1.5× in-relation，三模型）；(4)验收：held-out 门 3 模型 + K2 份额表；若 held-out 全败 → G2 降级描述学。GPU 预算 ~25min（3 模型×738×3 行推理）。
"""
if "## Phase 3153: 失败模态解剖" not in memo_txt:
    with io.open(MEMO, "a", encoding="utf-8") as f:
        f.write(sec)
    out.append("memo: appended 3153 section")
else:
    out.append("memo: 3153 section already present, skip")
# 落盘复核
memo_txt2 = io.open(MEMO, encoding="utf-8").read()
assert "## Phase 3153: 失败模态解剖" in memo_txt2 and "预注册 Phase 3154" in memo_txt2
out.append("memo: on-disk verify ok (len=%d)" % len(memo_txt2))

# ============ 3. daily ============
dln = ("3153 G1-P3 失败模态解剖闭环：全零 GPU 三模型；主判决 fingerprint_consistent_coverage_partial；"
       "ANOVA 读出层交互≈0 闭合 3152 rank10 之谜；M1 因子≈Pc(k*) 第 4 确证；ledger n=290；"
       "修 1 次（M1 门段循环结构 patch2）；3154（G2-P1 多关系族 K2）已预注册。\n")
if os.path.exists(DAILY):
    dt = io.open(DAILY, encoding="utf-8").read()
    if "3153" in dt:
        out.append("daily: 3153 already noted, skip")
    else:
        io.open(DAILY, "a", encoding="utf-8").write(dln)
        out.append("daily: appended")
else:
    io.open(DAILY, "w", encoding="utf-8").write(
        "# 2026-10-01\n\n" + dln)
    out.append("daily: created+appended")

# ============ 4. MEMORY.md 压缩重写 ============
NEW_MEMO = """# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）+ qwen3-14b + glm4-9b。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md；多线 MEMO 在 research\\{claude,deepseek,gemini,gpt5}\\docs\\；deepseek 线 CRLF+BOM。
- 脚本 tests\\glm5\\phase{N}_*.py；产物 tests\\glm5\\result\\rdc_query_construction_20260913\\phase{N}\\{name}\\；临时 tests\\gpt5_temp\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json（measurements 列表，n=290）。

## 强制流程
1. 闭环：execution 冻结→SMOKE→正式→判决→seal→Ledger→MEMO→daily→MEMORY→present→磁盘复核。
2. MEMO 标题简短：`## Phase {N}: 短标题（线-P）[hh:mm]`；纠错 append 不回改。
3. 预审计：静态对齐 result 键与断言；禁跨样本集 1e-9 断言；软门优先硬 assert。
4. 非 Phase 节有被外部进程删除风险 → 非 Phase 交付必须留 standalone 文档；MEMO 追加后 Grep 复核。
5. 幻影编辑缺陷：关键写入一律 Python patch + Grep/Read 落盘复核；禁止 bash 内联 python -c（反引号/反斜杠被吃）。

## 统计纪律（3102 五原则 + 3151 教训）
- cos 与 relative-L2 双报；门先验证可达域；秩相关平均秩；JVP 准≠映射混；次模=条件二阶差分。
- paired 门 margin=err_cand−err_B4，**负=候选优**，pass 当 m≤−2×MDE。
- numpy2：batch solve 需 [..., None] 尾维；one-hot+截距共线 → λ≥1e-3。

## 机制链状态（G1 线，3151–3153）
- **3151**：interaction_pair_generalizes_at_k3_only——glm4 M1@k3 三 seed 胜 B4（−0.029/−0.032/−0.025）；读出层 M1 全败；B4@k39=0.390。
- **3152**：k1_not_triggered_b4_additive_at_kstar_operator_line_kept（above=1/3）。三模型 k* 层位定律：B4@k* 0.046/0.008/0.081 vs 读出层 0.39/0.33/0.40（恶化 4.9–42×，argmin=k0）；M1 修复窗 k≤10–11、k≥15 过拟合；M2（嵌入中介）三模型再败=嵌入行实现 2 次否证；V5 类子空间能量 k* 高/读出层低（92/52/88 vs 15/18/12%）；glm4k1 双位级锚 drift 0（B4@k39、M1 margins）。
- **3153**：fingerprint_consistent_coverage_partial——读出层残差 ANOVA [类间5–6.5/加性34–38/**交互(i,c)0.0–0.3**/模板内57–60]% 三模型一致（fp_corr 0.995–0.999、spec 0.968–0.991）→ **交互网格能量≈0=读出层无低秩交互结构，rank10 顶格之谜闭合（残差无网格坐标）**；M1 因子列空间≈Pc(k*) 主角 0.979/0.889/0.977 vs Pc(KOUT) 0.14/0.34/0.38=k* 交互=类子空间第 4 确证；worst20 模态模型特异（glm4 错配 13/20、4b 散布 8/20、14b 均衡）；cov partial（4b 0.60）。锚：4b c222c7ff/14b a0049e59/glm4 18f2ef01/summary b9846738，M1 margins 与 3152 位级。
- 历史（3105–3150 压缩）：真值=记录级一阶矩广播；端口类 5 次确证（读出不读行坐标）；写入头组 L20–28 主写/L32 擦除；层位谱分离 iinj 峰 L17–19 vs cinj 平台 L23–27；3150 P0 制度冻结（metric_dict/counterexample_grep/ledger v3）；载体解码 NEG=多语言词素碎片/POS=文档骨架；3151 判决符号=rev-3151b（负=优）。

## 路线裁决（3150 方案冻结）
- 层相关写入算子族 {W_ℓ} + 层无关读出算子 P；端口替换取代方向去除（3148：减法非法）。
- 死线：K1 3模型未见组合（3152 已判 not_triggered）；K2 φ_ℓ(c) 与 W_ℓ 不可分离→弃条件门（**3154 检验**）；K3 top-50 覆盖<30%→弃单坐标。
- 排期：3154–3156 G2 关联机制（多关系族+held-out）→ 3157–3159 G3 自回归 → 3160 整合 v5.4；G4-SAE 3161+ 候选。
- 每条主线第一 Phase 必含"可能否证整条主线"的判决。

## 本机环境缺陷（Windows）
- bash shim 劣化：rm/ls/grep/cat 坏→python 文件化；-c stdout 丢→写文件再 Read；反引号/反斜杠被吃→chr(96)/避免内联。
- PowerShell stdout 捕获不稳；关键写入后 Grep 复核真实磁盘；GPU 测试逐模型防 OOM。
- 项目 Python：D:\\AI2050\\Ai2050-OpenOne\\.venv\\Scripts\\python.exe。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出；关键发现重复 3 次。

## 下一步
- **Phase 3154（G2-P1）已预注册**：3 关系族（is-a/has-a/made-of）×(i,c)×3 模板面板（GPU 采集，qwen3-4b+14b+glm4）；K2 可分离性检验（φ_ℓ(c) vs W_ℓ，交互份额>50%→弃条件门）；held-out 关系泛化门（2 训 1 测，err≤1.5×in-relation）。3155：G2-P2 视 3154 结果。复用资产：3151/3152 collect 协议、phase3152 b4_fit/als_complete、p3152_closeout 结构。
"""
io.open(MEMORY, "w", encoding="utf-8").write(NEW_MEMO)
out.append("memory: rewritten size=%d bytes" % os.path.getsize(MEMORY))

# ============ 5. 幂等自查 ============
led2 = json.load(io.open(LEDGER, encoding="utf-8"))
assert any(m.get("phase") == 3153 for m in led2["measurements"])
assert len(led2["measurements"]) == 290
out.append("final: ledger n=%d" % len(led2["measurements"]))
io.open(os.path.join(ROOT, "tests", "gpt5_temp", "p3153_closeout_out.txt"),
        "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
