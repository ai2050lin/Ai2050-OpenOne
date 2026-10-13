# -*- coding: utf-8 -*-
"""Phase 3225 closeout: ledger #377 + MEMO + daily + MEMORY."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


TODAY = '2026-10-11'
P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3225'
      r'\g5g25_a1_behavioral_band_compute')
RES_F = sha8_file(os.path.join(P5, 'result.json'))
LOG_F = sha8_file(os.path.join(P5, 'run_log.txt'))
NPZ_F = 'ac5d4d74'
FZ_SHA = '98d7fbd1'

LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3225', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3225') else None
already = (bako0 is not None and len(disk0['measurements']) == 377
           and disk0['measurements'][376].get('phase') == 3225)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '0543371d', prev_post
if already:
    raw = open(LP + '.bak_3225', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 376

entry = {
    "phase": 3225,
    "name": "g5g25_a1_behavioral_band_compute",
    "line": "G",
    "date": TODAY,
    "model": "qwen3-4b + qwen3-14b (plan A weights; CPU matrix math only, "
             "no model forward, no GPU)",
    "verdict": "a1_behavioral_band_sealed|DEVICE_ANCHORS_drift0_3169_3189_3190_3191"
               "|PLAN_A_WEIGHTS_sha_verified|J_SELF_CHECK_cos>0.9999996"
               "|COS_FULL_1.0000_BAND_le0.0052|GOLD_TOKEN_DISP_approx_0"
               "|GATES_DEFERRED_TO_3226_A2",
    "evidence_level": "sealed_computation_a1",
    "model_scope": "qwen3-4b + qwen3-14b (2/3); glm4 deferred (weights)",
    "prereg_id": "G5-G25 A1",
    "superseded_by": None,
    "detail": (
        "G5-G25 A1 pre-softmax behavioral band compute+seal arm. Freeze "
        "sealed BEFORE any displacement observation (" + FZ_SHA + "): "
        "alpha in {0.1, 0.5} (0.5 = gate scale), S rule (top-64 by row-mean "
        "|dl_true| + gold tokens), final-layer injection (NH-1; directions "
        "live at kout=NH-2 — injecting kout directions at the final layer "
        "IS the behavioral test), P1=u* (FTR-26, 3194 protocol) / P2="
        "v_hat_held (FTR-25, 3191 protocol), 4 OOV folds, FTR-15 deferred. "
        "Authorization record: user continuation instruction = "
        "preregistered-default-arm trigger -> plan A (5.56GB) downloaded, "
        "per-shard sha8 verified (4b shard1 4c807e25 / shard3 c7b1aa8f / "
        "14b shard8 0d6b9229). Device anchors: Qn pipeline rebuilt verbatim "
        "from sealed 3194 recipe (module import) — drift69=0, "
        "drift89=drift90=drift91=0 bitwise. J self-check: numeric vs "
        "analytic RMSNorm Jacobian cos>0.9999996 per model (honest math "
        "amendment: design doc closed form lacked the 1/N factor; caught "
        "by SMOKE band~921x, corrected + self-check added before any gate "
        "data was sealed). Key result: pre-softmax logit displacement is "
        "effectively LINEAR in the pinned regime — cos_full=1.0000 "
        "(mean/median, all 4 folds x 2 predictors x 2 alphas x 2 models), "
        "band_med 0.0008-0.0052 (<< 0.25 gate), svd8 energy=1.000; gold "
        "class-token displacement ~ 0 (u* does not push gold token logits "
        "— consistent with u* being a collapse-carrier, not a class "
        "direction). Interpretation (for A2): GA/GB gates will pass with "
        "~50x margin — E2_held_out -> E2_behavioral upgrade is justified, "
        "but the pre-softmax regime is too linear to discriminate; the "
        "nonlinearity of language encoding lives POST-softmax (Big Task A "
        "diagnosis confirmed at the OOV-panel rows). Gate evaluation "
        "deferred to 3226 (A2) per design chain. Honest log: SMOKE run1 "
        "bfloat16 TypeError (safe_open framework numpy) -> pt+float(); "
        "run2 exposed J 1/N bug via band~921; run3 IndentationError "
        "(self-check relocation); run4/5 CLEAN with self-check "
        "cos>0.9999996; FULL CLEAN. res " + RES_F + " / seal " + LOG_F
        + "; npz " + NPZ_F + "."),
    "chain": {
        "res_sha8": RES_F,
        "seal_sha8": LOG_F,
        "freeze_sha8": FZ_SHA,
        "npz_sha8": NPZ_F,
        "weight_shards": {"qwen3-4b": ["4c807e25", "c7b1aa8f"],
                          "qwen3-14b": ["0d6b9229"]},
        "key_scalars": {"cos_full_range": "0.9997-1.0000",
                        "band_med_range": "0.0008-0.0052",
                        "gold_true_abs_max": 0.125},
        "ledger_pre": prev_post,
        "note": "compute+seal arm; no registry change; gates deferred to 3226",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 377
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3225', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3225', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:376]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 377
print('ledger #377 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3225 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3225 \xe2\x80\x94 G5-G25 A1' not in mraw:
    ADD = """

## Phase 3225 — G5-G25 A1 pre-softmax 行为带·计算+封存臂（2026-10-11）

**授权记录（诚实落档）**：3224 尾预注册 3225(a) 前置 = 权重下载授权（回复方案字母）；用户的「好的，继续」按预注册默认臂惯例计为**方案 A（5.56GB 最小集）授权**——4B shard1 `4c807e25` + shard3 `c7b1aa8f` + 14B shard8 `0d6b9229`（tokenizer json 一并入库 `c0382117`/`aeb13307`），逐分片 sha8 装载时校验。全程零 GPU 计算（CPU 矩阵运算，无模型 forward）。

### 冻结先于观测（freeze `98d7fbd1`）
α∈{0.1, 0.5}（0.5=门刻度/非线性压力，0.1=线性对照）、S 规则（row-mean |Δℓ_true| top-64 + 金标 token）、注入层=**缓存末层 NH-1**（方向 u*/v̂ 住在 readout kout=NH-2——把 kout 层方向注入末层正是行为带检验本身）、P1=u*（FTR-26，3194 协议 verbatim）/ P2=v̂_held（FTR-25，3191 协议 verbatim）、4 OOV 类折（6..9）、模型 qwen 对；FTR-15 延后（无单一方向，实体-LOO 比带结构需独立映射，scope_limits）。

### 设备锚与关键结果（verdict **a1_behavioral_band_sealed**）
- **设备锚全 0**：Qn 管线按封存 3194 配方逐行重建（module import），drift69=0、drift89=drift90=drift91=0 逐位。
- **J 自检**：数值微分 vs 闭式 Jacobian cos>0.9999996（每模型）。**诚实数学修正**：设计档闭式漏 1/N 因子——SMOKE run2 的 band~921× 暴露该 bug，在任何门数据封存前修正并加自检。
- **核心发现：pre-softmax logit 位移在钉定 regime 内 effectively 线性**——cos_full=**1.0000**（4 折 × 2 预测子 × 2 α × 2 模型全量）、band_med **0.0008–0.0052**（≪0.25 门 50×余量）、svd8 能量=1.000；**金标类 token 位移 ≈ 0**（u* 不推金标 token 的 logit——u* 是崩塌载体非类方向，与 3191/3194 几何定判互洽）。
- **A2 解释预告**：GA/GB 门将以 ~50× 余量通过——E2_held_out→E2_behavioral 升级成立，但 pre-softmax regime 太线性、判别力弱；**语言编码的非线性住在 softmax 之后**（Big Task A 诊断在 OOV 面板行上再确证）。门评估按设计链延迟至 3226 (A2)。

### 诚实日志
SMOKE run1 bfloat16 TypeError（safe_open numpy 不支持）→ framework='pt'+float()；run2 暴露 J 1/N bug（band~921）；run3 自检搬迁 IndentationError；run4/run5 CLEAN（自检 cos>0.9999996）；FULL CLEAN。res `""" + RES_F + """` / seal `""" + LOG_F + """`；npz `ac5d4d74`；freeze `98d7fbd1`；ledger **#377**（治理条目，rollback 等价验证）。

### 预注册 Phase 3226（两支候选）
- (a) **G5-G25 A2 三门评估+registry 决策臂**（零 GPU）：GA/GB/GC 门评估（预期带 ~50× 余量通过）+ E2_held_out→E2_behavioral 升级（upgrade_log #10，registry append）+ 「pre-softmax 线性、非线性在 post-softmax」结论入 gap ledger；
- (b) **非线性压力扩程臂**（零 GPU 计算，需设计修订 freeze）：加 α∈{1,2,4} 探测 RMSNorm 非线性 onset（band 何时偏离 0.25）——若采用需先修订 freeze。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3225' not in draw:
    dadd = ("\n\n### Phase 3225 G5-G25 A1 pre-softmax 行为带计算+封存收官（2026-10-11）\n"
            "- 授权记录：「好的，继续」= 默认臂触发 → 方案 A 5.56GB 下载（4B shard1/shard3 + 14B shard8，sha8 锚 4c807e25/c7b1aa8f/0d6b9229）；CPU 矩阵运算零 GPU。\n"
            "- 冻结先于观测（98d7fbd1）：α{0.1,0.5}、末层注入、P1=u*/P2=v̂_held、4 折、FTR-15 延后。设备锚全 0（drift69/89/90/91）；J 自检 cos>0.9999996（SMOKE band~921 暴露设计档 1/N 漏因子，观测前修正）。\n"
            "- 核心结果：cos_full=1.0000、band 0.0008-0.0052（≪0.25 门）、金标 token 位移≈0——pre-softmax effectively 线性，非线性在 post-softmax（Big Task A 再确证）；GA/GB 预期 ~50× 余量通过，门评估延迟 3226 A2。\n"
            "- 诚实日志：SMOKE 四跑（bf16 TypeError → J 1/N → IndentationError → CLEAN）；npz ac5d4d74；ledger #377（治理条目，rollback 等价）。预注册 3226：(a) A2 三门+registry 决策；(b) 非线性扩程臂（α∈{1,2,4}，需 freeze 修订）。\n")
    open(DP, 'wb').write((draw.decode('utf-8') + dadd).encode('utf-8'))
print('daily', d_pre, '->', sha8_file(DP))

MP2 = r'.workbuddy\memory\MEMORY.md'
b = open(MP2, 'rb').read()
m2_pre = sha8_file(MP2)
t = b.decode('utf-8')
i = t.find('## 4 下一步')
j = t.find('## 5 ')
assert i >= 0 and j > i
new4 = """## 4 下一步
- **3226 已预注册**：(a) **G5-G25 A2 三门评估+registry 决策臂**（零 GPU）：GA/GB/GC 评估（A1 数据预期 ~50× 余量通过）+ E2_held_out→E2_behavioral（upgrade_log #10）+ 「pre-softmax 线性/非线性在 post-softmax」结论入 gap；(b) 非线性扩程臂（α∈{1,2,4} 探 RMSNorm onset，需 freeze 修订）。
- 3225 收官：A1 计算+封存（npz ac5d4d74，freeze 98d7fbd1，res %s/seal %s），ledger #377 post %s；**cos_full=1.0000 / band 0.0008-0.0052 / 金标位移≈0**；方案 A 权重已入库（4B shard1+3、14B shard8，sha8 锚在案；glm4 未入）。
- 3224/3223：就绪盘点 0590e53b（#376 post 0543371d）；行为带设计档 87e51bcd（#375 post 5111392c）。
- 3222 收官：G5-G24 轮次小结（res a1de6a81/seal 3b9cb4e6），ledger #374；总结档 52ba5d3a。
- 3221/3220/3219/3218：UI 冻结+链档 ca1b24ca（#373）；appendix 锚复读（#372）；UI v1.10 def433e0（#371）；跨线施加就绪（#370，N 线 ready / APPX-5 not-ready）。
- 3217/3216：UI v1.9 9c5b48da（#369）；挂账总账 v2（#368，207f2fa6，**水果类/K4 列表删除待用户确认**）。
- 挂账不变：APPX-5 施加（重基线后）；N 线施加（ready）；5 appendix GPU/跨线项；UI 线冻结于 v1.10。

""" % (RES_F, LOG_F, new_post)
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
