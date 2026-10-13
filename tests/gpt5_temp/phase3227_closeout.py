# -*- coding: utf-8 -*-
"""Phase 3227 closeout: ledger #379 + MEMO + daily + MEMORY."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


TODAY = '2026-10-11'
P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3227'
      r'\g5g26_e3_intervention_design')
RES_F = sha8_file(os.path.join(P5, 'result.json'))
LOG_F = sha8_file(os.path.join(P5, 'run_log.txt'))
FZ_SHA = '82dbb8b1'
DOC_C, DOC_F = 'a883c4da', '32968738'

LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3227', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3227') else None
already = (bako0 is not None and len(disk0['measurements']) == 379
           and disk0['measurements'][378].get('phase') == 3227)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '62b2d7d6', prev_post
if already:
    raw = open(LP + '.bak_3227', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 378

entry = {
    "phase": 3227,
    "name": "g5g26_e3_intervention_design",
    "line": "G",
    "date": TODAY,
    "model": "design-only (no model, no observation, no GPU)",
    "verdict": "e3_intervention_design_registered|DESIGN_SEALED_NOT_OBSERVED"
               "|GATES_GD_flip_GD_dose_GD_spec_GD_amp|CHAIN_3228_compute_3229_decision"
               "|CACHES_x6_REVERIFIED|LEDGER_UNTOUCHED",
    "evidence_level": "design_only",
    "model_scope": "qwen3-4b + qwen3-14b (2/3); glm4 deferred (W_U absent)",
    "prereg_id": "G5-G26",
    "superseded_by": None,
    "detail": (
        "G5-G26 criterion-2 (sufficiency) E3 intervention DESIGN arm "
        "(zero GPU, no observation). Freeze " + FZ_SHA + " before any "
        "observation. Sealed design " + DOC_C + "/" + DOC_F + ": "
        "minimal-intervention protocol licensing E2 -> E3 per 3223 "
        "completion criterion_2 — patch relation direction u_R at the "
        "FINAL layer of cached residuals, evaluate EXACT post-patch "
        "logits l' = W_U @ ln_f(h + a*u) on CPU (ground truth at any "
        "alpha; A1 linearity licenses the linear INTERPRETATION at gate "
        "scale 0.5 only; alpha=2 doubles as nonlinearity-onset probe). "
        "Judgment operationalization pinned to on-disk anchors: 3154 "
        "column schema (g/l/s/ti/ci/ho) + 3169 marg relation-readout "
        "cache; token-set distinctness inherited from 3169 D2. "
        "Pre-registered gates: GD_flip (flip-rate minus max-null >= 0.15 "
        "on >=3/4 folds x2 models), GD_dose (monotone in alpha "
        "{0.5,1,2}), GD_spec (cross-patch <= 1/3 matched), GD_amp (u* "
        "amplitude rank-correlates with per-class collapse >= 0.6, n=4 "
        "small-n scope). Taxonomy: E3_PASS iff GD_flip and GD_dose and "
        "GD_spec; FAIL -> F ledger. E3 taxonomy honesty: in-protocol "
        "causality only (last-layer residual injection bypasses upstream "
        "computation — recorded verbatim, no full-regeneration claim). "
        "Upgrade targets on E3_PASS (3229): FTR-22 family + FTR-26 -> E3 "
        "via upgrade_log #11 (registry v1.30). Honest log: SMOKE run1 "
        "glm4 cache path guess wrong (actual glm4\\collect.npz); run2 "
        "CLEAN; FULL CLEAN. res " + RES_F + " / seal " + LOG_F + "."),
    "chain": {
        "res_sha8": RES_F,
        "seal_sha8": LOG_F,
        "freeze_sha8": FZ_SHA,
        "design": {"content_sha8": DOC_C, "file_sha8": DOC_F},
        "caches_reverified": {"p3154": ["bb99438f", "6d142b9d", "cf2659f1"],
                              "p3169": ["36eb4ff0", "97f98575", "9cc3f8b4"]},
        "ledger_pre": prev_post,
        "note": "design-only; no registry/gap change; compute arm 3228, "
                "decision arm 3229",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 379
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3227', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3227', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:378]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post \
    and len(d2['measurements']) == 379
print('ledger #379 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3227 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3227 \xe2\x80\x94 G5-G26' not in mraw:
    ADD = """

## Phase 3227 — G5-G26 判据二（充分性）E3 干预设计草案（2026-10-11）

**执行模式**：design-only 零 GPU 无观测——冻结 `82dbb8b1` 先于任何观测；设计档 `e3_intervention_design_v1.json` 内容 **`a883c4da`**（文件 `32968738`）封存。

### 数据基础（D1–D3，在盘锚全过）
- **缓存 ×6 重验**：3154（bb99438f/6d142b9d/cf2659f1）+ 3169（36eb4ff0/97f98575/9cc3f8b4）；W_U 分片（qwen 对）+ u* npz `ac5d4d74`；A1 线性许可（cos_S ≥0.99999 / band ≤0.0052 @α≤0.5）。
- **关键设计洞察**：patch 后 logits `ℓ' = W_U·ln_f(h + α·u_R)` 在 CPU 上**任意 α 精确**（ln_f 直接求值，RMSNorm 非线性在函数内）——A1 许可的只是「线性解释」限于门刻度；α=2 行兼作非线性 onset 探针。
- **判断操作化钉死在盘锚**：seen 面板 3154 列 schema（g/l/s/ti/ci/ho）+ OOV 面板 3169 `marg` 关系 readout 缓存；token 集区分性继承 3169 D2 门；方向 u_R 按封存 T2 配方（3194 module import，设备锚 drift==0 已于 3225 再证）。

### 预注册门（D4，upgrade_log #11 候选）
- **GD_flip**：flip_rate(+u_R, α=1) − max(null) ≥ 0.15，≥3/4 折 × 2 模型（null = 范数匹配随机 ×10 种子 + 正交化对照）；
- **GD_dose**：flip_rate 对 α∈{0.5,1,2} 单调不减 ×2 模型；
- **GD_spec**：交叉 patch（u_isa→hasa 判断，反之亦然）翻转率 ≤ 匹配率 1/3 ×2 模型；
- **GD_amp**：u* 幅度（行投影）与 3169 逐类谱外崩塌 rank-corr ≥ 0.6 ×2 模型（n=4 小样本 scope limit 在案）。
- **分类学**：E3_PASS ⇔ GD_flip ∧ GD_dose ∧ GD_spec；部分 → E3_DIRECTIONAL；否则 FAIL 入 F 账本。升级目标（3229 决策）：FTR-22 族 + FTR-26 → E3（upgrade_log #11，registry v1.30）。

### E3 taxonomy 诚实边界（verbatim 入档）
协议内因果（G 线 E3 级定义）：末层残差注入绕过上游计算——**不作全再生（full regeneration）声明**。

### 诚实日志与预注册
SMOKE run1 glm4 缓存路径猜测错误（实际 `glm4\\collect.npz`）→ run2 CLEAN + FULL CLEAN；res `f374377b` / seal（run_log）；账本臂内未动（n=378，`62b2d7d6`）→ **#379**（治理条目，rollback 等价）。

### 预注册 Phase 3228（两支候选）
- (a) **G5-G26 计算+封存臂**（零 GPU 计算）：按本设计冻结种子/α/折，精确翻转矩阵 + null 对照 + 幅度相关封存 npz——默认；
- (b) **非线性扩程先行臂**（α∈{1,2,4} onset 单独先测，输出并入 3228 设计注记）。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3227' not in draw:
    dadd = ("\n\n### Phase 3227 G5-G26 判据二 E3 干预设计草案收官（2026-10-11）\n"
            "- design-only 零 GPU 无观测：冻结 82dbb8b1；设计档 a883c4da/32968738（criterion_2：方向 patch → 判断翻转 + u* 幅度预测崩塌）。\n"
            "- 关键洞察：patch 后 logits ℓ'=W_U·ln_f(h+α·u) CPU 任意 α 精确（A1 线性许可仅限解释@门刻度；α=2 兼 onset 探针）；判断操作化钉 3154 列 schema + 3169 marg readout。\n"
            "- 预注册门：GD_flip/GD_dose/GD_spec/GD_amp（taxonomy：E3_PASS ⇔ flip∧dose∧spec；FAIL 入 F 账本）；升级目标 FTR-22 族+FTR-26→E3（upgrade_log #11，registry v1.30）；E3=协议内因果（末层注入，不作全再生声明）。\n"
            "- 诚实日志：SMOKE run1 glm4 缓存路径猜错 → CLEAN；res f374377b；ledger #379（rollback 等价）。预注册 3228：(a) 计算+封存臂；(b) 非线性扩程先行臂。\n")
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
- **3228 已预注册**：(a) **G5-G26 计算+封存臂**（零 GPU 计算）：按 3227 设计冻结种子/α/折，精确翻转矩阵 + null 对照 + u* 幅度相关封存 npz；(b) 非线性扩程先行臂（α onset 先测）。
- 3227 收官：E3 干预设计草案（res %s，freeze 82dbb8b1，设计档 a883c4da/32968738），ledger #379 post %s；门 GD_flip/GD_dose/GD_spec/GD_amp 预注册（E3_PASS ⇒ FTR-22 族+FTR-26 升 E3，upgrade_log #11）；criterion_2 协议冻结。
- 3226 收官：A2 三门评估+registry 决策（res 48222035，freeze 1608c23d），ledger #378 post 62b2d7d6；**FTR-25+FTR-26 升 E2_behavioral（upgrade_log #10）**；registry v1.29 85b8dc1f；GAP-5 非线性定位（gap v1.7 4f45fda0：pre-softmax 线性、非线性在 post-softmax）。
- 3225 收官：A1 计算+封存（npz ac5d4d74，freeze 98d7fbd1，res ea1a0585），ledger #377；cos_full=1.0000 / band ≤0.0052 / 金标位移≈0；方案 A 权重入库（4B shard1+3、14B shard8）。
- 3224/3223：就绪盘点 0590e53b（#376）；行为带设计档 87e51bcd（#375）。
- 3222 收官：G5-G24 轮次小结（res a1de6a81），ledger #374；总结档 52ba5d3a。
- 3221/3220/3219/3218：UI 冻结+链档 ca1b24ca（#373）；appendix 锚复读（#372）；UI v1.10 def433e0（#371）；跨线施加就绪（#370，N 线 ready / APPX-5 not-ready）。
- 挂账不变：APPX-5 施加（重基线后）；N 线施加（ready）；5 appendix GPU/跨线项；UI 线冻结于 v1.10；**水果类/K4 列表删除待用户确认**。

""" % (RES_F, new_post)
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
