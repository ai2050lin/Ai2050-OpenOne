# -*- coding: utf-8 -*-
"""Phase 3228 closeout: ledger #380 + MEMO + daily + MEMORY (self-check on disk readback)."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


TODAY = '2026-10-11'
P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3228'
      r'\g5g26_e3_compute_seal')
RES_F = sha8_file(os.path.join(P5, 'result.json'))
LOG_F = sha8_file(os.path.join(P5, 'run_log.txt'))
FZ_SHA = 'c51bfea5'          # FULL freeze (v3 config)
FZ_SMOKE = 'ec0f0ccb'        # SMOKE v3 freeze
# superseded in-run history (honest log)
FZ_V1, FZ_V2, FZ_FULLV1 = 'aae60c9d', '5dd8b4f6', 'c076d6f6'
RES_FULLV1 = 'ef55a204'
NPZ_SHA = 'a844d985'
RES_C = '99693298'

LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3228', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3228') else None
already = (bako0 is not None and len(disk0['measurements']) == 380
           and disk0['measurements'][379].get('phase') == 3228)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'f21108e7', prev_post
if already:
    raw = open(LP + '.bak_3228', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 379

entry = {
    "phase": 3228,
    "name": "g5g26_e3_compute_seal",
    "line": "G",
    "date": TODAY,
    "model": "CPU matrix math only (zero GPU); qwen3-4b (LOCAL weights) + qwen3-14b",
    "verdict": "e3_compute_sealed|matched_flip_0_all_folds_x2models_x2R|null_flip_0"
               "|u_isa_ANTIDIRECTIONAL_dose_monotone|u_hasa_weak_7x|GD_amp_rho_0.40_0.80"
               "|C9_MAP_AND_PROVENANCE_CORRECTION|C6v3_marg_exact_cos_1.0|GATES_TO_3229",
    "evidence_level": "compute_only (gates deferred to 3229)",
    "model_scope": "qwen3-4b + qwen3-14b (2/3); glm4 deferred (W_U absent)",
    "prereg_id": "G5-G26",
    "superseded_by": None,
    "detail": (
        "G5-G26 E3 intervention compute+seal arm (zero GPU, default (a) per "
        "3227 design a883c4da). Sealed exact flip matrices + null controls + "
        "amplitude correlations: npz " + NPZ_SHA + ", res " + RES_C + ", "
        "freeze " + FZ_SHA + " (SMOKE " + FZ_SMOKE + "). "
        "IN-RUN HONEST HISTORY: freeze v1 " + FZ_V1 + " / v2 " + FZ_V2 + " / "
        "FULL v1 (freeze " + FZ_FULLV1 + ", res " + RES_FULLV1 + " dirty) all "
        "SUPERSEDED in-phase — SMOKE exposed the pre-registered C6 marg gate "
        "(0.90 blanket) failing; bf16-emulation arbitration (fp32-vs-bf16emul "
        "0.9927, both-vs-cached 0.55) ruled out machinery bug and exposed "
        "C9: the cached slot NH-1 is the POST-NORM pre-logit state (HF final-"
        "norm append; raw last-layer output not cached) — the design map "
        "l'=W_U@ln_f(h+a*u) was a DOUBLE-NORM misread; corrected map is "
        "EXACTLY LINEAR l'=W_U@(h+a*u) (GAP-5 coherent: nonlinearity lives "
        "upstream). C9 also found a W_U PROVENANCE SPLIT: 4B caches were "
        "collected with LOCAL models/hf/qwen3-4b weights (shard1 328a91d3) "
        "which DIFFER from the hub Qwen3-4B-Base snapshot (4c807e25) used by "
        "3225 plan A; corrected arm loads LOCAL 4B weights, 14B hub shard8 "
        "(0d6b9229 == local, exact). Cross-phase note for 3229: 3225 A1 used "
        "hub 4B W_U on local residuals; A1 linearity conclusion (map "
        "property) unaffected, provenance caveat travels with registry "
        "v1.29 E2_behavioral upgrades. CORRECTED DEVICE GATE C6v3: baseline "
        "direct-map class readout vs sealed 3169 bf16 marg argmax agreement "
        "0.9963 (4B) / 0.9941 (14B), centered-cos 1.0000 both, gap>=0.5 "
        "agreement 1.0000 — readout machinery exact. SUBSTANTIVE NUMBERS "
        "(corrected map, both panels, both models): matched flip-toward-R "
        "rate 0.0000 on ALL seen folds (4 cut-folds x 2 models x 2 R) and "
        "null flip rates 0 (the superseded double-norm map had amplified "
        "noise flips 2-4%); u_isa dose ANTI-DIRECTIONAL monotone (mean "
        "dD=m_isa-m_hasa shift: 4b seen -0.071/-0.142/-0.284, oov "
        "-0.069/-0.139/-0.278; 14b seen -0.063/-0.127/-0.253, oov "
        "-0.056/-0.112/-0.223 at alpha 0.5/1/2), u_hasa ~7x weaker same "
        "sign — the T2 relation-level directions (ctx-contrast recipe "
        "v_R=unit_mean(dCn[rel])) do NOT act as levers toward their own "
        "relation on the behavioral judgment readout (J = sign "
        "mean-logit{'是一种'=101158} - mean-logit{'有'=18830}, token sets "
        "pinned from sealed prompt ids, D2 distinctness replicated); "
        "judgment has power (baseline d0 std ~3.8, 55/45 both sides). "
        "GD_amp data sealed: per-class u* amplitude vs per-class 3169-B "
        "collapse Spearman 0.40 (4b) / 0.80 (14b) — partial support, n=4 "
        "scope. All device anchors drift==0 (3157 tc, 3183 dv, 3169 per-"
        "seed E, 3189/3190/3191 scalars, u* bitwise). Gate application "
        "(GD_flip/GD_dose/GD_spec/GD_amp; E3 taxonomy: FAIL -> F ledger) "
        "DEFERRED to 3229 per design chain — expectation recorded honestly: "
        "GD_flip/GD_dose/GD_spec FAIL, GD_amp partial."),
    "chain": {
        "res_sha8": RES_F,
        "seal_sha8": LOG_F,
        "freeze_sha8": FZ_SHA,
        "freeze_smoke_sha8": FZ_SMOKE,
        "superseded_in_run": {"freeze_v1": FZ_V1, "freeze_v2": FZ_V2,
                              "full_v1_freeze": FZ_FULLV1,
                              "full_v1_res_content": RES_FULLV1},
        "npz_sha8": NPZ_SHA,
        "c6v3_marg_agreement": {"qwen3-4b": 0.9963, "qwen3-14b": 0.9941},
        "w_u_provenance": {"qwen3-4b": "LOCAL 328a91d3 (hub 4c807e25 is a "
                            "different revision)",
                           "qwen3-14b": "hub 0d6b9229 == local"},
        "ledger_pre": prev_post,
        "note": "compute+seal; GD_* application + F-ledger/registry action "
                "= 3229",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 380
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3228', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3228', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:379]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post \
    and len(d2['measurements']) == 380
print('ledger #380 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3228 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3228 \xe2\x80\x94 G5-G26' not in mraw:
    ADD = """

## Phase 3228 — G5-G26 E3 计算+封存臂（2026-10-11）

**执行模式**：零 GPU CPU 矩阵计算（默认 (a)，按 3227 设计 `a883c4da`）；SMOKE v3 冻结 `ec0f0ccb` / FULL 冻结 `c51bfea5`；npz **`a844d985`**、res **`99693298`**、verdict `e3_compute_sealed`（0 矛盾）。

### 相位内诚实史（三次重封，全部在 FULL 前或同相内）
- **C6 v1→v3**：预注册 0.90 marg 一致率 blanket 门 SMOKE 即 FAIL（0.8169）→ bf16 模拟仲裁（fp32-vs-bf16emul 0.9927，两者-vs-cached 0.55）排除机械 bug → **C9 揭示真因**：HF 约定下缓存槽 NH-1 是 **post-norm pre-logit 态**（final norm 循环后 append，raw 末层输出不入缓存），设计图 `l'=W_U·ln_f(h+αu)` 系**双重归一化误读**；修正图**精确线性** `l'=W_U·(h+αu)`（与 GAP-5 自洽：非线性在上游）。
- **C9 出处分裂**：4B 缓存采集用**本地** `models/hf/qwen3-4b`（shard1 `328a91d3`）≠ hub Qwen3-4B-Base（`4c807e25`，3225 方案 A 所用）；修正臂 4B 改载本地权重，14B hub shard8 `0d6b9229` ==本地（精确）。**跨相注记（交 3229）**：3225 A1 用 hub 4B W_U 投影本地残差——A1 线性结论（图性质）不受影响，出处告警随 registry v1.29 E2_behavioral 升级走。
- **C6v3 设备门（真门）**：基线直接图层读出 vs 3169 bf16 marg 缓存 argmax 一致率 **0.9963（4B）/0.9941（14B）**、centered-cos **1.0000**、gap≥0.5 一致率 1.0000——读出机械精确闭合。

### 实质数字（修正图，双面板 × 双模型）
- **matched flip = 0.0000**：4 cut-折 × 2 模型 × 2 方向全零；null 翻转亦 0（被废止的双重归一化图曾放大出 2–4% 噪声翻转——真线性图下读出稳定）。
- **u_isa 剂量反方向且单调**（dD = m_isa−m_hasa 均移）：4b seen −0.071/−0.142/−0.284、oov −0.069/−0.139/−0.278；14b seen −0.063/−0.127/−0.253、oov −0.056/−0.112/−0.223（α=0.5/1/2）；u_hasa 弱 ~7× 同号。**T2 关系级方向（ctx-对比配方 v_R）在其自身关系的行为判断读出上不是杠杆，isa 方向是反平行杠杆**。
- 判断 readout 有功效（基线 d0 std≈3.8，两侧 55/45；token 集 {「是一种」=101158} vs {「有」=18830}，封存 prompt id 钉定，D2 区分性复现）。
- **GD_amp**：逐类 u* 幅度 vs 3169-B 逐类崩塌 Spearman **0.40（4b）/ 0.80（14b）**——部分支持（n=4 scope）。
- 设备锚全 drift==0（3157 tc / 3183 dv / 3169 逐种子 E / 3189/3190/3191 / u* 位级）；几何 cos(u_isa,u*)=0.0434/0.0897、cos(u_hasa,u*)=0.1015/0.0676。

### 移交 3229（门评估 + registry 决策臂）
GD_flip/GD_dose/GD_spec 按封存矩阵 Apply——诚实预期 **FAIL → F 账本**（候选 F20：关系级方向 patch 反平行/无效于行为判断读出，与 3188 DECOUPLED、GAP-5 链条自洽）；GD_amp 部分支持（1/2 模型过 0.6）。E3 taxonomy：FAIL 不隐藏。

### 预注册 Phase 3229（两支候选）
- (a) **G5-G26 门评估 + registry 决策臂**（默认）：按 3227 D4 逐字应用 GD_* 于封存矩阵 → verdict + F 账本/counter_evidence 处置 + 3225 出处告记入档；
- (b) **关系对比方向效力臂**：直接用 3157 关系对比方向 dR = h(isa)−h(hasa)（非 ctx-对比配方的 v_R）重测 flip 效力——检验反方向结果是否为 T2 配方特有。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3228' not in draw:
    dadd = ("\n\n### Phase 3228 G5-G26 E3 计算+封存臂收官（2026-10-11）\n"
            "- 零 GPU CPU 矩阵计算（默认 (a)）：npz a844d985 / res 99693298 / freeze c51bfea5（SMOKE ec0f0ccb）；verdict e3_compute_sealed（0 矛盾）。\n"
            "- 相位内三次重封诚实史：C6 v1 0.90 blanket FAIL → bf16 仲裁排机械 bug → C9 揭示缓存槽 NH-1 = post-norm pre-logit 态（设计图系双重归一化误读）→ 修正图精确线性 l'=W_U·(h+αu)；C6v3 门 0.9963/0.9941（centered-cos 1.0000）。\n"
            "- C9 出处分裂：4B 缓存采集用本地权重（328a91d3）≠ hub（4c807e25）；修正臂 4B 改载本地；14B hub==local；跨相注记：3225 A1 hub-W_U-on-本地残差（线性结论不受影响，告警交 3229）。\n"
            "- 实质：matched flip 全 0（null 亦 0）；u_isa 剂量反方向单调（−0.14/−0.13@α1），u_hasa 弱 7×——T2 关系级方向非行为读出杠杆；GD_amp ρ 0.40/0.80 部分。GD_* 应用移交 3229（诚实预期 FAIL→F 账本）。\n"
            "- 预注册 3229：(a) 门评估+registry 决策（默认）；(b) 关系对比方向 dR=h(isa)−h(hasa) 效力臂。\n")
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
- **3229 已预注册**：(a) **G5-G26 门评估 + registry 决策臂**（默认）：GD_* 逐字应用于 3228 封存矩阵 → 诚实预期 FAIL → F 账本（F20 候选）+ GD_amp 部分（0.40/0.80）；(b) 关系对比方向 dR=h(isa)−h(hasa) 效力臂（检验反方向结果是否 T2 配方特有）。
- 3228 收官：E3 计算+封存（npz a844d985，res 99693298，freeze c51bfea5），ledger #380 post %s；**C9 相位内修正**：缓存槽 NH-1 = post-norm pre-logit 态（设计图双重归一化误读→修正图精确线性）；**4B W_U 出处分裂**（本地 328a91d3 ≠ hub 4c807e25，修正臂改载本地；3225 A1 出处告警交 3229）；matched flip 全 0、u_isa 反方向剂量、GD_amp 0.40/0.80。
- 3227 收官：E3 干预设计草案（res f374377b，设计档 a883c4da/32968738），ledger #379 post f21108e7；门 GD_flip/GD_dose/GD_spec/GD_amp 预注册。
- 3226/3225：A2 三门评估 FTR-25+FTR-26 升 E2_behavioral（registry v1.29 85b8dc1f，GAP-5 入册 gap v1.7 4f45fda0）；A1 计算+封存（npz ac5d4d74，res ea1a0585）。
- 挂账不变：APPX-5 施加（重基线后）；N 线施加（ready）；5 appendix GPU/跨线项；UI 线冻结于 v1.10；**水果类/K4 列表删除待用户确认**。

""" % new_post
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))

# self-check: disk readback of all five writes
d3 = json.load(open(LP, encoding='utf-8'))
assert len(d3['measurements']) == 380 and d3['ledger_sha256_8'] == new_post
assert d3['measurements'][379]['phase'] == 3228
assert b'## Phase 3228' in open(MP, 'rb').read()
assert b'### Phase 3228' in open(DP, 'rb').read()
t3 = open(MP2, 'rb').read().decode('utf-8')
assert '3228 收官' in t3 and new_post in t3
print('self-check: ledger/MEMO/daily/MEMORY disk readback OK; closeout done')
