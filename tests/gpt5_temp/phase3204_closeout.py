# -*- coding: utf-8 -*-
"""Phase 3204 closeout: ledger #356 append + MEMO append + daily + MEMORY.
Registry v1.26 (112a46ee) + atlas_v1_28 (cfc389a8) already sealed by main script.
"""
import hashlib
import json


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

TODAY = '2026-10-10'

# ---------- 1. ledger append (#356) ----------
LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
pre8 = sha8(raw)
old = json.loads(raw.decode('utf-8'))
n0 = len(old['measurements'])
assert n0 == 355, n0
old_blob = json.dumps(old, ensure_ascii=False, indent=1, sort_keys=True)
old_post = old['ledger_sha256_8']
assert old_post == 'df_placeholder' or isinstance(old_post, str)
# capture previous post before we overwrite
prev_post = json.load(open(LP, encoding='utf-8'))['ledger_sha256_8']

entry = {
    "phase": 3204,
    "name": "g5g1_ftr15_heldout_recheck",
    "line": "G",
    "date": TODAY,
    "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5g1_ftr15_heldout_recheck|REGISTERED|H1_3158_ratios_bitwise_x3|H2_3180_chain_clean_ratio_bitwise_maxband_0.0300_0.0278_0.0182_folds_16x3_same_class_drift0|H3_registry_anchors_bitwise_upgrade_entry|scope_limits_E2_held_out_annotation_append|registry_v1.25_3bb33a24_to_v1.26_112a46ee|atlas_v1_28_cfc389a8_784_fields_clean",
    "evidence_level": "E2_predictive_held_out_annotation",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-G1",
    "superseded_by": None,
    "detail": (
        "G5-G1 FTR-15 held_out recheck arm (zero GPU; sealed p3158 summary + p3180 "
        "held-out result + registry v1.25 re-read; default prereg arm (a)). H1: "
        "sealed 3158 budget ratios bitwise == pinned full-precision constants "
        "(1.0283477199188433 / 1.3316768638966026 / 1.2472595295618603) + content "
        "double lock (res fa5cca12 / seal e0c60629). H2: sealed 3180 held-out chain "
        "clean - ratio_full_recompute bitwise == 3158 ratios x3 (device drift 0 x3, "
        "ratio_sealed == ratio_full_recompute x3), max_log_band 4dp "
        "0.0300/0.0278/0.0182 < 0.25, folds 16 x3 all same_class with max "
        "ratio_train < 3.0, n_c_gt1 10/16/15, gates G1_device/G2_heldout True + "
        "G2_class 3.0 + G2_log_band 0.25 + content double lock (res f1689e94 / seal "
        "a3c1b53e). H3: registry v1.25 FTR-15 anchors bitwise (p3158 ratios + "
        "p3180 ratio_full_recompute/max_log_band/all_same_class + res/seal shas) + "
        "upgrade_log E1->E2_predictive entry present + evidence_level "
        "E2_predictive with held_out tag. Registration: FTR-15 scope_limits append "
        "ONLY (E2_held_out 口径注记: held_out = entity-LOO fold 口径, E2_held_out is "
        "an E2_predictive strengthened-tier annotation, no new evidence level; "
        "local-equivalence still not extrapolated to global quotient), registry "
        "v1.25 (3bb33a24) -> v1.26 (112a46ee), atlas_v1_28.html (cfc389a8, 784/784 "
        "fields clean); upgrade_log / failures / all other features byte-preserved. "
        "Verdict ftr15_heldout_recheck_registered, H1-H3 0 contradictions. design "
        "8e3sha_ref_in_execution, res 263934c8 / seal 7bb722b1, final f9ee4152 / "
        "59100c6d. SMOKE passed first run (CLEAN, 0 contradictions)."),
    "chain": {
        "res_sha8": "263934c8",
        "seal_sha8": "7bb722b1",
        "final_res_sha8": "f9ee4152",
        "final_seal_sha8": "59100c6d",
        "registry_v126": "112a46ee",
        "atlas_v1_28": "cfc389a8",
        "registry_v125": "3bb33a24",
        "ledger_pre": prev_post,
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 356
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
open(LP + '.bak_3204', 'wb').write(raw)
open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:355]
back['ledger_sha256_8'] = prev_post
eq = json.dumps(old, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 356
print('ledger #356 ok, pre', pre8, '(entry pre', prev_post, ') -> post', new_post,
      '(bak_3204 saved, rollback-equal)')

# ---------- 2. MEMO append ----------
MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8(mraw)
assert m_pre == '0cc9cea8', m_pre
ADD = """

## Phase 3204 — G5-G1 FTR-15 held_out 补强复读臂（registry v1.25→v1.26，2026-10-10）

**预注册臂 (a) 执行完毕，SMOKE 一次 CLEAN，FULL 一次收官**。零 GPU；SRC 5 文件（p3158 summary `7ed85b36` + p3180 result `cb811310` + registry v1.25 `3bb33a24` + gap v1.5 `4fb3f37d` + registry_3162 `00f15e98`）sha 断言。design 与 res/seal 见 execution.json / result（res `263934c8` / seal `7bb722b1`，final `f9ee4152` / `59100c6d`）。

### 判定（H1–H3 全 CLEAN，0 contradictions）
- **H1**：封存 3158 预算比逐位回读 == 钉定全精度常量 ×3（1.0283477199188433 / 1.3316768638966026 / 1.2472595295618603）+ 内容双锁（res `fa5cca12` / seal `e0c60629`）。
- **H2**：封存 3180 held-out 链 CLEAN——ratio_full_recompute 逐位 == 3158 ×3（device drift 0 ×3、ratio_sealed==ratio_full_recompute ×3）；max_log_band 4dp 0.0300/0.0278/0.0182 <0.25；16 折 ×3 全 same_class 且 max ratio_train <3.0；n_c_gt1 10/16/15；门 G1_device/G2_heldout True、G2_class 3.0、G2_log_band 0.25；内容双锁（res `f1689e94` / seal `a3c1b53e`）。
- **H3**：registry v1.25 FTR-15 锚逐位（p3158 ratios + p3180 ratio_full_recompute/max_log_band/all_same_class + res/seal sha）+ 升级链 E1→E2_predictive 条目在册 + held_out tag。

### 登记（append-only）
**FTR-15 scope_limits append（E2_held_out 口径注记）**：held_out = 实体 LOO fold（16 锚=16 实体）口径；E2_held_out 为 E2_predictive 强化档注记，不新增独立 evidence_level；「局部等价」仍不外推为全局商结构。registry v1.25 `3bb33a24` → **v1.26 `112a46ee`** + **atlas_v1_28 `cfc389a8`**（784/784 字段，0/0/0/0）；upgrade_log / failures / 其余 25 特征字节保全断言全过。evidence_level 维持 E2_predictive（与 3201 FTR-24 口径注记同型）。

### 图谱现状（v1.26）
26 特征：E2×20（含 E2_held_out 口径注记 ×3：FTR-23/25/26 + FTR-15 强化档注记）、E3×3、E1×3；failures F1–F19；upgrade_log 9；gap v1.6 `0f6423e0`；ledger n=356。

### 预注册 Phase 3205（两支候选）
- (a) **G5-G3 FTR-14 A2 归因后复读臂**（零 GPU）：FTR-14 于 3177 gate fail（F16）/3178 A2 归因 FAIL×3 关系特异残差确证后入册 FTR-22/23——回读 3177/3178 封存标量 + registry v1.26 FTR-14/22/23 锚交叉逐位，评估 FTR-14 scope_limits 补写（gate fail 后续口径），registry v1.26→v1.27 仅 FTR-14 scope_limits append + atlas_v1_29。
- (b) **G5-G4 GAP-4 机制段 v1.7 草案**（零 GPU，report-only）：起草 port_residual_dominant 机制段定稿候选文本，不动账本。
- 默认执行 (a)；若 3177/3178 封存锚与 v1.26 交叉断言不齐备则降级 (b) 并诚实登记。
"""
open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8(open(MP, 'rb').read()))

# ---------- 3. daily ----------
DP = r'.workbuddy\memory\2026-10-10.md'
draw = open(DP, 'rb').read()
dadd = ("\n\n### Phase 3204 G5-G1 FTR-15 held_out 补强复读臂收官（2026-10-10）\n"
        "- SMOKE 一次 CLEAN；FULL 一次收官：H1 3158 比值逐位 + H2 3180 链（ratio 逐位×3/band 4dp/16 折×3 same_class/drift 0/门齐）+ H3 v1.25 FTR-15 锚逐位+升级条目。\n"
        "- 登记：FTR-15 scope_limits append（E2_held_out 口径注记），registry v1.25 3bb33a24→v1.26 112a46ee + atlas_v1_28 cfc389a8（784/784）。\n"
        "- 环境插曲：3.13.12 默认 python 缺 numpy（此前 Phase 均可跑），pip 补装后复跑正常；3.11.9 自带 numpy 可作备用。\n"
        "- ledger #356（rollback 等价验证）；预注册 3205：默认 (a) G5-G3 FTR-14 复读臂；(b) gap v1.7 草案。\n")
open(DP, 'wb').write((draw.decode('utf-8') + dadd).encode('utf-8'))
print('daily ->', sha8(open(DP, 'rb').read()))

# ---------- 4. MEMORY patch ----------
MP2 = r'.workbuddy\memory\MEMORY.md'
b = open(MP2, 'rb').read()
m2_pre = sha8(b)
t = b.decode('utf-8')
i = t.find('## 4 下一步')
j = t.find('## 5 ')
assert i >= 0 and j > i
new4 = """## 4 下一步
- **3205 已预注册 = G5-G3 FTR-14 A2 归因后复读臂**（零 GPU，默认）：回读 3177/3178 封存标量 + registry v1.26 FTR-14/22/23 锚交叉逐位，registry v1.26→v1.27 仅 FTR-14 scope_limits append + atlas_v1_29；候选 (b) gap v1.7 机制段草案（report-only）。
- 3204 收官：FTR-15 held_out 补强复读 REGISTERED（H1–H3 全 CLEAN），registry v1.25 3bb33a24→**v1.26 112a46ee** + atlas_v1_28 cfc389a8（784/784），ledger #356。
- 3203 收官：G5-F2 GAP-4 复读 CLEAN（report-only，registry 未动），ledger #355。
- 后续：FTR-04 已收官（3202）、GAP-4 已闭环复读（3203）、FTR-15 已补强（3204）、5 appendix；UI v1.6 客户端改造（用户询方案已给，零 script 版排队）。
- 挂账不变：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；水果类；K4。

"""
t = t[:i] + new4 + t[j:]
t = t.replace('registry v1.25 `3bb33a24`（26 特征）、atlas_v1_27 `739faeb7`、ledger n=355（#355 RE_READ_CONSISTENCY）。',
              'registry v1.26 `112a46ee`（26 特征）、atlas_v1_28 `cfc389a8`、ledger n=356。')
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8(open(MP2, 'rb').read()))
print('CLOSEOUT DONE')
