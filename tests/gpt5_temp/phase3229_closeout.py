# -*- coding: utf-8 -*-
"""Phase 3229 closeout: ledger #381 + MEMO + daily + MEMORY (self-check on disk readback)."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


TODAY = '2026-10-11'
P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3229'
      r'\g5g26_e3_gate_decision')
RES_F = sha8_file(os.path.join(P5, 'result.json'))
LOG_F = sha8_file(os.path.join(P5, 'run_log.txt'))
FZ_SHA = 'd0bb40e6'          # freeze (SMOKE==FULL, deterministic gates arm)
REG30_F, REG30_C = 'c11d13bd', '9ea9fc3e'
GAP18_F, GAP18_C = '7065ec4e', '09bf2fa9'
RES_C = None  # filled below from disk
_r = json.load(open(os.path.join(P5, 'result.json'), encoding='utf-8'))
RES_C = _r['res_sha8']

LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3229', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3229') else None
already = (bako0 is not None and len(disk0['measurements']) == 381
           and disk0['measurements'][380].get('phase') == 3229)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '8e593d22', prev_post
if already:
    raw = open(LP + '.bak_3229', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 380

entry = {
    "phase": 3229,
    "name": "g5g26_e3_gate_decision",
    "line": "G",
    "date": TODAY,
    "model": "CPU arithmetic on sealed npz (zero GPU)",
    "verdict": "e3_gate_e3_fail_f20_registered|GD_flip_FAIL_matched0_nullmax0.049"
               "|GD_dose_FAIL_degenerate_zero_K2|GD_spec_FAIL_inverted_K3"
               "|GD_amp_FAIL_1of2_0.40_0.80|registry_v1.30_F20|gap_v1.8_GAP5_note",
    "evidence_level": "gate_decision (E3 FAIL -> no E-level change)",
    "model_scope": "qwen3-4b + qwen3-14b (2/3); glm4 deferred",
    "prereg_id": "G5-G26",
    "superseded_by": None,
    "detail": (
        "G5-G26 E3 gate evaluation + registry decision arm (zero GPU, "
        "default (a) per 3228 prereg). Gates GD_flip/GD_dose/GD_spec "
        "applied VERBATIM from design a883c4da to the sealed 3228 flip "
        "matrices (res 99693298 / npz a844d985 / freeze c51bfea5); "
        "integrity: 1176 sealed agg cells reproduced from npz bit-exact "
        "before gate evaluation; freeze d0bb40e6 sealed pre-evaluation "
        "with 5 clarifications (K1 design-label fix: 'FTR-22 family' in "
        "upgrade_targets is a mislabel, relation-direction carrier = "
        "FTR-24 dv family; K2 dose degenerate-zero rule; K3 spec "
        "undefined-ratio rule; K4 fold basis; K5 3225 provenance warning "
        "routing). VERDICT per taxonomy: E3_FAIL (0/3 core gates). "
        "GD_flip FAIL: matched flip-toward-R rate 0.0000 on 4/4 seen "
        "folds x 2 models x 2 directions (deltas all <= 0 vs null_max up "
        "to 0.049 on OOV report); GD_dose FAIL: matched rates all-zero "
        "across alphas (K2 degenerate — monotonicity of nulls is not "
        "evidence); GD_spec FAIL: cross-patch rate UP TO ~8x matched "
        "(inverted specificity; K3 applied); GD_amp FAIL as gate "
        "(Spearman 0.40/0.80 = 1/2 models >= 0.6, n=4 scope). CONCLUSION "
        "(F20): the T2 ctx-contrast relation-level direction u_R is NOT "
        "a behavioral lever on the isa/hasa judgment readout under the "
        "exactly-linear last-layer pre-logit map (u_isa anti-directional "
        "monotone dD -0.139/-0.112 @a1, -0.278/-0.223 @a2, 4b/14b OOV); "
        "behavioral judgment is carried upstream — readout geometry != "
        "behavioral control; coheres 3188 DECOUPLED + GAP-5. REGISTRY "
        "v1.30 (file %s / content %s): F20 e3_sufficiency_gate_fail "
        "appended, upgrade_log #11 (FAIL record, no E-level change), "
        "scope_limits appends FTR-24 (E3 FAIL note) + FTR-25/FTR-26 "
        "(3225 hub-vs-local W_U provenance warning filed per K5), 23 "
        "features + legacy top-level byte-preserved. GAP v1.8 (file %s / "
        "content %s): GAP-5 mechanism_note appended — three-stage "
        "conclusion: upstream encoding carries judgment behavior -> "
        "last-layer readout is an exact linear projection (3228 C6v3 "
        "cos 1.0000) -> post-softmax distribution-level nonlinearity; "
        "causal levers for behavior must be sought upstream; last-layer "
        "geometric fingerprints do not extrapolate to behavioral "
        "control. Ledger READ-ONLY in this arm (head at freeze: n=380 "
        "post 8e593d22; reseal run observed head n=381 with "
        "byte-identical before/after within the run; #381 appended by "
        "this closeout only)."
        % (REG30_F, REG30_C, GAP18_F, GAP18_C)),
    "chain": {
        "res_sha8": RES_F,
        "seal_sha8": LOG_F,
        "freeze_sha8": FZ_SHA,
        "res_content_sha8": RES_C,
        "registry_v130": {"file": REG30_F, "content": REG30_C},
        "gap_v18": {"file": GAP18_F, "content": GAP18_C},
        "gate_pass": {"GD_flip": False, "GD_dose": False,
                      "GD_spec": False, "GD_amp_support": False},
        "e3_verdict": "E3_FAIL",
        "integrity_cells": 1176,
        "ledger_pre": prev_post,
        "note": "gate decision arm; next = 3230 per prereg",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 381
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3229', 'wb').write(raw)
open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3229', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:380]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post \
    and len(d2['measurements']) == 381
print('ledger #381 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3229 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if '## Phase 3229 — G5-G26' not in mraw.decode('utf-8'):
    ADD = """

## Phase 3229 — G5-G26 E3 门评估 + registry 决策臂（2026-10-11）

**执行模式**：零 GPU 纯算术（默认 (a)，按 3228 预注册）；freeze **`d0bb40e6`**（门评估前封存，含 5 条冻结澄清 K1–K5）；完整性 **1176 个封存 agg 单元自 npz 位级复现**后才评门；res 待 verify、verdict **`e3_gate_e3_fail_f20_registered`**（0 矛盾）。

### 门结果（3227 四门逐字应用于 3228 封存矩阵）
- **GD_flip FAIL**：matched flip-toward-R = 0.0000 于 4/4 seen 折 × 2 模型 × 2 方向（delta 全 ≤0；OOV report-only null 带最高 0.049@14b）。
- **GD_dose FAIL**：matched 率全 α 退化全零（K2 规则：全零序列的单调性不是剂量证据）。
- **GD_spec FAIL**：特异性反向——cross-patch（u_isa→hasa 判断）翻转率最高达 matched（u_hasa→hasa）的 ~8×，非 ≤1/3（K3：matched=0 时 cross 须恰为 0）。
- **GD_amp FAIL（作为门）**：Spearman 0.40（4b）/ 0.80（14b）= 1/2 模型 ≥0.6（n=4 scope）。
- **taxonomy 判定：E3_FAIL（0/3 核心门）→ F 账本 F20，不隐藏。**

### 定论（F20，e3_sufficiency_gate_fail）
T2 ctx-对比配方关系级方向 u_R 在末层 pre-logit 精确线性读出图上对 isa/hasa 判断行为**不是杠杆**（u_isa 反方向单调 dD −0.1389/−0.1117 @α1、−0.2777/−0.2234 @α2，OOV 4b/14b；u_hasa 弱 ~7–11×）——行为判断由上游承载，**读出几何 ≠ 行为控制**；与 3188 DECOUPLED、GAP-5 自洽。三段式：上游编码（承载行为）→ 末层线性读出（精确投影，3228 C6v3 cos 1.0000）→ post-softmax（分布级非线性）。

### 冻结澄清（K1–K5，评估前入 freeze）
K1 设计档 upgrade_targets「FTR-22 family」系标注笔误（FTR-22=端口校准；关系方向承载=FTR-24 Δv 族）；K2 剂量退化规则；K3 spec 未定义比值规则；K4 折基（seen 4 cut-折、OOV 3 模板 report-only）；K5 3225 出处告警（hub 4B W_U `4c807e25` vs 本地采集 `328a91d3`；A1 结论为图性质不受影响）随 FTR-25/26 scope_limits 入档。

### registry v1.30 + gap v1.8
- **registry v1.30**（file `c11d13bd` / content `9ea9fc3e`）：failures +**F20**、upgrade_log **#11**（FAIL 记录，E 级不变）、FTR-24（E3 FAIL 注）/FTR-25+FTR-26（出处告警）scope_limits append、provenance 锚 e3_flip_matrices `a844d985`；23 特征 + legacy 顶层字节保全。
- **gap v1.8**（file `7065ec4e` / content `09bf2fa9`）：GAP-5 mechanism_note append（三段式定论）；legacy ×4 + GAP-5 其余字段字节保全。
- 账本只读（n=380 post `8e593d22` before==after）。

### 预注册 Phase 3230（两支候选）
- (a) **G5-G27 关系对比方向效力臂**（默认）：用行级关系对比方向 dR = h(isa 面板)−h(hasa 面板)（3157 KOUT 缓存均值，非 ctx-对比配方 v_R）重测 flip 效力——检验「反方向/零效力」是否 T2 配方特有（3228 候选 (b) 继承）；
- (b) **GD_amp 扩样臂**：n=4 → n=10 类（3169 全类崩塌幅度 + 3225 npz u* 逐类投影，纯算术）——裁决幅度承载假设。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3229' not in draw:
    dadd = ("\n\n### Phase 3229 G5-G26 E3 门评估 + registry 决策臂收官（2026-10-11）\n"
            "- 零 GPU 纯算术（默认 (a)）：freeze d0bb40e6（评估前封存，K1–K5 澄清）；1176 封存 agg 单元 npz 位级复现；verdict e3_gate_e3_fail_f20_registered（0 矛盾）。\n"
            "- 四门全 FAIL：GD_flip matched 0/4 折×2 模型×2 方向；GD_dose 退化全零（K2）；GD_spec 反向 cross~8×matched（K3）；GD_amp 0.40/0.80=1/2。taxonomy：E3_FAIL → F20。\n"
            "- 定论（F20）：T2 ctx-对比配方 u_R 非行为读出杠杆（u_isa 反向单调），行为判断上游承载，读出几何≠行为控制；三段式入 GAP-5 mechanism_note（gap v1.8 7065ec4e/09bf2fa9）。\n"
            "- registry v1.30（c11d13bd/9ea9fc3e）：F20 + upgrade_log #11（E 级不变）+ FTR-24/25/26 scope append（含 3225 出处告警 K5）；字节保全。\n"
            "- 预注册 3230：(a) G5-G27 关系对比方向 dR=h(isa)−h(hasa) 效力臂（默认）；(b) GD_amp 扩样 n=4→10 类纯算术臂。\n")
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
- **3230 已预注册**：(a) **G5-G27 关系对比方向效力臂**（默认）：dR = h(isa 面板)−h(hasa 面板)（3157 KOUT 缓存均值，非 ctx-对比配方 v_R）重测 flip 效力——检验反方向/零效力是否 T2 配方特有；(b) GD_amp 扩样臂（n=4→10 类，3169 全类崩塌幅度 + u* 逐类投影，纯算术）。
- 3229 收官：E3 门评估 E3_FAIL → **F20**（registry v1.30 `c11d13bd`/`9ea9fc3e`，upgrade_log #11 E 级不变）+ gap v1.8 `7065ec4e`/`09bf2fa9`（GAP-5 三段式 mechanism_note）；freeze d0bb40e6；1176 agg 单元位级复现；四门全 FAIL（flip 0/dose 退化/spec 反向/amp 1-of-2）；K1 澄清：设计档「FTR-22 family」系笔误，关系方向承载=FTR-24。ledger #381 post %s。
- 3228 收官：E3 计算+封存（npz a844d985，res 99693298，freeze c51bfea5），ledger #380 post 8e593d22；C9 相位内修正（缓存槽 NH-1 = post-norm pre-logit 态，修正图精确线性）；4B W_U 出处分裂（本地 328a91d3 ≠ hub 4c807e25）。
- 3227/3226/3225：E3 干预设计（a883c4da/32968738，#379 f21108e7）；A2 三门 FTR-25+FTR-26 升 E2_behavioral（registry v1.29 85b8dc1f，GAP-5 入册 gap v1.7 4f45fda0，#378 62b2d7d6）；A1 计算+封存（npz ac5d4d74，res ea1a0585，#377 969b6791）。
- 挂账不变：APPX-5 施加（重基线后）；N 线施加（ready）；5 appendix GPU/跨线项；UI 线冻结于 v1.10；**水果类/K4 列表删除待用户确认**。

""" % new_post
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))

# self-check: disk readback of all five writes
d3 = json.load(open(LP, encoding='utf-8'))
assert len(d3['measurements']) == 381 and d3['ledger_sha256_8'] == new_post
assert d3['measurements'][380]['phase'] == 3229
assert '## Phase 3229' in open(MP, 'rb').read().decode('utf-8')
assert b'### Phase 3229' in open(DP, 'rb').read()
t3 = open(MP2, 'rb').read().decode('utf-8')
assert '3229 收官' in t3 and new_post in t3
print('self-check: ledger/MEMO/daily/MEMORY disk readback OK; closeout done')
