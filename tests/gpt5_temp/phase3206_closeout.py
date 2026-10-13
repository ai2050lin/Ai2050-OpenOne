# -*- coding: utf-8 -*-
"""Phase 3206 closeout: ledger #358 append + MEMO append + daily + MEMORY.
Registry v1.28 (5995bbe9) + atlas_v1_30 (ebc1e122) already sealed by main script.
"""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

TODAY = '2026-10-10'

# ---------- 1. ledger append (#358) ----------
LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
pre8 = sha8_file(LP)
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3206', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3206') else None
already = (bako0 is not None and len(disk0['measurements']) == 358
           and disk0['measurements'][357].get('phase') == 3206)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'bf6ed076', prev_post
if already:
    # idempotent replay from bak base (byte-identical rewrite)
    raw = open(LP + '.bak_3206', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 357

entry = {
    "phase": 3206,
    "name": "g5g5_ftr22_chain_recheck",
    "line": "G",
    "date": TODAY,
    "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5g5_ftr22_chain_recheck|REGISTERED|L1_3172_pooled_bitwise_x5_kcurves8_x3_overall_borderline_partial_recovery|L2_3169_gate_bitwise_eq_pooled0_k0_drift0_3171_ratioS_x3_mixed_across_models|L3_registry_v127_ftr22_three_anchors_bitwise_values_E2_predictive|scope_limits_residual_ledger_update_append|registry_v1.27_d5ce1c19_to_v1.28_5995bbe9|atlas_v1_30_ebc1e122_787_fields_clean",
    "evidence_level": "E2_predictive_chain_reread_annotation",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-G5",
    "superseded_by": None,
    "detail": (
        "G5-G5 FTR-22 upgrade chain re-read arm (zero GPU; sealed p3169 + p3171 + "
        "p3172 results + registry v1.27 re-read; default prereg arm (a)). L1: "
        "sealed 3172 pooled curve bitwise x5 (2.5388114997805062 / 2.031459140015479 "
        "/ 1.9429485491548533 / 1.8766676450829491 / 1.800176217907229) + "
        "per_model kcurves.8.ratio bitwise x3 (1.9485993218753288 / "
        "1.7100461949026464 / 1.767394689531845) + overall ratio_kmax "
        "1.800176217907229 / main_cls borderline_partial_recovery + content double "
        "lock (res 463f42c8 / seal cdb85525). L2: sealed 3169 gate bitwise == 3172 "
        "pooled.0 x3 (k0 drift 0: ratio_B / pooled_E_oov_B / pooled_E_seen_B) + "
        "gate verdict collapse_confirmed_gap4_open + content double lock (res "
        "49430a39 / seal 018c6024); sealed 3171 slots.k_main.ratio_S bitwise x3 "
        "(0.7511424513908266 / 0.7166531333803954 / 0.8205301477290912) + "
        "overall.main_cls mixed_across_models + content double lock (res 6a29201c "
        "/ seal cc7ccedd). L3: registry v1.27 FTR-22 three anchors (p3172/p3169/"
        "p3171) asserts bitwise vs sealed values (incl k0_drift_ratio/E_oov/E_seen "
        "== 0.0 x3) + values dict (ratio_k0/k8, E_oov_pooled_k0/k8, "
        "E_seen_pooled_k0/k8, k0_drift_ratio_vs_3169, p3171_ratio_S x3) + "
        "evidence_level E2_predictive. Registration: FTR-22 scope_limits append "
        "ONLY (残留 ~1.8x 成分挂账口径更新: 3175 残留归因 delta 0.0660 "
        "port_residual_dominant, ratio_in 1.8002 / ratio_out 1.8662; 3203 G5-F2 "
        "双链复读 CLEAN 闭环; 「残留成分未定位」->「closed@3175 端口残留主导, "
        "E3 因果收敛仍不成立」), registry v1.27 (d5ce1c19) -> v1.28 (5995bbe9), "
        "atlas_v1_30.html (ebc1e122, 787/787 fields clean); upgrade_log / failures "
        "/ all other features byte-preserved. Verdict "
        "ftr22_chain_recheck_registered, L1-L3 0 contradictions. SMOKE passed "
        "first run (CLEAN). design 530263ce, res 4229bae5 / seal 812ac34a, final "
        "7ad6c7d4 / f899b8cd."),
    "chain": {
        "res_sha8": "4229bae5",
        "seal_sha8": "812ac34a",
        "final_res_sha8": "7ad6c7d4",
        "final_seal_sha8": "f899b8cd",
        "registry_v128": "5995bbe9",
        "atlas_v1_30": "ebc1e122",
        "registry_v127": "d5ce1c19",
        "ledger_pre": prev_post,
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 358
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not (os.path.exists(LP + '.bak_3206') and len(disk0['measurements']) == 358
        and disk0['measurements'][357]['phase'] == 3206):
    open(LP + '.bak_3206', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3206', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:357]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 358
print('ledger #358 ok, pre', pre8, '(entry pre', prev_post, ') -> post', new_post,
      '(bak_3206 saved, rollback-equal)')

# ---------- 2. MEMO append ----------
MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
assert m_pre == 'e2dea1fc', m_pre
ADD = """

## Phase 3206 — G5-G5 FTR-22 升级链复读臂（registry v1.27→v1.28，2026-10-10）

**预注册臂 (a) 执行完毕，SMOKE 一次 CLEAN，FULL 一次收官**。零 GPU；SRC 6 文件（p3169 `5b51c2c1` + p3171 `a43cf48c` + p3172 `d8ddc481` + registry v1.27 `d5ce1c19` + gap v1.5 `4fb3f37d` + registry_3162 `00f15e98`）sha 断言。design 与 res/seal 见 execution.json / result（res `4229bae5` / seal `812ac34a`，final `7ad6c7d4` / `f899b8cd`，design `530263ce`）。

### 判定（L1–L3 全 CLEAN，0 contradictions）
- **L1**：封存 3172 pooled 曲线逐位回读 ×5（2.5388114997805062 / 2.031459140015479 / 1.9429485491548533 / 1.8766676450829491 / 1.800176217907229）+ per_model kcurves.8.ratio 逐位 ×3（1.9485993218753288 / 1.7100461949026464 / 1.767394689531845）+ overall ratio_kmax 1.800176217907229 / main_cls borderline_partial_recovery；内容双锁（res `463f42c8` / seal `cdb85525`）。
- **L2**：封存 3169 gate 逐位 == 3172 pooled.0 ×3（k0 drift 0：ratio_B / pooled_E_oov_B / pooled_E_seen_B）+ verdict collapse_confirmed_gap4_open；内容双锁（res `49430a39` / seal `018c6024`）。封存 3171 slots.k_main.ratio_S 逐位 ×3（0.7511424513908266 / 0.7166531333803954 / 0.8205301477290912）+ overall.main_cls mixed_across_models；内容双锁（res `6a29201c` / seal `cc7ccedd`）。
- **L3**：registry v1.27 FTR-22 三锚（p3172/p3169/p3171）asserts 逐位 vs 封存（含 k0_drift_ratio/E_oov/E_seen == 0.0 ×3）+ values 字典（ratio_k0/k8、E_oov/E_seen pooled k0/k8、k0_drift_ratio_vs_3169、p3171_ratio_S ×3）+ evidence_level E2_predictive。

### 登记（append-only）
**FTR-22 scope_limits append（残留 ~1.8x 成分挂账口径更新）**：该成分已在 3175 完成残留归因定判（delta 0.0660，ratio_in 1.8002 / ratio_out 1.8662，port_residual_dominant）并于 3203 G5-F2 双链复读 CLEAN 闭环（机制链 3169→3172→3175 + C 线五连 3188→3192）——「残留成分未定位」表述自本注记起更新为「已定位为端口残留主导（port_residual_dominant），因果收敛（E3）仍不成立但机制归属 closed@3175」；范围口径不变（中文面板+单一模板族+行为读出层+协议内干预）。registry v1.27 `d5ce1c19` → **v1.28 `5995bbe9`** + **atlas_v1_30 `ebc1e122`**（787/787 字段，0/0/0/0）；upgrade_log / failures / 其余 25 特征字节保全断言全过。evidence_level 维持 E2_predictive。

### 图谱现状（v1.28）
26 特征：E2×20（E2_held_out 口径注记 ×3 + FTR-15 强化档注记 + FTR-14 gate-fail 口径注记 + FTR-22 残留挂账口径注记）、E3×3、E1×3；failures F1–F19；upgrade_log 9；gap v1.6 `0f6423e0`；ledger n=358（#358 post 见 chain）。

### 预注册 Phase 3207（两支候选）
- (a) **G5-G6 5-appendix 挂账复读臂**（零 GPU，report-only）：回读 gap v1.5 appendix 5 条（title/status/note/anchor/anchor_sha8）逐字段 + 3192 gap v1.6 mechanism_note 交叉逐位 + 对应封存源文件 sha 链，评估各 appendix 注记是否需口径更新；不动账本。
- (b) **UI v1.6 客户端改造零 script 版开工**（非账本轮）：图谱透镜客户端化（atlas v1.30 数据面）。
- 默认执行 (a)；若 appendix 封存锚不齐备则降级 (b) 并诚实登记。
"""
open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

# ---------- 3. daily ----------
DP = r'.workbuddy\memory\2026-10-10.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
assert d_pre == '402cd205', d_pre
dadd = ("\n\n### Phase 3206 G5-G5 FTR-22 升级链复读臂收官（2026-10-10）\n"
        "- SMOKE 一次 CLEAN；FULL 一次收官：L1 3172 pooled 逐位×5 + kcurves.8 ×3 + overall borderline_partial_recovery；L2 3169 gate 逐位 == pooled.0（k0 drift 0）+ 3171 ratio_S ×3 mixed_across_models；L3 v1.27 FTR-22 三锚逐位 + values + E2_predictive。0 contradictions。\n"
        "- 登记：FTR-22 scope_limits append（残留 1.8x 成分挂账口径更新：3175 port_residual_dominant delta 0.0660 + 3203 双链闭环，「残留成分未定位」→「closed@3175」），registry v1.27 d5ce1c19→v1.28 5995bbe9 + atlas_v1_30 ebc1e122（787/787）。\n"
        "- 主脚本四块（run_recheck/finish_registry/flatten/main）用安全索引整块替换补丁生成（唯一标记 count 校验，无截断事故），py_compile 过；SMOKE/FULL 均一次过。\n"
        "- ledger #358（rollback 等价验证）；预注册 3207：默认 (a) G5-G6 5-appendix 挂账复读臂（report-only）；(b) UI v1.6 零 script 版开工。\n")
open(DP, 'wb').write((draw.decode('utf-8') + dadd).encode('utf-8'))
print('daily', d_pre, '->', sha8_file(DP))

# ---------- 4. MEMORY patch (section 4) ----------
MP2 = r'.workbuddy\memory\MEMORY.md'
b = open(MP2, 'rb').read()
m2_pre = sha8_file(MP2)
t = b.decode('utf-8')
i = t.find('## 4 下一步')
j = t.find('## 5 ')
assert i >= 0 and j > i
new4 = """## 4 下一步
- **3207 已预注册 = G5-G6 5-appendix 挂账复读臂**（零 GPU report-only，默认）：回读 gap v1.5 appendix 5 条逐字段 + 3192 gap v1.6 mechanism_note 交叉逐位 + 封存源 sha 链，评估口径更新；不动账本；候选 (b) UI v1.6 零 script 版开工（非账本轮）。
- 3206 收官：FTR-22 升级链复读 REGISTERED（L1–L3 全 CLEAN，k0 drift 0），registry v1.27 d5ce1c19→**v1.28 5995bbe9** + atlas_v1_30 ebc1e122（787/787），ledger #358；FTR-22 残留挂账口径 closed@3175。
- 3205 收官：FTR-14 归因后复读 REGISTERED（K1–K3 全 CLEAN），registry v1.26 112a46ee→**v1.27 d5ce1c19** + atlas_v1_29 b2841d0c（785/785），ledger #357。
- 已收官链：FTR-04（3202）/GAP-4（3203）/FTR-15（3204）/FTR-14（3205）/FTR-22（3206）。
- 后续：5 appendix；UI v1.6 客户端改造（用户询方案已给，零 script 版排队）。
- 挂账不变：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；水果类；K4。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
