# -*- coding: utf-8 -*-
"""Phase 3214 closeout: ledger #366 + MEMO + daily + MEMORY (report-only arm;
governance entry only)."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


TODAY = '2026-10-11'

LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3214', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3214') else None
already = (bako0 is not None and len(disk0['measurements']) == 366
           and disk0['measurements'][365].get('phase') == 3214)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '3bd36b2a', prev_post
if already:
    raw = open(LP + '.bak_3214', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 365

entry = {
    "phase": 3214,
    "name": "g5g14_ftr15_heldout_closed",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, report-only closure-confirmation arm)",
    "verdict": "g5g14_ftr15_heldout_closed|REGISTERED|H1_3158_ratios_bitwise_x3|H2_3180_chain_ratio_bitwise_drift0_band0.0300_0.0278_0.0182_16folds_sameclass_ncgt1_10_16_15|H3_3204_sealed_registry_append_verbatim_E2_heldout_note|H4_ledger_untouched_n365_3bd36b2a|closed@3204",
    "evidence_level": "report_only_closure_confirmation",
    "model_scope": "n/a",
    "prereg_id": "G5-G14",
    "superseded_by": None,
    "detail": (
        "G5-G14 FTR-15 held_out closed-loop confirmation arm (report-only, "
        "zero GPU; default prereg arm (a), per MEMO Phase 3213 tail). "
        "Pre-freeze probe found the held_out reinforcement recheck ALREADY "
        "closed — 3204 G5-G1 (registry v1.25->v1.26, FTR-15 scope_limits "
        "append) — so the arm ran as closure confirmation: H1 3158 sealed "
        "per-model products (file sha 90e734fd/1f532016/c49614fc) /ratio "
        "BITWISE == FTR-15 p3158 anchor asserts x3 (1.0283477199188433 / "
        "1.3316768638966026 / 1.2472595295618603); H2 3180 sealed product "
        "(res f1689e94 / seal a3c1b53e, verdict ftr15_upgraded_E2_"
        "predictive) per_model ratio_full_recompute bitwise == 3158 x3 == "
        "p3180 anchor asserts, drift == 0.0 x3, max_band bitwise == p3180 "
        "asserts (0.030024603607133406 / 0.027780384524118094 / "
        "0.018213681402158396 < 0.25), all_same_class true x3, 16 folds "
        "x3 all same_class, n_c_gt1 [10, 16, 15], gates G1_device/"
        "G2_heldout True; H3 3204 sealed product (res 263934c8 / seal "
        "7bb722b1 / design 588dcf87, verdict ftr15_heldout_recheck_"
        "registered, recheck clean) + registry v1.28 FTR-15 scope_limits "
        "3204 append verbatim (incl. E2_held_out 强化档注记不新增独立 "
        "evidence_level) + MEMO 3204 seg anchors on disk; H4 ledger "
        "untouched (n=365, 3bd36b2a before==after). Disposition: FTR-15 "
        "held_out closed@3204, E2_predictive with E2_held_out caliber "
        "note maintained, no registry action, drop from candidate "
        "follow-up list. Verdict ftr15_heldout_closed, H1-H4 0 "
        "contradictions. Honest log: SMOKE run1 CLEAN once; FULL clean "
        "(SMOKE first-try CLEAN, no fixes needed). design b8052e94, res "
        "a48005fa / seal 926be348."),
    "chain": {
        "res_sha8": "a48005fa",
        "seal_sha8": "926be348",
        "design_sha8": "b8052e94",
        "r3158": "90e734fd/1f532016/c49614fc (file bytes x3)",
        "r3180": "cb811310 (res f1689e94/seal a3c1b53e)",
        "r3204": "b1021387 (res 263934c8/seal 7bb722b1)",
        "registry_v128": "5995bbe9",
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 366
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3214', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3214', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:365]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 366
print('ledger #366 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3214 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3214 \xe2\x80\x94 G5-G14' not in mraw:
    ADD = """

## Phase 3214 — G5-G14 FTR-15 held_out 闭环确认臂（report-only，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 一次 CLEAN，FULL 一次收官**。零 GPU、report-only（H4 ledger 未动断言）。design `b8052e94`，res `a48005fa` / seal `926be348`，verdict **ftr15_heldout_closed**。

### 关键探针结论（预冻结）
FTR-15 held_out 补强复读**在册已闭环**：3204 G5-G1（registry v1.25→v1.26，FTR-15 scope_limits append）已复读封存 3158/3180 链。本臂转为**闭环确认复读**，不升版。

### 确认结论（H1–H4 全过）
- **H1**：3158 三模型封存产物（文件 sha `90e734fd`/`1f532016`/`c49614fc`）/ratio 与 FTR-15 p3158 锚 asserts **逐位一致** ×3（1.0283477199188433 / 1.3316768638966026 / 1.2472595295618603）。
- **H2**：3180 封存产物（res `f1689e94` / seal `a3c1b53e`，verdict ftr15_upgraded_E2_predictive）——ratio_full_recompute 逐位 == 3158 ×3 == p3180 锚；drift==0.0 ×3；max_band 逐位一致（0.030024603607133406 / 0.027780384524118094 / 0.018213681402158396 < 0.25）；all_same_class true ×3；16 折 ×3 全 same_class；n_c_gt1 [10,16,15]；G1_device/G2_heldout 门 True。
- **H3**：3204 封存产物（res `263934c8` / seal `7bb722b1` / design `588dcf87`，verdict ftr15_heldout_recheck_registered，recheck clean）+ registry v1.28 scope_limits 3204 append 逐字在盘（含 **E2_held_out 为 E2_predictive 强化档注记、不新增独立 evidence_level** 口径）+ MEMO 3204 节锚。
- **H4**：ledger 未动（n=365，3bd36b2a before==after）。

### disposition
FTR-15 held_out **closed@3204**：held_out 证据 = 实体 LOO fold（16 锚=16 实体）口径；E2_predictive + E2_held_out 强化档注记维持，无 registry 动作，自候选跟进列表移除。

### 诚实日志
SMOKE 一次 CLEAN、FULL 一次 CLEAN（零修正轮次）。

### 预注册 Phase 3215（两支候选）
- (a) **G5-G15 GAP-4 闭环确认复读臂**（零 GPU report-only）：回读 GAP-4 谱外崩塌归因链（3175 port_residual_dominant + 3192 amplitude 归因 + 3205 残留挂账口径），评估挂账处置。
- (b) **UI v1.9**：版本链/账本尾块锚详情展开（对齐 v1.8 块视图）。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3214' not in draw:
    dadd = ("\n\n### Phase 3214 G5-G14 FTR-15 held_out 闭环确认收官（report-only，2026-10-11）\n"
            "- 探针：补强复读在册已闭环（3204 G5-G1 v1.25→v1.26），本臂转闭环确认；SMOKE 一次 CLEAN、FULL 一次 CLEAN（零修正轮次）。\n"
            "- H1 3158 ratios 逐位 ×3 + H2 3180 链（ratio 逐位/drift 0/band<0.25/16 折 same_class/n_c_gt1 10/16/15）+ H3 3204 + scope_limits append + E2_held_out 口径注记 + H4 ledger 未动。\n"
            "- disposition closed@3204，无 registry 动作；ledger #366（治理条目，rollback 等价验证）；预注册 3215：默认 (a) GAP-4 闭环确认复读；(b) UI v1.9。\n")
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
- **3215 已预注册 = G5-G15 GAP-4 闭环确认复读臂**（零 GPU report-only，默认）：回读 GAP-4 归因链（3175 端口残留主导 + 3192 amplitude 归因 + 3205 挂账口径）；候选 (b) UI v1.9（版本链/账本尾锚详情）。
- 3214 收官：G5-G14 FTR-15 held_out 闭环确认 REGISTERED（res a48005fa/seal 926be348，design b8052e94），ledger #366；closed@3204（E2_predictive + E2_held_out 强化档注记维持，无 registry 动作）。
- 3213/3212/3211/3210：UI v1.8 ca76ec49 670/670（#365）；FTR-04 glm4 归因 closed@3202（#364）；UI v1.7 62ba1bd6（#363）；水果类/K4 盘点（#362，**列表删除待用户确认**）。
- 3209/3208/3207/3206：UI v1.6 bd230988（#361）；N 线草案 f29da2de（#360）；5-appendix 复读（#359）；FTR-22 链 + registry v1.28 5995bbe9 + atlas_v1_30 ebc1e122（#358）。
- 挂账不变：跨线账本补丁施加确认（APPX-5）；N 线补录施加（跨线确认后）；5 appendix GPU/跨线项。（水果类/K4/FTR-04/FTR-15 已闭环或待移除）

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
