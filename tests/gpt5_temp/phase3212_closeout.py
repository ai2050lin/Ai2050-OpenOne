# -*- coding: utf-8 -*-
"""Phase 3212 closeout: ledger #364 + MEMO + daily + MEMORY (report-only arm;
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
bako0 = json.loads(open(LP + '.bak_3212', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3212') else None
already = (bako0 is not None and len(disk0['measurements']) == 364
           and disk0['measurements'][363].get('phase') == 3212)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'd637e53f', prev_post
if already:
    raw = open(LP + '.bak_3212', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 363

entry = {
    "phase": 3212,
    "name": "g5g12_ftr04_attribution_closed",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, report-only closure-confirmation arm)",
    "verdict": "g5g12_ftr04_attribution_closed|REGISTERED|A1_3181_sealed_noise_level_p3181_anchor_bitwise|A2_3202_decomposition_rederived_0.1180952380952382_1.9661111513569065_registry_append_verbatim|A3_chain_3176FAIL_3181noise_3202decompose_E1_maintained|A4_ledger_untouched_n363_d637e53f|closed@3202",
    "evidence_level": "report_only_closure_confirmation",
    "model_scope": "n/a",
    "prereg_id": "G5-G12",
    "superseded_by": None,
    "detail": (
        "G5-G12 FTR-04 glm4 weak-specificity attribution closed-loop "
        "confirmation arm (report-only, zero GPU; default prereg arm (a), "
        "per MEMO Phase 3211 tail). Pre-freeze probe found the attribution "
        "ALREADY closed on the books — 3181 G5-B6 dual-arm noise_level "
        "verdict (registry v1.7->v1.8) + 3202 G5-F1 unit-level "
        "decomposition recheck (registry v1.24->v1.25) — so the arm ran as "
        "closure confirmation: A1 3181 sealed result (res 1e95a19d / seal "
        "462fd221 / design 7c567972, verdict noise_level) per_model "
        "scalars BITWISE equal to FTR-04 p3181 anchor asserts (kappa "
        "48118411.98113118/46016538.162839554/44953766.487345025, "
        "proj_ratio.glm4 0.029408887325559345, mean_margin.glm4 "
        "5.23258577598124, gates Ga/Gb false); A2 3202 sealed result (res "
        "af81e9f4 / seal 480bd271 / design 822996d4) decomposition "
        "re-derived from 3181 per_model within 1e-12 (loeo_gap "
        "0.1180952380952382 = 14b 0.5980952381 - glm4 0.4800000000; "
        "margin_ratio 1.9661111513569065 = 5.23258577598124 / "
        "2.661388585467299) + registry v1.28 FTR-04 scope_limits 3202 "
        "append verbatim + counter_evidence acc triple 0.6019/0.5981/"
        "0.4800 == per_model 4dp; A3 chain complete (3176 arm_a FAILED "
        "gate 0.6019/0.5981/0.4800 vs 0.5 -> 3181 noise_level "
        "_not_attributable -> 3202 unit-level decompose readout-"
        "instability-not-geometry), E1_repeatable maintained, MEMO 3181/"
        "3202 verbatim anchors on disk; A4 ledger untouched (n=363, "
        "d637e53f before==after). Disposition: FTR-04 glm4 attribution "
        "closed@3202, no registry action needed (closure text already in "
        "v1.28 scope_limits), drop from candidate follow-up list. Verdict "
        "ftr04_glm4_attribution_closed, A1-A4 0 contradictions. Honest "
        "log: SMOKE run1 DIRTY(1) — A2 scope_limits fragment lowercase "
        "'loeo' vs actual uppercase 'LOEO' (casing only); run2 SMOKE "
        "CLEAN; FULL clean. design b2e6a642, res 07041998 / seal 73024f1a."),
    "chain": {
        "res_sha8": "07041998",
        "seal_sha8": "73024f1a",
        "design_sha8": "b2e6a642",
        "r3181": "5ab51de6 (res 1e95a19d/seal 462fd221)",
        "r3202": "af7d8a70 (res af81e9f4/seal 480bd271)",
        "registry_v128": "5995bbe9",
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 364
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3212', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3212', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:363]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 364
print('ledger #364 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3212 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3212 \xe2\x80\x94 G5-G12' not in mraw:
    ADD = """

## Phase 3212 — G5-G12 FTR-04 glm4 弱特异性归因闭环确认臂（report-only，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 两跑（首跑 DIRTY(1) 大小写、修后 CLEAN），FULL 一次收官**。零 GPU、report-only（A4 ledger 未动断言）。design `b2e6a642`，res `07041998` / seal `73024f1a`，verdict **ftr04_glm4_attribution_closed**。

### 关键探针结论（预冻结）
FTR-04 glm4 弱特异性归因**在册已闭环**：3181 G5-B6 双臂定判 noise_level（registry v1.7→v1.8）+ 3202 G5-F1 单元级分解复读（registry v1.24→v1.25，FTR-04 scope_limits append）。本臂转为**闭环确认复读**，不升版。

### 确认结论（A1–A4 全过）
- **A1**：3181 封存产物（res `1e95a19d` / seal `462fd221` / design `7c567972`，verdict noise_level）per_model 标量与 FTR-04 p3181 锚 asserts **逐位一致**（κ 48118411.98113118/46016538.162839554/44953766.487345025、proj_ratio.glm4 0.029408887325559345、mean_margin.glm4 5.23258577598124、Ga/Gb 均 false）。
- **A2**：3202 封存产物（res `af81e9f4` / seal `480bd271` / design `822996d4`）decomposition 从 3181 标量 1e-12 内重推导成功——loeo_gap 0.1180952380952382（=14b 0.5980952381 − glm4 0.4800000000）、margin_ratio 1.9661111513569065（=5.23258577598124/2.661388585467299）；registry v1.28 scope_limits 3202 append 逐字在盘 + counter_evidence acc 三元组 0.6019/0.5981/0.4800 == per_model 4dp。
- **A3**：定判链条完整——3176 arm_a FAILED gate（acc 0.6019/0.5981/0.4800 vs 0.5 门）→ 3181 noise_level_not_attributable → 3202 读出不一致非几何缺陷（margin 5.2326 = qwen 最大 2.6614 的 1.966 倍，几何带健康：κ 比差 <25%、pr 比差 <20%、e90 同为 8）；**E1_repeatable 维持**；MEMO 3181/3202 verbatim 锚全在盘。
- **A4**：ledger 未动（n=363，d637e53f before==after）。

### disposition
FTR-04 glm4 归因 **closed@3202**：无需 registry 动作（闭环文本已在 v1.28 scope_limits），自候选跟进列表移除；弱特异性定性 = 读出不稳定（类对间隔分布散），非几何缺陷。

### 诚实日志
SMOKE run1 DIRTY(1)：A2 scope_limits 片段小写 `loeo` vs 实际大写 `LOEO`（仅大小写）；修后 run2 CLEAN、FULL CLEAN。封存数据未动。

### 预注册 Phase 3213（两支候选）
- (a) **UI v1.8 迭代**（非账本轮）：gap/appendix/N 线草案块对齐 v1.7 特征视图——锚 values/asserts 详情展开。
- (b) **G5-G13 FTR-15 held_out 复读臂**（零 GPU report-only）：回读 FTR-15 held_out 证据链（3204 v1.26 口径），评估升格/挂账处置。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3212' not in draw:
    dadd = ("\n\n### Phase 3212 G5-G12 FTR-04 glm4 归因闭环确认收官（report-only，2026-10-11）\n"
            "- 探针：归因在册已闭环（3181 noise_level + 3202 单元级分解复读），本臂转闭环确认；SMOKE 首跑 DIRTY(1)（LOEO 大小写）修后 CLEAN；FULL 一次收官。\n"
            "- A1 3181 p3181 锚逐位 + A2 3202 decomposition 重推导（0.1180952…/1.9661…，1e-12）+ A3 链条 3176FAIL→3181noise→3202decompose（E1_repeatable 维持）+ A4 ledger 未动。\n"
            "- disposition closed@3202，无 registry 动作；ledger #364（治理条目，rollback 等价验证）；预注册 3213：默认 (a) UI v1.8；(b) FTR-15 held_out 复读。\n")
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
- **3213 已预注册 = UI v1.8 迭代**（非账本轮，默认）：gap/appendix/N 线草案块加锚 values/asserts 详情展开（对齐 v1.7 特征视图）；候选 (b) G5-G13 FTR-15 held_out 复读臂（3204 v1.26 口径）。
- 3212 收官：G5-G12 FTR-04 glm4 归因闭环确认 REGISTERED（res 07041998/seal 73024f1a，design b2e6a642），ledger #364；closed@3202（3181 noise_level + 3202 分解复读，E1_repeatable 维持，无 registry 动作）。
- 3211 收官：UI v1.7 锚 values 视图 REGISTERED，atlas_client_v1.html 62ba1bd6（137,358 字节，611/611 字段），ledger #363；3210：水果类/K4 盘点 REGISTERED（#362），两项建议 close，**用户确认文档编辑待办**。
- 3209/3208/3207/3206：UI v1.6 bd230988（#361）；N 线草案 f29da2de（#360）；5-appendix 复读（#359）；FTR-22 链 + registry v1.28 5995bbe9 + atlas_v1_30 ebc1e122（#358）。
- 挂账不变：跨线账本补丁施加确认（APPX-5）；N 线补录施加（跨线确认后）；5 appendix GPU/跨线项；FTR-15 held_out（3213 候选）。水果类/K4/FTR-04 glm4 归因已闭环或待移除。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
