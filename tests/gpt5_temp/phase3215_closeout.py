# -*- coding: utf-8 -*-
"""Phase 3215 closeout: ledger #367 + MEMO + daily + MEMORY (report-only arm;
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
bako0 = json.loads(open(LP + '.bak_3215', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3215') else None
already = (bako0 is not None and len(disk0['measurements']) == 367
           and disk0['measurements'][366].get('phase') == 3215)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'f30394dc', prev_post
if already:
    raw = open(LP + '.bak_3215', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 366

entry = {
    "phase": 3215,
    "name": "g5g15_gap4_closed",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, report-only closure-confirmation arm)",
    "verdict": "g5g15_gap4_closed|REGISTERED|K1_scalars_bitwise_3169_2.8291_2.1258_2.7142_gt2_3172_k0_2.5388_k8_1.8002_3175_delta+0.0660_3192_amp_10.5984_7.1100_8.8909|K2_gap_v16_quantified_collapse_six_stage_chain_anchors|K3_memo_3169_3175_3192|K4_ledger_untouched_n366_f30394dc|mechanism_chain_closed",
    "evidence_level": "report_only_closure_confirmation",
    "model_scope": "n/a",
    "prereg_id": "G5-G15",
    "superseded_by": None,
    "detail": (
        "G5-G15 GAP-4 closed-loop confirmation arm (report-only, zero GPU; "
        "default prereg arm (a), per MEMO Phase 3214 tail). Pre-freeze "
        "probe found the GAP-4 mechanism chain ALREADY closed on the books "
        "(gap ledger v1.6 mechanism_note: 3169 quantified collapse -> 3171 "
        "encoding_missing negated -> 3172 port partial recovery -> 3175 "
        "residual port-intrinsic -> 3191 held_out -> 3192 amplitude "
        "carrier) so the arm ran as closure confirmation: K1 sealed "
        "scalars bitwise vs ledger evidence — 3169 per-model ratio_B "
        "2.8291123613715263/2.125753723276223/2.7142111478689683 all>2, "
        "3172 pooled curve k0 2.5388114997805062 -> k8 1.800176217907229 "
        "(band [1.5,2)), 3175 delta 0.06603206261005456 (|delta|<=0.15 "
        "port_residual_dominant), 3192 amp_ratio 10.598382949829102/"
        "7.110019207000732/8.890898704528809 all>=1.3; K2 gap v1.6 GAP-4 "
        "status quantified_collapse + six-stage mechanism_note phrases + "
        "anchor_sha8 six keys (p3169=5b51c2c1); K3 MEMO 3169/3175/3192 "
        "verbatim anchors; K4 ledger untouched (n=366, f30394dc "
        "before==after). Disposition: mechanism chain closed on the books; "
        "GAP-4 stays quantified_collapse BY DESIGN (collapse quantified + "
        "mechanism localized; full recovery not achieved and not pending); "
        "no registry/gap action; drop from candidate follow-up list. "
        "Verdict gap4_closed_loop_confirmed, K1-K4 0 contradictions. "
        "Honest log: SMOKE run1 crashed pre-gate (3172 pooled is a dict "
        "keyed by k-string, not a list — structure misread); fixed "
        "unpacking, run2 SMOKE CLEAN; FULL clean. design ecfead62, res "
        "1d2f037e / seal 254bff7c."),
    "chain": {
        "res_sha8": "1d2f037e",
        "seal_sha8": "254bff7c",
        "design_sha8": "ecfead62",
        "r3169": "5b51c2c1 (res 49430a39/seal 018c6024)",
        "r3172": "res 463f42c8/seal cdb85525",
        "r3175": "res cb76a177/seal 9e1a2f08",
        "r3192": "res 6221fe00/seal 58c4f183",
        "gl16": "0f6423e0",
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 367
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3215', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3215', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:366]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 367
print('ledger #367 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3215 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3215 \xe2\x80\x94 G5-G15' not in mraw:
    ADD = """

## Phase 3215 — G5-G15 GAP-4 闭环确认臂（report-only，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 两跑（首跑结构误读崩溃、修后 CLEAN），FULL 一次收官**。零 GPU、report-only（K4 ledger 未动断言）。design `ecfead62`，res `1d2f037e` / seal `254bff7c`，verdict **gap4_closed_loop_confirmed**。

### 关键探针结论（预冻结）
GAP-4 机制链**在册已闭环**（gap v1.6 mechanism_note 六段）：3169 量化 → 3171 encoding_missing 否定 → 3172 端口部分恢复 → 3175 残留=端口固有 → 3191 held_out → 3192 幅度载体。本臂转为**闭环确认复读**，不升版。

### 确认结论（K1–K4 全过）
- **K1**：四件封存产物 res/seal 锁定 + 标量与 gap ledger evidence **逐位一致**——3169 per-model ratio_B 2.8291123613715263 / 2.125753723276223 / 2.7142111478689683（全 >2 崩塌门）；3172 pooled 曲线 k0 2.5388114997805062 → k8 1.800176217907229（[1.5,2) borderline 带）；3175 delta 0.06603206261005456（≤0.15，port_residual_dominant）；3192 amp_ratio 10.598382949829102 / 7.110019207000732 / 8.890898704528809（全 ≥1.3）。
- **K2**：gap v1.6 GAP-4 status=quantified_collapse + mechanism_note 六段关键短语全在盘（mixed_across_models / borderline_partial_recovery / port_residual_dominant / 机制链闭环句 / u*_amplitude_carrier）+ anchor_sha8 六键（p3169=5b51c2c1）。
- **K3**：MEMO 3169/3175/3192 节 verbatim 锚。
- **K4**：ledger 未动（n=366，f30394dc before==after）。

### disposition
GAP-4 机制链**closed on the books**：崩塌已量化（2.54×）+ 机制已定位（one-hot 端口 ~29% + 端口固有残留 ~1.8× + u* 幅度载体）；**GAP-4 保持 quantified_collapse 是设计内状态**（全恢复未达成且非待办）——无 registry/gap 动作，自候选跟进列表移除。

### 诚实日志
SMOKE run1 门前置崩溃：3172 pooled 实为 k-字符串键 dict 而非 list（结构误读 `e['k']`）；修 unpack 后 run2 CLEAN、FULL CLEAN。封存数据未动。

### 预注册 Phase 3216（两支候选）
- (a) **G5-G16 挂账总账 v2 终审**（零 GPU report-only）：汇总 3210–3215 闭环结果 + 5 appendix + 跨线项，产出挂账清单 v2 处置表（水果类/K4 列表编辑仍待用户确认）。
- (b) **UI v1.9**：版本链/账本尾块锚详情展开（对齐 v1.8 块视图）。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3215' not in draw:
    dadd = ("\n\n### Phase 3215 G5-G15 GAP-4 闭环确认收官（report-only，2026-10-11）\n"
            "- 探针：机制链在册已闭环（gap v1.6 mechanism_note 六段），本臂转闭环确认；SMOKE run1 pooled 结构误读崩溃、修后 CLEAN；FULL 一次 CLEAN。\n"
            "- K1 四产物标量逐位（3169 ratios >2 / 3172 曲线 2.5388→1.8002 / 3175 delta +0.0660 / 3192 amp ≥1.3 ×3）+ K2 quantified_collapse 六段链 + K3 MEMO 锚 + K4 ledger 未动。\n"
            "- disposition：机制链 closed on the books，GAP-4 保持 quantified_collapse 属设计内；ledger #367（治理条目，rollback 等价验证）；预注册 3216：默认 (a) 挂账总账 v2 终审；(b) UI v1.9。\n")
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
- **3216 已预注册 = G5-G16 挂账总账 v2 终审**（零 GPU report-only，默认）：汇总 3210–3215 闭环 + 5 appendix + 跨线项，产出挂账清单 v2 处置表（水果类/K4 列表编辑仍待用户确认）；候选 (b) UI v1.9。
- 3215 收官：G5-G15 GAP-4 闭环确认 REGISTERED（res 1d2f037e/seal 254bff7c，design ecfead62），ledger #367；机制链 closed on the books（3169→3171→3172→3175→3192），GAP-4 保持 quantified_collapse 属设计内。
- 3214/3212：FTR-15 held_out closed@3204（#366）；FTR-04 glm4 closed@3202（#364）——归因类挂账全部闭环。
- 3213/3211：UI v1.8 ca76ec49 670/670（#365）；UI v1.7 62ba1bd6（#363）。
- 3210：水果类/K4 盘点（#362，**列表删除待用户确认**）；3209/3208/3207/3206：UI v1.6 bd230988（#361）；N 线草案 f29da2de（#360）；5-appendix 复读（#359）；FTR-22 链 + registry v1.28 5995bbe9（#358）。
- 挂账不变：跨线账本补丁施加确认（APPX-5）；N 线补录施加（跨线确认后）；5 appendix GPU/跨线项。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
