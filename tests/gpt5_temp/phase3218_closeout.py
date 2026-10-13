# -*- coding: utf-8 -*-
"""Phase 3218 closeout: ledger #370 + MEMO + daily + MEMORY (report-only arm;
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
bako0 = json.loads(open(LP + '.bak_3218', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3218') else None
already = (bako0 is not None and len(disk0['measurements']) == 370
           and disk0['measurements'][369].get('phase') == 3218)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '7978e4f7', prev_post
if already:
    raw = open(LP + '.bak_3218', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 369

entry = {
    "phase": 3218,
    "name": "g5g20_crossline_apply_readiness",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, report-only readiness inventory)",
    "verdict": "crossline_apply_readiness_registered|R1_nline_5_entries_itemized|R2_lcorr_4_ops_itemized_stale_baseline_finding_bbda63df_ne_7978e4f7|R3_checklist_sealed_e402da6e|R4_ledger_untouched_n369_7978e4f7|no_application_performed",
    "evidence_level": "report_only_readiness_inventory",
    "model_scope": "n/a",
    "prereg_id": "G5-G20",
    "superseded_by": None,
    "detail": (
        "G5-G20 cross-line apply readiness inventory arm (report-only, "
        "zero GPU, NO application; default prereg arm (a), per MEMO "
        "Phase 3217 tail). R1: N-line backfill draft f29da2de (5 entries "
        "N-P3..N-P7, all draft_pending_cross_line) itemized entry-wise "
        "with per-entry source keys; plan = single append batch of 5 to "
        "atlas_ledger measurements + one cross-line confirmation + bak/"
        "rollback. R2: APPX-5 ledger_corrections_v1 beb775bd "
        "(SEALED_BUT_NOT_APPLIED) 4 operations itemized (set_field "
        "ledger_sha256_8, unify_schema meas_id+phase n_total=304, "
        "enumerate_field evidence_level, append_measurement A-gate "
        "entry); CRITICAL FINDING: patch precondition "
        "target_sha256_before=bbda63df vs current ledger 7978e4f7 => "
        "baseline STALE, re-baseline required before any apply. R3: "
        "crossline_apply_checklist_v1.json sealed, content sha e402da6e "
        "(file d3f8c544), includes per-item apply_plan steps. R4: "
        "ledger untouched (n=369, 7978e4f7 before==after). Verdict "
        "crossline_apply_readiness_registered, R1-R4 0 contradictions. "
        "Honest log: SMOKE/FULL both clean first pass; stale-baseline "
        "finding is a checklist fact (patch predates 300+ governance "
        "entries), NOT a contradiction. design a6d5cdb5, res 976391dc / "
        "seal cd7eebe2."),
    "chain": {
        "res_sha8": "976391dc",
        "seal_sha8": "cd7eebe2",
        "design_sha8": "a6d5cdb5",
        "checklist": "e402da6e (file d3f8c544)",
        "draft": "f29da2de", "lcorr": "beb775bd",
        "ledger_pre": prev_post,
        "note": "report-only arm; governance entry only, atlas data untouched; "
                "no cross-line application performed",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 370
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3218', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3218', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:369]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 370
print('ledger #370 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3218 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3218 \xe2\x80\x94 G5-G20' not in mraw:
    ADD = """

## Phase 3218 — G5-G20 跨线施加就绪盘点（report-only，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE/FULL 各一次全 CLEAN**。零 GPU、report-only、**未做任何施加**（R4 ledger 未动断言）。design `a6d5cdb5`，res `976391dc` / seal `cd7eebe2`，verdict **crossline_apply_readiness_registered**；封存清单 `crossline_apply_checklist_v1.json` 内容 sha **`e402da6e`**（文件 `d3f8c544`）。

### 盘点结论（R1–R4 全过，0 矛盾）
- **R1 N 线补录**：草案 `f29da2de`（5 条 N-P3..N-P7，全 draft_pending_cross_line）逐条在册 + per-entry sources 键；施加方案 = 一次跨线确认 → bak → 单批 append 5 条 → 重算 post-sha → rollback 可回。
- **R2 APPX-5 补丁**：`ledger_corrections_v1` `beb775bd`（SEALED_BUT_NOT_APPLIED）4 操作逐条在册（set_field ledger_sha256_8 / unify_schema meas_id+phase n_total=304 / enumerate_field evidence_level / append_measurement A-gate 条目）；**关键发现：补丁前置 `target_sha256_before=bbda63df` 与当前账本 `7978e4f7` 不符 ⇒ 基线过期，施加前必须重基线**（对当前账本状态重算操作参数）。
- **R3**：清单封存（含逐项 apply_plan 步骤）；**R4**：ledger 未动（n=369，`7978e4f7` before==after）。

### disposition
跨线两项施加就绪状态定级：N 线 = **ready（基线无关，append 型）**；APPX-5 = **not-ready（基线过期，需重基线后方可施加）**。施加本身仍需用户跨线确认，本臂零变更。

### 预注册 Phase 3219（两支候选）
- (a) **UI v1.10**：挂账清单 v2（standing_ledger_v2 `207f2fa6`）+ 施加就绪状态作为客户端新块渲染（closed/open/pending/ready/not-ready 处置表，零 script）。
- (b) **G5-G21 5-appendix 复核**：对 APPX-1..3（GPU 型）做在盘锚哈希复读，确认零 GPU 不可关结论仍然成立（report-only）。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3218' not in draw:
    dadd = ("\n\n### Phase 3218 G5-G20 跨线施加就绪盘点收官（report-only，2026-10-11）\n"
            "- R1 N 线 5 条逐条在册（append 型，ready）+ R2 APPX-5 4 操作逐条在册，关键发现补丁基线 bbda63df 已过期（现 7978e4f7）⇒ not-ready 需重基线 + R3 清单封存 crossline_apply_checklist_v1 e402da6e + R4 ledger 未动。\n"
            "- SMOKE/FULL 一次全 CLEAN；ledger #370（治理条目，rollback 等价验证）；预注册 3219：默认 (a) UI v1.10；(b) 5-appendix 复核。\n")
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
- **3219 已预注册 = (a) UI v1.10**：挂账清单 v2 + 施加就绪状态客户端新块渲染（零 script）；候选 (b) G5-G21 5-appendix 在盘锚复读（report-only）。
- 3218 收官：G5-G20 跨线施加就绪盘点 REGISTERED（res 976391dc/seal cd7eebe2，design a6d5cdb5），ledger #370；checklist e402da6e。定级：N 线 ready（append 型）；APPX-5 not-ready（基线 bbda63df 过期 vs 现 7978e4f7，需重基线）。施加待用户跨线确认。
- 3217/3216：UI v1.9 9c5b48da 698/698（#369）；挂账总账 v2 终审（#368，standing_ledger_v2.json 207f2fa6：closed 5/open 4/pending 1，**水果类/K4 列表删除待用户确认**）。
- 3213/3211/3210：UI v1.8 ca76ec49（#365）；UI v1.7 62ba1bd6（#363）；水果类/K4 盘点（#362）。
- 3209/3208/3207/3206：UI v1.6 bd230988（#361）；N 线草案 f29da2de（#360）；5-appendix 复读（#359）；FTR-22 链 + registry v1.28 5995bbe9（#358）。
- 挂账不变：APPX-5 施加（重基线后，待确认）；N 线施加（ready，待确认）；5 appendix GPU/跨线项。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
