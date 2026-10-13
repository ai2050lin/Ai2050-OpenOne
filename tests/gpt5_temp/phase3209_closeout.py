# -*- coding: utf-8 -*-
"""Phase 3209 closeout: ledger #361 + MEMO + daily + MEMORY (non-ledger arm;
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
bako0 = json.loads(open(LP + '.bak_3209', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3209') else None
already = (bako0 is not None and len(disk0['measurements']) == 361
           and disk0['measurements'][360].get('phase') == 3209)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '383a05db', prev_post
if already:
    raw = open(LP + '.bak_3209', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 360

entry = {
    "phase": 3209,
    "name": "g5g8_ui_v16_client",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, client build arm)",
    "verdict": "g5g8_ui_v16_client|REGISTERED|C1_zero_script_no_handlers_no_js_uri|C2_census_E2x20_E3x3_E1x3|C3_blocks_gaps4_appendix5_ndraft5_ftr01_26|C4_datak_137_fields_clean|C5_ledger_untouched_n360_383a05db|atlas_client_v1.html_bd230988",
    "evidence_level": "client_build_report",
    "model_scope": "n/a",
    "prereg_id": "G5-G8",
    "superseded_by": None,
    "detail": (
        "G5-G8 UI v1.6 zero-script client build arm (non-ledger, zero GPU; "
        "default prereg arm (a)). Single-file pure HTML/CSS client "
        "atlas_client_v1.html (bd230988, 73507 bytes) embedding the atlas "
        "v1.30 data plane: registry v1.28 (5995bbe9) 26 features with "
        "family-grouped cards + details foldout (anchors/counter_evidence/"
        "scope_limits/replication) + CSS radio :checked evidence-level "
        "filtering (E2 20 / E3 3 / E1 3) + sticky anchor nav; gap ledger "
        "v1.6 (0f6423e0) 4 gaps with closed badges + GAP-4 mechanism_note; "
        "appendix x5 (gap v1.5 4fb3f37d) with anchor shas; N-line backfill "
        "draft x5 (f29da2de, draft_pending_cross_line, NOT applied); "
        "registry version chain v1.28..v1.21; ledger tail x5. C1 zero "
        "<script> / no on* handlers / no javascript: URI; C2 census E2 "
        "(17 E2_predictive + 3 E2_held_out) = 20 / E3_causal_scoped 3 / "
        "E1_repeatable 3; C3 blocks 4/5/5 + FTR-01..26 + APPX-1..5 present; "
        "C4 data-k verify 137/137 fields (0/0/0/0); C5 ledger untouched "
        "(n=360, 383a05db before==after). Verdict ui_v16_client_registered, "
        "C1-C5 0 contradictions. Honest log: SMOKE run1 DIRTY(2) — C2 "
        "expected literal keys (E3_protocol_causal) mismatched actual "
        "(E3_causal_scoped; E2 tier = 17+3) + C4 extra meta.created "
        "(expectation omitted); C2/CSS filter fixed to ^E2/^E3/^E1 prefix "
        "matching, created expectation added; run2 SMOKE CLEAN; FULL "
        "clean. design 9c444679, res eac7fe87 / seal fed536ad."),
    "chain": {
        "res_sha8": "eac7fe87",
        "seal_sha8": "fed536ad",
        "client_html": "bd230988",
        "registry_v128": "5995bbe9",
        "gl16": "0f6423e0",
        "draft": "f29da2de",
        "ledger_pre": prev_post,
        "note": "non-ledger arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 361
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3209', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3209', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:360]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 361
print('ledger #361 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3209 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3209 \xe2\x80\x94 G5-G8' not in mraw:
    ADD = """

## Phase 3209 — G5-G8 UI v1.6 客户端零 script 版（非账本轮，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 三跑 CLEAN（首跑 DIRTY(2) 判据/期望 bug，封存未动），FULL 一次收官**。零 GPU、非账本轮（C5 ledger 未动断言）。产物：**`atlas_client_v1.html`（`bd230988`，73,507 字节）**，data-k 137/137 字段全过（0/0/0/0）。design `9c444679`，res `eac7fe87` / seal `fed536ad`。

### 客户端能力（零 script：纯 HTML/CSS）
- **C1**：全文无 `<script>`、无 on* 内联处理器、无 javascript: URI——交互全部由 `<details>` 折叠 + CSS radio `:checked` 级联过滤 + sticky 锚点导航实现。
- **数据面**：registry v1.28 `5995bbe9` 26 特征（family 分组卡 + 锚/反证/范围/复现折叠）；gap v1.6 `0f6423e0` 4 条（closed 徽章 + GAP-4 mechanism_note）；appendix 5 条（gap v1.5 `4fb3f37d` 源，anchor_sha8 列）；N 线补录草案 5 条（`f29da2de`，draft_pending_cross_line，未施加标记）；registry 版本链 v1.28←v1.21；治理账本尾 5 条。
- **C2 census**：E2 档 20（17 E2_predictive + 3 E2_held_out）/ E3_causal_scoped 3 / E1_repeatable 3；CSS 过滤器前缀匹配（^E2/^E3/^E1）。
- **C4**：data-k 137 字段（26×(elevel+family+statement) + gap×3+mech + appendix×3 + ndraft×2 + 链 8 + meta×5 + 账本尾×5）verify 全过。

### 诚实日志
SMOKE 首跑 DIRTY(2)：C2 预期字面键（E3_protocol_causal）与实际（E3_causal_scoped；E2 档=17+3）不符 + C4 extra meta.created（期望字典漏收）——修判据与过滤器后二次 SMOKE CLEAN、FULL CLEAN；封存数据未动，客户端为新增产物。

### 预注册 Phase 3210（两支候选）
- (a) **G5-G9 水果类/K4 挂账盘点臂**（零 GPU report-only）：回读水果类与 K4 挂账历史出处与现状，评估可否关闭或转 appendix。
- (b) **UI v1.7 迭代**：客户端加 per-feature 原始锚 values 视图（registry anchors asserts 全量展示）。
- 默认执行 (a)；若挂账出处不可回读则降级 (b)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3209' not in draw:
    dadd = ("\n\n### Phase 3209 G5-G8 UI v1.6 零 script 客户端收官（非账本轮，2026-10-11）\n"
            "- SMOKE 首跑 DIRTY(2)（C2 census 键名/E2 档构成 + C4 meta.created 期望漏收；修判据后 CLEAN）；FULL 一次收官：C1 零 script + C2 census 20/3/3 + C3 blocks 4/5/5 + C4 data-k 137/137 + C5 ledger 未动。\n"
            "- 产物 atlas_client_v1.html bd230988（73,507 字节，纯 HTML/CSS，details/radio 过滤/锚点导航，内嵌 v1.28 数据面 + gap v1.6 + appendix + N 线草案 + 版本链 + 账本尾）。\n"
            "- ledger #361（治理条目，rollback 等价验证）；预注册 3210：默认 (a) 水果类/K4 挂账盘点；(b) UI v1.7 锚 values 视图。\n")
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
- **3210 已预注册 = G5-G9 水果类/K4 挂账盘点臂**（零 GPU report-only，默认）：回读水果类与 K4 挂账出处与现状，评估关闭或转 appendix；候选 (b) UI v1.7（per-feature anchors asserts 全量视图）。若挂账出处不可回读降级 (b)。
- 3209 收官：G5-G8 UI v1.6 零 script 客户端 REGISTERED，atlas_client_v1.html `bd230988`（73,507 字节，137/137 字段，纯 HTML/CSS），ledger #361；数据面 = registry v1.28 26 特征 + gap v1.6 + appendix 5 + N 线草案 5 + 版本链。
- 3208 收官：G5-G7 N 线 P3–P7 补录草案 REGISTERED（n_line_backfill_draft_v1.json f29da2de，draft_pending_cross_line 未施加），ledger #360；APPX-4 推进至「草案已封存待施加」。
- 3207 收官：G5-G6 5-appendix 复读 REPORT_ONLY_CLEAN，ledger #359；3206：FTR-22 升级链复读 REGISTERED（registry v1.28 5995bbe9 + atlas_v1_30 ebc1e122，ledger #358）。
- 挂账不变：跨线账本补丁施加确认（APPX-5）；N 线补录施加（跨线确认后）；水果类；K4。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
