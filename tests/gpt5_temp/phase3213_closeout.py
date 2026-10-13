# -*- coding: utf-8 -*-
"""Phase 3213 closeout: ledger #365 + MEMO + daily + MEMORY (non-ledger arm;
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
bako0 = json.loads(open(LP + '.bak_3213', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3213') else None
already = (bako0 is not None and len(disk0['measurements']) == 365
           and disk0['measurements'][364].get('phase') == 3213)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '63a42cbc', prev_post
if already:
    raw = open(LP + '.bak_3213', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 364

entry = {
    "phase": 3213,
    "name": "g5g13_ui_v18_block_anchor_details",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, client build arm)",
    "verdict": "g5g13_ui_v18_block_anchor_details|REGISTERED|C1_zero_script_no_handlers_no_js_uri|C2_census_E2x20_E3x3_E1x3|C3_blocks_gaps4_appendix5_ndraft5_ftr01_26|C4_datak_670_fields_611carried_59block_details|C5_ledger_untouched_n364_63a42cbc|atlas_client_v1.html_ca76ec49",
    "evidence_level": "client_build_report",
    "model_scope": "n/a",
    "prereg_id": "G5-G13",
    "superseded_by": None,
    "detail": (
        "G5-G13 UI v1.8 block anchor-details arm (non-ledger, zero GPU; "
        "default prereg arm (a), per MEMO Phase 3212 tail). Iterates the "
        "v1.7 client (phase 3211, 62ba1bd6): extends the anchor-details "
        "view to non-feature blocks — gap evidence_anchors (11 fields, "
        "per-anchor data-k) + gap anchor_sha8 per-key rows (15); appendix "
        "anchor/anchor_sha8 as data-k spans (10); N-line draft per-entry "
        "name + per-key sources (5+15) + new draft meta block status/"
        "target/governance (3) = 59 new fields; feature anchor values "
        "view (61 anchors / 352 asserts) carried over. Output "
        "atlas_client_v1.html (ca76ec49, 141577 bytes, up from 137358). "
        "C1 zero <script> / no on* / no javascript: URI; C2 census 26 "
        "(E2 20 / E3 3 / E1 3); C3 blocks present; C4 data-k verify "
        "670/670 (0/0/0/0); C5 ledger untouched (n=364, 63a42cbc "
        "before==after). Verdict ui_v18_block_anchor_details_registered, "
        "C1-C5 0 contradictions. Honest log: SMOKE run1 DIRTY(1) — gap "
        "evidence_anchors row wrapped span() concatenation in esc(), "
        "escaping the emitted data-k spans (11 missing / 4 mismatch); "
        "removed outer esc (values already escaped inside span); run2 "
        "SMOKE CLEAN; FULL clean. design e5a9ed86, res a398c290 / seal "
        "095624ed."),
    "chain": {
        "res_sha8": "a398c290",
        "seal_sha8": "095624ed",
        "design_sha8": "e5a9ed86",
        "client_html": "ca76ec49",
        "client_v17": "62ba1bd6",
        "client_v16": "bd230988",
        "registry_v128": "5995bbe9",
        "ledger_pre": prev_post,
        "note": "non-ledger arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 365
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3213', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3213', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:364]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 365
print('ledger #365 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3213 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3213 \xe2\x80\x94 G5-G13' not in mraw:
    ADD = """

## Phase 3213 — G5-G13 UI v1.8 块级锚详情（非账本轮，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 两跑（首跑 DIRTY(1) esc 嵌套、修后 CLEAN），FULL 一次收官**。零 GPU、非账本轮（C5 ledger 未动断言）。产物：**`atlas_client_v1.html`（`ca76ec49`，141,577 字节）**，data-k **670/670** 字段全过（0/0/0/0）。design `e5a9ed86`，res `a398c290` / seal `095624ed`。

### v1.8 新增能力（零 script 保持；v1.7 特征锚视图 61/352 原样保留）
- **gap 块**：evidence_anchors 逐锚 data-k（11 字段）+ anchor_sha8 逐键行（15 字段，原为一行 esc 拼接）。
- **appendix 块**：anchor / anchor_sha8 升级为 data-k span（10 字段，原为纯 esc 文本）。
- **N 线草案块**：逐条 name + sources 逐键 data-k（20 字段）；新增**草案元数据面板**（status / target / governance，3 字段——含「APPX-4/5 同型：只补不改；施加需跨线确认」治理口径上屏）。
- 合计 59 新字段：611（v1.7）→ **670**。

### 诚实日志
SMOKE 首跑 DIRTY(1)：gap evidence_anchors 行把 span() 拼接结果再套 esc()，发出标签被转义（11 missing / 4 mismatch）；去除外层 esc（span 内部已对值转义）后 run2 CLEAN、FULL CLEAN。封存数据未动。

### 预注册 Phase 3214（两支候选）
- (a) **G5-G14 FTR-15 held_out 复读臂**（零 GPU report-only）：回读 FTR-15 held_out 证据链（3204 registry v1.25→v1.26 口径），评估升格/挂账处置。
- (b) **UI v1.9**：版本链块逐版 scope 附 FTR 名单展开 + 账本尾条目锚详情（对齐 v1.8 块视图）。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3213' not in draw:
    dadd = ("\n\n### Phase 3213 G5-G13 UI v1.8 块级锚详情收官（非账本轮，2026-10-11）\n"
            "- SMOKE 首跑 DIRTY(1)（evidence_anchors 行 esc 嵌套转义了 data-k span；去外层 esc 后 CLEAN）；FULL 一次收官：C4 670/670（611 carried + 59 块级新字段）。\n"
            "- v1.8 增量：gap evidence_anchors 逐锚 + anchor_sha8 逐键、appendix anchor/anchor_sha8 data-k、草案 name/sources 逐键 + 元数据面板（status/target/governance）。\n"
            "- 产物 atlas_client_v1.html ca76ec49（141,577 字节）；ledger #365（治理条目，rollback 等价验证）；预注册 3214：默认 (a) FTR-15 held_out 复读；(b) UI v1.9。\n")
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
- **3214 已预注册 = G5-G14 FTR-15 held_out 复读臂**（零 GPU report-only，默认）：回读 FTR-15 held_out 证据链（3204 v1.25→v1.26 口径），评估升格/挂账处置；候选 (b) UI v1.9（版本链/账本尾锚详情展开）。
- 3213 收官：UI v1.8 块级锚详情 REGISTERED，atlas_client_v1.html `ca76ec49`（141,577 字节，670/670 = 611 carried + 59 块级字段），ledger #365。
- 3212/3211/3210：FTR-04 glm4 归因 closed@3202（#364，res 07041998）；UI v1.7 62ba1bd6 611/611（#363）；水果类/K4 盘点 REGISTERED（#362），**两项列表删除待用户确认**。
- 3209/3208/3207/3206：UI v1.6 bd230988（#361）；N 线草案 f29da2de（#360）；5-appendix 复读（#359）；FTR-22 链 + registry v1.28 5995bbe9 + atlas_v1_30 ebc1e122（#358）。
- 挂账不变：跨线账本补丁施加确认（APPX-5）；N 线补录施加（跨线确认后）；5 appendix GPU/跨线项。（水果类/K4/FTR-04 glm4 归因已闭环或待移除）

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
