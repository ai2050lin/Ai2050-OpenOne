# -*- coding: utf-8 -*-
"""Phase 3211 closeout: ledger #363 + MEMO + daily + MEMORY (non-ledger arm;
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
bako0 = json.loads(open(LP + '.bak_3211', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3211') else None
already = (bako0 is not None and len(disk0['measurements']) == 363
           and disk0['measurements'][362].get('phase') == 3211)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == '1aaab36d', prev_post
if already:
    raw = open(LP + '.bak_3211', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 362

entry = {
    "phase": 3211,
    "name": "g5g10_ui_v17_anchor_values",
    "line": "G",
    "date": TODAY,
    "model": "n/a (zero GPU, client build arm)",
    "verdict": "g5g10_ui_v17_anchor_values|REGISTERED|C1_zero_script_no_handlers_no_js_uri|C2_census_E2x20_E3x3_E1x3|C3_blocks_gaps4_appendix5_ndraft5_ftr01_26|C4_datak_611_fields_61anchors_352asserts|C5_ledger_untouched_n362_1aaab36d|atlas_client_v1.html_62ba1bd6",
    "evidence_level": "client_build_report",
    "model_scope": "n/a",
    "prereg_id": "G5-G10",
    "superseded_by": None,
    "detail": (
        "G5-G10 UI v1.7 anchor-values view arm (non-ledger, zero GPU; "
        "default prereg arm (a), per MEMO Phase 3210 tail). Iterates the "
        "v1.6 client (phase 3209, bd230988) with the per-feature RAW ANCHOR "
        "VALUES view: every registry v1.28 anchor's asserts dict rendered "
        "verbatim inside each feature foldout — 61 anchors (src/phase/role/"
        "tags each) and all 352 asserts (key = value, sorted) across 26 "
        "features, zero script preserved. Output atlas_client_v1.html "
        "(62ba1bd6, 137358 bytes, up from 73507). C1 zero <script> / no "
        "on* handlers / no javascript: URI; C2 census E2 (17 E2_predictive "
        "+ 3 E2_held_out) = 20 / E3_causal_scoped 3 / E1_repeatable 3; C3 "
        "blocks 4/5/5 + FTR-01..26 + APPX-1..5 present; C4 data-k verify "
        "611/611 fields (137 base + 61x2 anchor src/role + 352 asserts; "
        "0/0/0/0); C5 ledger untouched (n=362, 1aaab36d before==after). "
        "Cosmetic fix carried: v1.6 filter label 'E3_protocol_causal' "
        "corrected to 'E3_causal_scoped' (display-only, data-k unaffected). "
        "Verdict ui_v17_anchor_values_registered, C1-C5 0 contradictions. "
        "Honest log: SMOKE run1 CLEAN once (611/611); one run-banner log "
        "string still read Phase 3209 after patch — fixed (log text only, "
        "post-observation, no DESIGN/gate change) before FULL. design "
        "93fdcc0c, res ee976567 / seal e1a67926."),
    "chain": {
        "res_sha8": "ee976567",
        "seal_sha8": "e1a67926",
        "design_sha8": "93fdcc0c",
        "client_html": "62ba1bd6",
        "client_v16": "bd230988",
        "registry_v128": "5995bbe9",
        "ledger_pre": prev_post,
        "note": "non-ledger arm; governance entry only, atlas data untouched",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 363
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3211', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3211', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:362]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 363
print('ledger #363 ok (entry pre', prev_post, ') -> post', new_post,
      '(bak_3211 saved, rollback-equal)')

MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3211 \xe2\x80\x94 G5-G10' not in mraw:
    ADD = """

## Phase 3211 — G5-G10 UI v1.7 锚 values 视图（非账本轮，2026-10-11）

**预注册臂 (a) 执行完毕，SMOKE 一次 CLEAN，FULL 一次收官**。零 GPU、非账本轮（C5 ledger 未动断言）。产物：**`atlas_client_v1.html`（`62ba1bd6`，137,358 字节，v1.6 73,507 → 翻倍）**，data-k **611/611** 字段全过（0/0/0/0）。design `93fdcc0c`，res `ee976567` / seal `e1a67926`。

### v1.7 新增能力（零 script 保持）
- **per-feature 原始锚 values 视图**：registry v1.28 全部 **61 锚**（src/phase/role/tags 逐锚）+ **352 asserts**（key = value 全量逐条，键序 sorted）渲染进 26 特征折叠面板——此前 v1.6 仅显示锚计数。
- **C4 扩容**：137 基础字段 + 61×2（锚 src/role）+ 352（asserts 逐条）= 611 字段 data-k 校验全过；asserts 数值经 `json.dumps` 保真（bool 小写、浮点全精度）。
- **C1**：全文无 `<script>`、无 on* 处理器、无 javascript: URI（details/radio 过滤/锚点导航不变）。
- **顺带修正**：v1.6 过滤器显示标签 `E3_protocol_causal` → `E3_causal_scoped`（纯显示层，data-k 未涉及，诚实入档）。

### 诚实日志
SMOKE 一次 CLEAN（611/611）；补丁后残留一处 run 横幅 log 文案仍写 Phase 3209——FULL 前修正（仅日志文本，观测后改动，不涉及 DESIGN/判据/封存数据）。

### 预注册 Phase 3212（两支候选）
- (a) **G5-G12 FTR-04 glm4 弱特异性归因臂**（零 GPU report-only）：回读 FTR-04 glm4 特异性证据链（registry v1.5→v1.6 升格口径），评估升格/挂账/appendix 处置。
- (b) **UI v1.8**：gap/appendix/N 线草案块同样加锚 values/asserts 详情展开（对齐 v1.7 特征视图）。
- 默认执行 (a)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

DP = r'.workbuddy\memory\2026-10-11.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3211' not in draw:
    dadd = ("\n\n### Phase 3211 G5-G10 UI v1.7 锚 values 视图收官（非账本轮，2026-10-11）\n"
            "- SMOKE 一次 CLEAN；FULL 一次收官：C1 零 script + C2 census 20/3/3 + C3 blocks 4/5/5 + C4 data-k 611/611（137+61×2+352 asserts）+ C5 ledger 未动。\n"
            "- 产物 atlas_client_v1.html 62ba1bd6（137,358 字节，v1.6 翻倍）：26 特征折叠面板内嵌全部 61 锚 + 352 asserts 全量逐条；顺带修正 E3 过滤器标签。\n"
            "- ledger #363（治理条目，rollback 等价验证）；预注册 3212：默认 (a) FTR-04 glm4 归因复读；(b) UI v1.8 gap/appendix 锚详情。\n")
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
- **3212 已预注册 = G5-G12 FTR-04 glm4 弱特异性归因臂**（零 GPU report-only，默认）：回读 FTR-04 glm4 特异性证据链，评估升格/挂账/appendix 处置；候选 (b) UI v1.8（gap/appendix/N 草案锚详情展开）。
- 3211 收官：G5-G10 UI v1.7 锚 values 视图 REGISTERED，atlas_client_v1.html `62ba1bd6`（137,358 字节，611/611 字段 = 137+61×2+352 asserts，零 script），ledger #363；v1.6 bd230988 之上翻倍。
- 3210 收官：G5-G9 水果类/K4 挂账盘点 REGISTERED（res 3ae23744/seal 23008571），ledger #362；两项建议 close，**用户确认文档编辑待办**（挂账列表删除两项）。
- 3209/3208/3207/3206：UI v1.6 bd230988（#361）；N 线草案 f29da2de（#360）；5-appendix 复读（#359）；FTR-22 链 + registry v1.28 5995bbe9 + atlas_v1_30 ebc1e122（#358）。
- 挂账不变：跨线账本补丁施加确认（APPX-5）；N 线补录施加（跨线确认后）；5 appendix GPU/跨线项。（水果类/K4 待用户确认后自列表移除）

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
