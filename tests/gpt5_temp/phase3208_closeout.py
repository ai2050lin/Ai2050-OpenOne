# -*- coding: utf-8 -*-
"""Phase 3208 closeout: ledger #360 append + MEMO append + daily(10-11) +
MEMORY. Draft-only phase: backfill NOT applied; governance entry only.
"""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

TODAY = '2026-10-11'

# ---------- 1. ledger append (#360) ----------
LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3208', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3208') else None
already = (bako0 is not None and len(disk0['measurements']) == 360
           and disk0['measurements'][359].get('phase') == 3208)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'efe4a6de', prev_post
if already:
    raw = open(LP + '.bak_3208', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 359

entry = {
    "phase": 3208,
    "name": "g5g7_nline_backfill_draft",
    "line": "G",
    "date": TODAY,
    "model": "qwen3-4b+glm4-9b-chat-hf+qwen2.5-3b-instruct (zero GPU, deepseek-line re-read)",
    "verdict": "g5g7_nline_backfill_draft|DRAFT_REGISTERED|B1_15_sealed_sources_bitwise|B2_content_anchors_verbatim_x7_files|B3_ledger_untouched_n359_efe4a6de|5_draft_entries_N-P3..N-P7_draft_pending_cross_line|NOT_APPLIED_appx4_governance",
    "evidence_level": "report_only_draft_backfill",
    "model_scope": "deepseek N-line (qwen3-4b primary)",
    "prereg_id": "G5-G7",
    "superseded_by": None,
    "detail": (
        "G5-G7 N-line P3-P7 ledger backfill draft arm (draft-only, zero GPU; "
        "deepseek_temp Phase3-7 sealed artifacts re-read; default prereg arm "
        "(b), APPX-4 挂账联动). B1: 15 sealed source files sha bitwise "
        "(Phase3 e2_report d6de235f + memo 1d56f814 + wlog d0810382; Phase4 "
        "N1_design_seal bd05ba88 + memo cbe8452e + n1v2 report ad05f1f2; "
        "Phase5 memo b40de352 + n2f 248f0a46 + n2g d20cac53; Phase6 "
        "N2h1_design_seal 4d32b1aa + memo 373571ed + n2h1c 4fb86141; Phase7 "
        "N3_design_seal 66afc202 + memo 952f8833 + n3 51e19b7f). B2: content "
        "anchors verbatim (P3 'E2 条件化比较探针'; P4 seal main_axis_layer_scan/"
        "sense_split/ontology_margin; P5 memo 没有任何单头 / L31–L35; P6 report "
        "B_cat=+10.192 / B/A=0.958 / B_cat=+0.071 + seal 置换向量消融; P7 report "
        "'F2 color B/A=0.925 (lstar=L27)' + seal F1_is_a/F2_color/B_cat_loo). "
        "B3: shared atlas_ledger untouched (n=359, internal sha efe4a6de "
        "before==after, double assertion). Draft registered: "
        "n_line_backfill_draft_v1.json (f29da2de) with 5 entries N-P3..N-P7 "
        "(E2 条件化比较层扫描 / N1 主轴三段层扫描 / N2 重建源分布式重构 / N2h1 置换向量"
        "消类别分量承载 / N3 装置跨 L2/极性/语言族迁移), all "
        "draft_pending_cross_line — NOT applied (施加需跨线确认, APPX-4/5 "
        "governance). Verdict n_line_backfill_draft_registered, B1-B3 0 "
        "contradictions. SMOKE first run CLEAN (res 2ecbe4ca / seal "
        "03a52e6f); FULL clean (res 33cc81ac / seal 35b0345e). design "
        "2e96dae7."),
    "chain": {
        "res_sha8": "33cc81ac",
        "seal_sha8": "35b0345e",
        "draft_file": "f29da2de",
        "smoke_res_sha8": "2ecbe4ca",
        "smoke_seal_sha8": "03a52e6f",
        "ledger_pre": prev_post,
        "note": "backfill NOT applied; 5 entries pending cross-line confirmation",
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 360
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3208', 'wb').write(raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3208', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:359]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 360
print('ledger #360 ok, pre (entry pre', prev_post, ') -> post', new_post,
      '(bak_3208 saved, rollback-equal)')

# ---------- 2. MEMO append ----------
MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3208 \xe2\x80\x94 G5-G7' not in mraw:
    ADD = """

## Phase 3208 — G5-G7 N 线 P3–P7 补 Ledger 试臂（draft-only，2026-10-11）

**预注册臂 (b) 执行完毕，SMOKE 一次 CLEAN，FULL 一次收官**。零 GPU、draft-only（主账本零变更，双重未动断言）；SRC 15 文件（deepseek_temp Phase3–7 封存：P3 e2 report/memo/wlog `d6de235f`/`1d56f814`/`d0810382`；P4 N1 seal `bd05ba88` + memo `cbe8452e` + report `ad05f1f2`；P5 memo `b40de352` + n2f `248f0a46` + n2g `d20cac53`；P6 N2h1 seal `4d32b1aa` + memo `373571ed` + report `4fb86141`；P7 N3 seal `66afc202` + memo `952f8833` + report `51e19b7f`）。design `2e96dae7`，res `33cc81ac` / seal `35b0345e`（SMOKE res `2ecbe4ca` / seal `03a52e6f`）。

### 判定（B1–B3 全 CLEAN，0 contradictions）
- **B1**：15 封存源文件 sha 逐位全过。
- **B2**：内容锚 verbatim——P3「E2 条件化比较探针」；P4 seal `main_axis_layer_scan`/`sense_split`/`ontology_margin`；P5 memo「没有任何单头」+「L31–L35」；P6 report `B_cat=+10.192` / `B/A=0.958` / `B_cat=+0.071` + seal「置换向量消融」；P7 report「F2 color B/A=0.925 (lstar=L27)」+ seal `F1_is_a`/`F2_color`/`B_cat_loo`。
- **B3**：共享 atlas_ledger 未动（n=359、内部 sha `efe4a6de` before==after，双重断言）。

### 登记（draft-only）
**`n_line_backfill_draft_v1.json`（`f29da2de`）封存于 phase3208 输出目录**：5 条补录草案 N-P3..N-P7（E2 条件化比较层扫描 / N1 主轴三段层扫描 / N2 重建源分布式重构 / N2h1 置换向量消融-类别分量承载 / N3 装置跨 L2-极性-语言族迁移），全部 `draft_pending_cross_line`——**未施加**（施加需跨线确认，APPX-4/5 governance 同型）。ledger #360 仅记本 phase 治理活动，verdict `n_line_backfill_draft_registered`。

### APPX-4 挂账影响
N 线 P3–P7 补录由「未起草」推进为「草案已封存待施加」；施加动作仍挂账（跨线确认后一次性 append 5 条）。

### 预注册 Phase 3209（两支候选）
- (a) **UI v1.6 客户端零 script 版开工**（非账本轮）：图谱透镜客户端化（atlas v1.30 数据面 + registry v1.28 26 特征 + gap v1.6 + appendix 5 条 + N 线补录草案展示）。
- (b) **G5-G8 水果类/K4 挂账盘点臂**（零 GPU report-only）：回读水果类与 K4 挂账的历史出处与当前状态，评估是否可关闭或需转 appendix。
- 默认执行 (a)；用户如需继续账本线则切 (b)。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

# ---------- 3. daily (2026-10-11, create if absent) ----------
DP = r'.workbuddy\memory\2026-10-11.md'
if not os.path.exists(DP):
    open(DP, 'wb').write(('# 2026-10-11 工作日志\n').encode('utf-8'))
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
if b'### Phase 3208' not in draw:
    dadd = ("\n\n### Phase 3208 G5-G7 N 线 P3–P7 补 Ledger 试臂收官（draft-only，2026-10-11）\n"
            "- SMOKE 一次 CLEAN；FULL 一次收官：B1 15 封存源 sha 逐位 + B2 内容锚 verbatim ×7 文件 + B3 主账本未动双重断言（n=359 efe4a6de）。\n"
            "- 登记：n_line_backfill_draft_v1.json f29da2de（5 条 N-P3..N-P7 草案，draft_pending_cross_line，未施加，APPX-4 governance）；ledger #360 仅记治理活动。\n"
            "- APPX-4 推进：N 线补录由「未起草」→「草案已封存待施加」；施加仍挂账（跨线确认）。\n"
            "- 预注册 3209：默认 (a) UI v1.6 零 script 版开工；(b) 水果类/K4 挂账盘点。\n")
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
- **3209 已预注册 = UI v1.6 客户端零 script 版开工**（默认，非账本轮）：图谱透镜客户端化（atlas v1.30 数据面 + registry v1.28 26 特征 + gap v1.6 + appendix 5 条 + N 线补录草案展示）；候选 (b) 水果类/K4 挂账盘点臂（report-only）。
- 3208 收官：G5-G7 N 线 P3–P7 补录草案 REGISTERED（B1–B3 全 CLEAN），n_line_backfill_draft_v1.json f29da2de（draft_pending_cross_line，未施加），ledger #360；APPX-4 推进至「草案已封存待施加」。
- 3207 收官：G5-G6 5-appendix 复读 REPORT_ONLY_CLEAN（5 appendix 全 open），ledger #359。
- 3206 收官：FTR-22 升级链复读 REGISTERED（registry v1.28 `5995bbe9` + atlas_v1_30 `ebc1e122`，ledger #358；残留挂账 closed@3175）。
- 已收官链：FTR-04（3202）/GAP-4（3203）/FTR-15（3204）/FTR-14（3205）/FTR-22（3206）/appendix（3207）/N 线补录草案（3208）。
- 挂账不变：跨线账本补丁施加确认（APPX-5）；N 线补录施加（跨线确认后）；水果类；K4。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
