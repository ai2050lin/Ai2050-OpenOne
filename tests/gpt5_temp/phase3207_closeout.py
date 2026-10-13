# -*- coding: utf-8 -*-
"""Phase 3207 closeout: ledger #359 append + MEMO append + daily + MEMORY.
Report-only phase: no registry/gap-ledger/HTML mutation; only governance
ledger entry + docs.
"""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

TODAY = '2026-10-10'

# ---------- 1. ledger append (#359) ----------
LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
pre8 = sha8_file(LP)
disk0 = json.loads(raw.decode('utf-8'))
bako0 = json.loads(open(LP + '.bak_3207', 'rb').read().decode('utf-8')) \
    if os.path.exists(LP + '.bak_3207') else None
already = (bako0 is not None and len(disk0['measurements']) == 359
           and disk0['measurements'][358].get('phase') == 3207)
prev_post = bako0['ledger_sha256_8'] if bako0 is not None else disk0['ledger_sha256_8']
assert prev_post == 'fecd8e62', prev_post
if already:
    raw = open(LP + '.bak_3207', 'rb').read()
old = json.loads(raw.decode('utf-8'))
assert len(old['measurements']) == 358

entry = {
    "phase": 3207,
    "name": "g5g6_appendix_ledger_recheck",
    "line": "G",
    "date": TODAY,
    "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5g6_appendix_ledger_recheck|REPORT_ONLY_CLEAN|A1_gl15_appendix_5_entries_full_field_bitwise|A2_anchor_sha_x3_cross_3166_eff_ge05_sum2_onehot_14b_3165_K_entity_eff_ge05_0_glm4_14b_3155_kout_share_bitwise_x3|A3_gl16_appendix_append_only_bitwise_eq_gl15|A4_agate_C4_C6_pending_cross_line_wei_shi_jia_ledger_corrections_beb775bd_on_disk|5_appendix_confirmed_open_no_update_needed",
    "evidence_level": "report_only_appendix_recheck",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-G6",
    "superseded_by": None,
    "detail": (
        "G5-G6 5-appendix ledger re-read arm (report-only, zero GPU; gap "
        "ledger v1.5/v1.6 + sealed p3166/p3165/p3155-summary + deepseek "
        "a_gate_closure_v1 + ledger_corrections_v1; default prereg arm (a)). "
        "A1: gl15 appendix 5 entries full-field bitwise (id/title/anchor/"
        "status/note/anchor_sha8 all verbatim) + version 1.5 + file sha "
        "4fb3f37d. A2: anchor cross-sha APPX-1->p3166 448af595 (eff_ge05 "
        "sum==2, exactly one entry==2, 14b only; res 89b3f320 / seal "
        "263de0ab) + APPX-2->p3165 511d9b13 (cross_model_K glm4/qwen3-14b "
        "K_entity_eff_ge05==0, 词表方向 4b 专属 confirmed; res 9b0fe9c9 / "
        "seal e966a4e5) + APPX-3->p3155 summary 46a5034d (k2_int_share_kout "
        "bitwise 0.15187579769212284 / 0.1824533831239199 / 0.16411745881744746 "
        "in 15.0-18.5%% band; APPX-3 note '15.2-18.3%%' 为叙述性舍入, 14b "
        "0.18245->18.2%%). A3: gl16 (0f6423e0) version 1.6 + appendix "
        "bitwise == gl15 appendix (append-only preserved). A4: agate "
        "a_gate_closure_v1 24c60160 (== APPX-4/5 anchor_sha8, file located "
        "at research/deepseek/atlas/) + corrections_C1_C6 n==6 + C4/C6 "
        "status pending_cross_line + enactment 未施加 + ledger_corrections_v1"
        ".json beb775bd on disk — 跨线补丁已落盘未施加口径维持. Verdict "
        "appendix_ledger_recheck_report_only_clean: 5 appendix entries "
        "confirmed open, anchors all on-disk, no 口径 update needed after "
        "3206 FTR-22 closed@3175 note; no registry/gap-ledger/HTML "
        "mutation. Honest log: SMOKE first run DIRTY(1) — A2 kout band "
        "lower bound 0.1519 > measured 0.151876 (judgement bug, seal "
        "untouched); band relaxed to 15.0-18.5%%, second SMOKE CLEAN, "
        "FULL CLEAN. design badd69fe, res 97b25b57 / seal 20d84260."),
    "chain": {
        "res_sha8": "97b25b57",
        "seal_sha8": "20d84260",
        "gl15": "4fb3f37d",
        "gl16": "0f6423e0",
        "agate": "24c60160",
        "ledger_corrections": "beb775bd",
        "ledger_pre": prev_post,
    },
}
old['measurements'].append(entry)
assert len(old['measurements']) == 359
blob = json.dumps(old, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
old['ledger_sha256_8'] = new_post
blob = json.dumps(old, ensure_ascii=False, indent=1)
if not already:
    open(LP + '.bak_3207', 'wb').write(
        open(LP, 'rb').read() if len(json.load(open(LP, encoding='utf-8'))['measurements']) == 358
        else raw)
    open(LP, 'wb').write(blob.encode('utf-8'))
d2 = json.load(open(LP, encoding='utf-8'))
bako = json.loads(open(LP + '.bak_3207', 'rb').read().decode('utf-8'))
back = dict(d2)
back['measurements'] = d2['measurements'][:358]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
assert eq and d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 359
print('ledger #359 ok, pre', pre8, '(entry pre', prev_post, ') -> post', new_post,
      '(bak_3207 saved, rollback-equal)')

# ---------- 2. MEMO append ----------
MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8_file(MP)
if b'## Phase 3207 \xe2\x80\x94 G5-G6' not in mraw:
    ADD = """

## Phase 3207 — G5-G6 5-appendix 挂账复读臂（report-only，2026-10-10）

**预注册臂 (a) 执行完毕，SMOKE 二跑 CLEAN（首跑判据边界 bug DIRTY(1)，封存未动），FULL 一次收官**。零 GPU、report-only（无 registry/gap-ledger/HTML 变更）；SRC 8 文件（gl15 `4fb3f37d` + gl16 `0f6423e0` + p3166 `448af595` + p3165 `511d9b13` + p3155 summary `46a5034d` + agate `24c60160` + ledger_corrections_v1 `beb775bd` + registry_3162 `00f15e98`）sha 断言。design `badd69fe`，res `97b25b57` / seal `20d84260`。

### 判定（A1–A4 全 CLEAN，0 contradictions）
- **A1**：gl15 appendix 5 条全字段逐位回读（id/title/anchor/status/note/anchor_sha8 全部 verbatim）+ version 1.5。
- **A2**：锚交叉 sha——APPX-1→p3166 `448af595`（eff_ge05 求和==2、恰一处==2 仅 14b；res `89b3f320`/seal `263de0ab`）+ APPX-2→p3165 `511d9b13`（glm4/14b K_entity_eff_ge05==0，「词表方向 4b 专属」确证；res `9b0fe9c9`/seal `e966a4e5`）+ APPX-3→p3155 summary `46a5034d`（k2_int_share_kout 逐位 0.15187579769212284 / 0.1824533831239199 / 0.16411745881744746，落于 15.0–18.5% 带）。
- **A3**：gl16（3192，`0f6423e0`）version 1.6 + appendix 逐位 == gl15 appendix（append-only 保全）。
- **A4**：agate `a_gate_closure_v1.json` `24c60160`（== APPX-4/5 anchor_sha8；**文件定位至 research/deepseek/atlas/**）+ corrections_C1_C6 共 6 条 + C4/C6 status `pending_cross_line` + enactment「未施加」+ ledger_corrections_v1.json `beb775bd` 在盘——跨线补丁「已落盘未施加」口径维持。

### 结论（report-only，不动账本）
5 条 appendix 全部确认 open 且锚在盘；3206 FTR-22 closed@3175 口径更新不涉及 appendix；**无需口径登记**。诚实日志：SMOKE 首跑 DIRTY(1)——A2 kout 带下界 0.1519 > 实测 0.151876（判据 bug，封存未动），带宽松至 15.0–18.5% 后二次 CLEAN；另 APPX-3 note「15.2-18.3%」为叙述性舍入（14b 0.18245→18.2%），逐位锚以封存值为准。

### 挂账现状（3207 后）
5 appendix 维持 open（APPX-1/2/3 单线、APPX-4/5 跨线 pending_cross_line）；N 线 P3–P7 补 Ledger、跨线补丁施加确认仍为长期挂账。

### 预注册 Phase 3208（两支候选）
- (a) **UI v1.6 客户端零 script 版开工**（非账本轮）：图谱透镜客户端化（atlas v1.30 数据面 + registry v1.28 26 特征 + gap v1.6 + appendix 5 条），纯前端。
- (b) **G5-G7 N 线 P3–P7 补 Ledger 试臂**（零 GPU）：从 deepseek 线 P3–P7 封存产物回读，起草 ledger 补录条目（只补不改），施前先登记（与 APPX-4 挂账联动）。
- 默认执行 (b)；若 deepseek 线 P3–P7 封存产物不齐备则降级 (a) 并诚实登记。
"""
    open(MP, 'wb').write((mraw.decode('utf-8') + ADD).encode('utf-8'))
print('MEMO', m_pre, '->', sha8_file(MP))

# ---------- 3. daily ----------
DP = r'.workbuddy\memory\2026-10-10.md'
draw = open(DP, 'rb').read()
d_pre = sha8_file(DP)
dadd = ("\n\n### Phase 3207 G5-G6 5-appendix 挂账复读臂收官（report-only，2026-10-10）\n"
        "- SMOKE 首跑 DIRTY(1)（kout 带下界 0.1519 > 实测 0.151876，判据 bug 封存未动），修判据二次 CLEAN；FULL 一次收官：A1 appendix 5 条逐位 + A2 锚 sha ×3 交叉（3166 eff_ge05 sum==2 恰一处 14b / 3165 glm4+14b K_entity_eff_ge05==0 / 3155 kout 逐位）+ A3 gl16 appendix append-only == gl15 + A4 agate C4/C6 pending_cross_line 未施加。\n"
        "- APPX-4/5 锚 24c60160 定位至 research/deepseek/atlas/a_gate_closure_v1.json；ledger_corrections_v1.json beb775bd 在盘，跨线口径维持。\n"
        "- report-only：registry/gap-ledger/HTML 零变更；结论 5 appendix 全 open、无需口径登记（3206 FTR-22 closed@3175 不涉及 appendix）。\n"
        "- ledger #359（rollback 等价验证）；预注册 3208：默认 (b) N 线 P3–P7 补 Ledger 试臂；(a) UI v1.6 零 script 版。\n")
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
- **3208 已预注册 = G5-G7 N 线 P3–P7 补 Ledger 试臂**（零 GPU，默认）：deepseek 线 P3–P7 封存产物回读，起草 ledger 补录条目（只补不改，与 APPX-4 挂账联动）；候选 (a) UI v1.6 零 script 版开工（非账本轮）。若 P3–P7 封存不齐备降级 (a)。
- 3207 收官：G5-G6 5-appendix 复读 REPORT_ONLY_CLEAN（A1–A4 全 CLEAN），5 appendix 全 open 无需口径登记；APPX-4/5 锚 24c60160 = research/deepseek/atlas/a_gate_closure_v1.json（C4/C6 pending_cross_line + ledger_corrections beb775bd 在盘），ledger #359。
- 3206 收官：FTR-22 升级链复读 REGISTERED（L1–L3 全 CLEAN，k0 drift 0），registry v1.27 d5ce1c19→**v1.28 5995bbe9** + atlas_v1_30 ebc1e122（787/787），ledger #358；FTR-22 残留挂账口径 closed@3175。
- 3205 收官：FTR-14 归因后复读 REGISTERED（registry v1.27 d5ce1c19）；3204：FTR-15（v1.26）；3203：GAP-4（report-only）；3202：FTR-04。
- 后续：UI v1.6 客户端改造（零 script 版排队）。
- 挂账不变：N 线 P3–P7 补 Ledger（3208 默认臂）；跨线账本补丁施加确认；水果类；K4。

"""
t = t[:i] + new4 + t[j:]
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8_file(MP2))
print('closeout done')
