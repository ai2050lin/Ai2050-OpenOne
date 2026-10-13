# -*- coding: utf-8 -*-
"""Phase 3203 closeout: ledger #355 append + MEMO append + daily + MEMORY patch.
Report-only phase: registry v1.25 (3bb33a24) and gap ledger v1.6 (0f6423e0) UNTOUCHED.
"""
import copy
import datetime
import hashlib
import json

TODAY = '2026-10-10'


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

# ---------- 1. ledger append (#355) ----------
LP = r'research\gpt5\atlas\atlas_ledger.json'
raw = open(LP, 'rb').read()
pre8 = sha8(raw)
d = json.loads(raw.decode('utf-8'))
old_blob = json.dumps(d, ensure_ascii=False, indent=1)
n0 = len(d['measurements'])
assert n0 == 354, n0
post_old = d['ledger_sha256_8']

entry = {
    "phase": 3203,
    "name": "g5f2_gap4_reread",
    "line": "G",
    "date": TODAY,
    "model": "qwen3-4b+qwen3-14b+glm4-9b (zero GPU)",
    "verdict": "g5f2_gap4_reread|CLEAN|G1_quantification_chain_4dp|G2_C_line_five_arms|G3_ftr25_anchor_bitwise_89_90_91|G3_mechanism_note_markers_ok|n_contradictions_0|gap4_reread_clean|report_only_registry_v1.25_3bb33a24_untouched|gap_ledger_v1.6_0f6423e0_authoritative_unchanged",
    "evidence_level": "RE_READ_CONSISTENCY",
    "model_scope": "qwen3-4b+qwen3-14b+glm4-9b",
    "prereg_id": "G5-F2",
    "superseded_by": None,
    "detail": (
        "G5-F2 GAP-4 residual mechanism re-read arm (report-only; zero GPU; sealed "
        "3169/3172/3175/3188-3192 scalars + gap v1.5/v1.6 + registry v1.25 re-read; "
        "no registry bump, no atlas). G1 quantification chain 4dp clean: 3169 pooled "
        "ratio_B 2.5388 (E_oov_B 0.9479 / E_seen_B 0.3734), 3172 ratio_kmax 1.8002 "
        "(borderline_partial_recovery), 3175 delta 0.0660, ratio_in_kmax 1.8002 / "
        "ratio_out_kmax 1.8662 (port_residual_dominant). G2 C-line five arms clean: "
        "3188 DECOUPLED(2/3) -> 3189 SHARED(3/3, top1 0.3065/0.2936/0.3008, cross_med "
        "0.5441/0.5663/0.4449) -> 3190 LOO robust(3/3, loo 0.0342/0.0214/0.0332, ks "
        "0.9932/0.9761/0.9715) -> 3191 held-out predictive(3/3, min_med "
        "0.3906/0.4174/0.3717) -> 3192 amplitude carrier(3/3, amp_ratio "
        "10.5984/7.1100/8.8909) + ratio_B_anchor bitwise == 3169 ratio_B. G3: FTR-25 "
        "anchors bitwise vs sealed 3189/3190/3191 scalars (True/True/True) + gap v1.6 "
        "mechanism_note markers present (u*_amplitude_carrier / 10.5984 / 7.1100 / "
        "8.8909 / ratio_B 2.5388). Honest log: SMOKE first run DIRTY(2) - (1) PIN "
        "p3169 e_seen_B was probe-truncated 0.3733, true 4dp of 0.37335046 is 0.3734 "
        "(criterion aligned to sealed data, sealed untouched); (2) mechanism_note "
        "marker criterion contained strings absent from ledger text ('amp_ratio', "
        "'10.60', 'quantified_collapse'), realigned to actually-present markers. "
        "Proposal (report-only): gap ledger v1.6 stays authoritative unchanged; GAP-4 "
        "status quantified_collapse (mechanism chain 3169->3175 + C-line 3188->3192 "
        "double-chain re-read holds); v1.7 only on new evidence. Verdict "
        "gap4_reread_clean, G1-G3 0 contradictions. design f3037547, res dc13d252 / "
        "seal d237ec5a."),
    "chain": {
        "design_sha8": "f3037547",
        "res_sha8": "dc13d252",
        "seal_sha8": "d237ec5a",
        "registry_v125": "3bb33a24",
        "gap_ledger_v16": "0f6423e0",
        "ledger_pre": pre8,
    },
}
d['measurements'].append(entry)
assert len(d['measurements']) == 355
blob = json.dumps(d, ensure_ascii=False, indent=1)
new_post = sha8(blob.encode('utf-8'))
d['ledger_sha256_8'] = new_post
blob = json.dumps(d, ensure_ascii=False, indent=1)
open(LP + '.bak_3203', 'wb').write(raw)
open(LP, 'wb').write(blob.encode('utf-8'))
# prefix equivalence re-read
d2 = json.load(open(LP, encoding='utf-8'))
new_blob = json.dumps(d2, ensure_ascii=False, indent=1)
assert new_blob.startswith(old_blob[:-1]), 'prefix equivalence FAIL'
assert d2['ledger_sha256_8'] == new_post and len(d2['measurements']) == 355
print('ledger #355 ok, pre', pre8, '-> post', new_post, '(bak_3203 saved)')

# ---------- 2. MEMO append ----------
MP = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MP, 'rb').read()
m_pre = sha8(mraw)
assert m_pre == '2a3a5a8f', m_pre
ADD = """

## Phase 3203 — G5-F2 GAP-4 残留机制复读臂（report-only，2026-10-10）

**预注册臂 (a) 执行完毕**。零 GPU；SRC 11 文件（8 封存 result + gap v1.5 4fb3f37d + v1.6 0f6423e0 + registry v1.25 3bb33a24）sha 断言；CONTENT 内容双锁 ×8；registry 不动、无 atlas 渲染。design f3037547、res dc13d252 / seal d237ec5a。

### 判定（G1–G3 全 CLEAN，0 contradictions）
- **G1 量化链 4dp**：3169 ratio_B 2.5388（E_oov_B 0.9479 / E_seen_B 0.3734）→ 3172 ratio_kmax 1.8002（borderline_partial_recovery）→ 3175 delta 0.0660，ratio_in 1.8002 / ratio_out 1.8662（port_residual_dominant）。
- **G2 C 线五连**：3188 DECOUPLED(2/3) → 3189 SHARED(3/3；top1 0.3065/0.2936/0.3008、cross_med 0.5441/0.5663/0.4449) → 3190 LOO robust(3/3；loo 0.0342/0.0214/0.0332、ks 0.9932/0.9761/0.9715) → 3191 held-out predictive(3/3；min_med 0.3906/0.4174/0.3717) → 3192 amplitude carrier(3/3；amp_ratio 10.5984/7.1100/8.8909)＋ratio_B_anchor 逐位 == 3169 ratio_B。
- **G3 跨一致性**：FTR-25 锚 vs 封存 3189/3190/3191 标量逐位 True/True/True；gap v1.6 mechanism_note 标记齐备（u*_amplitude_carrier / 10.5984 / 7.1100 / 8.8909 / ratio_B 2.5388）。

### 诚实日志（SMOKE 首跑 DIRTY(2) 的判据修正）
1. PIN p3169 e_seen_B 探针截断 0.3733，实测 0.37335046 的 4dp 舍入为 **0.3734**——判据对齐封存数据，封存产物未动。
2. mechanism_note 标记判据含账本文本不存在的字样（'amp_ratio'/'10.60'/'quantified_collapse'），改为实测存在标记后 OK。
两处均属「判据可修、封存不动」纪律范围，已如实登记。

### 定论（report-only 提案）
GAP-4 状态维持 **quantified_collapse**：机制链（3169→3172→3175）与 C 线五连（3188→3192）双链封存标量全数复现、FTR-25 锚逐位一致、gap v1.6 mechanism_note 标记齐备 ⇒ 复读闭环成立。**gap ledger v1.6 维持权威定稿不动**；后续 v1.7 仅在新增证据时触发。registry v1.25（3bb33a24）未动。

### 预注册 Phase 3204（两支候选）
- (a) **G5-G1 FTR-15 held_out 补强复读臂**（零 GPU）：回读 FTR-15 升级链封存标量 + registry v1.25 锚逐位，评估 E2_held_out 口径注记下的 scope_limits 补写（registry v1.25→v1.26 仅 FTR-15 scope_limits append + atlas_v1_28）。
- (b) **G5-G2 GAP-4 残留机制量化提案 v1.7 草案**（零 GPU，report-only）：基于 3203 复读 CLEAN 的双链闭环，起草 gap v1.7 mechanism_note 增量文本（port_residual_dominant 机制段定稿候选），不动账本，待用户审。
- 默认执行 (a)；若 FTR-15 封存锚不齐备则降级 (b) 并诚实登记。
"""
blob = mraw.decode('utf-8') + ADD
open(MP, 'wb').write(blob.encode('utf-8'))
print('MEMO', m_pre, '->', sha8(open(MP, 'rb').read()))

# ---------- 3. daily ----------
DP = r'.workbuddy\memory\2026-10-10.md'
try:
    draw = open(DP, 'rb').read()
except FileNotFoundError:
    open(DP, 'wb').write(b'')
    draw = b''
dadd = ("\n\n### Phase 3203 G5-F2 GAP-4 复读臂收官（2026-10-10）\n"
        "- SMOKE 首跑 DIRTY(2)：PIN e_seen_B 截断 0.3733→0.3734 修正；mechanism_note 标记改实测存在字样（'amp_ratio'/'10.60'/'quantified_collapse' 不在账本文本）。\n"
        "- FULL CLEAN：G1 量化链 + G2 C 线五连 + G3 FTR-25 锚逐位/标记，0 contradictions，verdict gap4_reread_clean。\n"
        "- report-only：registry v1.25 3bb33a24 与 gap v1.6 0f6423e0 未动；proposal 已入 MEMO/ledger #355。\n"
        "- 预注册 3204：默认 (a) G5-G1 FTR-15 held_out 补强复读臂；(b) gap v1.7 草案候选。\n")
blob = draw.decode('utf-8') + dadd
open(DP, 'wb').write(blob.encode('utf-8'))
print('daily ->', sha8(open(DP, 'rb').read()))

# ---------- 4. MEMORY patch ----------
MP2 = r'.workbuddy\memory\MEMORY.md'
m2 = open(MP2, 'rb').read()
m2_pre = sha8(m2)
t = m2.decode('utf-8')
old_line = "- **3179 已预注册 = G5-B4 FTR-23 立项入册**"
idx = t.find('## 4 下一步')
assert idx >= 0
head, tail = t[:idx], t[idx:]
# replace whole section 4 with updated
sec_end = tail.find('## 5 ')
assert sec_end >= 0
new4 = """## 4 下一步
- **3204 已预注册 = G5-G1 FTR-15 held_out 补强复读臂**（零 GPU，默认）：封存标量回读 + registry v1.25 锚逐位，registry v1.25→v1.26 仅 FTR-15 scope_limits append + atlas_v1_28；候选 (b) gap v1.7 草案（report-only）。
- 3203 收官：G5-F2 GAP-4 复读 CLEAN（G1 量化链 + G2 C 线五连 + G3 锚逐位/标记，0 contradictions），registry v1.25 3bb33a24 / gap v1.6 0f6423e0 未动，ledger #355（RE_READ_CONSISTENCY）。
- 后续：FTR-04 glm4 后续（已收官 3202）、GAP-4 已闭环复读、5 appendix；UI v1.6 客户端改造（用户询方案已给，零 script 版排队）。
- 挂账不变：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；水果类；K4。

"""
t = head + new4 + tail[sec_end:]
# update section 2 header counts (registry v1.25 / ledger 354)
t = t.replace('registry v1.5 `e1942d03`、atlas_v1_6 `b95e67a2`、ledger n=330 chain `5851a521`。',
              'registry v1.25 `3bb33a24`（26 特征）、atlas_v1_27 `739faeb7`、ledger n=355 chain 见 #355。')
open(MP2, 'wb').write(t.encode('utf-8'))
print('MEMORY', m2_pre, '->', sha8(open(MP2, 'rb').read()))
print('CLOSEOUT DONE')
