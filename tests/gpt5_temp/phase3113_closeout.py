# -*- coding: utf-8 -*-
"""Phase 3113 closeout (idempotent):
Ledger -> MEMO Phase 3113 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3113'
        r'\omega_p111_artifact_writein')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
o = []

res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
assert res['verdict'] == \
    'belief_robust|within_unit_replicated|' \
    'write_in_concentrated'
assert res['A1']['slot8_median'] >= 0.85
assert res['A2']['slot8_median'] >= 0.70
assert res['B']['L28']['top8_share'] >= 0.50
assert res['sanity']['a1_determinism_b'] == 0.0
assert res['sanity']['residual_identity_rel_l2'] \
    < 0.02

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3113
           for m in led['measurements']):
    claim = (
        'Omega-P111 (3113, A offline on frozen 3105/'
        '3106 captures + B fresh GPU capture on '
        'rebuilt 3105 material, qwen3-4b, 81s) - '
        'artifact separation and write-in '
        'localization verdict.  Verdict belief_'
        'robust|within_unit_replicated|write_in_'
        'concentrated.  A1 DECISIVE ARTIFACT '
        'SEPARATION: 3105 P vs A1 pairs share pair/'
        'facts/query and differ in EXACTLY 1 token '
        '(line-k predicate); within-pair per-'
        'coordinate directional consistency median '
        '(direction-free) climbs 0.616 (slot0=L4) '
        '-> 0.899 (slot4=L20) -> 0.963 (slot6=L28) '
        '-> 0.952 (slot8=final norm) over 672 '
        'pairs - WITHIN-pair signal (0.952) even '
        'EXCEEDS cross-pair (0.920, 3111 baseline), '
        'so the truth broadcast CANNOT be a pair-'
        'identity / vocabulary data-construction '
        'artifact; it is carried by the truth-'
        'relevant 1-token difference itself.  '
        'Negative control A1-A2 (both truth=0, '
        'carries relation-presence cue): within-'
        'pair median 0.63, projection sign rate '
        '0.641 -> weak cue_leakage_present, far '
        'too small to explain 0.95.  Within-pair '
        'linear readout sign rate 1.000 on test '
        'pairs (centered: 0.31-0.45, w dominated '
        'by the DC component - consistent with '
        'all-moment broadcast).  A2 CROSS-MATERIAL '
        '(3106, 143 units, within-unit true-minus-'
        'false condition difference): median 0.629 '
        '(early) -> 0.951 (slot6=L28) -> 0.916 '
        '(final) -> within_unit_replicated; same '
        'slot6 peak shape as 3112.  B WRITE-IN '
        'LOCALIZATION (fresh hooks at layers '
        '12/20/24/28/32, last position, per-head '
        'at o_proj INPUT side with per-head o_proj '
        'column-block projection, zero-fit '
        'direction W_yes-W_no; rebuilt-material m '
        'AUC 0.9885 validates the rebuild): L12 '
        'writes nothing (ds -0.054); L20 MLP '
        'starts writing (ds_mlp +0.503, AUC '
        '0.9901); L24 peak build (ds_mlp +1.673, '
        'ds_block +2.399); L28 max write (ds_mlp '
        '+2.478) with CONCENTRATED head write '
        '(top-8 |ds| share 0.556, 25/32 heads '
        'within-pair frac>=0.70) -> write_in_'
        'concentrated; L32 MLP writes LARGE '
        'NEGATIVE (ds_mlp -3.970, mlp AUC 0.0061 '
        '= 0.9939 direction-free ANTI-correlated) '
        '- the late block actively CANCELS the '
        'truth direction it just built (erase/'
        'cleanup phase, descriptive pending '
        'ablation).  SLOT CLARIFICATION registered '
        '(corrects 3112 wording): 3105/3106 slot '
        'indices L0-L8 map to model layers '
        '4/8/12/16/20/24/28/32/36fn; 3112 "emerge '
        'L6" = model layer 28; write-in window '
        'L4-L6 = layers 20-28.  SYNTHESIS: the '
        'truth broadcast is a REAL within-pair '
        'belief signal (not artifact), WRITTEN '
        'mainly by MLPs at layers 20-28 with '
        'concentrated attention-head support at '
        '28, MAINTAINED to the end, and partly '
        'CANCELLED by the L32 MLP.  CAVEATS: (i) '
        'B uses rebuilt prompts (line order crc32 '
        'variant - original hash() unsalted '
        'unreproducible; structure and A6 '
        'multiset re-asserted, m AUC 0.9885); '
        '(ii) weak relation-presence cue leakage '
        'present; (iii) L32 erase interpretation '
        'descriptive, needs ablation; (iv) single '
        'model.  NEXT 3114: ablate top write-in '
        'heads + L28/L32 MLP contributions '
        '(causal test of write/erase), then T4 '
        'multi-step autoregression.')
    meas = {
        'meas_id': 'meas3113_omega_p111_'
                   'artifact_writein',
        'phase': 3113,
        'claim': claim,
        'verdict': 'belief_robust|within_unit_'
                   'replicated|write_in_'
                   'concentrated',
        'anchors': 'design_seal.json frozen before '
                   'computation: A gate slot8 '
                   'median 0.85/0.70; A2 gate 0.70; '
                   'B gate L28 top8 share 0.50/'
                   '0.25; direction W_yes-W_no '
                   'zero-fit; per-head cut at '
                   'o_proj INPUT side only',
        'artifacts': {
            'result': 'phase3113/omega_p111_'
                      'artifact_writein/'
                      'result.json',
            'seal': 'phase3113/omega_p111_'
                    'artifact_writein/'
                    'design_seal.json',
            'capture_b': 'phase3113/omega_p111_'
                         'artifact_writein/'
                         'capture_b.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, eager, '
                'batch 1, 64.2s capture of 2016 '
                'records); determinism 0; residual '
                'identity rel L2 7.6e-3; A6 20 '
                'pairs on rebuilt texts',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3113_omega_p111_artifact_writein')
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted')

# ---------- MEMO Phase 3113 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3113:' not in memo:
    sec = u'''## Phase 3113: Ω-P111 伪迹分离+写入端定位——配对内（仅差 1 token）单坐标中位 0.952 ≥ 跨对 0.920（伪迹解释被否定）；3106 unit 内 0.916 复现；写入端=MLP 主力（L20 起写 +0.50 → L28 峰值 +2.48、top-8 head 份额 0.556）且 L32 MLP 大幅负写入 −3.97（擦除相）；槽位索引澄清：3112 的 L6=模型层 28 [[NOW]]

**性质**：T3 第 7 Phase + 写入端首 Phase。A 部分纯离线（3105/3106 冻结 capture）；B 部分 GPU 新采集（qwen3-4b BF16，2016 记录，64.2s，hook 真实层 12/20/24/28/32）。总 81s。预注册（design_seal.json 先于一切统计）：A 门=slot8 配对内中位 ≥0.85 belief_robust / 0.70–0.85 partial_confound / <0.70 artifact_dominated；A2 门=slot8 ≥0.70 复现；B 门=L28 top-8 |ds| 份额 ≥0.50 concentrated / 0.25–0.50 moderate / <0.25 distributed。方向=零拟合 W_yes−W_no；per-head 切点只在 o_proj 输入侧（R55）。

### 0. 槽位索引澄清（纠错登记）
3105/3106 的槽位 L0–L8 对应模型层 **4/8/12/16/20/24/28/32/36fn**。3112 所写"emerge L6"实为**模型层 28**；"写入端窗口 L4–L6"= **模型层 20–28**。按纠错追加纪律在此更正，3112 原文不改。

### 1. A1 伪迹分离（决定性）：信念广播不是构造伪迹
3105 的 P vs A1 本就是最小对：同 pair、同事实块、同查询，**恰差 1 个 token**（line-k 谓词）。配对内逐坐标方向自由一致率中位（672 对）：

| slot(层) | 0(L4) | 1(L8) | 2(L12) | 3(L16) | 4(L20) | 5(L24) | 6(L28) | 7(L32) | 8(fn) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 配对内中位 | 0.616 | 0.662 | 0.777 | 0.795 | 0.899 | 0.946 | **0.963** | 0.964 | **0.952** |
| frac≥0.85 | .020 | .099 | .344 | .405 | .582 | .679 | .746 | .740 | .697 |

**配对内 0.952 甚至高于跨对 0.920（3111 基线）**——信号完全由真值相关的那 1 个 token 差异承载，pair 身份/词汇差异伪迹解释被否定。负对照 A1−A2（同为假、携带 relation-presence cue）：配对内中位仅 0.63，w 投影符号率 0.641（弱 cue 泄漏，远不足以解释 0.95）。配对内线性读出（train 对均值 w）：test 74 对符号率 **1.000**；中心化后 0.31–0.45（w 由 DC 分量主导，与"全阶广播"一致）。

### 2. A2 跨材料复现（3106，143 units）
unit 内真条件均值−假条件均值差分：slot5 0.888 → **slot6 0.951** → slot8 0.916 → **within_unit_replicated**。峰形与 3112 同（slot6=L28）。

### 3. B 写入端定位：MLP 主写、head 集中、L32 擦除
重建 3105 材料（行序 crc32 确定性变体，A6 multiset 20 对复验，重建材料 m AUC 0.9885 验证重建有效）；hook last 位置，per-head 切点在 o_proj 输入侧、经该 head 的 o_proj 列块投影回残差流；零拟合方向 w=W_yes−W_no；组件写入量 ds=配对内 (P−A1) 投影差均值：

| 层 | 12 | 20 | 24 | **28** | 32 |
| --- | --- | --- | --- | --- | --- |
| ds_MLP | −0.047 | **+0.503** | +1.673 | **+2.478** | **−3.970** |
| ds_block | −0.054 | +0.479 | +2.399 | +2.221 | −3.239 |
| MLP AUC | 0.349 | 0.990 | 0.983 | 0.991 | **0.006** |
| top-8 head 份额 | 0.557 | 0.735 | 0.819 | **0.556** | 0.802 |

**写入相**：L20 MLP 起写（AUC 0.990），L24–28 主力写入（ds 峰 +2.48）；L28 attention 写入**集中**（top-8 head 份额 0.556 ≥0.5 门，25/32 head 配对内 frac≥0.70）→ write_in_concentrated。**擦除相（新拼图）**：L32 MLP 大幅负写入（−3.97），其输出与真值方向**反向 AUC 0.994**——晚期块在主动抵消刚构建的真值方向分量。自检：determinism=0、残差恒等 rel L2=7.6e-03（h_out−h_in=o_proj(attn)+mlp）。

### 4. 综合（3107–3113）
真值广播是**真实的配对内信念信号**（非伪迹），由 **L20–28 的 MLP 为主力写入**、L28 attention head 集中辅助、**维持至末端**、**L32 MLP 部分擦除**。写入端从"窗口"细化到"组件+符号"：写入(+)/擦除(−)两相结构首次可见。

### 5. 硬伤
① B 用重建材料（行序 crc32 变体；原 hash() 加盐不可复现）——与原 capture 非同条文本，组件结论是材料族级；② relation-presence cue 弱泄漏（0.641）未完全排除；③ L32"擦除"是描述性观察，未经因果消融验证；④ centered 符号率 <0.5 的解释未定；⑤ 单模型（qwen3-4b）。

### 6. 3114 预注册（观测前冻结于 3114 seal）
① **因果消融**：置零/替换 L28 top-8 head 与 L28/L32 MLP 贡献（写入门控 ablation），检验写/擦两相的因果性（3101 教训：先逆向再复刻）；② **L32 擦除的目的性**：擦除是否服务下一步生成（擦除后状态 vs 下游读出）；③ 之后 T4 多步自回归 + 写入端条件化结构。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3113/omega_p111_artifact_writein/`（result.json、design_seal.json、run_log.txt、capture_b.npz 186.9MB）；脚本 `tests/glm5/phase3113_omega_p111_artifact_writein.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3113)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3113 Omega-P111 (offline A + GPU B, '
          'qwen3-4b, 81s): verdict belief_robust|'
          'within_unit_replicated|write_in_concentrated. '
          'A1 DECISIVE: within-pair (1-token minimal '
          'pairs, 672) per-coordinate median 0.952 at '
          'final >= cross-pair 0.920 -> truth broadcast '
          'is NOT a data-construction artifact; weak '
          'cue leakage 0.641 only. A2: 3106 within-unit '
          '0.916 replicated, slot6(=L28) peak 0.951. '
          'B: MLPs write the truth direction (L20 '
          '+0.50 -> L28 peak +2.48), L28 head write '
          'concentrated (top-8 share 0.556, 25/32 '
          'heads), and L32 MLP writes LARGE NEGATIVE '
          '(-3.97, anti-AUC 0.994) = erase phase. SLOT '
          'CLARIFICATION: 3112 L6 = model layer 28; '
          'window L4-L6 = layers 20-28. NEXT 3114: '
          'causal ablation of top heads + L28/L32 MLP '
          '(write/erase gates); then T4.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3113 Omega-P111' not in prev:
        try:
            with io.open(wl, 'a',
                         encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md update ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3113' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3112）',
        '## 机制链状态（3113）\n'
        '- 3113：伪迹分离+写入端判决 belief_robust|'
        'within_unit_replicated|write_in_concentrated。'
        '配对内（差 1 token）单坐标中位 0.952 ≥ 跨对 '
        '0.920——广播非伪迹；3106 unit 内 0.916 复现；'
        '写入端=MLP 主力 L20–28（峰 +2.48）+ L28 head '
        '集中（top8 0.556），L32 MLP 负写入 −3.97='
        '擦除相。槽位澄清：3112 L6=模型层 28。\n')
    mem_new = mem_new.replace(
        'max=3112', 'max=3113').replace(
        '下一 3113：**伪迹分离强对照（同词汇换真值）+ '
        '写入端组件级定位（L4-L6 谁写入广播）**→ '
        '之后 T4 多步自回归。',
        '下一 3114：**写/擦两相因果消融（L28 top8 head + '
        'L28/L32 MLP）**→ 之后 T4 多步自回归。')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
