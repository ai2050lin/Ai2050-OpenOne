# -*- coding: utf-8 -*-
"""Phase 3232 closeout: five writes (ledger #384 / MEMO / daily / MEMORY /
self-check), idempotent, disk-readback assertions."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PREV_POST = '271cdbf2'
LED_N = 384
REPORT = []


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# ---------- 1. ledger ----------
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
raw = open(LP, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
already = [e for e in led['measurements'] if e.get('phase') == 3232]
if not already:
    assert len(led['measurements']) == 383 and \
        led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3232,
        "name": "g5g29_gd_amp_class_robustness",
        "line": "G",
        "date": "2026-10-13",
        "model": "CPU matrix math on sealed caches (zero GPU)",
        "verdict": ("GD_AMP_CLASS_ROBUST|G1_3of3_models_x4of4_oov_classes|"
                    "G2_3of3_class_ratio_7.14_9.61_9.18|amp_ratio_c_"
                    "6.57_12.21|spearman_amp_collapse_0.79_0.83|"
                    "registry_v1.31_UNCHANGED|gap_v1.9_to_v1.10"),
        "evidence_level": ("robustness arm (no new feature -> no E-level "
                           "change; GAP-4 mechanism_note append "
                           "class-level robustness)"),
        "model_scope": "3/3 (qwen3-4b + qwen3-14b + glm4-9b; glm4 via "
                       "chain-recomputed u*, 3225 ustar anchor 4b/14b)",
        "prereg_id": "G5-G29",
        "superseded_by": None,
        "detail": (
            "G5-G29 GD_amp class-count robustness arm (zero GPU, default "
            "(a) per 3231 prereg). 3192 u*_amplitude_carrier was pooled "
            "over n=4 OOV classes; this arm expands the unit of analysis "
            "to n=10 per-class measurements (4 OOV + 6 seen) on the SAME "
            "sealed residual chain (3169 B-protocol verbatim, seeds 7/8/9, "
            "ridge lam 1e-3) with chain-recomputed u* for all 3 models "
            "(3225 npz ustar external anchor for 4b/14b: cos=0.99999988 "
            ">= 1-1e-6; 3225 med_abs drift == 0). Device anchors: 3169 "
            "per-seed E drift 0, 3189/3190/3191 scalars drift == 0 "
            "bitwise x3. C1 correction registered pre-observation: 3192 "
            "pooled med anchors propagate through the SVD vector u* "
            "(BLAS run-to-run wobble ~5e-4 rad; first smoke crashed at "
            "the bitwise assert BEFORE any per-class observation) -> "
            "relaxed to rel tol 1e-6 (observed 1.6e-9); independent "
            "verify later showed 0.00e+00 rel on this machine. Gates "
            "(frozen pre-observation): G1 per-OOV-class amp_ratio_c >= "
            "1.3 in >= 3/4 classes, G2 class_ratio = med(4 OOV class "
            "meds)/med(6 seen class meds) >= 1.3, each x >= 2/3 models. "
            "Results: G1 PASS 3/3 models (4b amp_ratio_c "
            "10.18/9.96/12.21/10.08; 14b 6.65/7.06/8.50/6.57; glm4 "
            "9.19/8.30/10.30/7.97); G2 PASS 3/3 (class_ratio 9.6117/"
            "7.1420/9.1832) => GD_AMP_CLASS_ROBUST: the carrier verdict "
            "is NOT a pooling artifact - every OOV class individually "
            "carries 6.6-12.2x the seen-class u* amplitude. Report-only: "
            "per-class u* energy share, per-class collapse amplitude, "
            "Spearman(med_amp_c, collapse_c) = 0.794/0.830/0.830 over 10 "
            "classes (bigger-collapse classes concentrate more u* "
            "amplitude). Res 89fb1440 / seal 653933fe / npz 0bbe1f42 / "
            "gap v1.10 378de1cd. Independent verify 8/8 PASS (two initial "
            "FAILs were verify-probe hash-convention bugs, sealed "
            "artifacts untouched). Registry v1.31 UNTOUCHED (ef5189e7). "
            "pre-append ledger file sha8 %s." % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LP + '.bak_3232', 'wb').write(raw)
    with open(LP, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('ledger: appended #384 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    assert len(already) == 1, ('duplicate 3232 entries', len(already))
    new_post = led['ledger_sha256_8']
    REPORT.append('ledger: 3232 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LP, encoding='utf-8'))
assert len(chk['measurements']) == LED_N, (len(chk['measurements']), LED_N)
stored = chk['ledger_sha256_8']
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == stored, ('ledger post recompute', rec, new_post,
                                   stored)
REPORT.append('ledger self-check: n=%d post=%s recompute OK (chain prev=%s)'
              % (LED_N, new_post, PREV_POST))

# ---------- 2. MEMO ----------
MP = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
memo = open(MP, encoding='utf-8').read()
SEC = '''

## Phase 3232: G5-G29 GD_amp 类数稳健性扩样臂 [2026-10-13 11:57]

**执行模式**：零 GPU CPU 矩阵运算（默认 (a)，按 3231 预注册）；design **`a636d023`**（任何门控观测前冻结）；res **`89fb1440`** / seal **`653933fe`**、npz **`0bbe1f42`**、gap v1.10 **`378de1cd`**、verdict **`GD_AMP_CLASS_ROBUST`**；registry v1.31 未动（`ef5189e7`）；账本只读（n=383 post `271cdbf2` before==after），closeout 追加 #384。

### 问题与设计
3192 的 u*_amplitude_carrier（amp_ratio 10.60/7.11/8.89 ≥1.3 ×3）是**池化**口径（876 OOV 行 4 类合判）。本臂把分析单元扩到 **n=10 逐类**（4 OOV + 6 seen）：同一封存残差链（3169 B 口径 verbatim，seeds 7/8/9，ridge λ=1e-3）+ 链内重算 u* ×3 模型（3225 npz ustar 外部锚：cos=0.99999988 ≥1−1e-6、med_abs drift==0）。门（冻结在先）：**G1** 逐 OOV 类 amp_ratio_c ≥1.3 于 ≥3/4 类 ×≥2/3 模型；**G2** class_ratio = med(4 OOV 类幅)/med(6 seen 类幅) ≥1.3 ×≥2/3 模型；双过 ⇒ ROBUST。

### C1 修正（观测前登记，诚实记录）
首次 SMOKE 在 3192 池化锚逐位断言处崩——med_amp 经 SVD 奇异**向量** u* 传导，跨运行 BLAS 线程抖动 ~5e-4 rad（cos(u*_run, u*_3225npz)=0.99999988）；而全部**标量锚**（3189/3190/3191，不依赖 Vt[0] 方向）保持 drift==0 逐位。崩点在任何逐类观测计算之前 ⇒ 池化锚改相对容差 1e-6（实测 1.6e-9；独立复核在本机复算 rel=0.00e+00）。

### 观测（主臂，×3 模型）
| 模型 | amp_ratio_c（4 OOV 类） | class_ratio | G1/G2 | spearman(amp,崩塌) |
|---|---|---|---|---|
| qwen3-4b | 10.18/9.96/12.21/10.08 | 9.6117 | PASS/PASS | 0.794 |
| qwen3-14b | 6.65/7.06/8.50/6.57 | 7.1420 | PASS/PASS | 0.830 |
| glm4-9b | 9.19/8.30/10.30/7.97 | 9.1832 | PASS/PASS | 0.830 |

### verdict
- **重复一**：G1 3/3 × 4/4 类、G2 3/3 全过（最低单类 6.57 仍为门 1.3 的 5 倍）——**u*_amplitude_carrier 在类级成立，非池化伪影**：每个 OOV 类独立携带 6.6–12.2 倍于 seen 类的 u* 投影幅度。
- **重复二**：report-only 全类（10 类）Spearman(类 med 幅, 类崩塌能量) = 0.79/0.83/0.83——崩塌越强的类在 u* 上集中越多幅度，类级幅度承载与 3169 全类崩塌幅度**共变**。
- **重复三**：C 线机制链加厚——3189 SHARED → 3190 LOO → 3191 held-out → 3192 池化 CARRIER → **3232 类级 ROBUST（n=10）**；GAP-4 mechanism_note append（gap v1.9→v1.10，append-only，legacy 逐位保全），registry 不动（无特征级变化）。

### 独立复核（verify.json，8/8 PASS）
源锚 ×11、registry 未动、逐类数独立重算（rel=0）、门逻辑复算、gap v1.10 推导（legacy x4 + GAP-4 字段保全 + append 形状）、npz==result、smoke/FULL 数字一致（仅 mode/res/seal 差）。两次初 FAIL 均 verify 探针哈希口径 bug（None 填充 vs 键剔除；mid 态多含 seal_sha8），封存产物未动。

### 预注册 Phase 3233（候选）
- (a) **GAP-5 上游机制臂设计**（默认，按 3231 预注册 (b)）：FTR-22 残留挂账已清，上游杠杆候选扫描先立项——设计文档 + 候选杠杆清单 + 门定义（零 GPU 设计臂）；
- (b) u* 内容归因类级扩展：u* 与 10 类 first-token embedding 差分方向的逐类对齐（3190 report-only 降级方案的正式化，零 GPU）。
'''
if '## Phase 3232: G5-G29 GD_amp 类数稳健性扩样臂' not in memo:
    with open(MP, 'a', encoding='utf-8', newline='') as f:
        f.write(SEC)
    REPORT.append('memo: Phase 3232 section appended')
else:
    REPORT.append('memo: Phase 3232 section already present (idempotent)')
memo_sha8 = fsha(MP)

# ---------- 3. daily ----------
DP = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-13.md')
DSEC = '''
### Phase 3232 — G5-G29 GD_amp 类数稳健性扩样臂（2026-10-13 11:57）
design `a636d023`（门冻结在先 + C1 修正登记）；零 GPU；res `89fb1440` / seal `653933fe` / npz `0bbe1f42` / verdict `GD_AMP_CLASS_ROBUST`。n=4 池化 → n=10 逐类：G1 3/3 模型 ×4/4 OOV 类（amp_ratio_c 6.65–12.21，最低为门 5 倍）、G2 3/3（class_ratio 9.6117/7.1420/9.1832）——u*_amplitude_carrier 类级成立，非池化伪影；spearman(类幅,类崩塌) 0.79–0.83。C1：3192 池化锚经 SVD 向量传导有 BLAS 跨运行抖动（~5e-4 rad），标量锚仍逐位 0；池化锚改 rel 1e-6（实测 1.6e-9，复核机内 0）。gap v1.9→v1.10（`378de1cd`）GAP-4 append；registry v1.31 未动。独立复核 8/8 PASS（两次初 FAIL 均 verify 探针 bug）。账本 #384 post `%s`（pre-append file `%s`）。3233 预注册：(a) GAP-5 上游机制臂设计（默认）；(b) u* 内容归因类级扩展。
''' % (new_post, pre_file_sha8)
daily = open(DP, encoding='utf-8').read() if os.path.exists(DP) else ''
if '### Phase 3232' not in daily:
    with open(DP, 'a', encoding='utf-8', newline='') as f:
        f.write(DSEC)
    REPORT.append('daily: Phase 3232 note appended')
else:
    REPORT.append('daily: Phase 3232 note already present (idempotent)')
daily_sha8 = fsha(DP)

# ---------- 4. MEMORY ----------
MEMP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
mem = open(MEMP, encoding='utf-8').read()
lines = mem.splitlines()
i_prereg = [i for i, l in enumerate(lines)
            if l.startswith('- **3232 已预注册**')]
i_close = [i for i, l in enumerate(lines) if l.startswith('- 3231 收官：')]
assert len(i_prereg) == 1 and len(i_close) == 1 and i_close[0] == i_prereg[0] + 1, \
    (i_prereg, i_close)
NEW_LINES = [
    ("- **3233 已预注册**：(a) **GAP-5 上游机制臂设计**（默认，按 3231 预注册 "
     "(b)）：上游杠杆候选扫描先立项——设计文档 + 候选杠杆清单 + 门定义（零 "
     "GPU 设计臂）；(b) u* 内容归因类级扩展（3190 report-only 降级方案正式化）。"),
    ("- 3232 收官：GD_AMP_CLASS_ROBUST（design `a636d023`，res `89fb1440`，"
     "npz `0bbe1f42`），ledger #384 post `%s`；n=4 池化 → n=10 逐类：G1 3/3 "
     "×4/4 类（amp_ratio_c 6.65–12.21）、G2 3/3（class_ratio 9.61/7.14/9.18）"
     "——u*_amplitude_carrier 类级成立非池化伪影；spearman(类幅,类崩塌) "
     "0.79–0.83；gap v1.10 `378de1cd`，registry v1.31 未动；C1：SVD 向量跨运行"
     "抖动 ~5e-4 rad，标量锚逐位 0、池化锚改 rel 1e-6（观测前登记）。"
     % new_post),
]
lines[i_prereg[0]:i_close[0] + 1] = NEW_LINES
with open(MEMP, 'w', encoding='utf-8', newline='') as f:
    f.write('\n'.join(lines) + '\n')
REPORT.append('memory: 3232 prereg+3231 closeout lines -> 3233 prereg + '
              '3232 closeout (anchor check OK)')
mem_sha8 = fsha(MEMP)

# ---------- 5. self-check readback ----------
memo2 = open(MP, encoding='utf-8').read()
daily2 = open(DP, encoding='utf-8').read()
mem2 = open(MEMP, encoding='utf-8').read()
assert '## Phase 3232: G5-G29 GD_amp 类数稳健性扩样臂' in memo2 and \
    'GD_AMP_CLASS_ROBUST' in memo2
assert '### Phase 3232' in daily2 and new_post in daily2
assert '3232 收官' in mem2 and new_post in mem2 and '3233 已预注册' in mem2
REPORT.append('self-check: MEMO/daily/MEMORY readback OK')
REPORT.append('final: ledger_post=%s memo_sha8=%s daily_sha8=%s '
              'memory_sha8=%s' % (new_post, memo_sha8, daily_sha8, mem_sha8))
with open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                       'phase3232_closeout_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(REPORT) + '\n')
print('CLOSEOUT OK')
