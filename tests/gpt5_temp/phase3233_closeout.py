# -*- coding: utf-8 -*-
"""Phase 3233 closeout: five writes (ledger #385 / MEMO / daily / MEMORY /
self-check), idempotent, disk-readback assertions."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PREV_POST = '0e47bfd9'
LED_N = 385
REPORT = []


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# ---------- 1. ledger ----------
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
raw = open(LP, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
already = [e for e in led['measurements'] if e.get('phase') == 3233]
if not already:
    assert len(led['measurements']) == 384 and \
        led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3233,
        "name": "g5g30_gap5_upstream_lever_design",
        "line": "G",
        "date": "2026-10-13",
        "model": "design-only (zero GPU, no observation)",
        "verdict": ("UPSTREAM_LEVER_DESIGN_REGISTERED|candidates_6_"
                    "C1_u_R_C2_u_star_C3_dR_C4_dv_C5_kstar_proj_"
                    "C6_head_redist|gates_UL_flip_0.15_UL_dose_"
                    "UL_spec_1of3_UL_amp_0.6_UL_loc_report_only_"
                    "kstar_band_1_5|k1_anchor_kstar3_x3models_"
                    "NL36_40_40|registry_v1.31_UNTOUCHED_"
                    "gap_v1.10_UNTOUCHED|chain_3234_compute_"
                    "3235_decision"),
        "evidence_level": ("design arm (no observation -> no E-level "
                           "change; freezes GAP-5 stage-1 upstream-lever "
                           "scan protocol with full downstream "
                           "recomputation)"),
        "model_scope": "qwen pair (3234 compute arm); glm4 deferred "
                       "(W_U absent from local hub)",
        "prereg_id": "G5-G30",
        "superseded_by": None,
        "detail": (
            "G5-G30 GAP-5 upstream-lever scan DESIGN arm (zero GPU, "
            "default (a) per 3232 prereg). GAP-5 stage-1 (upstream "
            "encoding carries behavior) is the last open stage: F20 "
            "(3229/3230) proved both direction recipes (u_R, dR) are "
            "non-levers AT the final layer. This arm freezes the "
            "upstream scan protocol: residual patch h_L += alpha*u at L "
            "in {0..NL-1} with FULL downstream recomputation (stronger "
            "in-protocol causality than 3227 last-layer injection), "
            "judgment = isa/hasa token-set margin sign, per-layer "
            "norm-matched + orthogonalized nulls. Six pre-registered "
            "candidate levers (u_R / u* / dR / dW / kstar-projection / "
            "head-redistribution report-only reservation). Gates: "
            "UL_flip >= 0.15 over nulls x >= 3/4 folds x 2 models (same "
            "ratio family that genuinely failed in-protocol at "
            "3229/3230 GA_flip=0), UL_dose monotone, UL_spec cross-patch "
            "<= 1/3, UL_amp rank-corr >= 0.6 (n=4 small-n scope), "
            "UL_loc REPORT-ONLY position prediction (argmax band "
            "overlap [kstar-2, kstar+2] = [1, 5]; 3152 k*=3 three "
            "models, depth 0.075-0.083). Verdict taxonomy: "
            "UPSTREAM_LEVER_FOUND / UPSTREAM_PARTIAL / UPSTREAM_ABSENT "
            "(absent => distributed-carrier confirmation, F20-family F "
            "ledger extension). Budget ~1.5-2 h GPU serial (coarse 3.2e4 "
            "+ fine 2.6e4 forwards). Design doc d74b6484 / file "
            "6a53616c, freeze 7e201c78, res e594f31e. Independent "
            "verify 39/39 PASS (three initial FAILs were verify-probe "
            "hash-domain bugs: self keys included; sealed artifacts "
            "untouched). Registry v1.31 UNTOUCHED (ef5189e7), gap v1.10 "
            "UNTOUCHED (378de1cd). pre-append ledger file sha8 %s."
            % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LP + '.bak_3233', 'wb').write(raw)
    with open(LP, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('ledger: appended #385 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    assert len(already) == 1, ('duplicate 3233 entries', len(already))
    new_post = led['ledger_sha256_8']
    REPORT.append('ledger: 3233 entry already present, post=%s (idempotent)'
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

## Phase 3233: G5-G30 GAP-5 上游杠杆扫描设计臂 [2026-10-13 12:05]

**执行模式**：零 GPU 纯设计（默认 (a)，按 3232 预注册；无观测、无模型前向）；freeze **`7e201c78`**（任何观测前）；设计文档内容 **`d74b6484`** / 文件 **`6a53616c`**；res **`e594f31e`**，verdict **`upstream_lever_design_registered`**（0 矛盾）；registry v1.31 未动（`ef5189e7`）、gap v1.10 未动（`378de1cd`）；账本只读（n=384 post `0e47bfd9` before==after），closeout 追加 #385。

### 问题与动机
GAP-5 三段式（3229）：上游编码承载行为 → 末层精确线性读出（3228 C6v3 cos 1.0000）→ post-softmax 分布级非线性。F20（3229/3230）已证**两个方向配方（u_R、dR）在末层都不是行为杠杆**——「上游是否有单一方向杠杆」是 GAP-5 链最后一段 open 问题（gap v1.10 mechanism_note）。

### 设计要点（全部冻结在先）
- **干预**：残差流 patch `h_L ← h_L + α·u`，L ∈ {0..NL−1}，**下游全量重算**（所有层 > L + final norm + W_U）——比 3227 末层注入（绕过上游计算）更强的协议内因果；判断 = isa/hasa token-set margin 符号（3169 D2 first-token distinctness 锚定）。
- **候选杠杆 ×6（预注册）**：C1 u_R（T2 链配方）、C2 u*（3189–3232 链公共核）、C3 dR（3230 配方）、C4 Δv=W_U 词对差、C5 k* 带秩-5 子空间投影分解（in-plane vs 正交补）、C6 头级再分配（3160/3161；**report-only 设计预留**，不进 3234 计算臂）。
- **门（量纲体检在案）**：UL_flip 峰层 flip_rate ≥0.15（与 GD_flip 同族比值量纲，**可失败性已协议内实证**——3229/3230 GA_flip=0）；UL_dose α{0.5,1,2} 单调；UL_spec cross-patch ≤1/3；UL_amp u* 通道逐类 margin 偏移与 3169 崩塌能量 rank-corr ≥0.6（n=4 small-n scope）；**UL_loc 为 report-only 位置预测**：峰层带应与 [k*−2, k*+2]=[1,5] 重叠（3152 实测锚：k*=3 ×3 模型，深度 0.075–0.083，NL=36/40/40）——预测可被记录为不符而不改 verdict（防事后调门）。
- **verdict taxonomy**：UPSTREAM_LEVER_FOUND（flip+dose+spec 全过 ×2 模型）→ 3235 registry 升级；UPSTREAM_PARTIAL（方向性记录 + F 账本注记）；**UPSTREAM_ABSENT（全网格 FAIL ×2 模型 ⇒ 分布式承载确证，F20-family F 账本扩展，不隐藏）**。
- **对照**：逐层 norm-matched random（10 pinned seeds）+ 正交化（u 投影出行态）——正阳性不能归因于范数膨胀。
- **预算（预注册，防中途改设计）**：粗扫 NL×2 方向×208 行×2 模型 ≈3.2e4 前向 + 峰带细扫 3 层×3α×2 方向×730 行×2 模型 ≈2.6e4 + 峰层对照 ×11 ⇒ ~1.5–2 h GPU 串行。

### 链
- **3234** = G5-G30 compute+seal 臂（GPU：粗扫 + 峰带剂量细扫 + 对照，qwen pair，freeze-before-observation seed pinning）；
- **3235** = G5-G30 门评估 + registry/gap 决策臂（UL_* 门施加；FOUND → upgrade_log；ABSENT → F 账本 + GAP-5 mechanism_note append）。

### 独立复核（verify.json，39/39 PASS）
结果/设计/冻结三层身份哈希复算（自键剔除口径）、数据锚 ×6 重验、k* 锚独立重读（k*=3 ×3 模型 + NL 36/40/40）、UL_loc 带 [1,5] 文本、账本 n=384 post `0e47bfd9` 复算、registry/gap sha、SMOKE/FULL 冻结内容一致、无观测键泄漏。3 个初 FAIL 均 verify 探针哈希域 bug（把已赋值的自身 sha 键带进哈希域），封存产物未动。

### 预注册 Phase 3234（候选）
- (a) **G5-G30 compute+seal 臂**（默认，按本设计执行）：逐层 patch 全重算粗扫 + 峰带剂量细扫 + 双对照，qwen pair，GPU ~1.5–2 h，封存 npz + freeze-before-observation；
- (b) u* 内容归因类级扩展（3232 预注册 (b) 顺延，零 GPU）。
'''
if '## Phase 3233: G5-G30 GAP-5 上游杠杆扫描设计臂' not in memo:
    with open(MP, 'a', encoding='utf-8', newline='') as f:
        f.write(SEC)
    REPORT.append('memo: Phase 3233 section appended')
else:
    REPORT.append('memo: Phase 3233 section already present (idempotent)')
memo_sha8 = fsha(MP)

# ---------- 3. daily ----------
DP = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-13.md')
DSEC = '''
### Phase 3233 — G5-G30 GAP-5 上游杠杆扫描设计臂（2026-10-13 12:05）
零 GPU 纯设计；freeze `7e201c78`、设计文档 `d74b6484`/file `6a53616c`、res `e594f31e`、verdict `upstream_lever_design_registered`。冻结 GAP-5 stage-1 上游扫描协议：逐层 patch + **下游全量重算**（强于 3227 末层注入）；候选杠杆 ×6（u_R/u*/dR/Δv/k* 投影/头级预留）；门 UL_flip 0.15（可失败性由 3229/3230 GA_flip=0 实证）/dose/spec 1/3/amp 0.6/UL_loc report-only（k* 带 [1,5]）；verdict taxonomy 含 UPSTREAM_ABSENT ⇒ 分布式承载确证（F20 扩展）。k* 实测锚：3152 k*=3 ×3 模型。独立复核 39/39 PASS（3 初 FAIL 均 verify 探针自键哈希 bug，产物未动）。账本 #385 post `%s`（pre file `%s`）。registry v1.31 / gap v1.10 未动。3234 预注册：(a) compute+seal GPU 臂（默认）；(b) u* 内容归因类级扩展。
''' % (new_post, pre_file_sha8)
daily = open(DP, encoding='utf-8').read() if os.path.exists(DP) else ''
if '### Phase 3233' not in daily:
    with open(DP, 'a', encoding='utf-8', newline='') as f:
        f.write(DSEC)
    REPORT.append('daily: Phase 3233 note appended')
else:
    REPORT.append('daily: Phase 3233 note already present (idempotent)')
daily_sha8 = fsha(DP)

# ---------- 4. MEMORY ----------
MEMP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
mem = open(MEMP, encoding='utf-8').read()
if '3234 已预注册' in mem and '3233 收官' in mem and         '3233 已预注册' not in mem:
    REPORT.append('memory: already at 3234-prereg/3233-closeout state '
                  '(idempotent skip)')
else:
    lines = mem.splitlines()
    i_prereg = [i for i, l in enumerate(lines)
                if l.startswith('- **3233 已预注册**')]
    i_close = [i for i, l in enumerate(lines)
               if l.startswith('- 3232 收官：')]
    assert len(i_prereg) == 1 and len(i_close) == 1 and         i_close[0] == i_prereg[0] + 1, (i_prereg, i_close)
    NEW_LINES = [
        ('- **3234 已预注册**：(a) **G5-G30 compute+seal 臂**（默认，GPU '
         '~1.5-2 h）：按 3233 设计执行——逐层 patch 下游全量重算粗扫 + 峰带'
         '剂量细扫 + 双对照，qwen pair；UPSTREAM_ABSENT ⇒ 分布式承载确证；'
         '(b) u* 内容归因类级扩展（零 GPU，顺延）。'),
        ('- 3233 收官：upstream_lever_design_registered（freeze `7e201c78`，'
         'design `d74b6484`，res `e594f31e`），ledger #385 post `%s`；零 GPU '
         '设计臂：候选杠杆 ×6 + 门 UL_flip/dose/spec/amp + UL_loc report-only'
         '（k*=3 带 [1,5]，3152 实测锚）；registry v1.31 `ef5189e7` / gap '
         'v1.10 `378de1cd` 未动；独立复核 39/39 PASS。' % new_post),
    ]
    lines[i_prereg[0]:i_close[0] + 1] = NEW_LINES
    with open(MEMP, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(lines) + '\n')
    REPORT.append('memory: 3233 prereg+3232 closeout lines -> 3234 prereg + '
                  '3233 closeout (anchor check OK)')
mem_sha8 = fsha(MEMP)

# ---------- 5. self-check readback ----------
memo2 = open(MP, encoding='utf-8').read()
daily2 = open(DP, encoding='utf-8').read()
mem2 = open(MEMP, encoding='utf-8').read()
assert '## Phase 3233: G5-G30 GAP-5 上游杠杆扫描设计臂' in memo2 and \
    'upstream_lever_design_registered' in memo2
assert '### Phase 3233' in daily2 and new_post in daily2
assert '3233 收官' in mem2 and new_post in mem2 and '3234 已预注册' in mem2
assert len(mem2) < 3000, ('MEMORY too long', len(mem2))
REPORT.append('self-check: MEMO/daily/MEMORY readback OK (mem len=%d)'
              % len(mem2))
REPORT.append('final: ledger_post=%s memo_sha8=%s daily_sha8=%s '
              'memory_sha8=%s' % (new_post, memo_sha8, daily_sha8, mem_sha8))
with open(os.path.join(ROOT, 'tests', 'gpt5_temp',
                       'phase3233_closeout_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(REPORT) + '\n')
print('CLOSEOUT OK')
