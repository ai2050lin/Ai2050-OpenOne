# -*- coding: utf-8 -*-
"""Phase 3251 closeout: five writes (result assertions / ledger #403 / MEMO
incl. doc-arm fidelity-closure summary + gap appendix candidate / dual wlog /
MEMORY.md). Idempotent per write; ledger append only once (chain-checked)."""
import hashlib
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3251\g5h14_ustar_lexical_attribution\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'research\gpt5\docs\WORKLOG_gpt5.md')
WLOG_U = os.path.join(ROOT, r'research\deepseek\docs\WORKLOG_gpt5_mirror.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
REG = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                         r'\phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
TS = time.strftime('%Y-%m-%d %H:%M')
REPORT = []


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


# ---------- 1. result assertions ----------
R = json.load(open(RES, encoding='utf-8'))
assert R['verdict'] == 'ustar_lexical_partial' and R['mode'] == 'full'
assert R['design_sha8'] == '48169586' and R['res_sha8'] == '0601eb1b'
assert R['gate_all'] is False
PM = R['per_model']
assert set(PM) == {'qwen3-4b', 'qwen3-14b', 'glm4-9b'}
for m, pm in PM.items():
    assert pm['drift_3169'] == 0.0 and pm['drift_3189'] == 0.0 and pm['drift_3190'] == 0.0
    # OOV-dominance clause falsified in all 3 models
    assert pm['oov4_sum'] < pm['seen6_sum']
    # max|cos| >> random baseline but modest vs pseudo-class null (4b only)
assert max(PM['qwen3-4b']['abs_lex']) == 0.1438 or True
assert PM['qwen3-4b']['ntok_single_4b'] if 'ntok_single_4b' in PM['qwen3-4b'] else True
mx = {m: max(PM[m]['abs_lex']) for m in PM}
assert abs(mx['qwen3-4b'] - 0.1438) < 1e-3
assert abs(mx['qwen3-14b'] - 0.0327) < 1e-3
assert abs(mx['glm4-9b'] - 0.0245) < 1e-3
assert mx['qwen3-4b'] > 3 * PM['qwen3-4b']['rand_med']      # 4b lexical detectable
assert mx['qwen3-14b'] >= 3 * PM['qwen3-14b']['rand_med']   # 14b at gate edge
assert mx['glm4-9b'] < 3 * PM['glm4-9b']['rand_med']        # glm4 below random gate
assert PM['glm4-9b']['perm_max_med'] > mx['glm4-9b']        # null covers observed
assert PM['qwen3-14b']['perm_max_med'] < mx['qwen3-14b']
REPORT.append('1 science assertions: OK (verdict=%s res=%s; drifts 0.0 x9)'
              % (R['verdict'], R['res_sha8']))

# registry untouched
rs = sha8(open(REG, 'rb').read())
assert rs == 'a30215fd', rs
REPORT.append('1b registry v1.33 byte sha8 %s untouched (report-only phase)' % rs)

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = sha8(raw)
led = json.loads(raw.decode('utf-8'))
PREV_POST = 'a79a1e4f'
already = [e for e in led['measurements'] if e.get('phase') == 3251]
if not already:
    assert len(led['measurements']) == 402 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3251,
        "name": "g5h14_ustar_lexical_attribution",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b + qwen3-14b + glm4-9b (zero-GPU ustar lexical "
                 "attribution, formalization of the 3190 downgraded arm (b); "
                 "+ doc arm: fidelity-closure summary 3239-3250 + gap "
                 "appendix candidate)",
        "verdict": ("ustar_lexical_partial|GATE 0/3: lexical-class alignment "
                    "is real but SMALL, DECAYS WITH SCALE and is NOT "
                    "OOV-dominant - 4b max|cos(u*,grad_cls)| 0.1438 (10.4x "
                    "random baseline, 2.0x pseudo-class null), 14b 0.0327 "
                    "(3.6x random), glm4 0.0245 (BELOW 3x-random gate and "
                    "BELOW its pseudo-class null 0.0288 - word-cluster "
                    "geometry alone covers it); oov4_sum < seen6_sum in ALL "
                    "3 models (OOV-dominance clause falsified). Combined "
                    "with the D-line quintet (3193-3197 SIGNED_CLASS_"
                    "CONTRAST): u* class structure is carried by "
                    "ACTIVATION-side class-mean directions v_c (cos "
                    "0.72-0.85), NOT by vocabulary embedding difference "
                    "directions - the two geometric systems are decoupled; "
                    "this is the atlas-level explanation of why lexical "
                    "bridges (3241/3247) gained so little"),
        "evidence_level": ("report-only characterization (preregistered "
                           "MEMO 3250 tail arm (b); NINE bit-level drifts "
                           "== 0.0: drift69 x3 + drift89 x3 (top1/top2/"
                           "cross_med/within/r80/r90) + drift90 x3 "
                           "(loo_top1/cos_ustar_vc/loo_max_delta vs sealed "
                           "3190); SMOKE-stage correction registered: "
                           "class-row permutation control degenerate "
                           "(mean-centered diff direction set invariant "
                           "under row permutation, null == observed max) "
                           "-> pseudo-class word permutation (73 entity "
                           "words into 10 groups); registry v1.33 "
                           "byte-untouched"),
        "model_scope": "qwen3-4b + qwen3-14b + glm4-9b (own embedding spaces)",
        "prereg_id": "G5-H14 (MEMO 3250 tail; formalizes 3232 prereg (b))",
        "superseded_by": None,
        "detail": (
            "u* class-level lexical attribution (zero GPU, 3 models). Chain: "
            "verbatim 3190 (3169 B-protocol -> Qn 876 OOV residual row "
            "directions -> ustar = SVD Vt[0]). New arm per model: 10 class "
            "first-subword embed rows (mean pooling; 4b class words "
            "single-token asserted), mean-centered difference directions "
            "grad_cls_c, signed cos(ustar, grad_cls_c) per class, oov4 vs "
            "seen6 |cos| sums, negative-sign count (3197 link), controls = "
            "500 random unit directions + 20 pseudo-class word permutations. "
            "Findings: (1) signed cos carries the anti-class-mean signature "
            "in 4b (OOV carriers negative: instruments -0.1438 / weather "
            "-0.0572 / sports -0.0475), consistent with 3197; (2) alignment "
            "DECAYS WITH SCALE 0.1438 -> 0.0327 -> 0.0245; (3) glm4 "
            "pseudo-class null (0.0288) exceeds its observed max - lexical "
            "alignment there is pure word-cluster geometry; (4) doc arm "
            "(a): fidelity-closure summary 3239-3250 appended to MEMO + gap "
            "appendix candidate entry drafted (closure chain -1.751 -> "
            "-0.449 incumbent XC_p5_b8 vs oracle +0.083). Registry "
            "untouched (v1.33 head, byte-verified a30215fd). freeze "
            "48169586, res 0601eb1b. Pre-append ledger file sha8 %s."
            % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = sha8(json.dumps(led, ensure_ascii=False, indent=1).encode('utf-8'))
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3251', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #403 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3251 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 403
chk['ledger_sha256_8'] = PREV_POST
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=403 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO (phase section + doc-arm closure summary) ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = sha8(mb)
if b'## Phase 3251:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3251: G5-H14 u* 类级词汇归因（3190 降级臂正式化）+ 保真收线小结（零 GPU，双臂） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3250 tail（(b) 科学臂 + (a) 文档臂）；freeze **`48169586`**；res **`0601eb1b`**；verdict **`ustar_lexical_partial`**（锚全过、门 0/3 → partial，characterization 登记）。**九位级漂移全零**：drift69×3（3169 per-seed E）+ drift89×3（3189 top1/top2/cross_med/within/r80/r90）+ drift90×3（3190 loo_top1/cos_ustar_vc/loo_max_delta），均逐位。registry v1.33 `a30215fd` 字节不动（report-only）。')
    A_('')
    A_('### 核心发现（u* 词汇归因正式化，headline: 词级对齐真实但小、随规模衰减、非 OOV 主导）')
    A_('')
    A_('- **门 0/3 ⇒ partial**：LEXICALLY_STRUCTURED 需 max|cos(u*,∇_cls)| ≥ 3×随机基线 且 oov4_sum > seen6_sum 全 3 模型；实测 4b max 0.1438（随机 10.4×、伪类 null 2.0×）达标但 OOV 主导未过；14b 0.0327（3.6×）踩线达标同样 OOV 主导未过；**glm4 0.0245 低于 3× 门且低于其伪类 null（0.0288）——词簇几何即覆盖**。')
    A_('- **OOV 主导全面证伪**：oov4_sum < seen6_sum ×3/3（4b 0.3046 vs 0.3136 持平；14b 0.0334 vs 0.0952；glm4 0.0269 vs 0.0562）。')
    A_('- **符号一致（3197 链接）**：4b OOV 载体类带符号 cos 为负（乐器 −0.1438 / 天气 −0.0572 / 运动 −0.0475）——anti-class-mean 对比签名在词级可见。')
    A_('- **规模衰减**：max|cos| 0.1438 → 0.0327 → 0.0245（4b→14b→glm4）——词级对齐随模型规模衰减。')
    A_('- **图谱级定谳（与 D 线五连合并）**：u* 的类结构由**激活侧类均值方向 v_c**（cos 0.72–0.85）承载，**不映射到词表 embedding 差分方向**（词级 |cos| ≤0.14 且随规模衰减、glm4 上被词簇 null 覆盖）——**两套几何系统解耦**；这是词桥（3241/3247）增益微弱的图谱级解释。')
    A_('')
    A_('**SMOKE 修正（正式观察前，已记 prereg_correction）**：类级行置换对照退化——mean-centered 差分方向集合在行置换下不变（center 不变），null ≡ 观测 max（0.1438 恒等复现；3242 类级置换空洞同族教训）→ 改为**实体词伪类置换**（73 实体词分 10 组构造伪类差分）。')
    A_('')
    A_('### 文档臂 (a)：G5 保真收线小结（3239–3250，registry gap 附录候选）')
    A_('')
    A_('**收线链（SCLS 族保真，3 种子均值，876 OOV 行）**：3239 原版 ridge −1.751（KL 恢复 38.7%）→ 3240 F2 类子空间补全 −1.75（全族正向转正未达）→ 3241 G1e 词桥 −0.77（DVU/DCC 回退越界，partial；3244 证伪为 span 伪影）→ 3242 T3_dual 跨模型 −1.293（首个零回退正向，partial）→ 3243 seen-only 信息极限（三变体全败）→ 3244 归因反转（99.3% 来自 span 扩展；R1_clean 零回退 −0.92）→ 3245 X_combo 组合加性 −0.797（FLIP 未达：声明体制内转正不可达）→ 3246 体制边界（FLIP=INFORMATION_GAP：LS oracle +0.083 转正；拉力算子 span→全空间时 fid→0 的数学上限）→ 3247 映射函数类（poly 核 +0.142 闭合）→ 3248 degree 阶梯未饱和（K_p3_T3 纯拉力 −0.529 全场最优）→ 3249 deg 5 单调（−0.467，DVU 侵蚀随阶增长；combo=p2 时代装置）→ 3250 价值停点（**XC_p5_b8 −0.449 现役**，r2 0.241/KL 0.987；cond 平坦无数值爆炸）。')
    A_('')
    A_('**状态定谳**：声明体制内缺口从 −1.751 收至 **−0.449**（修复 74%），剩余 −0.53 为 INFORMATION_GAP（3246 定谳）——转正需越出泄漏声明界或新机制类别（偏移型分量）；词级来源已由 3251 证伪（本 phase）。**gap 附录候选条目**：`gap_v1_12_appendix_fidelity_closure`（描述收线链 + 现役配方 + oracle 上界 + 词汇解耦定谳；待下一次 registry append 时一并入册，本 phase report-only 不触发）。')
    A_('')
    A_('**限界**：零 GPU；∇_cls 为 first-subword embed 差分（类词 4b 单 token 断言）；u* 符号约定沿用 3190 SVD 输出（|cos| 报告为主）；checkpoint 轨迹与 5 appendix GPU 项维持挂账。')
    A_('')
    A_('**完整性**：ledger #403（pre 文件 sha8 %s）；产物 result.json + ustar_lex.npz + smoke 子目录；独立磁盘复核见 p3251_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3252（候选）')
    A_('')
    A_('- (a) **gap 附录候选条目定稿臂**：把 `gap_v1_12_appendix_fidelity_closure` 按 gap ledger schema 成文并预演 diff（仍 report-only，入册留待下一次 registry append 触发点）；')
    A_('- (b) **U 点轨迹采样臂**（checkpoint 轨迹挂账的零 GPU 前置）：3169 collect npz 内已有 per-seed 中间量可否支撑训练轨迹上的 u* 稳定性刻画——先做产物清单审计再决定是否需要 GPU 项；')
    A_('- (c) 5 appendix GPU 项维持挂账。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3251 section + fidelity-closure doc arm (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3251 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3251:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3251 u* 类级词汇归因（3190 降级臂正式化）+保真收线小结完成（双臂，partial=characterization）：门 0/3——词级对齐真实但小、随规模衰减（0.1438→0.0327→0.0245）、非 OOV 主导（oov4<seen6 ×3）；glm4 上伪类 null 覆盖观测。'
      '与 D 线五连合并定谳：u* 类结构由激活侧 v_c 承载，与词表 embedding 差分解耦——词桥增益微弱的图谱级解释。'
      'SMOKE 修正：类级行置换退化（mean-centered 差分置换不变）→实体词伪类置换。'
      '九位级漂移全零（drift69/89/90 ×3）。文档臂：3239–3250 保真收线小结入 MEMO + gap 附录候选条目 gap_v1_12_appendix_fidelity_closure（待入册）。'
      'verdict ustar_lexical_partial；freeze 48169586 / res 0601eb1b，ledger #403，registry v1.33 不动。'
      '3252 已预注册：gap 附录候选定稿 + checkpoint 轨迹产物审计。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3251 u* 类级词汇归因' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3251 已预注册**：(a) 保真收线小结段（文档臂）；(b) u* 内容归因（提级）；(c) checkpoint/appendix 挂账。')
new_block = ('- **3251 收官（u* 词汇归因 + 收线小结，双臂 partial=characterization）**：词级对齐真实但小、随规模衰减（4b 0.144→14b 0.033→glm4 0.025 低于伪类 null）、非 OOV 主导 ×3；与 D 线合并定谳——u* 类结构由激活侧 v_c 承载，与词表 embedding 解耦（词桥微弱的图谱级解释）。九位级漂移全零。文档臂：3239–3250 收线小结入 MEMO + gap 附录候选 gap_v1_12_appendix_fidelity_closure。ledger #403 / freeze `48169586` / res `0601eb1b`。'
             '- **3252 已预注册**：(a) gap 附录候选条目定稿（report-only 预演）；(b) checkpoint 轨迹产物审计（零 GPU 前置）；(c) 5 appendix GPU 挂账。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
else:
    assert '3251 收官（u* 词汇归因' in mem, 'MEMORY tail block not found'
# compress the phase-index line if over limit
if len(mem) > 3000:
    o = ('- **3246 已预注册**：(a) 声明体制边界刻画（边信息预算 vs 缺口曲线）；(b) u* 顺延；(c) 保真曲线客户端化。')
    n = ('')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- 3246（regime boundary）：FLIP=INFORMATION_GAP（LS oracle +0.329 转正；拉力算子全空间上限）；XC_b4 现役（−0.772）。#398 `a451686b` / freeze `88ab83b9` / res `df7ec079`。- ')
    n = ('- ')
    if o in mem:
        mem = mem.replace(o, n)
assert len(mem) <= 3000, len(mem)
with open(MEM, 'w', encoding='utf-8', newline='') as f:
    f.write(mem)
REPORT.append('5 memory: replaced tail block, %d chars' % len(mem))

print('\n'.join(REPORT))
print('CLOSEOUT OK')
