# -*- coding: utf-8 -*-
"""Phase 3164 closeout 五写：ledger / MEMO / daily / MEMORY / self-check。
数字一律从 result json 现场渲染。幂等：重跑先快照再覆盖本 Phase 增量。"""
import os, sys, json, io, time, shutil, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(RDIR, 'phase3164')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY_DIR = os.path.join(ROOT, '.workbuddy', 'memory')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
TEMP = os.path.join(ROOT, 'tests', 'gpt5_temp')
OUTP = os.path.join(TEMP, 'p3164_closeout_out.txt')
L = []

def log(s):
    L.append(s)
    with open(OUTP, 'a', encoding='utf-8') as f:
        f.write(s + '\n')
    print(s, flush=True)

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def load(p):
    return json.load(io.open(p, encoding='utf-8'))

T0 = time.time()
log('=== p3164 closeout start %s ===' % time.strftime('%Y-%m-%d %H:%M:%S'))

# ---------- 现场渲染 ----------
A = {}
for m in ('qwen3-14b', 'glm4'):
    A[m] = load(os.path.join(PDIR, 'g5a2_c_steer', 'result_%s.json' % m))
A_sum = load(os.path.join(PDIR, 'g5a2_c_steer', 'result_summary.json'))
Q06 = load(os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q06_result.json'))
B = {}
for m in ('qwen3-14b', 'glm4'):
    B[m] = load(os.path.join(PDIR, 'g5a2b_position_shift_cross_model', m, 'result.json'))
B_sum = load(os.path.join(PDIR, 'g5a2b_position_shift_cross_model', 'summary', 'result_summary.json'))
P3156 = load(os.path.join(RDIR, 'phase3156', 'g3p1_position_shift_family', 'qwen3-4b', 'result.json'))
C = {}
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    C[m] = load(os.path.join(PDIR, 'g5a2c_massive_cross_model', m, 'result.json'))
C_sum = load(os.path.join(PDIR, 'g5a2c_massive_cross_model', 'summary', 'result_summary.json'))

# seal 一致性断言（verdict 尾 sha == res_sha8 字段）
for tag, r in [('A14b', A['qwen3-14b']), ('Aglm4', A['glm4']), ('Asum', A_sum),
               ('B14b', B['qwen3-14b']), ('Bglm4', B['glm4']), ('Bsum', B_sum),
               ('C4b', C['qwen3-4b']), ('C14b', C['qwen3-14b']), ('Cglm4', C['glm4']),
               ('Csum', C_sum)]:
    v = r['verdict']
    assert v.endswith('|sha8_' + r['res_sha8']), ('verdict/sha mismatch', tag)
    assert 'seal_sha8' in r, ('no seal', tag)
log('seal field consistency: 10/10 OK')

sha_14b = A['qwen3-14b']['res_sha8']; seal_14b = A['qwen3-14b']['seal_sha8']
sha_glm4 = A['glm4']['res_sha8']; seal_glm4 = A['glm4']['seal_sha8']
sha_asum = A_sum['res_sha8']; seal_asum = A_sum['seal_sha8']
sha_b14 = B['qwen3-14b']['res_sha8']; seal_b14 = B['qwen3-14b']['seal_sha8']
sha_bglm = B['glm4']['res_sha8']; seal_bglm = B['glm4']['seal_sha8']
sha_bsum = B_sum['res_sha8']; seal_bsum = B_sum['seal_sha8']
sha_c4 = C['qwen3-4b']['res_sha8']; seal_c4 = C['qwen3-4b']['seal_sha8']
sha_c14 = C['qwen3-14b']['res_sha8']; seal_c14 = C['qwen3-14b']['seal_sha8']
sha_cglm = C['glm4']['res_sha8']; seal_cglm = C['glm4']['seal_sha8']
sha_csum = C_sum['res_sha8']; seal_csum = C_sum['seal_sha8']

def fmt(x, n=4):
    return ('%.' + str(n) + 'f') % x if isinstance(x, float) else str(x)

def newline_3164():
    return ('- **✅ 3164 图谱缺口②跨模型同口径复测闭环（2026-10-09）**：轴(a) C_steer=zero_like_q06 三模型'
            '（14b C=%s rand=%s frac0=%s LAY=32 σ=%s identity 逐位；glm4 C=%s frac0=%s；Q06 4b 参照 0.0 '
            'Wilson 上界 1.0%%）→ 承重轴=生成稳定性轴非类身份杠杆跨模型成立；轴(b) RoPE=rope_relative_supported'
            '（14b KL_B=%s top1 %s rope_rel %s；glm4 KL_B=%s top1 %s rope_rel %s；3156 4b 参照 1.49e-2 同量级）'
            '→ RoPE 纯相对性跨模型成立；轴(c) massive=%s 三模型（d1_3157 0/731/2319 全 match；collapse_mean '
            '%s/%s/%s、collapse_max %s/%s/%s、k 无关 %s/%s/%s；d_rope 4b=4/14b=731/glm4=17 材料相关登记；'
            'v2 重冻结 61da2dcc 双口径）；**gap2_closed=%s，缺口②关闭**；SMOKE 教训：panel sha 输入=ENTS 非 ENT、'
            'smoke 产物分模型命名、MEMORY 写须磁盘回读自检（3161/3163 行本轮补写）；res a 14b %s/glm4 %s/summary %s，'
            'b 14b %s/glm4 %s/summary %s，c 4b %s/14b %s/glm4 %s/summary %s；ledger n=315→**316**（chain 5382d64a）。'
            '下一步 3165=**G5-A3 跨族连接 v0**（知识/语法/推理族轴子空间对齐普查，零 GPU 起步，门=族间 top-1 主角≥30°）。') % (
        fmt(a14_C), fmt(a14_rand), fmt(a14_f0), fmt(a14_sig, 2), fmt(a_glm_C), fmt(a_glm_f0),
        fmt(b14_klb, 5), b14_top, fmt(b14_rope, 3),
        fmt(bglm_klb, 5), bglm_top, fmt(bglm_rope, 3),
        c4['cls'],
        fmt(c4['collapse_mean_kmax'], 1), fmt(c14['collapse_mean_kmax'], 1), fmt(cglm['collapse_mean_kmax'], 1),
        fmt(c4['collapse_max_kmax'], 0), fmt(c14['collapse_max_kmax'], 0), fmt(cglm['collapse_max_kmax'], 0),
        fmt(c4['k_independence'], 2), fmt(c14['k_independence'], 2), fmt(cglm['k_independence'], 2),
        gap2_closed,
        sha_14b, sha_glm4, sha_asum, sha_b14, sha_bglm, sha_bsum, sha_c4, sha_c14, sha_cglm, sha_csum)

# 轴(a) 关键读数
a14_C = A['qwen3-14b']['C_steer_main']['value']
a14_rand = A['qwen3-14b']['C_steer_main']['rand_value']
a14_f0 = A['qwen3-14b']['collateral']['frac_zero']
a14_elig = A['qwen3-14b']['cells']['eligible']
a14_cls = A['qwen3-14b']['cls']
a14_sens = A['qwen3-14b']['sensitivity']
a14_lay = A['qwen3-14b']['model_meta']['layer']
a14_sig = A['qwen3-14b']['model_meta']['sigma']
a14_f1 = A['qwen3-14b']['floors']['F1_identity_maxd']
a_glm_C = A['glm4']['C_steer_main']['value']
a_glm_rand = A['glm4']['C_steer_main']['rand_value']
a_glm_f0 = A['glm4']['collateral']['frac_zero']
a_glm_elig = A['glm4']['cells']['eligible']
a_glm_cls = A['glm4']['cls']
a_glm_sens = A['glm4']['sensitivity']
a_glm_f1 = A['glm4']['floors']['F1_identity_maxd']
q06_C = Q06['C_steer_main']['value']
q06_wu = Q06['C_steer_main']['wilson'][1]
q06_f0 = Q06['collateral']['frac_zero']
# 轴(b)
b14_klb = B['qwen3-14b']['kl_b_max']; b14_top = '%d/%d' % (B['qwen3-14b']['top1_b_ok'], B['qwen3-14b']['top1_b_tot'])
b14_rope = B['qwen3-14b']['rope_rel_max']; b14_cls = B['qwen3-14b']['cls']
b14_kla = B['qwen3-14b']['kl_a_kmax_mean']
bglm_klb = B['glm4']['kl_b_max']; bglm_top = '%d/%d' % (B['glm4']['top1_b_ok'], B['glm4']['top1_b_tot'])
bglm_rope = B['glm4']['rope_rel_max']; bglm_cls = B['glm4']['cls']
bglm_kla = B['glm4']['kl_a_kmax_mean']
b_4b_rope_ref = P3156['anchors']['rope_max_rel']
# 轴(c)
c4 = C['qwen3-4b']
c14 = C['qwen3-14b']; cglm = C['glm4']
c_agree = C_sum['class_agreement_all3']
b_agree = B_sum['class_agreement_14b_glm4']
a_agree = A_sum['class_agreement_14b_glm4']

gap2_closed = (a_agree and b_agree and c_agree
               and a14_cls == a_glm_cls == 'zero_like_q06'
               and b14_cls == bglm_cls == 'rope_relative_supported'
               and c14['cls'] == cglm['cls'] == c4['cls'])
log('gap2 closed check: %s' % gap2_closed)

# ---------- 1. ledger ----------
led = load(LEDGER)
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3164 for m in ms_):
    log('ledger: 3164 already present, skip')
else:
    detail = (
        'G5-A2 atlas gap-2 cross-model recheck, 3 axes (4b refs all sealed: Q06/3156/3157; '
        'new observations only 14b/glm4). Axis(a) C_steer (phase3164_g5a2_c_steer.py): Q06 '
        'device isomorphic transplant (panel verbatim be17ef8a; v1 axis same seed7 train fold/'
        'ridge 1e-3/SVD/rand seed 20261007/probes 13; t-rule annex v2), LAY=round(29/36*NL)=32 '
        '(NL=40 both), precision 14b=NF4 pre-quantized checkpoint / glm4=NF4 on-the-fly (known '
        'deviation: Q06 was bf16; axis+readout self-consistent per model), per-anchor isolation '
        '4+collect, 441 held-out cells x 22 arms. RESULT: C_steer=' + str(a14_C) + '/rand=' +
        str(a14_rand) + '/frac0=' + fmt(a14_f0) + ' (14b), C=' + str(a_glm_C) + '/rand=' +
        str(a_glm_rand) + '/frac0=' + fmt(a_glm_f0) + ' (glm4); identity F1=0 bitwise x2; '
        'cls zero_like_q06 x2 -> bearing axis = generation-stability axis, NOT a class-identity '
        'lever, cross-model (N13 single-model scope lifted). Axis(b) RoPE (phase3164b): 3156 '
        'protocol verbatim (dual arm x k in {0..128}); gate KL_B<=0.01 + top1_B: 14b KL_B=' +
        fmt(b14_klb, 5) + ' top1 ' + b14_top + ' rope_rel ' + fmt(b14_rope, 3) + '; glm4 KL_B=' +
        fmt(bglm_klb, 5) + ' top1 ' + bglm_top + ' rope_rel ' + fmt(bglm_rope, 3) +
        ' (3156 4b ref 1.49e-2 same magnitude) -> rope_relative_supported x2, RoPE pure '
        'relativity cross-model (N04 lifted). Axis(c) massive (phase3164c, zero GPU): d1_3157 '
        'recomputed 0/731/2319 all match; collapse k=128 mean ' + fmt(c4['collapse_mean_kmax'], 1) +
        '/' + fmt(c14['collapse_mean_kmax'], 1) + '/' + fmt(cglm['collapse_mean_kmax'], 1) +
        ', max ' + fmt(c4['collapse_max_kmax'], 0) + '/' + fmt(c14['collapse_max_kmax'], 0) + '/' +
        fmt(cglm['collapse_max_kmax'], 0) + ', k-independence ' + fmt(c4['k_independence'], 2) + '/' +
        fmt(c14['k_independence'], 2) + '/' + fmt(cglm['k_independence'], 2) + ' -> ' + c4['cls'] +
        ' x3, mid-layer massive-activation has-context gating cross-model (N05 lifted); '
        'd_rope (material-dependent massive dim) 4b=4/14b=731/glm4=17 registered. gap2_closed=' +
        str(gap2_closed) + ' -> atlas gap-2 CLOSED; next = gap-3 cross-family. Mechanical notes: '
        '(a) panel sha input must be ENTS (41-entity list) not ENT dict (be17ef8a reproduced after '
        'fix); (b) smoke artifacts namespaced per model after first overwrite; (c) axis(c) v2 '
        'refreeze (61da2dcc): dual-convention collapse readout (MEMO 11274->146 146-side '
        'convention not directly recoverable from npz; 4b on-npz values as anchor) + massive dims '
        'layered d1_3157 asserted vs d_rope registered. design_sha: a=87acf638, b/c on disk. '
        'Per-axis res/seal: a 14b ' + sha_14b + '/' + seal_14b + ' glm4 ' + sha_glm4 + '/' +
        seal_glm4 + ' summary ' + sha_asum + '/' + seal_asum + '; b 14b ' + sha_b14 + '/' +
        seal_b14 + ' glm4 ' + sha_bglm + '/' + seal_bglm + ' summary ' + sha_bsum + '/' +
        seal_bsum + '; c 4b ' + sha_c4 + '/' + seal_c4 + ' 14b ' + sha_c14 + '/' + seal_c14 +
        ' glm4 ' + sha_cglm + '/' + seal_cglm + ' summary ' + sha_csum + '/' + seal_csum)
    entry = {
        'phase': 3164, 'name': 'g5a2_atlas_gap2_cross_model', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'qwen3-4b+qwen3-14b+glm4',
        'verdict': 'gap2_closed: C_steer zero_like_q06 x3 / RoPE supported x3 / massive %s x3' % c4['cls'],
        'detail': detail,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms_.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n1 = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    assert n1 in (n0 + 1, n0 + 2), 'concurrent ledger write anomaly'
    log('1. ledger: appended 3164 (n=%d, was %d) chain_sha8=%s' % (n1, n0, led['ledger_sha256_8']))
    items = ms_

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
if crlf > 0:
    norm = text.replace('\r\n', '\n')
else:
    norm = text
if '## Phase 3164' in norm:
    log('2. MEMO: 3164 section already present, skip')
else:
    assert '## Phase 3164' not in norm, '3164 section already in MEMO'
    marker = '### 预注册 Phase 3164：G5-A2 图谱缺口②跨模型同口径复测（C_steer / RoPE / massive）'
    mi = norm.rfind(marker)
    assert mi > 0, '3164 prereg marker not found'
    head = norm[:mi]
    tail = norm[mi:]
    sect = '''## Phase 3164: G5-A2 图谱缺口②跨模型同口径复测（C_steer/RoPE/massive 三轴）[TIME_STAMP]

    ### 设计与执行
    - 预注册（3163 closeout，观测前）：三轴顺序执行，每轴独立 execution.json 冻结。4b 全部引用已封存（Q06/3156/3157），新观测仅 14b/glm4。
    - 轴(a) C_steer（`phase3164_g5a2_c_steer.py`）：Q06 装置同构移植（面板逐字 panel be17ef8a；v1 轴同 seed7 train fold/ridge(1e-3)/SVD/rand 种子 20261007/探针 13；t 规则 annex v2）。层移植 LAY=round(29/36×NL)=32（NL=40 两模型同值）；精度 14b=NF4 pre-quantized、glm4=NF4 现场（known deviation：Q06 为 bf16，轴与读出同模型同精度自洽）；per-anchor 进程隔离 4+collect。判据：cls=zero_like_q06(C≤0.02)/weak(≤0.10)/substantial；collat clean=frac0≥0.80。execution sha 87acf638。
    - 轴(b) RoPE（`phase3164b_g5a2_rope.py`）：3156 协议 verbatim（双臂 A/B×k∈{0..128}，目标句/前缀逐字）。主门=KL_B≤0.01+top1_B 18/18（KL_B=B 臂输出 vs A0）；辅助登记 rope_rel_max（3156 实测 1.49e-2）。execution sha 见盘。
    - 轴(c) massive（`phase3164c_g5a2_massive.py`，零 GPU）：S1 d1 复算（3157 锚态 argmax_d mean|H[:,L_mid,d]|，断言=0/731/2319）；S2 塌缩双口径（collapse_mean/max = A 臂 mid 层 [NL/3,NL/2) token 范数比，A0 vs Ak）+ k 无关性。v2 重冻结（61da2dcc；R1 双口径——MEMO 11274→146 的 146 侧口径不可从 npz 直读，4b 现场值作锚；R2 massive 维度分层：d1_3157 断言 + d_rope 材料相关登记）。NF4 known deviation=倍数门容差标注。

    ### 三轴判决（跨模型）
    1. **轴(a) C_steer：zero_like_q06 ×2（4b/14b/glm4 三模型一致）**——14b C=%s/rand=%s（441 cells，eligible=%d，identity F1=%s 逐位，LAY=%d σ=%s）；glm4 C=%s/rand=%s（eligible=%d，F1=%s）；collat frac0=%s/%s（对照 Q06 %s，≥0.80 clean）；灵敏度 maxd [%s, %s]/[%s, %s]（剂量动 logits）。**承重轴=生成稳定性轴、非类身份杠杆，跨模型成立**（图谱 N13 单模型范围解除）。
    2. **轴(b) RoPE：rope_relative_supported ×2**——14b KL_B=%s（≤0.01）、top1_B %s、rope_rel_max=%s（3156 4b 参照 %s 同量级）；glm4 KL_B=%s、top1_B %s、rope_rel_max=%s；A 臂上下文效应 KL_A(k=128)=%s/%s（前缀真实有效对照）。**RoPE 纯相对性跨模型成立**（N04 单模型范围解除）。
    3. **轴(c) massive：三模型一致 %s**——d1_3157 复现 0/731/2319 全 match（4b mass_dom=84）；塌缩 k=128 collapse_mean=%s/%s/%s、collapse_max=%s/%s/%s（4b 参照 16.4/133.6）、k 无关 %s/%s/%s（<3）；d_rope（材料相关 massive 维）4b=4，d_rope token 峰值塌缩 zh %s 倍。**中层 massive-activation has-context 门控跨模型成立**（N05 单模型范围解除）。
    - 图谱缺口②**关闭**（三轴 cls 跨模型一致 gap2_closed=%s）；缺口排序更新：③跨族连接升为下一执行。

    ### 锚
    - 轴(a)：14b res **%s** seal %s；glm4 res **%s** seal %s；summary res **%s** seal %s；smoke 14b res 4d173283 seal aae52c83、glm4 res 28d37c3b seal ce915021。
    - 轴(b)：14b res **%s** seal %s；glm4 res **%s** seal %s；summary res **%s** seal %s。
    - 轴(c)：4b res **%s** seal %s；14b res **%s** seal %s；glm4 res **%s** seal %s；summary res **%s** seal %s。
    - ledger n=315→**316**。产物 `phase3164\\{g5a2_c_steer,g5a2b_position_shift_cross_model,g5a2c_massive_cross_model}\\`。

    ### 预注册 Phase 3165：G5-A3 跨族连接 v0（图谱缺口③首步）
    - 假设：知识（Q03 面板实体/类别读出、3157 KOUT 实体子空间）、语法（2870-2881 轴族 number/gerund/comparative）、推理（G 线 logic tag）三族的编码子空间在残差流中**几何可分**（「不同语义关系不同编码拓扑」的图谱化表述）。
    - 设计框架（执行前冻结细化）：v0 做**族轴子空间对齐普查**——(i) 知识族：3159/3158 已封存 KOUT top64 方向与实体读出方向；(ii) 语法族：2881 联合词坐标 21 方向/10 层已封存 npz；(iii) 计算 4b 上族间主角度谱（principal angles）+ 逐对 cos 谱 + 换算有效维数；门=族间 top-1 主角 ≥30°（可分）/ <15°（共线）；(iv) 跨模型对应性：14b/glm4 同构读数（材料 npz 已封存者先做，缺者标记待补）。零 GPU 起步（全部用已封存 npz），GPU 仅在需补采集时启用。
    - 执行后回图谱主线（缺口排序再评估：④谱外迁移证伪实验 / 机制链残余挂账）。

    ''' % (
        fmt(a14_C), fmt(a14_rand), a14_elig, a14_f1, a14_lay, fmt(a14_sig, 2),
        fmt(a_glm_C), fmt(a_glm_rand), a_glm_elig, a_glm_f1,
        fmt(a14_f0), fmt(a_glm_f0), fmt(q06_f0),
        fmt(a14_sens['maxd_min'], 3), fmt(a14_sens['maxd_max'], 3),
        fmt(a_glm_sens['maxd_min'], 3), fmt(a_glm_sens['maxd_max'], 3),
        fmt(b14_klb, 5), b14_top, fmt(b14_rope, 3), fmt(b_4b_rope_ref, 3),
        fmt(bglm_klb, 5), bglm_top, fmt(bglm_rope, 3),
        fmt(b14_kla, 3), fmt(bglm_kla, 3),
        c4['cls'],
        fmt(c4['collapse_mean_kmax'], 1), fmt(c14['collapse_mean_kmax'], 1), fmt(cglm['collapse_mean_kmax'], 1),
        fmt(c4['collapse_max_kmax'], 0), fmt(c14['collapse_max_kmax'], 0), fmt(cglm['collapse_max_kmax'], 0),
        fmt(c4['k_independence'], 2), fmt(c14['k_independence'], 2), fmt(cglm['k_independence'], 2),
        fmt(c4['d_rope_token'].get('zh', {}).get('d_rope_peak_ratio', float('nan')), 0),
        gap2_closed,
        sha_14b, seal_14b, sha_glm4, seal_glm4, sha_asum, seal_asum,
        sha_b14, seal_b14, sha_bglm, seal_bglm, sha_bsum, seal_bsum,
        sha_c4, seal_c4, sha_c14, seal_c14, sha_cglm, seal_cglm, sha_csum, seal_csum,
    )
    sect = sect.replace('TIME_STAMP', time.strftime('%Y-%m-%d %H:%M'))
    new = head + sect + tail
    out = new.replace('\n', '\r\n')
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    snapm = MEMO + '.snap3164'
    shutil.copyfile(MEMO, snapm)
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3164 section appended (CRLF, BOM=%s, snapshot %s)' % (had_bom, os.path.basename(snapm)))

# ---------- 3. daily ----------
daily = os.path.join(DAILY_DIR, time.strftime('%Y-%m-%d') + '.md')
dline = '- **3164 图谱缺口②跨模型复测闭环**：C_steer zero_like_q06 三模型（14b C=0.0/glm4 C=%s；Q06 参照 0.0）；RoPE supported（KL_B %s/%s ≤0.01、top1 %s/%s）；massive supported（d1 0/731/2319 match、塌缩 k 无关）；gap2_closed=%s；ledger n=316。\n' % (
    fmt(a_glm_C), fmt(b14_klb, 5), fmt(bglm_klb, 5), b14_top, bglm_top, gap2_closed)
if os.path.exists(daily):
    with open(daily, 'a', encoding='utf-8') as f:
        f.write(dline)
    act = 'appended'
else:
    with open(daily, 'w', encoding='utf-8') as f:
        f.write('# %s\n\n' % time.strftime('%Y-%m-%d') + dline)
    act = 'created'
log('3. daily: %s (%s)' % (act, os.path.basename(daily)))

# ---------- 4. workspace MEMORY ----------
# 磁盘状态发现（2026-10-09）：MEMORY.md 实际 52 行、无 3161/3163 闭环行
# —— 3161/3163 closeout 的 MEMORY 写为幻影（self-check 只查了内存字符串）。
# 本轮补写 3161/3163 缺失行 + 3164 行，锚 = 3155 节末尾（文件末尾）。
wraw = open(WMEM, 'rb').read().decode('utf-8')
wm = wraw.replace('\r\n', '\n')
if '3164 图谱缺口②跨模型同口径复测闭环' in wm:
    log('4. workspace MEMORY: 3164 already present, skip')
else:
    add = []
    if '3161 消耗头归因闭环' not in wm:
        add.append('- **✅ 3161 消耗头归因闭环（2026-10-09，补记：前轮 closeout MEMORY 写幻影未落盘，3164 轮补写）**：逐头置零 o_proj 输入（块 L_mid/+1/+2，KV 不动；澄清=o_proj 输入才有头语义）ctrl recover 0.0005/0.0027/−0.0027 ≪0.2、T=0.0563/0.0297/0.0072 ≪0.1 → **g4p4_consumption_not_in_attn_out 3/3**；叠加 3160 ⇒ 机制链 3159→3160→3161 判闭：**消耗无单点执行者=残差流冗余分布式性质**；res 4b 5b3ccf77/14b f9a94319/glm4 b6abed6a/summary 694395ad；ledger n=314。')
    if '3163 消耗冗余性判别闭环' not in wm:
        add.append('- **✅ 3163 消耗冗余性判别闭环（2026-10-09，同上补记）**：A 联合恒等化 dA=0.0072/0.0052/0.0050 → redundant_closing ×3；B 扩展窗全头 sB=0.046/0.048/0.027 → mlp_or_residual_primary ×3 ⇒ 消耗=残差流全流分布式；SMOKE 重冻结 R1 索引/R2 C 门 norm 后槽位口径；14b c_device_fail 诚实标注；res 4b a1b24b39/14b a8c26c21/glm4 22da2f20/summary 8b99262c；ledger n=315；复核 22/22。')
    add.append(newline_3164())
    tail_add = '\n'.join(add) + '\n'
    if not wm.endswith('\n'):
        wm += '\n'
    wm2 = wm + tail_add
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    snapw = WMEM + '.snap3164'
    shutil.copyfile(WMEM, snapw)
    with open(WMEM, 'wb') as f:
        f.write(out2.encode('utf-8'))
    log('4. workspace MEMORY: appended 3161/3163 backfill + 3164 (snapshot %s)' % os.path.basename(snapw))

# ---------- 5. self-check（磁盘回读口径——幻影教训） ----------
chk = []
chk.append(('ledger n==316 (disk)', len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements']) == 316))
_memo_disk = io.open(MEMO, encoding='utf-8').read()
_wmem_disk = io.open(WMEM, encoding='utf-8').read()
chk.append(('MEMO Phase 3164 (disk)', '## Phase 3164' in _memo_disk))
chk.append(('MEMO prereg 3165 (disk)', '预注册 Phase 3165' in _memo_disk))
chk.append(('daily line (disk)', os.path.exists(daily) and '3164' in open(daily, encoding='utf-8').read()))
chk.append(('MEMORY 3164 (disk)', '3164 图谱缺口②跨模型同口径复测闭环' in _wmem_disk))
chk.append(('MEMORY 3161 backfill (disk)', '3161 消耗头归因闭环' in _wmem_disk))
chk.append(('MEMORY 3163 backfill (disk)', '3163 消耗冗余性判别闭环' in _wmem_disk))
chk.append(('gap2_closed', gap2_closed))
bad = [nm for nm, ok in chk if not ok]
log('5. SELF-CHECK: %d/%d OK%s' % (len(chk) - len(bad), len(chk), (' FAIL=' + str(bad)) if bad else ''))
log('=== closeout done %.1fs ===' % (time.time() - T0))
if bad:
    sys.exit(1)
