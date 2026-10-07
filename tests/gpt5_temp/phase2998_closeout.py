# -*- coding: utf-8 -*-
"""Phase 2998 closeout: Ledger 137 -> L14 -> MEMO append ->
workspace log -> MEMORY.md."""
import hashlib
import io
import json
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase2998'
     r'\omega_f2_s_c_injection_glm4')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_DIR = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory')
LOGF = R + r'\closeout_log.txt'
o = []


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['final_verdict']
assert verdict == 'sep_curve_nonmonotone_glm4', verdict
assert res['anchor_all_ok'] is True

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
already = any(m.get('phase') == 2998
              for m in led['measurements'])
if not already:
claim = (
    'Omega-F2 s_c injection machine on GLM4-9B: NOT '
    'REPLICATED (sep_curve_nonmonotone_glm4). 2945 machine '
    'rebuilt GLM4-native: 98 cells verbatim (2996 glm arm), '
    'dirs_g full 40-layer rebuild bit-equal to 2996 npz '
    '(a1 0.0), Vt8_g + u39 readout, in-session dcks = c8'
    '(null0)-c8(func) per word (2939 coords construction '
    'verified equivalent), xdir = dcks_S @ Vt8_S, '
    'single-layer injection at L17/18/19, s in '
    '{0.25..2.0} x K3, dimensionless cross-model gates '
    '(SEP_REF=0.5*sep_f, STEEP_REF=0.2*sep_f). RESULT: '
    'injection armed and propagating (ratio rises with s) '
    'but the language separation is nearly IMMUNE: ratio '
    '<= 0.027 at s=2 (qwen 2945 reached 0.86), sep '
    'collapse <= 0.7% vs qwen ~46%, L19 slightly REVERSES '
    '(85.7->85.9). The three-subspace displacement '
    'direction is ~98% laundered between L17-19 attn_in '
    'and the u39 readout on GLM4. Structural note: GLM4 '
    'null0 baseline itself carries sep 61.4 (72% of func '
    '85.7) - class separation is carried by the word token '
    'w, not the context word (qwen null0 ~ 0). Anchors '
    '9/9 incl. cross-run bit determinism (run4/run5 '
    'identical npz hash).')
meas = {
    'meas_id': 'meas2998_omega_f2_inj',
    'phase': 2998,
    'claim': claim,
    'verdict': verdict,
    'anchors': '9/9 (a0 order; a8 collision; a1 dirs_g '
               'bit-equal 0.0 x3 rows; a7 unit 1e-16; a2 '
               'det 0.0; a6 sep_f 85.7>0; a3/a4 identity '
               '2.5e-14; a5 K3 det 1.4e-14)',
    'artifacts': {
        'result': 'phase2998/omega_f2_s_c_injection_glm4/'
                  'result.json',
        'npz': 'phase2998/omega_f2_s_c_injection_glm4/'
               'omega_f2_s_c_injection_glm4.npz'},
    'hashes': {
        'npz_sha256_8': seal['npz_sha256_8'],
        'result_sha256_8': seal['result_sha256_8'],
        'script_sha256_8': exe['script_sha256_8']},
    'note': '5 runs: run1 a0 wrong word-order protocol + '
            'uninitialized a5_diff crash; run2 injection '
            'never armed (empty layer set + vec unset, '
            'ratio flat 0); run3 ALL three layers injected '
            'simultaneously (identical curves); run4 '
            'protocol-faithful; run5 authoritative '
            'replication (identical npz hash). Scale is '
            'self-calibrated dimensionless (qwen absolute '
            'gates 100/40 not copied). Negative registered '
            'per discipline.',
}
led['measurements'].append(meas)
assert len(led['measurements']) == 137
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
assert all((c.get('meas_id')
            if isinstance(c, dict) else c)
           != 'meas2998_omega_f2_inj'
           for c in l14['connects'])
l14['connects'].append({
    'meas_id': 'meas2998_omega_f2_inj',
    'phase': 2998,
    'axis': 'lang',
    'verdict': verdict,
    'grade_change': '2945 s_c machine on glm4: '
                    'non-monotone/immune (ratio<=0.027 '
                    'vs qwen 0.86; L19 reverses)'})
led.pop('ledger_sha256_8')
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
led['ledger_sha256_8'] = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(led, f, ensure_ascii=False, indent=1)
o.append('ledger n=%d l14=%d sha=%s'
         % (len(led['measurements']),
            len(l14['connects']),
            led['ledger_sha256_8']))

# ---------- MEMO append ----------
memo = io.open(MEMO, encoding='utf-8').read()
assert '## Phase 2998:' not in memo
sec = u'''## Phase 2998: Ω-F2 s_c 注入机器 GLM4 版——强洗消负结果 [%(created)s]

**判决：`sep_curve_nonmonotone_glm4`**（run5 权威，297.6s，锚 9/9；run4/run5 npz hash 相同=跨 run bit 级确定性）

### 设计（预注册冻结）
GLM4 原生重建 2945 注入机器：98 cells verbatim（2996 glm arm 词表）；pass1 全 40 层 dirs_g 重建（a1：L17/18/19 vs 2996 npz dirs_attn_glm **bit 级 0.00**——2996 捕获协议跨卡复现）；Vt8_g=SVD[:8]、u39=dirs_g[39] 读出；in-session dcks=c8(null0)−c8(func) 逐词（与 2939 coords 构造同构，已核 2939 源码）；xdir=dcks_S@Vt8_S（S_IDX=0,1,4）；单层注入 L17/18/19、s∈{0.25..2.0}×K3、中位读出。**跨模型刻度无量纲化**（qwen 绝对门 100/40 是 2560 维 u35 量，禁复制）：SEP_REF=0.5×sep_f、STEEP_REF=0.2×sep_f；ratio 门 0.86±0.3 不变（无量纲）。

### 结果（负，如实登记）
- 注入已武装且在传播（ratio 随 s 单调升 0.009→0.027），但语言分离近乎**免疫**：s=2 时 ratio ≤0.027（qwen 2945 同刻度 0.86），sep 塌缩 ≤0.7%%（qwen ~46%%），L17/18 近平坦、**L19 反向微升**（85.7→85.9）；
- T1 fail（spearman 未达 −0.9、陡降 ≪0.2×sep_f）、T2 s_c 不存在（sep 从未低于 42.9）；
- 结构注记：GLM4 null0 基线本身 sep=61.4（func 的 72%%；qwen null0≈0）——**GLM4 类分离由词 token w 携带、上下文词贡献小**，与 qwen 的上下文敏感结构不同。

### 解读（命名前审计链已过）
xdir 方向在 GLM4 L17-19 attn_in→u39 读出之间被**~98%% 洗消**。三个候选：(a) Ω-E 读出轴防御的 GLM4 版（防御更早/更强）；(b) 语言机制层位偏移（2996 K2 显示 GLM4 注册表 L7/10/13，可能定型更早）；(c) 2995 式 L39 末层对齐使 u39 对中带扰动不敏感。判别实验=敏感带深度扫描（s=2 单点扫层）——2999 主选。

### 硬伤（4 笔，均删产物重跑）
run1 a0 词序协议错（2996 glm src 是 F_en+C_en+24L_CAND+F_fr+C_fr=98 而非 2972 的 74）+ anchor-fail 路径 a5_diff 未初始化崩溃；run2 注入未武装（inj.layers 恒空 + inj.vec 未设→ratio 平 0）；run3 三层同时注入（曲线逐值相同暴露）；run4 协议保真权威；run5 复现（npz hash 相同）。correction_note 全登记。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase2998/omega_f2_s_c_injection_glm4/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger 137 / L14 %(l14)d。

**接续**：候选 2999：A（主选）GLM4 敏感带深度扫描（s=2 单点注入扫 L8-31，定位洗消带/等效 s_c）；B qwen↔GLM4 位移传播比逐层剖面（定量防御带）；C 2989 T3 加密复测+k 剂量；D Ω-E 错误吸引子操作化。
''' % {'created': created,
       'script8': exe['script_sha256_8'],
       'result8': seal['result_sha256_8'],
       'npz8': seal['npz_sha256_8'],
       'exec8': seal['exec_sha256_8'],
       'l14': len(l14['connects'])}
memo += '\n' + sec
with io.open(MEMO, 'w', encoding='utf-8') as f:
    f.write(memo)
o.append('memo +%d chars' % len(sec))

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-20.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
line = ('- Phase 2998 Omega-F2 s_c injection glm4: '
        'verdict sep_curve_nonmonotone_glm4 (negative, '
        'anchors 9/9, ratio<=0.027 vs qwen 0.86, L19 '
        'reverses); ledger 137/L14 %d.\n'
        % len(l14['connects']))
if 'Phase 2998' not in prev:
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
o.append('wlog appended')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
