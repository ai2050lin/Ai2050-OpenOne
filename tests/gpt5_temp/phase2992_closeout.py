"""Phase 2992 closeout: ledger -> MEMO -> wslog -> MEMORY."""
import hashlib
import io
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs',
                    'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                      'atlas_ledger.json')
RES = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913',
                   'phase2992',
                   'sparse_dictionary_registration')
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\tmp_closeout2992_log.txt')
out = []


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


r = json.load(io.open(os.path.join(RES, 'result.json'),
                      encoding='utf-8'))
e = json.load(io.open(os.path.join(RES, 'execution.json'),
                      encoding='utf-8'))
s = json.load(io.open(os.path.join(RES, 'seal.json'),
                      encoding='utf-8'))
assert r['final_verdict'] == \
    'dictionary_aligned_snapshot_only'
created = e['created']
stamp = '%s %s' % (created[:10], created[11:16])
npz8 = s['npz_sha256_8']
res8 = s['result_sha256_8']
exe8 = e.get('script_sha256_8', 'NA')

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('meas_id') ==
           'meas2992_sparse_dictionary_registration'
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': 'meas2992_sparse_dictionary_registration',
        'phase': 2992,
        'claim': ('Lightweight overcomplete dictionary on '
                  'the L6-12 band (k-means k=96 + OMP k=16 '
                  'per layer): atom-to-readout-axis '
                  'alignment PASSES the 2931 random-'
                  'rotation null gate (p_fam_min 0.014 '
                  'after Bonferroni x14, 12/14 layer-axis '
                  'pairs at the null floor, max cos 0.100 '
                  '@11_cls vs null p95 0.035) BUT causal '
                  'load is redirected downstream: '
                  'realized d_lang_u effect at L34 input '
                  'keeps magnitude (ratio 1.06) yet flips '
                  'sign (agreement 0.405 < coin flip), '
                  'only 2.9x random-atom control - '
                  'feature-level alignment is a snapshot '
                  'property; full SAE not chartered'),
        'verdict': r['final_verdict'],
        'anchors': '7/7 (six bit-level 0.00; a1 Vt8 '
                   '3.04e-08; a2/a3 headC vs 2987, a4 act '
                   'vs 2989, a8 axes rebuild)',
        'artifacts': {
            'result': 'phase2992/'
                      'sparse_dictionary_registration/'
                      'result.json',
            'npz': 'phase2992/'
                   'sparse_dictionary_registration/'
                   'sparse_dictionary_registration.npz'},
        'hashes': {'npz_sha256_8': npz8,
                   'result_sha256_8': res8,
                   'script_sha256_8': exe8},
        'note': ('runs 1-3 died pre-verdict: cross-space '
                 'matmul (o_proj input is 4096-d head '
                 'concat, not 2560 residual), null-count '
                 'reachability violation (14/201=0.0697 > '
                 'gate, N_NULL raised to 1000), and '
                 'with_kwargs pre-hook must return '
                 '((args,),kwargs); run4 guard denominator '
                 'miscalibrated (|h|max vs bf16 ULP of '
                 'h-dh), run5 reran with fixed guard '
                 '(3.5e-03 = 2^-9), verdict reproduced '
                 'deterministically')})
    for l in led['linkage']:
        if l['link_id'] == \
                'L14_readout_spectrum_cross_model':
            cons = l['connects']
            if not any(isinstance(c, dict)
                       and c.get('phase') == 2992
                       for c in cons):
                cons.append({
                    'phase': 2992,
                    'via': 'sparse_dictionary_registration',
                    'adds': 'dictionary atom alignment = '
                            'snapshot property (null-'
                            'passed, causally redirected '
                            'downstream); SAE gate: not '
                            'chartered'})
led.pop('ledger_sha256_8', None)
h = hashlib.sha256(json.dumps(
    led, sort_keys=True,
    ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = h
with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(led, f, ensure_ascii=False, indent=1)
n_meas = len(led['measurements'])
l14 = [l for l in led['linkage']
       if l['link_id'] == 'L14_readout_spectrum_cross_model']
n14 = len([c for c in l14[0]['connects']
           if isinstance(c, dict)])
out.append('ledger n=%d L14=%d hash=%s' % (n_meas, n14, h))

# ---------- 2. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
tag = '## Phase 2992:'
if tag not in memo:
    sec = (
        '## Phase 2992: 稀疏字典登记——对齐超 null 但因果载荷下游重定向 '
        '[%(stamp)s]\n\n'
        '**判决：`dictionary_aligned_snapshot_only`**（run5 权威，'
        '39.6s，锚 7/7，其中六重 bit 级 0.00）。\n\n'
        '### 设计\n'
        '方案 v4 P2 门控实验：全量 SAE 训练立项前先回答"轻量过完备字典'
        '是否值得登记"。重跑 74 词×{L2,L16N} 双条件（op/x17/down_proj/'
        'L34 层输入四 hook 并挂，bit 级锚对 2987/2989）；T1 层 6-12 '
        '各在 148 样本上 k-means 字典（k=96 种子化）+ OMP(k=16)；T2 '
        '原子×读出轴（c_lang/c_cls）最大对齐 vs 1000 Haar 随机方向 '
        'null（2931 口径），Bonferroni ×14 族校正；T3 12 分层细胞 '
        'lang 轴 m=8 消融双口径：直路解析预测 dh·c_lang（残差恒等路径）'
        'vs L34 层输入残差沿 d_lang_u 的 realized 下游效应 + 随机原子'
        '对照。\n\n'
        '### 核心结果（重复三遍）\n'
        '**字典原子对读出轴的快照对齐是真的（p_fam_min 0.014<0.05，'
        '12/14 层×轴 p_raw 触 null 地板 0.001，最大 cos 0.100@11_cls、'
        '0.097@12_lang ≈ null p95 0.035 的 3 倍）；但其因果载荷在下游被'
        '系统性重定向：realized 幅度中位比 1.06（幅度守恒）而符号一致率'
        '仅 0.405（低于掷硬币），top-8 原子消融下游效应仅为随机原子对照'
        '的 2.9 倍——特征级对齐是快照属性，方向重写发生在层间传播中，'
        '2983/2985/2990 的"对单点操作化关闭"在字典原子粒度复现。全量 '
        'SAE 训练暂不立项：对齐可登记，因果承诺不可登记。**\n\n'
        '**字典原子对读出轴的快照对齐是真的；但因果载荷在下游被系统性'
        '重定向（符号一致率 0.405<0.5、幅度比 1.06）；特征对齐=快照'
        '属性，SAE 暂不立项。**\n\n'
        '- T1 recon 门过：层 6-11 中位 0.31-0.38，层 12 独低 0.112'
        '（L12 激活最可压缩，与 2989/2990 cls-L12 单点集中互证）；\n'
        '- T2：仅 7_cls（p 0.294）与 10_lang（p 0.056）未过 null；\n'
        '- T3：84 行全存档，mod_guard 中位 3.5e-03=bf16 ULP 2^-9'
        '（消融生效确证）；特异度 2.9x 有但非主载荷。\n\n'
        '### 解释（重复三遍）\n'
        '**快照对齐（几何）与因果载荷（传播）在字典原子粒度上分离：'
        '几何上原子确实携带读出轴信息（超随机旋转 null 3 倍），传播上'
        '其贡献方向被后续层重写（符号翻转六成而幅度守恒）——与 2983 '
        '正交重定向通道、2985 方向重写非旋转、2990 单点操作化关闭构成'
        '同一结论在头级/神经元级/特征级的三粒度收敛：本模型的语义载'
        '荷是分布式且传播动态的，任何静态对齐登记都只是快照。**\n\n'
        '### 硬伤（如实登记，均删产物重跑）\n'
        '1. run1：T3 直路预测 Wd@dh 方向反；run2 同位置暴露根因——'
        'M34v 是 4096 维头拼接空间（o_proj 输入=32×128）而非 2560 '
        '残差，pred_prof 口径作废，改残差空间 realized 口径（L34 层'
        '输入×d_lang_u）；\n'
        '2. run1 暴露判据可达性违规：N_NULL=200 时 p 地板 14/201='
        '0.0697>0.05，Bonferroni 门数学不可达——N_NULL 提至 1000'
        '（地板 0.014）；教训：判据可达性先检须包含 null 数×族校正'
        '联合检查；\n'
        '3. run3：with_kwargs pre-hook 返回裸 tensor 崩（协议要求 '
        '((args,),kwargs) 元组）；\n'
        '4. run4：mod_guard 分母错配（|h|max 低估 SwiGLU 稀疏激活下 '
        'bf16 ULP），归一化改 |h-dh|max 后=3.5e-03，run5 权威重跑'
        '判决确定性复现。\n\n'
        '### 产物\n'
        '- result.json sha256_8=%(res8)s；npz sha256_8=%(npz8)s；'
        'script sha256_8=%(exe8)s；execution created=%(created)s。\n\n'
        % {'stamp': stamp, 'res8': res8, 'npz8': npz8,
           'exe8': exe8, 'created': created})
    memo = memo.rstrip('\n') + '\n\n' + sec
    io.open(MEMO, 'w', encoding='utf-8').write(memo)
    out.append('memo appended')
else:
    out.append('memo already has 2992')
tail = memo.split(tag)[1] if tag in memo else ''
residue = '%(' in tail
out.append('memo residue %% = %s' % residue)

# ---------- 3. workspace log ----------
ws = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\2026-09-20.md')
line = ('- Phase 2992 稀疏字典登记：判决 '
        'dictionary_aligned_snapshot_only（run5，锚 7/7）。对齐超 '
        'null（p_fam 0.014，cos 峰 0.10≈3x p95）但因果符号一致率 '
        '0.405<0.5、幅度比 1.06、特异度 2.9x——特征对齐=快照属性，'
        'SAE 不立项。硬伤 4 笔登记（跨空间 matmul/null 可达性/'
        'hook 元组协议/guard 分母）。Ledger 131 / L14 99 / hash '
        '%s。\n' % h)
wtxt = io.open(ws, encoding='utf-8').read() \
    if os.path.exists(ws) else ''
if 'Phase 2992' not in wtxt:
    if not wtxt:
        io.open(ws, 'a', encoding='utf-8').write(
            '# 2026-09-20 工作日志\n')
    io.open(ws, 'a', encoding='utf-8').write(line)
    out.append('wslog appended')

# ---------- 4. MEMORY.md ----------
MP = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\MEMORY.md')
mm = io.open(MP, encoding='utf-8').read()
old2991 = ('2991 载体迁移：头级谱反相关（cos -0.45）重排，'
           'MLP 注册表保守（重叠 41-66/128，z 30-55）——'
           '头=路由，神经元=存储。')
add992 = ('2992 稀疏字典：对齐超 null（p_fam 0.014，cos 峰 '
          '0.10）但符号一致率 0.405<0.5——特征对齐=快照属性，'
          'SAE 不立项。')
if add992 not in mm:
    if old2991 in mm:
        mm = mm.replace(old2991, old2991 + add992, 1)
    else:
        out.append('WARN 2991 anchor miss')
mm = mm.replace('## 机制链状态（2936-2991）',
                '## 机制链状态（2936-2992）', 1)
old_next = ('- max=2991，下一个 **2992**（A 主选 P2 稀疏字典登记 '
            'L6-12，null 校准前置；B T3 加密+k 剂量；C 因果耦合；'
            'D P3 Ω-D/E）。方案 v4 见 '
            r'research\gpt5\docs\plan_v4_micro_macro_merge.md。')
new_next = ('- max=2992，下一个 **2993**（A 主选 P3 顺延：Ω-D 逻辑'
            '签名带适用域标签；B Ω-E 竞争-滞后操作化；C 2989 T3 '
            '加密+k 剂量；D 字典原子×注册表因果耦合）。方案 v4 见 '
            r'research\gpt5\docs\plan_v4_micro_macro_merge.md。')
if old_next in mm:
    mm = mm.replace(old_next, new_next, 1)
else:
    out.append('WARN next anchor miss')
if len(mm) > 3000:
    out.append('WARN memory %d chars, extra compress'
               % len(mm))
    for a, b in (
            ('2963 类效应 p2e-4→', '2963 类效应→'),
            ('2964 载体 L34/h15→', '2964 载体 h15→'),
            ('（消失 97%）', ''),
            ('（I-rho 0.94）', ''),
            ('（g 0.89）', '')):
        mm = mm.replace(a, b, 1)
io.open(MP, 'w', encoding='utf-8').write(mm)
out.append('memory chars=%d max2992=%s next2993=%s'
           % (len(mm), 'max=2992' in mm,
              'next **2993**' in mm))

out.append('CLOSEOUT DONE')
io.open(LOG, 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
