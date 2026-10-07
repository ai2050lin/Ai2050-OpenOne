"""Phase 2994 closeout: ledger -> MEMO -> wslog -> MEMORY."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs',
                    'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                      'atlas_ledger.json')
RES = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913',
                   'phase2994', 'competition_hysteresis')
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\tmp_closeout2994_log.txt')
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
assert r['final_verdict'] == 'competition_floor_void'
created = e['created']
stamp = '%s %s' % (created[:10], created[11:16])
npz8 = s['npz_sha256_8']
res8 = s['result_sha256_8']
exe8 = e.get('script_sha256_8', 'NA')

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('meas_id') ==
           'meas2994_competition_hysteresis'
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': 'meas2994_competition_hysteresis',
        'phase': 2994,
        'claim': ('Omega-E NEGATIVE (plan v4 P3): mislead '
                  'injection along the class axis (per-word '
                  'mirror, L17 input, dose 1.6*n17, 2977 '
                  'protocol) does NOT displace the L34 '
                  'readout toward the wrong pole (median D '
                  '-0.16, 17/37 positive, p=0.15) - the '
                  'identity-path prediction (g*n17~32.9) '
                  'survives only 5.9% (median) for the '
                  'class axis while RANDOM directions pass '
                  'AMPLIFIED (168%): the erasure is class-'
                  'axis-specific = an active readout-axis '
                  'defense (competitive rebalancing, '
                  '2950/2983/2990 now demonstrated by '
                  'causal injection). Spatial washout '
                  'criteria (recovery fraction > 0 and > '
                  'random control) NOT passed - ratios are '
                  'degenerate on near-erased displacements '
                  '(registered: S1 p=1e-4 NOT claimable, '
                  'nondegenerate gate missing). Error-'
                  'attractor NOT operationalized in the '
                  'stateless forward; "phase transition" '
                  'stays banned'),
        'verdict': r['final_verdict'],
        'anchors': '8/8 (a2/a3 base vs 2986/2987 bit 0.00; '
                   'a4/a5 determinism+locality 0.00; a6 '
                   'guard med 2.13e-03 = bf16 ULP; a1 Vt8 '
                   '3.04e-08)',
        'artifacts': {
            'result': 'phase2994/competition_hysteresis/'
                      'result.json',
            'npz': 'phase2994/competition_hysteresis/'
                   'competition_hysteresis.npz'},
        'hashes': {'npz_sha256_8': npz8,
                   'result_sha256_8': res8,
                   'script_sha256_8': exe8},
        'note': ('run1 a5 dict-slice KeyError; run2 a6 '
                 'guard denominator miscalibrated (|d|max '
                 'vs max(|base|max,|d|max) -> 9.2e-2 '
                 'artifact; THIRD guard-denominator lesson '
                 'after 2992 run4), run3 authoritative. '
                 'Applicability: stateless forward only, '
                 'temporal KV-cache hysteresis untested; '
                 'L16 protocol, 37 en words')})
    for l in led['linkage']:
        if l['link_id'] == \
                'L14_readout_spectrum_cross_model':
            cons = l['connects']
            if not any(isinstance(c, dict)
                       and c.get('phase') == 2994
                       for c in cons):
                cons.append({
                    'phase': 2994,
                    'via': 'competition_hysteresis',
                    'adds': 'class-axis injection erased '
                            '(5.9% survival) while random '
                            'amplified (168%) = active '
                            'readout-axis defense; '
                            'error-attractor not '
                            'operationalized (stateless)'})
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
out.append('ledger n=%d L14 total=%d hash=%s'
           % (n_meas, len(l14[0]['connects']), h))

# ---------- 2. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
tag = '## Phase 2994:'
if tag not in memo:
    sec = (
        '## Phase 2994: Omega-E 竞争-滞后——类轴主动擦除、错误吸引子'
        '未操作化 [%(stamp)s]\n\n'
        '**判决：`competition_floor_void`**（run3 权威，34.5s，'
        '锚 8/8：base prof/headC 对 2986/2987 bit 0.00，确定性/'
        '局部性 0.00，注入守卫 med 2.13e-03=bf16 ULP）。\n\n'
        '### 设计\n'
        '方案 v4 P3 顺延（v3 Omega-E）：37 en 词（2977 口径）×L16'
        '（2986/2987 filler verbatim）；误导方向 m=unit(res34 类间'
        '均值差)，逐词镜像轴 a_i=s_i·m（F 推向 C 极、C 推向 F 极）；'
        'L17 self_attn 输入原位注入 delta=g·n17·a（2977 协议），'
        '剂量爬升 {0.1..1.6}@词位 + 空间洗出位置网格 {0,1,4,7,10,13}'
        '@1.6 + 低剂量 0.4 对照 + 逐词种子随机方向对照。判据（冻结）：'
        'T1 词位位移>0（单侧符号置换）；T2 恢复分数 R=D(p0)/D(p13)>0；'
        'T3 R>R_rand；判决映射四支。无状态前向的适用域限制预登记：'
        '时序 KV-cache 滞后未测。\n\n'
        '### 核心结果（重复三遍）\n'
        '**误导注入在词位不能把读出推向错误极（中位 D=-0.16，17/37 '
        '正，p=0.15）——恒等路径预测 g·n17≈32.9，类轴存活仅 5.9%%，'
        '而随机方向反而放大到 168%%：擦除是类轴特异的=读出轴存在主动'
        '防御（竞争重平衡 2950/2983/2990 首次获得因果注入端证明）。'
        '空间洗出判据未过（近零位移上的比值退化），错误吸引子在无状态'
        '前向内未操作化，"相变"命名继续禁用。**\n\n'
        '**类轴注入被主动擦除（存活 6%%）而随机方向反放大（168%%）——'
        '读出轴有防御；滞后判据未过，错误吸引子未操作化。**\n\n'
        '- 事后诊断（标注 post-hoc，npz 复算）：mis |D@word| med '
        '1.89 vs rand 56.17（恒等预测 32.9）；mis |D13| med 0.84 '
        'vs rand 54.55——擦除集中在读出相关轴，离轴扰动自由传播；\n'
        '- S1 剂量持续性 p=1e-4 **不可主张**：R 比值建立在近零位移上'
        '（10/37 词 D13 低于 floor），退化统计量非退化门缺失——本 '
        'Phase 自己踩中统计纪律红线，登记为流程教训；\n'
        '- u35 副轨迹已存档（proj_u_*）。\n\n'
        '### 硬伤（如实登记，均删产物重跑）\n'
        '1. run1：a5 锚对 dict 切片 KeyError（op 是 {li:arr} 字典）；\n'
        '2. run2：a6 注入守卫分母错配——|delta|max 低估 bf16 加法'
        '舍入（误差由操作数 ULP 主导），max 伪影 9.2e-2；归一化改 '
        'max(|base|max, |delta|max) 后 med 2.13e-03——**守卫分母'
        '教训第三次**（2992 run4 同款），对策入 MEMORY：bf16 守卫'
        '分母一律取操作数最大幅度；\n'
        '3. run3 权威，判决确定性成立。\n\n'
        '### 产物\n'
        '- result.json sha256_8=%(res8)s；npz sha256_8=%(npz8)s；'
        'script sha256_8=%(exe8)s；execution created=%(created)s。\n\n'
        % {'stamp': stamp, 'res8': res8, 'npz8': npz8,
           'exe8': exe8, 'created': created})
    memo = memo.rstrip('\n') + '\n\n' + sec
    io.open(MEMO, 'w', encoding='utf-8').write(memo)
    out.append('memo appended')
else:
    out.append('memo already has 2994')
tail = memo.split(tag)[1] if tag in memo else ''
out.append('memo residue %% = %s' % ('%(' in tail))

# ---------- 3. workspace log ----------
ws = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\2026-09-20.md')
line = ('- Phase 2994 Omega-E 竞争-滞后：判决 '
        'competition_floor_void（run3，锚 8/8）。类轴注入被主动擦除'
        '（存活 5.9%%，随机方向反放大 168%%）——读出轴防御首次因果证明；'
        '洗出比值退化，错误吸引子未操作化。硬伤 2 笔（dict 切片/'
        '守卫分母第三次）登记。Ledger 133 / hash %s。\n' % h)
wtxt = io.open(ws, encoding='utf-8').read() \
    if os.path.exists(ws) else ''
if 'Phase 2994' not in wtxt:
    io.open(ws, 'a', encoding='utf-8').write(line)
    out.append('wslog appended')

# ---------- 4. MEMORY.md ----------
MP = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\MEMORY.md')
mm = io.open(MP, encoding='utf-8').read()
old2993 = ('2993 Ω-D 逻辑签名：第三类在场（F<L<C p 4e-4）且长度'
           '稳健（d≈2.0 五档）——非上下文涌现。')
add994 = ('2994 Ω-E 竞争-滞后：类轴注入主动擦除（存活 6%，随机反'
          '放大 168%）——读出轴防御；错误吸引子未操作化。')
if add994 not in mm:
    if old2993 in mm:
        mm = mm.replace(old2993, old2993 + add994, 1)
    else:
        out.append('WARN 2993 anchor miss')
mm = mm.replace('## 机制链状态（2936-2993）',
                '## 机制链状态（2936-2994）', 1)
old_next = ('- max=2993，下一个 **2994**（A 主选 P3 顺延：Ω-E 竞争-'
            '滞后操作化；B Ω-F 跨模型 GLM4-9B 卡片复制——顺序不可反；'
            'C 2989 T3 加密+k 剂量；D 逻辑签名头级因果复测）。方案 v4 '
            '见 ' + r'research\gpt5\docs\plan_v4_micro_macro_merge.md。')
new_next = ('- max=2994，下一个 **2995**（A 主选 Ω-F 跨模型卡片复制 '
            'GLM4-9B——卡片集 v2 带适用域标签，锚结构按新模型重建；'
            'B 2989 T3 加密+k 剂量；C 逻辑签名头级因果复测；D KV-cache '
            '时序滞后补遗）。方案 v4 见 '
            + r'research\gpt5\docs\plan_v4_micro_macro_merge.md。')
if old_next in mm:
    mm = mm.replace(old_next, new_next, 1)
else:
    out.append('WARN next anchor miss')
for a, b in (
        ('2990 L12#2：消融 p=0.42——cls 集中=快照属性；',
         '2990 L12#2：消融 p=0.42=快照属性；'),
        ('2992 稀疏字典：对齐超 null（p_fam 0.014，cos 峰 0.10）'
         '但符号一致率 0.405<0.5——特征对齐=快照属性，SAE 不立项。',
         '2992 稀疏字典：对齐超 null 但符号一致率 0.405——特征对齐'
         '=快照，SAE 不立项。'),
        ('——非上下文涌现。', '——非涌现。')):
    if a in mm:
        mm = mm.replace(a, b, 1)
if len(mm) > 3000:
    out.append('WARN memory %d chars' % len(mm))
io.open(MP, 'w', encoding='utf-8').write(mm)
out.append('memory chars=%d max2994=%s star2995=%s'
           % (len(mm), 'max=2994' in mm,
              '**2995**' in mm))

out.append('CLOSEOUT DONE')
io.open(LOG, 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
