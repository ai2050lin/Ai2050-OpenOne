"""Phase 2991 closeout: ledger -> MEMO -> wslog -> MEMORY."""
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
                   'phase2991', 'carrier_migration_atlas')
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\tmp_closeout2991_log.txt')
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
assert r['final_verdict'] == 'carrier_identity_rearranged'
created = e['created']
stamp = '%s %s' % (created[:10], created[11:16])
npz8 = s['npz_sha256_8']
res8 = s['result_sha256_8']
exe8 = e.get('script_sha256_8', 'NA')

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('meas_id') ==
           'meas2991_carrier_migration_atlas'
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': 'meas2991_carrier_migration_atlas',
        'phase': 2991,
        'claim': ('L34 F/C carrier identity is '
                  'protocol-relative and migrates by '
                  'readout-geometry reassignment, not '
                  'circuit relocation: head spectrum '
                  'anti-correlates L2 vs L16N '
                  '(cos -0.448, beyond-gate fail, '
                  'p_cos 0.515) while the 2989 MLP '
                  'neuron registry stays conservative '
                  'under context (top-128 overlap '
                  '41-66/128, z 31-55); h15 flips sign '
                  '(0.849 -> -0.271), h11/h23 were '
                  'rank-6/7 secondary carriers at L2'),
        'verdict': r['final_verdict'],
        'anchors': '9/9 (six bit-level 0.00; a2/a3 '
                   'headC vs 2987, a4 act vs 2989, '
                   'a8 axes rebuild)',
        'artifacts': {
            'result': 'phase2991/carrier_migration_atlas/'
                      'result.json',
            'npz': 'phase2991/carrier_migration_atlas/'
                   'carrier_migration_atlas.npz'},
        'hashes': {'npz_sha256_8': npz8,
                   'result_sha256_8': res8,
                   'script_sha256_8': exe8},
        'note': ('execution.json freeze was missing in '
                 'runs 1-4 (process violation); run5 '
                 'refroze and reran, verdict reproduced '
                 'deterministically')})
    for l in led['linkage']:
        if l['link_id'] == 'L14_readout_spectrum_cross_model':
            cons = l['connects']
            if not any(isinstance(c, dict)
                       and c.get('phase') == 2991
                       for c in cons):
                cons.append({
                    'phase': 2991,
                    'via': 'carrier_migration_atlas',
                    'adds': 'head carrier = readout '
                            'routing (protocol-relative); '
                            'MLP registry = content storage '
                            '(context-conservative)'})
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
tag = '## Phase 2991:'
if tag not in memo:
    sec = (
        '## Phase 2991: 载体迁移图谱——头级身份重排、神经元注册表保守 '
        '[%(stamp)s]\n\n'
        '**判决：`carrier_identity_rearranged`**（run5 权威，43.2s，'
        '锚 9/9 含六重 bit 级 0.00）。\n\n'
        '### 设计\n'
        '方案 v4 P1b：2987 发现 h15 载体 len-2 专属且 L16N 迁移至 '
        'h11/h23，但其迁移本质（连续重加权 vs 离散重排）未定。离线 '
        '复用 2987 npz headC (5 条件×74×32) 做全头普查（perm_p '
        'verbatim，N_PERM=10000），重跑 74 词×{L2,L16N} 双条件前向'
        '（op/x17/down_proj 三 hook 并挂）。\n\n'
        '### 核心结果\n'
        '- T1 五条件全头图谱：L2 显著载体 11 头（h15 contrast '
        '0.849 top1）；L16N 仅 6 头且 h15 **反号**（-0.271）；'
        'h15 曲线 0.849→0.217→-0.271→-0.574→0.084——上下文不仅'
        '杀死 h15 还翻转其符号；L16R（随机 token）反而 14 头显著。\n'
        '- T2（主判决）：谱 cos12 = **-0.448**（L2 vs L16N 反相关），'
        '秩相关 0.562，p_cos 0.515（变化幅度落标签置换 null 内），'
        'Jaccard 0.214；**h11/h23 在 L2 已是第 6/7 位次级载体**，'
        'h15 在 L16N 仍是 |c| 第一（0 位）但反号且不显著。\n'
        '- T3 神经元侧（2989 delta 口径 verbatim）：**top-128 '
        '注册表在 L16N 高度保守**——与 2989 名册重叠 41-66/128'
        '（z 31-55 vs 随机 1.7）；轴谱 cos lang 0.37-0.82、'
        'cls 0.33-0.87。\n'
        '- T4 合成：头级谱变化 |0.448| vs 神经元级平均 0.694'
        '——**头级身份重组比神经元注册表变化剧烈得多**。\n\n'
        '### 解释（重复三遍）\n'
        '**头级"载体"是读出路由的表面对象（协议相对、谱反相关），'
        '其下的 MLP 神经元注册表是内容存储（上下文保守、名册过半'
        '存活）——2987 的"载体迁移"不是计算电路迁移，而是 L34 读出'
        '几何在头间的重新分配；2989 的分布式注册表不受协议切换影响。'
        '头=路由、神经元=存储的双层分工首次被同一实验体系并表证实。**\n\n'
        '### 硬伤与流程违规（如实登记）\n'
        '1. run1：dirs27 键错（2939 npz 无此键；正确来源 2927 '
        'dirs_word，2939 仅提供 Vt8 锚）；\n'
        '2. run2：SRC_2927 常量漏定义；\n'
        '3. run3：a1w words 格式错（裸词 vs 存档 F:en:he 复合'
        '口径）+ result 构造在 anchor_fail 路径引用未定义 atlas'
        '（预初始化缺口第三次）；\n'
        '4. **run4 结果正确但缺 execution.json 冻结块（流程违规：'
        '预注册未在执行前落盘）——run5 补冻结重跑，判决确定性复现**。\n\n'
        '### 产物\n'
        '- result.json sha256_8=%(res8)s；npz sha256_8=%(npz8)s；'
        'script sha256_8=%(exe8)s；execution created=%(created)s。\n\n'
        % {'stamp': stamp, 'res8': res8, 'npz8': npz8,
           'exe8': exe8, 'created': created})
    memo = memo.rstrip('\n') + '\n\n' + sec
    io.open(MEMO, 'w', encoding='utf-8').write(memo)
    out.append('memo appended')
else:
    out.append('memo already has 2991')
# placeholder residue check
tail = memo.split(tag)[1] if tag in memo else ''
residue = ('%(' in tail) or ('%(stamp)' in tail)
out.append('memo residue %% = %s' % residue)

# ---------- 3. workspace log ----------
ws = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\2026-09-20.md')
line = ('- Phase 2991 载体迁移图谱：判决 '
        'carrier_identity_rearranged（run5，锚 9/9）。头级谱反相关'
        '（cos -0.448）+ h15 反号，MLP top-128 注册表保守（重叠 '
        '41-66/128，z 31-55）——头=读出路由、神经元=内容存储。'
        '流程违规登记：run1-4 缺 execution.json 冻结，run5 补冻结'
        '重跑判决复现。Ledger 130 / L14 98 / hash %s。\n' % h)
wtxt = io.open(ws, encoding='utf-8').read() \
    if os.path.exists(ws) else ''
if 'Phase 2991' not in wtxt:
    io.open(ws, 'a', encoding='utf-8').write(line)
    out.append('wslog appended')

# ---------- 4. MEMORY.md ----------
MP = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\MEMORY.md')
mm = io.open(MP, encoding='utf-8').read()
old2990 = ('2990 L12#2 深挖：快照极端但消融 p=0.42+无剂量门+h15 '
           '无耦合——cls 集中也是快照属性，单点操作化双轴关闭')
new2990 = '2990 L12#2：快照极端但消融 p=0.42——cls 集中=快照属性'
add991 = ('；2991 载体迁移：头级谱反相关（cos -0.45）重排，'
          'MLP 注册表保守（重叠 41-66/128，z 30-55）——'
          '头=读出路由，神经元=存储')
if old2990 in mm:
    mm = mm.replace(old2990, new2990, 1)
if '2991 载体迁移' not in mm:
    mm = mm.replace(new2990, new2990 + add991, 1)
old_next = ('- max=2990，下一个 **2991**（A 主选 P1b 载体迁移图谱：'
            '2987 h11/h23 x 2989 注册表并表；B T3 加密+k 剂量；'
            'C P2 稀疏字典；D 30 词集激活-读出错位）。方案 v4 见 '
            r'research\gpt5\docs\plan_v4_micro_macro_merge.md。')
new_next = ('- max=2991，下一个 **2992**（A 主选 P2 稀疏字典登记：'
            'L6-12 承重带 MLP overcomplete dict，随机 null 校准前置；'
            'B 2989 T3 加密复测+k 剂量；C 载体-注册表因果耦合；'
            'D 方案 v4 P3 Ω-D/E）。方案 v4 见 '
            r'research\gpt5\docs\plan_v4_micro_macro_merge.md。')
if old_next in mm:
    mm = mm.replace(old_next, new_next, 1)
else:
    out.append('WARN next anchor miss')
mm = mm.replace('## 机制链状态（2936-2970，34 卡入册）',
                '## 机制链状态（2936-2991）', 1)
if len(mm) > 3000:
    out.append('WARN memory %d chars, extra compress'
               % len(mm))
    mm = mm.replace('语言主调制×词类英文特化',
                    '语言主调制×词类特化', 1)
io.open(MP, 'w', encoding='utf-8').write(mm)
out.append('memory chars=%d max2991=%s next2992=%s'
           % (len(mm), 'max=2991' in mm,
              'next **2992**' in mm))

out.append('CLOSEOUT DONE')
io.open(LOG, 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
