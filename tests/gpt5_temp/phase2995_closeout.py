# -*- coding: utf-8 -*-
"""Phase 2995 closeout: Ledger -> MEMO -> ws log -> MEMORY."""
import hashlib
import io
import json
import time

LED = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
       r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WS = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\2026-09-20.md')
MEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\MEMORY.md')
R = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase2995'
     r'\omega_f1_glm4_panel')
OUTLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\tmp_closeout2995_log.txt')

r = json.load(io.open(R + r'\result.json', encoding='utf-8'))
s = json.load(io.open(R + r'\seal.json', encoding='utf-8'))
e = json.load(io.open(R + r'\execution.json', encoding='utf-8'))
out = []

# ---------- 1. ledger ----------
led = json.load(io.open(LED, encoding='utf-8'))
hc_old = led.pop('ledger_sha256_8')
meas_id = 'meas2995_omega_f1_glm4_panel'
if not any(m.get('meas_id') == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'phase': 2995,
        'claim': 'Omega-F1 first panel on GLM4-9B (plan v4 P3 '
                 '/ v3 Omega-F): weight-bearing-band card '
                 'replication. T1 M2963 lang-class separation '
                 'REPLICATED (F median 8.25 vs C 14.75 on the '
                 'GLM4 language axis at L39, two-sided '
                 'label-perm p=0.0014; direction matches qwen '
                 'F-low pole). T2 M2947 head concentration '
                 'REPLICATED (top1 h13, maxT min p=5.0e-4 = '
                 '2000-perm floor). T3 M2989 MLP registry '
                 'NOT replicated (top-128 |up-row . dirs_mlp| '
                 'z=-1.75/-2.25/-1.89 at L7/10/13, BELOW Haar '
                 'null; qwen 2989 z=31-55) - first cross-model '
                 'divergence point. L-class position DIVERGES: '
                 'L median 22.81 sits ABOVE C (qwen: F<L<C '
                 'middle) - logic-class signature rearranged '
                 'across models (descriptive). Grades: '
                 'replicated/replicated/not_replicated -> '
                 'omega_f1_panel_partial',
        'verdict': r['final_verdict'],
        'anchors': '7/7 (a1 sweep determinism 0.00; a2 unit '
                   'norm 1.1e-16; a3 tokenizer exact; a4 '
                   'headC recompute 1.7e-16; a5 registry '
                   'refetch 0.00; a6 perm rerun identical; '
                   'a7 batch==1 asserted)',
        'artifacts': {
            'result': 'phase2995/omega_f1_glm4_panel/'
                      'result.json',
            'npz': 'phase2995/omega_f1_glm4_panel/'
                   'omega_f1_glm4_panel.npz'},
        'hashes': {
            'npz_sha256_8': s['npz_sha256_8'],
            'result_sha256_8': s['result_sha256_8'],
            'script_sha256_8': e.get('script_sha256_8')},
        'note': 'run1 pos-1 row slice error; run2 numpy '
                'int64 json keys; run3 T1 null-pool bug (L '
                'words pooled into C: F vs C+L mismatched '
                'null); run4 DOUBLE-PERMUTATION bug (pool[pm] '
                'paired with lab[pm] recovers original groups '
                '- perm |d| constant 6.5008; fixed to single-'
                'perm positional split); run5 authoritative. '
                'Applicability: glm4-9b len-2 en classes, '
                'weight-side snapshot, causal ablation '
                'untested; T3 caliber dirs_mlp=post-LN2 '
                'space (space-matched re-audit candidate)'})
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    if not any(isinstance(c, dict) and c.get('phase') == 2995
               for c in l14['connects']):
        l14['connects'].append({
            'phase': 2995,
            'via': 'omega_f1_glm4_panel',
            'adds': 'GLM4-9B: lang-class separation and head '
                    'concentration replicate (T1 p=0.0014, T2 '
                    'maxT floor); MLP registry alignment does '
                    'NOT (z negative vs qwen 31-55); L-class '
                    'axis position rearranged (L above C)'})
calc = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = calc
io.open(LED, 'w', encoding='utf-8').write(
    json.dumps(led, ensure_ascii=False, indent=1))
l14c = [l for l in led['linkage']
        if l.get('link_id')
        == 'L14_readout_spectrum_cross_model'][0]
out.append('ledger n=%d hash=%s L14=%d p2995=%s' % (
    len(led['measurements']), calc,
    len(l14c['connects']),
    any(isinstance(c, dict) and c.get('phase') == 2995
        for c in l14c['connects'])))

# ---------- 2. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
tag = '## Phase 2995:'
if tag not in memo:
    tmpl = '''

## Phase 2995: Ω-F1 GLM4-9B 承重带三卡复制 %(stamp)s

**判决：`omega_f1_panel_partial`**（run5 权威，107.9s，锚 7/7：双 sweep 确定性 0.00、单位范数 1.1e-16、tokenizer 恒等、headC/注册表重构 1.7e-16/0.00、置换重放计数恒等、batch==1 断言）

### 原理与设计
方案 v4 P3 顺延（v3 Ω-F 首面板）：跨模型复制顺序不可反——先复制后找新原语。第二模型 glm4-9b-chat-hf（hidden 4096 / 40 层 / 32 头×128 / GQA kv=2 / 融合 gate_up_proj，bf16 全确定性探针 bit 0.0）。承重带三卡同协议复制：词源 2972 exec cells + 2993 L 候选，GLM4 tokenizer 单 token 过滤后 98 词（en 61 = F15+C22+L24；非 en 37）；seq=[the,w] pos-1 双空间捕获（attn_in=post-LN1、mlp_in=post-LN2，口径分账禁混用）；dirs_attn/dirs_mlp = unit(en 均值 − 非 en 均值)（2927 word-probe 口径逐层重建）。T1 M2963 语言类分离（L39 投影，F∪C 池单置换位置切分 N=4999 双侧）；T2 M2947 头集中（C39=u39@Wo39 头切片，32 头 maxT N=2000）；T3 M2989 MLP 注册表（L7/10/13 up 行×dirs_mlp，top-128 均值，Haar null N=1000）。

### 结果（重复三遍）
**跨模型第一版 Scaling Law（卡片级）：语言类分离与头级读出集中是跨规模稳健机制（GLM4-9B 复制：T1 F 中位 8.25 vs C 14.75，p=0.0014，F 低极方向与 qwen 一致；T2 top1 h13，maxT min p=5.0e-4 触 2000-perm 地板）；MLP 注册表对齐是首个跨模型分化点（T3 三层 z 全负 -1.75/-2.25/-1.89，obs 低于 Haar null，与 qwen 2989 的 z 31-55 鲜明反差）；且逻辑词签名跨模型重排（L 中位 22.81 高于 C——qwen 2993 为 F<L<C 中间位，descriptive）。头=路由、类分离=读出几何的骨干跨模型守恒；神经元级对齐随架构（融合 SwiGLU、口径空间）而变。**
- 判据分级：T1 复制 / T2 复制 / T3 不复制 → omega_f1_panel_partial（per-card 登记入 ledger）；
- 适用域标签（plan v4 P0）：glm4-9b / len-2 / en 类 / weight-side 快照；因果消融未测（untested）；T3 口径 dirs_mlp=post-LN2 空间（与 2989 残差 dirs 的空间匹配复审列为候选）；
- T2 top-5 头 d：h13 +0.147、h12 −0.108、h9 +0.103、h5 +0.088、h29 +0.084——头级语言读出在 GLM4 同样稀疏集中（top1 share 0.115 在 perm null p95=0.130 之下但 maxT 挣得）。

### 硬伤（4 笔，均删产物重跑；两笔统计方法学教训入册）
1. run1：pos-1 行切片错误（ai[0,:,:] 形状 (2,4096) 赋 (40,4096)）；
2. run2：numpy int64 字典键入 json 崩（转 int() 纪律再犯再登记）；
3. run3：**T1 null 池 bug**——置换标签把 L 词并入 C 组（实为 F vs C+L），null 与检验不同分布，p=1.0 虚报；
4. run4：**双置换 bug（方法学，最重）**——pool[pm] 配 lab[pm] 同索引置换恒复原原始分组，perm |d| 常数 6.5008、p=1.0；sanity 探针定位后改单置换+位置切分（nF/nF 后段），a6 重放计数 6 vs 6 恢复真实波动。教训：置换检验必须单置换（值序列置换后按位置切分）或符号翻转，严禁 values 与 labels 用同一 pm 各自置换后配对。run5 权威，判决确定性成立。

### 入账
Ledger %(nled)d 条 / L14 %(nl14)d / hash %(led8)s；产物 hash：npz %(npz8)s、result %(res8)s、script %(scr8)s。工作区日志、MEMORY.md 同步。

### 接续
下一 Phase 2996：A（主选）Ω-F2 续卡复制（s_c 开关注入机器卡 2945/2953 + 词盲卡 2940 + L16 稳健卡，GLM4 注入协议重建）；B T3 空间口径复审（dirs_mlp post-LN2 vs 残差 dirs 空间匹配重检，判"注册表不复制"是否口径伪影）；C L 词跨模型位置差异定量（en/fr 轴端点检验）；D 2989 T3 加密复测。
'''
    stamp = e['created'].replace('T', ' ')[:16]
    body = tmpl % {
        'stamp': stamp,
        'nled': len(led['measurements']),
        'nl14': len(l14c['connects']),
        'led8': calc,
        'npz8': s['npz_sha256_8'],
        'res8': s['result_sha256_8'],
        'scr8': e.get('script_sha256_8')}
    io.open(MEMO, 'a', encoding='utf-8').write(body)
out.append('memo appended residue_pct=%s hashes=%s' % (
    '%(' in body.split(tag)[1] if tag in body else 'n/a',
    all(h in body for h in [s['npz_sha256_8'],
                            s['result_sha256_8'],
                            e.get('script_sha256_8')])))

# ---------- 3. workspace log ----------
wsl = io.open(WS, encoding='utf-8').read()
entry = ('\n\n## Phase 2995 Ω-F1 GLM4-9B 承重带三卡复制 '
         '(2026-09-20)\n'
         '- 判决 omega_f1_panel_partial（run5 权威，锚 7/7）；'
         'T1 语言类分离复制（p=0.0014）+ T2 头集中复制'
         '（maxT 地板 5.0e-4），T3 MLP 注册表不复制'
         '（z 全负 vs qwen 31-55）——首张跨模型分化卡；'
         'L 词轴位重排（L>C，qwen 为中间）descriptive\n'
         '- 硬伤 4 笔：pos 切片 / int64 json 键 / T1 null 池'
         '混 L / **双置换 bug**（pool[pm]+lab[pm] 恒复原组，'
         '单置换位置切分修复；教训入册）\n'
         '- Ledger 134 / L14 102 / hash %s；MEMO 2995 节、'
         'seal、npz 落盘\n' % calc)
if 'Phase 2995 Ω-F1' not in wsl:
    io.open(WS, 'a', encoding='utf-8').write(entry)
out.append('wslog appended=%s' % ('Phase 2995'
                                  in io.open(WS,
                                             encoding='utf-8')
                                  .read()))

# ---------- 4. MEMORY ----------
mm = io.open(MEM, encoding='utf-8').read()
pairs = [
    ('2994 Ω-E：类轴注入被擦除',
     '2994 Ω-E：类轴注入被擦除、随机反放大=读出轴防御'),
    ('（F<L<C p 4e-4）', '（p 4e-4）'),
    ('2979 权威：转正分布式', '2979 转正分布式'),
    ('2967 塌缩八层', '2967 塌缩'),
    ('2984 h12 消融归宿=功能严格局部化', '2984 h12 归宿局部化'),
    ('bf16 守卫/修改分母一律取操作数最大幅度（2992/2994 三次）。',
     'bf16 守卫/修改分母取操作数最大幅度（三次）；置换检验单置换+位置切分，禁 values/labels 同 pm 双置换（2995）。'),
    ('max=2994', 'max=2995'),
    ('**2995**（A 主选', '**2996**（A 主选'),
]
for a, b in pairs:
    if a in mm:
        mm = mm.replace(a, b, 1)
add = ('；2995 Ω-F1 GLM4：类分离+头集中复制（p 0.0014/maxT 地板），'
       '注册表不复制（z 负）首分化卡')
marker = '2994 Ω-E：类轴注入被擦除、随机反放大=读出轴防御'
if marker in mm and '2995 Ω-F1' not in mm:
    mm = mm.replace(marker, marker + add, 1)
if len(mm) > 3000:
    for a, b in (
            ('2970 延迟载体', '2970 延迟'),
            ('2973 fr 塌缩=方向重写', '2973 fr 塌缩=重写'),
            ('2985 perp 通道身份=单方向 86%% 能量但词级不稳定'
             '+非轴锁定（重写非旋转）',
             '2985 perp 通道=重写非旋转')):
        if a in mm:
            mm = mm.replace(a, b, 1)
io.open(MEM, 'w', encoding='utf-8').write(mm)
out.append('memory chars=%d max2995=%s next2996=%s dblperm=%s'
           % (len(mm), 'max=2995' in mm, '**2996**' in mm,
              '双置换' in mm))

io.open(OUTLOG, 'w', encoding='utf-8').write(
    '\n'.join(out))
print('closeout done')
